import argparse
import logging
import os
import torch
import traceback
from dotenv import load_dotenv
import huggingface_hub
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer, Trainer, DataCollatorForTokenClassification
import transformers
from trl import SFTTrainer, SFTConfig
import yaml
from accelerate import PartialState

from vllm import SamplingParams
from tuning_models.utils.grading_dataset import GradingDataset, GradingDatasetWithRetrieval
from tuning_models.utils.custom_trainer import CustomTrainer
from tuning_models.utils.dataset_eval_utils import eval_correctness_dataset, eval_closure, EvalCallback
from utils.model_utils import get_basemodel_loadstring, load_model, load_tokenizer
from tuning_models.utils.training_utils import load_dataset, find_latest_checkpoint
from torch.distributed.elastic.multiprocessing.errors import record

def analyze_processed_datasets(base_train_dataset, base_val_dataset, train_dataset, val_dataset, save_dir, save_processed_datasets):
    """
    Export datasets to JSONL format for human-readable verification.
    """
    import json
    import os
    
    # Create output directory
    export_dir = os.path.join(save_dir, "dataset_exports")
    os.makedirs(export_dir, exist_ok=True)
    
    # Export base_train_dataset (raw data)
    base_train_path = os.path.join(export_dir, "base_train_dataset.jsonl")
    with open(base_train_path, 'w') as f:
        for i, text in enumerate(base_train_dataset.texts):
            i < 3 and print(f"Final train dataset example {i}: {text}")
            entry = {
                "text": text,
            }
            if save_processed_datasets:
                f.write(json.dumps(entry) + '\n')
    if save_processed_datasets:
        print(f"Exported base_train_dataset ({len(base_train_dataset.texts)} examples) to {base_train_path}")
    
    # Export base_val_dataset (raw data)
    base_val_path = os.path.join(export_dir, "base_val_dataset.jsonl")
    with open(base_val_path, 'w') as f:
        for i, text in enumerate(base_val_dataset.texts):
            i < 3 and print(f"Final val dataset example {i}: {text}")
            entry = {
                "text": text,
            }
            if save_processed_datasets:
                f.write(json.dumps(entry) + '\n')
    if save_processed_datasets:
        print(f"Exported base_val_dataset ({len(base_val_dataset.texts)} examples) to {base_val_path}")
    
    if save_processed_datasets:
        # Export train_dataset (tokenized)
        train_path = os.path.join(export_dir, "train_dataset.jsonl")
        with open(train_path, 'w') as f:
            for i in range(len(train_dataset)):
                entry = {
                    "input_ids": train_dataset[i]["input_ids"],
                    "labels": train_dataset[i]["labels"]
                }
                f.write(json.dumps(entry) + '\n')
        print(f"Exported train_dataset ({len(train_dataset)} examples) to {train_path}")
        
        # Export val_dataset (tokenized)
        val_path = os.path.join(export_dir, "val_dataset.jsonl")
        with open(val_path, 'w') as f:
            for i in range(len(val_dataset)):
                entry = {
                    "input_ids": val_dataset[i]["input_ids"],
                    "labels": val_dataset[i]["labels"]
                }
                f.write(json.dumps(entry) + '\n')
        print(f"Exported val_dataset ({len(val_dataset)} examples) to {val_path}")
    
    print(f"All datasets exported to {export_dir}")

@record
def main(args):
    # Debug, only enable certain features when we are not running in a distributed process.
    accelerate_fixed = PartialState().num_processes == 1
    print(f"Running in Single Process Mode: {accelerate_fixed}")

    # Check for existing checkpoints if resumption is requested
    resume_from_checkpoint = None
    if args.checkpoints_to_resume_from_path:
        print(f"Checking for checkpoints in: {args.checkpoints_to_resume_from_path}")
        resume_from_checkpoint = find_latest_checkpoint(args.checkpoints_to_resume_from_path)
        if resume_from_checkpoint:
            print(f"Found checkpoint to resume from: {resume_from_checkpoint}")
        else:
            print("No checkpoints found, starting fresh training")
    
    # Accelerate Variables: Device map should be "auto" for single GPU or None for multi-GPU (Trainer will handle accelerate)
    device_map = "auto" if accelerate_fixed else None

    # def Load Model and Tokenizer
    if args.load_existing_model:
        model_dir, model_name = os.path.split(args.model)
        model_info = load_model(model_name, model_dir, full_32_precision=False, device_map=device_map)
        model = model_info["model"]
        tokenizer = model_info["tokenizer"]
    # elif args. #TODO: Allow evaluation of reasoning model by adding in VLLM initialization and separation of GRPO and verbalized. 
    elif args.verbalized:
        from vllm import LLM
        tokenizer = load_tokenizer(args.model)
        model = LLM(model=get_basemodel_loadstring(args.model), tensor_parallel_size=args.tensor_parallel_size, trust_remote_code=True)
    else:
        load_string = get_basemodel_loadstring(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            load_string,
            device_map=device_map,
        )
        tokenizer = AutoTokenizer.from_pretrained(load_string)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if not args.eval_only: #When evaluating with VLLM, you should disable use cache.
        model.config.use_cache = False 
    
    # Required for verbalized evaluations
    vllm_sampling_params = SamplingParams(
        min_p = 0.1,
        top_p = 1.0,
        top_k = -1,
        seed = 3407,
        stop = [tokenizer.eos_token],
        include_stop_str_in_output = True,
        max_tokens = args.max_seq_length
    )

    ## Create Dataset - all processes load but with main process coordination for operations like caching that assume only one process.
    distributed_state = PartialState()
    
    # with distributed_state.main_process_first():
    base_train_dataset, base_val_dataset = load_dataset(
        args.dataset,
        tokenizer,
        ntrain=args.ntrain,
        eval_start_p=args.eval_start_p,
        train_end_p=args.train_end_p,
        datafile=args.datafile,
        shuffle=args.shuffle,
        train_on_all_tokens=args.train_on_all_tokens,
        exclude_from_train=args.exclude_from_train,
        include_model_response=(not args.disable_include_model_response),
        chroma_path=args.chroma_path,
        collection_name=args.collection_name,
        k_retrieve=args.k_retrieve,
        chroma_source_datafile=args.chroma_source_datafile,
        verbalized=args.verbalized,
        use_reasoning=args.use_reasoning,
        use_model_name_in_prompt=args.use_model_name_in_prompt,
        dataset_save_path=args.dataset_save_path,
        dataset_load_path=args.dataset_load_path,
        embedding_function=args.embedding_function,
        group_by_question=args.group_by_question,
        specialty_prompttype=args.specialty_prompttype,
    )
    
    train_dataset = base_train_dataset.to_hf_dataset_tokenized()
    val_dataset = base_val_dataset.to_hf_dataset_tokenized()

    # Examine processed datasets and optionally export to JSONL format for verification
    analyze_processed_datasets(base_train_dataset, base_val_dataset, train_dataset, val_dataset, args.save_dir, args.save_processed_datasets)

    if args.cache_dataset_only:
        return

    # Get unwrapped model for evaluation (will be properly unwrapped after trainer initialization)
    eval_model = model

    ## Evaluate model prior to training, eval dataset
    print("BEGINNING EVALUATIONS in eval_only mode." if args.eval_only else "INITIAL RESULTS FOR UNEDITED MODEL")
    if args.eval_only:
        if accelerate_fixed:
            eval_correctness_dataset(base_val_dataset, eval_model, tokenizer, args.device, args.save_dir, plot_sufix="_eval_only", save_predictions_path=args.save_predictions_path, verbalized=args.verbalized, vllm_sampling_params=vllm_sampling_params, use_reasoning=args.use_reasoning)
        return
    else:
        if accelerate_fixed:
            eval_correctness_dataset(base_val_dataset, eval_model, tokenizer, args.device, args.save_dir, plot_sufix="_base", verbalized=args.verbalized, vllm_sampling_params=vllm_sampling_params, use_reasoning=args.use_reasoning)
    
    ## Create Eval Callback For Training
    try:
        # Note: eval_model will be properly unwrapped after trainer initialization
        closured_eval_function = eval_closure(base_val_dataset, eval_model, tokenizer, args.device, args.save_dir, plot_sufix="_during_training", verbalized=args.verbalized, vllm_sampling_params=vllm_sampling_params, use_reasoning=args.use_reasoning)
        eval_callback = EvalCallback(closured_eval_function)
    except:
        print("Error: eval_callback construction failed")
        traceback.print_exc()
        eval_callback = None

    # def Load PEFT / Lora
    if args.use_unsloth:
        if FastLanguageModel is None:
            raise ImportError("Unsloth is not installed but --use_unsloth was specified. Please install unsloth or remove --use_unsloth flag.")
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.rank,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=args.alpha,
            lora_dropout=args.lora_dropout,
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
    else:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        target_modules_types = {
            "default": ["q_proj", "v_proj"],
            "mlp_down_only": ["down_proj"],
            "combined": ["q_proj", "v_proj", "down_proj"],
            "value_only": ["v_proj"],
            "q_k_only": ["q_proj", "k_proj"],
            "q_v": ["q_proj", "v_proj"],
            "k_o": ["k_proj", "o_proj"],
            "mlps_only": ["up_proj", "down_proj", "gate_proj"],
            "attentions_only": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "all-linear": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        }

        peft_config = LoraConfig(
            lora_alpha=args.alpha,
            lora_dropout=args.lora_dropout,
            r=args.rank,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules_types[args.target_modules],
            # layers_to_transform = args.layers_to_transform # TEMP
        )

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        trainable_params_tuple = model.get_nb_trainable_parameters()
        print(f"{trainable_params_tuple=}")

    # Configure Training Arguments
    calculated_save_steps = int((0.15 * len(train_dataset) * 3) / ((4 if args.training_mode == "qlora" else 1) * (args.train_batch_size))) + 1
    calculated_save_steps = (len(train_dataset) * args.epochs) / ((4 if args.training_mode == "qlora" else 1) * args.train_batch_size) * (1.05/args.n_evals)
    calculated_save_steps = int(calculated_save_steps)
    logger.info(f"Calculated save steps {calculated_save_steps}")

    train_configs = SFTConfig(
        output_dir=args.saving_directory,
        overwrite_output_dir=False,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size, 
        per_device_eval_batch_size=args.eval_batch_size, 
        learning_rate=args.learning_rate,  #3e-5 if args.training_mode == "full_ft" else 5e-5 ,  #5e-6, 1e-5, 
        weight_decay=0.01, # Good default according to huggingface team https://discuss.huggingface.co/t/does-the-default-weight-decay-of-0-0-in-transformers-adamw-make-sense/1180/2
        eval_strategy="steps",
        save_strategy="steps",
        report_to="wandb",
        label_smoothing_factor=args.label_smoothing,
        # dataset_text_field="text",
        # log_level="info",
        # load_best_model_at_end=True,
        # metric_for_best_model="eval_accuracy",
        # greater_is_better=True,

        #Controlled arguments
        optim = "adamw_torch",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_steps= calculated_save_steps*2 if args.testing else calculated_save_steps,
        save_steps= calculated_save_steps*2 if args.testing else calculated_save_steps,  # used to be args.eval_save_steps
        gradient_checkpointing= args.gradient_checkpointing,
    )

    train_configs.gradient_checkpointing = train_configs.gradient_checkpointing
    if train_configs.gradient_checkpointing:
        train_configs.gradient_checkpointing_kwargs = {"use_reentrant": False} # Should be true if using Zero-3

    # Save model parameters information
    os.makedirs(os.path.join(args.saving_directory), exist_ok=True)
    log_param_file_path = os.path.join(args.saving_directory, "params.txt")
    with open(log_param_file_path, "w") as log_file:
        log_file.write("run_name "+os.getenv('SHELLS_LAUNCHER_LOG_NAME', args.runname))
        log_file.write("\nargs for current run: "+str(args))
        log_file.write("\ntraining_arguments "+str(train_configs))
    print(flush=True)
        

    # Create Trainer
    if args.use_custom_trainer and accelerate_fixed:
        trainer = CustomTrainer(
                model=model,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                args=train_configs,
                # compute_metrics=compute_metrics,
                # data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
                data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer, padding="longest", return_tensors="pt"),
                processing_class=tokenizer
            )
    else:
        trainer = Trainer(
                model=model,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                args=train_configs,
                # compute_metrics=compute_metrics,
                # data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
                data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer, padding="longest", return_tensors="pt"),
                processing_class=tokenizer
            )
    if eval_callback is not None and accelerate_fixed:
        try:
            trainer.add_callback(eval_callback)
        except:
            print("Error: add_callback failed")
            traceback.print_exc()
    
    # Deprecated
    eval_model = trainer.model
    
    # Train
    trainer.train()
    trainer.save_state()
    print("FINISHED TRAINING")

    print("FINAL RESULTS\n")
    if accelerate_fixed:
        final_results = eval_correctness_dataset(base_val_dataset, eval_model, tokenizer, args.device, args.save_dir, plot_sufix="_trained", save_predictions_path=args.save_predictions_path)
    print("FINAL RESULTS END\n")

    # Save model, this should be the last block
    print(f"saving_directory {args.saving_directory}")
    model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model  # Take care of distributed/parallel training
    if args.existing_model_path != "":
        # Merge and unload
        print("Var model, trained model before merging new adapter", model_to_save)
        model_to_save = model_to_save.merge_and_unload()
        print("Var model, trained model after merging new adapter", model_to_save)
    model_to_save.save_pretrained(args.saving_directory)
    
    print(f"Saved model {model_to_save}")
    print(f"Saving_directory {args.saving_directory}")

    if accelerate_fixed:
        print(f"Final results {final_results}")


if __name__ == "__main__":
    # TODO: Remove all unused arguments, check carefully.
    import logging, argparse, os, torch, random
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger = logging.getLogger(__name__)

    # -------------------- ARGPARSE -------------------------------------------
    parser = argparse.ArgumentParser()

    # DATA
    parser.add_argument("--dataset", default="MMLU_TF",
                        choices=["MMLU_TF", "logistics", "MMLU",
                                 "num_classification", "mapped_classification", 
                                 "CORRECTNESS_TUNE", "CORRECTNESS_TUNE_RETRIEVAL"])
    parser.add_argument("--datafile", type=str, default=None)
    parser.add_argument("--ntrain", type=int, default=0)
    parser.add_argument("--eval_start_p", type=float, default=.75)
    parser.add_argument("--train_end_p",  type=float, default=.75)
    parser.add_argument("--model",        default="gemma-2b")
    parser.add_argument("--use_model_name_in_prompt", action="store_true", help="If true, replaces 'Model Alpha' with model name from dataset metadata.")
    parser.add_argument("--specialty_prompttype", type=str, default=None, help="Specialty prompt type for prompt generation")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--group_by_question", action="store_true", 
                        help="Group all examples with the same input_prompt together for optimization. Mutually exclusive with --shuffle.")
    parser.add_argument("--exclude_from_train", type=str, nargs='*', default=[], help="List of subjects to exclude from training dataset")
    parser.add_argument("--save_processed_datasets", action="store_true")

    # TRAINING
    parser.add_argument("--train_on_all_tokens", action="store_true")
    parser.add_argument("--load_existing_model", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--use_custom_trainer", action="store_true")
    parser.add_argument("--use_reasoning", action="store_true")
    parser.add_argument("--use_unsloth", action="store_true", help="Use unsloth optimization for faster training")
    parser.add_argument("--max_seq_length", type=int, default=32768, help="Maximum sequence length for model")
    parser.add_argument("--label_smoothing", type=float, default=0.0, help="Label smoothing factor for training")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--training_mode", default="lora",
                        choices=["lora", "qlora", "full_ft", "base", "ICL"])

    # MISC
    parser.add_argument("--save_dir", default="results/model_checkpoints", help="path to save graphs, etc")
    parser.add_argument("--saving_directory", required=False, help="path to save model")
    parser.add_argument("--device",   default="cuda")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--cache_dataset_only", action="store_true")
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--n_evals",  type=int, default=7)
    parser.add_argument("--existing_model_path", type=str, default="")
    parser.add_argument("--runname", type=str, default="temp")
    parser.add_argument("--save_predictions_path", default=None)
    parser.add_argument("--disable_include_model_response", action="store_true", help="Disable including model response in GradingDataset")

    # LoRA hyper-params
    parser.add_argument("--alpha",        type=int,   default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--rank",         type=int,   default=16)
    parser.add_argument("--target_modules", default="default")
    
    # GRPO Turtel et al. (2025) replication
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for generation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps.")
    parser.add_argument("--disable_scale_rewards", dest="scale_rewards", action="store_false", help="Disable reward scaling. Paper suggests disabling this.")
    parser.add_argument("--loss_type", type=str, default="bnpo", choices=["grpo", "bnpo", "dr_grpo"], help="Loss type for GRPO training.")
    parser.add_argument("--reward_function", type=str, default="rlcr",
                        help="The reward function to use for GRPO training.")

    # CHECKPOINT RESUMPTION
    parser.add_argument("--checkpoints_to_resume_from_path", type=str, default=None, 
                       help="Path to directory containing checkpoints to resume from. If provided, will automatically find and resume from the latest checkpoint.")
    
    # argument for testing
    parser.add_argument("--testing", action="store_true")

    # RETRIEVAL ARGUMENTS  
    parser.add_argument("--chroma_path", type=str, default=None, help="Path to store/load the Chroma database")
    parser.add_argument("--collection_name", type=str, default="training-examples", help="Name of the Chroma collection")
    parser.add_argument("--k_retrieve", type=int, default=15, help="Number of similar examples to retrieve for each query")
    parser.add_argument("--chroma_source_datafile", type=str, default=None, help="Path to source dataset for building Chroma collection (if different from main datafile)")
    parser.add_argument("--embedding_function", type=str, default=None, help="Embedding function to use for ChromaDB collection (None for default, 'ReasonIR' for ReasonIR-8B)")

    # DATASET CACHING ARGUMENTS
    parser.add_argument("--dataset_save_path", type=str, default=None, help="Path to save processed datasets to (using pickle), this path is for caching only, we will only load if dataset_load_path is specified.")
    parser.add_argument("--dataset_load_path", type=str, default=None, help="Path to load cached datasets from (using pickle), we'll only load if this path is specified.")

    # VLLM arguments, move to grpo script once that is complete
    parser.add_argument("--verbalized", action="store_true", help="Load an VLLM model, instead of predicting yes/no, predict a probability percentage. This is bundled with eval_only.")
    parser.add_argument("--tensor_parallel_size", type=int, default=4, help="VLLM argument")

    def parse_layers(val: str):
        lst = eval(val)        # trusted environment
        if not isinstance(lst, list):
            lst = [lst]
        flat = []
        for x in lst:
            flat.extend(x if isinstance(x, list) else [x])
        return list(map(int, flat))

    args = parser.parse_args()
    # args.layers_to_transform = parse_layers(args.layers_to_transform)
    args.unsupervised = True
    if args.testing:
        args.eval_start_p = .995
        args.train_end_p = .005
    os.makedirs(args.save_dir, exist_ok=True)

    # Check for mutually exclusive arguments
    if args.shuffle and args.group_by_question:
        parser.error("--shuffle and --group_by_question are mutually exclusive. Choose one or neither. --shuffle undoes the organization done by grouping.")

    main(args)