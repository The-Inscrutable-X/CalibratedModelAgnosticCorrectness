#!/bin/bash
: << 'SCRIPT_DESCRIPTION'
Evaluate the generalized model trained with k in context examples
vs 
specific model trained with k in context examples
model trained with no in context examples directly
generalized model trained with 0 in context examples
Base model

For a setting all of them were trained on
SCRIPT_DESCRIPTION

#SETUP
module purge
module load python/3.12.2
source ttfs_venv/bin/activate
set -x

#CONFIGS
dataset="CORRECTNESS_TUNE" #  "Spider" "GSM8k" "MMLU_STEM" "MMLU_humanities" "MMLU_social_sciences" wikitext2; "GSM8k" 
seeds=(0)
seed=0
lr=1e-5
train_end_p=.70
results_dir="results"
# device="4,5,6,7"
# eval_device="4,5,6,7"
device="2"
eval_device="2"
model_to_evals=( \
    "Llama-3.1-8B-Instruct" \
)
calibration_types=("spline_calibration" "isotonic_regression")
ntrain=0
epoch=1
t_batch=1
rank=32
# runname=Alpha_Model_Llama-3.1-8B-Instruct_unhinted | Alpha_Model_gemma-3-27b-it_unhinted
runnames=(
    Alpha_Model_Llama-3.1-8B-Instruct_unhinted
)
k_retrieves=(0)
reward_function="none"
gradient_accumulation_steps=4

accelerate_config="scripts/configs/GPUn4DeepspeedZero2.yaml"
echo "Accelerate config file content:"
cat $accelerate_config
echo ""

for runname in "${runnames[@]}"
do
    for model_to_eval in "${model_to_evals[@]}"
    do
        for k_retrieve in "${k_retrieves[@]}"
        do
            tunename="${model_to_eval}_${dataset}_k${k_retrieve}_eval_ptrue_v2"
            export VLLM_CACHE_ROOT="$results_dir/$runname/${tunename}/vllm_cache"
            rm -rf "$VLLM_CACHE_ROOT"
            mkdir -p "$VLLM_CACHE_ROOT"
            
            CUDA_VISIBLE_DEVICES=$device python -m dynamics_project.tuning_models.standard_tuning_sft \
                --dataset $dataset \
                --ntrain $ntrain \
                --eval_start_p 0.75 \
                --train_end_p $train_end_p \
                --model $model_to_eval \
                --save_dir $results_dir/$runname/${tunename} \
                --device "cuda" \
                --train_batch_size $t_batch \
                --training_mode "lora" \
                --alpha 16 \
                --lora_dropout 0.0 \
                --rank $rank \
                --learning_rate $lr \
                --datafile "$results_dir/$runname/graded_dataset_generations.jsonl" \
                --target_modules "default" \
                --runname $runname \
                --save_predictions_path "$results_dir/$runname/${tunename}/attributed_graded_dataset_generations.jsonl" \
                --epochs $epoch \
                --n_evals 5 \
                --use_custom_trainer \
                --chroma_path "$results_dir/$runname/chroma_train" \
                --k_retrieve $k_retrieve \
                --gradient_accumulation_steps $gradient_accumulation_steps \
                --shuffle \
                --use_model_name_in_prompt \
                --gradient_checkpointing \
                --dataset_save_path "$results_dir/$runname/dataset_cache" \
                --load_existing_model \
                --eval_only \
                --save_processed_datasets

            # for calibration_type in "${calibration_types[@]}"
            # do
                # calibrated_results_savedir="posthoc_${calibration_type}_calibration_results"
                # CUDA_VISIBLE_DEVICES=$device python3 -m dynamics_project.tuning_models.posthoc_calibration \
                #     --dataset $dataset \
                #     --datafile "$results_dir/$runname/graded_dataset_generations.jsonl" \
                #     --model ${model_to_eval}_lora_model \
                #     --checkpoints_dir "$results_dir/$runname/${tunename}" \
                #     --load_lora \
                #     --eval_start_p 0.75 \
                #     --train_end_p $train_end_p \
                #     --calibration_p 0.05 \
                #     --ntrain $ntrain \
                #     --calibration_type $calibration_type \
                #     --load_eval_probabilities_path "$results_dir/$runname/${tunename}/attributed_graded_dataset_generations.jsonl" \
                #     --save_dir $results_dir/$runname/${tunename}/${calibrated_results_savedir} \
                #     --save_calibration_predictions_path "$results_dir/$runname/${tunename}/${calibrated_results_savedir}/attributed_graded_calibration_dataset_generations.jsonl" \
                #     --save_calibrated_eval_predictions_path "$results_dir/$runname/${tunename}/${calibrated_results_savedir}/calibrated_attributed_graded_dataset_generations.jsonl"
            # done
            rm -rf "$VLLM_CACHE_ROOT"
        done
    done
done