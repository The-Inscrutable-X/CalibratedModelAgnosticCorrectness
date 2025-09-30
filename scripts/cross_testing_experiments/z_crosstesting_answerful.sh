#!/bin/bash
: << 'SCRIPT_DESCRIPTION'
Can models predict the correctness of other models, or do models only have a special ability to predict the correctness of their own generations?
\begin{itemize}
    \item Show Llama3 $\to$ Llama3, vs Qwen $\to$ Llama3, vs (new) Llama3 $\to$ Qwen, vs (new) Qwen $\to$ Qwen | this must done under the \textbf{answerless setting in order to marginalize impact from parametric knowledge}.
    \item We have only Llama3 $\to$ Llama3, vs Qwen $\to$ Llama3 in the parametric with answers setting.
\end{itemize}
SCRIPT_DESCRIPTION

#SETUP
module purge
module load python/3.12.2
source ttfs_venv/bin/activate
set -x

#CONFIGS
dataset="CORRECTNESS_TUNE"
seeds=(0)
seed=0
lr=1e-5
train_end_p=.70
results_dir="results"
# device="4,5,6,7"
# eval_device="4,5,6,7"
device="0,1,2,3"
eval_device="0,1,2,3"
model_to_trains=("Qwen2.5-7B-Instruct" "Llama-3.1-8B-Instruct")
calibration_types=("spline_calibration" "isotonic_regression" "beta_calibration" "platt")
ntrain=0
epoch=1
t_batch=1
rank=32
runnames=("Alpha_Model_Qwen2.5-7B-Instruct_unhinted" "Alpha_Model_Llama-3.1-8B-Instruct_unhinted")
k_retrieves=(0)
k_retrieve=0
gradient_accumulation_steps=4

accelerate_config="scripts/configs/GPUn4DeepspeedZero2.yaml"
echo "Accelerate config file content:"
cat $accelerate_config
echo ""

for runname in "${runnames[@]}"
do
    for model_to_train in "${model_to_trains[@]}"
    do
        for k_retrieve in "${k_retrieves[@]}"
        do
            tunename="${model_to_train}_${dataset}_k${k_retrieve}_e${epoch}_tb${t_batch}_TrainEndP${train_end_p}_rank${rank}_lr${lr}_answerless_sft_tune"
            export VLLM_CACHE_ROOT="$results_dir/$runname/${tunename}/vllm_cache"
            rm -rf "$VLLM_CACHE_ROOT"
            mkdir -p "$VLLM_CACHE_ROOT"
            
            CUDA_VISIBLE_DEVICES=$device accelerate launch --config_file $accelerate_config -m tuning_models.standard_tuning_sft \
                --dataset $dataset \
                --ntrain $ntrain \
                --eval_start_p 0.75 \
                --train_end_p $train_end_p \
                --model $model_to_train \
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
                --saving_directory "$results_dir/$runname/${tunename}/${model_to_train}_lora_model" \
                --save_predictions_path "$results_dir/$runname/${tunename}/attributed_graded_dataset_generations.jsonl" \
                --epochs $epoch \
                --n_evals 5 \
                --use_custom_trainer \
                --k_retrieve $k_retrieve \
                --gradient_accumulation_steps $gradient_accumulation_steps\
                --shuffle \
                --gradient_checkpointing \
                --dataset_save_path "$results_dir/$runname/dataset_cache" \
                --use_model_name_in_prompt

            CUDA_VISIBLE_DEVICES=$device python -m tuning_models.standard_tuning_sft \
                --dataset $dataset \
                --ntrain $ntrain \
                --eval_start_p 0.75 \
                --train_end_p $train_end_p \
                --model "$results_dir/$runname/${tunename}/${model_to_train}_lora_model" \
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
                --saving_directory "$results_dir/$runname/${tunename}/${model_to_train}_lora_model" \
                --save_predictions_path "$results_dir/$runname/${tunename}/attributed_graded_dataset_generations.jsonl" \
                --epochs $epoch \
                --n_evals 5 \
                --use_custom_trainer \
                --k_retrieve $k_retrieve \
                --gradient_accumulation_steps $gradient_accumulation_steps\
                --shuffle \
                --gradient_checkpointing \
                --dataset_save_path "$results_dir/$runname/dataset_cache" \
                --load_existing_model \
                --eval_only \
                --use_model_name_in_prompt

            # for calibration_type in "${calibration_types[@]}"
            # do
            #     calibrated_results_savedir="posthoc_${calibration_type}_calibration_results"
            #     CUDA_VISIBLE_DEVICES=$device python3 -m tuning_models.posthoc_calibration \
            #         --dataset $dataset \
            #         --datafile "$results_dir/$runname/graded_dataset_generations.jsonl" \
            #         --model ${model_to_train}_lora_model \
            #         --checkpoints_dir "$results_dir/$runname/${tunename}" \
            #         --load_lora \
            #         --eval_start_p 0.75 \
            #         --train_end_p $train_end_p \
            #         --calibration_p 0.05 \
            #         --ntrain $ntrain \
            #         --calibration_type $calibration_type \
            #         --load_eval_probabilities_path "$results_dir/$runname/${tunename}/attributed_graded_dataset_generations.jsonl" \
            #         --save_dir $results_dir/$runname/${tunename}/${calibrated_results_savedir} \
            #         --save_calibration_predictions_path "$results_dir/$runname/${tunename}/${calibrated_results_savedir}/attributed_graded_calibration_dataset_generations.jsonl" \
            #         --save_calibrated_eval_predictions_path "$results_dir/$runname/${tunename}/${calibrated_results_savedir}/calibrated_attributed_graded_dataset_generations.jsonl" \
            #         --use_model_name_in_prompt \
            #         --disable_include_model_response
            # done
            rm -rf "$VLLM_CACHE_ROOT"
        done
    done
done
