from typing import Tuple

import transformers
from transformers.integrations import WandbCallback
from transformers import TrainingArguments
from tuning_models.utils.grading_dataset import GradingDatasetWithRetrieval
from tuning_models.utils.prompt_utils import truncate_after_yes_no
from tuning_models.utils.simple_generation_utils import generate_offline
from utils import calib_tools
import deepspeed

import numpy as np
import torch
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve


import copy
import os
import traceback
import re

def score_completion(completion) -> Tuple[int,float]:
    prob_patterns = [
        r"ANSWER_CORRECT_CHANCE:\s*([0-9.]+)%",
        r"ANSWER_CORRECT_PROBABILITY:\s*([0-9.]+)%"
    ]
    probability = -1
    for pattern in prob_patterns:
        match = re.search(pattern, completion, re.IGNORECASE)
        if match:
            probability = float(match.group(1)) / 100.0
    if probability == -1:
        print(f"Warning: Probability={probability} | {completion=}")
    prediction = probability > 0.5
    return prediction, probability


@torch.inference_mode()
def eval_correctness_dataset(
    dataset, model, tokenizer, device, save_dir = None, plot_sufix="", save_predictions_path=None,
    verbose=True, return_var="average_correct", verbalized=False, vllm_sampling_params=None, batched_eval=True, lora_request=None, use_reasoning=False, using_unsloth=False
) -> float:
    """
    This version can also eval correctness for verbalized. Verbose will print all results. Disabling verbose will result in the Confusion Matrix, ECE, and related metrics to be lost as those are not saved.
    """
    def _eval_body(dataset):
        base_dataset: GradingDatasetWithRetrieval = dataset
        dataset = dataset.format_for_grpo() if verbalized else dataset.to_hf_dataset()

        total_correct = 0
        total_datapoints = 0
        predictions_for_attribution = [] #Original dataset information updated with model predictions
        all_probs_yes = []
        all_probs_no = []
        all_max_probs = []
        all_labels = []
        all_preds = []
        all_subjects = []
        all_cors = []
        has_subjects = False

        if batched_eval and verbalized:
            # Batched vLLM evaluation
            prompts = [i["prompt"] for i in dataset]
            completions = generate_offline(
                model, tokenizer, prompts,
                use_vllm=True, verbose=verbose, sampling_params=vllm_sampling_params, using_unsloth=using_unsloth, batched_eval=batched_eval, lora_request=lora_request, use_reasoning=use_reasoning
            )
            for idx, (i, model_completion) in enumerate(zip(dataset, completions)):
                input_text = i["prompt"]
                prediction_label, probability_of_yes = score_completion(model_completion)
                greedy_token_decoded = "yes" if prediction_label else "no"
                prob_yes = probability_of_yes
                prob_no = 1 - probability_of_yes
                gt_label = int(base_dataset.output_info_list[idx]["is_correct"] == "yes")
                all_preds.append(prediction_label)
                all_probs_yes.append(prob_yes)
                all_probs_no.append(prob_no)
                all_max_probs.append(np.max([prob_yes, prob_no]))
                all_cors.append(int(gt_label == prediction_label))
                all_labels.append(gt_label)
                total_correct += int(gt_label == prediction_label)
                total_datapoints += 1
                """Gather Data for Per Subject Plot"""
                if hasattr(base_dataset, "output_info_list") and len(base_dataset.output_info_list) > idx:
                    info = base_dataset.output_info_list[idx]
                    if isinstance(info, dict) and "subject" in info:
                        has_subjects = True
                        all_subjects.append(info["subject"])
                    else:
                        all_subjects.append(None)
                else:
                    all_subjects.append(None)
                """Gather data for chorma attribution"""
                attribution_entry = copy.deepcopy(base_dataset.output_info_list[idx])
                attribution_entry.update({
                    "greedy_token_decoded": greedy_token_decoded,
                    "prob_yes": float(prob_yes),
                    "prob_no": float(prob_no),
                    "training_example": input_text,
                    "correctness_model_completion": model_completion
                })
                predictions_for_attribution.append(attribution_entry)
                (idx < 3) and print(f"{prob_yes=} {prob_no=} {gt_label=} {model_completion=}")
        else:
            # Original per-example loop
            for idx, i in enumerate(dataset):
                if verbalized:
                    """Make Predictions and Collect Data For Graphing and Metrics"""
                    input_text = i["prompt"]
                    idx < 3 and print("Input text:", input_text + "End") #TODO: Pass through generation configs
                    generate_kwargs = dict(max_new_tokens=2048, top_p=0.8, temperature=0.7, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id) # vllm defaults: 'repetition_penalty': 1.05, 'temperature': 0.7, 'top_k': 20, 'top_p': 0.8
                    model_completion = generate_offline(model, tokenizer, input_text, generate_kwargs, use_vllm=True, verbose=(idx < 3), sampling_params=vllm_sampling_params, using_unsloth=using_unsloth, lora_request=lora_request, use_reasoning=use_reasoning)
                    final_result = {"correctness_model_completion": model_completion}
                    prediction_label, probability_of_yes = score_completion(model_completion)
                    if probability_of_yes == -1:
                        print(f"Cause {input_text}")
                    greedy_token_decoded = "yes" if prediction_label else "no" # We convert label to 'yes' 'no' for simplicity, we do not actually decode yes/no in verbalized mode

                    prob_yes = probability_of_yes
                    prob_no = 1 - probability_of_yes
                    gt_label = int(base_dataset.output_info_list[idx]["is_correct"] == "yes")

                    all_preds.append(prediction_label)
                    all_probs_yes.append(prob_yes)
                    all_probs_no.append(prob_no)
                    all_max_probs.append(np.max([prob_yes, prob_no]))
                    all_cors.append(int(gt_label == prediction_label))
                    all_labels.append(gt_label)
                    total_correct += int(gt_label == prediction_label)
                    total_datapoints += 1
                    idx < 3 and print(f"{prob_yes=} {prob_no=} {gt_label=} {model_completion=}")

                else: #Unverbalized
                    """Make Predictions and Collect Data For Graphing and Metrics"""
                    # Find the index of "yes" and "no" tokens in the vocab
                    yes_token_id = tokenizer.encode("yes", add_special_tokens=False)
                    no_token_id = tokenizer.encode("no", add_special_tokens=False)
                    # If tokenizer splits "yes" or "no" into multiple tokens, take the first
                    yes_token_id = yes_token_id[0] if isinstance(yes_token_id, list) else yes_token_id
                    no_token_id = no_token_id[0] if isinstance(no_token_id, list) else no_token_id
                    input_text = truncate_after_yes_no(i["text"])
                    idx < 3 and print("Input text:", input_text + "End")
                    model_device = next(model.parameters()).device
                    input_ids = tokenizer(input_text, return_tensors="pt", padding="longest", truncation=False)["input_ids"].to(model_device)
                    prompt_input_ids = input_ids[:, :-1]
                    outputs = model(prompt_input_ids)

                    """Aggregate Results"""
                    greedy_token = torch.argmax(outputs.logits[:, -1], dim=-1)
                    greedy_token_decoded = tokenizer.decode(greedy_token).strip().lower()
                    label_decoded = tokenizer.decode(input_ids[:, -1]).strip().lower()
                    idx < 3 and print(f"{greedy_token=} {tokenizer.decode(greedy_token)=} {input_ids[:, -1]=} | {greedy_token_decoded=} {label_decoded=}")
                    correct = greedy_token_decoded == label_decoded
                    total_correct += correct
                    total_datapoints += 1

                    # Track Labels
                    logits = outputs.logits[:, -1]
                    probs = torch.softmax(logits, dim=-1).detach().cpu().numpy().flatten()
                    if label_decoded == "yes":
                        label = 1
                    elif label_decoded == "no":
                        label = 0
                    else:
                        raise Exception(f"Unknown label: {label_decoded}")

                    # Track Preds for Confusion Matrix
                    if greedy_token_decoded == "yes":
                        pred_label = 1
                    elif greedy_token_decoded == "no":
                        pred_label = 0
                    else:
                        print(f"Warning:error: {greedy_token_decoded=} is not 'yes' or 'no'")
                        try:
                            print(f"Cause of warning, decoded input (input_ids[:, :-1]) to model: {tokenizer.decode(prompt_input_ids[0])}")
                        except:
                            pass
                        pred_label = -1
                    all_preds.append(pred_label)

                    prob_yes = probs[yes_token_id] if yes_token_id < len(probs) else 0.0
                    prob_no = probs[no_token_id] if no_token_id < len(probs) else 0.0
                    all_probs_yes.append(prob_yes)
                    all_probs_no.append(prob_no)
                    all_max_probs.append(np.max([prob_yes, prob_no]))
                    all_cors.append(int(correct))
                    all_labels.append(label)
                    idx < 3 and print(f"{prob_yes=} {prob_no=} {label=} | {np.max([prob_yes, prob_no])=}")
                    final_result = dict()

                """Gather Data for Per Subject Plot"""
                # For per-subject calibration
                if hasattr(base_dataset, "output_info_list") and len(base_dataset.output_info_list) > idx:
                    info = base_dataset.output_info_list[idx]
                    if isinstance(info, dict) and "subject" in info:
                        has_subjects = True
                        all_subjects.append(info["subject"])
                    else:
                        all_subjects.append(None)
                else:
                    all_subjects.append(None)

                """Gather data for chorma attribution"""
                # Build attribution entry based on original info
                if hasattr(base_dataset, "output_info_list") and len(base_dataset.output_info_list) > idx:
                    attribution_entry = copy.deepcopy(base_dataset.output_info_list[idx])
                else:
                    attribution_entry = {}
                attribution_entry.update({
                    "greedy_token_decoded": greedy_token_decoded,
                    "prob_yes": float(prob_yes),
                    "prob_no": float(prob_no),
                    "training_example": input_text
                })
                attribution_entry.update(final_result)
                predictions_for_attribution.append(attribution_entry)

        """Evaluate Accuracy"""
        average_correct = total_correct / total_datapoints
        verbose and print(f"{average_correct=}")

        """Evaluate Calibration Metrics"""
        all_max_probs_np = np.array(all_max_probs)
        all_cors_np = np.array(all_cors)
        all_labels_np = np.array(all_labels)
        all_probs_yes_np = np.array(all_probs_yes)

        try:
            ece = calib_tools.ece(all_labels_np, all_probs_yes_np)
        except Exception as e:
            print("ECE calculation failed:")
            traceback.print_exc()
            ece = None

        try:
            rmsce = calib_tools.rmsce(all_cors_np, all_max_probs_np)
        except Exception as e:
            print("RMSCE calculation failed:")
            traceback.print_exc()
            rmsce = None

        try:
            auroc = roc_auc_score(all_labels_np, all_probs_yes_np)
        except Exception as e:
            print("AUROC calculation failed:")
            traceback.print_exc()
            auroc = None
        verbose and print(f"Binary ECE: {ece}, RMSCE: {rmsce}, AUROC: {auroc}")

        """Confusion Matrix"""
        cm = confusion_matrix(all_labels_np, np.array(all_preds), labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        if verbose:
            print("\nConfusion-matrix (rows = true 0/1, cols = pred 0/1)")
            print(cm)
            print(f"TP={tp}  FP={fp}\nFN={fn}  TN={tn}\n")

        """Calibration Plot Grouped By Subject"""
        if has_subjects and save_dir is not None:
            try:
                import matplotlib.pyplot as plt
                import os
                subjects = set([str(s) for s in all_subjects if s is not None])
                for subject in subjects:
                    subject_probs = [p for p, s in zip(all_max_probs, all_subjects) if s == subject]
                    subject_labels = [l for l, s in zip(all_labels, all_subjects) if s == subject]
                    if len(subject_probs) < 5:
                        continue  # skip subjects with too few samples
                    # Bin the probabilities for calibration curve
                    bins = np.linspace(0, 1, 11)
                    binids = np.digitize(subject_probs, bins) - 1
                    accs = []
                    confs = []
                    for b in range(len(bins)-1):
                        bin_labels = [subject_labels[i] for i in range(len(binids)) if binids[i] == b]
                        bin_probs = [subject_probs[i] for i in range(len(binids)) if binids[i] == b]
                        if len(bin_labels) == 0:
                            continue
                        accs.append(np.mean(bin_labels))
                        confs.append(np.mean(bin_probs))
                    plt.clf()
                    plt.scatter(confs, accs)
                    minv = min(confs + accs)
                    maxv = max(confs + accs)
                    x = np.arange(minv, maxv, 0.01)
                    plt.plot(x, x, c="r")
                    plt.xlabel("Confidence")
                    plt.ylabel("Accuracy")
                    plt.title(f"Calibration Curve: {subject}")
                    os.makedirs(save_dir, exist_ok=True)
                    plt.savefig(os.path.join(save_dir, f"calibration_{subject}{plot_sufix}.png"))
            except Exception as e:
                print(f"Per-subject calibration plotting failed: {e}")
                traceback.print_exc()

        if save_dir is not None:
            import matplotlib.pyplot as plt, os

            os.makedirs(save_dir, exist_ok=True)

            """Calibration Curve (binned by confidence)"""
            try:
                prob_true, prob_pred = calibration_curve(
                    all_labels_np, all_probs_yes_np, n_bins=10, strategy="uniform" 
                )
                prob_true = prob_true[prob_true>=0]
                prob_pred = prob_pred[prob_pred>=0]
                # Find how many points are in each bin
                bins = np.linspace(0, 1, 11) # 10 bins from 0 to 1
                counts, _ = np.histogram(all_probs_yes_np, bins=bins, density=False)
                counts = counts[counts>0]
                total_counts = np.sum(counts)
                markersize = (counts / total_counts) * 45 # If you encounter divison by 0, total_counts is incorrect
                markersize = markersize[markersize>0]
                print(f"Counts: {counts}")
                print(f"prob_preds: {prob_pred}")
                plt.clf()
                try:
                    plt.scatter(prob_pred, prob_true, marker="o", s=markersize)
                    plt.title(f"Reliability Diagram (all subjects) ECE: {ece} | (marksize~counts)")
                except:
                    plt.scatter(prob_pred, prob_true, marker="x")
                    plt.title(f"Reliability Diagram (all subjects) ECE: {ece} | (marksize=None)")
                plt.plot([0, 1], [0, 1], linestyle="--")
                plt.xlabel("Confidence")
                plt.ylabel("Empirical Accuracy")
                plt.savefig(os.path.join(save_dir, f"calibration_overall{plot_sufix}.png"))
                print(f"Saved Reliability curve to", str(os.path.join(save_dir, f"calibration_overall{plot_sufix}.png")))
            except:
                print("Realiability curve drawing failed")
                traceback.print_exc()

            """Create histogram with same bins as reliability curve"""
            try:
                plt.clf()
                bins = np.linspace(0, 1, 11) # 10 bins from 0 to 1
                bin_centers = (bins[:-1] + bins[1:]) / 2
                # Separate data by labels
                probs_label_1 = all_probs_yes_np[all_labels_np == 1]
                probs_label_0 = all_probs_yes_np[all_labels_np == 0]
                # Create histograms
                hist_1, _ = np.histogram(probs_label_1, bins=bins)
                hist_0, _ = np.histogram(probs_label_0, bins=bins)
                # Plot both histograms
                plt.bar(bin_centers, hist_1, width=0.08, alpha=0.7, label='Label=1 (Yes)', color='blue')
                plt.bar(bin_centers, hist_0, width=0.08, alpha=0.7, label='Label=0 (No)', color='red')
                plt.xlabel("Confidence")
                plt.ylabel("Count")
                plt.title("Distribution of Confidence Scores by True Label")
                plt.legend()
                plt.savefig(os.path.join(save_dir, f"confidence_histogram{plot_sufix}.png"))
                print(f"Saved confidence histogram to", str(os.path.join(save_dir, f"confidence_histogram{plot_sufix}.png")))
            except:
                print("Histogram failed")
                traceback.print_exc()

            """AUROC Curve"""
            fpr, tpr, _ = roc_curve(all_labels_np, all_probs_yes_np)
            plt.clf()
            plt.plot(fpr, tpr, marker=".")
            plt.plot([0, 1], [0, 1], linestyle="--")
            plt.xlabel("False-positive rate")
            plt.ylabel("True-positive rate")
            plt.title(f"ROC Curve (AUROC={auroc:0.3f})")
            plt.savefig(os.path.join(save_dir, f"roc_curve{plot_sufix}.png"))
            print(f"Saved AUROC curve to", str(os.path.join(save_dir, f"roc_curve{plot_sufix}.png")))

        """Save data"""
        if save_predictions_path is not None:
            base_dataset.save_predictions_for_attribution(predictions_for_attribution, save_predictions_path)
        if return_var == "average_correct":
            return average_correct
        elif return_var == "predictions_for_attribution":
            return predictions_for_attribution
    
    print(f"Evaluation starting on {model}")
    if hasattr(model, 'module') and hasattr(model.module, 'ds_engine') and isinstance(model, deepspeed.DeepSpeedEngine):
        with deepspeed.zero.GatheredParameters(model.module.parameters(), modifier_rank=0):
            print("Unsharding Parameters For Evaluations")
            out = _eval_body(dataset)
    else:
        print("Not Unsharding Parameters For Evaluations")
        out = _eval_body(dataset)
    return out


def eval_closure(base_val_dataset, model, tokenizer, device, save_dir, plot_sufix="_base", verbalized=False, vllm_sampling_params=None, use_reasoning=False):
    def evoke_eval(plot_sufix_sufix):
        return eval_correctness_dataset(base_val_dataset, model, tokenizer, device, save_dir, plot_sufix=plot_sufix+plot_sufix_sufix, verbalized=verbalized, vllm_sampling_params=vllm_sampling_params, use_reasoning=use_reasoning)
    return evoke_eval


class EvalCallback(WandbCallback):
    def __init__(self, closured_eval_func):
        super().__init__()
        self.eval_func = closured_eval_func

    def on_evaluate(self, args: TrainingArguments, state: transformers.TrainerState, control: transformers.TrainerControl, metrics, **kwargs):
        average_correct = self.eval_func("_epoch"+str(state.epoch))
        metrics["eval_average_correct"] = average_correct
        return super().on_evaluate(args, state, control, metrics=metrics, **kwargs)