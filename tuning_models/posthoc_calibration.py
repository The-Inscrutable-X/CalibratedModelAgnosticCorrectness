import argparse
import copy
import json
import os

import torch

from tuning_models.utils.dataset_eval_utils import eval_correctness_dataset
from tuning_models.utils.training_utils import load_dataset
from tuning_models.utils.grading_dataset import GradingDataset
from tuning_models.utils.prompt_utils import truncate_after_yes_no
from utils.model_utils import load_model, ModelInfo, get_basemodel_loadstring, load_tokenizer

# Add additional imports needed for model loading
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import PartialState

# Calibration Tools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ml_insights as mli
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from betacal import BetaCalibration
print(mli.__version__)
from utils import calib_tools
import traceback


def process_predictions_for_calibration(predictions_for_attribution):
    """
    Process predictions_for_attribution into format needed for calibration.
    
    Args:
        predictions_for_attribution: List of dicts from eval_correctness_dataset
        
    Returns:
        tuple: (uncalibrated_probs, true_labels) as numpy arrays
    """
    uncalibrated_probs = []  # uncalibrated probabilities for "yes" class
    true_labels = []  # true binary labels (1 for "yes", 0 for "no")
    
    for entry in predictions_for_attribution:
        # Extract uncalibrated probability for "yes" class
        prob_yes = float(entry["prob_yes"])
        uncalibrated_probs.append(prob_yes)
        
        # Extract true label - need to determine what the correct answer was
        if "is_correct" in entry:
            # If correct answer is stored
            true_label = 1 if entry["is_correct"].lower() == "yes" else 0
        else:
            # Fallback: check if the training example ends with "yes" or "no"
            training_example = entry["training_example"]
            if training_example.strip().lower().endswith("yes"):
                true_label = 1
            elif training_example.strip().lower().endswith("no"):
                true_label = 0
            else:
                raise ValueError(f"Cannot determine true label for entry: {entry}")
        
        true_labels.append(true_label)
    
    # Convert to numpy arrays
    uncalibrated_probs = np.array(uncalibrated_probs)
    true_labels = np.array(true_labels)
    
    return uncalibrated_probs, true_labels


def evaluate_predictions(calibrated_probs, true_labels, save_dir=None, plot_suffix="_calibrated", verbose=True):
    """
    Evaluate calibrated predictions with the same metrics as eval_correctness_dataset.
    
    Args:
        calibrated_probs: Array of calibrated probabilities for positive class
        true_labels: Array of true binary labels (0/1)
        save_dir: Directory to save plots and metrics
        plot_suffix: Suffix for saved plot files
        verbose: Whether to print results (matches eval_correctness_dataset)
    
    Returns:
        dict: Dictionary containing all computed metrics
    """
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
    from utils import calib_tools
    import matplotlib.pyplot as plt
    import traceback
    
    results = {}
    
    print("\n" + "="*50)
    print(f"EVALUATION WITH {plot_suffix} PROBABILITIES")
    print("="*50)
    
    # 1. Accuracy
    calibrated_preds = (calibrated_probs > 0.5).astype(int)
    average_correct = np.mean(calibrated_preds == true_labels)
    results['average_correct'] = average_correct
    verbose and print(f"{average_correct=}")

    # 2. Calibration Metrics - match the format exactly
    all_labels_np = np.array(true_labels)
    all_probs_yes_np = np.array(calibrated_probs)
    all_cors_np = (calibrated_preds == true_labels).astype(int)
    all_max_probs_np = np.maximum(calibrated_probs, 1 - calibrated_probs)

    try:
        ece = calib_tools.ece(all_labels_np, all_probs_yes_np)
        results['ece'] = ece
    except Exception as e:
        print("ECE calculation failed:")
        traceback.print_exc()
        ece = None
        results['ece'] = None

    try:
        rmsce = calib_tools.rmsce(all_cors_np, all_max_probs_np)
        results['rmsce'] = rmsce
    except Exception as e:
        print("RMSCE calculation failed:")
        traceback.print_exc()
        rmsce = None
        results['rmsce'] = None

    try:
        auroc = roc_auc_score(all_labels_np, all_probs_yes_np)
        results['auroc'] = auroc
    except Exception as e:
        print("AUROC calculation failed:")
        traceback.print_exc()
        auroc = None
        results['auroc'] = None
    
    # Print in the exact same format as eval_correctness_dataset
    verbose and print(f"Binary ECE: {ece}, RMSCE: {rmsce}, AUROC: {auroc}")

    # 3. Confusion Matrix - match the format exactly
    cm = confusion_matrix(all_labels_np, np.array(calibrated_preds))
    tn, fp, fn, tp = cm.ravel()
    results['confusion_matrix'] = {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}
    if verbose:
        print("\nConfusion-matrix (rows = true 0/1, cols = pred 0/1)")
        print(cm)
        print(f"TP={tp}  FP={fp}\nFN={fn}  TN={tn}\n")

    if save_dir is not None:
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # 4. Reliability Diagram - match the format exactly
        try:
            prob_true, prob_pred = calibration_curve(
                all_labels_np, all_probs_yes_np, n_bins=10, strategy="uniform" 
            )
            bins = np.linspace(0, 1, 11) # 10 bins from 0 to 1
            counts, _ = np.histogram(all_probs_yes_np, bins=bins, density=False)
            print(f"Counts: {counts}")
            counts = counts[counts != 0]
            total_counts = np.sum(counts)
            markersize = (counts / total_counts) * 45 #If you encounter divison by 0, perhaps total_counts is incorrect
            markersize = markersize + 5
            plt.clf()
            plt.scatter(prob_pred, prob_true, marker="o", s=markersize)

            plt.plot([0, 1], [0, 1], linestyle="--")
            plt.xlabel("Confidence")
            plt.ylabel("Empirical Accuracy")
            plt.title(f"Reliability Diagram (all subjects) ECE: {ece}")
            plt.savefig(os.path.join(save_dir, f"calibration_overall{plot_suffix}.png"))
            print(f"Saved Reliability curve to", str(os.path.join(save_dir, f"calibration_overall{plot_suffix}.png")))
        except:
            print("Realiability curve drawing failed")
            traceback.print_exc()
        
        # 5. Confidence Histogram - match the format exactly
        try:
            plt.clf()
            bins = np.linspace(0, 1, 11)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            probs_label_1 = all_probs_yes_np[all_labels_np == 1]
            probs_label_0 = all_probs_yes_np[all_labels_np == 0]
            hist_1, _ = np.histogram(probs_label_1, bins=bins)
            hist_0, _ = np.histogram(probs_label_0, bins=bins)
            
            plt.bar(bin_centers, hist_1, width=0.08, alpha=0.7, label='Label=1 (Yes)', color='blue')
            plt.bar(bin_centers, hist_0, width=0.08, alpha=0.7, label='Label=0 (No)', color='red')
            plt.xlabel("Confidence")
            plt.ylabel("Count")
            plt.title("Distribution of Confidence Scores by True Label")
            plt.legend()
            plt.savefig(os.path.join(save_dir, f"confidence_histogram{plot_suffix}.png"))
            print(f"Saved confidence histogram to", str(os.path.join(save_dir, f"confidence_histogram{plot_suffix}.png")))
        except:
            print("Histogram failed")
            traceback.print_exc()
        
        # 8. ROC Curve
        try:
            fpr, tpr, _ = roc_curve(true_labels, calibrated_probs)
            plt.clf()
            plt.plot(fpr, tpr, marker=".")
            plt.plot([0, 1], [0, 1], linestyle="--")
            plt.xlabel("False-positive rate")
            plt.ylabel("True-positive rate")
            plt.title(f"ROC Curve{plot_suffix} (AUROC={results.get('auroc', 'N/A'):.3f})")
            plt.savefig(os.path.join(save_dir, f"roc_curve{plot_suffix}.png"))
            print(f"Saved ROC curve to {os.path.join(save_dir, f'roc_curve{plot_suffix}.png')}")
            plt.close()
        except Exception as e:
            print(f"ROC curve failed: {e}")
    
    return results


def eval_calibration_fit(y_calib_1, calibset_preds_uncalib_1, y_test_1, testset_preds_uncalib_1, testset_calibrated_probs, tvec, tvec_pred, args, ):
    try:
        bins = np.linspace(0, 1, 11)
        mli.plot_reliability_diagram(y_calib_1, calibset_preds_uncalib_1, error_bars=False, bins=bins);
        # tvec = np.linspace(.01, .99, 99)
        # plt.plot(tvec, lr.predict_proba(tvec.reshape(-1,1))[:,1]);
        plt.plot(tvec, tvec_pred)
        plt.title('Calibration Curve on Calibration Data');
        plt.savefig(os.path.join(args.save_dir, f"reliability_calibrationdata.png"))
        plt.clf()
        print(f"Saved Reliability curve 1 to", str(os.path.join(args.save_dir, f"reliability_calibrationdata.png")))

        mli.plot_reliability_diagram(y_test_1, testset_preds_uncalib_1, error_bars=False, bins=bins);
        # tvec = np.linspace(.01, .99, 99)
        # plt.plot(tvec, lr.predict_proba(tvec.reshape(-1,1))[:,1])
        plt.plot(tvec, tvec_pred)
        plt.title('Calibration Curve on Eval Data');
        plt.savefig(os.path.join(args.save_dir, f"reliability_evaldata.png"))
        plt.clf()
        print(f"Saved Reliability curve 2 to", str(os.path.join(args.save_dir, f"reliability_evaldata.png")))

        # Diagnostic: check for problematic values in calibrated probabilities
        print(f"\nCalibrated probabilities diagnostics:")
        print(f"  Min: {np.min(testset_calibrated_probs):.6f}")
        print(f"  Max: {np.max(testset_calibrated_probs):.6f}")
        print(f"  Mean: {np.mean(testset_calibrated_probs):.6f}")
        print(f"  Std: {np.std(testset_calibrated_probs):.6f}")
        print(f"  Values < 0: {np.sum(testset_calibrated_probs < 0)}")
        print(f"  Values > 1: {np.sum(testset_calibrated_probs > 1)}")
        print(f"  NaN values: {np.sum(np.isnan(testset_calibrated_probs))}")
        print(f"  Inf values: {np.sum(np.isinf(testset_calibrated_probs))}")
        
        mli.plot_reliability_diagram(y_test_1, testset_calibrated_probs, bins=bins);
        plt.title('Reliability Diagram on Test Data\n after Calibration');
        plt.savefig(os.path.join(args.save_dir, f"post_calibration_reliability_evaldata.png"))
        plt.clf()
        print(f"Saved Reliability curve 3 to", str(os.path.join(args.save_dir, f"post_calibration_reliability_evaldata.png")))

    except:
        print("ml_insights analysis failed")
        traceback.print_exc()

def main(args):
    """
    This function will load in the calibration dataset, run the loaded model on it to obtain probabilities using a function that is a subset of eval_correctness_dataset without the plotting, the input will be of the same form as the input to attribute.py.
    Calibrate the model's probabilities by training on this calibration dataset.
    If the model was a lora model previously trained, then we may load the probabilities of the model from output of standard_tuning.py, but is equally able to simply rerun the model to get raw uncalibrated probabilities to be transformed.
    Then the probabilities of the model, transformed by the calibration procedure, will be evaluated on the eval dataset.
    The eval will be done via an edit of eval_correctness_dataset, meaning it produces all the plots as well as the data 
    That can be used by attribute.py.
    """
    # Debug, only enable certain features when we are not running in a distributed process.
    accelerate_fixed = PartialState().num_processes == 1
    print(f"Running in Single Process Mode: {accelerate_fixed}")
    
    # Accelerate Variables: Device map should be "auto" for single GPU or None for multi-GPU (Trainer will handle accelerate)
    device_map = "auto" if accelerate_fixed else None

    # Load Model and Tokenizer (mirrored from standard_tuning_sft)
    if args.load_existing_model:
        model_dir, model_name = os.path.split(args.model)
        model_info = load_model(model_name, model_dir, full_32_precision=False, device_map=device_map)
        model = model_info["model"]
        tokenizer = model_info["tokenizer"]
    elif args.verbalized:
        from vllm import LLM
        tokenizer = load_tokenizer(args.model)
        model = LLM(model=get_basemodel_loadstring(args.model), tensor_parallel_size=args.tensor_parallel_size, trust_remote_code=True)
    else:
        model_info: ModelInfo = load_model(args.model, checkpoints_dir=args.checkpoints_dir, lora_model=args.load_lora)
        tokenizer = model_info["tokenizer"]
        model = model_info["model"]
    
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Required for verbalized evaluations (mirrored from standard_tuning_sft)
    from vllm import SamplingParams
    vllm_sampling_params = SamplingParams(
        min_p = 0.1,
        top_p = 1.0,
        top_k = -1,
        seed = 3407,
        stop = [tokenizer.eos_token],
        include_stop_str_in_output = True,
        max_tokens = args.max_seq_length
    )
    # Load same dataset as used in standard_tuning
    calib_ds, eval_ds = load_dataset(
        args.dataset,
        tokenizer=tokenizer,
        ntrain=args.ntrain,
        eval_start_p=args.eval_start_p,
        train_end_p=args.train_end_p,
        calibration_p=args.calibration_p,
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
        override_splits=args.override_splits,
    )
    print(f"Calibration dataset size: {len(calib_ds)}")
    print(f"Evaluation dataset size: {len(eval_ds)}")

    # Run model on calibration dataset to get uncalibrated probabilities
    print("Running model on calibration dataset...")
    predictions_for_attribution = eval_correctness_dataset(calib_ds, model, tokenizer, device=args.device, \
        save_predictions_path=args.save_calibration_predictions_path, verbose=False, return_var="predictions_for_attribution", \
        verbalized=args.verbalized, vllm_sampling_params=vllm_sampling_params, use_reasoning=args.use_reasoning)
    # Process predictions_for_attribution into format needed for calibration
    calibset_preds_uncalib_1, y_calib_1 = process_predictions_for_calibration(predictions_for_attribution)
    print(f"Calibration set: {len(calibset_preds_uncalib_1)} examples")
    print(f"True label distribution: {np.mean(y_calib_1):.3f} fraction 'yes'")
    print(f"Mean uncalibrated prob_yes: {np.mean(calibset_preds_uncalib_1):.3f}")

    # Similarly process evaluation dataset to get test set predictions
    print("Running model on evaluation dataset...")
    if args.load_eval_probabilities_path:
        eval_predictions_for_attribution = GradingDataset.load_predictions_for_attribution(args.load_eval_probabilities_path)
    else:
        eval_predictions_for_attribution = eval_correctness_dataset(eval_ds, model, tokenizer, device=args.device, \
            save_predictions_path=None, verbose=False, return_var="predictions_for_attribution", \
            verbalized=args.verbalized, vllm_sampling_params=vllm_sampling_params, use_reasoning=args.use_reasoning)
    testset_preds_uncalib_1, y_test_1 = process_predictions_for_calibration(eval_predictions_for_attribution)
    print(f"Test set: {len(testset_preds_uncalib_1)} examples")
    print(f"True label distribution: {np.mean(y_test_1):.3f} fraction 'yes'")
    print(f"Mean uncalibrated prob_yes: {np.mean(testset_preds_uncalib_1):.3f}")

    if args.calibration_type == "platt":
        # Taken from https://www.youtube.com/watch?v=7h1-muiW97s&t=816s
        # Fit Platt scaling (logistic calibration)
        lr = LogisticRegression(C=99999999999, solver='lbfgs')
        lr.fit(calibset_preds_uncalib_1.reshape(-1,1), y_calib_1)

        calibset_platt_probs = lr.predict_proba(calibset_preds_uncalib_1.reshape(-1,1))[:,1]
        testset_platt_probs = lr.predict_proba(testset_preds_uncalib_1.reshape(-1,1))[:,1]

        print(f"\nPlatt calibration diagnostics:")
        print(f"  Calibset probs - Min: {np.min(calibset_platt_probs):.6f}, Max: {np.max(calibset_platt_probs):.6f}")
        print(f"  Testset probs - Min: {np.min(testset_platt_probs):.6f}, Max: {np.max(testset_platt_probs):.6f}")
        print(f"  LogReg coef: {lr.coef_[0][0]:.6f}, intercept: {lr.intercept_[0]:.6f}")

        testset_calibrated_probs = testset_platt_probs
        calibset_calibrated_probs = calibset_platt_probs

        tvec = np.linspace(.01, .99, 99)
        tvec_pred = lr.predict_proba(tvec.reshape(-1,1))[:,1]
        eval_calibration_fit(y_calib_1=y_calib_1, calibset_preds_uncalib_1=calibset_preds_uncalib_1, y_test_1=y_test_1, \
            testset_preds_uncalib_1=testset_preds_uncalib_1, testset_calibrated_probs=testset_calibrated_probs, tvec=tvec, \
            tvec_pred=tvec_pred, args=args)
    elif args.calibration_type == "isotonic_regression":
        iso = IsotonicRegression(out_of_bounds = 'clip')
        iso.fit(calibset_preds_uncalib_1, y_calib_1)
        calibset_calibrated_probs = iso.predict(calibset_preds_uncalib_1)
        testset_calibrated_probs = iso.predict(testset_preds_uncalib_1)
        tvec = np.linspace(.01, .99, 99)
        tvec_pred = iso.predict(tvec)
        eval_calibration_fit(y_calib_1=y_calib_1, calibset_preds_uncalib_1=calibset_preds_uncalib_1, y_test_1=y_test_1, \
            testset_preds_uncalib_1=testset_preds_uncalib_1, testset_calibrated_probs=testset_calibrated_probs, tvec=tvec, \
            tvec_pred=tvec_pred, args=args)
    elif args.calibration_type == "beta_calibration":
        # Fit three-parameter beta calibration
        bc = BetaCalibration()
        bc.fit(calibset_preds_uncalib_1, y_calib_1)
        calibset_calibrated_probs = bc.predict(calibset_preds_uncalib_1)
        testset_calibrated_probs = bc.predict(testset_preds_uncalib_1)
        tvec = np.linspace(.01, .99, 99)
        tvec_pred = bc.predict(tvec)
        eval_calibration_fit(y_calib_1=y_calib_1, calibset_preds_uncalib_1=calibset_preds_uncalib_1, y_test_1=y_test_1, \
            testset_preds_uncalib_1=testset_preds_uncalib_1, testset_calibrated_probs=testset_calibrated_probs, tvec=tvec, \
            tvec_pred=tvec_pred, args=args)
    elif args.calibration_type == "spline_calibration":
        # Define SplineCalib object with optional knot_sample_size and reg_param configuration
        spline_kwargs = {}
        
        if args.knot_sample_size is not None:
            print(f"Using spline calibration with knot_sample_size={args.knot_sample_size}")
            spline_kwargs['knot_sample_size'] = args.knot_sample_size
        
        # Configure regularization parameter grid
        if args.reg_param_num_values is not None:
            reg_param_vec = np.logspace(-4, 4, args.reg_param_num_values)  # 0.0001 to 10000
            spline_kwargs['reg_param_vec'] = reg_param_vec
            print(f"Using {args.reg_param_num_values} regularization parameter values from {reg_param_vec[0]:.6f} to {reg_param_vec[-1]:.6f}")
        
        # Configure unity prior
        if args.spline_unity_prior:
            spline_kwargs['unity_prior'] = True  # This enables unity prior in ml_insights SplineCalib
            print("Using spline calibration with unity prior")
        
        if spline_kwargs:
            splinecalib = mli.SplineCalib(**spline_kwargs)
        else:
            print("Using spline calibration with default settings")
            splinecalib = mli.SplineCalib()
        
        splinecalib.fit(calibset_preds_uncalib_1, y_calib_1)
        calibset_calibrated_probs = splinecalib.predict(calibset_preds_uncalib_1)
        testset_calibrated_probs = splinecalib.predict(testset_preds_uncalib_1)
        tvec = np.linspace(.01, .99, 99)
        tvec_pred = splinecalib.predict(tvec)
        eval_calibration_fit(y_calib_1=y_calib_1, calibset_preds_uncalib_1=calibset_preds_uncalib_1, y_test_1=y_test_1, \
            testset_preds_uncalib_1=testset_preds_uncalib_1, testset_calibrated_probs=testset_calibrated_probs, tvec=tvec, \
            tvec_pred=tvec_pred, args=args)

    """Save testset_calibrated_probs as the new probabilities for eval_predictions_for_attribution"""
    for idx, entry in enumerate(eval_predictions_for_attribution):
        entry["prob_yes"] = float(testset_calibrated_probs[idx])
        entry["prob_no"] = float(1.0 - testset_calibrated_probs[idx])
    if args.save_calibrated_eval_predictions_path is not None:
        print(f"Saving calibrated evaluation predictions:")
        GradingDataset.save_predictions_for_attribution(eval_predictions_for_attribution, args.save_calibrated_eval_predictions_path)

    """Evaluate predictions"""
    calibrated_results = evaluate_predictions(
        calibrated_probs=testset_calibrated_probs,
        true_labels=y_test_1, 
        save_dir=args.save_dir,
        plot_suffix="_calibrated",
        verbose=True
    )

    # Evaluate uncalibrated predictions for comparison
    uncalibrated_results = evaluate_predictions(
        calibrated_probs=testset_preds_uncalib_1,
        true_labels=y_test_1,
        save_dir=args.save_dir, 
        plot_suffix="_uncalibrated",
        verbose=True
    )

    # Print comparison
    print("\n" + "="*50)
    print("SUMMARY COMPARISON")
    print("="*50)
    print(f"{'Metric':<15} {'Uncalibrated':<15} {'Calibrated':<15} {'Improvement':<15}")
    print("-" * 60)
    for metric in ['average_correct', 'ece', 'rmsce', 'auroc']:
        uncal_val = uncalibrated_results.get(metric, 'N/A')
        cal_val = calibrated_results.get(metric, 'N/A')
        if uncal_val != 'N/A' and cal_val != 'N/A':
            # Mark and calculate the polarity of the metric, high is better or lower is better.
            improvement = cal_val - uncal_val if metric == 'average_correct' or metric == 'auroc' else uncal_val - cal_val
            print(f"{metric:<15} {uncal_val:<15.4f} {cal_val:<15.4f} {improvement:<15.4f}")
        else:
            print(f"{metric:<15} {uncal_val:<15} {cal_val:<15} {'N/A':<15}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    # --- dataset-loading args (mirrored from standard_tuning) -----------------
    p.add_argument("--dataset", default="CORRECTNESS_TUNING")
    p.add_argument("--datafile", type=str, default=None)
    p.add_argument("--ntrain", type=int, default=0)
    p.add_argument("--eval_start_p", type=float, default=0.75)
    p.add_argument("--train_end_p", type=float, default=0.70)
    p.add_argument("--train_on_all_tokens", action="store_true")
    p.add_argument("--model", default="gemma-2b", help="Model to post hoc calibrate")
    p.add_argument("--load_lora", action="store_true")
    p.add_argument("--load_existing_model", action="store_true")
    p.add_argument("--exclude_from_train", type=str, nargs='*', default=[], help="List of subjects to exclude from training dataset")
    p.add_argument("--checkpoints_dir", type=str)
    p.add_argument("--use_model_name_in_prompt", action="store_true", help="If true, replaces 'Model Alpha' with model name from dataset metadata.")
    p.add_argument("--specialty_prompttype", type=str, default=None, help="Specialty prompt type for prompt generation")
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--group_by_question", action="store_true", 
                        help="Group all examples with the same input_prompt together for optimization. Mutually exclusive with --shuffle.")

    # RETRIEVAL ARGUMENTS (mirrored from standard_tuning)
    p.add_argument("--chroma_path", type=str, default=None, help="Path to store/load the Chroma database")
    p.add_argument("--collection_name", type=str, default="training-examples", help="Name of the Chroma collection")
    p.add_argument("--k_retrieve", type=int, default=15, help="Number of similar examples to retrieve for each query")
    p.add_argument("--chroma_source_datafile", type=str, default=None, help="Path to source dataset for building Chroma collection (if different from main datafile)")
    p.add_argument("--embedding_function", type=str, default=None, help="Embedding function to use for ChromaDB collection (None for default, 'ReasonIR' for ReasonIR-8B)")

    # DATASET CACHING ARGUMENTS (mirrored from standard_tuning)
    p.add_argument("--dataset_save_path", type=str, default=None, help="Path to save processed datasets to (using pickle), this path is for caching only, we will only load if dataset_load_path is specified.")
    p.add_argument("--dataset_load_path", type=str, default=None, help="Path to load cached datasets from (using pickle), we'll only load if this path is specified.")

    # VLLM arguments (mirrored from standard_tuning)
    p.add_argument("--verbalized", action="store_true", help="Load an VLLM model, instead of predicting yes/no, predict a probability percentage. This is bundled with eval_only.")
    p.add_argument("--use_reasoning", action="store_true", help="Whether the verbalized eval_correctness should use reasoning tokens.")
    p.add_argument("--tensor_parallel_size", type=int, default=4, help="VLLM argument")
    p.add_argument("--max_seq_length", type=int, default=32768, help="Maximum sequence length for model, when running eval_correctness in verbalized mode")
    
    # Calibration tuning params
    p.add_argument("--calibration_p", type=float, default=0.05, help="This will select examples from the slice between [train_end_p, train_end_p+calibration_p], we will run the model to collect a similar information to what eval_correctness_dataset produces for the eval_dataset")
    p.add_argument("--calibration_type", default="platt")
    p.add_argument("--knot_sample_size", type=int, default=None, help="Number of knots to use for spline calibration. If None, uses default behavior of SplineCalib")
    p.add_argument("--reg_param_num_values", type=int, default=None, help="Number of regularization parameter values to try for spline calibration (log-spaced between 0.0001 and 10000). Default 100 for better results than ml_insights default of 17")
    p.add_argument("--spline_unity_prior", action="store_true", help="Use unity prior for spline calibration (sets prior_prob=None to use uniform prior)")
    p.add_argument("--save_calibration_predictions_path", default=None, required=True)
    p.add_argument("--load_eval_probabilities_path", default=None, type=str, help="If the model was a lora model previously trained, then we may load the probabilities of the model from output of standard_tuning.py, but is equally possible to simply rerun the model to get raw uncalibrated probabilities to be transformed by specifying None for this argument.")
    p.add_argument("--save_calibrated_eval_predictions_path", default=None, type=str, help="Path to save calibrated evaluation predictions for attribution")
    p.add_argument("--save_dir", default=None, type=str, help="Directory to save calibration plots and metrics")
    p.add_argument("--disable_include_model_response", action="store_true")
    p.add_argument("--override_splits", action="store_true", 
                   help="If true, subset calibration split according to calibration_p, p is relative to full loaded jsonl dataset size.")

    # MISC
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    
    # Check for mutually exclusive arguments
    if args.shuffle and args.group_by_question:
        p.error("--shuffle and --group_by_question are mutually exclusive. Choose one or neither. --shuffle undoes the organization done by grouping.")
    
    main(args)
