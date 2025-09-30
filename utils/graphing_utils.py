import numpy as np
from dynamics_project.utils.calib_tools import rmsce, ece
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
import os
import numpy as np

def plot_calibration_graphs(per_model_cors: np.ndarray, 
                            per_model_max_probs: np.ndarray, 
                            model: str, 
                            args: argparse.Namespace, 
                            extra_data_dict: dict = {}) -> None:
    """Plot calibration graphs and statistics for a given model and its predictions.

    Args:
        per_model_cors (numpy array): Correctness values for each layer's predictions.
        per_model_max_probs (numpy array): Maximum confidence values for each layer's predictions.
        model (str): Name of the model.
        args (argparse.Namespace): Parsed arguments for the model.
        extra_data_dict (dict, optional): Extra data dictionary for additional information.

    Returns:
        None

    Todos:
        - Implement additional statistical tests or analyses for further insights.
    """
    # Calculate calibration statistics by layer
    from sklearn.metrics import roc_auc_score, roc_curve, auc, RocCurveDisplay
    per_model_cors = np.array(per_model_cors).transpose([1,0]).astype(int)
    per_model_max_probs = np.array(per_model_max_probs).transpose([1,0,2])        
    per_model_ece = []
    per_model_rmsce = []
    per_model_acc_layerwise = []
    per_model_conf_layerwise = []
    per_model_auc_roc_layerwise = []
    per_model_MAE_layerwise = []

    # AUC ROC for every layer: calculate ground truth labels for all examples based on last layer
    predictions = (per_model_max_probs[-1,:,0] > 0.5).astype(int)
    total_labels = np.where(per_model_cors[-1,:], predictions, 1 - predictions)

    for layer in range(per_model_cors.shape[0]):
        # Per layer for all examples
        layer_for_all_examples_rmsce = rmsce(per_model_cors[layer,:], np.max(per_model_max_probs[layer,:,:], axis=-1))
        layer_for_all_examples_ece = ece(per_model_cors[layer,:], np.max(per_model_max_probs[layer,:,:], axis=-1))

        per_model_acc_layerwise.append(np.mean(per_model_cors[layer,:]))
        per_model_conf_layerwise.append(np.mean(np.max(per_model_max_probs[layer,:,:], axis=-1)))
        per_model_MAE_layerwise.append(np.mean(np.abs(total_labels-per_model_max_probs[layer,:,0])))

        per_model_rmsce.append(layer_for_all_examples_rmsce)
        per_model_ece.append(layer_for_all_examples_ece)

        # AUC ROC for every layer
        layer_roc_auc_score = roc_auc_score(total_labels, per_model_max_probs[layer,:,0])
        per_model_auc_roc_layerwise.append(layer_roc_auc_score)
        print(f"\nLayer_roc_auc_score layer {layer}:", layer_roc_auc_score)
        print(f"MAE score layer {layer}:", np.abs(total_labels-per_model_max_probs[layer,:,0]))
        print(f"Accuracy score layer {layer}:", np.mean(per_model_cors[layer,:]))

        if layer == per_model_cors.shape[0] - 1:
            # Calculate AUC ROC and Plot
            plt.clf()
            fpr, tpr, thresholds = roc_curve(total_labels, per_model_max_probs[layer,:,0])
            # print(thresholds, all_probs)
            roc_auc = auc(fpr, tpr)
            display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                            estimator_name=f'{model} {args.probe_type}')
            display.plot()
            plt.savefig("{}/{}_{}roc_plot.png".format(args.out_dir, model, args.probe_type), bbox_inches='tight')
            print("Final layer AUCROC:", layer_roc_auc_score)
            plt.clf()
            

    # Plot confidence vs correctness violin plots
    last_layer_is_correct = per_model_cors[-1,:]
    # print("per_model_cors[-1,:].shape", per_model_cors.shape, per_model_cors[-1,:].shape)
    # print("per_model_max_probs[:,-1,:].shape", per_model_max_probs.shape, per_model_max_probs[-1,:,:].shape, per_model_max_probs[-1,:,:], np.max(per_model_max_probs[-1,:,:], axis=-1))
    last_layer_confidences = np.max(per_model_max_probs[-1,:,:], axis=-1)
    confidence_violin_plot(last_layer_is_correct, last_layer_confidences, model, args)


    extra_data_dict = {}
    run_graph = np.all([key in extra_data_dict for key in ["all_accs", "all_confs", "all_cors", "all_max_probs"]])
    if run_graph:
        # Calculate last layer calibration statistics (to ensure they match with those calculated via hidden states)
        avg_max_prob = np.mean(extra_data_dict["all_max_probs"])
        acc = np.mean(extra_data_dict["all_cors"])
        rms_ce = rmsce(np.array(extra_data_dict["all_cors"]), np.array(extra_data_dict["all_max_probs"]))
        expected_calibration_error = ece(np.array(extra_data_dict["all_cors"]), np.array(extra_data_dict["all_max_probs"]))
        print("{} average max confidence: {:.3f}, accuracy: {:.3f}, RMS-CE: {:.3f}, ECE: {:.3f}".format(model, avg_max_prob, acc, rms_ce, expected_calibration_error))

        # Plot confidence accuracy graph for last layer
        plt.clf()
        plt.scatter(extra_data_dict["all_confs"], extra_data_dict["all_accs"])
        min = np.minimum(np.min(extra_data_dict["all_confs"]), np.min(extra_data_dict["all_accs"]))
        max = np.maximum(np.max(extra_data_dict["all_confs"]), np.max(extra_data_dict["all_accs"]))
        x = np.arange(min, max, 0.01)
        y = np.arange(min, max, 0.01)
        plt.plot(x, y, c="r")
        plt.xlabel("Confidence")
        plt.ylabel("Accuracy")
        plt.title(f"Confidence Accuracy {model} {args.probe_type} (points are averages by subject)")
        plt.savefig("{}/{}_{}calibration.png".format(args.out_dir, model, args.probe_type), bbox_inches='tight')
        print("graphs saved at {}/{}_{}calibration.png".format(args.out_dir, model, args.probe_type))

    # Plot RMS-CE and ECE change for all layers
    num_layers = len(per_model_ece)

    fig, ax = plt.subplots(figsize=(10, 8))

    # First heatmap for ECE
    im1 = ax.imshow(np.vstack([per_model_ece]).T, aspect='auto', extent=[-0.5, 0.5, -0.5, num_layers - 0.5], origin='lower', cmap='Reds', interpolation='nearest')
    # Add text labels for each pixel value
    for i in range(num_layers):
        ax.text(0, i, f'{per_model_ece[i]:.3f}', ha='center', va='center', color='black', fontsize=8)
        
    # # Second heatmap for RMSCE
    # im2 = ax.imshow(np.vstack([per_model_rmsce]).T, aspect='auto', extent=[0.5, 1.5, -0.5, num_layers - 0.5], origin='lower', cmap='Oranges', interpolation='nearest')
    # # Add text labels for each pixel value
    # for i in range(num_layers):
    #     ax.text(1, i, f'{per_model_rmsce[i]:.3f}', ha='center', va='center', color='black', fontsize=8)

    # Second heatmap for AUCROC
    im2 = ax.imshow(np.vstack([per_model_auc_roc_layerwise]).T, aspect='auto', extent=[0.5, 1.5, -0.5, num_layers - 0.5], origin='lower', cmap='Oranges', interpolation='nearest')
    # Add text labels for each pixel value
    for i in range(num_layers):
        ax.text(1, i, f'{per_model_auc_roc_layerwise[i]:.3f}', ha='center', va='center', color='black', fontsize=8)

    # Third heatmap for Average Grading Correctness
    im3 = ax.imshow(np.vstack([per_model_MAE_layerwise]).T, aspect='auto', extent=[1.5, 2.5, -0.5, num_layers - 0.5], origin='lower', cmap='Reds', interpolation='nearest')
    # Add text labels for each pixel value
    for i in range(num_layers):
        ax.text(2, i, f'{per_model_MAE_layerwise[i]:.3f}', ha='center', va='center', color='black', fontsize=8)

    # Fourth heatmap for Average Confidence
    im4 = ax.imshow(np.vstack([per_model_conf_layerwise]).T, aspect='auto', extent=[2.5, 3.5, -0.5, num_layers - 0.5], origin='lower', cmap='Blues', interpolation='nearest')
    # Add text labels for each pixel value
    for i in range(num_layers):
        ax.text(3, i, f'{per_model_conf_layerwise[i]:.3f}', ha='center', va='center', color='black', fontsize=8)

    # # Fifth heatmap for Average Accuracy
    im5 = ax.imshow(np.vstack([per_model_acc_layerwise]).T, aspect='auto', extent=[3.5, 4.5, -0.5, num_layers - 0.5], origin='lower', cmap='Greens', interpolation='nearest')
    # Add text labels for each pixel value
    for i in range(num_layers):
        ax.text(4, i, f'{per_model_acc_layerwise[i]:.3f}', ha='center', va='center', color='black', fontsize=8)

    # # Add colorbars
    # cbar1 = fig.colorbar(im1, ax=ax, label='ECE')
    # # cbar2 = fig.colorbar(im2, ax=ax, label='RMSCE')
    # cbar2 = fig.colorbar(im2, ax=ax, label='AUCROC')
    # cbar3 = fig.colorbar(im3, ax=ax, label='Avg Grading MAE')
    # cbar4 = fig.colorbar(im4, ax=ax, label='Avg Grading Confidence')
    # cbar5 = fig.colorbar(im5, ax=ax, label='Avg Accuracy')

    # Set plot properties
    ax.set_xlim(-0.5, 4.5)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xticklabels(['ECE', 'AUCROC', 'Avg Grading MAE', 'Avg Grading Confidence', "Avg Accuracy"], rotation=45)  # used to be RMSCE
    ax.set_yticks(range(num_layers))
    # Assuming you have a list of layer names
    layer_names = range(num_layers)  # Provide your list of layer names here
    ax.set_yticklabels(layer_names)
    ax.set_ylabel('Layers')
    ax.set_title("{} {} layerwise calibration".format(model, args.probe_type))
    fig.tight_layout()
    plt.savefig("{}/{}_{}layerwise_calibration.png".format(args.out_dir, model, args.probe_type), bbox_inches='tight')




def confidence_violin_plot(last_layer_is_correct, last_layer_confidences, model, args):
    # Plot confidence vs correctness violin plots
    data = pd.DataFrame({
        'Correctness': last_layer_is_correct,
        'Confidence': last_layer_confidences
    })
    # print("Raw violin plot data", data[:5])
    data['Correctness'] = data['Correctness'].apply(lambda x: 'Correct' if x == 1 else 'Incorrect')
    plt.clf()
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Correctness', y='Confidence', data=data, split=False, inner='stick', inner_kws={"linewidths":.0001})

    # Calculating and overlaying quartiles
    categories = data['Correctness'].unique()
    for i, category in enumerate(categories):
        subset = data[data['Correctness'] == category]['Confidence']
        q1 = subset.quantile(0.25)
        q2 = subset.quantile(0.50)
        q3 = subset.quantile(0.75)
        plt.plot([i - 0.05, i + 0.05], [q1, q1], color='red', lw=2)  # Make quartile lines red and thicker
        plt.plot([i - 0.05, i + 0.05], [q2, q2], color='red', lw=2)
        plt.plot([i - 0.05, i + 0.05], [q3, q3], color='red', lw=2)

    plt.title(f'Confidence Distribution by Correctness {model} {args.probe_type}')
    plt.xlabel('Correctness')
    plt.ylabel('Confidence Level')
    plt.savefig("{}/{}_{}confidence_vs_correctness_violin.png".format(args.out_dir, model, args.probe_type), bbox_inches='tight')
    print("confidence vs correctness violin plots saved at {}/{}_{}violin.png".format(args.out_dir, model, args.probe_type))


def create_similarity_plot(per_subject_A_to_B_vectors, results_savedir, args, size_multiplier = 2):
    # Create Plot
    # Convert tensors to numpy arrays and store in a list
    vectors = [v.detach().to("cpu").numpy() for v in per_subject_A_to_B_vectors.values()]

    # Calculate cosine similarity matrix
    num_vectors = len(vectors)
    similarity_matrix = np.zeros((num_vectors, num_vectors))
    for i in range(num_vectors):
        for j in range(num_vectors):
            similarity_matrix[i, j] = np.dot(vectors[i], vectors[j]) / (np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j]))

    # Create plot
    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(figsize=(6.4*size_multiplier, 4.8*size_multiplier))
    im = ax.imshow(similarity_matrix, cmap="viridis", vmin=-1, vmax=1)

    # Show numbers above each square
    for i in range(num_vectors):
        for j in range(num_vectors):
            text = ax.text(j, i, f"{similarity_matrix[i, j]:.2f}", ha="center", va="center", color="w")

    # Set tick labels to subject IDs
    ax.set_xticks(np.arange(num_vectors))
    ax.set_yticks(np.arange(num_vectors))
    ax.set_xticklabels(per_subject_A_to_B_vectors.keys())
    ax.set_yticklabels(per_subject_A_to_B_vectors.keys())

    # Rotate tick labels and align them properly
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add colorbar and labels
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Cosine Similarity", rotation=-90, va="bottom")
    ax.set_xlabel("Group")
    ax.set_ylabel("Group")

    # Display the plot
    plt.tight_layout()
    if args.testing:
        save_path = os.path.join(results_savedir, f"fig_{args.model}_{args.type}_subsequence_A_to_B_vectors_testing.png")
    else:
        save_path = os.path.join(results_savedir, f"fig_{args.model}_{args.type}_subsequence_A_to_B_vectors.png")
    plt.savefig(save_path, bbox_inches='tight')


def plot_raw_vectors(per_subject_A_to_B_vectors, results_savedir, args, size_multiplier=3):
    # Convert tensors to numpy arrays and store in a list
    vectors = [v.unsqueeze(0).detach().to("cpu").numpy() for v in per_subject_A_to_B_vectors.values()]
    
    # Number of subjects
    num_subjects = len(vectors)
    
    # Create plot
    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(figsize=(6.4*size_multiplier+3, 4.8*size_multiplier))
    ax.set_xticks([])
    ax.set_yticks([])
    subjects = list(per_subject_A_to_B_vectors.keys())

    # Plot each vector as an image
    for i, vector in enumerate(vectors):
        ax = fig.add_subplot(num_subjects, 1, i+1)
        im = ax.imshow(vector, aspect='auto', interpolation='nearest', cmap='winter')
        ax.text(1, 0.5, f'{list(per_subject_A_to_B_vectors.keys())[i]}', va='center', ha='right', transform=ax.transAxes, fontsize=12, rotation=0)
        ax.set_yticks([])

    # Add colorbar
    ax.set_xlabel("Vector dimensions")
    cbar = fig.colorbar(im, ax=fig.get_axes(), orientation='horizontal')
    cbar.set_label("Raw Values")
    
    # Save the plot
    if args.testing:
        save_path = os.path.join(results_savedir, f"fig_{args.model}_{args.type}_raw_vectors_testing.png")
    else:
        save_path = os.path.join(results_savedir, f"fig_{args.model}_{args.type}_raw_vectors.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()