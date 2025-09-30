import json
import os
from matplotlib import pyplot as plt
import numpy as np
import torch

"""Display"""
# Function to printed sorted components in terms of importance
def importance_plot(args, accumulated_gradients, sort=True, plot_name=None, aggregation_function=torch.mean, plot_path=None):
    ## Calculate one value for each matrix for each parameter's gradients from accumulated_gradients
    gradient_importance = {
        name: aggregation_function(torch.abs(grad)).item() 
        for name, grad in accumulated_gradients.items()
    }
    ## Compute statistics from accumulated_gradients
    statistics = {}
    for name, matrix in accumulated_gradients.items():
        flattened = matrix.flatten().to(torch.float32)
        flattened = torch.abs(flattened)
        # Compute statistics
        mean = torch.mean(flattened)
        max_value = torch.max(flattened)
        min_value = torch.min(flattened)
        std = torch.std(flattened)
        # quantiles = torch.quantile(flattened, torch.tensor([0.25, 0.5, 0.75]))
        statistics[name] = (mean, max_value, min_value, std)
    # Sanity Check
    for name, grad in accumulated_gradients.items():
        if grad == None:
            args.logger.info(f"{name}GRAD IS NONE<")
    ## Sort by average magnitude in descending order
    if sort:
        sorted_gradients = dict(sorted(
            gradient_importance.items(),
            key=lambda x: x[1],
            reverse=True
        ))
    else:
        sorted_gradients = gradient_importance
    ## Prepare data for plotting
    names = list(sorted_gradients.keys())
    values = list(sorted_gradients.values())
    lower_errors = [statistics[name][0].item() - statistics[name][2].item() for name in names]
    upper_errors = [statistics[name][1].item() - statistics[name][0].item() for name in names]
    values_array = np.array(values)
    normalized_values = (values_array - values_array.min()) / (values_array.max() - values_array.min())
    colors = plt.cm.viridis(normalized_values)
    ## Create visualization
    plt.figure(figsize=(20, 25))
    names = list(sorted_gradients.keys())
    values = list(sorted_gradients.values())
    # Create bar plot
    names = list(sorted_gradients.keys())
    values = list(sorted_gradients.values())
    lower_errors = [statistics[name][0].item() - statistics[name][2].item() for name in names]
    upper_errors = [statistics[name][1].item() - statistics[name][0].item() for name in names]
    xerr = np.array([lower_errors, upper_errors])
    bars = plt.barh(range(len(names)), values, color=colors)
    # Customize plot labels
    plt.ylabel('Parameter Names')
    plt.xlabel('Average Gradient Magnitude')
    plt.title('Parameter Importance by Average Gradient Magnitude')
    plt.yticks(range(len(names)), names, fontsize=8)
    # Add value labels on the bars
    for i, v in enumerate(values):
        min_value = statistics[names[i]][2].item()
        max_value = statistics[names[i]][1].item()
        std = statistics[names[i]][3].item()
        plt.text(v, i, f'{v:.3e} | (min:{min_value:.2e}, max:{max_value:.2e}, std:{std:.2e})', va='center', fontsize=8)
        # plt.text(v, i, f'{v:.3e}', va='center', fontsize=8)
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    # Save plot
    if not plot_path:
        plot_path = os.path.join(args.results_dir, args.run_name, f"{f'{plot_name}_' if plot_name else ''}by_module_{args.serial_number}.png")
    plt.savefig(plot_path)
    plt.close()
    args.logger.info(f"Gradient importance plot saved to {plot_path}")
    return sorted_gradients


def plot_layer_differences(differences: dict, save_path: str, title: str = "Layer-wise Differences"):
    """
    Plot the zd (scalar) differences for each layer in a horizontal bar chart.
    
    Args:
        differences: Dictionary with layer names as keys and dict containing 'zd' values as values
        save_path: Path to save the plot
        title: Title for the plot
    """
    # Extract layer names and zd values
    names = list(differences.keys())
    values = [differences[name]['zd'].to("cpu") for name in names]
    
    # Create normalized colors
    values_array = np.array(values)
    normalized_values = (values_array - values_array.min()) / (values_array.max() - values_array.min())
    colors = plt.cm.viridis(normalized_values)
    
    # Create visualization
    plt.figure(figsize=(20, 25))
    
    # Create bar plot
    bars = plt.barh(range(len(names)), values, color=colors)
    
    # Customize plot labels
    plt.ylabel('Layer Names')
    plt.xlabel('Aggregate Difference (zd)')
    plt.title(title)
    plt.yticks(range(len(names)), names, fontsize=8)
    
    # Add value labels on the bars
    for i, v in enumerate(values):
        plt.text(v, i, f'{v:.4f}', va='center', fontsize=8)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path)
    plt.close()
    print(f"Layer differences plot saved to {save_path}")


"""#Visualization Plots"""

import torch.nn.functional as F

def visualize_two_matrices(
    args,
    matrix_name: str,
    matrix_type_1: str,
    importances_1: torch.Tensor,
    matrix_type_2: str,
    importances_2: torch.Tensor,
    sort_by: str = 'none',  # 'none', 'row', 'col', 'flatten'
    pool_size: int = 1,
    cmap: str = 'seismic',
    bins: int = 50,
    save_path: str = None
) -> None:
    """
    Visualize two importance matrices side by side, their difference, and histograms of their distributions.
    White is anchored at zero. Histograms are on a log scale.
    Matrices are padded before pooling if required.
    """
    # Move tensors to CPU and detach
    imp1 = importances_1.detach().cpu()
    imp2 = importances_2.detach().cpu()

    # Create difference
    diff = imp1 - imp2

    # Convert to numpy
    imp1_np = imp1.numpy()
    imp2_np = imp2.numpy()
    diff_np = diff.numpy()

    # Sorting logic
    def sort_by_row(imp_ref: np.ndarray):
        # Sort rows by the mean absolute magnitude
        row_mags = np.sum(np.abs(imp_ref), axis=1)
        sorted_indices = np.argsort(-row_mags)  # descending order
        return sorted_indices, None

    def sort_by_col(imp_ref: np.ndarray):
        # Sort columns by mean absolute magnitude
        col_mags = np.sum(np.abs(imp_ref), axis=0)
        sorted_indices = np.argsort(-col_mags)
        return None, sorted_indices

    def sort_by_flatten(imp_ref: np.ndarray):
        # Flatten and sort by magnitude
        flat = imp_ref.flatten()
        sorted_indices = np.argsort(-np.abs(flat))
        return sorted_indices

    # Apply sorting if needed
    if sort_by == 'row':
        row_idx, _ = sort_by_row(imp1_np)
        imp1_np = imp1_np[row_idx, :]
        imp2_np = imp2_np[row_idx, :]
        diff_np = diff_np[row_idx, :]
    elif sort_by == 'col':
        _, col_idx = sort_by_col(imp1_np)
        imp1_np = imp1_np[:, col_idx]
        imp2_np = imp2_np[:, col_idx]
        diff_np = diff_np[:, col_idx]
    elif sort_by == 'flatten':
        rows, cols = imp1_np.shape
        fidx = sort_by_flatten(imp1_np)
        imp1_flat = imp1_np.flatten()[fidx]
        imp2_flat = imp2_np.flatten()[fidx]
        diff_flat = diff_np.flatten()[fidx]
        imp1_np = imp1_flat.reshape(rows, cols)
        imp2_np = imp2_flat.reshape(rows, cols)
        diff_np = diff_flat.reshape(rows, cols)
    elif sort_by == "none":
        pass
    else:
        raise Exception(f"Sorting order not specified, {sort_by}")

    def pad_for_pooling(arr: np.ndarray, pool: int):
        if pool <= 1:
            return arr
        # Pad so that arr.shape is divisible by pool_size
        r, c = arr.shape
        pad_rows = (pool - (r % pool)) % pool
        pad_cols = (pool - (c % pool)) % pool
        print("pad_rows, pad_cols", pad_rows, pad_cols)
        t = torch.tensor(arr, dtype=torch.float64).unsqueeze(0).unsqueeze(0)
        t = F.pad(t, (0, pad_cols, 0, pad_rows), mode='constant', value=float('-inf'))
        # After pooling, we will get rid of the padded rows/cols by slicing back if necessary
        return t

    def apply_pooling(t: torch.Tensor, pool: int, orig_shape):
        if pool <= 1:
            return t
        pooled = F.max_pool2d(t, kernel_size=pool)
        # We padded with -inf, so pooling won't select these unless everything is -inf.
        # After pooling, it’s smaller; no need to slice because we padded to a multiple of pool.
        return pooled.squeeze(0).squeeze(0).numpy()

    # Pad and pool
    orig_shape = imp1_np.shape
    t1 = pad_for_pooling(imp1_np, pool_size)
    t2 = pad_for_pooling(imp2_np, pool_size)
    td = pad_for_pooling(diff_np, pool_size)

    print("Shapes", t1.shape, t2.shape, td.shape)
    pooled_imp1 = apply_pooling(t1, pool_size, orig_shape)
    pooled_imp2 = apply_pooling(t2, pool_size, orig_shape)
    pooled_diff = apply_pooling(td, pool_size, orig_shape)

    # Prepare figure
    fig, axs = plt.subplots(2, 3, figsize=(36, 24))
    fig.suptitle(
        f"Visualizations of {matrix_name}\n"
        f"{matrix_type_1} vs {matrix_type_2}, pool_size={pool_size}, sort_by={sort_by}",
        fontsize=20
    )

    # Plot heatmaps
    # imp1
    max_val = np.max(np.abs(pooled_imp1))
    vmin, vmax = -max_val, max_val
    im1 = axs[0, 0].imshow(pooled_imp1, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    axs[0, 0].set_title(f"{matrix_type_1}", fontsize=16)
    fig.colorbar(im1, ax=axs[0, 0])

    # imp2
    max_val = np.max(np.abs(pooled_imp2))
    vmin, vmax = -max_val, max_val
    im2 = axs[0, 1].imshow(pooled_imp2, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    axs[0, 1].set_title(f"{matrix_type_2}", fontsize=16)
    fig.colorbar(im2, ax=axs[0, 1])

    # difference
    max_val = np.max(np.abs(pooled_diff))
    vmin, vmax = -max_val, max_val
    im3 = axs[0, 2].imshow(pooled_diff, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    axs[0, 2].set_title(f"Difference ({matrix_type_1} - {matrix_type_2})", fontsize=16)
    fig.colorbar(im3, ax=axs[0, 2])

    # Histograms with log scale
    def plot_hist(ax, data, title):
        if len(data[np.isfinite(data)]) < np.prod(data.shape):
            print(f"Filtering out {np.prod(data.shape) - len(data[np.isfinite(data)])} datapoints, due to being infinite")
            data = data[np.isfinite(data)]
        # Use log scale on y-axis
        hist_values, _, _ = ax.hist(data.flatten(), bins=bins, color='blue', alpha=0.7)
        ax.set_yscale('log')
        ax.set_title(title, fontsize=16)
        # Summary stats
        dmin, dmax = data.min(), data.max()
        ax.annotate(
            f"min={dmin:.4e}\nmax={dmax:.4e}",
            xy=(0.7, 0.7),
            xycoords='axes fraction',
            fontsize=12,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.5)
        )

    # Original (not pooled) data hist
    plot_hist(axs[1, 0], imp1_np, f"Histogram of {matrix_type_1}")
    plot_hist(axs[1, 1], imp2_np, f"Histogram of {matrix_type_2}")
    plot_hist(axs[1, 2], diff_np, "Histogram of Difference")

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save the plot
    if save_path is None:
        save_path = os.path.join(
            args.results_dir, 
            args.run_name, 
            f"{matrix_name}_{matrix_type_1}_vs_{matrix_type_2}_pool{pool_size}_sort{sort_by}_{args.serial_number}.png"
        )

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Logging
    args.logger.info(f"Two-matrix visualization saved to {save_path}")
    args.logger.info(
        f"Shapes before pooling: {importances_1.shape}, "
        f"Stats for {matrix_type_1}: min={imp1_np.min()}, max={imp1_np.max()}, mean={imp1_np.mean()} "
        f"Stats for {matrix_type_2}: min={imp2_np.min()}, max={imp2_np.max()}, mean={imp2_np.mean()}"
        f"Stats for difference: min={diff_np.min()}, max={diff_np.max()}, mean={diff_np.mean()}"
    )

# Function to visualize importances per weight for a single matrix
# Load importances per weight for a single matrix
# Visualize importances per weight for a single matrix via imshow
# Option to sort weight rows by importance for one prompt vs another prompt
# Option to expose the per *neuron* importance and cross layer pattern we previously captured, and sort by neuron importance
# This should put neurons that are important and most activated at the left of the graph, while neurons not important at the bottom. 
# It can also put neurons important for another task to the other side of the graph. And you can see the distance between them. 
# If neurons are shared, then they would be in the middle. 
def visualize_matrix(args, matrix_name: str, importances: torch.Tensor, 
                    comparison_importances: torch.Tensor = None,
                    sort_by: str = 'none',
                    max_pool_ratio: float = -1.,
                    cmap: str = 'seismic',
                    save_path: str = None) -> None:
    """
    Visualize importance values for a single weight matrix with optional comparison.
    
    Args:
        args: Arguments object containing logger
        matrix_name: Name of the matrix being visualized
        importances: Primary importance tensor of shape [rows, cols]
        comparison_importances: Optional second importance tensor for comparison
        sort_by: How to sort neurons ('primary', 'comparison', 'difference')
        cmap: Colormap to use for visualization
        save_path: Where to save the plot (defaults to standard naming scheme)

        Don't take absolute values.
    """
    plt.figure(figsize=(25, 20))
    
    # Convert to numpy and get absolute values for sorting
    imp_array = importances.detach().cpu().numpy()
        
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(imp_array[:, :], cmap=cmap, aspect='auto')
    ax.set_title(f'Weight Importances\n{matrix_name}')
    plt.colorbar(im)
    
    plt.tight_layout()
    
    # Save the plot
    if save_path:
        save_path = save_path
    else:
        save_path = os.path.join(args.results_dir, args.run_name, f"{matrix_name}_{args.serial_number}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    args.logger.info(f"Matrix visualization saved to {save_path}")

def visualize_matrix_interactive(args, plot_name, matrix):
    """
    Creates and saves an interactive, zoomable heatmap of a given gradient matrix.
    The colormap is centered at zero, so near-zero values look neutral, and larger
    magnitudes become more intensely colored. The resulting HTML file can be opened
    in a browser to view and interact with the plot.

    Args:
        args: An object containing configuration and logging attributes.
              Must have attributes: results_dir, run_name, serial_number, and logger.
        matrix: A torch.Tensor or NumPy array representing the gradient matrix.
        plot_name: Optional string to prepend to the output file name.

    Returns:
        None
    """
    # Ensure matrix is a float32 torch.Tensor
    if not isinstance(matrix, torch.Tensor):
        matrix = torch.tensor(matrix, dtype=torch.float32)
    matrix = matrix.to(torch.float32)

    abs_vals = torch.abs(matrix)
    max_value = torch.max(abs_vals).item()
    if max_value == 0:
        max_value = 1e-8  # Prevent division by zero if all values are zero

    # Convert to a NumPy array for Plotly
    matrix_np = matrix.cpu().numpy()

    # Create a Plotly figure with a diverging colormap centered at zero
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix_np,
            colorscale='RdBu',
            zmin=-max_value,
            zmax=max_value,
            colorbar=dict(title='Gradient Value')
        )
    )

    fig.update_layout(
        title='Gradient Magnitude Heatmap',
        xaxis_title='Index (Dimension 2)',
        yaxis_title='Index (Dimension 1)',
        autosize=True
    )

    # Save the figure as an HTML file
    html_path = os.path.join(
        args.results_dir,
        args.run_name,
        f"{f'{plot_name}_' if plot_name else ''}gradient_matrix_{args.serial_number}.html"
    )
    image_path = html_path[:-len(".html")]+".png"
    fig.write_html(html_path)
    fig.write_image(image_path)

    args.logger.info(f"Gradient matrix heatmap HTML saved to {html_path}, image to {image_path}")

# Function to visualize importances per weight for all matrices
# Load importances per weight for all matrices
# Visualize importances per weight for all matrices via imshow by calling the single matrix function
# Calculate summary statistics such as gradient distribution (history over all parameters)
# Show this summary statistic over all layers

# Function to visualize what knapsack picked / did not pick
def visualize_weights_heatmap_interactive(args, all_keys, picked_keys, total_value, total_cost, plot_name=None, output_file_html=None):
    # Parse data into a list of dictionaries
    data = []
    for key in all_keys:
        parts = key.split('.')
        layer = 'unknown'
        weight_type = 'unknown'
        if 'layers' in parts:
            idx = parts.index('layers')
            layer = parts[idx + 1] if idx + 1 < len(parts) else 'unknown'
            weight_type = '.'.join(parts[idx + 2:]) if idx + 2 < len(parts) else 'unknown'
        status = 1 if key in picked_keys else 0  # Use numeric values for heatmap coloring
        data.append({'Layer': layer, 'Weight Type': weight_type, 'Status': status})
    
    # Create a DataFrame
    df = pd.DataFrame(data)

    # Pivot the DataFrame to create a matrix for the heatmap
    heatmap_data = df.pivot(index='Weight Type', columns='Layer', values='Status')

    # Create the heatmap using Plotly
    fig = px.imshow(
        heatmap_data,
        color_continuous_scale=['red', 'green'],  # Red for discarded, green for kept
        labels=dict(color="Status"),
        title=f"Kept (Green) and Discarded (Red) Weights<br>Total Value: {total_value}, Total Cost: {total_cost}"
    )
    fig.update_layout(
        xaxis_title="Layer",
        yaxis_title="Weight Type",
        height=800,  # Adjust the height for better readability
        width=1200   # Adjust the width for better readability
    )
    if output_file_html == None:
        output_file_html = os.path.join(args.results_dir, args.run_name, f"{f"{plot_name}_"if plot_name else ""}module_selection_vis_{args.serial_number}.png")
    fig.write_image(output_file_html)
    print(f"Interactive plot saved as {output_file_html}")

# Function to printed sorted components in terms of importance
def importance_plot(args, accumulated_gradients, sort=True, plot_name=None, aggregation_function=torch.mean, plot_path=None):
    ## Calculate one value for each matrix for each parameter's gradients from accumulated_gradients
    gradient_importance = {
        name: aggregation_function(torch.abs(grad)).item() 
        for name, grad in accumulated_gradients.items()
    }
    ## Compute statistics from accumulated_gradients
    statistics = {}
    for name, matrix in accumulated_gradients.items():
        flattened = matrix.flatten().to(torch.float32)
        flattened = torch.abs(flattened)
        # Compute statistics
        mean = torch.mean(flattened)
        max_value = torch.max(flattened)
        min_value = torch.min(flattened)
        std = torch.std(flattened)
        # quantiles = torch.quantile(flattened, torch.tensor([0.25, 0.5, 0.75]))
        statistics[name] = (mean, max_value, min_value, std)
    # Sanity Check
    for name, grad in accumulated_gradients.items():
        if grad == None:
            args.logger.info(f"{name}GRAD IS NONE<")
    ## Sort by average magnitude in descending order
    if sort:
        sorted_gradients = dict(sorted(
            gradient_importance.items(),
            key=lambda x: x[1],
            reverse=True
        ))
    else:
        sorted_gradients = gradient_importance
    ## Prepare data for plotting
    names = list(sorted_gradients.keys())
    values = list(sorted_gradients.values())
    lower_errors = [statistics[name][0].item() - statistics[name][2].item() for name in names]
    upper_errors = [statistics[name][1].item() - statistics[name][0].item() for name in names]
    values_array = np.array(values)
    normalized_values = (values_array - values_array.min()) / (values_array.max() - values_array.min())
    colors = plt.cm.viridis(normalized_values)
    ## Create visualization
    plt.figure(figsize=(20, 25))
    names = list(sorted_gradients.keys())
    values = list(sorted_gradients.values())
    # Create bar plot
    names = list(sorted_gradients.keys())
    values = list(sorted_gradients.values())
    lower_errors = [statistics[name][0].item() - statistics[name][2].item() for name in names]
    upper_errors = [statistics[name][1].item() - statistics[name][0].item() for name in names]
    xerr = np.array([lower_errors, upper_errors])
    bars = plt.barh(range(len(names)), values, color=colors)
    # Customize plot labels
    plt.ylabel('Parameter Names')
    plt.xlabel('Average Gradient Magnitude')
    plt.title('Parameter Importance by Average Gradient Magnitude')
    plt.yticks(range(len(names)), names, fontsize=8)
    # Add value labels on the bars
    for i, v in enumerate(values):
        min_value = statistics[names[i]][2].item()
        max_value = statistics[names[i]][1].item()
        std = statistics[names[i]][3].item()
        plt.text(v, i, f'{v:.3e} | (min:{min_value:.2e}, max:{max_value:.2e}, std:{std:.2e})', va='center', fontsize=8)
        # plt.text(v, i, f'{v:.3e}', va='center', fontsize=8)
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    # Save plot
    if not plot_path:
        plot_path = os.path.join(args.results_dir, args.run_name, f"{f'{plot_name}_' if plot_name else ''}by_module_{args.serial_number}.png")
    plt.savefig(plot_path)
    plt.close()
    args.logger.info(f"Gradient importance plot saved to {plot_path}")
    return sorted_gradients

# Function to filter for only mlp and atten layers
def filter_importances_dict(importances, configuration="mlp_atten_only"):
    if "mlp_atten_only":
        importances = {k:v for k, v in importances.items() if ".mlp." in k or ".self_attn." in k}
    elif "linear_only":
        raise Exception("Not implemented")
    return importances


