import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import seaborn as sns

# ------------------------------------------------------------------------------
# Function: plot_ppl_with_sem_ci
#
# Description:
#     Generates a bar plot visualizing the PPL (Perplexity) scores from selected
#     experiments along with their Standard Error of the Mean (SEM) and 95% 
#     Confidence Interval (CI). Each bar represents the mean of the CI range, 
#     with the actual CI indicated by vertical error bars and SEM shown as a 
#     small red overlay bar centered on the mean.
#
# Parameters:
#     csv_path (str): Path to the CSV file containing experiment results.
#     row_ids (list[int]): List of row IDs (experiment IDs) to include in the plot.
#     save_path (str): Path where the generated plot image will be saved.
#
# Behavior:
#     - Loads data from the provided CSV file.
#     - Extracts PPL values, SEM, and CI bounds for the selected experiments.
#     - Constructs a grouped bar chart with CI error bars and SEM visual cues.
#     - Saves the resulting figure to the specified file path.
#
# Output:
#     An image file is saved at `save_path` showing the plotted bars with CI and SEM.
# ------------------------------------------------------------------------------
def plot_ppl_with_sem_ci(csv_path, row_ids, save_path="ErrorBars.png"):
    df = pd.read_csv(csv_path)
    df_filtered = df[df['ID'].isin(row_ids)].copy()

    # Extract required values
    ids = df_filtered['ID'].tolist()
    sem_values = df_filtered['SEM PPL'].astype(float).tolist()

    # Parse confidence interval (CI) strings and convert to float
    ci_split = df_filtered['CI PPL Test'].str.split('-', expand=True)
    ci_split.columns = ['ci_lower', 'ci_upper']
    ci_split = ci_split.astype(float)

    # Compute CI means and error margins
    ci_means = ((ci_split['ci_lower'] + ci_split['ci_upper']) / 2).tolist()
    ci_errors = ((ci_split['ci_upper'] - ci_split['ci_lower']) / 2).tolist()

    # Set up plot positions
    x = np.arange(len(ids))
    bar_width = 0.75        # Main bar width
    sem_bar_width = 0.1     # Narrower SEM bar width

    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot PPL bars with confidence intervals
    PPL_COLOR = 'skyblue'
    ax.bar(x, ci_means, bar_width, yerr=ci_errors, capsize=5, color=PPL_COLOR)

    # Overlay SEM bars on top of PPL mean bars
    for i, (x_pos, ci_mean, sem) in enumerate(zip(x, ci_means, sem_values)):
        ax.bar(x_pos, 2 * sem, sem_bar_width, bottom=ci_mean - sem, color='red')

    # Configure axes and labels
    ax.set_xlabel('Learning Rate', fontsize=16)
    ax.set_ylabel('PPL', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels([f'lr={lr}' for lr in df_filtered['Learning Rate'].astype(str).tolist()], fontsize=14)
    ax.tick_params(axis='y', labelsize=14)

    # Custom legend
    custom_legend = [
        Line2D([0], [0], color=PPL_COLOR, lw=10, label='PPL'),
        Line2D([0], [0], color='black', lw=2, label='CI'),
        Line2D([0], [0], color='red', lw=6, label='SEM')
    ]
    ax.legend(handles=custom_legend, fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path)



# ------------------------------------------------------------------------------
# Function: plot_ppl_heatmap
#
# Description:
#     Generates a heatmap visualization of PPL (Perplexity) scores extracted
#     from a CSV file for a subset of experiment configurations. The x-axis 
#     represents the Batch Size, and the y-axis represents the Learning Rate.
#     The color intensity in each cell reflects the PPL value.
#
# Parameters:
#     csv_path (str): Path to the CSV file containing experiment results.
#     selected_ids (list[int]): List of row IDs (experiment IDs) to include in the heatmap.
#     save_path (str): Path where the generated plot image will be saved.
#
# Behavior:
#     - Reads experiment results from the given CSV file.
#     - Filters the data to include only rows matching the selected experiment IDs.
#     - Constructs a pivot table with Learning Rate as rows and Batch Size as columns.
#     - Uses seaborn to create a heatmap of the PPL scores.
#
# Output:
#     Displays the heatmap plot showing PPL across combinations of Learning Rate and Batch Size.
# ------------------------------------------------------------------------------
def plot_ppl_heatmap(csv_path, selected_ids, save_path="Heatmap.png"):
    # Read CSV file
    df = pd.read_csv(csv_path)

    # Filter rows based on the selected experiment IDs
    df_filtered = df[df['ID'].isin(selected_ids)].copy()

    # Ensure correct data types (no more warning)
    df_filtered.loc[:, 'Learning Rate'] = df_filtered['Learning Rate'].astype(float)
    df_filtered.loc[:, 'Batch Size'] = df_filtered['Batch Size'].astype(int)
    df_filtered.loc[:, 'PPL Test'] = df_filtered['PPL Test'].astype(float)

    # Pivot the data for heatmap plotting
    pivot_table = df_filtered.pivot_table(
        index='Learning Rate',
        columns='Batch Size',
        values='PPL Test'
    )

    # Sort axes for clearer visualization
    pivot_table = pivot_table.sort_index(ascending=False)  # Learning rate: high to low (top to bottom)
    pivot_table = pivot_table[sorted(pivot_table.columns)]  # Batch size: low to high (left to right)

    # Generate the heatmap
    plt.figure(figsize=(10, 6))
    # Heatmap with green-to-red colors and larger axis labels
    ax = sns.heatmap(
        pivot_table,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn_r",             # green to red
        linewidths=.5,
        xticklabels=True,
        yticklabels=True,
        annot_kws={
            "fontsize": 14,         # Font size
            # "color": "white",       # Text color (puoi cambiare il colore)
            "fontweight": "bold"    # Bold text
        },
    )

    # use matplotlib.colorbar.Colorbar object
    cbar = ax.collections[0].colorbar
    # here set the labelsize by 20
    cbar.ax.tick_params(labelsize=14)

    # Set larger font size for axis labels
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("Batch Size", fontsize=16)
    plt.ylabel("Learning Rate", fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path)




# -----------------------------------------  MAIN --------------------------------------------------
filename = 'experiments.csv'
plot_ppl_heatmap(filename, [4,5,6,7,8,9,10,11,12,13,14,15], 'HeatmapLSTM.png')

