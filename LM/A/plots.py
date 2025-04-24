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

def plot_ppl_with_sem_ci_AdamW(csv_path, row_ids, save_path="ErrorBars.png"):
    df = pd.read_csv(csv_path)
    df_filtered = df[df['ID'].isin(row_ids)].copy()

    # Estrazione dei valori
    ids = df_filtered['ID'].tolist()
    sem_values = df_filtered['SEM PPL'].astype(float).tolist()
    dim_pairs = list(zip(df_filtered['Hidden Size'], df_filtered['Embedding Size']))

    # Parsing intervallo di confidenza (CI)
    ci_split = df_filtered['CI PPL Test'].str.split('-', expand=True)
    ci_split.columns = ['ci_lower', 'ci_upper']
    ci_split = ci_split.astype(float)

    # Calcolo media CI e margini
    ci_means = ((ci_split['ci_lower'] + ci_split['ci_upper']) / 2).tolist()
    ci_errors = ((ci_split['ci_upper'] - ci_split['ci_lower']) / 2).tolist()

    # Posizioni nel plot
    x = np.arange(len(ids))
    bar_width = 0.75
    sem_bar_width = 0.1

    fig, ax = plt.subplots(figsize=(8, 6))

    # Mappa dimensioni -> colori
    def get_color(dim_pair):
        return 'green' if dim_pair == (400, 600) else 'orange'

    # Barre con CI
    for i, (x_pos, ci_mean, ci_err, dim_pair) in enumerate(zip(x, ci_means, ci_errors, dim_pairs)):
        color = get_color(dim_pair)
        ax.bar(x_pos, ci_mean, bar_width, yerr=ci_err, capsize=5, color=color)

    # Barre SEM
    for i, (x_pos, ci_mean, sem) in enumerate(zip(x, ci_means, sem_values)):
        ax.bar(x_pos, 2 * sem, sem_bar_width, bottom=ci_mean - sem, color='red')

    # Asse e etichette
    ax.set_xlabel('Dropout', fontsize=16)
    ax.set_ylabel('PPL', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels([f'dr={d}' for d in df_filtered['Droput Emb']], fontsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_ylim(50,120)

    # Legenda personalizzata
    custom_legend = [
        Line2D([0], [0], color='green', lw=10, label='400-600'),
        Line2D([0], [0], color='orange', lw=10, label='600-900'),
        Line2D([0], [0], color='black', lw=2, label='CI'),
        Line2D([0], [0], color='red', lw=6, label='SEM')
    ]
    ax.legend(handles=custom_legend, fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path)



# ------------------------------------------------------------------------------
# Function: plot_ppl_with_sem_ci_with_reference
#
# Description:
#     Generates a bar plot visualizing the PPL (Perplexity) scores from selected
#     experiments along with their Standard Error of the Mean (SEM) and 95% 
#     Confidence Interval (CI). A separate reference experiment is also displayed
#     as the first column, using distinct colors to highlight its role as a baseline.
#
# Parameters:
#     csv_path (str): Path to the CSV file containing experiment results.
#     row_ids (list[int]): List of row IDs (experiment IDs) to include in the plot.
#     reference_id (int): ID of the reference experiment to be shown separately.
#     save_path (str): Path where the generated plot image will be saved.
#
# Output:
#     An image file is saved at `save_path` showing the plotted bars with CI, SEM,
#     and the reference experiment highlighted.
# ------------------------------------------------------------------------------
def plot_ppl_with_sem_ci_with_reference(csv_path, row_ids, reference_id, save_path="ErrorBarsWithReference.png"):
    df = pd.read_csv(csv_path)

    # Include the reference ID separately
    all_ids = [reference_id] + row_ids
    df_filtered = df[df['ID'].isin(all_ids)].copy()

    # Sort to ensure reference is first
    df_filtered['SortOrder'] = df_filtered['ID'].apply(lambda x: 0 if x == reference_id else 1)
    df_filtered = df_filtered.sort_values(by='SortOrder')

    ids = df_filtered['ID'].tolist()
    sem_values = df_filtered['SEM PPL'].astype(float).tolist()

    # Parse CI strings
    ci_split = df_filtered['CI PPL Test'].str.split('-', expand=True)
    ci_split.columns = ['ci_lower', 'ci_upper']
    ci_split = ci_split.astype(float)

    ci_means = ((ci_split['ci_lower'] + ci_split['ci_upper']) / 2).tolist()
    ci_errors = ((ci_split['ci_upper'] - ci_split['ci_lower']) / 2).tolist()

    x = np.arange(len(ids))
    bar_width = 0.75
    sem_bar_width = 0.1

    fig, ax = plt.subplots(figsize=(7, 6))

    # Plot bars with different colors for reference
    for i, (id_, x_pos, ci_mean, ci_err, sem) in enumerate(zip(ids, x, ci_means, ci_errors, sem_values)):
        if id_ == reference_id:
            bar_color = 'steelblue'
            sem_color = 'red'
            label = 'Reference'
        else:
            bar_color = 'skyblue'
            sem_color = 'red'
            label = None

        ax.bar(x_pos, ci_mean, bar_width, yerr=ci_err, capsize=5, color=bar_color)
        ax.bar(x_pos, 2 * sem, sem_bar_width, bottom=ci_mean - sem, color=sem_color)

    # Labels
    ax.set_ylabel('PPL', fontsize=16)
    ax.set_xticks(x)
    xticklabels = ['plain LSTM'] + [f'dropout={lr}' for lr in df_filtered[df_filtered['ID'] != reference_id]['Droput Emb'].astype(str).tolist()]
    ax.set_xticklabels(xticklabels, fontsize=14)
    ax.set_ylim(40, 200)
    ax.tick_params(axis='y', labelsize=14)

    # Custom legend
    custom_legend = [
        Line2D([0], [0], color='skyblue', lw=10, label='PPL'),
        Line2D([0], [0], color='black', lw=2, label='CI'),
        Line2D([0], [0], color='red', lw=6, label='SEM'),
        Line2D([0], [0], color='steelblue', lw=10, label='plain LSTM PPL'),
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



# ------------------------------------------------------------------------------
# Function: plot_ppl_by_batchsize_with_ci
#
# Description:
#     Creates a scatter plot with error bars to visualize PPL (Perplexity) scores 
#     across different combinations of Batch Size and Learning Rate. For each 
#     batch size, three models with distinct learning rates (0.5, 1, 2) are 
#     displayed side-by-side. Each point shows the mean of the 95% Confidence 
#     Interval (CI), with CI shown as vertical error bars and SEM (Standard Error 
#     of the Mean) indicated as smaller red ticks.
#
# Parameters:
#     csv_path (str): Path to the CSV file containing experiment results.
#     selected_ids (list[int]): List of row IDs (experiment IDs) to include in the plot.
#     save_path (str): Path where the generated plot image will be saved.
#
# Output:
#     Saves the resulting scatter plot with CI and SEM indications to the specified file path.
# ------------------------------------------------------------------------------
def plot_ppl_by_batchsize_with_ci(csv_path, selected_ids, save_path="PPL_vs_BatchSize.png"):
    # Carica e filtra dati
    df = pd.read_csv(csv_path)
    df_filtered = df[df['ID'].isin(selected_ids)].copy()

    # Converti i tipi
    df_filtered['Learning Rate'] = df_filtered['Learning Rate'].astype(float)
    df_filtered['Batch Size'] = df_filtered['Batch Size'].astype(int)
    df_filtered['SEM PPL'] = df_filtered['SEM PPL'].astype(float)

    # Parsing CI
    ci_split = df_filtered['CI PPL Test'].str.split('-', expand=True)
    ci_split.columns = ['ci_lower', 'ci_upper']
    ci_split = ci_split.astype(float)
    df_filtered['ci_mean'] = (ci_split['ci_lower'] + ci_split['ci_upper']) / 2
    df_filtered['ci_error'] = (ci_split['ci_upper'] - ci_split['ci_lower']) / 2

    # Batch sizes e learning rates da usare
    batch_sizes = [16, 32, 64, 128]
    learning_rates = [0.5, 1, 2]
    lr_colors = {0.5: 'blue', 1: 'green', 2: 'orange'}

    # Posizioni discrete asse x
    x_base_positions = np.arange(len(batch_sizes))  # es. [0, 1, 2, 3]
    offset_width = 0.2  # spazio tra i punti nel gruppo
    lr_offsets = {0.5: -offset_width, 1: 0, 2: offset_width}

    # Plot setup
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot per ciascun punto
    for lr in learning_rates:
        df_lr = df_filtered[df_filtered['Learning Rate'] == lr]
        for i, bs in enumerate(batch_sizes):
            row = df_lr[df_lr['Batch Size'] == bs]
            if row.empty:
                continue
            row = row.iloc[0]
            x = x_base_positions[i] + lr_offsets[lr]
            y = row['ci_mean']
            ci = row['ci_error']
            sem = row['SEM PPL']

            # Punto + CI
            ax.errorbar(x, y, yerr=ci, fmt='o', color=lr_colors[lr], capsize=8, markersize=10, elinewidth=4)

            # SEM come tick rosso
            ax.errorbar(x, y, yerr=sem, fmt='_', color='red', capsize=8, elinewidth=4)

    # Asse x discreto
    ax.set_xticks(x_base_positions)
    ax.set_xticklabels([str(bs) for bs in batch_sizes], fontsize=14)
    ax.set_xlabel("Batch Size", fontsize=16)
    ax.set_ylabel("PPL", fontsize=16)
    ax.tick_params(axis='y', labelsize=14)

    # Legenda custom
    custom_legend = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=lr_colors[0.5], label='lr=0.5', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=lr_colors[1], label='lr=1', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=lr_colors[2], label='lr=2', markersize=10),
        Line2D([0], [0], color='black', lw=2, label='95% CI'),
        Line2D([0], [0], color='red', lw=2, label='SEM')
    ]
    ax.legend(handles=custom_legend, fontsize=14, ncol=2, loc='upper left')

    plt.tight_layout()
    plt.savefig(save_path)



# -----------------------------------------  MAIN --------------------------------------------------
filename = 'experiments.csv'
plot_ppl_with_sem_ci_AdamW(filename, [26,27,28,29,30,31], save_path="AdamWDIM.png")

