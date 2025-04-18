import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

def plot_ppl_with_sem_ci(csv_path, row_ids, save_path="ErrorBars.png"):
    df = pd.read_csv(csv_path)
    df_filtered = df[df['ID'].isin(row_ids)].copy()

    ids = df_filtered['ID'].tolist()
    sem_values = df_filtered['SEM PPL'].astype(float).tolist()

    ci_split = df_filtered['CI PPL Test'].str.split('-', expand=True)
    ci_split.columns = ['ci_lower', 'ci_upper']
    ci_split = ci_split.astype(float)

    ci_means = ((ci_split['ci_lower'] + ci_split['ci_upper']) / 2).tolist()
    ci_errors = ((ci_split['ci_upper'] - ci_split['ci_lower']) / 2).tolist()

    x = np.arange(len(ids))
    bar_width = 0.4
    sem_bar_width = 0.05

    fig, ax = plt.subplots(figsize=(10, 6))
    # RNN = skyblue
    # LSTM = orange
    ax.bar(x, ci_means, bar_width, yerr=ci_errors, capsize=5, color='skyblue')

    for i, (x_pos, ci_mean, sem) in enumerate(zip(x, ci_means, sem_values)):
        ax.bar(x_pos, 2 * sem, sem_bar_width, bottom=ci_mean - sem, color='red')

    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('PPL')
    ax.set_title('PPL: CI & SEM')
    ax.set_xticks(x)
    ax.set_xticklabels([f'lr={lr}' for lr in df_filtered['Learning Rate'].astype(str).tolist()])

    # Aggiunta legenda manuale
    custom_legend = [
        Line2D([0], [0], color='orange', lw=10, label='PPL'),
        Line2D([0], [0], color='black', lw=2, label='CI'),
        Line2D([0], [0], color='red', lw=6, label='SEM')
    ]
    ax.legend(handles=custom_legend)

    plt.tight_layout()
    plt.savefig(save_path)

# Esempio d'uso:
filename = 'experiments.csv'
plot_ppl_with_sem_ci(filename, [2,1,3])

