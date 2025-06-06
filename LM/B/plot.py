import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_ppl_with_ci_custom_style():
    # Dati raw
    data = [
        {"label": "RNN", "ppl": 157.00, "ci": "140.01 - 177.29"},
        {"label": "LSTM", "ppl": 139.17, "ci": "123.29 - 156.41"},
        {"label": "LSTM-DO", "ppl": 121.88, "ci": "107.76 - 137.12"},
        {"label": "ADAM", "ppl": 105.69, "ci": "92.6 - 121.71"},
        {"label": "WeightTying", "ppl": 116.06, "ci": "103.18 - 130.04"},
        {"label": "VarDropout", "ppl": 86.89, "ci": "76.99 - 97.74"},
        {"label": "NTAvSGD", "ppl": 84.64, "ci": "74.95 - 95.13"},
    ]

    df = pd.DataFrame(data)

    # Colori RGB associati ai label
    label_colors = {
        "RNN": (222, 226, 230),       
        "LSTM": (198, 222, 241),      
        "LSTM-DO": (255, 231, 170),   
        "ADAM": (205, 151, 255),      
        "WeightTying": (236,154,154),
        "VarDropout": (220, 238, 209),
        "NTAvSGD": (221,181,143)
    }

    # Parse CI
    df[['ci_lower', 'ci_upper']] = df['ci'].str.split(' - ', expand=True).astype(float)
    df['err_lower'] = df['ppl'] - df['ci_lower']
    df['err_upper'] = df['ci_upper'] - df['ppl']

    def rgb_to_mpl(rgb):
        return tuple(v / 255 for v in rgb)

    # Plot PPL
    plt.figure(figsize=(10, 5))
    x_positions = range(len(df))

    for i, row in df.iterrows():
        label = row['label']
        color = rgb_to_mpl(label_colors.get(label, (0, 0, 0)))  # default nero
        x = x_positions[i]

        plt.errorbar(label, row['ppl'],
                     yerr=[[row['err_lower']], [row['err_upper']]],
                     fmt='o',
                     capsize=5,
                     linewidth=2,
                     markersize=16,
                     color=color,
                     ecolor='black',
                     markeredgecolor='black',
                     markeredgewidth=1.5)

    plt.title('Test Set Perplexity (PPL) with 95% Confidence Intervals')
    plt.ylabel('Perplexity (PPL)')
    plt.ylim(50, 185)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/ppl_plot.png', dpi=300)

plot_ppl_with_ci_custom_style()
