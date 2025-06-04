import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from io import StringIO



def save_plot(model_name, plot_name):
    """
    Saves the current matplotlib figure in:
        models/{model_name}/plots/{plot_name}.png
    """
    save_dir = os.path.join('models', model_name, 'plots')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{plot_name}.png"), dpi=300, bbox_inches='tight')



def train_plot(model_name, sampled_epochs, losses_train, losses_dev):
    # Define fixed colors
    COLOR_TRAIN = '#1f77b4'  # Matplotlib default blue
    COLOR_DEV = '#ff7f0e'    # Matplotlib default orange

    plt.figure(num=3, figsize=(8, 5))
    plt.title('Train and Dev Losses')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')

    # Set fixed axis limits
    # plt.xlim(0, 200)   # X-axis from 0 to 200
    # plt.ylim(0, 9)     # Y-axis from 0 to 9

    # Plot full training curve
    all_epochs = list(range(1, len(losses_train) + 1))
    plt.plot(all_epochs, losses_train, label='Train Loss', color=COLOR_TRAIN, alpha=0.7)

    # Plot dev loss only at sampled epochs
    plt.plot(sampled_epochs, losses_dev, label='Dev Loss', color=COLOR_DEV, marker='o')

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    save_plot(model_name, "training")
    # plt.show()
    plt.close()



def slot_plot(model_name, uploaded_json):
    # Extract slot f1 scores, excluding 'total'
    slot_f1 = {k: v['f'] for k, v in uploaded_json['results_dev'].items() if k != 'total'}

    # Sort alphabetically by slot tag name
    sorted_slot_f1 = dict(sorted(slot_f1.items()))

    # Plotting
    plt.figure(figsize=(14, 10))
    bars = plt.barh(list(sorted_slot_f1.keys()), list(sorted_slot_f1.values()), color='skyblue')
    plt.xlabel("F1 Score")
    plt.title("Slot Tag F1 Scores (Alphabetical Order)")
    plt.xlim(0, 1.05)
    mean_f1 = sum(sorted_slot_f1.values()) / len(sorted_slot_f1)
    plt.axvline(x=mean_f1, color='r', linestyle='--', label=f"Mean F1: {mean_f1:.2f}")
    plt.legend()
    plt.grid(axis='x')
    plt.tight_layout()

    # Optional: Add value labels
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + 0.2, f'{width:.2f}', va='center')

    # Save the plot
    save_plot(model_name, "slot_f1")
    # plt.show()
    plt.close()


def intent_plot(model_name, uploaded_json, metric='f1-score'):
    # Extract intent metrics
    intent_metrics = {}
    intent_res = uploaded_json['intent_res']
    for intent, results in intent_res.items():
        if intent not in ['accuracy', 'ci_95_beta', 'macro avg', 'weighted avg']:
            if metric == 'precision':
                val = results['precision']
            elif metric == 'recall':
                val = results['recall']
            elif metric == 'f1-score':
                val = results['f1-score']
            else:
                raise ValueError("metric must be one of 'precision', 'recall', or 'f1-score'")
            
            intent_metrics[intent] = val

    # Sort alphabetically by intent name
    sorted_metrics = dict(sorted(intent_metrics.items()))

    # Plotting
    plt.figure(figsize=(14, 10))
    bars = plt.barh(list(sorted_metrics.keys()), list(sorted_metrics.values()), color='skyblue')
    plt.xlabel(metric.capitalize())
    plt.title(f"Intent Classification - {metric.capitalize()}")
    plt.xlim(0, 1.05)
    
    # Add macro and weighted average lines
    macro_avg = intent_res['macro avg'].get(metric)
    weighted_avg = intent_res['weighted avg'].get(metric)

    if macro_avg is not None:
        plt.axvline(x=macro_avg, color='g', linestyle='--', label="Macro Avg")
    if weighted_avg is not None:
        plt.axvline(x=weighted_avg, color='orange', linestyle='--', label="Weighted Avg")
    
    plt.grid(axis='x')
    plt.legend()
    plt.tight_layout()

    # Optional: Add value labels
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + 0.2, f'{width:.2f}', va='center')

    # Save the plot
    save_plot(model_name, f"intent-{metric}")
    plt.close()




def plot_all(model_name):
    """
    This function is used to plot all the plots for the model.
    """
    # Load the training_data JSON file
    with open(os.path.join('models', model_name, 'training_data.json'), 'r') as f:
        training_data = json.load(f)

    # Plot the training and dev losses
    train_plot(model_name, training_data['sampled_epochs'], training_data['losses_train'], training_data['losses_dev'])


    # Load the dev_data JSON file
    with open(os.path.join('models', model_name, 'dev_data.json'), 'r') as f:
        dev_data = json.load(f)

    slot_plot(model_name, dev_data)         # Plot the slot f1 scores
    intent_plot(model_name, dev_data)       # Plot the intent accuracies

    print(f"Plots saved in models/{model_name}/plots/")




def plot_slot_and_intent_errorbars_with_new_data(
    save_path_slot="plots/SlotF1ErrorBars.png",
    save_path_intent="plots/IntentAccErrorBars.png"
):
    # Dati hardcoded in formato CSV-like
    data = """
    label,learning_rate,n_layers,hidden_size,embedding_size,batch_size,dropout,slot_f1,95% CI,intent_acc,95% CI (beta)
SimpleIAS,0.001,1,200,300,128,,0.9253,0.9156 - 0.935,0.9272,0.9083 - 0.9424
Bidirectional,0.001,1,600,900,128,,0.948,0.9398 - 0.9561,0.944,0.9269 - 0.9572
Dropout,0.001,2,600,900,32,0.5,0.9526,0.9448 - 0.9604,0.9619,0.9473 - 0.9726
    """

    # Carico i dati in un DataFrame
    df = pd.read_csv(StringIO(data.strip()))

    # Converto le metriche a float
    df['slot_f1'] = df['slot_f1'].astype(float)
    df['intent_acc'] = df['intent_acc'].astype(float)

    # Funzione per parsare gli intervalli di confidenza
    def parse_ci(ci_str):
        lower, upper = map(float, ci_str.replace(" ", "").split('-'))
        mean = (lower + upper) / 2
        error = upper - mean
        return mean, error

    # Estrai Slot F1 e CI
    slot_means = df['slot_f1'].tolist()
    slot_errors = [parse_ci(ci)[1] for ci in df['95% CI']]
    
    # Estrai Intent Accuracy e CI
    intent_means = df['intent_acc'].tolist()
    intent_errors = [parse_ci(ci)[1] for ci in df['95% CI (beta)']]

    labels = df['label'].tolist()
    x = np.arange(len(df))
    bar_width = 0.6

    # --- Plot Slot F1 ---
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.bar(x, slot_means, bar_width, yerr=slot_errors, capsize=5, color='lightgreen', align='center')
    ax.set_ylabel('Slot F1 Score', fontsize=14)
    ax.set_title('Slot F1 Scores with 95% Confidence Interval', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_ylim(0.91, 0.965)

    custom_legend = [
        Line2D([0], [0], color='lightgreen', lw=10, label='Model'),
        Line2D([0], [0], color='black', lw=2, label='95% CI'),
    ]

    ax.legend(handles=custom_legend, fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path_slot)
    plt.close()

    # --- Plot Intent Accuracy ---
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.bar(x, intent_means, bar_width, yerr=intent_errors, capsize=5, color='purple')
    ax.set_ylabel('Intent Accuracy', fontsize=14)
    ax.set_title('Intent Accuracy with 95% Confidence Interval', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_ylim(0.9, 0.975)

    custom_legend = [
        Line2D([0], [0], color='purple', lw=10, label='Model'),
        Line2D([0], [0], color='black', lw=2, label='95% CI'),
    ]
    ax.legend(handles=custom_legend, fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path_intent)
    plt.close()



plot_slot_and_intent_errorbars_with_new_data()