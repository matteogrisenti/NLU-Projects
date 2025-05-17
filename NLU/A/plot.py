import os
import json
import numpy as np
import matplotlib.pyplot as plt


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
    plt.xlim(0, 200)   # X-axis from 0 to 200
    plt.ylim(0, 9)     # Y-axis from 0 to 9

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


# plot_all('SimpleIAS_lr-0,0001_hid-200_emb-300_batch-128_layers-1')
# plot_all('SimpleIAS_lr-0,001_hid-200_emb-300_batch-128_layers-1')