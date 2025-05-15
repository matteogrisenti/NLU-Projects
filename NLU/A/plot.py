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
    plt.show()



def slot_plot(model_name, uploaded_json):
        # Extract slot f1 scores, excluding 'total'
    slot_f1 = {k: v['f'] for k, v in uploaded_json['slot_results'].items() if k != 'total'}

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
    plt.show()



def intent_plot(model_name, uploaded_json):
    # Extract intent accuracies
    intent_acc = {
        k: v['accuracy'] for k, v in uploaded_json['intent_results'].items()
        if k not in ['accuracy', 'macro avg', 'weighted avg']
    }

    # Sort alphabetically by intent name
    sorted_intent_acc = dict(sorted(intent_acc.items()))

    # Plotting
    plt.figure(figsize=(14, 10))
    bars = plt.barh(list(sorted_intent_acc.keys()), list(sorted_intent_acc.values()), color='skyblue')
    plt.xlabel("Accuracy")
    plt.title("Intent Classification Accuracy")
    plt.xlim(0, 1.05)
    plt.axvline(x=uploaded_json['intent_results']['accuracy'], color='r', linestyle='--', label="Overall Accuracy")
    plt.grid(axis='x')
    plt.legend()
    plt.tight_layout()

    # Optional: Add value labels
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + 0.2, f'{width:.2f}', va='center')

    # Save the plot
    save_plot(model_name, "intent_accuracy")
    plt.show()




def plot_all(model_name):
    """
    This function is used to plot all the plots for the model.
    """
    # Load the training_data JSON file
    with open(os.path.join('models', model_name, 'training_data.json'), 'r') as f:
        training_data = json.load(f)

    # Plot the training and dev losses
    train_plot(model_name, training_data['sampled_epochs'], training_data['losses_train'], training_data['losses_dev'])


    # Load the test_data JSON file
    with open(os.path.join('models', model_name, 'test_data.json'), 'r') as f:
        test_data = json.load(f)

    slot_plot(model_name, test_data)         # Plot the slot f1 scores
    intent_plot(model_name, test_data)       # Plot the intent accuracies



plot_all('SimpleIAS_lr-0,0001_hid-200_emb-300_batch-128_layers-1')