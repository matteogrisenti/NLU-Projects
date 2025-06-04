import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
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
    plt.xlim(0, 40)   # X-axis from 0 to 200
    plt.ylim(0, 5)     # Y-axis from 0 to 9

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

    print(f"Plots saved in models/{model_name}/plots/")




def plot_heatmap_with_annotations():
    ''' 
    This function is used to plot the dev performance of Base Bert model in a heat map
    '''

    # Raw data
    data = """
    base,0.0001,128,0.1,0.9844,0.9784 - 0.9903,0.996,0.9856 - 0.9988
    base,0.0001,32,0.1,0.9841,0.9781 - 0.99,0.994,0.9825 - 0.9978
    base,0.0001,64,0.1,0.9826,0.9763 - 0.9888,0.9859,0.9713 - 0.9931
    base,0.0001,32,0.05,0.98,0.9733 - 0.9866,0.9859,0.9713 - 0.9931
    base,0.0001,64,0.05,0.9838,0.9777 - 0.9898,0.998,0.9889 - 0.9995
    base,0.0001,128,0.05,0.9826,0.9763 - 0.9888,0.988,0.974 - 0.9943
    base,0.0001,32,0.3,0.9835,0.9774 - 0.9895,0.992,0.9796 - 0.9967
    base,0.0001,64,0.3,0.9811,0.9746 - 0.9876,0.992,0.9796 - 0.9967
    base,0.0001,128,0.3,0.9826,0.9764 - 0.9888,0.992,0.9796 - 0.9967
    base,0.0001,32,0.5,0.9814,0.9749 - 0.9878,0.994,0.9825 - 0.9978
    base,0.0001,64,0.5,0.9817,0.9753 - 0.9881,0.99,0.9768 - 0.9956
    base,0.0001,128,0.5,0.984,0.9781 - 0.99,0.99,0.9768 - 0.9956
    """

    # Parse the data
    records = []
    for line in data.strip().split('\n'):
        parts = line.split(',')
        batch_size = int(parts[2])
        dropout = float(parts[3])
        slot_f1 = float(parts[4])
        intent_acc = float(parts[6])
        total = slot_f1 + intent_acc
        records.append({
            'batch_size': batch_size,
            'dropout': dropout,
            'slot_f1': slot_f1,
            'intent_acc': intent_acc,
            'total': total,
            'annotation': f"{slot_f1:.3f}\n{intent_acc:.3f}"
        })

    # Convert to DataFrame
    df = pd.DataFrame(records)
    df_pivot = df.pivot(index='batch_size', columns='dropout', values='total')
    annotations = df.pivot(index='batch_size', columns='dropout', values='annotation')

    # Plot heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        df_pivot,
        annot=annotations,
        fmt='',
        cmap='RdYlGn',  
        linewidths=0.5,
        cbar_kws={'label': 'Slot F1 + Intent Accuracy'}
    )
    plt.title('Heatmap of Slot F1 + Intent Accuracy (LR=0.0001)\nEach cell shows Slot F1 / Intent Accuracy')
    plt.xlabel('Dropout Rate')
    plt.ylabel('Batch Size')
    plt.tight_layout()
    # plt.show()
    plt.savefig('heatmap.png', dpi=300)
    plt.close()




def plot_slot_intent_with_ci():
    # Raw data
    data = [
        {"label": "SimpleIAS", "slot_f1": 0.9253, "slot_ci": "0.9156 - 0.935", "intent_acc": 0.9272, "intent_ci": "0.9083 - 0.9424"},
        {"label": "Bidirectional", "slot_f1": 0.948, "slot_ci": "0.9398 - 0.9561", "intent_acc": 0.944, "intent_ci": "0.9269 - 0.9572"},
        {"label": "Dropout", "slot_f1": 0.9526, "slot_ci": "0.9448 - 0.9604", "intent_acc": 0.9619, "intent_ci": "0.9473 - 0.9726"},
        {"label": "Bert Base", "slot_f1": 0.9555, "slot_ci": "0.9479 - 0.9631", "intent_acc": 0.9765, "intent_ci": "0.9643 - 0.9845"},
        {"label": "Bert Large", "slot_f1": 0.9581, "slot_ci": "0.9507 - 0.9655", "intent_acc": 0.9765, "intent_ci": "0.9643 - 0.9845"},
    ]

    df = pd.DataFrame(data)

    label_colors = {
        "SimpleIAS": (222, 226, 230),        
        "Bidirectional": (246, 189, 96),  
        "Dropout": (166, 225, 250),          
        "Bert Base": (251, 184, 221),         
        "Bert Large": (220, 238, 209),           
    }

    # Parse CI
    df[['slot_ci_lower', 'slot_ci_upper']] = df['slot_ci'].str.split(' - ', expand=True).astype(float)
    df[['intent_ci_lower', 'intent_ci_upper']] = df['intent_ci'].str.split(' - ', expand=True).astype(float)

    df['slot_err_lower'] = df['slot_f1'] - df['slot_ci_lower']
    df['slot_err_upper'] = df['slot_ci_upper'] - df['slot_f1']
    df['intent_err_lower'] = df['intent_acc'] - df['intent_ci_lower']
    df['intent_err_upper'] = df['intent_ci_upper'] - df['intent_acc']

    # Convert RGB 0-255 to 0-1
    def rgb_to_mpl(rgb):
        return tuple([v / 255 for v in rgb])

    # Plot Slot F1
    plt.figure(figsize=(10, 5))
    x_positions = range(len(df))  # Numeric x for plotting

    for i, row in df.iterrows():
        label = row['label']
        color = rgb_to_mpl(label_colors.get(label, (0, 0, 0)))  # Default black
        x = x_positions[i]
        y = row.slot_f1

        plt.errorbar(label, row['slot_f1'],
                     yerr=[[row['slot_err_lower']], [row['slot_err_upper']]],
                     fmt='o',
                     capsize=5,
                     linewidth=2,
                     markersize=16,
                     color=color,
                     ecolor='black',
                     markeredgecolor='black',
                     markeredgewidth=1.5)
    
    plt.title('Slot F1 Score with 95% Confidence Intervals')
    plt.ylabel('Slot F1 Score')
    plt.ylim(0.9, 1.0)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/slot_plot.png', dpi=300)
    plt.close()

    # Plot Intent Accuracy
    plt.figure(figsize=(10, 5))
    for i, row in df.iterrows():
        label = row['label']
        color = rgb_to_mpl(label_colors.get(label, (0, 0, 0)))
        x = x_positions[i]
        y = row.intent_acc

        plt.errorbar(label, row['intent_acc'],
                     yerr=[[row['intent_err_lower']], [row['intent_err_upper']]],
                     fmt='o',
                     capsize=5,
                     linewidth=2,
                     markersize=16,
                     color=color,
                     ecolor='black',
                     markeredgecolor='black',
                     markeredgewidth=1.5)

    plt.title('Intent Accuracy with 95% Confidence Intervals')
    plt.ylabel('Intent Accuracy')
    plt.ylim(0.9, 1.0)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/intent_plot.png', dpi=300)
    plt.close()


# Call the function
# plot_heatmap_with_annotations()
plot_slot_intent_with_ci()
