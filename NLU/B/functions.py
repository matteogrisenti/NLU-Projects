import os
import sys
import csv
import json
import torch
import numpy as np
import torch.nn as nn
import scipy.stats as st
import torch.optim as optim

from tqdm import tqdm
from copy import deepcopy
from pprint import pformat
from conll import evaluate
from sklearn.metrics import classification_report



def model_name(label, lr, batch_size, dropout):
    name = f"{label}_lr-{str(lr).replace('.', ',')}_batch-{batch_size}"
    if dropout is not None:
        name += f"_drop-{str(dropout).replace('.', ',')}"
    return name




def save_training(sampled_epochs, losses_train, losses_dev, name): 
    """
    Saves training data (epochs, train/dev losses) into a JSON file for future plotting.

    Args:
        sampled_epochs (list): List of epoch numbers where loss was recorded.
        losses_train (list): Training losses per sampled epoch.
        losses_dev (list): Validation/dev losses per sampled epoch.
        name (str): Name of the model/experiment (used in filename).
    """
    
    # Ensure directory exists
    save_dir = os.path.join('models', name)
    os.makedirs(save_dir, exist_ok=True)
    file_path = save_dir + '/training_data.json'
    
    # Create dictionary to save
    training_data = {
        "sampled_epochs": sampled_epochs,
        "losses_train": losses_train,
        "losses_dev": losses_dev,
    }
    
    with open(file_path, 'w') as f:
        json.dump(training_data, f, indent=4)
    
    print(f"\tTraining data saved to: {file_path}")




def save_model(name, model, bert_type, num_intent_labels, num_slot_labels):
    path = 'bin/others/' + name + '.pt'
    saving_object = { 
        "model": model.state_dict(), 
        "bert_type": bert_type, 
        "num_intent_labels": num_intent_labels, 
        "num_slot_labels": num_slot_labels 
    }

    print(f"\tSaving model to {path}:")
    print(f"\t\tBERT type: {bert_type}, num_intent_labels: {num_intent_labels}, num_slot_labels: {num_slot_labels}")
    torch.save(saving_object, path)




def save_dev(bert_type, lr, batch_size, dropout, slot_f1, f1_ci_95, intent_accuracy, ci_95_beta):
    """
    Saves training/validation or test results along with hyperparameters to a CSV file.

    Args:
        bert_type (str): Type of BERT model used: 'bert-base-uncased' or 'bert-large-uncased'.
        lr (float): Learning rate used in training.
        batch_size (int): Batch size used in training.
        dropout (float or None): Dropout rate used in training, None if not applied.
        slot_f1 (float): F1 score for slot filling task.
        f1_ci_95 (tuple): 95% confidence interval for the slot F1 score as a tuple (lower_bound, upper_bound).
        intent_accuracy (float): Accuracy for intent classification task.
        ci_95_beta (tuple): 95% confidence interval for intent accuracy as a tuple (lower_bound, upper_bound).
    """
    
    # Create file if it doesn't exist, append otherwise
    filename = 'results/dev.csv'
    file_exists = os.path.isfile(filename)

    # Format bert type 
    if bert_type == 'bert-base-uncased':
        label = 'base'
    elif bert_type == 'bert-large-uncased':
        label = 'large'
    else:
        raise ValueError(f"Unknown BERT type: {bert_type}")

    # Prepare data to write
    data = {
        'label': label,
        'learning_rate': lr,
        'batch_size': batch_size,
        'dropout': dropout or None,
        'slot_f1': round(slot_f1, 4),             # Slot F1 score rounded to 2 decimal places
        '95% CI': f"{round(f1_ci_95[0], 4)} - {round(f1_ci_95[1], 4)}",  # 95% CI for F1 score
        'intent_acc': round(intent_accuracy, 4),   # Intent accuracy rounded to 2 decimal places
        '95% CI (beta)': f"{round(ci_95_beta[0], 4)} - {round(ci_95_beta[1], 4)}"  # 95% CI for intent accuracy
    }

    # Define fieldnames for CSV header
    fieldnames = [
        'Bert Type', 'Learning Rate', 'batch_size', 'dropout', 'slot_f1', '95% CI', 'intent_acc', '95% CI (beta)'
    ]

    # Write to CSV
    with open(filename, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()  # Write header only once

        writer.writerow(data)  # Write the row with results and hyperparams

    print(f"\tResults saved to {filename}")




def save_dev_results(dev_results, name):
    """
    Saves the development results to a JSON file.

    Args:
        dev_results (dict): Development results containing loss and metrics.
        name (str): Name of the model/experiment (used in filename).
    """
    
    # Ensure directory exists
    save_dir = os.path.join('models', name)
    os.makedirs(save_dir, exist_ok=True)
    file_path = save_dir + '/dev_data.json'
    
    with open(file_path, 'w') as f:
        json.dump(dev_results, f, indent=4)
    
    print(f"\tDevelopment results saved to: {file_path}")




def train_loop(data, optimizer, criterion_slots, criterion_intents, model, device, clip=5):
    """
    Trains the model for one epoch.

    Args:
        data (DataLoader): DataLoader object that yields batches of training samples.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        criterion_slots (nn.CrossEntropyLoss): Loss function for slot filling.
        criterion_intents (nn.CrossEntropyLoss): Loss function for intent detection.
        model (nn.Module): The neural network model to be trained.
        device (torch.device): CPU or CUDA device.
        clip (float): Max norm for gradient clipping.

    Returns:
        list: List of loss values per batch for monitoring/tracking.
    """

    model.train()       # Set model to training mode 
    loss_array = []     # To store loss values for each batch

    # Iterate over each batch in the training data
    for sample in data:
        optimizer.zero_grad() # Clear previous gradients

        # Move tensors to device
        input_ids = sample['input_ids'].to(device)
        attention_mask = sample['attention_mask'].to(device)
        intent_labels = sample['intent_label'].to(device)
        slot_labels = sample['slot_labels'].to(device)

        # Forward pass
        slot_logits, intent_logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Compute intent loss
        loss_intent = criterion_intents(intent_logits, intent_labels)

        # Compute slot loss
        loss_slot = criterion_slots(slot_logits, slot_labels)

        # Total loss (equal weight)
        loss = loss_intent + loss_slot
        loss_array.append(loss.item())      # Save the loss value for logging/plotting

        # Optional question: Is there another way to combine these losses?
        # Yes! For example:
        # - Weighted sum: loss = α * loss_intent + β * loss_slot
        # - Task-specific weighting or dynamic loss balancing methods                          

        # Backward pass: compute gradients
        loss.backward() 

        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  

        # Update model parameters using the optimizer
        optimizer.step() 

    return loss_array




def eval_loop(data, tokenizer, criterion_slots, criterion_intents, intent_list, slot_list, model, device):
    """
    Evaluation loop over the dataset. Computes loss and metrics for intent classification
    and slot filling tasks.

    Args:
        data (DataLoader): DataLoader object that yields batches of samples.
        tokenizer: (BertTokenizer): Tokenizer used to convert text to token IDs.
        criterion_slots (nn.CrossEntropyLoss): Loss function for slot filling.
        criterion_intents (nn.CrossEntropyLoss): Loss function for intent detection.
        intent_list (list): List of intent labels.
        slot_list (list): List of slot labels.
        model (nn.Module): The neural network model in evaluation mode.
        device (torch.device): CPU or CUDA device.

    Returns:
        results (dict): Dictionary with slot-level evaluation metrics.
        report_intent (dict): Classification report for intents.
        loss_array (list): List of batch losses during evaluation.
    """

    model.eval()        # Set model to evaluation mode
    loss_array = []     # Store batch losses
    
    ref_intents = []    # Ground truth intent labels
    hyp_intents = []    # Predicted intent labels
    
    ref_slots = []      # Ground truth slot labels
    hyp_slots = []      # Predicted slot labels

    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            # Move inputs to device
            input_ids = sample['input_ids'].to(device)
            attention_mask = sample['attention_mask'].to(device)
            intent_labels = sample['intent_label'].to(device)
            slot_labels = sample['slot_labels'].to(device)

            # Forward pass
            slot_logits, intent_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # Compute losses
            loss_intent = criterion_intents(intent_logits, intent_labels)   # Compute intent loss
            loss_slot = criterion_slots(slot_logits, slot_labels)           # Compute slot loss
            loss = loss_intent + loss_slot                                  # Total loss (equal weight)
            loss_array.append(loss.item()) 


            # --- Intent evaluation ---
            pred_intents = torch.argmax(intent_logits, dim=1).tolist()      # Take argmax over intent logits
            gold_intents = intent_labels.tolist()                           # Ground truth intent labels
            hyp_intents.extend([intent_list[i] for i in pred_intents])      # Convert predicted IDs to labels
            ref_intents.extend([intent_list[i] for i in gold_intents])      # Convert ground truth IDs to labels


            # --- Slot evaluation ---
            pred_slots = torch.argmax(slot_logits, dim=2)    # Shape: (batch, seq_len)
            for i in range(len(input_ids)):
                seq_len = (attention_mask[i] == 1).sum().item()  # number of real tokens

                input_tokens = input_ids[i][:seq_len].tolist()
                pred_slot_ids = pred_slots[i][:seq_len].tolist()
                true_slot_ids = slot_labels[i][:seq_len].tolist()

                # Convert input token IDs to words
                words = tokenizer.convert_ids_to_tokens(input_tokens)

                # Convert gold and predicted slot IDs to labels
                gold_slots = [slot_list[sid] for sid in true_slot_ids]
                pred_slots_labels = [slot_list[sid] for sid in pred_slot_ids]

                # Add to references and hypotheses
                ref_slots.append(list(zip(words, gold_slots)))
                hyp_slots.append(list(zip(words, pred_slots_labels)))
    try:  
        # Evaluate slot filling using a custom evaluate function 
        results = evaluate(ref_slots, hyp_slots)

    except Exception as ex:
        # Handle cases where the model predicts unseen/invalid slot labels
        print("Warning:", ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print(hyp_s.difference(ref_s))
        results = {"total":{"f":0}}     # Default if evaluation fails
    
    # Generate classification report for intents
    report_intent = classification_report(
        ref_intents, 
        hyp_intents, 
        zero_division=False, 
        output_dict=True
    )


    # Calculate beta 95 confidence interval for intent accuracy 
    correct = sum(r == h for r, h in zip(ref_intents, hyp_intents))
    total = len(ref_intents)

    ci_beta_low, ci_beta_high = st.beta.interval(0.95, correct + 1, total - correct + 1)
    report_intent['ci_95_beta'] = (ci_beta_low, ci_beta_high)


    # Calculate the sem and 95 confidence interval for slot F1 score
    slot_f1 = results['total']['f']
    n_slots = results['total']['s']
    if n_slots > 0:
        sem_f1 = (slot_f1 * (1 - slot_f1) / n_slots) ** 0.5
        ci_f1_low, ci_f1_high = st.norm.interval(0.95, loc=slot_f1, scale=sem_f1)
    else:
        sem_f1 = 0
        ci_f1_low, ci_f1_high = 0, 0

    results['total']['f1_ci_95'] = (ci_f1_low, ci_f1_high)
    results['total']['sem'] = sem_f1
    
    return results, report_intent, loss_array
