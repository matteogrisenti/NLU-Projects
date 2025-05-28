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

from utils import get_slots_intents_lists



def model_name(bert_type, lr, batch_size, dropout):

     # Format bert type 
    if bert_type == 'bert-base-uncased':
        label = 'base'
    elif bert_type == 'bert-large-uncased':
        label = 'large'
    else:
        raise ValueError(f"Unknown BERT type: {bert_type}")
    
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




def save_model(name, model, bert_type, dropout, num_intent_labels, num_slot_labels):
    path = 'bin/others/' + name + '.pt'
    saving_object = { 
        "model": model.state_dict(), 
        "bert_type": bert_type, 
        "dropout": dropout,
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
        'Bert Type', 'Learning Rate', 'Batch Size', 'Dropout', 'Slot F1', '95% CI', 'Intent Acc', '95% CI (beta)'
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

    count_batch = 0; 

    # Iterate over each batch in the training data
    for batch in data:
        count_batch=count_batch+1
        optimizer.zero_grad() # Clear previous gradients

        # Move tensors to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        intent_labels = batch['intent_label'].to(device)
        slot_labels = batch['slot_labels'].to(device)

        print(f"=== BATCH {count_batch} DEBUG START ===")
        print("input_ids.shape:", input_ids.shape)
        print("attention_mask.shape:", attention_mask.shape)
        print("slot_labels.shape:", slot_labels.shape)
        print("intent_labels.shape:", intent_labels.shape)

        # Forward pass
        slot_logits, intent_logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Compute intent loss
        loss_intent = criterion_intents(intent_logits, intent_labels)
        print("loss intent: ", loss_intent)
        
        # Flatten logits and labels for slot filling
        slot_logits_flat = slot_logits.view(-1, slot_logits.shape[-1])          # [batch_size * seq_len, num_classes]
        slot_labels_flat = slot_labels.view(-1)  
        
        # Compute slot loss
        loss_slot = criterion_slots(slot_logits_flat, slot_labels_flat)
        print("loss slot: ", loss_slot)

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

        print("=== BATCH DEBUG END ===")

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




def train_model(
    model,
    train_loader,
    dev_loader,
    tokenizer,
    criterion_slots, 
    criterion_intents,
    n_epochs=200,
    patience=3,
    clip=5,
    eval_every=5,
    model_name="best_model",
    device=None,
    hyperparameters=None
):
    """
    Trains a joint intent detection and slot filling model with early stopping.

    Args:
        model (nn.Module): The neural network model to train.
        train_loader (DataLoader): DataLoader for training data.
        dev_loader (DataLoader): DataLoader for validation data.
        tokenizer: (BertTokenizer): Tokenizer used to convert text to token IDs. 
        optimizer (torch.optim.Optimizer): Optimizer used for parameter updates.
        criterion_slots (nn.CrossEntropyLoss): Loss function for slot filling.
        criterion_intents (nn.CrossEntropyLoss): Loss function for intent classification.
        n_epochs (int): Maximum number of epochs to train.
        patience (int): Number of epochs without improvement before early stopping.
        clip (float): Gradient norm clipping threshold.
        eval_every (int): Frequency (in epochs) of evaluation on dev set.
        model_name (str): Name of the model ( used to save it's performance ).
        device (str or torch.device): Device to run the model on ('cuda', 'cpu', or None).
                                     If None, uses CUDA if available.
        hyperparameters (dict): Hyperparameters for the model (optional).

    Returns:
        best_model (nn.Module): The best saved model based on dev performance.
    """
    
    # Set default device if not provided
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Create logging file
    save_dir = os.path.join('models', model_name)
    os.makedirs(save_dir, exist_ok=True)
    log_file = save_dir + "/training.txt"
    print("\nTraining started...")
    print(f"\tLogging training output to {log_file}")

    # Redirect stdout to both console and file
    class Logger:
        def __init__(self, file):
            self.file = file
            self.stdout = sys.stdout
            
        def write(self, data):
            self.stdout.write(data)
            self.file.write(data)
            
        def flush(self):
            self.stdout.flush()
            self.file.flush()

    # Open log file and redirect output
    with open(log_file, 'w') as f:
        sys.stdout = Logger(f)

        try:
            losses_train = []       # To store training losses
            losses_dev = []         # To store dev losses
            sampled_epochs = []     # To store epochs where dev loss was recorded
            
            best_model = None       # To store the best model
            best_f1 = 0.0           # Initialize best F1 score
            no_improvement = 0      # Counter for early stopping

            dev_results = {}        # Development results of the best model

            # Define the optimizer 
            optimizer = optim.Adam(model.parameters(), lr=hyperparameters['learnin_rate'])

            for epoch in tqdm(range(1, n_epochs + 1)):

                # Training step
                model.train()
                loss = train_loop(train_loader, optimizer, criterion_slots, criterion_intents, model, device, clip=clip)
                losses_train.append(np.mean(loss))

                # Evaluation step
                if epoch % eval_every == 0:
                    sampled_epochs.append(epoch)

                    model.eval()
                    results_dev, intent_res, loss_dev = eval_loop(dev_loader, tokenizer, criterion_slots, criterion_intents, model, device)
                    losses_dev.append(np.mean(loss_dev))

                    current_f1 = results_dev['total']['f']
                    print(f"Epoch {epoch} | Dev Slot F1: {current_f1:.4f} | Intent Acc: {intent_res['accuracy']:.4f}")

                    # Save best model
                    if current_f1 > best_f1:
                        print(f"New best F1: {current_f1:.4f}")
                        best_f1 = current_f1
                        no_improvement = 0
                        best_model = deepcopy(model)        # Save the best model
                        dev_results = {
                            "loss_dev": loss_dev,
                            "results_dev": results_dev,
                            "intent_res": intent_res
                        }
                    else:
                        no_improvement += 1

                    # Early stopping
                    if no_improvement >= patience:
                        print("Early stopping triggered.")
                        break

            save_training(sampled_epochs, losses_train, losses_dev, model_name)  # Save the training data in a JSON file
            save_dev_results(dev_results, model_name)                            # Save the dev results in a JSON file
            
            # Save the dev results in a CSV file
            save_dev(hyperparameters['bert_type'], hyperparameters['lr'], hyperparameters['batch_size'], 
                     hyperparameters['dropout'], dev_results['results_dev']['total']['f'], dev_results['results_dev']['total']['f1_ci_95'],
                     dev_results['intent_res']['accuracy'], dev_results['intent_res']['ci_95_beta'])  
            
            # Save best model
            save_model(model_name, best_model, hyperparameters['bert_type'], hyperparameters['dropout'], hyperparameters['num_intents_label', hyperparameters['num_slots_label']])# Save the best model

            print("Training completed.")

        finally:
            sys.stdout = sys.stdout.stdout   # Restore stdout

    return best_model




def save_test(bert_type, lr, batch_size, dropout, slot_f1, f1_ci_95, intent_accuracy, ci_95_beta):
    """
    Saves training/validation or test results along with hyperparameters to a CSV file.

    Args:
        bert_type (str): Type of BERT model used: 'bert-base-uncased' or 'bert-large-uncased'.
        lr (float): Learning rate.
        batch_size (int): Batch size used during training.
        dropout (float): Dropout rate.
        slot_f1: Slot F1 score.
        f1_ci_95: 95% confidence interval for F1 score.
        intent_accuracy: Intent accuracy.
        ci_95_beta: 95% confidence interval for intent accuracy.
    """
    
    # Create file if it doesn't exist, append otherwise
    filename = 'results/test.csv'
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
        'dropout': dropout or 'None',
        'slot_f1': round(slot_f1, 4),             # Slot F1 score rounded to 2 decimal places
        '95% CI': f"{round(f1_ci_95[0], 4)} - {round(f1_ci_95[1], 4)}",  # 95% CI for F1 score
        'intent_acc': round(intent_accuracy, 4),   # Intent accuracy rounded to 2 decimal places
        '95% CI (beta)': f"{round(ci_95_beta[0], 4)} - {round(ci_95_beta[1], 4)}"  # 95% CI for intent accuracy
    }

    # Define fieldnames for CSV header
    fieldnames = [
        'Bert Type', 'Learning Rate', 'Batch Size', 'Dropout', 'Slot F1', '95% CI', 'Intent Acc', '95% CI (beta)'
    ]

    # Write to CSV
    with open(filename, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()  # Write header only once

        writer.writerow(data)  # Write the row with results and hyperparams

    print(f"\tResults saved to {filename}")




def save_test_results(results_test, intent_test, model_name):
    """
    Save test results in a nicely formatted way to JSON file.

    Args:
        results_test (dict): Slot filling results (F1 scores per tag)
        intent_test (dict): Intent classification results (precision/recall/f1)
        model_name (str): Name of the model for saving purposes
    """
    # Create directories if not exist
    save_dir = os.path.join('models', model_name)
    os.makedirs(save_dir, exist_ok=True)
    file_path_json = save_dir + '/test_data.json'

    slot_f1 = results_test['total']['f']
    intent_acc = intent_test['accuracy']

    full_results = {
        "slot_results": results_test,
        "intent_results": intent_test,
        "metrics": {
            "slot_f1": slot_f1,
            "intent_accuracy": intent_acc
        }
    }

    with open(file_path_json, 'w', encoding='utf-8') as fj:
        json.dump(full_results, fj, indent=4, ensure_ascii=False)

    print(f"\tResults saved to: {file_path_json}")




def test_model(
    model,
    test_loader,
    tokenizer,
    criterion_slots,
    criterion_intents,
    model_name="best_model",
    device=None,
    hyperparameters=None
):
    """
    Loads the best model and evaluates it on the test dataset.

    Args:
        model (nn.Module): Model architecture (untrained/unloaded).
        test_loader (DataLoader): DataLoader for test data.
        tokenizer: (BertTokenizer): Tokenizer used to convert text to token IDs.
        criterion_slots (nn.CrossEntropyLoss): Loss function for slots.
        criterion_intents (nn.CrossEntropyLoss): Loss function for intents.
        lang (object): Language object containing label mappings.
        model_name (str): name of the best model was saved.
        device (str or torch.device): Device to run the model on ('cuda', 'cpu', or None).
                                      If None, uses CUDA if available.
        hyperparameters (dict): Hyperparameters for the model (optional).

    Returns:
        results_test (dict): Dictionary with test slot metrics.
        intent_test (dict): Classification report for intents.
    """

    # Set default device if not provided
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results_test, intent_test, _ = eval_loop(test_loader, tokenizer, criterion_slots, criterion_intents, model)

    save_test_results(results_test, intent_test, model_name)
    # Save the dev results in a CSV file
    save_test(hyperparameters['batch_type'], hyperparameters['lr'], hyperparameters['batch_size'], 
              hyperparameters['dropout'], results_test['total']['f'], results_test['total']['f1_ci_95'],
              intent_test['accuracy'], intent_test['ci_95_beta'])  
    
    print('Slot F1:', results_test['total']['f'])
    print('Intent Accuracy:', intent_test['accuracy'])

    return results_test, intent_test