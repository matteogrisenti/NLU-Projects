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



def init_weights(mat):
    """
    Applies custom weight initialization to all modules in the given model.

    Xavier uniform is used for input-to-hidden connections,
    orthogonal initialization for hidden-to-hidden connections in RNNs,
    and small uniform values for linear layers.
    """
    for m in mat.modules():     # Iterate over all modules in the model
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            # Special handling for RNN-based layers
            for name, param in m.named_parameters():
                # Input-to-hidden weights
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
                # Hidden-to-hidden weights
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
                # Bias terms
                elif 'bias' in name:
                    param.data.fill_(0)     # Set biases to zero
        else:
            if type(m) in [nn.Linear]:
                # Initialize linear layer weights with small uniform values
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)     # Initialize bias with small constant




def model_name(label, lr, hid_size, emb_size, batch_size, dropout, n_layer):
    name = f"{label}_lr-{str(lr).replace('.', ',')}_hid-{hid_size}_emb-{emb_size}_batch-{batch_size}_layers-{n_layer}"
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




def save_model(name, model, optimizer, hid_size, out_slot, out_int, emb_size, vocab_len, n_layer, pad_index, dropout=None):
    path = 'bin/others/' + name + '.pt'
    saving_object = { "model": model.state_dict(), 
                      "optimizer": optimizer.state_dict(), 
                      "hid_size": hid_size,
                      "out_slot": out_slot,
                      "out_int": out_int,
                      "emb_size": emb_size,
                      "vocab_len": vocab_len,
                      "n_layer": n_layer,
                      "pad_index": pad_index
                    }
    if dropout is not None:
        saving_object["dropout"] = dropout

    print(f"\tSaving model to {path}:")
    print(f"\t\t hid_size: {hid_size} \n\t\t out_slot: {out_slot} \n\t\t out_int: {out_int} \n\t\t emb_size: {emb_size} \n\t\t vocab_len: {vocab_len} \n\t\t n_layer: {n_layer} \n\t\t pad_index: {pad_index} \n\t\t dropout: {dropout}")
    torch.save(saving_object, path)





def save_dev(label, lr, n_layer, hid_size, emb_size, batch_size, dropout, 
                 slot_f1, f1_ci_95, intent_accuracy, ci_95_beta):
    """
    Saves training/validation or test results along with hyperparameters to a CSV file.

    Args:
        label (str): Name of the experiment/run.
        lr (float): Learning rate.
        n_layer (int): Number of LSTM layers.
        hid_size (int): Hidden size of LSTM.
        emb_size (int): Embedding size.
        batch_size (int): Batch size used during training.
        dropout (float): Dropout rate.
        slot_f1: Slot F1 score.
        f1_ci_95: 95% confidence interval for F1 score.
        intent_accuracy: Intent accuracy.
        ci_95_beta: 95% confidence interval for intent accuracy.
    """
    
    # Create file if it doesn't exist, append otherwise
    filename = 'results/dev.csv'
    file_exists = os.path.isfile(filename)

    # Prepare data to write
    data = {
        'label': label,
        'learning_rate': lr,
        'n_layers': n_layer,
        'hidden_size': hid_size,
        'embedding_size': emb_size,
        'batch_size': batch_size,
        'dropout': dropout or None,
        'slot_f1': round(slot_f1, 4),             # Slot F1 score rounded to 2 decimal places
        '95% CI': f"{round(f1_ci_95[0], 4)} - {round(f1_ci_95[1], 4)}",  # 95% CI for F1 score
        'intent_acc': round(intent_accuracy, 4),   # Intent accuracy rounded to 2 decimal places
        '95% CI (beta)': f"{round(ci_95_beta[0], 4)} - {round(ci_95_beta[1], 4)}"  # 95% CI for intent accuracy
    }

    # Define fieldnames for CSV header
    fieldnames = [
        'label', 'learning_rate', 'n_layers', 'hidden_size', 'embedding_size',
        'batch_size', 'dropout', 'slot_f1', '95% CI', 'intent_acc', '95% CI (beta)'
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




def train_loop(data, optimizer, criterion_slots, criterion_intents, model, clip=5):
    """
    Trains the model for one epoch.

    Args:
        data (DataLoader): DataLoader object that yields batches of training samples.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        criterion_slots (nn.CrossEntropyLoss): Loss function for slot filling.
        criterion_intents (nn.CrossEntropyLoss): Loss function for intent detection.
        model (nn.Module): The neural network model to be trained.
        clip (float): Max norm for gradient clipping.

    Returns:
        list: List of loss values per batch for monitoring/tracking.
    """

    model.train()       # Set model to training mode 
    loss_array = []     # To store loss values for each batch

    # Iterate over each batch in the training data
    for sample in data:
        optimizer.zero_grad() # Clear previous gradients

        # Forward pass: compute model outputs
        # Inputs:
        #   - utterances: token indices of shape (batch_size, seq_len)
        #   - slots_len: lengths of sequences for pack_padded_sequence
        slots, intent = model(sample['utterances'], sample['slots_len'])

        # DEBUG: Print intent label info
        # print("Intent targets:", sample['intents'])
        # print("Intent shape:", sample['intents'].shape)
        # print("Intent min/max:", sample['intents'].min().item(), sample['intents'].max().item())
        # print("Num intent classes:", intent.shape[1])

        # VALIDATE intent labels
        num_classes = intent.shape[1]
        assert (sample['intents'] >= 0).all(), "Negative intent labels found!"
        assert (sample['intents'] < num_classes).all(), f"Intent label >= {num_classes} found!"


        # Compute intent loss
        # intent: (batch_size, num_intents)
        # sample['intents']: (batch_size,)
        loss_intent = criterion_intents(intent, sample['intents'])

        # Compute slot loss
        # slots: (batch_size, num_slots, seq_len) <- permuted for CrossEntropyLoss
        # sample['y_slots']: (batch_size, seq_len)
        loss_slot = criterion_slots(slots, sample['y_slots'])

        # Combine the losses
        # This is a simple multi-task learning setup where both tasks are equally weighted
        loss = loss_intent + loss_slot 

        # Optional question: Is there another way to combine these losses?
        # Yes! For example:
        # - Weighted sum: loss = α * loss_intent + β * loss_slot
        # - Task-specific weighting or dynamic loss balancing methods
                                       
        loss_array.append(loss.item()) # Save the loss value for logging/plotting

        # Backward pass: compute gradients
        loss.backward() 

        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  

        # Update model parameters using the optimizer
        optimizer.step() 

    return loss_array




def eval_loop(data, criterion_slots, criterion_intents, model, lang):
    """
    Evaluation loop over the dataset. Computes loss and metrics for intent classification
    and slot filling tasks.

    Args:
        data (DataLoader): DataLoader object that yields batches of samples.
        criterion_slots (nn.CrossEntropyLoss): Loss function for slot filling.
        criterion_intents (nn.CrossEntropyLoss): Loss function for intent detection.
        model (nn.Module): The neural network model in evaluation mode.
        lang (object): An object containing mappings between IDs and labels:
                       - lang.id2intent: maps intent ID to intent name
                       - lang.id2slot: maps slot ID to slot name
                       - lang.id2word: maps word ID to actual word

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
            # Forward pass: get model outputs
            slots, intents = model(sample['utterances'], sample['slots_len'])

            # Compute losses
            loss_intent = criterion_intents(intents, sample['intents'])
            loss_slot = criterion_slots(slots, sample['y_slots'])
            loss = loss_intent + loss_slot 
            loss_array.append(loss.item())

            # Intent prediction: take argmax over classes
            # print("Intent logits:", torch.argmax(intents, dim=1).tolist())
            # print("Valid intent IDs:", list(lang.id2intent.keys()))
            out_intents = [lang.id2intent[x] 
                           for x in torch.argmax(intents, dim=1).tolist()] 
            
            # Ground truth intents
            gt_intents = [lang.id2intent[x] for x in sample['intents'].tolist()]

            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)
            
            # Slot prediction: take argmax over slot logits
            output_slots = torch.argmax(slots, dim=1)

            # For each sequence in the batch
            for id_seq, seq in enumerate(output_slots):
                length = sample['slots_len'].tolist()[id_seq]
                utt_ids = sample['utterance'][id_seq][:length].tolist()
                gt_ids = sample['y_slots'][id_seq].tolist()

                # Ground truth slots (convert from IDs to labels)
                gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]

                # Input utterance words (convert from IDs to actual words)
                utterance = [lang.id2word[elem] for elem in utt_ids]

                # Predicted slots (take only up to valid length)
                to_decode = seq[:length].tolist()

                # Reference slots: pair each word with its ground truth label
                ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])
                tmp_seq = []

                # Hypothesis slots: pair each word with predicted label
                for id_el, elem in enumerate(to_decode):
                    tmp_seq.append((utterance[id_el], lang.id2slot[elem]))
                hyp_slots.append(tmp_seq)

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
    lang,
    optimizer,
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
        lang (object): Language object containing label mappings.
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

            
            for epoch in tqdm(range(1, n_epochs + 1)):

                # Training step
                model.train()
                loss = train_loop(train_loader, optimizer, criterion_slots, criterion_intents, model, clip=clip)
                losses_train.append(np.mean(loss))

                # Evaluation step
                if epoch % eval_every == 0:
                    sampled_epochs.append(epoch)

                    model.eval()
                    results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots, criterion_intents, model, lang)
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
            save_dev(hyperparameters['label'], hyperparameters['lr'], hyperparameters['n_layer'],
                     hyperparameters['hid_size'], hyperparameters['emb_size'], hyperparameters['batch_size'], 
                     hyperparameters['dropout'], dev_results['results_dev']['total']['f'], dev_results['results_dev']['total']['f1_ci_95'],
                     dev_results['intent_res']['accuracy'], dev_results['intent_res']['ci_95_beta'])  
            # Save best model
            save_model(model_name, best_model, optimizer, hyperparameters['hid_size'],  len(lang.slot2id), len(lang.intent2id), hyperparameters['emb_size'],
                       len(lang.word2id), hyperparameters['n_layer'], lang.PAD_TOKEN, hyperparameters['dropout'])  # Save the best model

            print("Training completed.")

        finally:
            sys.stdout = sys.stdout.stdout   # Restore stdout

    return best_model




def save_test(label, lr, n_layer, hid_size, emb_size, batch_size, dropout, 
              slot_f1, f1_ci_95, intent_accuracy, ci_95_beta):
    """
    Saves training/validation or test results along with hyperparameters to a CSV file.

    Args:
        label (str): Name of the experiment/run.
        lr (float): Learning rate.
        n_layer (int): Number of LSTM layers.
        hid_size (int): Hidden size of LSTM.
        emb_size (int): Embedding size.
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

    # Prepare data to write
    data = {
        'label': label,
        'learning_rate': lr,
        'n_layers': n_layer,
        'hidden_size': hid_size,
        'embedding_size': emb_size,
        'batch_size': batch_size,
        'dropout': dropout or None,
        'slot_f1': round(slot_f1, 4),             # Slot F1 score rounded to 2 decimal places
        '95% CI': f"{round(f1_ci_95[0], 4)} - {round(f1_ci_95[1], 4)}",  # 95% CI for F1 score
        'intent_acc': round(intent_accuracy, 4),   # Intent accuracy rounded to 2 decimal places
        '95% CI (beta)': f"{round(ci_95_beta[0], 4)} - {round(ci_95_beta[1], 4)}"  # 95% CI for intent accuracy
    }

    # Define fieldnames for CSV header
    fieldnames = [
        'label', 'learning_rate', 'n_layers', 'hidden_size', 'embedding_size',
        'batch_size', 'dropout', 'slot_f1', '95% CI', 'intent_acc', '95% CI (beta)'
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
    criterion_slots,
    criterion_intents,
    lang,
    model_name="best_model",
    device=None,
    hyperparameters=None
):
    """
    Loads the best model and evaluates it on the test dataset.

    Args:
        model (nn.Module): Model architecture (untrained/unloaded).
        test_loader (DataLoader): DataLoader for test data.
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

    results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, model, lang)

    save_test_results(results_test, intent_test, model_name)
    # Save the dev results in a CSV file
    save_test(hyperparameters['label'], hyperparameters['lr'], hyperparameters['n_layer'],
              hyperparameters['hid_size'], hyperparameters['emb_size'], hyperparameters['batch_size'], 
              hyperparameters['dropout'], results_test['total']['f'], results_test['total']['f1_ci_95'],
              intent_test['accuracy'], intent_test['ci_95_beta'])  
    
    print('Slot F1:', results_test['total']['f'])
    print('Intent Accuracy:', intent_test['accuracy'])

    return results_test, intent_test