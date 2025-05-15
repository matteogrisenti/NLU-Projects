import os
import sys
import csv
import json
import torch
import numpy as np
import torch.nn as nn
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





def save_model(epoch, model, optimizer, w2id, slot2id, intent2id, name):
    path = 'bin/' + name + '.pt'
    saving_object = { "epoch": epoch, 
                      "model": model.state_dict(), 
                      "optimizer": optimizer.state_dict(), 
                      "w2id": w2id, 
                      "slot2id": slot2id, 
                      "intent2id": intent2id
                    }
    torch.save(saving_object, path)




def save_results(label, lr, n_layer, hid_size, emb_size, batch_size, dropout, 
                 results_slot, intent_report, mod='TRAIN'):
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
        results_slot (dict): Slot filling results from evaluate(ref, hyp).
                             Expected key: 'total' -> {'f': f1_score}
        intent_report (dict): Intent classification report from sklearn.
                              Expected key: 'accuracy'
        mod (str): Mode: 'TRAIN' or 'TEST'. Determines which file to write to.
    """
    
    # Determine filename based on mode
    if mod == 'TRAIN':
        filename = 'results/dev.csv'
    elif mod == 'TEST':
        filename = 'results/test.csv'
    else:
        raise ValueError("Invalid mode. Use 'TRAIN' or 'TEST'.")
    
    # Create file if it doesn't exist, append otherwise
    file_exists = os.path.isfile(filename)
    
    # Extract metrics
    slot_f1 = results_slot['total']['f']
    intent_acc = intent_report['accuracy']

    # Prepare data to write
    data = {
        'label': label,
        'learning_rate': lr,
        'n_layers': n_layer,
        'hidden_size': hid_size,
        'embedding_size': emb_size,
        'batch_size': batch_size,
        'dropout': dropout,
        'slot_f1': slot_f1,
        'intent_acc': intent_acc,
    }

    # Define fieldnames for CSV header
    fieldnames = [
        'label', 'learning_rate', 'n_layers', 'hidden_size', 'embedding_size',
        'batch_size', 'dropout', 'slot_f1', 'intent_acc', 'timestamp'
    ]

    # Write to CSV
    with open(filename, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()  # Write header only once

        writer.writerow(data)  # Write the row with results and hyperparams

    print(f"\tResults saved to {filename}")




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
    device=None
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
            losses_train = []
            losses_dev = []
            sampled_epochs = []

            best_f1 = 0.0
            no_improvement = 0

            
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
                        best_model = deepcopy(model)
                    else:
                        no_improvement += 1

                    # Early stopping
                    if no_improvement >= patience:
                        print("Early stopping triggered.")
                        break

            save_training(sampled_epochs, losses_train, losses_dev, model_name)  # Save the training data
            save_model(epoch, best_model, optimizer, lang.word2id, lang.slot2id, lang.intent2id, model_name)  # Save best model

            print("Training completed.")

        finally:
            sys.stdout = sys.stdout.stdout   # Restore stdout

    return model




def save_test(results_test, intent_test, model_name):
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
    device=None
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

    Returns:
        results_test (dict): Dictionary with test slot metrics.
        intent_test (dict): Classification report for intents.
    """
    path = 'bin/' + model_name + '.pt'
    print(f"\nTest for model {model_name}")

    # Set default device if not provided
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # Load the saved checkpoint
        checkpoint = torch.load(path, map_location=device, weights_only=True)

        model.load_state_dict(checkpoint['model'])      # Load the model state dictionary
        model.to(device)                                # Move model to appropriate device
        model.eval()                                    # Set model to evaluation mode

        print(f"\tModel loaded successfully from {path}")
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found at {path}. Make sure the model was saved correctly.")
    
    except KeyError as e:
        raise KeyError(f"Missing key in state_dict: {e}. The saved model might be incomplete or incompatible.")

    except RuntimeError as e:
        raise RuntimeError(f"Error loading model: {e}. Check that the model architecture matches the saved weights.")
    
    except Exception as e:
        raise Exception(f"Unexpected error occurred while loading the model: {e}")

    results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, model, lang)

    save_test(results_test, intent_test, model_name)
        
    print('Slot F1:', results_test['total']['f'])
    print('Intent Accuracy:', intent_test['accuracy'])

    return results_test, intent_test