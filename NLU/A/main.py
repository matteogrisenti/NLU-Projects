import os
import torch
import shutil
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from tqdm import tqdm

from model import ModelIAS, BiModelIAS, DoModelIAS
from utils import init_dataloader, init_dataloader_test
from functions import init_weights, model_name, train_model,test_model
from plot import plot_all


def training_dev_model(hyperparameters):
    """
    This function is used to train and dev a model, it will also plot the results.

    Args:
        hyperparameters (dict): A dictionary containing the hyperparameters for the model.
            - label (str): The label for the model.
            - hid_size (int): The hidden size of the model.
            - emb_size (int): The embedding size of the model.
            - n_layer (int): The number of layers in the model.
            - batch_size (int): The batch size for training.
            - lr (float): The learning rate for the optimizer.
            - clip (float): The gradient clipping value.
            - dropout (float): The dropout rate for the model.
    """
    # Unpack hyperparameters
    label = hyperparameters['label']
    hid_size = hyperparameters['hid_size']
    emb_size = hyperparameters['emb_size']
    n_layer = hyperparameters['n_layer']
    batch_size = hyperparameters['batch_size']
    lr = hyperparameters['lr']
    clip = hyperparameters['clip']
    dropout = hyperparameters['dropout']

    name = model_name(label, lr, hid_size, emb_size, batch_size, dropout, n_layer)

    # ------------------------------------- DATASET MENAGMENT -----------------------------------------------
    train_loader, dev_loader, _ , lang = init_dataloader(batch_size, PAD_TOKEN, device, name)

    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)
    vocab_len = len(lang.word2id)


    # -------------------------------------- MODEL DEFINITION ------------------------------------------------
    # model = ModelIAS(hid_size, out_slot, out_int, emb_size, vocab_len, n_layer=n_layer, pad_index=PAD_TOKEN).to(device)
    # model = BiModelIAS(hid_size, out_slot, out_int, emb_size, vocab_len, n_layer=n_layer, pad_index=PAD_TOKEN).to(device)
    model = DoModelIAS(hid_size, out_slot, out_int, emb_size, vocab_len, n_layer=n_layer, pad_index=PAD_TOKEN, dropout=dropout).to(device)
    model.apply(init_weights)



    # ------------------------------------------ TRAINING ---------------------------------------------------
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss() # Because we do not have the pad token

    n_epochs = 200
    patience = 3

    model = train_model(model, train_loader, dev_loader, lang, optimizer, criterion_slots, criterion_intents, 
                        n_epochs, patience, clip, model_name=name, device=device, hyperparameters=hyperparameters)

    # ------------------------------------------ PLOTTING ---------------------------------------------------
    plot_all(name)


def testing_model(hyperparameters):
    """
    This function is used to test a model

    Args:
        hyperparameters (dict): A dictionary containing the hyperparameters for the model.
            - label (str): The label for the model.
            - hid_size (int): The hidden size of the model.
            - emb_size (int): The embedding size of the model.
            - n_layer (int): The number of layers in the model.
            - batch_size (int): The batch size for training.
            - lr (float): The learning rate for the optimizer.
            - clip (float): The gradient clipping value.
            - dropout (float): The dropout rate for the model.
    """
    # Unpack hyperparameters
    label = hyperparameters['label']
    hid_size = hyperparameters['hid_size']
    emb_size = hyperparameters['emb_size']
    n_layer = hyperparameters['n_layer']
    batch_size = hyperparameters['batch_size']
    lr = hyperparameters['lr']
    clip = hyperparameters['clip']
    dropout = hyperparameters['dropout']

    name = model_name(label, lr, hid_size, emb_size, batch_size, dropout, n_layer)
    old_path = os.path.join('bin', 'others', f"{name}.pt")
    new_path = os.path.join('bin', f"{name}.pt")

    # Move the file if it hasn't been moved already
    if os.path.exists(old_path) and not os.path.exists(new_path):
        shutil.move(old_path, new_path)

    # Load the checkpoint dict
    checkpoint = torch.load(new_path, weights_only=False)

    # Rebuild the model architecture
    # model = BiModelIAS(checkpoint['hid_size'], checkpoint['out_slot'], checkpoint['out_int'], checkpoint['emb_size'], checkpoint['vocab_len'], n_layer=checkpoint['n_layer'], pad_index=checkpoint['pad_index']).to(device)
    model = DoModelIAS(checkpoint['hid_size'], checkpoint['out_slot'], checkpoint['out_int'], checkpoint['emb_size'], checkpoint['vocab_len'], n_layer=checkpoint['n_layer'], pad_index=checkpoint['pad_index'], dropout=checkpoint['dropout']).to(device)
    
    # Load saved weights into model
    model.load_state_dict(checkpoint['model'])

    # Set model to evaluation mode
    model.eval()

    # Load the test data
    test_loader, lang = init_dataloader_test(batch_size, PAD_TOKEN, device, name)

    # Define the loss functions
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss() # Because we do not have the pad token

    # Test the model
    test_model(model, test_loader, criterion_slots, criterion_intents, lang, name, device=device, hyperparameters=hyperparameters)





# ------------------------------------------ SETUP ------------------------------------------------------
device = 'cuda:0' # cuda:0 means we are using the GPU with id 0, if you have multiple GPU
os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # Used to report errors on CUDA side

PAD_TOKEN = 0

# -------------------------------------- HYPERPARAMETERS ------------------------------------------------
label = 'Dropout'

hid_size = 900                # originally 200
emb_size = 600                # originally 300
n_layer = 1             # originally 1
batch_size = 32                # originally 128

lr = 0.001                           # originally 0.0001
clip = 5                             # originally 5
dropout = 0.5 

'''
for i in range(len(n_layer)):

    hyperparameters = {
        'label': label,
        'hid_size': hid_size,
        'emb_size': emb_size,
        'n_layer': n_layer[i],
        'batch_size': batch_size,
        'lr': lr,
        'clip': clip,
        'dropout': dropout
    }

    #---------------------------------- TRAINING AND DEVLOPMENT -----------------------------------------------
    training_dev_model(hyperparameters)
'''
# ------------------------------------------ TESTING ----------------------------------------------------
hyperparameters = {
    'label': label,
    'hid_size': hid_size,
    'emb_size': emb_size,
    'n_layer': n_layer,
    'batch_size': batch_size,
    'lr': lr,
    'clip': clip,
    'dropout': dropout
}

testing_model(hyperparameters)







