import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from tqdm import tqdm

from model import ModelIAS
from utils import init_dataloader
from functions import init_weights, model_name, train_model,test_model
from plot import plot_all


device = 'cuda:0' # cuda:0 means we are using the GPU with id 0, if you have multiple GPU
os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # Used to report errors on CUDA side

PAD_TOKEN = 0

# -------------------------------------- HYPERPARAMETERS ------------------------------------------------
label = 'SimpleIAS'

hid_size = 200
emb_size = 300
n_layer = 1
batch_size = 128

lr = 0.0001 # learning rate
clip = 5 # Clip the gradient
dropout = None

name = model_name(label, lr, hid_size, emb_size, batch_size, dropout, n_layer)

# ------------------------------------- DATASET MENAGMENT -----------------------------------------------
train_loader, dev_loader, test_loader, lang = init_dataloader(batch_size, PAD_TOKEN, device, name)

out_slot = len(lang.slot2id)
out_int = len(lang.intent2id)
vocab_len = len(lang.word2id)


# -------------------------------------- MODEL DEFINITION ------------------------------------------------
model = ModelIAS(hid_size, out_slot, out_int, emb_size, vocab_len, n_layer=n_layer, pad_index=PAD_TOKEN).to(device)
model.apply(init_weights)



# ------------------------------------------ TRAINING ---------------------------------------------------
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
criterion_intents = nn.CrossEntropyLoss() # Because we do not have the pad token

n_epochs = 200
patience = 3

model = train_model(model, train_loader, dev_loader, lang, optimizer, criterion_slots, criterion_intents, 
                    n_epochs, patience, clip, model_name=name, device=device)


# ------------------------------------------ TESTING ----------------------------------------------------
results_test, intent_test = test_model(model, test_loader, criterion_slots, criterion_intents, lang, 
                                       name, device=device)


# ------------------------------------------ PLOTTING ---------------------------------------------------
plot_all(name)