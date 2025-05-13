import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from tqdm import tqdm

from model import ModelIAS
from utils import init_dataloader
from functions import init_weights, model_name, train_loop, eval_loop


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


# ------------------------------------- DATASET MENAGMENT -----------------------------------------------
train_loader, dev_loader, test_loader, lang = init_dataloader(batch_size, PAD_TOKEN, device)

out_slot = len(lang.slot2id)
out_int = len(lang.intent2id)
vocab_len = len(lang.word2id)


# -------------------------------------- MODEL DEFINITION ------------------------------------------------
model = ModelIAS(hid_size, out_slot, out_int, emb_size, vocab_len, n_layer=n_layer, pad_index=PAD_TOKEN).to(device)
model.apply(init_weights)
model_name = model_name(label, lr, hid_size, emb_size, batch_size, dropout, n_layer)


# ------------------------------------------ TRAINING ---------------------------------------------------
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
criterion_intents = nn.CrossEntropyLoss() # Because we do not have the pad token

n_epochs = 200
patience = 3

losses_train = []
losses_dev = []
sampled_epochs = []

best_f1 = 0

for x in tqdm(range(1,n_epochs)):

    loss = train_loop(
        train_loader, 
        optimizer, 
        criterion_slots, 
        criterion_intents, 
        model, 
        clip=clip
    )

    # Check the performance every 5 epochs
    if x % 5 == 0: 
        sampled_epochs.append(x)
        losses_train.append(np.asarray(loss).mean())

        results_dev, intent_res, loss_dev = eval_loop(
            dev_loader, 
            criterion_slots, 
            criterion_intents, 
            model, 
            lang
        )

        losses_dev.append(np.asarray(loss_dev).mean())
        
        f1 = results_dev['total']['f']
        # For decreasing the patience you can also use the average between slot f1 and intent accuracy
        if f1 > best_f1:
            best_f1 = f1
            # Here you should save the model
            patience = 3
        else:
            patience -= 1
        if patience <= 0: # Early stopping with patience
            break # Not nice but it keeps the code clean

results_test, intent_test, _ = eval_loop(
    test_loader, 
    criterion_slots, 
    criterion_intents, 
    model, 
    lang
)

print('Slot F1: ', results_test['total']['f'])
print('Intent Accuracy:', intent_test['accuracy'])