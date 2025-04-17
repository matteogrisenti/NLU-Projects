import math
import copy

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch import optim
from functools import partial
from torch.utils.data import DataLoader


from utils import read_file, Lang, PennTreeBank, collate_fn
from model import LM_RNN
from functions import init_weights, train_loop, eval_loop, test_eval_loop, path_define, plot_training_progress, save_model, save_experiment_results


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", DEVICE)



# --------------------------------------------  HYPERPARAMETERS ------------------------------------------------
LABEL = 'DROPOUT'
BATCH_SIZE = 64     # Original 64
HID_SIZE = 200      # Original 200
EMB_SIZE = 300      # Original 300
DROPOUT_EMB = 0.2
DROPOUT_OUT = 0.2
LR = 1
OPTIMIZER = 'SGD'   # SGD or Adam
CLIP = 5            # Clip the gradient -> avoid exploding gradients

print("HYPERPARAMETERS:")
print("\tBatch size: ", BATCH_SIZE)
print("\tHidden size: ", HID_SIZE)  
print("\tEmbedding size: ", EMB_SIZE)
print("\tLearning rate: ", LR)
print("\tDropout embedding: ", DROPOUT_EMB)
print("\tDropout output: ", DROPOUT_OUT)
print("\tGradient clipping: ", CLIP)



# ---------------------------------------------  DATASET MENAGMENT  ----------------------------------------------
train_raw = read_file("dataset/PennTreeBank/ptb.train.txt")
dev_raw = read_file("dataset/PennTreeBank/ptb.valid.txt")
test_raw = read_file("dataset/PennTreeBank/ptb.test.txt")

lang = Lang(train_raw, ["<pad>", "<eos>"])
#print(len(lang.word2id))

train_dataset = PennTreeBank(train_raw, lang)
dev_dataset   = PennTreeBank(dev_raw, lang)
test_dataset  = PennTreeBank(test_raw, lang)

print("DATASET:")
print("\tTrain dataset size: ", len(train_dataset))
print("\tDev dataset size: ", len(dev_dataset))
print("\tTest dataset size: ", len(test_dataset))

# Dataloader instantiation ( set of batch )
# NB: You can reduce the batch_size if the GPU memory is not enough
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,  collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"], DEVICE=DEVICE),  shuffle=True)
dev_loader   = DataLoader(dev_dataset,   batch_size=BATCH_SIZE*2, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"], DEVICE=DEVICE))
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE*2, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"], DEVICE=DEVICE))

print("\tNumber of batches in train_loader:", len(train_loader))
print("\tNumber of batches in dev_loader:", len(dev_loader))
print("\tNumber of batches in test_loader:", len(test_loader))



# --------------------------------------------- MODEL MENAGEMENT ----------------------------------------------
# Experiment also with a smaller or bigger model by changing hid and emb sizes 
# NB: A large model tends to overfit
# Don't forget to experiment with a lower training batch size
# Increasing the back propagation steps can be seen as a regularization step

vocab_len = len(lang.word2id)
model = LM_RNN(EMB_SIZE, HID_SIZE, vocab_len, pad_index=lang.word2id["<pad>"], out_dropout=DROPOUT_OUT, emb_dropout=DROPOUT_EMB).to(DEVICE)
model.apply(init_weights)

if OPTIMIZER == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=LR)
elif OPTIMIZER == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=LR)

criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')


n_epochs = 100
last_epoch = 0
patience = 3        # stop training if no improvement in 3 epochs in row -> model is overfitting
losses_train = []
losses_dev = []
ppl_list_dev = []
sampled_epochs = []
best_ppl = math.inf
best_model = None
pbar = tqdm(range(1,n_epochs))


#If the PPL is too high try to change the learning rate
for epoch in pbar:
    loss = train_loop(train_loader, optimizer, criterion_train, model, CLIP)    
    if epoch % 1 == 0:
        last_epoch += 1 
        sampled_epochs.append(epoch)
        losses_train.append(np.asarray(loss).mean())
        ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model, test=False)

        losses_dev.append(np.asarray(loss_dev).mean())
        ppl_list_dev.append(ppl_dev)

        pbar.set_description("PPL: %f" % ppl_dev)

        if  ppl_dev < best_ppl: # the lower, the better
            best_ppl = ppl_dev
            best_model = copy.deepcopy(model).to('cpu')
            patience = 3
        else:
            patience -= 1
            
        if patience <= 0: # Early stopping with patience
            print(" Early stopping at epoch ", last_epoch, " \n\tBest PPL: ", best_ppl, "\n\tLast PPL:", ppl_list_dev[-3:])
            break # Not nice but it keeps the code clean

best_model.to(DEVICE)



# --------------------------------------------- POST TRAINING -----------------------------------------
# 
path = path_define(LABEL, LR, HID_SIZE, EMB_SIZE, DROPOUT_EMB, DROPOUT_OUT, OPTIMIZER)

# Save the model
save_model(best_model, LABEL, LR, dropout_emb=DROPOUT_EMB, dropout_out=DROPOUT_OUT)

# Plot the model
plot_training_progress(sampled_epochs, losses_train, losses_dev, ppl_list_dev, filename=LABEL, lr=LR, dropout_emb=DROPOUT_EMB, dropout_out=DROPOUT_OUT)

# Evaluate the model on the test set
final_ppl, final_loss, sem_loss, ci_loss, sem_ppl, ci_ppl = test_eval_loop(test_loader, criterion_eval, best_model, test=True)    
print('Test ppl: ', final_ppl)

# Save the results in a CSV file
save_experiment_results(LABEL ,LR, HID_SIZE, EMB_SIZE, DROPOUT_EMB, DROPOUT_OUT, OPTIMIZER, last_epoch, final_ppl, final_loss, sem_loss, ci_loss, sem_ppl, ci_ppl)