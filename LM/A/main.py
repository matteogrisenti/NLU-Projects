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
from functions import init_weights, train_loop, eval_loop, plot_training_progress, save_model, save_experiment_results


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", DEVICE)


train_raw = read_file("dataset/PennTreeBank/ptb.train.txt")
dev_raw = read_file("dataset/PennTreeBank/ptb.valid.txt")
test_raw = read_file("dataset/PennTreeBank/ptb.test.txt")

lang = Lang(train_raw, ["<pad>", "<eos>"])
# print(len(lang.word2id))

train_dataset = PennTreeBank(train_raw, lang)
dev_dataset   = PennTreeBank(dev_raw, lang)
test_dataset  = PennTreeBank(test_raw, lang)

# print("Train dataset size: ", len(train_dataset))
# print("Dev dataset size: ", len(dev_dataset))
# print("Test dataset size: ", len(test_dataset))

# Dataloader instantiation ( set of batch )
# NB: You can reduce the batch_size if the GPU memory is not enough
train_loader = DataLoader(train_dataset, batch_size=64,  collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"], DEVICE=DEVICE),  shuffle=True)
dev_loader   = DataLoader(dev_dataset,   batch_size=128, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"], DEVICE=DEVICE))
test_loader  = DataLoader(test_dataset,  batch_size=128, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"], DEVICE=DEVICE))


# Model configuration 
hid_size = 200  # Original 200
emb_size = 300  # Original 300
dropout_emb = 0.1
dropout_out = 0.1
lr = 1
clip = 5 # Clip the gradient -> avoid exploding gradients

print("Network Configuration:")
print("\tHidden size: ", hid_size)  
print("\tEmbedding size: ", emb_size)
print("\tLearning rate: ", lr)
print("\tDropout embedding: ", dropout_emb)
print("\tDropout output: ", dropout_out)

# Experiment also with a smaller or bigger model by changing hid and emb sizes 
# NB: A large model tends to overfit
# Don't forget to experiment with a lower training batch size
# Increasing the back propagation steps can be seen as a regularization step


vocab_len = len(lang.word2id)

model = LM_RNN(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"], out_dropout=dropout_out, emb_dropout=dropout_emb).to(DEVICE)
model.apply(init_weights)

optimizer = optim.SGD(model.parameters(), lr=lr)

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
    loss = train_loop(train_loader, optimizer, criterion_train, model, clip)    
    if epoch % 1 == 0:
        last_epoch += 1 
        sampled_epochs.append(epoch)
        losses_train.append(np.asarray(loss).mean())
        ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)

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

# Save the model
save_model(best_model, "DROP-OUT", lr, dropout_emb=dropout_emb, dropout_out=dropout_out)

# Plot the model
plot_training_progress(sampled_epochs, losses_train, losses_dev, ppl_list_dev, filename='DROP-OUT', lr=lr, dropout_emb=dropout_emb, dropout_out=dropout_out)

final_ppl, final_loss, ci_loss, ci_ppl = eval_loop(test_loader, criterion_eval, best_model)    

print('Test ppl: ', final_ppl)
save_experiment_results("DROP-OUT",lr, hid_size, emb_size, dropout_emb, dropout_out, 'SGD', last_epoch, final_ppl, final_loss, ci_loss, ci_ppl)