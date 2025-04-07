import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os


def train_loop(data, optimizer, criterion, model, clip=5):
    # data: training datatest containing samples. Each sample is a dictionary with:
    #       - "source": a sequence of IDs representing the input.
    #       - "target": a sequence of IDs representing the output.
    #       - "number_tokens": the number of tokens in the sequence.
    # optimizer: optimizer to update the weights of the model.
    # criterion: loss function to compute the loss.
    # model: the model to be trained.
    # clip: gradient clipping value to avoid exploding gradients.

    model.train()
    loss_array = []
    number_of_tokens = []
    
    for sample in data:
        optimizer.zero_grad()               # Zeroing the gradient
        output = model(sample['source'])    # Forward pass: compute predicted outputs by passing inputs to the model
        
        loss = criterion(output, sample['target'])                  # Compute the loss
        loss_array.append(loss.item() * sample["number_tokens"])
        number_of_tokens.append(sample["number_tokens"])

        loss.backward() # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid explosioning gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  

        optimizer.step() # Update the weights
    
    # return the average loss over the batch
    return sum(loss_array)/sum(number_of_tokens)




def eval_loop(data, eval_criterion, model):
    # data: training datatest containing samples. Each sample is a dictionary with:
    #       - "source": a sequence of IDs representing the input.
    #       - "target": a sequence of IDs representing the output.
    #       - "number_tokens": the number of tokens in the sequence.
    # eval_criterion: loss function to compute the evaluation.
    # model: the model to be evaluated.

    model.eval()
    loss_to_return = []
    loss_array = []
    number_of_tokens = []

    # softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            output = model(sample['source'])
            loss = eval_criterion(output, sample['target'])
            loss_array.append(loss.item())
            number_of_tokens.append(sample["number_tokens"])
            
    ppl = math.exp(sum(loss_array) / sum(number_of_tokens))
    loss_to_return = sum(loss_array) / sum(number_of_tokens)

    return ppl, loss_to_return




def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)




def save_model(model, name, learning_rate):
    path = f'bin/{name}_LM-{learning_rate}.pt'
    torch.save(model.state_dict(), path)



# Plotting functions of the losses during training and validation
def plot_training_progress(sampled_epochs, losses_train, losses_dev, ppl_dev_values, filename='plot.png'):
    
    # Plot two graphs and save the result in a PNG file:
    # 1. Loss trend during training (blue) and validation (red).
    # 2. Perplexity (PPL) trend during validation (green).

    # Parameters:
    # - sampled_epochs: list of sampled epochs
    # - loss_train: list of loss values ​​during training
    # - loss_dev: list of loss values ​​during validation
    # - ppl_dev_values: list of PPL values ​​during validation

    fig, axes = plt.subplots(2, 1, figsize=(8, 10))
    
    # Primo grafico: Loss Function
    axes[0].plot(sampled_epochs, losses_train, linestyle='-', color='b', label='Training Loss')
    axes[0].plot(sampled_epochs, losses_dev, linestyle='-', color='r', label='Validation Loss')
    axes[0].set_xlabel('Epoche')  # Corretto
    axes[0].set_ylabel('Loss')    # Corretto
    axes[0].set_title('Loss Trend')  # Corretto
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # Secondo grafico: Perplexity
    axes[1].plot(sampled_epochs, ppl_dev_values, marker='s', linestyle='-', color='g', label='Validation PPL')
    axes[1].set_xlabel('Epoche')
    axes[1].set_ylabel('Perplexity (PPL)')
    axes[1].set_title('Perplexity Trend')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)

    # Creare il percorso completo del file
    plt.tight_layout()
    filepath = os.path.join('plots', filename)

    plt.savefig(filepath, dpi=300)




