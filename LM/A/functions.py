import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.stats as st

# ------------------------------------------------------------------------------
# Function: train_loop
#
# Description:
#     Executes one full pass (epoch) of training over the provided dataset. 
#     For each sample, the function performs:
#         - Forward pass through the model
#         - Loss computation
#         - Backward pass to compute gradients
#         - Gradient clipping to prevent exploding gradients
#         - Parameter update via the optimizer
#
# Parameters:
#     data (iterable): A dataset or dataloader providing training samples.
#                      Each sample should be a dictionary containing:
#                          - "source": input token IDs (tensor)
#                          - "target": target token IDs (tensor)
#                          - "number_tokens": number of tokens in the target sequence
#     optimizer (torch.optim.Optimizer): The optimizer used for updating model parameters.
#     criterion (callable): The loss function used to compute the training loss.
#     model (nn.Module): The model being trained.
#     clip (float, optional): Maximum allowed norm for gradients. Used to prevent
#                             gradient explosion (default: 5).
#
# Returns:
#     avg_loss (float): The average loss across all tokens in the dataset.
# ------------------------------------------------------------------------------
def train_loop(data, optimizer, criterion, model, clip=5):
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



# ------------------------------------------------------------------------------
# Function: eval_loop
#
# Description: Evaluates the performance of a trained language model on a given dataset
# (typically the dev set). Computes key evaluation metrics such as:
#     - Perplexity (PPL)
#     - Average loss
#
# Parameters:
#     data (iterable): A dataset or dataloader providing evaluation samples.
#                      Each sample should be a dictionary with the following keys:
#                          - "source": input token IDs (tensor)
#                          - "target": target token IDs (tensor)
#                          - "number_tokens": number of tokens in the target sequence
#     eval_criterion (callable): The loss function used for evaluation (e.g., nn.CrossEntropyLoss).
#     model (nn.Module): The trained model to be evaluated.
#
# Returns:
#     ppl (float): Perplexity over the dataset.
#     loss_to_return (float): Average loss normalized by the number of tokens.
# ------------------------------------------------------------------------------
def eval_loop(data, eval_criterion, model):
    
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



# ------------------------------------------------------------------------------
# Function: test_eval_loop
#
# Description: Evaluates the performance of a trained language model on a given dataset
# (typically the test set). Computes key evaluation metrics such as:
#     - Perplexity (PPL)
#     - Average loss
#     - Standard Error of the Mean (SEM) for both loss and perplexity
#     - 95% Confidence Intervals (CI) for both loss and perplexity
#
# Parameters:
#     data (iterable): A dataset or dataloader providing evaluation samples.
#                      Each sample should be a dictionary with the following keys:
#                          - "source": input token IDs (tensor)
#                          - "target": target token IDs (tensor)
#                          - "number_tokens": number of tokens in the target sequence
#     eval_criterion (callable): The loss function used for evaluation (e.g., nn.CrossEntropyLoss).
#     model (nn.Module): The trained model to be evaluated.
#
# Returns:
#     ppl (float): Perplexity over the dataset.
#     loss_to_return (float): Average loss normalized by the number of tokens.
#     sem_loss (float or None): Standard Error of the Mean for the normalized loss.
#     ci_loss (tuple or None): 95% Confidence Interval for the normalized loss.
#     sem_ppl (float or None): Standard Error of the Mean for the perplexity.
#     ci_ppl (tuple or None): 95% Confidence Interval for the perplexity.
# ------------------------------------------------------------------------------
def test_eval_loop(data, eval_criterion, model, test=False):
    
    model.eval()
    loss_to_return = []
    loss_array = []
    loss_array_norm = []
    number_of_tokens = []
    sem_loss = None
    sem_ppl = None
    ci_loss = None
    ci_ppl = None

    # softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            output = model(sample['source'])
            loss = eval_criterion(output, sample['target'])

            loss_array.append(loss.item())
            number_of_tokens.append(sample["number_tokens"])
            loss_array_norm.append(loss.item() / sample["number_tokens"])
            
    ppl = math.exp(sum(loss_array) / sum(number_of_tokens))
    loss_to_return = sum(loss_array) / sum(number_of_tokens)

    # Confidence interval for the loss
    losses = np.array(loss_array_norm)
    ppl_values = np.exp(losses)

    # SEM computation 
    sem_loss = st.sem(losses)  
    sem_ppl  = st.sem(ppl_values)
    # print('TEST SEM Loss:', sem_loss)
    # print('TEST SEM PPL:', sem_ppl)

    #CI computation
    ci_loss = st.t.interval(0.95, len(losses)-1, loc=np.mean(losses), scale=sem_loss)
    ci_ppl = (np.exp(ci_loss[0]), np.exp(ci_loss[1]))

    return ppl, loss_to_return, sem_loss, ci_loss, sem_ppl, ci_ppl



# ------------------------------------------------------------------------------
# Function: init_weights
#
# Description:
#     Initializes the weights of neural network modules with custom strategies. 
#     This function is typically called once at the beginning of training to ensure
#     stable and well-scaled weight initialization.
#     - For RNN-based layers (GRU, LSTM, RNN):
#         - Input-hidden weights (`weight_ih`) are initialized using Xavier Uniform
#         - Hidden-hidden weights (`weight_hh`) are initialized using Orthogonal initialization
#         - Biases are set to zero
#     - For Linear layers:
#         - Weights are initialized uniformly in the range [-0.01, 0.01]
#         - Biases are initialized to 0.01
#
# Parameters:
#     mat (nn.Module): The model or submodule whose parameters will be initialized.
#                      This function recursively applies initialization to all supported
#                      submodules within `mat`.
# ------------------------------------------------------------------------------
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



# ------------------------------------------------------------------------------
# Function: path_define
#
# Description:
#     Constructs a descriptive and unique string identifier for saving model-relatedmartifacts. 
#     The identifier includes key hyperparameters to ensure easy traceability and reproducibility of experiments.
#
# Parameters:
#     LABEL (str): Identifier for the model or experiment.
#     LR (float): Learning rate used in training.
#     HID_SIZE (int): Size of the hidden layers in the model.
#     EMB_SIZE (int): Size of the embedding vectors.
#     DROPOUT_EMB (float): Dropout rate applied to the embedding layer.
#     DROPOUT_OUT (float): Dropout rate applied to the output layer.
#     OPTIMIZER (str): Optimizer used for training (e.g., 'Adam', 'SGD').
#
# Returns:
#     path (str): A formatted string with all the hyperparameters embedded,
#                 suitable for use in filenames or directory paths.
# ------------------------------------------------------------------------------
def path_define(LABEL, LR, HID_SIZE, EMB_SIZE, DROPOUT_EMB, DROPOUT_OUT, OPTIMIZER):
    path = f"{LABEL}_lr-{str(LR).replace('.', ',')}_hid-{HID_SIZE}_emb-{EMB_SIZE}"
    path += f"_dropEmb-{str(DROPOUT_EMB).replace('.', ',')}_dropOut-{str(DROPOUT_OUT).replace('.', ',')}"
    path += f"_{OPTIMIZER}"
    return path



# ------------------------------------------------------------------------------
# Function: save_model
#
# Description:
#     Saves the state dictionary (model weights) of the provided model to a file
#     at the specified path. The file is saved in the 'bin' directory with a 
#     ".pt" extension. 
#
# Parameters:
#     model (nn.Module): The model whose state dictionary is to be saved.
#     path (str): The path that encode the hyperparameters of the model.
# ------------------------------------------------------------------------------
def save_model(model, path):
    path = f'bin/' + path + f'.pt'
    torch.save(model.state_dict(), path)



# ------------------------------------------------------------------------------
# Function: plot_training_progress
#
# Description:
#     Generates and saves two plots showing the evolution of training metrics:
#       1. Training and validation loss over sampled epochs
#       2. Validation perplexity (PPL) over sampled epochs
#
#     The plots are saved as a PNG file in the "plots/" directory. The filename
#     includes key hyperparameters (e.g., learning rate, dropout rates) for easy
#     identification of the experiment.
#
# Parameters:
#     sampled_epochs (list[int]): Epoch indices at which metrics were sampled.
#     losses_train (list[float]): Training loss values corresponding to the sampled epochs.
#     losses_dev (list[float]): Validation loss values corresponding to the sampled epochs.
#     ppl_dev_values (list[float]): Perplexity values on the validation set.
#
# Behavior:
#     - Fixed X-axis (epochs: 0–100) and Y-axis limits for consistency.
#     - Dynamically scales Y-axis for PPL if necessary.
#     - Output file is saved at: plots/{filename}_lr-{lr}_emb_{dropout_emb}_out_{dropout_out}.png
# ------------------------------------------------------------------------------
def plot_training_progress(sampled_epochs, losses_train, losses_dev, ppl_dev_values, path='PLOT'):
    
    fig, axes = plt.subplots(2, 1, figsize=(8, 10))
    
    # Primo grafico: Loss Function
    axes[0].plot(sampled_epochs, losses_train, linestyle='-', color='b', label='Training Loss')
    axes[0].plot(sampled_epochs, losses_dev, linestyle='-', color='r', label='Validation Loss')
    axes[0].set_xlabel('Epoche')  
    axes[0].set_ylabel('Loss')   
    axes[0].set_title('Loss Trend')  
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)

    axes[0].set_xlim(0, 100)  # Limiti fissi per l'asse X
    axes[0].set_ylim(1, 9)    # Limiti fissi per l'asse Y

    # Secondo grafico: Perplexity
    axes[1].plot(sampled_epochs, ppl_dev_values, marker='s', linestyle='-', color='g', label='Validation PPL')
    axes[1].set_xlabel('Epoche')
    axes[1].set_ylabel('Perplexity (PPL)')
    axes[1].set_title('Perplexity Trend')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)

    axes[1].set_xlim(0, 100)  # Limiti fissi per l'asse X
    y_max = 400
    if(max(ppl_dev_values) > 500): y_max = max(ppl_dev_values) 
    axes[1].set_ylim(50, y_max)    # Limiti fissi per l'asse Y

    filepath = os.path.join('plots', path + '.png')
    plt.savefig(filepath, dpi=300)



# ------------------------------------------------------------------------------
# Function: get_last_experiment_id
#
# Description:
#     Retrieves the last experiment ID recorded in the `experiments.csv` file.
#
# Behavior:
#     - Automatically initializes the file with a header if it does not exist.
#     - Handles empty or malformed files gracefully.
#
# Returns:
#     int: The last experiment ID found in the CSV file, or 0 if no experiments exist.
# ------------------------------------------------------------------------------
def get_last_experiment_id(filename):

    # If the file does not exist, create it and write the header
    if not os.path.isfile(filename):
        with open(filename, 'w') as f:
            f.write('ID,Network Type,Learning Rate,Hidden Size,Embedding Size,Dropout Emb,Dropout Out,Optimizer,Epochs,PPL Test,Norm Loss Test,SEM Loss,CI Norm Loss Test,SEM PPL,CI PPL Test\n')
        return 0  # First experiment

    # Read existing file
    with open(filename, 'r') as f:
        lines = f.readlines()
        if len(lines) <= 1:
            return 0  # File exists but only header is present

        last_line = lines[-1].strip().split(',')

        try:
            return int(last_line[0])
        except ValueError:
            return 0  # If parsing fails, default to 0



# ------------------------------------------------------------------------------
# Function: save_experiment_results
#
# Description:
#     Appends a new row to the `experiments.csv` file, logging key details and
#     evaluation metrics from a trained model experiment. This includes model
#     configuration, optimizer, number of training epochs, and test set performance
#     such as perplexity, normalized loss, standard error, and confidence intervals.
#
# Parameters:
#     network_type (str): The type/name of the network architecture used.
#     lr (float): Learning rate used during training.
#     hidden_size (int): Size of the hidden layers in the model.
#     emb_size (int): Dimensionality of the embedding layer.
#     dropout_emb (float): Dropout rate applied to the embedding layer.
#     dropout_out (float): Dropout rate applied to the output layer.
#     optimizer (str): Optimizer used (e.g., 'Adam', 'SGD').
#     epoche (int): Number of epochs the model was trained.
#     test_ppl (float): Perplexity on the test set.
#     lest_loss_norm (float): Normalized test loss.
#     sem_loss (float): Standard Error of the Mean (SEM) for test loss.
#     ci_loss (tuple): 95% Confidence Interval for test loss.
#     sem_ppl (float): Standard Error of the Mean (SEM) for test perplexity.
#     ci_ppl (tuple): 95% Confidence Interval for test perplexity.
#
# Behavior:
#     - Automatically retrieves the last experiment ID and increments it.
#     - Creates the CSV file with a header if it does not exist.
#     - Appends all values (rounded to 2 decimals) to the file.
#
# Output:
#     A new line is added to 'experiments.csv' recording the current experiment.
# ------------------------------------------------------------------------------
def save_experiment_results(network_type, lr, hidden_size, emb_size, dropout_emb, dropout_out, 
                            optimizer, epoche, test_ppl, lest_loss_norm, sem_loss, ci_loss, 
                            sem_ppl, ci_ppl):
    filename = 'experiments.csv'

    experiment_id = get_last_experiment_id(filename) + 1    # Leggi l'ultimo ID
    file_exists = os.path.isfile(filename)                  # Check if the file exists  

    # If the file does not exist, create it and write the header
    with open(filename, 'a') as f:
        f.write(f'{experiment_id},{network_type},{lr},{hidden_size},{emb_size},{dropout_emb},{dropout_out},{optimizer},{epoche},{round(test_ppl, 2)},{round(lest_loss_norm, 2)},{round(sem_loss, 2)},{round(ci_loss[0], 2)}-{round(ci_loss[1], 2)},{round(sem_ppl, 2)},{round(ci_ppl[0], 2)}-{round(ci_ppl[1], 2)}\n')
