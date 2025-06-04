import torch

from model import LM_RNN, LM_LSTM, LM_LSTM_DO
from utils import read_file, init_lang, Lang, PennTreeBank
from functions import train_model, init_weights


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", DEVICE)


# Uncomment this line once to initialize dataset structure and vocab:
# It creates a dev set from the original train data and saves global slot/intent mappings
init_lang()


# -------------------------------------------- TRAINING ------------------------------------------------
#  HYPERPARAMETERS 
LABEL = 'ADAM'      # RNN, LSTM, ADAMW
BATCH_SIZE = 32     # Original 64
HID_SIZE = 200                            # Original 200
EMB_SIZE = 300                            # Original 300
N_LAYERS = 1                              # Original 1
DROPOUT = 0.2
LR = [0.1,0.01,0.001, 0.0001]
OPTIMIZER = 'AdamW'   # SGD or AdamW
CLIP = 5            # Clip the gradient -> avoid exploding gradients

lang = Lang.load_from_file()        
vocab_len = len(lang.word2id)           # Compute the Vocabular Len to understand the dimension of the Linear layer
pad_index = lang.word2id["<pad>"]       # Get the ID of the pad token 


for lr in LR:
    hyperparameters = {
        'label': LABEL,
        'batch_size': BATCH_SIZE, 
        'hid_size': HID_SIZE,
        'emb_size': EMB_SIZE,
        'n_layers': N_LAYERS,
        'dropout_emb': DROPOUT,
        'dropout_out': DROPOUT,
        'learning_rate': lr,
        'optimizer': OPTIMIZER, 
        'clip': 5
    }

    # Define the model to be trained
    # model = LM_RNN(hyperparameters['emb_size'], hyperparameters['hid_size'], vocab_len, 
    #                pad_index=pad_index ).to(DEVICE)
    # model = LM_LSTM(hyperparameters['emb_size'], hyperparameters['hid_size'], vocab_len,
    #                 pad_index=pad_index ).to(DEVICE)
    model = LM_LSTM_DO(hyperparameters['emb_size'], hyperparameters['hid_size'], vocab_len,
                       pad_index=pad_index, out_dropout=hyperparameters['dropout_out'], 
                       emb_dropout=hyperparameters['dropout_emb']).to(DEVICE)
    # Initializa the weight of the model
    model.apply(init_weights)
    
    # Train the model
    train_model( model, hyperparameters, DEVICE )

