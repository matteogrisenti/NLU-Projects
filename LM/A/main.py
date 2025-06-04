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
LABEL = 'ADAMW'      # RNN, LSTM, ADAMW
BATCH_SIZE = 128     # Original 64
HID_SIZE = 200                            # Original 200
EMB_SIZE = 300                            # Original 300
N_LAYERS = 1                              # Original 1
DROPOUT_EMB = None
DROPOUT_OUT = None
LR = [5,1,0.1]
OPTIMIZER = 'SGD'   # SGD or AdamW
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
        'dropout_emb': None,
        'dropout_out': None,
        'learning_rate': lr,
        'optimizer': OPTIMIZER, 
        'clip': 5
    }

    # Define the model to be trained
    model = LM_RNN(hyperparameters['emb_size'], hyperparameters['hid_size'], vocab_len, 
                    pad_index=pad_index ).to(DEVICE)
    # model = LM_LSTM(EMB_SIZE, HID_SIZE, vocab_len, pad_index=lang.word2id["<pad>"], out_dropout=DROPOUT_OUT, emb_dropout=DROPOUT_EMB).to(DEVICE)
    # model = LM_LSTM_DO(EMB_SIZE, HID_SIZE, vocab_len, pad_index=lang.word2id["<pad>"], out_dropout=DROPOUT_OUT, emb_dropout=DROPOUT_EMB, n_layers=N_LAYERS).to(DEVICE)
    
    # Initializa the weight of the model
    model.apply(init_weights)
    
    # Train the model
    train_model( model, hyperparameters, DEVICE )

