import torch

from utils import Lang, init_lang
from functions import train_model, init_weights
from model import LM_LSTM_VD, LM_LSTM_WT


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", DEVICE)


# Uncomment this line once to initialize dataset structure and vocab:
# It creates a dev set from the original train data and saves global slot/intent mappings
init_lang()

lang = Lang.load_from_file()        
vocab_len = len(lang.word2id)           # Compute the Vocabular Len to understand the dimension of the Linear layer
pad_index = lang.word2id["<pad>"]       # Get the ID of the pad token 



# -------------------------------------------- TRAINING ------------------------------------------------
#  HYPERPARAMETERS 
LABEL = 'NTAvSGD'       # WeightTying, VarDropout, NTAvSGD
BATCH_SIZE = 32                         
SIZE = [700]                  
N_LAYERS = 1                              
DROPOUT = [0.7]
LR = 3
OPTIMIZER = 'NTAvSGD'   # SGD or NTAvSGD
CLIP = 5            # Clip the gradient -> avoid exploding gradients


for sz in SIZE:
    for do in DROPOUT:
        hyperparameters = {
            'label': LABEL,
            'batch_size': BATCH_SIZE, 
            'hid_size': sz,
            'emb_size': sz,
            'n_layers': N_LAYERS,
            'dropout': do,
            'learning_rate': LR,
            'optimizer': OPTIMIZER, 
            'clip': 5
        }

        # Define the model to be trained
        # model = LM_LSTM_WT(hyperparameters['emb_size'], hyperparameters['hid_size'], vocab_len,
        #                    pad_index=pad_index).to(DEVICE)
        model = LM_LSTM_VD(hyperparameters['emb_size'], hyperparameters['hid_size'], vocab_len,
                           pad_index=pad_index, dropout = hyperparameters['dropout']).to(DEVICE)
        
        # Initializa the weight of the model
        model.apply(init_weights)
        
        # Train the model
        train_model( model, hyperparameters, DEVICE )
