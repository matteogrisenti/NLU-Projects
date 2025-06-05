import torch

from utils import Lang, init_lang
from functions import train_model, init_weights, test_model
from model import LM_LSTM_VD, LM_LSTM_WT


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", DEVICE)


# Uncomment this line once to initialize dataset structure and vocab:
# It creates a dev set from the original train data and saves global slot/intent mappings
init_lang()

lang = Lang.load_from_file()        
vocab_len = len(lang.word2id)           # Compute the Vocabular Len to understand the dimension of the Linear layer
pad_index = lang.word2id["<pad>"]       # Get the ID of the pad token 


'''
# -------------------------------------------- TRAINING ------------------------------------------------
#  HYPERPARAMETERS 
LABEL = 'NTAvSGD'       # WeightTying, VarDropout, NTAvSGD
BATCH_SIZE = 32                         
SIZE = 450                  
N_LAYERS = 2                             
DROPOUT = [0.5, 0.7]
LR = 3
OPTIMIZER = 'NTAvSGD'   # SGD or NTAvSGD
CLIP = 5            # Clip the gradient -> avoid exploding gradients


for do in DROPOUT:
    hyperparameters = {
        'label': LABEL,
        'batch_size': BATCH_SIZE, 
        'hid_size': SIZE,
        'emb_size': SIZE,
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
                        pad_index=pad_index, n_layers= hyperparameters['n_layers'], dropout = hyperparameters['dropout']).to(DEVICE)
    
    # Initializa the weight of the model
    model.apply(init_weights)
    
    # Train the model
    train_model( model, hyperparameters, DEVICE )
 '''


# -------------------------------------------- TESTING ------------------------------------------------
WeightTying = {                     # WeightTying,1,300,300,2,32,None,SGD,17,119.32,102.36-138.95
    'label': 'WeightTying',
    'batch_size': 32, 
    'hid_size': 300,
    'emb_size': 300,
    'n_layers': 1,
    'dropout': None,
    'learning_rate': 2,
    'optimizer': 'SGD', 
    'clip': 5
}
model = LM_LSTM_WT(WeightTying['emb_size'], WeightTying['hid_size'], vocab_len, pad_index=pad_index )
test_model(model, WeightTying, DEVICE)

VarDropout = {                    # 12,VarDropout,1,700,700,2,32,0.7,SGD,60,89.27,76.25-104.51
    'label': 'VarDropout',
    'batch_size': 32, 
    'hid_size': 700,
    'emb_size': 700,
    'n_layers': 1,
    'dropout': 0.7,
    'learning_rate': 2,
    'optimizer': 'SGD', 
    'clip': 5
}
model = LM_LSTM_VD(VarDropout['emb_size'], VarDropout['hid_size'], vocab_len,
                    pad_index=pad_index, dropout=VarDropout['dropout'] )
test_model(model, VarDropout, DEVICE)

NTAvSGD = {                    # 13,NTAvSGD,1,700,700,3,32,0.7,NTAvSGD,59,87.35,74.52-102.41
    'label': 'NTAvSGD',
    'batch_size': 32, 
    'hid_size': 700,
    'emb_size': 700,
    'n_layers': 1,
    'dropout': 0.7,
    'learning_rate': 3,
    'optimizer': 'NTAvSGD', 
    'clip': 5
}
model = LM_LSTM_VD(NTAvSGD['emb_size'], NTAvSGD['hid_size'], vocab_len,
                    pad_index=pad_index, dropout=NTAvSGD['dropout'] )
test_model(model, NTAvSGD, DEVICE)

