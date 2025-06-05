import torch

from model import LM_RNN, LM_LSTM, LM_LSTM_DO
from utils import read_file, init_lang, Lang, PennTreeBank
from functions import train_model, init_weights, test_model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", DEVICE)


# Uncomment this line once to initialize dataset structure and vocab:
# It creates a dev set from the original train data and saves global slot/intent mappings
# init_lang()

lang = Lang.load_from_file()        
vocab_len = len(lang.word2id)           # Compute the Vocabular Len to understand the dimension of the Linear layer
pad_index = lang.word2id["<pad>"]       # Get the ID of the pad token 


''' 
# -------------------------------------------- TRAINING ------------------------------------------------
#  HYPERPARAMETERS 
LABEL = 'ADAM'      # RNN, LSTM, ADAMW
BATCH_SIZE = 128     # Original 64
HID_SIZE = 200                           # Original 200
EMB_SIZE = 300                           # Original 300
N_LAYERS = [2,3]                              # Original 1
DROPOUT = 0.3
LR = 0.001
OPTIMIZER = 'AdamW'   # SGD or AdamW
CLIP = 5            # Clip the gradient -> avoid exploding gradients


for nl in N_LAYERS:
    hyperparameters = {
        'label': LABEL,
        'batch_size': BATCH_SIZE, 
        'hid_size': HID_SIZE,
        'emb_size': EMB_SIZE,
        'n_layers': nl,
        'dropout_emb': DROPOUT,
        'dropout_out': DROPOUT,
        'learning_rate': LR,
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
'''

# -------------------------------------------- TESTING ------------------------------------------------
rnn = {                     # 2,RNN,1,200,300,1,128,None,None,SGD,36,164.44,139.29-194.92
    'label': 'RNN',
    'batch_size': 128, 
    'hid_size': 200,
    'emb_size': 300,
    'n_layers': 1,
    'dropout_emb': None,
    'dropout_out': None,
    'learning_rate': 1,
    'optimizer': 'SGD', 
    'clip': 5
}
model = LM_RNN(rnn['emb_size'], rnn['hid_size'], vocab_len, pad_index=pad_index )
test_model(model, rnn, DEVICE)

lstm = {                    # 13,LSTM,1,200,300,2,32,None,None,SGD,15,144.84,123.87-169.2
    'label': 'LSTM',
    'batch_size': 32, 
    'hid_size': 200,
    'emb_size': 300,
    'n_layers': 1,
    'dropout_emb': None,
    'dropout_out': None,
    'learning_rate': 2,
    'optimizer': 'SGD', 
    'clip': 5
}
model = LM_LSTM(lstm['emb_size'], lstm['hid_size'], vocab_len, pad_index=pad_index )
test_model(model, lstm, DEVICE)

dropout = {                    # 17,LSTM-DO,1,200,300,2,32,0.2,0.2,SGD,45,126.47,108.11-147.66
    'label': 'LSTM-DO',
    'batch_size': 32, 
    'hid_size': 200,
    'emb_size': 300,
    'n_layers': 1,
    'dropout_emb': 0.2,
    'dropout_out': 0.2,
    'learning_rate': 2,
    'optimizer': 'SGD', 
    'clip': 5
}
model = LM_LSTM_DO(dropout['emb_size'], dropout['hid_size'], vocab_len,
                    pad_index=pad_index, out_dropout=dropout['dropout_out'], 
                    emb_dropout=dropout['dropout_emb'])
test_model(model, dropout, DEVICE)

adam = {                        # 31,ADAM,1,600,900,0.0005,128,0.5,0.5,AdamW,40,114.96,95.65-143.14
    'label': 'ADAM',
    'batch_size': 128, 
    'hid_size': 600,
    'emb_size': 900,
    'n_layers': 1,
    'dropout_emb': 0.5,
    'dropout_out': 0.5,
    'learning_rate': 0.0005,
    'optimizer': 'AdamW', 
    'clip': 5
}
model = LM_LSTM_DO(adam['emb_size'], adam['hid_size'], vocab_len,
                    pad_index=pad_index, out_dropout=adam['dropout_out'], 
                    emb_dropout=adam['dropout_emb'])
test_model(model, adam, DEVICE)




