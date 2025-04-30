import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np


class VariationalDropout(nn.Module):
    def __init__(self):
        super(VariationalDropout, self).__init__()

    def forward(self, x, dropout=0.5):
        # x is the sequence of embedding: (batch_size, seq_len, emb_size)

        # The dropout mask is used only during training
        if not self.training or dropout == 0:
            return x

        # Create the dropout mask
        dropout_mask = x.new_empty((x.size(0), 1, x.size(2))).bernoulli_(1 - dropout)
        # x.new_empty((x.size(0), 1, x.size(2)))    ->     Same size as x but with 1 in the second dimension
        # .bernoulli_(1 - dropout)                  ->     Create a mask with 1s and 0s, where 1s are kept and 0s are dropped out
        
        # NB: The second dimension is 1 because we want to apply the same dropout mask to all time steps in the sequence
        # We have a different dropout mask for each sample in the batch which is applied to all time steps in the sequence.
        
        # The mask is rescaled: we are dividiving for a number between 0 and 1 (1 - dropout). 
        # This allows to keep the mean of the output activation the same as the dropout will not be aplied. 
        # So we are put to zero some activations, but we are also scaling the rest of the activations to keep the mean the same.
        dropout_mask = dropout_mask.div_(1 - dropout)

        # The dropout mask is applied to the input tensor
        return x * dropout_mask



class LM_LSTM_WT(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, n_layers=1, dropout=0.5):
        super(LM_LSTM, self).__init__()

        # For weight tying, the embedding size must be equal to the hidden size
        assert emb_size == hidden_size, "Weight tying requires emb_size == hidden_size"

        # From Token to Embedding
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        
        # From Embedding to Hidden State
        self.rnn = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.pad_token = pad_index


        # From Hidden State to Output
        self.output = nn.Linear(hidden_size, output_size)

        # Weight tying: the output layer's weights are tied to the embedding layer's weights
        self.output.weight = self.embedding.weight
        
    def forward(self, input_sequence):
        # 1. Embedding of the Token Input: (seq_len, batch_size) -> (batch_size, seq_len, emb_size)
        emb = self.embedding(input_sequence)

        # 2. LSTM: (batch_size, seq_len, emb_size) -> (batch_size, seq_len, hidden_size)
        rnn_out, _  = self.rnn(emb)

        # 3. Decoding: (batch_size, seq_len, hidden_size) -> (batch_size, seq_len, output_size)
        output = self.output(rnn_out).permute(0,2,1)

        return output 



class LM_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, n_layers=1, dropout=0.5):
        super(LM_LSTM, self).__init__()

        # For weight tying, the embedding size must be equal to the hidden size
        assert emb_size == hidden_size, "Weight tying requires emb_size == hidden_size"

        # From Token to Embedding
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        
        # From Embedding to Hidden State
        self.rnn = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.pad_token = pad_index

        # Variational Dropout
        self.variational_dropout = VariationalDropout()
        self.dropout = dropout  

        # From Hidden State to Output
        self.output = nn.Linear(hidden_size, output_size)

        # Weight tying: the output layer's weights are tied to the embedding layer's weights
        self.output.weight = self.embedding.weight
        
    def forward(self, input_sequence):
        # 1. Embedding of the Token Input: (seq_len, batch_size) -> (batch_size, seq_len, emb_size)
        emb = self.embedding(input_sequence)

        # 2. Variational Dropout on the Embedding ( Input of the LSTM )
        emb = self.variational_dropout(emb, self.dropout)

        # 3. LSTM: (batch_size, seq_len, emb_size) -> (batch_size, seq_len, hidden_size)
        rnn_out, _  = self.rnn(emb)

        # 4. Variational Dropout on the Hidden ( Output of the LSTM )
        rnn_out = self.variational_dropout(rnn_out, self.dropout)

        # 5. Decoding: (batch_size, seq_len, hidden_size) -> (batch_size, seq_len, output_size)
        output = self.output(rnn_out).permute(0,2,1)

        return output 