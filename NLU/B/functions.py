import os
import sys
import csv
import json
import torch
import numpy as np
import torch.nn as nn
import scipy.stats as st
import torch.optim as optim

from tqdm import tqdm
from copy import deepcopy
from pprint import pformat
from conll import evaluate
from sklearn.metrics import classification_report



def init_weights(mat):
    """
    Applies custom weight initialization to all modules in the given model.

    Xavier uniform is used for input-to-hidden connections,
    orthogonal initialization for hidden-to-hidden connections in RNNs,
    and small uniform values for linear layers.
    """
    for m in mat.modules():     # Iterate over all modules in the model
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            # Special handling for RNN-based layers
            for name, param in m.named_parameters():
                # Input-to-hidden weights
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
                # Hidden-to-hidden weights
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
                # Bias terms
                elif 'bias' in name:
                    param.data.fill_(0)     # Set biases to zero
        else:
            if type(m) in [nn.Linear]:
                # Initialize linear layer weights with small uniform values
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)     # Initialize bias with small constant