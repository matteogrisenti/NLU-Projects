import os
import torch.nn as nn

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # Used to report errors on CUDA side

from transformers import BertTokenizer, BertModel
from pprint import pprint

from functions import model_name, train_model
from model import BertIntentSlot
from utils import (
    init_dataset,
    get_slots_intents_lists_len,
    get_train_dev_dataloader,
)

PAD_TOKEN = -100
N_EPOCHES = 200
PATIENTE = 3
CLIP = 5
DEVICE = 'cuda:0'

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")  # Download the tokenizer

'''
# init_dataset()
Run only one time to extract dev set from the original train dataset. This ensure the same train/dev split
for all the models, and this create a more fixed environment for the experiments. This allows to focus only 
on the model and not on the data split. This function also extract a global list for slots and intents 
from all the set ( train, dev and test ) and save it in a json file. 
'''

len_slot_list, len_intent_list = get_slots_intents_lists_len()

hyperparameters = {
    'bert_type' : 'bert-base-uncased',
    'learnin_rate': 0.001, 
    'batch_size': 64, 
    'dropout' : 0.1,
    'num_slots_label': len_slot_list,
    'num_intents_label': len_intent_list
}

train_loader, dev_loader = get_train_dev_dataloader(tokenizer, hyperparameters['batch_size'], PAD_TOKEN)

name = model_name(hyperparameters['bert_type'], hyperparameters['learnin_rate'], 
                  hyperparameters['batch_size'], hyperparameters['dropout']) 

model = BertIntentSlot(hyperparameters['bert_type'], hyperparameters['num_intents_label'],
                       hyperparameters['num_slots_label'], hyperparameters['dropout'])
model.init_classification_heads()

criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
criterion_intents = nn.CrossEntropyLoss()

model = train_model(model, train_loader, dev_loader, tokenizer, criterion_slots, criterion_intents, 
                    N_EPOCHES, PATIENTE, CLIP, model_name=name, device=DEVICE, hyperparameters=hyperparameters)
