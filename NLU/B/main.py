import os
import torch.nn as nn

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # Used to report errors on CUDA side

from transformers import BertTokenizer, BertModel
from pprint import pprint

from functions import model_name, train_model, test_model
from model import BertIntentSlot
from utils import init_dataset, get_slots_intents_len, get_train_dev_dataloader, get_test_dataloader
from plot import plot_all


SLOTS_PAD_TOKEN = -100
N_EPOCHES = 40
PATIENTE = 3
CLIP = 5
DEVICE = 'cuda:0'

# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")  # Download the tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")  # Download the tokenizer

'''
init_dataset(tokenizer, SLOTS_PAD_TOKEN)
# Run only one time to extract dev set from the original train dataset. This ensure the same train/dev split
# for all the models, and this create a more fixed environment for the experiments. This allows to focus only 
# on the model and not on the data split. This function also extract a global list for slots and intents 
# from all the set ( train, dev and test ) and save it in a json file. 
'''


def train(model, hyperparameters, name):
    model.init_classification_heads()

    train_loader, dev_loader = get_train_dev_dataloader(tokenizer, hyperparameters['batch_size'])

    model = train_model(model, train_loader, dev_loader, tokenizer, criterion_slots, criterion_intents, 
                        N_EPOCHES, PATIENTE, CLIP, model_name=name, device=DEVICE, hyperparameters=hyperparameters)

    plot_all(name)

def test(model, hyperparameters):
    test_loader = get_test_dataloader(tokenizer, hyperparameters['batch_size'])

    results_test, intent_test = test_model(model, test_loader, tokenizer, criterion_slots, criterion_intents, 
                                        model_name=name, device=DEVICE, hyperparameters=hyperparameters)

len_slot_list, len_intent_list = get_slots_intents_len()



BATCH_SIZES = 64 #[32, 64, 128]
DROPOUTS = 0.1  #[0.05, 0.3, 0.5]
LEARNING_RATES = [0.0005, 0.0001, 0.00005, 0.00001]

for lr in LEARNING_RATES:
    hyperparameters = {
        'bert_type' : 'bert-large-uncased',
        'learning_rate': lr, 
        'batch_size': BATCH_SIZES, 
        'dropout' : DROPOUTS,
        'num_slots_label': len_slot_list,
        'num_intents_label': len_intent_list
    }

    name = model_name(hyperparameters['bert_type'], hyperparameters['learning_rate'], 
                    hyperparameters['batch_size'], hyperparameters['dropout']) 

    model = BertIntentSlot(hyperparameters['bert_type'], hyperparameters['num_intents_label'],
                        hyperparameters['num_slots_label'], hyperparameters['dropout'])

    criterion_slots = nn.CrossEntropyLoss(ignore_index=SLOTS_PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss()

    train(model, hyperparameters, name)
    # test(model, hyperparameters)