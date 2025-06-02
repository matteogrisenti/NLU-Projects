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


# Uncomment this line once to initialize dataset structure and vocab:
# It creates a dev set from the original train data and saves global slot/intent mappings
# init_dataset(tokenizer, SLOTS_PAD_TOKEN) 


# Trains the model using the provided hyperparameters and datasets.
def train(model, hyperparameters, name, criterion_slots, criterion_intents):
    # Initialize classification weights
    model.init_classification_heads()   

    # Get train and dev dataloaders
    train_loader, dev_loader = get_train_dev_dataloader(tokenizer, hyperparameters['batch_size'])

    # Train the model
    model = train_model(
        model, train_loader, dev_loader, tokenizer, 
        criterion_slots, criterion_intents, 
        N_EPOCHES, PATIENTE, CLIP, 
        model_name=name, 
        device=DEVICE, 
        hyperparameters=hyperparameters
    )

    # Plot all the training plots 
    plot_all(name)


# Evaluates the trained model on the test set.
def test(model, hyperparameters, name, criterion_slots, criterion_intents):
    # Get test dataloader
    test_loader = get_test_dataloader(tokenizer, hyperparameters['batch_size'])

    # Run evaluation
    results_test, intent_test =  test_model(
        model, test_loader, tokenizer, 
        criterion_slots, criterion_intents, 
        model_name=name, 
        device=DEVICE, 
        hyperparameters=hyperparameters
    )


# Retrieve the number of unique slot labels and intent labels
len_slot_list, len_intent_list = get_slots_intents_len()


# ----------------------------------------- TRAINING ------------------------------------------------------
''' 
BATCH_SIZES = [ 16, 32, 64 ]
DROPOUTS = [ 0.1, 0.15 ]
LEARNING_RATES = 0.00001


for do in DROPOUTS:
    for bs in BATCH_SIZES:
        # Define fixed hyperparameters for training
        hyperparameters = {
            'bert_type' : 'bert-large-uncased',
            'learning_rate': LEARNING_RATES, 
            'batch_size': bs, 
            'dropout' : do,
            'num_slots_label': len_slot_list,
            'num_intents_label': len_intent_list
        }

        # Generate a unique model name based on hyperparameters
        name = model_name(hyperparameters['bert_type'], hyperparameters['learning_rate'], 
                        hyperparameters['batch_size'], hyperparameters['dropout']) 

        # Instantiate the model
        model = BertIntentSlot(hyperparameters['bert_type'], hyperparameters['num_intents_label'],
                            hyperparameters['num_slots_label'], hyperparameters['dropout'])

        # Define loss functions
        criterion_slots = nn.CrossEntropyLoss(ignore_index=SLOTS_PAD_TOKEN)
        criterion_intents = nn.CrossEntropyLoss()

        train(model, hyperparameters, name, criterion_slots, criterion_intents)    
'''


# ------------------------------------------ TESTING ------------------------------------------------------
# Define fixed hyperparameters of the model to be tested
hyperparameters = {
    'bert_type' : 'bert-large-uncased',
    'learning_rate': 0.00001, 
    'batch_size': 64, 
    'dropout' : 0.15,
    'num_slots_label': len_slot_list,
    'num_intents_label': len_intent_list
}

# Retrive the model name based on hyperparameters
name = model_name(hyperparameters['bert_type'], hyperparameters['learning_rate'], 
                        hyperparameters['batch_size'], hyperparameters['dropout']) 

# Initialize the model structure to be tested ( the weight were updated with the data saved in the bin )
model = BertIntentSlot(hyperparameters['bert_type'], hyperparameters['num_intents_label'],
                    hyperparameters['num_slots_label'], hyperparameters['dropout'])

criterion_slots = nn.CrossEntropyLoss(ignore_index=SLOTS_PAD_TOKEN)
criterion_intents = nn.CrossEntropyLoss()

test(model, hyperparameters, name, criterion_slots, criterion_intents)