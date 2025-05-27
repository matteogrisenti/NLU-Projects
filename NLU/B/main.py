import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from transformers import BertTokenizer, BertModel
from pprint import pprint

from utils import (
    init_dataset,
    get_train_dev_rawset,
    preprocess_raw,
    get_test_rawset,
    get_slots_intents_lists,
    AtisDataset,
    test_AtisDataset
)
    
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")  # Download the tokenizer

'''
# init_dataset()
Run only one time to extract dev set from the original train dataset. This ensure the same train/dev split
for all the models, and this create a more fixed environment for the experiments. This allows to focus only 
on the model and not on the data split. This function also extract a global list for slots and intents 
from all the set ( train, dev and test ) and save it in a json file. 
'''

train_raw, dev_raw = get_train_dev_rawset()     # load raw datasets from json files

train_records = preprocess_raw(train_raw)       # split the words in the utterance and the slots
dev_records = preprocess_raw(dev_raw)           # split the words in the utterance and the slots

# Load the lists of slots and intents from the json file
slot_list, intent_list = get_slots_intents_lists()

train_dataset = AtisDataset(
    records=train_records,
    tokenizer=tokenizer,
    slot_list=slot_list,
    intent_list=intent_list
)


'''
# test_AtisDataset(train_dataset, train_records)
Uning during the development of this project to test the AtisDataset and if it correclty menage
the sub-tokenisation. 
'''