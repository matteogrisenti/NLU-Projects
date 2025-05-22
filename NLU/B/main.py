from transformers import BertTokenizer, BertModel
from pprint import pprint

from utils import (
    init_dataset,
    get_train_dev_rawset,
    preprocess_raw,
    get_test_rawset,
    get_slots_intents_lists,
    AtisDataset
)
    
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")  # Download the tokenizer

'''
Run only one time to extract dev set from the original train dataset. This ensure the same train/dev split
for all the models, and this create a more fixed environment for the experiments. This allows to focus only 
on the model and not on the data split. This function also extract a global list for slots and intents and save it in a json file. 
'''
# init_dataset()

train_raw, dev_raw = get_train_dev_rawset()     # load raw datasets from json files

train_records = preprocess_raw(train_raw)       # split the words in the utterance and the slots
dev_records = preprocess_raw(dev_raw)           # split the words in the utterance and the slots

slot_list, intent_list = get_slots_intents_lists()


inputs = tokenizer(["I saw a man with a telescope", "StarLord was here",  "I didn't"], return_tensors="pt", padding=True)
pprint(inputs)
for row in inputs['input_ids']:
    tokens = tokenizer.convert_ids_to_tokens(row.tolist())
    print(tokens)