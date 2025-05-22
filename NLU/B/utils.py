import json
import torch
import os

from collections import Counter
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

# Load JSON dataset from file
def load_data(path):
    """
    Loads dataset from a JSON file.
    
    Args:
        path (str): Path to JSON file
        
    Returns:
        list: List of samples, each sample is a dictionary
    """
    dataset = []
    with open(path) as f:
        dataset = json.loads(f.read())
    return dataset




def extract_dev_set():
    """
    Loads original train set, extracts dev set via stratified split,
    and saves the split into separate JSON files.
    """
    print("Loading datasets...")
    tmp_train_raw = load_data(os.path.join('dataset', 'train.json'))
    print('\tTrain samples:', len(tmp_train_raw))

    portion = 0.10  # Use 10% of training as validation set

    intents = [x['intent'] for x in tmp_train_raw]
    count_y = Counter(intents)

    labels = []
    inputs = []
    mini_train = []

    for id_y, y in enumerate(intents):
        if count_y[y] > 1:
            inputs.append(tmp_train_raw[id_y])
            labels.append(y)
        else:
            mini_train.append(tmp_train_raw[id_y])  # Keep singleton intents in training

    X_train, X_dev, _, _ = train_test_split(inputs, labels, test_size=portion,
                                            random_state=42,
                                            shuffle=True,
                                            stratify=labels)

    X_train.extend(mini_train)

    os.makedirs('dataset', exist_ok=True)

    with open(os.path.join('dataset', 'train_split.json'), 'w', encoding='utf-8') as f:
        json.dump(X_train, f, ensure_ascii=False, indent=2)

    with open(os.path.join('dataset', 'dev_split.json'), 'w', encoding='utf-8') as f:
        json.dump(X_dev, f, ensure_ascii=False, indent=2)

    print("\nSaved split datasets to 'dataset/train_split.json' and 'dataset/dev_split.json'")




def get_train_dev_rawset():
    """
    Loads the already-saved train and dev sets from JSON files.

    Returns:
        tuple: train_raw, dev_raw
    """
    print("Loading split datasets...")
    train_path = os.path.join('dataset', 'train_split.json')
    dev_path = os.path.join('dataset', 'dev_split.json')

    if not os.path.exists(train_path) or not os.path.exists(dev_path):
        raise FileNotFoundError("Split files not found. Run extract_dev_set() first.")

    train_raw = load_data(train_path)
    dev_raw = load_data(dev_path)

    print('\t - TRAIN size:', len(train_raw))
    print('\t - DEV size:', len(dev_raw))

    return train_raw, dev_raw




def get_test_rawset():
    """
    Loads the already-saved test sets from JSON files.

    Returns:
        tuple: test_raw
    """
    print("Loading split datasets...")
    test_path = os.path.join('dataset', 'test.json')
    
    if not os.path.exists(test_path):
        raise FileNotFoundError("Test files not found.")

    test_raw = load_data(test_path)

    print('\t - TEST size:', len(test_raw))

    return test_raw




def preprocess_raw(raw_data):
    '''
    Preprocess the raw data in order to get a from that will be used by the BERT Tokenizzer.
    It simply split the utterance in an array of  words and the the slots in an array of slots
    '''
    return [
        {
            "words": ex["utterance"].split(),
            "slots": ex["slots"].split(),
            "intent_label": ex["intent"]
        }
        for ex in raw_data
    ]




def post_slots_intents_lists(slots, intents):
    """
    Save slot and intent label lists to a JSON file.

    Args:
        slots (list): List of slot label strings (e.g., ["O", "B-fromloc", ...])
        intents (list): List of intent label strings (e.g., ["flight", "hotel", ...])
    """
    data = {
        "slots": list(slots),     # convert set to list
        "intents": list(intents)  # convert set to list
    }

    json_file = "dataset/slot_intent_lists.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f"Vocab JSON saved to {json_file}")

def get_slots_intents_lists():
    """
    Load slot and intent label lists from a JSON file.

    Returns:
        tuple: (slots, intents), both as lists of strings
    """
    json_file = "dataset/slot_intent_lists.json"
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return list(data["slots"]), list(data["intents"])




def init_dataset():
    """
    Initializes the dataset:
    - Splits the original dataset into train/dev
    - Extracts global list of slots and intents
    - Saves them in a JSON file
    """
    # 1) Divide the original train set in train_split and dev_split
    extract_dev_set()      

    # 2) Define a global list of all intents and slots in all the sets. 
    #    This is done becouse we do not wat unk labels, however this depends on the research purpose
    train_raw, dev_raw = get_train_dev_rawset()     # load the raw data from the json files
    test_raw = get_test_rawset()

    corpus = train_raw + dev_raw + test_raw                           # merge all the set toghether 
    slots = set(sum([line['slots'].split() for line in corpus],[]))   # set of all the slots
    intents = set([line['intent'] for line in corpus])                # set of all the intents

    post_slots_intents_lists(slots, intents)        # save the list in a json file, so they are always reacable 
    print('Dataset initialized correctly')



class AtisDataset(Dataset):
    """
    PyTorch Dataset for ATIS using a non-Fast tokenizer.
    - Manually tokenizes each word and aligns slot labels.
    - Ignores sub-token labels unless label_all_tokens is True.
    """

    def __init__(self, records, tokenizer, slot_list, intent_list, max_length=50, label_all_tokens=False):
        self.records = records
        self.tokenizer = tokenizer
        self.slot_list = slot_list
        self.intent_list = intent_list
        self.max_length = max_length
        self.label_all_tokens = label_all_tokens

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        ex = self.records[idx]
        words = ex['words']
        slots = ex['slots']
        intent = self.intent_list.index(ex['intent_label'])

        # Manually tokenize words and align slot labels
        sub_tokens = []
        aligned_slot_labels = []

        for word, slot in zip(words, slots):
            word_pieces = self.tokenizer.tokenize(word)
            if not word_pieces:
                word_pieces = [self.tokenizer.unk_token]
            sub_tokens.extend(word_pieces)

            if self.label_all_tokens:
                aligned_slot_labels.extend([self.slot_list.index(slot)] * len(word_pieces))
            else:
                aligned_slot_labels.append(self.slot_list.index(slot))
                aligned_slot_labels.extend([-100] * (len(word_pieces) - 1))

        # Truncate if needed
        if len(sub_tokens) > self.max_length - 2:
            sub_tokens = sub_tokens[:self.max_length - 2]
            aligned_slot_labels = aligned_slot_labels[:self.max_length - 2]

        # Add special tokens
        sub_tokens = [self.tokenizer.cls_token] + sub_tokens + [self.tokenizer.sep_token]
        aligned_slot_labels = [-100] + aligned_slot_labels + [-100]

        # Convert to input IDs and attention mask
        input_ids = self.tokenizer.convert_tokens_to_ids(sub_tokens)
        attention_mask = [1] * len(input_ids)

        return {
            'word_pieces': sub_tokens,  
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'slot_labels': torch.tensor(aligned_slot_labels, dtype=torch.long),
            'intent_label': intent
        }



def collate_fn(batch, pad_token=0):
    """
    Custom collate_fn to pad a batch of examples and stack labels:
    - input_ids, attention_mask: padded to max length in batch
    - slot_labels: padded with -100 (ignored index)
    - intent_label: stacked into a tensor of shape (batch_size,)
    Returns a dict of batched tensors.
    """
    # Extract lists of tensors
    input_ids_list = [ex['input_ids'] for ex in batch]
    attention_list = [ex['attention_mask'] for ex in batch]
    slot_list_labels = [ex['slot_labels'] for ex in batch]
    intent_labels = torch.stack([ex['intent_label'] for ex in batch])

    # Pad sequences (batch_first=True)
    input_ids_padded = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token)
    attention_padded = pad_sequence(attention_list, batch_first=True, padding_value=0)
    slot_padded = pad_sequence(slot_list_labels, batch_first=True, padding_value=-100)

    # Return batch dict
    return {
        'input_ids': input_ids_padded,
        'attention_mask': attention_padded,
        'slot_labels': slot_padded,
        'intent_label': intent_labels
    }