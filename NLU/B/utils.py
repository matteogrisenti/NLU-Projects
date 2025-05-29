import json
import torch
import os

from collections import Counter
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
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
    print("\tLoading split datasets...")
    train_path = os.path.join('dataset', 'train_split.json')
    dev_path = os.path.join('dataset', 'dev_split.json')

    if not os.path.exists(train_path) or not os.path.exists(dev_path):
        raise FileNotFoundError("\tSplit files not found. Run extract_dev_set() first.")

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
    print("\tLoading split datasets...")
    test_path = os.path.join('dataset', 'test.json')
    
    if not os.path.exists(test_path):
        raise FileNotFoundError("Test files not found.")

    test_raw = load_data(test_path)

    print('\t - TEST size:', len(test_raw))

    return test_raw




def preprocess_raw(raw_data):
    '''
    Preprocess the raw data in order to get a form that will be used by the BERT Tokenizzer.
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


def get_slots_intents_lists():
    return None


class Lang():
    def __init__(self, intents, slots, tokenizer, slots_pad_token):
        self.words_pad_token = tokenizer.pad_token 
        self.words_pad_token_id = tokenizer.pad_token_id
        self.slots_pad_token_id = slots_pad_token
        
        # Build vocabularies
        self.slot2id = self.lab2id(slots, pad=True)
        self.intent2id = self.lab2id(intents)

        # Reverse mappings
        self.id2slot = {v:k for k, v in self.slot2id.items()}
        self.id2intent = {v:k for k, v in self.intent2id.items()}

        # Save also the number of slots and intents
        self.len_slots = len(self.slot2id)
        self.len_intents = len(self.intent2id)

        # Save the Lang in Json file
        self.save_json()

    def lab2id(self, elements, pad=False):
        vocab = {}
        if pad:
            vocab['pad'] = self.slots_pad_token_id  # Ensure consistent padding ID
        for label in sorted(set(elements)):     # Sort for consistency
            vocab[label] = len(vocab)
        return vocab

    def to_dict(self):
        return {
            'words_pad_token': self.words_pad_token,
            'words_pad_token_id': self.words_pad_token_id,
            'slots_pad_token_id': self.slots_pad_token_id,

            'intent2id': self.intent2id,
            'id2intent': self.id2intent,

            'slot2id': self.slot2id,
            'id2slot': self.id2slot,

            'len_slots': self.len_slots,
            'len_intents': self.len_intents
        }
    
    @classmethod
    def from_dict(cls, data):
        obj = cls.__new__(cls)  # create instance without calling __init__

        obj.words_pad_token = data.get('words_pad_token')
        obj.words_pad_token_id = data.get('words_pad_token_id')
        obj.slots_pad_token_id = data.get('slots_pad_token_id')

        obj.intent2id = data.get('intent2id')
        obj.id2intent = {int(k): v for k, v in data.get('id2intent', {}).items()}

        obj.slot2id = data.get('slot2id')
        obj.id2slot = {int(k): v for k, v in data.get('id2slot', {}).items()}

        obj.len_slots = data.get('len_slots') 
        obj.len_intents = data.get('len_intents')

        return obj

    @classmethod
    def load_from_file(cls, json_path="dataset/lang.json"):
        """Load Lang object directly from JSON file"""
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Language file not found at {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return cls.from_dict(data)

    def save_json(self, json_file="dataset/lang.json"):
        """Save vocabularies as JSON for later loading"""
        os.makedirs(os.path.dirname(json_file), exist_ok=True)
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"\tVocab JSON saved to {json_file}")



def get_slots_intents_len(json_path="dataset/lang.json"):
    """Returns the number of slots and intents from the lang.json file."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Language file not found at {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data.get('len_slots'), data.get('len_intents')




def init_dataset(tokenizer, slots_pad_token):
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

    lang = Lang(intents, slots, tokenizer, slots_pad_token )
    print('Dataset initialized correctly')




class AtisDataset(Dataset):
    """
    PyTorch Dataset for the ATIS task using a BERT tokenizer.
    Handles sub-token alignment for slot labels. Truncates if sequence > max_length.
    Padding is deferred to collate_fn for dynamic batching.
    """

    def __init__(self, records, tokenizer, max_length=50, label_all_tokens=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_all_tokens = label_all_tokens

        # Load lang object automatically from JSON
        self.lang = Lang.load_from_file()

        # Initialize lists to store processed data
        self.inputs = []
        self.attention_masks = []
        self.token_type_ids = []
        self.slot_labels = []
        self.slot_label_masks = []
        self.intent_labels = []

        self._preprocess(records)

    def _preprocess(self, records):
        for item in records:
            words = item['utterance'].split()
            slots = item['slots'].split()
            intent = item['intent']

            tokens = []
            slot_label_ids = []
            slot_mask = []

            for word, slot_label in zip(words, slots):
                word_tokens = self.tokenizer.tokenize(word)
                if not word_tokens:
                    word_tokens = [self.tokenizer.unk_token]

                tokens.extend(word_tokens)
                label_id = self.lang.slot2id.get(slot_label, self.lang.slot2id['pad'])

                # Assign label only to first subword if label_all_tokens=False
                if self.label_all_tokens:
                    slot_label_ids.extend([label_id] * len(word_tokens))
                    slot_mask.extend([1] * len(word_tokens))
                else:
                    slot_label_ids.extend([label_id] + [self.lang.slot2id['pad']] * (len(word_tokens) - 1))
                    slot_mask.extend([1] + [0] * (len(word_tokens) - 1))

            # Add special tokens
            tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
            slot_label_ids = [self.lang.slot2id['pad']] + slot_label_ids + [self.lang.slot2id['pad']]
            slot_mask = [0] + slot_mask + [0]

            # Convert tokens to IDs
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            # Truncate if needed
            if len(input_ids) > self.max_length:
                input_ids = input_ids[:self.max_length]
                attention_mask = [1] * self.max_length
                token_type_ids = [0] * self.max_length
                slot_label_ids = slot_label_ids[:self.max_length]
                slot_mask = slot_mask[:self.max_length]
            else:
                attention_mask = [1] * len(input_ids)
                token_type_ids = [0] * len(input_ids)

            # Append to dataset
            self.inputs.append(torch.tensor(input_ids, dtype=torch.long))
            self.attention_masks.append(torch.tensor(attention_mask, dtype=torch.long))
            self.token_type_ids.append(torch.tensor(token_type_ids, dtype=torch.long))
            self.slot_labels.append(torch.tensor(slot_label_ids, dtype=torch.long))
            self.slot_label_masks.append(torch.tensor(slot_mask, dtype=torch.bool))
            self.intent_labels.append(torch.tensor(self.lang.intent2id[intent], dtype=torch.long))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs[idx],
            'attention_mask': self.attention_masks[idx],
            'token_type_ids': self.token_type_ids[idx],
            'slot_labels': self.slot_labels[idx],
            'slot_label_mask': self.slot_label_masks[idx],
            'intent_label': self.intent_labels[idx]
        }


def collate_fn_factory(words_pad_token_id, slots_pad_token_id):
    def collate_fn(batch):
        """
        Custom collate function to pad sequences dynamically.
        """
        input_ids_list = [ex['input_ids'] for ex in batch]
        attention_list = [ex['attention_mask'] for ex in batch]
        slot_list_labels = [ex['slot_labels'] for ex in batch]
        intent_labels = torch.stack([ex['intent_label'] for ex in batch])

        # Pad sequences
        input_ids_padded = pad_sequence(input_ids_list, batch_first=True, padding_value=words_pad_token_id)
        attention_padded = pad_sequence(attention_list, batch_first=True, padding_value=0)
        slot_padded = pad_sequence(slot_list_labels, batch_first=True, padding_value=slots_pad_token_id)

        return {
            'input_ids': input_ids_padded,
            'attention_mask': attention_padded,
            'slot_labels': slot_padded,
            'intent_labels': intent_labels
        }
    return collate_fn



def get_train_dev_dataloader(tokenizer, batch_size):
    train_raw, dev_raw = get_train_dev_rawset()     # load raw datasets from json files

    lang = Lang.load_from_file()
    words_pad = lang.words_pad_token_id
    slots_pad = lang.slots_pad_token_id

    # Define the collate function with dinamic words and slots pad
    collate_fn = collate_fn_factory(words_pad, slots_pad)  

    train_dataset = AtisDataset(
        records=train_raw,
        tokenizer=tokenizer,
    )

    dev_dataset = AtisDataset(
        records = dev_raw,
        tokenizer = tokenizer,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=int(batch_size/2), collate_fn=collate_fn)
    print('\tTrain and Dev DataLoader initializated')
    
    return train_loader, dev_loader