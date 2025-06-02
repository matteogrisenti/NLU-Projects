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




class Lang():
    """
    A class to manage vocabularies (intent labels, slot labels, special tokens) for NLU tasks.

    Attributes:
        words_pad_token (str): The actual string used for padding input tokens.
        words_pad_token_id (int): Tokenizer ID for the padding token.
        slots_pad_token_id (int): Padding value used for slot labels.

        slot2id (dict): Mapping from slot names to IDs.
        id2slot (dict): Inverse mapping from slot IDs to names.
        intent2id (dict): Mapping from intent names to IDs.
        id2intent (dict): Inverse mapping from intent IDs to names.

        len_slots (int): Number of unique slot labels including padding.
        len_intents (int): Number of unique intents.
    """

    def __init__(self, intents, slots, tokenizer, slots_pad_token):
        """
        Initializes the language object by building mappings for intents and slots.

        Args:
            intents (list): List of all intent labels in the dataset.
            slots (list): List of all slot labels in the dataset.
            tokenizer (transformers.PreTrainedTokenizer): Tokenizer used for input encoding.
            slots_pad_token (int): Padding value for slot labels.
        """
        # Get the pad token and its ID from the tokenizer
        self.words_pad_token = tokenizer.pad_token 
        self.words_pad_token_id = tokenizer.pad_token_id
        self.slots_pad_token_id = slots_pad_token
        
        # Build mappings for slots and intents
        self.slot2id = self.lab2id(slots, pad=True)     # Add 'pad' as a special label
        self.intent2id = self.lab2id(intents)           # No pad needed for intent classification

        # Create reverse mappings for decoding
        self.id2slot = {v:k for k, v in self.slot2id.items()}
        self.id2intent = {v:k for k, v in self.intent2id.items()}

        # Store sizes of vocabularies
        self.len_slots = len(self.slot2id)
        self.len_intents = len(self.intent2id)

        # Save the vocabulary to a JSON file for future use
        self.save_json()

    def lab2id(self, elements, pad=False):
        """
        Converts a list of labels into a dictionary mapping from label to unique ID.

        Args:
            elements (list): List of labels (e.g., slots or intents).
            pad (bool): Whether to include a special 'pad' label at the beginning.

        Returns:
            dict: Mapping from label name to ID.
        """
        vocab = {}
        if pad:
            vocab['pad'] = self.slots_pad_token_id  # Use provided pad token ID
        for label in sorted(set(elements)):         # Sort for consistent order
            vocab[label] = len(vocab)
        return vocab

    def to_dict(self):
        """
        Serializes the Lang object to a dictionary for saving to JSON.

        Returns:
            dict: All attributes in a serializable format.
        """
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
        """
        Deserializes a dictionary back into a Lang object.

        Args:
            data (dict): Dictionary loaded from a JSON file.

        Returns:
            Lang: Reconstructed Lang object.
        """
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
        """
        Loads a Lang object from a JSON file.

        Args:
            json_path (str): Path to the saved lang JSON file.

        Returns:
            Lang: Loaded Lang object.

        Raises:
            FileNotFoundError: If the JSON file does not exist.
        """
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Language file not found at {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return cls.from_dict(data)

    def save_json(self, json_file="dataset/lang.json"):
        """
        Saves the current Lang object as a JSON file.

        Args:
            json_file (str): Path where the JSON should be saved.
        """
        os.makedirs(os.path.dirname(json_file), exist_ok=True)
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"\tVocab JSON saved to {json_file}")




def get_slots_intents_len(json_path="dataset/lang.json"):
    """
    Retrieves the number of unique slots and intents from the saved Lang JSON file.

    Useful for initializing model output heads.

    Args:
        json_path (str): Path to the lang.json file.

    Returns:
        tuple: (len_slots, len_intents)

    Raises:
        FileNotFoundError: If the JSON file does not exist.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Language file not found at {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data.get('len_slots'), data.get('len_intents')




def init_dataset(tokenizer, slots_pad_token):
    """
    Initializes the dataset by:
    - Splitting the original training set into train and dev sets
    - Building global vocabularies of intents and slots
    - Saving them into a JSON file for later use

    This ensures consistency across train, dev, and test sets.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer used for input encoding.
        slots_pad_token (int): Padding value for slot labels.
    """
    # Step 1: Split the original dataset into train and dev
    extract_dev_set()      

    # Step 2: Load raw datasets
    train_raw, dev_raw = get_train_dev_rawset()     
    test_raw = get_test_rawset()

    # Step 3: Merge all records to collect all possible slots and intents
    corpus = train_raw + dev_raw + test_raw                           
    slots = set(sum([line['slots'].split() for line in corpus],[]))    # Extract all unique slot labels from the corpus
    intents = set([line['intent'] for line in corpus])                 # Extract all unique intent labels from the corpus

    # Step 4: Create and save the Lang object
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




# Factory function to create a custom collate function with dynamic padding tokens
def collate_fn_factory(words_pad_token_id, slots_pad_token_id):
    """
    Returns a collate function that dynamically pads sequences in a batch.
    
    Args:
        words_pad_token_id (int): Token ID used to pad input word tokens.
        slots_pad_token_id (int): Token ID used to pad slot label tokens.

    Returns:
        collate_fn (function): A function that takes a batch of examples and returns a padded batch.
    """

    def collate_fn(batch):
        """
        Collates a batch of data by padding variable-length sequences.

        Args:
            batch (list of dicts): Each dict contains 'input_ids', 'attention_mask',
                                   'slot_labels', 'slot_label_mask', and 'intent_label'.

        Returns:
            dict: A dictionary containing padded tensors for each field in the batch.
        """
         
        # Extract lists of each component from the batch
        input_ids_list = [ex['input_ids'] for ex in batch]
        attention_list = [ex['attention_mask'] for ex in batch]
        slot_labels_list = [ex['slot_labels'] for ex in batch]
        slot_label_mask_list = [ex['slot_label_mask'] for ex in batch]  
        intent_labels = torch.stack([ex['intent_label'] for ex in batch])

        # Pad sequences to the length of the longest sequence in the batch
        input_ids_padded = pad_sequence(input_ids_list, batch_first=True, padding_value=words_pad_token_id)
        attention_padded = pad_sequence(attention_list, batch_first=True, padding_value=0)
        slot_labels_padded = pad_sequence(slot_labels_list, batch_first=True, padding_value=slots_pad_token_id)
        slot_label_mask_padded = pad_sequence(slot_label_mask_list, batch_first=True, padding_value=0)  # pad mask with 0

        # Return a dictionary of padded tensors
        return {
            'input_ids': input_ids_padded,
            'attention_mask': attention_padded,
            'slot_labels': slot_labels_padded,
            'slot_label_mask': slot_label_mask_padded,  
            'intent_labels': intent_labels,  
        }
    return collate_fn




def get_train_dev_dataloader(tokenizer, batch_size):
    """
    Creates DataLoader objects for training and development datasets.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to encode text.
        batch_size (int): Number of samples per batch.

    Returns:
        tuple: (train_loader, dev_loader) - PyTorch DataLoaders for train and dev sets.
    """

    # Load raw JSON data for training and validation
    train_raw, dev_raw = get_train_dev_rawset()     # load raw datasets from json files

    # Load language object to retrieve padding token IDs
    lang = Lang.load_from_file()
    words_pad = lang.words_pad_token_id
    slots_pad = lang.slots_pad_token_id

    # Create a collate function using the padding token IDs
    collate_fn = collate_fn_factory(words_pad, slots_pad)  

    # Create Dataset objects for train and dev
    train_dataset = AtisDataset(
        records=train_raw,
        tokenizer=tokenizer,
    )

    dev_dataset = AtisDataset(
        records = dev_raw,
        tokenizer = tokenizer,
    )

    # Create DataLoader objects
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=int(batch_size/2), collate_fn=collate_fn)

    print('\tTrain and Dev DataLoader initializated')
    
    return train_loader, dev_loader




def get_test_dataloader(tokenizer, batch_size):
    """
    Creates a DataLoader object for the test dataset.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to encode text.
        batch_size (int): Number of samples per batch.

    Returns:
        test_loader (DataLoader): PyTorch DataLoader for the test set.
    """

    # Load raw JSON data for testing
    test_raw = get_test_rawset()  

    # Load language object to retrieve padding token IDs
    lang = Lang.load_from_file()
    words_pad = lang.words_pad_token_id
    slots_pad = lang.slots_pad_token_id

    # Use the same collate function with dynamic padding tokens
    collate_fn = collate_fn_factory(words_pad, slots_pad)

    # Create Dataset object for test
    test_dataset = AtisDataset(
        records=test_raw,
        tokenizer=tokenizer,
    )

    # Create DataLoader object (no shuffling for test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

    print('\tTest DataLoader initialized')

    return test_loader