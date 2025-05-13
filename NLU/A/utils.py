import os
import json
import torch
import torch.utils.data as data

from sklearn.model_selection import train_test_split
from collections import Counter
from pprint import pformat
from torch.utils.data import DataLoader


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


# Split dataset into train, dev, test
def get_train_dev_test_set(): 
    """
    Loads original train/test sets, extracts dev set via stratified split,
    and returns cleaned train/dev/test datasets.

    Returns:
        tuple: train_raw, dev_raw, test_raw
    """
    print("Loading datasets...")
    tmp_train_raw = load_data(os.path.join('dataset','train.json'))
    test_raw = load_data(os.path.join('dataset','test.json'))

    print('\tTrain samples:', len(tmp_train_raw))
    print('\tTest samples:', len(test_raw))

    portion = 0.10  # Use 10% of training as validation set

    intents = [x['intent'] for x in tmp_train_raw]  # stratify on intents
    count_y = Counter(intents)

    labels = []
    inputs = []
    mini_train = []

    # Separate examples by intent frequency
    for id_y, y in enumerate(intents):
        if count_y[y] > 1:   # If some intents occurs only once, we put them in training
            inputs.append(tmp_train_raw[id_y])
            labels.append(y)
        else:
            mini_train.append(tmp_train_raw[id_y])   # Keep singleton intents in training

    # Stratify and split
    X_train, X_dev, y_train, y_dev = train_test_split(inputs, labels, test_size=portion, 
                                                      random_state=42, 
                                                      shuffle=True,
                                                      stratify=labels)
    X_train.extend(mini_train)
    train_raw = X_train
    dev_raw = X_dev

    y_test = [x['intent'] for x in test_raw]


    ''' Intent counts
    print('Train Intent Counts:')
    pprint(dict(sorted(Counter(y_train).items())))
    print('Dev Intent Counts:')
    pprint(dict(sorted(Counter(y_dev).items())))
    print('Test Intent Counts:')
    pprint(dict(sorted(Counter(y_test).items())))
    print('='*89)
    '''

    ''' Intent distributions
    print('Train:')
    pprint({k:round(v/len(y_train),3)*100 for k, v in sorted(Counter(y_train).items())})
    print('Dev:'), 
    pprint({k:round(v/len(y_dev),3)*100 for k, v in sorted(Counter(y_dev).items())})
    print('Test:') 
    pprint({k:round(v/len(y_test),3)*100 for k, v in sorted(Counter(y_test).items())})
    print('='*89)
    '''


    print('\n\tTRAIN size:', len(train_raw))
    print('\tDEV size:', len(dev_raw))
    print('\tTEST size:', len(test_raw), '\n')

    return train_raw, dev_raw, test_raw




#--------------------- DICTIONARY Word/Slot/Intent to ID and ID to Word/Slot/Intent ------------------------
FILE_PATH = 'vocabularies.txt'

# Language class to map words, slots and intents to IDs
class Lang():
    def __init__(self, words, intents, slots, PAD_TOKEN=0, cutoff=0):
        """
        Initializes vocabulary mappings from lists of words, intents and slots.
        
        Args:
            words (list): List of words from utterances
            intents (list): List of intent labels
            slots (list): List of slot tags
            cutoff (int): Minimum word frequency to include
        """
        self.PAD_TOKEN = PAD_TOKEN
        self.word2id = self.w2id(words, cutoff=cutoff, unk=True)
        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents, pad=False)
        self.id2word = {v:k for k, v in self.word2id.items()}
        self.id2slot = {v:k for k, v in self.slot2id.items()}
        self.id2intent = {v:k for k, v in self.intent2id.items()}

        self.save_vocab()
        
    def w2id(self, elements, cutoff=None, unk=True):
        """Map words to IDs"""
        vocab = {'pad': self.PAD_TOKEN}
        if unk:
            vocab['unk'] = len(vocab)
        count = Counter(elements)
        for k, v in count.items():
            if v > cutoff:
                vocab[k] = len(vocab)
        return vocab
    
    def lab2id(self, elements, pad=True):
        """Map labels (slots or intents) to IDs"""
        vocab = {}
        if pad:
            vocab['pad'] = self.PAD_TOKEN
        for elem in elements:
                vocab[elem] = len(vocab)
        return vocab
    
    def save_vocab(self):
        """Save vocabularies to file"""
        with open(FILE_PATH, 'w', encoding='utf-8') as f:

            def write_section(title, data):
                f.write(f"\n{title}\n")
                f.write("-" * len(title) + "\n")
                f.write(pformat(data, indent=2, width=100) + "\n\n")

            write_section("Word Vocabulary:", self.word2id)
            write_section("Intent Vocabulary:", self.intent2id)
            write_section("Slot Vocabulary:", self.slot2id)
            write_section("Sorted Word Vocabulary (by ID):", sorted(self.word2id.items(), key=lambda x: x[1]))
            write_section("Sorted Intent Vocabulary (by ID):", sorted(self.intent2id.items(), key=lambda x: x[1]))
            write_section("Sorted Slot Vocabulary (by ID):", sorted(self.slot2id.items(), key=lambda x: x[1]))

        print(f"\tVocab saved to {FILE_PATH}")




#---------------------------------------------- TOURC DATASET ------------------------------------------------
# Dataset class for NLU task
class IntentsAndSlots (data.Dataset):
    def __init__(self, dataset, lang, unk='unk'):
        """
        Custom Dataset for Intent Detection and Slot Filling.

        Args:
            dataset (list): List of samples
            lang (Lang): Vocabulary object
            unk (str): Unknown token
        """
        self.utterances = []    # array of all the utterances
        self.intents = []       # array of all the intents
        self.slots = []         # array of all the slots
        self.unk = unk
        
        # Populate the arrays with the sample of the dataset
        for x in dataset:
            self.utterances.append(x['utterance'])
            self.slots.append(x['slots'])
            self.intents.append(x['intent'])

        # create a copy of the utterances, intents and slots arrray with their dictionary ids
        self.utt_ids = self.mapping_seq(self.utterances, lang.word2id)
        self.slot_ids = self.mapping_seq(self.slots, lang.slot2id)
        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        '''
        Given an ID it return the correlated Sample where the utterance, slot and intent are already 
        traduced in their id (dictionary)
        '''
        utt = torch.Tensor(self.utt_ids[idx])
        slots = torch.Tensor(self.slot_ids[idx])
        intent = self.intent_ids[idx]
        sample = {'utterance': utt, 'slots': slots, 'intent': intent}
        return sample
    
    # Auxiliary methods
    def mapping_lab(self, data, mapper):
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]
    
    def mapping_seq(self, data, mapper): # Map sequences to number
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq.split():
                if x in mapper:
                    tmp_seq.append(mapper[x])
                else:
                    tmp_seq.append(mapper[self.unk])
            res.append(tmp_seq)
        return res
    


    
# Collate function for padding sequences
def collate_fn(data, PAD_TOKEN = 0, device='cpu'):
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape 
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(PAD_TOKEN)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq # We copy each sequence into the matrix
        # print(padded_seqs)
        padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
        return padded_seqs, lengths
    
    data.sort(key=lambda x: len(x['utterance']), reverse=True)   # Sort data by seq lengths
    new_item = {}

    # Group the data for the keys: utterance, slots and intent
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]
        
    # We just need one length for packed pad seq, since len(utt) == len(slots)
    src_utt, _ = merge(new_item['utterance'])
    y_slots, y_lengths = merge(new_item["slots"])
    intent = torch.LongTensor(new_item["intent"])
    
    src_utt = src_utt.to(device) # We load the Tensor on our selected device
    y_slots = y_slots.to(device)
    intent = intent.to(device)
    y_lengths = torch.LongTensor(y_lengths).to(device)
    
    new_item["utterances"] = src_utt
    new_item["intents"] = intent
    new_item["y_slots"] = y_slots
    new_item["slots_len"] = y_lengths

    return new_item




# Main function to initialize everything
def init_dataloader(batch_size = 128, PAD_TOKEN=0, device='cpu'):
    """
    Initialize train, dev, test dataloaders and language model.

    Args:
        device (str): Device to use ('cuda' or 'cpu')

    Returns:
        tuple: train_loader, dev_loader, test_loader, lang
    """
    # Step 1: Get train/dev/test splits
    train_raw, dev_raw, test_raw = get_train_dev_test_set()

    # Step 2: Build vocab from train + dev + test
    print("\tBuilding vocabulary...")
    words = sum([x['utterance'].split() for x in train_raw], [])    # No set() since we want to compute the cutoff
    corpus = train_raw + dev_raw + test_raw     # We do not wat unk labels, however this depends on the research purpose
    intents = [x['intent'] for x in corpus]
    slots = sum([x['slots'].split() for x in corpus], [])

    lang = Lang(words, intents, slots, PAD_TOKEN, cutoff=0)    # Create the Dictionaries

    # Step 3: Create dataset objects
    train_dataset = IntentsAndSlots(train_raw, lang)
    dev_dataset = IntentsAndSlots(dev_raw, lang)
    test_dataset = IntentsAndSlots(test_raw, lang)

    # Step 4: Create dataloaders
    print("\tCreating dataloaders...")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=lambda x: collate_fn(x, PAD_TOKEN, device), shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=(batch_size/2), collate_fn=lambda x: collate_fn(x, PAD_TOKEN, device))
    test_loader = DataLoader(test_dataset, batch_size=(batch_size/2), collate_fn=lambda x: collate_fn(x, PAD_TOKEN, device))

    print("Dataloaders initialized!")
    return train_loader, dev_loader, test_loader, lang