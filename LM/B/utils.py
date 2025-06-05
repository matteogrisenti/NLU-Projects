import os
import json
import torch
import torch.utils.data as data

# LOADING CORPUS: organize each text as a list of sentence
# and add an end of sentence token <eos> to each sentence
def read_file(path, eos_token="<eos>"):
    output = []                                             # List of sentences

    with open(path, "r") as f:                              # Open the file in read mode
        for line in f.readlines():                          # Read each line in the file
            output.append(line.strip() + " " + eos_token)   # Add the end of sentence token to each line
    
    return output




# LANG: This class computes and stores our vocab: Word to ids and ids to word
# NB: the all process is word -> id ( made by LANG ) ans id -> embedding vector ( made by the model )
class Lang():

    def __init__(self, corpus, special_tokens=[]):
        self.word2id = self.get_vocab(corpus, special_tokens)
        self.id2word = {v:k for k, v in self.word2id.items()}
        
        self.save_json()
    
    # GET VOCABULARY: create a dictionary that maps each word to an index
    def get_vocab(self, corpus, special_tokens=[]):
        output = {}                     # Dictionary to store the mapping of words to ids
        i = 0                           # Counter for the ids
       
        for st in special_tokens:        # Add special tokens to the mapping
            output[st] = i               # Add the special token to the mapping
            i += 1
       
        for sentence in corpus:             # For each sentence in the corpus
            for w in sentence.split():      # For each word in the sentence
                if w not in output:
                    output[w] = i
                    i += 1
                    
        return output
    
    # Serializes the Lang object to a dictionary for saving to JSON.
    def to_dict(self):
        return {
            'word2id': self.word2id,
            'id2word': self.id2word,
        }
    
    # Save the Lang in a json file
    def save_json(self):
        json_file="dataset/lang.json"
        os.makedirs(os.path.dirname(json_file), exist_ok=True)
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"\tVocab JSON saved to {json_file}")
    
    # Load a Lang object from the json file
    @classmethod
    def load_from_file(cls):
        json_path="dataset/lang.json"
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Language file not found at {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        obj = cls.__new__(cls)      # create instance without calling __init__
        obj.word2id = data.get('word2id')
        obj.id2word = data.get('id2word')
        return obj
    



# This function create a Lang object and save it in a json file 
def init_lang():
    train_raw = read_file("dataset/PennTreeBank/ptb.train.txt")
    lang = Lang(train_raw, ["<pad>", "<eos>"])
    



# PENNTREEBANK: This class is a dataset that will be used to train the model. 
class PennTreeBank (data.Dataset):
    # Mandatory methods are __init__, __len__ and __getitem__

    def __init__(self, corpus, lang):
        # corpus: list of sentences
        # lang: vocabulary mapping words to ids

        self.source = []    # list of list of all token in a sentence exept for the last 
        self.target = []    # list of list of all token in a sentence exept for the first
        
        for sentence in corpus:
            self.source.append(sentence.split()[0:-1])  # We get from the first token till the second-last token
            self.target.append(sentence.split()[1:])    # We get from the second token till the last token
        
        # Convert sentences to ids using the mapping computed in Lang class
        self.source_ids = self.mapping_seq(self.source, lang)
        self.target_ids = self.mapping_seq(self.target, lang)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        # Returns 1 sample of the dataset, which is a dictionary with:
        # - "source": a sequence of IDs (torch.LongTensor) representing the input.
        # - "target": a sequence of IDs (torch.LongTensor) representing the output.

        src = torch.LongTensor(self.source_ids[idx])
        trg = torch.LongTensor(self.target_ids[idx])

        sample = {'source': src, 'target': trg}
        return sample
    
    # Auxiliary methods
    def mapping_seq(self, data, lang):
        # Map sequences of tokens to corresponding computed in Lang class
        
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq:
                if x in lang.word2id:
                    tmp_seq.append(lang.word2id[x])
                else:
                    print('OOV found!')
                    print('You have to deal with that') 
                    break
            res.append(tmp_seq)
        return res
    



# COLLATE FUNCTION: This function is used to pad the sequences in a batch to the same length.
def collate_fn(data, pad_token, device):

    def merge(sequences):   # This function pads the sequences to the same length
        lengths = [len(seq) for seq in sequences]           # Get the lengths of each sequence
        max_len = 1 if max(lengths)==0 else max(lengths)

        # Create padded_seqs: a matrix of size (number of sequences, max_len) to store the padded sequences
        
        # 1) Fill the sequences with the pad token
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(pad_token)

        # 2) Copy the sequences into the matrix
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq # We copy each sequence into the matrix

        padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
        return padded_seqs, lengths
    
    # Sort data by seq lengths
    # By sorting in descending order (longest to shortest), you can do padding more efficiently.
    data.sort(key=lambda x: len(x["source"]), reverse=True) 
    
    new_item = {}   # Create a new dictionary to store the padded sequences
    for key in data[0].keys():  
        new_item[key] = [d[key] for d in data]

    source, _       = merge(new_item["source"])
    target, lengths = merge(new_item["target"])
    
    # Move the padded sequences to the specified device
    new_item["source"] = source.to(device)
    new_item["target"] = target.to(device)

    new_item["number_tokens"] = sum(lengths)
    return new_item