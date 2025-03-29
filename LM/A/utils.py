import torch
import torch.utils.data as data

# LOADING CORPUS: organize each line as a sentence
# and add an end of sentence token <eos> to each sentence
# group the sentences in a list
def read_file(path, eos_token="<eos>"):
    output = []
    with open(path, "r") as f:
        for line in f.readlines():
            output.append(line.strip() + " " + eos_token)
    return output




# LANG: This class computes and stores our vocab: Word to ids and ids to word
class Lang():

    def __init__(self, corpus, special_tokens=[]):
        self.word2id = self.get_vocab(corpus, special_tokens)
        self.id2word = {v:k for k, v in self.word2id.items()}
    
    # GET VOCABULARY: create a dictionary that maps each word to an index
    def get_vocab(self, corpus, special_tokens=[]):
        output = {}
        i = 0 
        for st in special_tokens:
            output[st] = i
            i += 1
        for sentence in corpus:
            for w in sentence.split():
                if w not in output:
                    output[w] = i
                    i += 1
        return output
    



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
        # Returns a sample of the dataset, which is a dictionary with:
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
def collate_fn(data, pad_token, DEVICE):

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
    new_item["source"] = source.to(DEVICE)
    new_item["target"] = target.to(DEVICE)

    new_item["number_tokens"] = sum(lengths)
    return new_item