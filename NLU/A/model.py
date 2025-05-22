import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class ModelIAS(nn.Module):
    """
    A neural network model for Joint Intent Detection and Slot Filling (SLU - Spoken Language Understanding).
    
    The model uses an LSTM to encode utterances and produces:
        - Slot labels for each token in the input (sequence labeling)
        - An intent label for the entire utterance (classification)
    """

    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len, n_layer=1, pad_index=0):
        """
        Initializes the model layers.

        Args:
            hid_size (int): Size of the hidden state in LSTM.
            out_slot (int): Number of slot labels.
            out_int (int): Number of intent classes.
            emb_size (int): Dimension of word embeddings.
            vocab_len (int): Vocabulary size.
            n_layer (int): Number of LSTM layers.
            pad_index (int): Index of the padding token in the vocabulary.
        """
        super(ModelIAS, self).__init__()

        # Word Embedding Layer
        # Converts token indices into dense vectors of size emb_size
        # padding_idx ensures that the embedding for pad tokens is not updated
        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)
        
        # LSTM Encoder
        # Encodes the embedded utterance into a sequence of hidden states
        # batch_first=True means input/output tensors are shaped (batch_size, seq_len, features)
        self.utt_encoder = nn.LSTM(emb_size, hid_size, n_layer, bidirectional=False, batch_first=True)   
        
        # Output layer for slot filling: maps hidden states to slot logits 
        self.slot_out = nn.Linear(hid_size, out_slot)
        
        # Output layer for intent detection: maps final hidden state to intent logits
        self.intent_out = nn.Linear(hid_size, out_int)
        
        
    def forward(self, utterance, seq_lengths):
        """
        Forward pass of the model.

        Args:
            utterance (torch.Tensor): Input tensor of shape (batch_size, seq_len)
            seq_lengths (torch.Tensor): Lengths of sequences in the batch

        Returns:
            slots (torch.Tensor): Slot logits of shape (batch_size, out_slot, seq_len)
            intent (torch.Tensor): Intent logits of shape (batch_size, out_int)
        """

        # Step 1: Convert token indices to embeddings
        utt_emb = self.embedding(utterance) # Shape: (batch_size, seq_len, emb_size)
        
        # Step 2: Pack the padded sequence to skip computation on padding tokens
        packed_input = pack_padded_sequence(
            utt_emb, 
            seq_lengths.cpu().numpy(),  # Ensure lengths are numpy array (compatibility)
            batch_first=True)
        
        # Step 3: Pass through LSTM encoder
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input) 
       
        # Step 4: Unpack the sequence back to original structure
        utt_encoded, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        
        # Step 5: Get final hidden state for intent classification
        # Take hidden state from last layer of LSTM
        last_hidden = last_hidden[-1,:,:]  # Shape: (batch_size, hid_size)
        
        # Step 6: Compute slot logits for each time step
        slots = self.slot_out(utt_encoded)   # Shape: (batch_size, seq_len, out_slot)
        
        # Step 7: Compute intent logits using final hidden state
        intent = self.intent_out(last_hidden)
        
        # Step 8: Permute slot logits for compatibility with CrossEntropyLoss
        slots = slots.permute(0,2,1) # From (B, L, C) -> (B, C, L)
        
        # Slot size: batch_size, classes, seq_len
        return slots, intent
    



class BiModelIAS(nn.Module):
    """
    A neural network model for Joint Intent Detection and Slot Filling (SLU - Spoken Language Understanding).
    Now supports bidirectional LSTM.
    """

    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len, n_layer=1, pad_index=0):
        """
        Initializes the model layers.

        Args:
            hid_size (int): Size of the hidden state in one direction of LSTM.
            out_slot (int): Number of slot labels.
            out_int (int): Number of intent classes.
            emb_size (int): Dimension of word embeddings.
            vocab_len (int): Vocabulary size.
            n_layer (int): Number of LSTM layers.
            pad_index (int): Index of the padding token in the vocabulary.
        """
        super(BiModelIAS, self).__init__()
        
        self.bidirectional = True
        self.num_directions = 2 

        # Word Embedding Layer
        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)

        # Bidirectional LSTM Encoder
        self.utt_encoder = nn.LSTM(
            emb_size, 
            hid_size, 
            n_layer, 
            bidirectional=self.bidirectional, 
            batch_first=True
        )

        # Output layer for slot filling: input is hidden_size * 2 due to bidirection
        self.slot_out = nn.Linear(hid_size * self.num_directions, out_slot)

        # Output layer for intent detection
        self.intent_out = nn.Linear(hid_size * self.num_directions, out_int)


    def forward(self, utterance, seq_lengths):
        """
        Forward pass of the model.

        Args:
            utterance (torch.Tensor): Input tensor of shape (batch_size, seq_len)
            seq_lengths (torch.Tensor): Lengths of sequences in the batch

        Returns:
            slots (torch.Tensor): Slot logits of shape (batch_size, out_slot, seq_len)
            intent (torch.Tensor): Intent logits of shape (batch_size, out_int)
        """

        # Step 1: Embedding
        utt_emb = self.embedding(utterance)  # (batch_size, seq_len, emb_size)

        # Step 2: Packing
        packed_input = pack_padded_sequence(
            utt_emb, 
            seq_lengths.cpu().numpy(), 
            batch_first=True,
            enforce_sorted=False
        )

        # Step 3: LSTM
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input)

        # Step 4: Unpack
        utt_encoded, _ = pad_packed_sequence(packed_output, batch_first=True)

        # Step 5: Intent classification, concatenate the last hidden states from both directions
        last_hidden = torch.cat((last_hidden[-2], last_hidden[-1]), dim=1)  # (batch_size, hid_size * 2)

        # Step 6: Slot filling
        slots = self.slot_out(utt_encoded)  # (batch_size, seq_len, out_slot)

        # Step 7: Intent logits
        intent = self.intent_out(last_hidden)  # (batch_size, out_int)

        # Step 8: Permute slot logits
        slots = slots.permute(0, 2, 1)  # (batch_size, out_slot, seq_len)

        return slots, intent




class DoModelIAS(nn.Module):
    """
    A neural network model for Joint Intent Detection and Slot Filling (SLU - Spoken Language Understanding).
    Now supports bidirectional LSTM with dropout.
    """

    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len, n_layer=1, pad_index=0, dropout=0.3):
        """
        Initializes the model layers.

        Args:
            hid_size (int): Size of the hidden state in one direction of LSTM.
            out_slot (int): Number of slot labels.
            out_int (int): Number of intent classes.
            emb_size (int): Dimension of word embeddings.
            vocab_len (int): Vocabulary size.
            n_layer (int): Number of LSTM layers.
            pad_index (int): Index of the padding token in the vocabulary.
            dropout (float): Dropout probability.
        """
        super(DoModelIAS, self).__init__()
        
        self.bidirectional = True
        self.num_directions = 2 

        # Word Embedding Layer
        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)

        # Bidirectional LSTM Encoder
        self.utt_encoder = nn.LSTM(
            emb_size, 
            hid_size, 
            n_layer, 
            bidirectional=self.bidirectional, 
            batch_first=True,
            dropout=dropout if n_layer > 1 else 0.0  # LSTM dropout only works if n_layer > 1
        )

        # Dropout Layer
        self.dropout = nn.Dropout(dropout)

        # Output layers
        self.slot_out = nn.Linear(hid_size * self.num_directions, out_slot)
        self.intent_out = nn.Linear(hid_size * self.num_directions, out_int)

    def forward(self, utterance, seq_lengths):
        """
        Forward pass of the model.

        Args:
            utterance (torch.Tensor): Input tensor of shape (batch_size, seq_len)
            seq_lengths (torch.Tensor): Lengths of sequences in the batch

        Returns:
            slots (torch.Tensor): Slot logits of shape (batch_size, out_slot, seq_len)
            intent (torch.Tensor): Intent logits of shape (batch_size, out_int)
        """

        # Step 1: Embedding
        utt_emb = self.embedding(utterance)  # (batch_size, seq_len, emb_size)

        # Step 2: Packing
        packed_input = pack_padded_sequence(
            utt_emb, 
            seq_lengths.cpu().numpy(), 
            batch_first=True,
            enforce_sorted=False
        )

        # Step 3: LSTM
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input)

        # Step 4: Unpack
        utt_encoded, _ = pad_packed_sequence(packed_output, batch_first=True)

        # Step 5: Apply dropout to LSTM output
        utt_encoded = self.dropout(utt_encoded)

        # Step 6: Intent classification, concatenate the last hidden states from both directions
        last_hidden = torch.cat((last_hidden[-2], last_hidden[-1]), dim=1)  # (batch_size, hid_size * 2)
        last_hidden = self.dropout(last_hidden)

        # Step 7: Slot filling
        slots = self.slot_out(utt_encoded)  # (batch_size, seq_len, out_slot)

        # Step 8: Intent logits
        intent = self.intent_out(last_hidden)  # (batch_size, out_int)

        # Step 9: Permute slot logits
        slots = slots.permute(0, 2, 1)  # (batch_size, out_slot, seq_len)

        return slots, intent