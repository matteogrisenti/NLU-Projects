import torch.nn as nn
from transformers import BertModel       # BERT model script from: huggingface.co


class BertIntentSlot(nn.Module):
    def __init__(self, bert_type: str, num_intent_labels: int, num_slot_labels: int, dropout=0.1):
        """
        Joint BERT model for intent classification and slot filling (without CRF).
        
        Args:
            bert_type (str): Name of the pre-trained BERT model to load (e.g., 'bert-base-uncased').
            num_intent_labels (int): Number of possible intent classes.
            num_slot_labels (int): Number of possible slot labels (BIO or BILOU tags).
            dropout (float): Dropout probability for regularization.
        """
        super(BertIntentSlot, self).__init__()

        # Load pre-trained BERT model
        self.bert = BertModel.from_pretrained(bert_type)

        # Dropout helps prevent overfitting during fine-tuning 
        self.dropout = nn.Dropout(dropout)

        # Intent classifier: uses [CLS] token's embedding (first token) for sentence-level classification
        self.intent_classifier = nn.Linear(self.bert.config.hidden_size , num_intent_labels)

        # Slot classifier: uses token-level BERT outputs to predict slot labels for each token
        self.slot_classifier = nn.Linear(self.bert.config.hidden_size, num_slot_labels)

        print(f"[DEBUG] New model initialized with:\n"
                f"\t- BERT type: {bert_type} (hidden size: {self.bert.config.hidden_size})\n"
                f"\t- Intent classifier output dim: {num_intent_labels}\n"
                f"\t- Slot classifier output dim: {num_slot_labels}\n"
                f"\t- Dropout rate: {dropout}\n ")
        

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        Forward pass through the model.
        
        Args:
            input_ids (Tensor): Token IDs of shape (batch_size, seq_len).
            attention_mask (Tensor): Attention mask to avoid attending to [PAD] tokens.
            token_type_ids (Tensor):Segment IDs to distinguish two sentences (useful in tasks with two inputs: answer to the question). 
                                    In our intent/slot classification case we have only one sentence in input,
                                    so this parameter is not meaningful for us but bert expects it in input 

        Returns:
            slot_logits (Tensor): Logits for slot filling task, shape (batch_size, seq_len, num_slot_labels).
            intent_logits (Tensor): Logits for intent classification task, shape (batch_size, num_intent_labels).
        """
        # BERT model returns:
        # - last_hidden_state: token embeddings (for all tokens including subwords)
        # - pooler_output: embedding of [CLS] token (used for intent classification: assign a emded to each start sentence)
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids,
            return_dict=True                        # allows access via named fields: outputs.last_hidden_state, outputs.pooler_output
        )

        sequence_output = outputs.last_hidden_state     # Shape: (batch_size, seq_len, hidden_size)
        pooled_output = outputs.pooler_output           # Shape: (batch_size, hidden_size), from [CLS] token

        # Apply dropout before classification heads
        pooled_output = self.dropout(pooled_output)
        sequence_output = self.dropout(sequence_output)

        # Compute intent logits from [CLS] embedding
        intent_logits = self.intent_classifier(pooled_output)

        # Compute slot logits from token-level embeddings
        slot_logits = self.slot_classifier(sequence_output)

        return slot_logits, intent_logits


    def init_classification_heads(self):
        '''
        Initialize the weights of the classification heads using Xavier initialization.
        NB: I have to change the init_weight of part A becouse it can interfere with the init of the BERT model.
        '''
        # Xavier init for your classifiers
        for classifier in [self.intent_classifier, self.slot_classifier]:
            nn.init.xavier_uniform_(classifier.weight)
            if classifier.bias is not None:
                classifier.bias.data.fill_(0.0)



