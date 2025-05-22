# BERT model script from: huggingface.co
from transformers import BertTokenizer, BertModel

'''
# ------------------------------------------------------
# 1. Load ATIS dataset (Section 4.1)
# ------------------------------------------------------
raw_datasets = load_dataset('atis')
intent_list = raw_datasets['train'].features['intent_label'].names
slot_list = raw_datasets['train'].features['slots'].feature.names
num_intent_labels = len(intent_list)
num_slot_labels = len(slot_list)


# ------------------------------------------------------
# 2. Tokenizer: WordPiece (Section 3.1)
# ------------------------------------------------------
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
label_all_tokens = False  # Only label first sub-token as per Section 3.2

# ------------------------------------------------------
# 3. Data preprocessing & sub-token alignment (Section 3.2)
# ------------------------------------------------------
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples['tokens'],
        truncation=True,
        is_split_into_words=True,
        max_length=50  # as in Section 4.2
    )

    all_slot_labels = []

    for i, slot_seq in enumerate(examples['slots']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        slot_labels = []
        for word_idx in word_ids:
            if word_idx is None:
                slot_labels.append(-100)  # ignore special tokens
            elif word_idx != previous_word_idx:
                # first sub-token: use actual slot label index
                slot_labels.append(slot_list.index(slot_seq[word_idx]))
            else:
                # subsequent sub-tokens: ignore or label all depending
                slot_labels.append(-100 if not label_all_tokens else slot_list.index(slot_seq[word_idx]))
            previous_word_idx = word_idx
        all_slot_labels.append(slot_labels)
    tokenized_inputs['slot_labels'] = all_slot_labels
    # Intent labels aligned to sentence ([CLS]) classification
    tokenized_inputs['intent_label'] = [intent_list.index(x) for x in examples['intent_label']]
    return tokenized_inputs

# Apply preprocessing
train_dataset = raw_datasets['train'].map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=raw_datasets['train'].column_names
)
valid_dataset = raw_datasets['validation'].map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=raw_datasets['validation'].column_names
)

data_collator = DataCollatorWithPadding(tokenizer)

# ------------------------------------------------------
# 4. Model Definition (Section 3.2 and 3.3)
# ------------------------------------------------------
class BertForJointIntentSlot(BertPreTrainedModel):
    def __init__(self, config, num_intent_labels, num_slot_labels, use_crf=False):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # Eq.1: Intent classifier on [CLS] (h1)
        self.intent_classifier = nn.Linear(config.hidden_size, num_intent_labels)
        # Eq.2: Slot classifier on token outputs
        self.slot_classifier = nn.Linear(config.hidden_size, num_slot_labels)
        self.use_crf = use_crf
        if use_crf:
            # CRF layer to model label dependencies (Section 3.3)
            self.crf = CRF(num_slot_labels, batch_first=True)
        self.init_weights()

    def forward(self, input_ids, attention_mask, token_type_ids=None,
                intent_label=None, slot_labels=None):
        # BERT encoder
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False
        )
        sequence_output, pooled_output = outputs  # sequence_output: (batch, seq_len, hidden)

        # Intent head
        pooled_output = self.dropout(pooled_output)
        intent_logits = self.intent_classifier(pooled_output)

        # Slot head
        sequence_output = self.dropout(sequence_output)
        slot_logits = self.slot_classifier(sequence_output)  # (batch, seq_len, num_slot_labels)

        loss = None
        if intent_label is not None and slot_labels is not None:
            # Cross-entropy for intent
            intent_loss_fct = nn.CrossEntropyLoss()
            intent_loss = intent_loss_fct(intent_logits, intent_label)
            # Slot loss: optionally with CRF
            if self.use_crf:
                # CRF neg-log-likelihood
                slot_loss = -self.crf(slot_logits, slot_labels, mask=attention_mask.bool(), reduction='mean')
            else:
                # Token classification loss ignoring -100
                slot_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                # reshape logits and labels
                active_loss = slot_labels.view(-1) != -100
                active_logits = slot_logits.view(-1, slot_logits.size(-1))[active_loss]
                active_labels = slot_labels.view(-1)[active_loss]
                slot_loss = slot_loss_fct(active_logits, active_labels)

            loss = intent_loss + slot_loss  # joint objective Eq.3

        return {
            'loss': loss,
            'intent_logits': intent_logits,
            'slot_logits': slot_logits
        }

# Initialize model
model = BertForJointIntentSlot.from_pretrained(
    'bert-base-uncased',
    num_intent_labels=num_intent_labels,
    num_slot_labels=num_slot_labels,
    use_crf=False  # set True to include CRF layer
)

# ------------------------------------------------------
# 5. Metrics: accuracy & seqeval (Section 2)
# ------------------------------------------------------
intent_metric = load_metric('accuracy')
slot_metric = load_metric('seqeval')

# Also compute sentence-level frame accuracy (exact match)
def frame_accuracy(pred_intents, true_intents, pred_slots, true_slots):
    correct = 0
    total = len(true_intents)
    for pi, ti, ps, ts in zip(pred_intents, true_intents, pred_slots, true_slots):
        if pi == ti and ps == ts:
            correct += 1
    return correct / total


def compute_metrics(eval_pred):
    (intent_logits, slot_logits), labels = eval_pred
    # Intent
    intent_preds = torch.argmax(torch.tensor(intent_logits), dim=1).tolist()
    intent_labels = labels['intent_label']
    intent_acc = intent_metric.compute(predictions=intent_preds, references=intent_labels)
    # Slot
    slot_preds = torch.argmax(torch.tensor(slot_logits), dim=2).tolist()
    slot_labels = labels['slot_labels']
    # Convert indices to tags, ignore -100
    true_labels = [[slot_list[l] for l in seq if l != -100] for seq in slot_labels]
    true_preds = [[slot_list[p] for (p, l) in zip(seq_p, seq_l) if seq_l != -100]
                  for seq_p, seq_l in zip(slot_preds, slot_labels)]
    slot_f1 = slot_metric.compute(predictions=true_preds, references=true_labels)
    # Sentence-level
    frame_acc = frame_accuracy(intent_preds, intent_labels, true_preds, true_labels)

    return {
        'intent_accuracy': intent_acc['accuracy'],
        'slot_f1': slot_f1['overall_f1'],
        'frame_accuracy': frame_acc
    }

# ------------------------------------------------------
# 6. TrainingArguments (Section 4.2)
# ------------------------------------------------------
training_args = TrainingArguments(
    output_dir='bert-atis-joint',
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    learning_rate=5e-5,
    num_train_epochs=30,  # tune on dev from [1,5,10,20,30,40]
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='slot_f1'
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# ------------------------------------------------------
# 7. Run training
# ------------------------------------------------------
trainer.train()
'''