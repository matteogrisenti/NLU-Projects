import torch
from utils import read_file, Lang, PennTreeBank
from functions import train_model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", DEVICE)



# --------------------------------------------- DATASET MANAGEMENT ----------------------------------------------
train_raw = read_file("dataset/PennTreeBank/ptb.train.txt")
dev_raw = read_file("dataset/PennTreeBank/ptb.valid.txt")
test_raw = read_file("dataset/PennTreeBank/ptb.test.txt")

lang = Lang(train_raw, ["<pad>", "<eos>"])

train_dataset = PennTreeBank(train_raw, lang)
dev_dataset   = PennTreeBank(dev_raw, lang)
test_dataset  = PennTreeBank(test_raw, lang)

print("DATASET:")
print("\tTrain dataset size: ", len(train_dataset))
print("\tDev dataset size: ", len(dev_dataset))
print("\tTest dataset size: ", len(test_dataset))
print("\tVocab size: ", len(lang.word2id))



# --------------------------------------------  HYPERPARAMETERS ------------------------------------------------
LABEL = 'ADAMW'      # RNN, LSTM
BATCH_SIZE = 128     # Original 64
HID_SIZE = [200]                            # Original 200
EMB_SIZE = [300]                            # Original 300
N_LAYERS = [4,5]                            # Original 1
DROPOUT_EMB = 0.5
DROPOUT_OUT = 0.5
LR = 0.001
OPTIMIZER = 'AdamW' # SGD or Adam
CLIP = 5            # Clip the gradient -> avoid exploding gradients



# -------------------------------------------- TRAINING ------------------------------------------------
for j in range(len(N_LAYERS)):
    n_layers = N_LAYERS[j]
    for i in range(len(EMB_SIZE)):
        #d o_emb = DROPOUT_EMB[i]
        # do_out = DROPOUT_OUT[i]
        # lr = LR[i]
        # batchsize = BATCH_SIZE[i]
        hid_size = HID_SIZE[i]
        emb_size = EMB_SIZE[i]


        # print("Training with learning rate: ", lr)
        # print("Training with batch size: ", batchsize)
        # print("Training with hidden size: ", hid_size)
        print("Training with embedding size: ", emb_size)
        # print("Training with hidden size: ", hid_size, " and embedding size: ", emb_size)
        # print("Training with dropout embedding: ", do_emb, " and dropout output: ", do_out)

        train_model(
            train_dataset,
            dev_dataset,
            test_dataset,
            lang,
            BATCH_SIZE=BATCH_SIZE,
            HID_SIZE=hid_size,
            EMB_SIZE=emb_size,
            N_LAYERS=n_layers,
            LR=LR,
            DROPOUT_EMB=DROPOUT_EMB,
            DROPOUT_OUT=DROPOUT_OUT,
            CLIP=CLIP,
            OPTIMIZER=OPTIMIZER,
            DEVICE=DEVICE,
            LABEL=LABEL
        )

