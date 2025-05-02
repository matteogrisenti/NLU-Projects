import torch
from utils import read_file, Lang, PennTreeBank
from functions import train_model, train_model_nt_avsgd


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
BATCH_SIZE = 32      # Original 64
HID_SIZE = 700       # Original 200
EMB_SIZE = 700       # Original 300
N_LAYERS = 1   # Original 1
LR = 2
DROPOUT = [0.5, 0.7]
CLIP = 5            # Clip the gradient -> avoid exploding gradients



# -------------------------------------------- TRAINING ------------------------------------------------
for j in range(len(DROPOUT)):
    dropout = DROPOUT[j]
    
    print("Training with dropout: ", dropout)
    # print("Training with learning rate: ", lr)
    # print("Training with batch size: ", batchsize)
    # print("Training with hidden size: ", hid_size)
    # print("Training with embedding size: ", emb_size)
    # print("Training with hidden size: ", hid_size, " and embedding size: ", emb_size)

    train_model_nt_avsgd(
        train_dataset, 
        dev_dataset,
        test_dataset,
        lang,
        BATCH_SIZE,
        HID_SIZE,
        EMB_SIZE,
        N_LAYERS,
        LR,
        DROPOUT,
        CLIP,
        DEVICE,
    )

