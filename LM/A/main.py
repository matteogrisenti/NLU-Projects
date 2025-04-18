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



# --------------------------------------------  HYPERPARAMETERS ------------------------------------------------
LABEL = 'RNN'
BATCH_SIZE = 64     # Original 64
HID_SIZE = 200      # Original 200
EMB_SIZE = 300      # Original 300
DROPOUT_EMB = None
DROPOUT_OUT = None
LR = [2, 1, 0.5]
OPTIMIZER = 'SGD'   # SGD or Adam
CLIP = 5            # Clip the gradient -> avoid exploding gradients



# -------------------------------------------- TRAINING ------------------------------------------------
for lr in LR:
    print("Training with learning rate: ", lr)
    train_model(
        train_dataset,
        dev_dataset,
        test_dataset,
        lang,
        BATCH_SIZE=BATCH_SIZE,
        HID_SIZE=HID_SIZE,
        EMB_SIZE=EMB_SIZE,
        LR=lr,
        DROPOUT_EMB=DROPOUT_EMB,
        DROPOUT_OUT=DROPOUT_OUT,
        CLIP=CLIP,
        OPTIMIZER=OPTIMIZER,
        DEVICE=DEVICE,
        LABEL=LABEL
    )

