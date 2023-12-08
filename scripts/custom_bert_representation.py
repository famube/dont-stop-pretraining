import torch
import numpy as np
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, RobertaForSequenceClassification, AutoModel, AutoTokenizer
from tqdm import tqdm, trange
import pickle
from sklearn.datasets import dump_svmlight_file
from sklearn.preprocessing import LabelEncoder
import os
import argparse


class Dataset(torch.utils.data.Dataset):

    def __init__(self, samples, label_encoder, tokenizer, max_seq_len=256):
        self.labels = label_encoder.transform([sample["cls"] for sample in samples])
        self.texts = [sample["text"] for sample in samples]
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        features = self.tokenizer(self.texts[idx],
                                   padding='max_length', max_length=self.max_seq_len, truncation=True,
                                   return_tensors="pt")
        return features, torch.tensor(self.labels[idx])

def load_pkl(filename):
    with open(filename, "rb") as infile:
        return pickle.load(infile)


def my_pooler(hidden_states):
    #Dimensions: 4(last hidden states) * batches * tokens * 768
    last4 = torch.stack([hidden_states[-x] for x in range(1, 5)])
    #print("last4.shape", last4.shape)
    last4_avg = torch.mean(last4, dim=0)
    #print("last4_avg.shape", last4_avg.shape)
    token_avg = torch.mean(last4_avg, dim=1)
    #print("token_avg.shape", token_avg.shape)
    return token_avg


def train(model, train_dataset, args):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-08)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    epochs = args.epochs
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    for epoch in trange(epochs, desc = 'Epoch'):

        # Set model to training mode
        model.train()

        # Tracking variables
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        for batch in tqdm(train_dataloader):
            X,y = batch
            mask = X['attention_mask'].to(device)
            input_id = X['input_ids'].squeeze(1).to(device)
            optimizer.zero_grad()
            # Forward pass
            train_output = model(input_id,
                                 token_type_ids=None,
                                 attention_mask=mask,
                                 labels=y.to(device), return_dict=True)
            # Backward pass
            train_output["loss"].backward()
            optimizer.step()
            # Update tracking variables
            tr_loss += train_output["loss"].item()
            nb_tr_examples += input_id.size(0)
            nb_tr_steps += 1
        print("Epoch", epoch, "Loss:", tr_loss/nb_tr_steps)


# Text classification benchmark datasets (TeCBench)
# Gera representacoes bert (modelo customizado) para dataset

def represent_tecbench_data(args):
    indir = args.input_dir + "/" + args.dataset
    outdir = args.output_dir + "/" + args.dataset

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    samples = load_pkl(indir + "/samples.pkl")
    labels = [sample["cls"]for sample in samples]
    label_enc = LabelEncoder()
    label_enc.fit(labels)

    if args.finetune_with == None:
        model = AutoModel.from_pretrained(args.model_name_or_path,
                                          output_attentions = False,
                                          output_hidden_states = True)

    else:
        model = RobertaForSequenceClassification.from_pretrained(args.model_name_or_path,
                                                                 num_labels = len(label_enc.classes_),
                                                                 output_attentions = False,
                                                                 output_hidden_states = True)
    try:
        os.makedirs(outdir)
    except:
        pass

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if args.finetune_with != None:
        if args.finetune_with == "train":
            train_ids = load_pkl(f"{indir}/fold_{args.fold}/train.pkl")
        else:
            train_ids = load_pkl(args.finetune_with)
        train_samples = [samples[i] for i in train_ids]
        train_dataset = Dataset(train_samples, label_enc, tokenizer, max_seq_len=args.max_seq_len)
        train(model, train_dataset, args)
        model.eval() #Sai do modo treino
    else:
        model.to(device)


    for part in ["test", "train", "val"]:
        outname = f"{outdir}/{part}{args.fold}"
        ids = load_pkl(f"{indir}/fold_{args.fold}/{part}.pkl")
        samples_part = [samples[i] for i in ids]
        dataset = Dataset(samples_part, label_enc, tokenizer, max_seq_len=args.max_seq_len)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        with torch.no_grad():
            tensors = []
            for X, y in tqdm(dataloader):
                mask = X['attention_mask'].to(device)
                input_id = X['input_ids'].squeeze(1).to(device)
                output = model(input_ids=input_id,
                               attention_mask=mask,return_dict=True)
                hidden_states = output["hidden_states"]
                tens = my_pooler(hidden_states)
                tensors.append(tens.cpu())
            X = torch.cat(tensors, dim=0).numpy()
            y = np.array([instance["cls"] for instance in samples_part])
            print("X.shape, y.shape:", X.shape, y.shape)
            dump_svmlight_file(X, y, outname, zero_based=False)

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--dataset", default="webkb", type=str, help="The name of the TeCBench dataset."
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        default="/home/fabiano/tecbench",
        help="TeCBench input directory.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="bert_vectors",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="bert-base-cased",
        help="The pre-trained bert model name or path that will be used.",
    )

    parser.add_argument(
        "--finetune_with",
        type=str,
        default=None,
        help="The indices of samples that will be used to fine-tune the model. To use all train ids, simply set --finetune_with train",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Fold index",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5
    )

    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=256
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5
    )

    args = parser.parse_args()
    print(args)
    represent_tecbench_data(args)

main()


#for fold in range(3):
#    model_path = f"../input/dont-stop-pretraining/models/{dataset}/fold{fold}"
    #model_path = "bert-base-cased"
#    represent_tecbench_data(dataset, model_path, fold=fold)



