import gc
import os
import sys
import time
from datetime import datetime
from math import floor

import numpy as np
import pandas as pd
import torch
import torchmetrics
from torch import nn
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from classifier import BertClassifier

model_options = {
    "biobert": "dmis-lab/biobert-v1.1",
    "pubmed_abstract": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
    "scibert": "allenai/scibert_scivocab_uncased",
    "pubmed_fulltext": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    "medbert": "Charangan/MedBERT",
    "basebert": "bert-base-uncased",
    "tinybert": "prajjwal1/bert-tiny",
    "minibert": "prajjwal1/bert-mini",
    "smallbert": "prajjwal1/bert-small",
    "mediumbert": "prajjwal1/bert-medium"
}


class Dataset(torch.utils.data.Dataset):
    """PyTorch Dataset class for our systematic review datasets.
    """

    def __init__(self, df, current_model):
        """Creates the dataset
              Params:
                df: dataset in a dataframe 
        """

        tokenizer = AutoTokenizer.from_pretrained(current_model)

        labels_dict = {
            'Excluded': 0,
            'Included': 1,
        }

        self.labels = []
        label = [0, 0]
        for decision in df['decision']:
            label[labels_dict[decision]] = 1
            self.labels.append(label)
            label = [0, 0]

        self.texts = [tokenizer(text, padding='max_length', max_length=256, truncation=True,
                                return_tensors="pt") for text in df['titleabstract']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y


def train(model_name, train_path, val_path, learning_rate, epochs, batch_size):
    """ Function to train the model.
        Params:
          - model: the model to be trained
          - train_data: traing data (Pandas DataFrame format)
          - val_data: validation data (Pandas DataFrame format)
          - learning_rate: learning rate
          - epochs: the number of epochs for training
    """
    start_time = datetime.now()
    save_path = os.path.join("classifier/models", model_name, start_time.strftime("%d.%m %H.%M"))
    log_path = os.path.join(save_path, "logs")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    summary_log = open(os.path.join(log_path, "summary"), 'w')
    summary_log.write(f"batch_size: {batch_size} \nepochs: {epochs} \ndata: {train_path} \n \n")

    current_model = model_options[model_name]

    print("Get model")
    model = BertClassifier(hidden=768, model_type=current_model)  # 768 or 128
    model.train()

    print("Retrieving data")
    train_data = pd.read_csv(train_path)
    val_data = pd.read_csv(val_path)
    train_set, val = Dataset(train_data, current_model), Dataset(val_data, current_model)

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=30)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    print("Building optimizer")
    # loss_weights = torch.Tensor([1., 17.])  # pick the weights
    # criterion = nn.CrossEntropyLoss(weight=loss_weights)
    criterion = nn.BCELoss()
    acc = torchmetrics.classification.BinaryAccuracy(threshold=0.5)
    precision = torchmetrics.classification.BinaryPrecision(threshold=0.5)
    recall = torchmetrics.classification.BinaryRecall(threshold=0.5)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0005)

    num_training_steps = epochs * len(train_dataloader)
    # lr_schedule = lr_scheduler.ReduceLROnPlateau(optimizer=optimizer)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()
        acc = acc.cuda()
        precision = precision.cuda()
        recall = recall.cuda()
        print("Using Cuda")

    print("Training")
    length = len(train_dataloader)
    for epoch_num in range(1, epochs + 1):

        total_loss_train = 0

        time_begin = time.time()
        current_time = datetime.now()
        current_time = current_time.strftime("%H:%M:%S")
        print(f"epoch: {epoch_num} time: {current_time}")
        i = 0
        epoch_log = open(os.path.join(log_path, "epoch_log " + str(epoch_num)), 'w')

        for train_input, train_label in train_dataloader:
            i += 1

            train_label = train_label.float().to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            model.zero_grad()
            optimizer.zero_grad()

            output, attentions = model(input_id, mask)
            # output, .. = model(input_id, mask)

            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()

            result = output.argmax(dim=1)

            batch_acc = acc(result, train_label[:, 1])
            batch_precision = precision(result, train_label[:, 1])
            batch_recall = recall(result, train_label[:, 1])

            batch_loss.backward()

            # if i % 2 == 0:
            optimizer.step()

            torch.cuda.empty_cache()
            sys.stdout.flush()
            gc.collect()

            #     lr_schedule.step(val_loss)
            #     learning_rate = optimizer.param_groups[0]["lr"]

            step_time = time.time()
            elapsed_time = step_time - time_begin
            batch_time = elapsed_time / i
            remaining_time = batch_time * (length - i)

            remaining_hours = remaining_time // 3600
            remaining_minutes = floor(((remaining_time / 3600) - remaining_hours) * 60)
            remaining_seconds = floor(((remaining_time / 60) - floor(remaining_time / 60)) * 60)

            elapsed_hours = elapsed_time // 3600
            elapsed_minutes = floor(((elapsed_time / 3600) - elapsed_hours) * 60)
            elapsed_seconds = floor(((elapsed_time / 60) - floor(elapsed_time / 60)) * 60)

            log = f"[{epoch_num}/{epochs}]: [{i}/{length}] loss:{batch_loss:.5f} " \
                  f"lr:{learning_rate:.6f} elapsed time:{elapsed_hours:.0f}:{elapsed_minutes:.0f}:{elapsed_seconds:.0f} " \
                  f"time remaining:{remaining_hours:.0f}:{remaining_minutes:.0f}:{remaining_seconds:.0f}"
            epoch_log.write(log + "\n")
            print(log)

            # if i >= 5:
            #     break

        train_acc = acc.compute()
        acc.reset()

        train_precision = precision.compute()
        precision.reset()

        train_recall = recall.compute()
        recall.reset()

        total_loss_val = 0
        print("Validating")
        with torch.no_grad():
            for val_input, val_label in val_dataloader:
                val_label = val_label.float().to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output, attentions = model(input_id, mask)

                val_loss = criterion(output, val_label)
                total_loss_val += val_loss.item()

                result = output.argmax(dim=1)

                batch_acc = acc(result, val_label[:, 1])
                batch_precision = precision(result, val_label[:, 1])
                batch_recall = recall(result, val_label[:, 1])

                sys.stdout.flush()
                gc.collect()

        val_acc = acc.compute()
        acc.reset()

        val_precision = precision.compute()
        precision.reset()

        val_recall = recall.compute()
        recall.reset()

        train_log = f"EPOCH {epoch_num} TRAIN average Loss: {(total_loss_train / len(train_dataloader)):.6f} Accuracy: {train_acc:.6f} Recall: {train_recall:.4f} Precision: {train_precision:.4f}"
        val_log = f"EPOCH {epoch_num} VALID average Loss: {(total_loss_val / len(val_dataloader)):.6f} Accuracy: {val_acc:.6f} Recall: {val_recall:.4f} Precision: {val_precision:.4f}"

        epoch_log.write(train_log + "\n")
        epoch_log.write(val_log)
        epoch_log.close()

        summary_log.write(f"{train_log}\n")
        summary_log.write(f"{val_log}\n")

        print(train_log)
        print(val_log)

        # model_name = "medbert" + str(epoch_num) + ".pt"

        torch.save(model.state_dict(),
                   os.path.join(save_path, f"{model_name}_epoch_{epoch_num}_{current_time.replace(':', '.')}.pt"))
    summary_log.close()


if __name__ == "__main__":
    # nltk.download('stopwords')

    # pick the model and create the tokenizer

    # read the training & validation data
    train_path = os.path.join("data", "sex_diff_train.csv")
    val_path = os.path.join("data", "sex_diff_val.csv")

    #
    # model = BertClassifier(hidden=768, model_type=current_model)

    LR = 2e-5
    EPOCHS = 5

    train('biobert', train_path, val_path, LR, EPOCHS, 25)
    #
    # torch.save(model.state_dict(), "bio.pt")
    #
    # model.load_state_dict(torch.load("bio.pt"))
