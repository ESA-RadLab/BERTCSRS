#!/usr/bin/env python
# coding: utf-8
import gc
import os
import sys
import time
from datetime import datetime

import nltk
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_scheduler

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

        labels = {
            'Excluded': 0,
            'Included': 1,
        }

        self.labels = [labels[label] for label in df['decision']]
        self.texts = [tokenizer(text, padding='max_length', max_length=512, truncation=True,
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


def train(model_name, train_data, val_data, learning_rate, epochs):
    """ Function to train the model.
        Params:
          - model: the model to be trained
          - train_data: traing data (Pandas DataFrame format)
          - val_data: validation data (Pandas DataFrame format)
          - learning_rate: learning rate
          - epochs: the number of epochs for training
    """
    print("Get model")
    model = BertClassifier(hidden=768, model_type=model_name)

    print("Retrieving data")
    train, val = Dataset(train_data, model_name), Dataset(val_data, model_name)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=8, shuffle=True, num_workers=0)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=8)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    print("Building optimizer")
    loss_weights = torch.Tensor([1., 17.])  # pick the weights
    criterion = nn.CrossEntropyLoss(weight=loss_weights)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    num_training_steps = epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()
        print("Using Cuda")

    print("Training")
    length = len(train_dataloader)
    for epoch_num in range(epochs):
        tp_t = 0
        fn_t = 0

        tp_v = 0
        fn_v = 0
        fp_v = 0

        total_acc_train = 0
        total_loss_train = 0
        total_recall_train = 0

        time_begin = time.time()
        current_time = datetime.now()
        current_time = current_time.strftime("%H:%M:%S")
        print(f"epoch: {epoch_num} time: {current_time}")
        i = 0
        for train_input, train_label in train_dataloader:
            i += 1
            print(f"Batch: {i}/{length}")

            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            output, attentions = model(input_id, mask)
            # output, .. = model(input_id, mask)

            batch_loss = criterion(output, train_label.long())
            total_loss_train += batch_loss.item()

            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            for ind, out in enumerate(output.argmax(dim=1)):
                if out == train_label[ind] and train_label[ind] == 1:
                    tp_t += 1
                elif out != train_label[ind] and train_label[ind] == 1:
                    fn_t += 1

            batch_loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            learning_rate = lr_scheduler.get_last_lr()


            sys.stdout.flush()
            gc.collect()

            total_acc_val = 0
            total_loss_val = 0

            # print("Validating")
            # with torch.no_grad():
            #     for val_input, val_label in val_dataloader:
            #         val_label = val_label.to(device)
            #         mask = val_input['attention_mask'].to(device)
            #         input_id = val_input['input_ids'].squeeze(1).to(device)
            #
            #         output, attentions = model(input_id, mask)
            #
            #         batch_loss = criterion(output, val_label.long())
            #         total_loss_val += batch_loss.item()
            #
            #         acc = (output.argmax(dim=1) == val_label).sum().item()
            #
            #         for ind, out in enumerate(output.argmax(dim=1)):
            #             if out == val_label[ind] and val_label[ind] == 1:
            #                 tp_v += 1
            #             elif out != val_label[ind] and val_label[ind] == 1:
            #                 fn_v += 1
            #             elif val_label[ind] == 0 and out == 1:
            #                 fp_v += 1
            #
            #         total_acc_val += acc
            #
            #         sys.stdout.flush()
            #         gc.collect()

            current_time = datetime.now()
            current_time = current_time.strftime("%H:%M:%S")
            log = f"[{epoch_num + 1}/{epochs}]: [{i}/{length}] loss:{batch_loss:.5f} lr:{learning_rate} time: {current_time}"  # time:{time_remain:.2f}h total:{time_remain_total:.2f}h"
            # outfile.write(log + "\n")
            # if i % 5 == 0:
            print(log)

        if tp_t + fn_t > 0:
            recall_t = tp_t / (tp_t + fn_t)
        else:
            recall_t = 0

        if tp_v + fn_v > 0:
            recall_v = tp_v / (tp_v + fn_v)
        else:
            recall_v = 0

        if tp_v + fp_v > 0:
            precision_v = tp_v / (tp_v + fp_v)
        else:
            precision_v = 0

        print('EPOCH ', epoch_num)
        print("Train loss", {total_loss_train / len(train_data)})
        print("Train Accuracy", {total_acc_train / len(train_data)})
        print("Train Recall", recall_t)
        # print("Validation loss", {total_loss_val / len(val_data)})
        # print("Validation Accuracy", {total_acc_val / len(val_data)})
        print("Validation Recall", recall_v)
        print('Val precision', precision_v)
        print('val tp', tp_v, 'fp', fp_v, 'fn', fn_v)

        model_name = "biobert" + str(epoch_num) + ".pt"
        os.path.join('..\\models', '')
        torch.save(model.state_dict(), "../models/" + model_name)


if __name__ == "__main__":
    nltk.download('stopwords')

    # pick the model and create the tokenizer

    # read the training & validation data
    train_path = os.path.join("..", "data", "cns_train_aug_900.csv")
    val_path = os.path.join("..", "data", "cns_val_80.csv")
    train_data = pd.read_csv(train_path)
    val_data = pd.read_csv(val_path)

    current_model = model_options['biobert']
    #
    # model = BertClassifier(hidden=768, model_type=current_model)

    LR = 2e-5
    EPOCHS = 5

    train(current_model, train_data, val_data, LR, EPOCHS)
    #
    # torch.save(model.state_dict(), "bio.pt")
    #
    # model.load_state_dict(torch.load("bio.pt"))
