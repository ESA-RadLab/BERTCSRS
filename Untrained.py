import gc
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertModel

from bert_classifier_train import Dataset

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

model_type = model_options['biobert']

bert = BertModel.from_pretrained(model_type, output_attentions=True)

test_data = pd.read_csv('data/cns_test.csv')

val = Dataset(test_data, model_type)

val_dataloader = torch.utils.data.DataLoader(val, batch_size=30)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
if use_cuda:
    model = bert.cuda()
    # criterion = criterion.cuda()
    print("Using Cuda")

total_acc_val = 0
total_loss_val = 0
tp_v = 0
fn_v = 0
fp_v = 0

print("Validating")
with torch.no_grad():
    for val_input, val_label in val_dataloader:
        val_label = val_label.to(device)
        mask = val_input['attention_mask'].to(device)
        input_id = val_input['input_ids'].squeeze(1).to(device)

        attentions = bert(input_id, mask)
        attentions

        # val_loss = criterion(output, val_label.long())
        # total_loss_val += val_loss.item()

        acc = (output.argmax(dim=1) == val_label).sum().item()

        for ind, out in enumerate(output.argmax(dim=1)):
            if out == val_label[ind] and val_label[ind] == 1:
                tp_v += 1
            elif out != val_label[ind] and val_label[ind] == 1:
                fn_v += 1
            elif val_label[ind] == 0 and out == 1:
                fp_v += 1

        total_acc_val += acc

        sys.stdout.flush()
        gc.collect()


if tp_v + fn_v > 0:
    recall_v = tp_v / (tp_v + fn_v)
else:
    recall_v = 0

if tp_v + fp_v > 0:
    precision_v = tp_v / (tp_v + fp_v)
else:
    precision_v = 0