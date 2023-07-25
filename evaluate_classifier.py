import gc
import sys

import nltk
import numpy as np
import pandas as pd
import torch
import itertools

# import torchmetrics
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryRecall, BinaryPrecision, BinaryF1Score, BinaryCohenKappa
from sklearn.metrics import confusion_matrix

# from classifier import BertClassifier
from classifier_old import BertClassifier
from bert_classifier_train import Dataset

nltk.download('stopwords')

model_options = {
    "biobert": ["dmis-lab/biobert-v1.1", 768],
    "pubmed_abstract": ["microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract", 768],
    "scibert": ["allenai/scibert_scivocab_uncased", 768],
    "pubmed_fulltext": ["microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", 768],
    "medbert": ["Charangan/MedBERT", 768],
    "basebert": ["bert-base-uncased", 768],
    "tinybert": ["prajjwal1/bert-tiny", 128],
    "minibert": ["prajjwal1/bert-mini", 768],
    "smallbert": ["prajjwal1/bert-small", 768],
    "mediumbert": ["prajjwal1/bert-medium", 768]
}


# class Dataset(torch.utils.data.Dataset):
#     """PyTorch Dataset class for our systematic review datasets.
#     """
#
#     def __init__(self, df):
#         """Creates the dataset
#               Params:
#                 df: dataset in a dataframe
#         """
#         self.labels = [labels[label] for label in df['Decision']]
#         self.texts = [tokenizer(text, padding='max_length', max_length=512, truncation=True,
#                                 return_tensors="pt") for text in df['titleabstract']]
#
#     def classes(self):
#         return self.labels
#
#     def __len__(self):
#         return len(self.labels)
#
#     def get_batch_labels(self, idx):
#         # Fetch a batch of labels
#         return np.array(self.labels[idx])
#
#     def get_batch_texts(self, idx):
#         # Fetch a batch of inputs
#         return self.texts[idx]
#
#     def __getitem__(self, idx):
#         batch_texts = self.get_batch_texts(idx)
#         batch_y = self.get_batch_labels(idx)
#         return batch_texts, batch_y


def test(bert_name, model_path, data_path, batch_size):
    # pick the model and create the tokenizer
    current_model = model_options[bert_name][0]
    hidden_layer = model_options[bert_name][1]

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = BertClassifier(hidden=hidden_layer, model_type=current_model)  # 128 or 768

    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict, strict=False)

    model.to(device)

    model.eval()

    test_data = pd.read_csv(data_path)

    test_dataset = Dataset(test_data, current_model)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    torch.cuda.empty_cache()

    total_acc_train = 0
    total_loss_train = 0
    total_recall_train = 0
    full_output = []

    tp = 0
    fn = 0
    fp = 0

    acc = BinaryAccuracy(threshold=0.5)
    precision = BinaryPrecision(threshold=0.5)
    recall = BinaryRecall(threshold=0.5)
    recall_4 = BinaryRecall(threshold=0.4)
    recall_3 = BinaryRecall(threshold=0.3)
    auroc = BinaryAUROC(thresholds=5)
    f1 = BinaryF1Score()
    cohen = BinaryCohenKappa()

    if use_cuda:
        acc = acc.cuda()
        precision = precision.cuda()
        recall = recall.cuda()
        recall_4 = recall_4.cuda()
        recall_3 = recall_3.cuda()
        auroc = auroc.cuda()
        f1 = f1.cuda()
        cohen = cohen.cuda()

    for train_input, train_label in test_dataloader:

        train_label = train_label.float().unsqueeze(-1).to(device)
        mask = train_input['attention_mask'].to(device)
        input_id = train_input['input_ids'].squeeze(1).to(device)

        output, attentions = model(input_id, mask)
        # full_output.append(output[:].detach().cpu().numpy())

        # acc = (output.argmax(dim=1) == train_label).sum().item()
        # total_acc_train += acc

        batch_acc = acc(output, train_label)
        batch_precision = precision(output, train_label)
        batch_recall = recall(output, train_label)
        recall_4(output, train_label)
        recall_3(output, train_label)
        auroc(output, train_label)
        batch_f1 = f1(output, train_label)
        batch_cohen = cohen(output, train_label)

        for ind, out in enumerate(output.argmax(dim=1)):
            if out == 1 and train_label[ind] == 1:
                tp += 1
            elif out == 0 and train_label[ind] == 1:
                fn += 1
            elif train_label[ind] == 0 and out == 1:
                fp += 1

        torch.cuda.empty_cache()
        sys.stdout.flush()
        gc.collect()

    test_acc = acc.compute()
    acc.reset()

    test_precision = precision.compute()
    precision.reset()

    test_recall = recall.compute()
    recall.reset()
    test_recall4 = recall_4.compute()
    recall_4.reset()
    test_recall3 = recall_3.compute()
    recall_3.reset()

    test_auroc = auroc.compute()
    auroc.reset()

    test_f1 = f1.compute()
    f1.reset()

    test_cohen = cohen.compute()
    cohen.reset()

    # all_logits = []
    # for array in full_output:
    #     all_logits.append(list(np.argmax(array, axis=1)[:]))
    #
    # all_logits = list(itertools.chain.from_iterable(all_logits))
    #
    # true_vals = pd.get_dummies(test_data['Decision']).values
    # true_vals = list(np.argmax(true_vals, axis=1))
    #
    # metric = BinaryRecall()
    # recall = metric(torch.tensor(all_logits), torch.tensor(true_vals))
    #
    # metric = BinaryPrecision()
    # precision = metric(torch.tensor(all_logits), torch.tensor(true_vals))
    #
    # metric = BinaryF1Score()
    # F1 = metric(torch.tensor(all_logits), torch.tensor(true_vals))
    #
    # metric = BinaryCohenKappa()
    # cohen = metric(torch.tensor(all_logits), torch.tensor(true_vals))
    #
    # # cohen
    #
    # metric = BinaryAUROC()
    # auc = metric(torch.tensor(all_logits), torch.tensor(true_vals))

    # print(f"{auc}, {cohen}, {F1}, {precision}, {recall}")
    print(f"acc:{test_acc:.4f} precision:{test_precision:.4f} recall:{test_recall:.4f} recall4:{test_recall4:.4f} recall3:{test_recall3:.4f} " 
          f"auroc:{test_auroc:.4f} f1:{test_f1:.4f} cohen:{test_cohen:.4f}")

    #
    # auc


def wss(R, y_true, y_pred):
    cfmat = confusion_matrix(y_true, y_pred)
    tn_, fp_, fn_, tp_ = cfmat.ravel()  # instead of doing a call for each
    N = np.sum(cfmat)
    if N <= 0:
        print("N = {}!!!".format(N))
    return (tn_ + fn_) / N - (1 - R)


def wss95(y_true, y_pred):
    return wss(0.95, y_true, y_pred)


# wss95(true_vals, all_logits)
if __name__ == "__main__":
    data_path = "data/cns_test.csv"
    # model_path = "models/pubmed_abstract/25.07_14.06/pubmed_abstract_epoch_6.pt"
    # model_path = "models/pubmed_abstract/24.07_13.27/pubmed_abstract_epoch_9_13.37.33.pt"
    model_path = "models/Original/cns.pt"
    bert_name = "biobert"
    batch_size = 24

    test(bert_name, model_path, data_path, batch_size)
