import nltk
import numpy as np
import pandas as pd
import torch
import itertools

from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torchmetrics.classification import BinaryAUROC, BinaryRecall, BinaryPrecision, BinaryF1Score, BinaryCohenKappa
from sklearn.metrics import confusion_matrix

from classifier import BertClassifier

nltk.download('stopwords')

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

# pick the model and create the tokenizer
current_model = model_options['biobert']
tokenizer = AutoTokenizer.from_pretrained(current_model)

labels = {'Excluded': 0,
          'Included': 1,
          }


class Dataset(torch.utils.data.Dataset):
    """PyTorch Dataset class for our systematic review datasets.
    """

    def __init__(self, df):
        """Creates the dataset
              Params:
                df: dataset in a dataframe 
        """
        self.labels = [labels[label] for label in df['Decision']]
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


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

model = BertClassifier(hidden=768, model_type=current_model)

state_dict = torch.load("/content/gdrive/MyDrive/ESA/Models/biobert0.pt")
model.load_state_dict(state_dict)

model.to(device)

model.eval()

test_data = pd.read_csv('/content/gdrive/MyDrive/ESA/data70-10-20/cns_test.csv')

test = Dataset(test_data)
test_dataloader = torch.utils.data.DataLoader(test, batch_size=8)

torch.cuda.empty_cache()

total_acc_train = 0
total_loss_train = 0
total_recall_train = 0
full_output = []

tp = 0
fn = 0
fp = 0

for train_input, train_label in test_dataloader:

    train_label = train_label.to(device)
    mask = train_input['attention_mask'].to(device)
    input_id = train_input['input_ids'].squeeze(1).to(device)

    output, attentions = model(input_id, mask)
    full_output.append(output[:].detach().cpu().numpy())

    acc = (output.argmax(dim=1) == train_label).sum().item()
    total_acc_train += acc

    for ind, out in enumerate(output.argmax(dim=1)):
        if out == 1 and train_label[ind] == 1:
            tp += 1
        elif out == 0 and train_label[ind] == 1:
            fn += 1
        elif train_label[ind] == 0 and out == 1:
            fp += 1

all_logits = []
for array in full_output:
    all_logits.append(list(np.argmax(array, axis=1)[:]))

all_logits = list(itertools.chain.from_iterable(all_logits))

true_vals = pd.get_dummies(test_data['Decision']).values
true_vals = list(np.argmax(true_vals, axis=1))

metric = BinaryRecall()
metric(torch.tensor(all_logits), torch.tensor(true_vals))

metric = BinaryPrecision()
metric(torch.tensor(all_logits), torch.tensor(true_vals))

metric = BinaryF1Score()
metric(torch.tensor(all_logits), torch.tensor(true_vals))

metric = BinaryCohenKappa()
cohen = metric(torch.tensor(all_logits), torch.tensor(true_vals))

# cohen


metric = BinaryAUROC()
auc = metric(torch.tensor(all_logits), torch.tensor(true_vals))


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


wss95(true_vals, all_logits)
