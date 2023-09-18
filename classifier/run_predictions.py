import nltk
import numpy as np
import pandas as pd
import torch
import itertools
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from classifier import BertClassifier2525 as Bert

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

    def __init__(self, df):
        self.texts = [tokenizer(text, padding='max_length', max_length=512, truncation=True,
                                return_tensors="pt") for text in df['titleabstract']]

    def __len__(self):
        return len(self.texts)

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)

        return batch_texts


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

model = Bert(hidden=768, model_type=current_model)

state_dict = torch.load("/content/gdrive/MyDrive/ESA/Models/biobert0.pt")
model.load_state_dict(state_dict)

model.to(device)

model.eval()

predict_data = pd.read_csv('/content/gdrive/MyDrive/ESA/data70-10-20/cns_test.csv')

test = Dataset(predict_data)
test_dataloader = torch.utils.data.DataLoader(test, batch_size=8)

torch.cuda.empty_cache()

full_output = []

for train_input in test_dataloader:
    mask = train_input['attention_mask'].to(device)
    input_id = train_input['input_ids'].squeeze(1).to(device)

    output, attentions = model(input_id, mask)
    full_output.append(output[:].detach().cpu().numpy())

all_logits = []
for array in full_output:
    all_logits.append(list(np.argmax(array, axis=1)[:]))

all_logits = list(itertools.chain.from_iterable(all_logits))

all_logits = np.array(all_logits)

all_logits.sum() / 999

# histogram of prediction probabilities
# import itertools
# out = list(itertools.chain.from_iterable(full_output))
# ones = []
# for i in out:
#   ones.append(i[1])
# plt.hist(ones)
# plt.show() 

predict_data['Prediction'] = all_logits

predict_data['Prediction'] = predict_data['Prediction'].replace(0, 'Excluded')
predict_data['Prediction'] = predict_data['Prediction'].replace(1, 'Included')

# predict_data.columns

to_save = predict_data

to_save.to_csv('cns_1k.csv')

# commandline: !cp 'cns_1k.csv' -d '/content/gdrive/MyDrive/ESA/'
