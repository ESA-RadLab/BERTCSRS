import re

import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader


htmlcleaner = re.compile('<.*?>')


def cleanhtml(text):
    cleantext = re.sub(htmlcleaner, '', text)
    return cleantext


class Reader(Dataset):
    """PyTorch Dataset class for our systematic review datasets.
    """

    def __init__(self, df, tokenizer, old_model):
        """Creates the dataset
              Params:
                df: dataset in a dataframe
        """

        labels_dict = {
            'Excluded': 0,
            'Included': 1,
        }

        if old_model:
            self.labels = []
            label = [0, 0]
            for decision in df['decision']:
                label[labels_dict[decision]] = 1
                self.labels.append(label)
                label = [0, 0]
        else:
            self.labels = [labels_dict[label.strip()] for label in df['decision']]

        self.texts = [tokenizer(cleanhtml(text), padding='max_length', max_length=512, truncation=True,
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


def load(path, tokenizer, batch_size, old_model=False, shuffle=True):
    data = pd.read_csv(path)
    dataset = Reader(data, tokenizer, old_model)
    dataloader = DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=0)
    return dataloader


class Reader_run(Dataset):
    """PyTorch Dataset class for our systematic review datasets.
    """

    def __init__(self, df, tokenizer):
        """Creates the dataset
              Params:
                df: dataset in a dataframe
        """

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


def load_run(path, tokenizer, batch_size, shuffle=True):
    data = pd.read_csv(path)
    dataset = Reader_run(data, tokenizer)
    dataloader = DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=0)
    return dataloader
