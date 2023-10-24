import re

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from torchtext.vocab import vocab
import torchtext as tt
import collections
import torch

from torch.utils.data import Dataset, DataLoader

nltk.download('wordnet')


class Reader(Dataset):
    """PyTorch Dataset class for our systematic review datasets.
    """

    def __init__(self, df, tokenizer, old_model, vocab=None):
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
            self.labels = [labels_dict[label] for label in df['decision']]

        self.texts = [tokenizer(text, padding='max_length', max_length=512, truncation=True,
                                return_tensors="pt") for text in df['titleabstract']]
        cleaned_texts = [self.clean_text(text) for text in df['titleabstract']]

        if vocab is None:
            self.vocab = self.get_vocab(cleaned_texts)
        else:
            self.vocab = vocab

        encoded_texts = [self.encode_text(self.vocab, text) for text in cleaned_texts]
        self.encoded_texts = np.array(encoded_texts)

    def clean_text(self, text):
        stripped = re.sub(r'[^\w\s]', '', text).lower()
        stop_words = stopwords.words('english')
        token = word_tokenize(stripped)
        cleaned_token = []
        for word in token:
            if word not in stop_words:
                cleaned_token.append(word)

        lemmatizer = WordNetLemmatizer()
        lemmatized = [lemmatizer.lemmatize(word) for word in cleaned_token]

        if len(lemmatized) > 250:
            result = lemmatized[:250]
        else:
            padding = ["<pad>"] * (250 - len(lemmatized))
            result = lemmatized + padding

        return result

    def get_vocab(self, training_corpus):
        # add special characters
        # padding, end of line, unknown term
        counter_obj = collections.Counter()

        # build vocab from training corpus
        for item in training_corpus:  # apply preprocessing on each text
            counter_obj.update(item)

        result = vocab(counter_obj, min_freq=5, specials=["<pad>", "<unk>"])
        result.set_default_index(result['<unk>'])
        return result

    def encode_text(self, vocab, text):
        encoded_text = []
        for word in text:
            encoded_text.append(vocab[word])
        return encoded_text

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

    def get_batch_encoded_texts(self, idx):
        # Fetch a batch of inputs
        return self.encoded_texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        batch_encoded_texts = self.get_batch_encoded_texts(idx)
        return batch_texts, batch_y, batch_encoded_texts


def load(path, tokenizer, batch_size, old_model=False, shuffle=True, vocab=None):
    data = pd.read_csv(path)
    dataset = Reader(data, tokenizer, old_model, vocab)
    dataloader = DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=0)
    return dataloader, dataset.vocab
