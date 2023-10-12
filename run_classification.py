import gc
import os
import sys

import nltk
import pandas as pd
import torch
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryRecall, BinaryPrecision, BinaryFBetaScore, \
    BinaryPrecisionRecallCurve
from transformers import AutoTokenizer

import reader
from classifier import BertClassifier25 as Bert
from classifier_old import BertClassifierOld

nltk.download('stopwords')

model_options = {
    "biobert": ["dmis-lab/biobert-v1.1", 768],
    "pubmed_abstract": ["microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract", 768],
    "scibert": ["allenai/scibert_scivocab_uncased", 768],
    "pubmed_fulltext": ["microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", 768],
    "medbert": ["Charangan/MedBERT", 768],
    "basebert": ["bert-base-uncased", 768],
    "tinybert": ["prajjwal1/bert-tiny", 128],
    "minibert": ["prajjwal1/bert-mini", 256],
    "smallbert": ["prajjwal1/bert-small", 512],
    "mediumbert": ["prajjwal1/bert-medium", 512]
}


def test(bert_name, version, epoch, data_path, output_path, batch_size, old_model=False):
    current_model = model_options[bert_name][0]
    N_hidden_layer = model_options[bert_name][1]

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if old_model:
        model = BertClassifierOld(hidden=N_hidden_layer, model_type=current_model)
    else:
        model = Bert(hidden=N_hidden_layer, model_type=current_model)

    # model_path = f"models/{bert_name}/{version}/{bert_name}_{version}_epoch_{epoch}.pt"
    model_path = "models/Kfold/pubmed_abstract_29.08_11.24_epoch_7.pt"

    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict, strict=False)

    model = model.to(device)

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(current_model)
    test_dataloader = reader.load_run(data_path, tokenizer, batch_size, shuffle=False)

    torch.cuda.empty_cache()

    full_output = []

    i = 0
    for test_input in test_dataloader:
        i += 1

        mask = test_input['attention_mask'].to(device)
        input_id = test_input['input_ids'].squeeze(1).to(device)

        output, attentions = model(input_id, mask)

        full_output.extend(output[:, 0].detach().cpu().numpy())

        torch.cuda.empty_cache()
        sys.stdout.flush()
        gc.collect()

        # if i >= 5:  # debug
        #     break
        print(f"Batch: {i}")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_data = pd.read_csv(data_path)
    output_data['prediction'] = full_output
    output_data.to_csv(os.path.join(output_path, f"{bert_name}_{version}_epoch{epoch}.csv"), index=False, lineterminator="\r\n")


if __name__ == "__main__":
    modelname = "pubmed_abstract"
    version = "29.08_11.24"
    epoch = 7

    data_path = os.path.join("data", "FULL_cns.csv")
    output_folder_path = os.path.join("data", "FULL")
    # model_path = f"models/{modelname}/{version}/{modelname}_epoch_{epoch}.pt"

    batch_size = 8

    test(modelname, version, epoch, data_path, output_folder_path, batch_size)
    # test(bert_name, model_path, data_path, batch_size, True)