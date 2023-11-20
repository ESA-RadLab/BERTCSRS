import gc
import os
import sys
import torch
import reader
import pandas as pd
from transformers import AutoTokenizer
from classifier import BertClassifier25 as Bert

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


def run(bert_name, version, epoch, model_path, data_path, output_path, batch_size):
    current_model = model_options[bert_name][0]
    n_hidden_layer = model_options[bert_name][1]

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model_name = f"{bert_name}_{version}_epoch_{epoch}"

    print(f"Get model {model_name}")
    model = Bert(hidden=n_hidden_layer, model_type=current_model)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    print("Get tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(current_model)

    print("Retrieving data")
    dataloader = reader.load_run(data_path, tokenizer, batch_size, shuffle=False)

    full_output = []
    i = 0
    length = len(dataloader)

    print("Start classification")
    for input in dataloader:
        i += 1

        mask = input['attention_mask'].to(device)
        input_id = input['input_ids'].squeeze(1).to(device)

        output, attentions = model(input_id, mask)

        full_output.append(output[:, 0].detach().cpu().numpy())

        torch.cuda.empty_cache()
        sys.stdout.flush()
        gc.collect()

        print(f"Batch: {i}/{length}")

        # if i >= 5:  # debug
        #     break

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_data = pd.read_csv(data_path)
    output_data['prediction'] = full_output
    output_data.to_csv(os.path.join(output_path, f"{bert_name}_{version}_epoch{epoch}.csv"), index=False,
                       lineterminator="\r\n")


if __name__ == "__main__":
    modelname = "pubmed_abstract"
    version = "08.11_10.32"
    epoch = 6
    batch_size = 10

    data_path = os.path.join("../data", "all_ref_SD.csv")
    output_folder = os.path.join("../data", "FULL", "SD")
    model_path = f"models/{modelname}/{version}/{modelname}_epoch_{epoch}.pt"
    # model_path = "models/Kfold/Final/CNS_pubmed_abstract_02.11_11.59_epoch_9.pt"

    run(modelname, version, epoch, data_path, model_path, output_folder, batch_size)
