import gc
import os
import sys
import reader
import torch
import pandas as pd
from torchmetrics.classification import BinaryAccuracy, BinaryRecall, BinaryPrecision, BinaryFBetaScore
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
    "mediumbert": ["prajjwal1/bert-medium", 512],
    "roberta_pubmed": ["raynardj/roberta-pubmed", 768]
}


def test(bert_name, data_path, output_path, model_path, batch_size):
    current_model = model_options[bert_name][0]
    hidden_layer = model_options[bert_name][1]

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    print(f"Load model {model_path[-1][:-3]}")
    model = Bert(hidden=hidden_layer, model_type=current_model)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    print("Retrieving data")
    tokenizer = AutoTokenizer.from_pretrained(current_model)
    test_dataloader = reader.load(data_path, tokenizer, batch_size, shuffle=False)

    torch.cuda.empty_cache()

    acc = BinaryAccuracy(threshold=0.5)
    acc_3 = BinaryAccuracy(threshold=0.3)
    acc_2 = BinaryAccuracy(threshold=0.2)
    precision = BinaryPrecision(threshold=0.5)
    precision_3 = BinaryPrecision(threshold=0.3)
    precision_2 = BinaryPrecision(threshold=0.2)
    recall = BinaryRecall(threshold=0.5)
    recall_3 = BinaryRecall(threshold=0.3)
    recall_2 = BinaryRecall(threshold=0.2)
    fB = BinaryFBetaScore(beta=2., threshold=0.5)
    fB_3 = BinaryFBetaScore(beta=2., threshold=0.3)
    fB_2 = BinaryFBetaScore(beta=2., threshold=0.2)

    if use_cuda:
        acc = acc.cuda()
        acc_3 = acc_3.cuda()
        acc_2 = acc_2.cuda()
        precision = precision.cuda()
        precision_3 = precision_3.cuda()
        precision_2 = precision_2.cuda()
        recall = recall.cuda()
        recall_3 = recall_3.cuda()
        recall_2 = recall_2.cuda()
        fB = fB.cuda()
        fB_3 = fB_3.cuda()
        fB_2 = fB_2.cuda()

    full_output = []

    i = 0
    print("Start testing")
    for test_input, test_label in test_dataloader:
        i += 1
        test_label = test_label.unsqueeze(-1)
        test_label = test_label.float().to(device)

        mask = test_input['attention_mask'].to(device)
        input_id = test_input['input_ids'].squeeze(1).to(device)

        output = model(input_id, mask)

        full_output.extend(output[:, 0].detach().cpu().numpy())

        acc(output, test_label)
        acc_3(output, test_label)
        acc_2(output, test_label)
        precision(output, test_label)
        precision_3(output, test_label)
        precision_2(output, test_label)
        recall(output, test_label)
        recall_3(output, test_label)
        recall_2(output, test_label)
        fB(output, test_label)
        fB_3(output, test_label)
        fB_2(output, test_label)

        torch.cuda.empty_cache()
        sys.stdout.flush()
        gc.collect()

        # if i >= 5:  # debug
        #     break

    test_acc = acc.compute()
    acc.reset()
    test_acc3 = acc_3.compute()
    acc_3.reset()
    test_acc2 = acc_2.compute()
    acc_2.reset()

    test_precision = precision.compute()
    precision.reset()
    test_precision3 = precision_3.compute()
    precision_3.reset()
    test_precision2 = precision_2.compute()
    precision_2.reset()

    test_recall = recall.compute()
    recall.reset()
    test_recall3 = recall_3.compute()
    recall_3.reset()
    test_recall2 = recall_2.compute()
    recall_2.reset()

    test_fB = fB.compute()
    fB.reset()
    test_fB3 = fB_3.compute()
    fB_3.reset()
    test_fB2 = fB_2.compute()
    fB_2.reset()

    output_folder = os.path.join(*output_path.split("/")[:-1])

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_data = pd.read_csv(data_path)
    output_data['prediction'] = full_output
    output_data.to_csv(os.path.join(output_path), index=False, lineterminator="\r\n")

    print(
        f"recall:{test_recall:.4f} precision:{test_precision:.4f} fBeta:{test_fB:.4f} acc:{test_acc:.4f} recall3:{test_recall3:.4f} "
        f"precision3:{test_precision3:.4f} fBeta3:{test_fB3:.4f} acc3:{test_acc3:.4f} recall2:{test_recall2:.4f} precision2:{test_precision2:.4f}"
        f"fBeta2:{test_fB2:.4f} acc2:{test_acc2:.4f} ")

    return test_recall.cpu().numpy(), test_precision.cpu().numpy(), test_acc.cpu().numpy(), test_fB.cpu().numpy()


if __name__ == "__main__":
    modelname = "pubmed_abstract"
    version = "20.11_10.34"
    epoch = 15
    # model_path = f"models/{modelname}/{version}/{modelname}_{version}_epoch_{epoch}.pt"

    data_path = "Kfolds/data/CNS/Final/fold_2/cns_val_raw.csv"
    model_path = "models/pubmed_abstract_08.11_10.32_epoch_6.pt"
    output_path = "Kfolds/output/CNS/20.11_09.55/fold_2/pubmed_abstract_08.11_10.32_epoch_6_val.csv"

    batch_size = 10

    test(modelname, data_path, output_path, model_path, batch_size)
