import gc
import sys
import nltk
import numpy as np
import torch
import reader

from transformers import AutoTokenizer
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryRecall, BinaryPrecision, BinaryF1Score, BinaryCohenKappa, BinaryFBetaScore
from sklearn.metrics import confusion_matrix
from classifier_old import BertClassifierOld
from classifier import BertClassifier25 as bert


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


def test(bert_name, model_path, data_path, batch_size, threshold, old_model=False):
    current_model = model_options[bert_name][0]
    hidden_layer = model_options[bert_name][1]

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if old_model:
        model = BertClassifierOld(hidden=hidden_layer, model_type=current_model)
    else:
        model = bert(hidden=hidden_layer, model_type=current_model)

    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict, strict=False)

    model.to(device)

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(current_model)
    test_dataloader = reader.load(data_path, tokenizer, batch_size, old_model)

    torch.cuda.empty_cache()

    acc = BinaryAccuracy(threshold=threshold)
    precision = BinaryPrecision(threshold=threshold)
    recall = BinaryRecall(threshold=threshold)
    auroc = BinaryAUROC(thresholds=10)
    fB = BinaryFBetaScore(beta=2., threshold=threshold)

    cohen = BinaryCohenKappa()

    if use_cuda:
        acc = acc.cuda()
        precision = precision.cuda()
        recall = recall.cuda()
        auroc = auroc.cuda()
        fB = fB.cuda()
        cohen = cohen.cuda()

    for train_input, train_label in test_dataloader:
        # train_label = train_label.float().unsqueeze(-1).to(device)
        if not old_model:
            train_label = train_label.unsqueeze(-1)
        train_label = train_label.float().to(device)

        mask = train_input['attention_mask'].to(device)
        input_id = train_input['input_ids'].squeeze(1).to(device)

        output, attentions = model(input_id, mask)
        # full_output.append(output[:].detach().cpu().numpy())

        # acc = (output.argmax(dim=1) == train_label).sum().item()
        # total_acc_train += acc

        batch_acc = acc(output, train_label)
        batch_precision = precision(output, train_label)
        batch_recall = recall(output, train_label)
        auroc(output, train_label)
        batch_fB = fB(output, train_label)
        batch_cohen = cohen(output, train_label)


        torch.cuda.empty_cache()
        sys.stdout.flush()
        gc.collect()

    test_acc = acc.compute()
    acc.reset()

    test_precision = precision.compute()
    precision.reset()

    test_recall = recall.compute()
    recall.reset()

    test_auroc = auroc.compute()
    auroc.reset()

    test_fB = fB.compute()
    fB.reset()

    test_cohen = cohen.compute()
    cohen.reset()

    print(f"recall:{test_recall:.4f} precision:{test_precision:.4f} fBeta:{test_fB:.4f} acc:{test_acc:.4f} "
          f"auroc:{test_auroc:.4f}")

if __name__ == "__main__":
    data_path = "data/sex_diff_test.csv"
    # model_path = "models/pubmed_abstract/25.07_14.06/pubmed_abstract_epoch_6.pt"
    # model_path = "models/pubmed_abstract/24.07_13.27/pubmed_abstract_epoch_9_13.37.33.pt"
    model_path = "models/Original/sex_diff.pt"
    bert_name = "pubmed_abstract"  # sex_diff: pubmed_abstract cns: biobert
    batch_size = 10

    test(bert_name, model_path, data_path, batch_size, True)
    # test(bert_name, model_path, data_path, batch_size)
