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
from classifier import BertClassifier


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


def test(bert_name, model_path, data_path, batch_size, old_model=False):
    current_model = model_options[bert_name][0]
    hidden_layer = model_options[bert_name][1]

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if old_model:
        model = BertClassifierOld(hidden=hidden_layer, model_type=current_model)
    else:
        model = BertClassifier(hidden=hidden_layer, model_type=current_model)

    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict, strict=False)

    model.to(device)

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(current_model)
    test_dataloader = reader.load(data_path, tokenizer, batch_size, old_model)

    torch.cuda.empty_cache()

    acc = BinaryAccuracy(threshold=0.5)
    precision = BinaryPrecision(threshold=0.5)
    precision_3 = BinaryPrecision(threshold=0.3)
    precision_1 = BinaryPrecision(threshold=0.1)
    recall = BinaryRecall(threshold=0.5)
    recall_1 = BinaryRecall(threshold=0.1)
    recall_3 = BinaryRecall(threshold=0.3)
    auroc = BinaryAUROC(thresholds=5)
    fB = BinaryFBetaScore(beta=2)
    cohen = BinaryCohenKappa()

    if use_cuda:
        acc = acc.cuda()
        precision = precision.cuda()
        precision_3 = precision_3.cuda()
        precision_1 = precision_1.cuda()
        recall = recall.cuda()
        recall_1 = recall_1.cuda()
        recall_3 = recall_3.cuda()
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
        precision_3(output, train_label)
        precision_1(output, train_label)
        batch_recall = recall(output, train_label)
        recall_1(output, train_label)
        recall_3(output, train_label)
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
    test_precision_3 = precision_3.compute()
    precision_3.reset()
    test_precision_1 = precision_1.compute()
    precision_1.reset()

    test_recall = recall.compute()
    recall.reset()
    test_recall1 = recall_1.compute()
    recall_1.reset()
    test_recall3 = recall_3.compute()
    recall_3.reset()

    test_auroc = auroc.compute()
    auroc.reset()

    test_fB = fB.compute()
    fB.reset()

    test_cohen = cohen.compute()
    cohen.reset()

    print(f"acc:{test_acc:.4f} precision:{test_precision:.4f} recall:{test_recall:.4f} precision3:{test_precision_3:.4f} recall3:{test_recall3:.4f} precision1:{test_precision_1:.4f} recall1:{test_recall1:.4f} " 
          f"auroc:{test_auroc:.4f} fBeta:{test_fB:.4f}")


# def wss(R, y_true, y_pred):
#     cfmat = confusion_matrix(y_true, y_pred)
#     tn_, fp_, fn_, tp_ = cfmat.ravel()  # instead of doing a call for each
#     N = np.sum(cfmat)
#     if N <= 0:
#         print("N = {}!!!".format(N))
#     return (tn_ + fn_) / N - (1 - R)
#
#
# def wss95(y_true, y_pred):
#     return wss(0.95, y_true, y_pred)


# wss95(true_vals, all_logits)
if __name__ == "__main__":
    data_path = "data/sex_diff_test.csv"
    # model_path = "models/pubmed_abstract/25.07_14.06/pubmed_abstract_epoch_6.pt"
    # model_path = "models/pubmed_abstract/24.07_13.27/pubmed_abstract_epoch_9_13.37.33.pt"
    model_path = "models/Original/sex_diff.pt"
    bert_name = "pubmed_abstract"  # sex_diff: pubmed_abstract cns: biobert
    batch_size = 10

    test(bert_name, model_path, data_path, batch_size, True)
    # test(bert_name, model_path, data_path, batch_size)
