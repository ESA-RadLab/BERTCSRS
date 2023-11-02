import gc
import os
import sys
import pandas as pd
import torch
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryRecall, BinaryPrecision, BinaryFBetaScore, \
    BinaryPrecisionRecallCurve
from transformers import AutoTokenizer

import reader
from classifier import BertClassifier50 as Bert
from classifier import RobertaClassifier50 as Roberta
from classifier_old import BertClassifierOld


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


def test(bert_name, version, epoch, data_path, output_path, batch_size, old_model=False):
    current_model = model_options[bert_name][0]
    hidden_layer = model_options[bert_name][1]

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if old_model:
        model = BertClassifierOld(hidden=hidden_layer, model_type=current_model)
    elif bert_name == "roberta_pubmed":
        model = Roberta(hidden=hidden_layer, model_type=current_model)
    else:
        model = Bert(hidden=hidden_layer, model_type=current_model)

    model_path = f"models/{bert_name}/{version}/{bert_name}_{version}_epoch_{epoch}.pt"
    # model_path = "../models/Kfold/pubmed_abstract_29.08_12.14_epoch_4.pt"

    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict, strict=False)

    model.to(device)

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(current_model)
    test_dataloader = reader.load(data_path, tokenizer, batch_size, old_model, shuffle=False)

    torch.cuda.empty_cache()

    acc = BinaryAccuracy(threshold=0.5)
    acc_3 = BinaryAccuracy(threshold=0.3)
    acc_1 = BinaryAccuracy(threshold=0.2)
    precision = BinaryPrecision(threshold=0.5)
    precision_3 = BinaryPrecision(threshold=0.3)
    precision_1 = BinaryPrecision(threshold=0.2)
    recall = BinaryRecall(threshold=0.5)
    recall_1 = BinaryRecall(threshold=0.2)
    recall_3 = BinaryRecall(threshold=0.3)
    # auroc = BinaryAUROC(thresholds=10)
    fB = BinaryFBetaScore(beta=2., threshold=0.5)
    fB_3 = BinaryFBetaScore(beta=2., threshold=0.3)
    fB_1 = BinaryFBetaScore(beta=2., threshold=0.2)
    PRcurve = BinaryPrecisionRecallCurve(thresholds=10)


    if use_cuda:
        acc = acc.cuda()
        acc_3 = acc_3.cuda()
        acc_1 = acc_1.cuda()
        precision = precision.cuda()
        precision_3 = precision_3.cuda()
        precision_1 = precision_1.cuda()
        recall = recall.cuda()
        recall_1 = recall_1.cuda()
        recall_3 = recall_3.cuda()
        # auroc = auroc.cuda()
        fB = fB.cuda()
        fB_3 = fB_3.cuda()
        fB_1 = fB_1.cuda()
        PRcurve = PRcurve.cuda()

    full_output = []

    i = 0
    for test_input, test_label in test_dataloader:
        i += 1
        # test_label = test_label.float().unsqueeze(-1).to(device)
        if not old_model:
            test_label = test_label.unsqueeze(-1)
        test_label = test_label.float().to(device)

        mask = test_input['attention_mask'].to(device)
        input_id = test_input['input_ids'].squeeze(1).to(device)

        output, attentions = model(input_id, mask)
        # full_output.append(output[:].detach().cpu().numpy())

        # acc = (output.argmax(dim=1) == test_label).sum().item()
        # total_acc_train += acc

        full_output.extend(output[:, 0].detach().cpu().numpy())

        batch_acc = acc(output, test_label)
        acc_3(output, test_label)
        acc_1(output, test_label)
        batch_precision = precision(output, test_label)
        precision_3(output, test_label)
        precision_1(output, test_label)
        batch_recall = recall(output, test_label)
        recall_1(output, test_label)
        recall_3(output, test_label)
        # auroc(output, test_label)
        batch_fB = fB(output, test_label)
        fB_3(output, test_label)
        fB_1(output, test_label)

        # curve = PRcurve(output, test_label.int())

        torch.cuda.empty_cache()
        sys.stdout.flush()
        gc.collect()

        # if i >= 5:  # debug
        #     break

    test_acc = acc.compute()
    acc.reset()
    test_acc3 = acc_3.compute()
    acc_3.reset()
    test_acc1 = acc_1.compute()
    acc_1.reset()

    test_precision = precision.compute()
    precision.reset()
    test_precision3 = precision_3.compute()
    precision_3.reset()
    test_precision1 = precision_1.compute()
    precision_1.reset()

    test_recall = recall.compute()
    recall.reset()
    test_recall1 = recall_1.compute()
    recall_1.reset()
    test_recall3 = recall_3.compute()
    recall_3.reset()

    # test_auroc = auroc.compute()
    # auroc.reset()

    test_fB = fB.compute()
    fB.reset()
    test_fB3 = fB_3.compute()
    fB_3.reset()
    test_fB1 = fB_1.compute()
    fB_1.reset()

    # PRcurve.compute()
    #
    # fig_, ax_ = PRcurve.plot()

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_data = pd.read_csv(data_path)
    output_data['prediction'] = full_output
    # output_data = output_data.to_frame()
    output_data.to_csv(os.path.join(output_path, f"{bert_name}_{version}_epoch{epoch}.csv"), index=False, lineterminator="\r\n")

    print(
        f"recall:{test_recall:.4f} precision:{test_precision:.4f} fBeta:{test_fB:.4f} acc:{test_acc:.4f} recall3:{test_recall3:.4f} "
        f"precision3:{test_precision3:.4f} fBeta3:{test_fB3:.4f} acc3:{test_acc3:.4f} recall2:{test_recall1:.4f} precision2:{test_precision1:.4f} "
        f"fBeta2:{test_fB1:.4f} acc2:{test_acc1:.4f} ")
        # f"auroc:{test_auroc:.4f}")

    return test_recall.cpu().numpy(), test_precision.cpu().numpy(), test_acc.cpu().numpy(), test_fB.cpu().numpy()

    # plt.show()

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
    modelname = "pubmed_abstract"
    version = "24.10_14.20"
    epoch = 1

    data_path = os.path.join("../data", "crowd_cns.csv")
    output_path = os.path.join("../data", "output_cns.csv")
    # model_path = f"models/{modelname}/{version}/{modelname}_epoch_{epoch}.pt"

    batch_size = 10

    test(modelname, version, epoch, data_path, output_path, batch_size)
    # test(bert_name, model_path, data_path, batch_size, True)
