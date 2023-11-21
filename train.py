import gc
import math
import os
import sys
import time
import torch
import reader

from datetime import datetime
from math import floor
from torch import nn
from torch.optim import RAdam
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryFBetaScore, BinaryAUROC
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


def train(model_name, train_path, val_path, learning_rate, epochs, batch_size, dropout, pos_weight):
    """ Function to train the model.
        Params:
          - model: the model to be trained
          - train_data: training data (Pandas DataFrame format)
          - val_data: validation data (Pandas DataFrame format)
          - learning_rate: learning rate
          - epochs: the number of epochs for training
    """
    start_time = datetime.now()
    version = start_time.strftime("%d.%m_%H.%M")
    save_path = os.path.join("models", model_name, version)
    log_path = os.path.join(save_path, "logs")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    summary_log = open(os.path.join(save_path, "#summary.txt"), 'w')
    summary_log.write(f"batch_size: {batch_size} \nepochs: {epochs} \ndata: {train_path} \n \n")

    valid_result = []
    Fbeta_result = []
    auroc_result = []
    recall_result = []

    current_model = model_options[model_name][0]
    hidden_layer = model_options[model_name][1]

    print(f"Get model {model_name}")
    model = Bert(hidden=hidden_layer, model_type=current_model, dropout=dropout, sigma=False)

    print("Retrieving data")
    tokenizer = AutoTokenizer.from_pretrained(current_model)
    train_dataloader = reader.load(train_path, tokenizer, batch_size)
    val_dataloader = reader.load(val_path, tokenizer, batch_size)

    print("Building optimizer")
    pos_weight = torch.tensor([pos_weight])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = RAdam(model.parameters(), lr=learning_rate)

    acc = BinaryAccuracy(threshold=0.5)
    acc_3 = BinaryAccuracy(threshold=0.3)
    acc_2 = BinaryAccuracy(threshold=0.2)
    precision = BinaryPrecision(threshold=0.5)
    precision_3 = BinaryPrecision(threshold=0.3)
    precision_2 = BinaryPrecision(threshold=0.2)
    recall = BinaryRecall(threshold=0.5)
    recall_3 = BinaryRecall(threshold=0.3)
    recall_2 = BinaryRecall(threshold=0.2)
    fB = BinaryFBetaScore(beta=4., threshold=0.5)
    fB_3 = BinaryFBetaScore(beta=4., threshold=0.3)
    fB_2 = BinaryFBetaScore(beta=4., threshold=0.2)
    auroc = BinaryAUROC(thresholds=100)  ## binned due to memory usage during training

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()
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
        auroc = auroc.cuda()
        print("Using Cuda")

    print("Training")
    length = len(train_dataloader)
    lowest_val_loss = math.inf
    counter = 0
    for epoch_num in range(1, epochs + 1):
        model.train()

        total_loss_train = 0

        time_begin = time.time()
        current_time = datetime.now()
        current_time = current_time.strftime("%H:%M:%S")
        print(f"epoch: {epoch_num} time: {current_time}")
        i = 0
        epoch_log = open(os.path.join(log_path, "epoch_log " + str(epoch_num) + ".txt"), 'w')

        for train_input, train_label in train_dataloader:
            i += 1

            train_label = train_label.float().unsqueeze(-1).to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            model.zero_grad()
            optimizer.zero_grad()

            output, attentions = model(input_id, mask)

            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()

            acc(output, train_label)
            precision(output, train_label)
            recall(output, train_label)
            fB(output, train_label)

            batch_loss.backward()
            optimizer.step()

            torch.cuda.empty_cache()
            sys.stdout.flush()
            gc.collect()

            step_time = time.time()
            elapsed_time = step_time - time_begin
            batch_time = elapsed_time / i
            remaining_time = batch_time * (length - i)

            remaining_hours = remaining_time // 3600
            remaining_minutes = floor(((remaining_time / 3600) - remaining_hours) * 60)
            remaining_seconds = floor(((remaining_time / 60) - floor(remaining_time / 60)) * 60)

            elapsed_hours = elapsed_time // 3600
            elapsed_minutes = floor(((elapsed_time / 3600) - elapsed_hours) * 60)
            elapsed_seconds = floor(((elapsed_time / 60) - floor(elapsed_time / 60)) * 60)

            log = f"[{epoch_num}/{epochs}]: [{i}/{length}] loss:{batch_loss:.5f} " \
                  f"lr:{learning_rate:.6f} elapsed time: {elapsed_hours:.0f}:{elapsed_minutes:.0f}:{elapsed_seconds:.0f} " \
                  f"time remaining: {remaining_hours:.0f}:{remaining_minutes:.0f}:{remaining_seconds:.0f}"

            epoch_log.write(log + "\n")
            if i % 10 == 0:
                print(log)

            # if i >= 2:  # debug
            #     break

        train_acc = acc.compute()
        acc.reset()

        train_precision = precision.compute()
        precision.reset()

        train_recall = recall.compute()
        recall.reset()

        total_loss_val = 0
        print("Validating")
        model.eval()
        i = 0

        with torch.no_grad():
            for val_input, val_label in val_dataloader:
                i += 1
                val_label_int = val_label.to(device)
                val_label = val_label.float().unsqueeze(-1).to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output, attentions = model(input_id, mask)

                val_loss = criterion(output, val_label)
                total_loss_val += val_loss.item()

                acc(output, val_label)
                acc_3(output, val_label)
                acc_2(output, val_label)
                precision(output, val_label)
                precision_3(output, val_label)
                precision_2(output, val_label)
                recall(output, val_label)
                recall_2(output, val_label)
                recall_3(output, val_label)
                fB(output, val_label)
                fB_3(output, val_label)
                fB_2(output, val_label)
                auroc(output, val_label_int)

                sys.stdout.flush()
                gc.collect()

                # if i >= 2:  # debug
                #     break

        learning_rate = optimizer.param_groups[0]["lr"]

        val_acc = acc.compute()
        acc.reset()
        val_acc3 = acc_3.compute()
        acc_3.reset()
        val_acc1 = acc_2.compute()
        acc_2.reset()

        val_precision = precision.compute()
        precision.reset()
        val_precision3 = precision_3.compute()
        precision_3.reset()
        val_precision1 = precision_2.compute()
        precision_2.reset()

        val_recall = recall.compute()
        recall.reset()
        val_recall1 = recall_2.compute()
        recall_2.reset()
        val_recall3 = recall_3.compute()
        recall_3.reset()

        val_fB = fB.compute()
        fB.reset()
        val_fB3 = fB_3.compute()
        fB_3.reset()
        val_fB1 = fB_2.compute()
        fB_2.reset()

        val_auroc = auroc.compute()
        auroc.reset()

        avg_val_loss = total_loss_val / len(val_dataloader)
        avg_train_loss = total_loss_train / len(train_dataloader)

        train_log = f"EPOCH {epoch_num} TRAIN avloss: {avg_train_loss:.6f} Acc: {train_acc:.6f} Recall: {train_recall:.4f} Precision: {train_precision:.4f}"
        val_log = f"EPOCH {epoch_num} VALID avloss: {avg_val_loss:.6f} val_auroc: {val_auroc:.4f}\n" \
                  f"Acc5: {val_acc:.6f} Recall5: {val_recall:.4f} Precision5: {val_precision:.4f} Fbeta5: {val_fB}\n" \
                  f"Acc3: {val_acc3:.6f} Recall3: {val_recall3:.4f} Precision3: {val_precision3:.4f} Fbeta3: {val_fB3}\n" \
                  f"Acc2: {val_acc1:.6f} Recall2: {val_recall1:.4f} Precision2: {val_precision1:.4f} Fbeta2: {val_fB1}\n"

        epoch_log.write(train_log + "\n")
        epoch_log.write(val_log)
        epoch_log.close()

        summary_log.write(f"{train_log}\t")
        summary_log.write(f"{val_log}\n")

        print(train_log)
        print(val_log)

        valid_result.append(avg_val_loss)
        Fbeta_result.append(val_fB)
        auroc_result.append(val_auroc)
        recall_result.append(val_recall)

        model_path = os.path.join(save_path, f"{model_name}_{version}_epoch_{epoch_num}.pt")

        torch.save(model.state_dict(), model_path)
        model.load_state_dict(torch.load(model_path))

        if avg_val_loss > lowest_val_loss:
            counter += 1
            if counter == 3:
                print("Early stop")
                break
        else:
            lowest_val_loss = avg_val_loss
            counter = 0

    del model, tokenizer
    torch.cuda.empty_cache()

    summary_log.close()
    return valid_result, Fbeta_result, recall_result, auroc_result, version


if __name__ == "__main__":
    train_path = os.path.join("data", "cns_balanced_new1.csv")
    val_path = os.path.join("data", "cns_val_new1.csv")

    LR = 2e-5
    EPOCHS = 15
    batch_size = 15

    train('minibert', train_path, val_path, LR, EPOCHS, batch_size, 0.2, 10)

