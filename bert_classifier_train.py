import gc
import os
import sys
import time
import torch
import torchmetrics
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryAUROC, BinaryFBetaScore

import reader

from datetime import datetime
from math import floor
from torch import nn
from torch.optim import AdamW, lr_scheduler
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


def train(model_name, train_path, val_path, learning_rate, epochs, batch_size, dropout, pos_weight, gamma, step_size):
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

    # valid_log = open(os.path.join(save_path, "valid_log.txt"), 'w')
    valid_result = []

    current_model = model_options[model_name][0]
    hidden_layer = model_options[model_name][1]

    print("Get model")
    # torch.manual_seed(5223)
    model = Bert(hidden=hidden_layer, model_type=current_model, dropout=dropout, sigma=False)

    print("Retrieving data")
    tokenizer = AutoTokenizer.from_pretrained(current_model)
    train_dataloader = reader.load(train_path, tokenizer, batch_size)
    val_dataloader = reader.load(val_path, tokenizer, batch_size)

    print("Building optimizer")
    # loss_weights = torch.Tensor([1., 17.])  # pick the weights
    # criterion = nn.CrossEntropyLoss(weight=loss_weights)
    pos_weight = torch.tensor([pos_weight])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0005)

    acc = BinaryAccuracy(threshold=0.5)
    acc_3 = BinaryAccuracy(threshold=0.3)
    acc_1 = BinaryAccuracy(threshold=0.2)
    precision = BinaryPrecision(threshold=0.5)
    precision_3 = BinaryPrecision(threshold=0.3)
    precision_1 = BinaryPrecision(threshold=0.2)
    recall = BinaryRecall(threshold=0.5)
    recall_1 = BinaryRecall(threshold=0.2)
    recall_3 = BinaryRecall(threshold=0.3)
    auroc = BinaryAUROC(thresholds=10)
    fB = BinaryFBetaScore(beta=2., threshold=0.5)
    fB_3 = BinaryFBetaScore(beta=2., threshold=0.3)
    fB_1 = BinaryFBetaScore(beta=2., threshold=0.2)

    # num_training_steps = epochs * len(train_dataloader)
    lr_schedule = lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()
        acc = acc.cuda()
        acc_3 = acc_3.cuda()
        acc_1 = acc_1.cuda()
        precision = precision.cuda()
        precision_3 = precision_3.cuda()
        precision_1 = precision_1.cuda()
        recall = recall.cuda()
        recall_1 = recall_1.cuda()
        recall_3 = recall_3.cuda()
        auroc = auroc.cuda()
        fB = fB.cuda()
        fB_3 = fB_3.cuda()
        fB_1 = fB_1.cuda()
        print("Using Cuda")

    print("Training")
    length = len(train_dataloader)
    for epoch_num in range(1, epochs + 1):
        # if epoch_num > decayepoch:
        #     learning_rate = learning_rate * gamma
        #     optimizer
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
            # output, .. = model(input_id, mask)

            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()

            # result = output.argmax(dim=1).unsqueeze(-1)

            acc(output, train_label)
            # acc_3(output, train_label)
            # acc_1(output, train_label)
            precision(output, train_label)
            # precision_3(output, train_label)
            # precision_1(output, train_label)
            recall(output, train_label)
            # recall_1(output, train_label)
            # recall_3(output, train_label)
            # auroc(output, train_label)
            fB(output, train_label)
            # fB_3(output, train_label)
            # fB_1(output, train_label)

            batch_loss.backward()

            # if i % 2 == 0:
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

            # if i >= 5:  # debug
            #     break

        train_acc = acc.compute()
        acc.reset()
        # train_acc3 = acc_3.compute()
        # acc_3.reset()
        # train_acc1 = acc_1.compute()
        # acc_1.reset()

        train_precision = precision.compute()
        precision.reset()
        # train_precision3 = precision_3.compute()
        # precision_3.reset()
        # train_precision1 = precision_1.compute()
        # precision_1.reset()

        train_recall = recall.compute()
        recall.reset()
        # train_recall1 = recall_1.compute()
        # recall_1.reset()
        # train_recall3 = recall_3.compute()
        # recall_3.reset()

        # train_auroc = auroc.compute()
        # auroc.reset()

        # train_fB = fB.compute()
        # fB.reset()
        # train_fB3 = fB_3.compute()
        # fB_3.reset()
        # train_fB1 = fB_1.compute()
        # fB_1.reset()

        total_loss_val = 0
        print("Validating")
        model.eval()
        i = 0
        with torch.no_grad():
            for val_input, val_label in val_dataloader:
                i += 1
                val_label = val_label.float().unsqueeze(-1).to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output, attentions = model(input_id, mask)

                val_loss = criterion(output, val_label)
                total_loss_val += val_loss.item()

                acc(output, val_label)
                acc_3(output, val_label)
                acc_1(output, val_label)
                precision(output, val_label)
                precision_3(output, val_label)
                precision_1(output, val_label)
                recall(output, val_label)
                recall_1(output, val_label)
                recall_3(output, val_label)
                auroc(output, val_label)
                fB(output, val_label)
                fB_3(output, val_label)
                fB_1(output, val_label)

                # result = output.argmax(dim=1).unsqueeze(-1)

                # batch_acc = acc(output, val_label)
                # batch_precision = precision(output, val_label)
                # batch_recall = recall(output, val_label)

                sys.stdout.flush()
                gc.collect()

                # if i >= 5:  # debug
                #     break

        lr_schedule.step()
        # if epoch_num == step_size:
        #     for g in optimizer.param_groups:
        #         g['lr'] = learning_rate * gamma

        learning_rate = optimizer.param_groups[0]["lr"]

        val_acc = acc.compute()
        acc.reset()
        val_acc3 = acc_3.compute()
        acc_3.reset()
        val_acc1 = acc_1.compute()
        acc_1.reset()

        val_precision = precision.compute()
        precision.reset()
        val_precision3 = precision_3.compute()
        precision_3.reset()
        val_precision1 = precision_1.compute()
        precision_1.reset()

        val_recall = recall.compute()
        recall.reset()
        val_recall1 = recall_1.compute()
        recall_1.reset()
        val_recall3 = recall_3.compute()
        recall_3.reset()

        val_auroc = auroc.compute()
        auroc.reset()

        val_fB = fB.compute()
        fB.reset()
        val_fB3 = fB_3.compute()
        fB_3.reset()
        val_fB1 = fB_1.compute()
        fB_1.reset()

        train_log = f"EPOCH {epoch_num} TRAIN avloss: {(total_loss_train / len(train_dataloader)):.6f} Acc: {train_acc:.6f} Recall: {train_recall:.4f} Precision: {train_precision:.4f}"
        val_log = f"EPOCH {epoch_num} VALID avloss: {(total_loss_val / len(val_dataloader)):.6f} \n" \
                  f"Acc5: {val_acc:.6f} Recall5: {val_recall:.4f} Precision5: {val_precision:.4f} Fbeta5: {val_fB}\n" \
                  f"Acc3: {val_acc3:.6f} Recall3: {val_recall3:.4f} Precision3: {val_precision3:.4f} Fbeta3: {val_fB3}\n" \
                  f"Acc2: {val_acc1:.6f} Recall2: {val_recall1:.4f} Precision2: {val_precision1:.4f} Fbeta2: {val_fB1}\n auroc: {val_auroc}"

        epoch_log.write(train_log + "\n")
        epoch_log.write(val_log)
        epoch_log.close()

        summary_log.write(f"{train_log}\t")
        summary_log.write(f"{val_log}\n")

        # valid_log.write(f"{epoch_num} {total_loss_val / len(val_dataloader)}")
        valid_result.append(total_loss_val / len(val_dataloader))

        print(train_log)
        print(val_log)

        model_path = os.path.join(save_path, f"{model_name}_{version}_epoch_{epoch_num}.pt")

        torch.save(model.state_dict(), model_path)
        model.load_state_dict(torch.load(model_path))

    summary_log.close()
    return valid_result, version


if __name__ == "__main__":
    train_path = os.path.join("data", "cns_balanced_new1.csv")
    val_path = os.path.join("data", "cns_val_new1.csv")

    LR = 2e-5
    EPOCHS = 10

    train('mediumbert', train_path, val_path, LR, EPOCHS, 15, 0.2)

