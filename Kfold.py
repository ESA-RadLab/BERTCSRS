import os
import pandas as pd
import torch
import xlwt as xlwt
from openpyxl.reader.excel import load_workbook
from tqdm import tqdm

import bert_classifier_train
from evaluation import evaluate_output, evaluate_classifier, compare_output

bert = 'pubmed_abstract'

fold_path = "Kfolds"
folds = os.listdir(fold_path)
folds.sort()

# book = xlwt.Workbook(encoding="utf-8")
#
# sheet = book.add_sheet(bert)

best_recall_list = []
best_precision_list = []
best_threshold_list = []
valid_result_list = []
version_list = []
fp_list = []
fn_list = []

for fold in tqdm(folds):
    train_path = os.path.join(fold_path, fold, "cns_balanced_raw.csv")
    val_path = os.path.join(fold_path, fold, "cns_val_raw.csv")

    LR = 2e-5
    EPOCHS = 8  # debug
    batch_size = 2  # debug
    dropout = 0.2
    step_size = 5
    gamma = 1
    pos_weight = 10

    valid_result, version = bert_classifier_train.train(bert, train_path, val_path, LR, EPOCHS, batch_size, dropout,
                                                        pos_weight, gamma, step_size)

    valid_result_list.append(valid_result)
    version_list.append(version)

    best_epoch = valid_result.index(min(valid_result)) + 1
    print(f"Best epoch: {best_epoch}")

    data_path = os.path.join(fold_path, fold, "cns_fulltest_raw.csv")

    batch_size = 2  # debug

    evaluate_classifier.test(bert, version, best_epoch, data_path, batch_size)

    output_path = "..\\output"
    precision_list, recall_list, threshold_list = evaluate_output.evaluate(bert, version, best_epoch, output_path)

    best_threshold = 0.5
    best_precision = 0
    best_recall = 0

    for i, precision in enumerate(precision_list):
        recall = recall_list[i]
        threshold = threshold_list[i]
        if precision > 0.4 and precision >= best_precision:
            if recall >= best_recall:
                best_recall = recall
                best_precision = precision
                best_threshold = threshold

    print(f"best_recall: {best_recall} best_precision: {best_precision} best_threshold: {best_threshold}")

    best_threshold_list.append(best_threshold)
    best_recall_list.append(best_recall)
    best_precision_list.append(best_precision)

    true_pos, true_neg, false_pos, false_neg = compare_output.compare(best_threshold, bert, version, best_epoch, output_path, fold)

    fp_list.append(false_pos)
    fn_list.append(false_neg)

    # break  # debug

# folds = folds[0]  # debug

Kfold_results = pd.DataFrame({"Fold": folds, "Version": version_list, "Recall": best_recall_list, "Precision": best_precision_list,
                              "Threshold": best_threshold_list, "False Neg": fn_list, "False Pos": fp_list, "Val loss": valid_result_list})

with pd.ExcelWriter("path_to_file.xlsx", if_sheet_exists='overlay') as writer:
    Kfold_results.to_excel(writer, sheet_name=bert)
