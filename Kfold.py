import os
import shutil
from datetime import datetime

import pandas as pd
import train
from evaluation import evaluate_output, evaluate_classifier, compare_output

# import zipfile

bert = 'roberta_pubmed'

fold_path = "Kfolds/data/CNS/divided"
folds = os.listdir(fold_path)
folds.sort()

filtered_folds = [fold for fold in folds if "fold" in fold]
folds = filtered_folds

LR = 2e-5
EPOCHS = 20
batch_size = 10
dropout = 0.2
step_size = 5
gamma = 1
pos_weight = 10

_, _, free_disk_space = shutil.disk_usage("/")
free_disk_space = free_disk_space / (2 ** 30)
required_disk_space = (len(folds) - 1) * 0.45 + EPOCHS * 0.45
if free_disk_space < required_disk_space:
    raise Exception(
        f"Not enough Disk Space! Required Disk Space: {required_disk_space} GB. Free Disk Space: {free_disk_space:.2f} GB")

best_epoch_list = []
best_recall_list = []
best_precision_list = []
best_threshold_list = []
valid_result_list = []
version_list = []
fp_list = []
fn_list = []
recall5_list = []
precision5_list = []
accuracy5_list = []
Fbeta5_list = []

start_time = datetime.now()
attempt = start_time.strftime("%d.%m_%H.%M")

for fold in folds:
    print("\n" + fold)
    train_path0 = os.path.join(fold_path, fold, "cns_balanced_raw0.csv")
    train_path1 = os.path.join(fold_path, fold, "cns_balanced_raw1.csv")
    train_path2 = os.path.join(fold_path, fold, "cns_balanced_raw2.csv")
    val_path = os.path.join(fold_path, fold, "cns_val_raw.csv")

    valid_result, Fbeta_result, recall_result, version = train.train(bert, train_path0, train_path1, train_path2, val_path, LR, EPOCHS, batch_size,
                                                                     dropout, pos_weight)

    valid_result_list.append(min(valid_result))
    version_list.append(version)

    best_valid = valid_result.index(min(valid_result))
    best_Fbeta = Fbeta_result.index(max(Fbeta_result))
    if best_Fbeta == best_valid:
        best_epoch = best_Fbeta + 1
    else:
        best_recall_result = [recall_result[best_valid], recall_result[best_Fbeta]]
        if best_recall_result.index(max(best_recall_result)) == 0:
            best_epoch = best_valid + 1
        else:
            best_epoch = best_Fbeta + 1

    best_epoch_list.append(best_epoch)
    print(f"Best epoch: {best_epoch}")

    for i in range(1, EPOCHS + 1):
        if i != best_epoch and os.path.exists(f"models/{bert}/{version}/{bert}_{version}_epoch_{i}.pt"):
            os.remove(f"models/{bert}/{version}/{bert}_{version}_epoch_{i}.pt")

    data_path = os.path.join(fold_path, fold, "cns_fulltest_raw.csv")

    test_batch_size = 8

    output_path = f"Kfolds/output/CNS/{attempt}/{fold}"

    recall5, precision5, accuracy5, Fbeta5 = evaluate_classifier.test(bert, version, best_epoch, data_path, output_path,
                                                                      test_batch_size)

    precision_list, recall_list, threshold_list = evaluate_output.evaluate(bert, version, best_epoch, output_path)

    best_threshold = 0.5
    best_precision = 0
    best_recall = 0
    min_precision = 0.4

    while best_precision == 0 and min_precision > 0.2:
        for i, precision in enumerate(precision_list):
            precision = float(precision)
            recall = float(recall_list[i])
            threshold = float(threshold_list[i])
            if precision > min_precision and precision >= best_precision:
                if recall >= best_recall and threshold >= 0.05:
                    best_recall = recall
                    best_precision = precision
                    best_threshold = threshold
        min_precision -= 0.05

    print(f"best_recall: {best_recall} best_precision: {best_precision} best_threshold: {best_threshold}")

    best_threshold_list.append(best_threshold)
    best_recall_list.append(best_recall)
    best_precision_list.append(best_precision)

    recall5_list.append(recall5)
    precision5_list.append(precision5)
    accuracy5_list.append(accuracy5)
    Fbeta5_list.append(Fbeta5)

    true_pos, true_neg, false_pos, false_neg = compare_output.compare(0.5, bert, version, best_epoch, output_path)

    fp_list.append(false_pos)
    fn_list.append(false_neg)

    summary_path = os.path.join("models", bert, version, "#summary.txt")

    summary_file = open(summary_path, 'r')
    summary_content = summary_file.read()
    print("\nSUMMARY:\n" + summary_content)
    summary_file.close()

    # break  # debug

# folds = folds[0]  # debug

Kfold_results = pd.DataFrame({"Fold": folds, "Version": version_list, "Epoch": best_epoch_list, "Recall": recall5_list,
                              "Precision": precision5_list,
                              "Accuracy": accuracy5_list, "Fbeta": Fbeta5_list, "Best Recall": best_recall_list,
                              "Best Precision": best_precision_list,
                              "Best Threshold": best_threshold_list, "False Neg(0.5)": fn_list,
                              "False Pos(0.5)": fp_list, "Val loss": valid_result_list})

if os.path.exists("Kfolds/Kfold_results_CNS.xlsx"):
    with pd.ExcelWriter("Kfolds/Kfold_results_CNS.xlsx", mode='a', if_sheet_exists='new') as writer:
        Kfold_results.to_excel(writer, sheet_name=bert)
else:
    Kfold_results.to_excel("Kfolds/Kfold_results_CNS.xlsx", sheet_name=bert)

best_fold = valid_result_list.index(min(valid_result_list))
print(f"\nBest fold: {best_fold}\n")

# for i, version in enumerate(version_list):
#     epoch = best_epoch_list[i]
#     logsname = f"models/{bert}/{version}/logs"
#     summaryname = f"models/{bert}/{version}/#summary.txt"
#     fn_name = f"output/{bert}_{version}_epoch{epoch}_false_neg.csv"
#     fp_name = f"output/{bert}_{version}_epoch{epoch}_false_pos.csv"
#     outputname = f"output/{bert}_{version}_epoch{epoch}.csv"
#     results_name = "Kfolds/Kfold_results_SD.xlsx"
#
#     with zipfile.ZipFile(f"{bert}_{version}_fold{i}.zip", mode="w") as zip:
#         zip.write(logsname, os.path.basename(logsname))
#         zip.write(summaryname, os.path.basename(summaryname))
#         zip.write(outputname, os.path.basename(outputname))
#         zip.write(fn_name, os.path.basename(fn_name))
#         zip.write(fp_name, os.path.basename(fp_name))
#         zip.write(results_name, os.path.basename(results_name))
