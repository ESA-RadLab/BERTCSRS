import os
import pandas as pd
from tqdm import tqdm
import bert_classifier_train
from evaluation import evaluate_output, evaluate_classifier, compare_output

bert = 'biobert'

fold_path = "Kfolds\data"
folds = os.listdir(fold_path)
folds.sort()

for k, fold in enumerate(folds):
    if "fold" not in fold:
        folds.pop(k)

best_epoch_list = []
best_recall_list = []
best_precision_list = []
best_threshold_list = []
valid_result_list = []
version_list = []
fp_list = []
fn_list = []

for fold in tqdm(folds):
    print("\n" + fold)
    train_path = os.path.join(fold_path, fold, "cns_balanced_raw.csv")
    val_path = os.path.join(fold_path, fold, "cns_val_raw.csv")

    LR = 2e-5
    EPOCHS = 8
    batch_size = 15
    dropout = 0.2
    step_size = 5
    gamma = 1
    pos_weight = 10

    valid_result, version = bert_classifier_train.train(bert, train_path, val_path, LR, EPOCHS, batch_size, dropout,
                                                        pos_weight, gamma, step_size)

    valid_result_list.append(min(valid_result))
    version_list.append(version)

    best_epoch = valid_result.index(min(valid_result)) + 1
    best_epoch_list.append(best_epoch)
    print(f"Best epoch: {best_epoch}")

    data_path = os.path.join(fold_path, fold, "cns_fulltest_raw.csv")

    batch_size = 13

    evaluate_classifier.test(bert, version, best_epoch, data_path, batch_size)

    # test_output = pd.read_csv(os.path.join("output", f"{bert}_{version}_epoch{best_epoch}_TEST.csv"))
    # val_output = pd.read_csv(os.path.join("output", f"{bert}_{version}_epoch{best_epoch}_VAL.csv"))
    #
    # full_output = test_output.append(val_output)
    # full_output.to_csv(os.path.join("output", f"{bert}_{version}_epoch{best_epoch}.csv"), index=False, lineterminator="\r\n")

    output_path = "output"
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
                if recall >= best_recall:
                    best_recall = recall
                    best_precision = precision
                    best_threshold = threshold
        min_precision -= 0.05

    print(f"best_recall: {best_recall} best_precision: {best_precision} best_threshold: {best_threshold}")

    best_threshold_list.append(best_threshold)
    best_recall_list.append(best_recall)
    best_precision_list.append(best_precision)

    true_pos, true_neg, false_pos, false_neg = compare_output.compare(0.5, bert, version, best_epoch, output_path)

    fp_list.append(false_pos)
    fn_list.append(false_neg)

    # break  # debug

# folds = folds[0]  # debug

Kfold_results = pd.DataFrame({"Fold": folds, "Version": version_list, "Epoch": best_epoch_list, "Recall": best_recall_list, "Precision": best_precision_list,
                              "Threshold": best_threshold_list, "False Neg(0.5)": fn_list, "False Pos(0.5)": fp_list, "Val loss": valid_result_list})


if os.path.exists("Kfolds/Kfold_results.xlsx"):
    with pd.ExcelWriter("Kfolds/Kfold_results.xlsx", mode='a', if_sheet_exists='new') as writer:
        Kfold_results.to_excel(writer, sheet_name=bert)
else:
    Kfold_results.to_excel("Kfolds/Kfold_results.xlsx", sheet_name=bert)