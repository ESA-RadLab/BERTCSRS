import os
import pandas as pd

fold_path = "Kfolds\output"
folds = os.listdir(fold_path)
folds.sort()

filtered_folds = [fold for fold in folds if "fold" in fold]
folds = filtered_folds

FN_titleabstracts = []
FN_count = []
FN_prediction = []

FP_titleabstracts = []
FP_count = []
FP_prediction = []

for fold in folds:
    files = os.listdir(os.path.join(fold_path, fold))
    files.sort()

    for file_name in files:
        if "false_neg" in file_name:
            file_path = os.path.join(fold_path, fold, file_name)
            FN_df = pd.read_csv(file_path)
            for j, FN in enumerate(FN_df['titleabstract']):
                present = False
                for i, titleabstract in enumerate(FN_titleabstracts):
                    if FN == titleabstract:
                        old_count = FN_count[i]
                        prediction = FN_df.loc[j, ['prediction']][0]

                        FN_prediction[i] = (FN_prediction[i] * old_count + prediction) / (old_count + 1)
                        FN_count[i] = old_count + 1
                        present = True
                        break
                if not present:
                    FN_titleabstracts.append(FN)
                    FN_count.append(1)
                    FN_prediction.append(FN_df.loc[j, ['prediction']][0])
        if "false_pos" in file_name:
            file_path = os.path.join(fold_path, fold, file_name)
            FP_df = pd.read_csv(file_path)
            for j, FP in enumerate(FP_df['titleabstract']):
                present = False
                for i, titleabstract in enumerate(FP_titleabstracts):
                    if FP == titleabstract:
                        old_count = FP_count[i]
                        prediction = FP_df.loc[j, ['prediction']][0]

                        FP_prediction[i] = (FP_prediction[i] * old_count + prediction) / (old_count + 1)
                        FP_count[i] = old_count + 1
                        present = True
                        break
                if not present:
                    FP_titleabstracts.append(FP)
                    FP_count.append(1)
                    FP_prediction.append(FP_df.loc[j, ['prediction']][0])

FN_results = pd.DataFrame({"Title-abstract": FN_titleabstracts, "Count": FN_count, "Average prediction": FN_prediction})
FN_results = FN_results.sort_values("Count", ascending=False)
FP_results = pd.DataFrame({"Title-abstract": FP_titleabstracts, "Count": FP_count, "Average prediction": FP_prediction})
FP_results = FP_results.sort_values("Count", ascending=False)

if os.path.exists("Kfolds/Kfold_results.xlsx"):
    with pd.ExcelWriter(f"{fold_path}/Titleabstracts.xlsx", mode='a', if_sheet_exists='clear') as writer:
        FN_results.to_excel(writer, sheet_name="False Negatives")
else:
    FN_results.to_excel(f"{fold_path}/Titleabstracts.xlsx", sheet_name="False Negatives")

with pd.ExcelWriter(f"{fold_path}/Titleabstracts.xlsx", mode='a', if_sheet_exists='new') as writer:
    FP_results.to_excel(writer, sheet_name="False Positives")
