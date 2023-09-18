import os
import pandas as pd

fold_path = "Kfolds/output/SD"
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

FN_duplicate = []
FP_duplicate = []

duplicates = pd.read_excel("data/duplicates.xlsx")

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
                    if duplicates["duplicates"].eq(FN).any():
                        FN_duplicate.append(True)
                    else:
                        FN_duplicate.append(False)
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
                    if duplicates["duplicates"].eq(FP).any():
                        FP_duplicate.append(True)
                    else:
                        FP_duplicate.append(False)

FN_results = pd.DataFrame({"Title-abstract": FN_titleabstracts, "Count": FN_count, "Average prediction": FN_prediction, "Duplicate": FN_duplicate})
FN_results = FN_results.sort_values("Count", ascending=False)
FP_results = pd.DataFrame({"Title-abstract": FP_titleabstracts, "Count": FP_count, "Average prediction": FP_prediction, "Duplicate": FP_duplicate})
FP_results = FP_results.sort_values("Count", ascending=False)

combined_results = []

for _, result in FN_results.iterrows():
    if int(result['Count']) >= 5 and result["Duplicate"] is False:
        combined_results.append(result)

for _, result in FP_results.iterrows():
    if int(result['Count']) >= 5 and result["Duplicate"] is False:
        combined_results.append(result)

combined_results = pd.DataFrame(combined_results)
combined_results.to_csv(f"{fold_path}/Titleabstracts_to_review_SD.csv")

if os.path.exists(f"{fold_path}/Titleabstracts_SD.xlsx"):
    with pd.ExcelWriter(f"{fold_path}/Titleabstracts_SD.xlsx", mode='a', if_sheet_exists='replace') as writer:
        FN_results.to_excel(writer, sheet_name="False Negatives")
else:
    FN_results.to_excel(f"{fold_path}/Titleabstracts_SD.xlsx", sheet_name="False Negatives")

with pd.ExcelWriter(f"{fold_path}/Titleabstracts_SD.xlsx", mode='a', if_sheet_exists='replace') as writer:
    FP_results.to_excel(writer, sheet_name="False Positives")
