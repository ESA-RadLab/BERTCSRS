import os
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import shuffle

data_path = 'sources/SD_Screening_data.csv'
data_label = "sd"

fold_path = f"../Kfolds/data/{data_label.upper()}/"

full = pd.read_csv(data_path)
print(f"Original length: {len(full)}")

full = full.rename(columns={"ScreeningDecisions": 'decision'})
full = full[~(full['Abstract'].isnull() & full['Title'].isnull())]
full.drop_duplicates(["Title", "Abstract"], inplace=True)
full = full.reset_index(drop=True)

num_blanks = 0

titleabstract_list = []
for i, row in full.iterrows():
    if type(row['Abstract']) == str and type(row['Title']) == str:
        titleabstract = row['Title'] + '. ' + row['Abstract']
        titleabstract_list.append(titleabstract)
    elif type(row['Title']) == str:
        titleabstract = row['Title']
        titleabstract_list.append(titleabstract)
        num_blanks += 1
    elif type(row['Abstract']) == str:
        titleabstract = row['Abstract']
        titleabstract_list.append(titleabstract)
        num_blanks += 1

full = full.reset_index(drop=True)

print(f"Processed length: {len(full)}")
print(f"N blanks: {num_blanks}")

full.insert(0, "titleabstract", titleabstract_list)

full.to_csv(f'processed_datasets/{data_label.lower()}_full_all_columns.csv')

full = full[['titleabstract', 'decision']]
full.to_csv(f'processed_datasets/{data_label.lower()}_full.csv')

skf = StratifiedKFold(n_splits=6)

result = skf.split(full, full['decision'])

for i, (train_fold_indices, full_test_fold_indices) in enumerate(result):
    test_fold = full.iloc[full_test_fold_indices]
    train_fold = full.iloc[train_fold_indices]

    train_fold, val_data = train_test_split(train_fold, test_size=0.1, stratify=train_fold['decision'])

    if not os.path.exists(f'{fold_path}fold_{i}'):
        os.makedirs(f'{fold_path}fold_{i}')

    train_fold.reset_index(drop=True, inplace=True)

    df_downsampled = train_fold.groupby("decision").sample(frac=0.33)
    df_balanced = pd.concat([train_fold[train_fold["decision"] == "Included"], df_downsampled[df_downsampled["decision"] == "Excluded"]])
    shuffle(df_balanced).reset_index(drop=True, inplace=True)

    print(f"train: {len(train_fold)}")
    print(f"balanced: {len(df_balanced)}")
    print(f"test: {len(test_fold)}")
    print(f"val: {len(val_data)}")

    train_fold.to_csv(f'{fold_path}fold_{i}/{data_label.lower()}_train_raw.csv')
    df_balanced.to_csv(f'{fold_path}fold_{i}/{data_label.lower()}_balanced_raw.csv')
    test_fold.to_csv(f'{fold_path}fold_{i}/{data_label.lower()}_test_raw.csv')
    val_data.to_csv(f'{fold_path}fold_{i}/{data_label.lower()}_val_raw.csv')
