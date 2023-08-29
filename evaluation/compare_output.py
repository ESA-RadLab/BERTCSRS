import os

import pandas as pd


def compare(threshold, bert, version, epoch, data_path, fold=''):
    output_data_path = os.path.join(data_path, f"{bert}_{version}_epoch{epoch}.csv")

    df_output = pd.read_csv(output_data_path)
    decisions = df_output['decision']
    predictions = df_output['prediction']

    true_pos = []
    true_neg = []
    false_pos = []
    false_neg = []

    for i, prediction in enumerate(predictions):
        decision = decisions[i]
        if prediction >= threshold and decision == "Included":
            true_pos.append(df_output.iloc[i, [2, 3, 4]])
        elif prediction < threshold and decision == "Excluded":
            true_neg.append(df_output.iloc[i, [2, 3, 4]])
        elif prediction >= threshold and decision == "Excluded":
            false_pos.append(df_output.iloc[i, [2, 3, 4]])
        else:
            false_neg.append(df_output.iloc[i, [2, 3, 4]])

    print(len(true_pos), len(true_neg), len(false_pos), len(false_neg))

    df_fn = pd.DataFrame(false_neg)
    if len(false_neg) > 0:
        df_fn = df_fn.sort_values("prediction", ascending=True)
    df_fn.to_csv(f"{data_path[:-4]}_fold{fold}_false_neg.csv")

    df_fp = pd.DataFrame(false_pos)
    if len(false_pos) > 0:
        df_fp = df_fp.sort_values("prediction", ascending=False)
    df_fp.to_csv(f"{data_path[:-4]}_fold{fold}_false_pos.csv")

    return len(true_pos), len(true_neg), len(false_pos), len(false_neg)

if __name__ == "__main__":
    threshold = 0.5
    data_path = "../output"
    bert_name = "pubmed_abstract"
    epoch = 12
    version = "23.08_14.27"

    compare(threshold, bert_name, version, epoch, data_path, fold = '1')
