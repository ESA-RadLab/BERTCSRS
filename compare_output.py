import pandas as pd

threshold = 0.5
data_path = "output/biobert_08.08_08.41_epoch7.csv"

df_output = pd.read_csv(data_path)
decisions = df_output['decision']
predictions = df_output['prediction']

true_pos = []
true_neg = []
false_pos = []
false_neg = []

for i, prediction in enumerate(predictions):
    decision = decisions[i]
    if prediction >= threshold and decision == "Included":
        true_pos.append(df_output.iloc[i, [3,4,5]])
    elif prediction < threshold and decision == "Excluded":
        true_neg.append(df_output.iloc[i, [3,4,5]])
    elif prediction >= threshold and decision == "Excluded":
        false_pos.append(df_output.iloc[i, [3,4,5]])
    else:
        false_neg.append(df_output.iloc[i, [3,4,5]])

print(len(true_pos), len(true_neg), len(false_pos), len(false_neg))

df_fn = pd.DataFrame(false_neg)
df_fn.to_csv(data_path[:-4] + "_false_neg.csv")

df_fp = pd.DataFrame(false_pos)
df_fp.to_csv(data_path[:-4] + "_false_pos.csv")
