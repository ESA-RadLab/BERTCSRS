import os

import pandas as pd

threshold = 0.23
datalabel = "cns"
data_path = '../data/output/FULL/CNS/pubmed_abstract_20.11_10.34_epoch15.csv'
full_model_name = "pubmed_abstract_20.11_10.34_epoch15"

df_output = pd.read_csv(data_path)

decision_dict = {
    False: 'Excluded',
    True: 'Included',
}

decision_bool = df_output['prediction'] > threshold

decision = [decision_dict[out] for out in decision_bool]

included = decision_bool.sum()
total = len(decision_bool)

print(included)
print(included / total)

df_output['decision'] = decision
df_included = df_output[decision_bool]
df_included = df_included.loc[:, 'titleabstract']

df_excluded = df_output[~decision_bool]
df_excluded = df_excluded[df_excluded["prediction"] > 0.13]

output_folder = "../data/output/FULL"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

df_output.to_csv(f"{output_folder}/{datalabel.upper()}/{datalabel.upper()}_{full_model_name}_TH{threshold}_DECISION.csv")
df_included.to_csv(f"{output_folder}/{datalabel.upper()}/{datalabel.upper()}_{full_model_name}_TH{threshold}_INCLUDED.csv")
df_excluded.to_excel(f"{output_folder}/{datalabel.upper()}/{datalabel.upper()}_{full_model_name}_TH{threshold}_check.xlsx")
