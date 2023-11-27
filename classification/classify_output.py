import os

import pandas as pd

threshold = 0.23
datalabel = "cns"
data_path = '../data/output/FULL/CNS/pubmed_abstract_20.11_10.34_epoch15.csv'

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
df_excluded = df_excluded[df_excluded["prediction"] > 0.25]

output_folder = "../data/output/FULL"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

df_output.to_csv(f"{output_folder}/{datalabel.upper()}/{datalabel.upper()}_DECISION2.csv")
df_included.to_csv(f"{output_folder}/{datalabel.upper()}/{datalabel.upper()}_INCLUDED2.csv")
df_excluded.to_excel(f"{output_folder}/{datalabel.upper()}/{datalabel.upper()}_check2.xlsx")
