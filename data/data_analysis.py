import random

import numpy as np
import pandas as pd
import regex as re
from numpy.random import Generator, choice

data_path = "../Kfolds/data/SD/fold_2/"

file = pd.read_csv(data_path + "sd_train_raw.csv")
# file = file.drop_duplicates(['titleabstract'])
titleabstracts = file["titleabstract"]



included = []
excluded = []
duplicates = []
title_abstracts_lengths = []
above_256 = 0
truncated = []
n_included = 0
included_truncated = 0
# file_matrix = file.as_matrix()
for i, text in enumerate(file["titleabstract"]):
    length = len(re.findall(r'\w+', text))
    if length > 256:
        above_256 += 1
        truncated.append(length - 256)
        if file["decision"][i] == "Included":
            included_truncated += 1
    title_abstracts_lengths.append(length)
    for j, testtext in enumerate(file["titleabstract"][i + 1:]):
        if text == testtext:
            duplicates.append(text)
    if file["decision"][i] == "Included":
        n_included += 1
        included.append(file.loc[i, ["decision", "titleabstract"]])
    else:
        # if i % 3 == 0:
        excluded.append(file.loc[i, ["decision", "titleabstract"]])

# balanced = included + excluded + included + included
# # excluded.sample()
# random.shuffle(balanced)
# # Generator.shuffle(excluded)
# df_balanced = pd.DataFrame(balanced, columns=['decision', 'titleabstract'])
# df_balanced.to_csv(data_path + "sd_triple_balanced_raw.csv")

df = pd.DataFrame({"duplicates": duplicates})
df.to_excel("../data/duplicates.xlsx")

print(f"number of texts: {len(titleabstracts)}")
# print(f"number of included: {n_included} ({(n_included / len(titleabstracts) * 100):.2f}%)")
print(f"max: {max(title_abstracts_lengths)}")
print(f"min: {min(title_abstracts_lengths)}")
print(f"mean: {np.mean(title_abstracts_lengths):.2f}")
print(f"median: {np.median(title_abstracts_lengths)}")

print(f"truncated: {above_256} ({(above_256 / len(title_abstracts_lengths) * 100):.2f}%)")
# print(f"truncated included: {included_truncated} ({(included_truncated / n_included * 100):.2f}%)")
# print(f"max: {max(truncated)}")
# print(f"min: {min(truncated)}")
# print(f"mean: {np.mean(truncated):.2f}")
# print(f"median: {np.median(truncated)}")

print(f"#duplicates: {len(duplicates)}")

# print(f"balanced: {len(balanced)}")
