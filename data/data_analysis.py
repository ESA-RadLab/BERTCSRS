import numpy as np
import pandas as pd
import regex as re

file = pd.read_csv("sex_diff_aug.csv")
titleabstracts = file["titleabstract"]


duplicates = []
title_abstracts_lengths = []
above_256 = 0
truncated = []
included = 0
included_truncated = 0
for i, text in enumerate(file["titleabstract"]):
    length = len(re.findall(r'\w+', text))
    if length > 256:
        above_256 += 1
        truncated.append(length - 256)
        if file["decision"][i] == "Included":
            included_truncated += 1
    title_abstracts_lengths.append(length)
    for j, testtext in enumerate(file["titleabstract"][i+1:]):
        if text == testtext:
            duplicates.append([text, i, j])
    if file["decision"][i] == "Included":
        included += 1


print(f"number of texts: {len(titleabstracts)}")
print(f"number of included: {included}")
print(f"max: {max(title_abstracts_lengths)}")
print(f"min: {min(title_abstracts_lengths)}")
print(f"mean: {np.mean(title_abstracts_lengths):.2f}")
print(f"median: {np.median(title_abstracts_lengths)}")

print(f"truncated: {above_256} ({(above_256 / len(title_abstracts_lengths) * 100):.2f}%)")
print(f"truncated included: {included_truncated} ({(included_truncated / included * 100):.2f}%)")
print(f"max: {max(truncated)}")
print(f"min: {min(truncated)}")
print(f"mean: {np.mean(truncated):.2f}")
print(f"median: {np.median(truncated)}")

print(f"#duplicates: {len(duplicates)}")
