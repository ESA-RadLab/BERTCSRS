import numpy as np
import pandas as pd
import regex as re

file = pd.read_csv("cns_test.csv")
title_abstracts_lengths = [len(re.findall(r'\w+', text)) for text in file["titleabstract"]]
print(f"number of texts: {len(title_abstracts_lengths)}")
print(f"max: {max(title_abstracts_lengths)}")
print(f"min: {min(title_abstracts_lengths)}")
print(f"mean: {np.mean(title_abstracts_lengths):.2f}")
print(f"median: {np.median(title_abstracts_lengths)}")

above_256 = 0
truncated = []
for l in title_abstracts_lengths:
    if l > 256:
        above_256 += 1
        truncated.append(l - 256)
print(f"truncated: {above_256} ({(above_256 / len(title_abstracts_lengths) * 100):.2f}%)")
print(f"max: {max(truncated)}")
print(f"min: {min(truncated)}")
print(f"mean: {np.mean(truncated):.2f}")
print(f"median: {np.median(truncated)}")
