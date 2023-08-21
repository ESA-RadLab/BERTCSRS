import re

import pandas as pd

data_df = pd.read_csv("data/cns_val_new1.csv")

remove_words = ["purpose", "purposes", "objective", "result", "results", "method", 'objectives', 'methods', 'design',
                'study population', 'measurements', 'conclusions', 'materials', 'methods', 'limitations', 'setting',
                'patients', 'intervention', 'main outcome', 'measures', 'background', 'conclusion']


def clean_text(text):
    text = re.sub(r'\W+', ' ', str(text).lower())
    text = ' '.join(word for word in text.split() if word not in remove_words)
    # text = "\"" + text + "\""
    return str(text)


data_df["titleabstract"] = data_df["titleabstract"].apply(clean_text)
for column in data_df.columns.values.tolist():
    if "Unnamed" in column:
        data_df = data_df.drop(columns=column)


data_df.to_csv("data/cns_val_clean.csv")
