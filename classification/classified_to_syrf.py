import numpy as np
import pandas as pd

data_label = "CNS"
full_model_name = "pubmed_abstract_20.11_10.34_epoch15"
threshold = "0.23"

included_data_path = f"../data/output/FULL/{data_label.upper()}/{data_label.upper()}_INCLUDED.csv"
training_source_path = f"../data/sources/{data_label.upper()}_Screening_data.csv"
all_column_data_path = f"../data/processed_datasets/all_ref_{data_label.upper()}_all_columns.csv"

df_training = pd.read_csv(training_source_path)
df_results = pd.read_csv(included_data_path)
df_original = pd.read_csv(all_column_data_path)

syrf_df = pd.merge(df_results, df_original, how="inner", left_on="titleabstract", right_on="titleabstract")
syrf_df = syrf_df[~syrf_df["Title"].isin(df_training["Title"])]

syrf_df = syrf_df.rename(columns={"Author": "Authors", "Journal":"PublicationName", "Unnamed: 23":"AlternateName", "URL":"Url", "Type of the Article":"ReferenceType", "Unnamed: 22": "Doi"})

syrf_df["PdfRelativePath"] = np.nan
syrf_df["CustomId"] = np.nan

syrf_df = syrf_df[["Title", "Authors", "PublicationName", "AlternateName", "Abstract", "Url", "AuthorAddress", "Year", "Doi", "ReferenceType", "Keywords", "PdfRelativePath", "CustomId"]]

syrf_df["Keywords"] = syrf_df["Keywords"].str.replace("_x000D_\n", ';', regex=False)

for column in syrf_df.columns:
    if type(syrf_df[column][0]) is str:
        syrf_df[column] = syrf_df[column].str.replace("_x000D_", '', regex=False)


syrf_df['Year'] = syrf_df['Year'].astype(dtype=pd.Int64Dtype())

print(len(df_results))
print(len(syrf_df))

syrf_df.to_csv(f"../data/output/SYRF_{data_label}_{full_model_name}_TH_{threshold}.csv", index=False, sep=",")
