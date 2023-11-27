import numpy as np
import pandas as pd

data_path = "../data/output/FULL/CNS/CNS_INCLUDED2.csv"
df_training = pd.read_csv("../data/sources/CNS_Screening_data.csv")
df_results = pd.read_csv(data_path)
df_original = pd.read_csv("../data/processed_datasets/all_ref_CNS_all_columns.csv")

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

syrf_df.to_csv("../data/output/SYRF_CNS_pubmed_abstract_20.11_10.34_epoch15_TH_0.23.csv", index=False, sep=",")
