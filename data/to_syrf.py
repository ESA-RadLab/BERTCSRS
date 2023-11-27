import pandas as pd

data_path_results = "../Kfolds/output/SD/with_titles/Titleabstracts_to_review_SD.csv"
data_path_original = "processed_datasets/sd_full_all_columns.csv"

df_results = pd.read_csv(data_path_results)
df_original = pd.read_csv(data_path_original)

syrf_df = pd.merge(df_results, df_original, how="inner", left_on="titleabstract", right_on="titleabstract")

syrf_df = syrf_df[["Title", "Authors", "PublicationName", "AlternateName", "Abstract", "Url", "AuthorAddress", "Year", "Doi", "ReferenceType", "Keywords", "PdfRelativePath", "CustomId"]]
syrf_df['Year'] = syrf_df['Year'].astype(dtype=pd.Int64Dtype())

syrf_df.to_csv("Kfolds/output/CNS/with_titles/Titleabstracts_to_SYRF_CNS_titles.csv", index=False, sep=",")
