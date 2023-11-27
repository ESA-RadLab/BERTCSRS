import pandas as pd

data_path = 'All references (Esa neuro).xlsx'

# full = pd.read_csv(data_path)
full = pd.read_excel(data_path)

print(len(full))

full = full[~(full['Abstract'].isnull() & full['Title'].isnull())]
full.drop_duplicates(["Title", "Abstract"], inplace=True)
full = full.reset_index(drop=True)

num_blanks = 0

titleabstract_list = []
for i, row in full.iterrows():
    if type(row['Abstract']) == str and type(row['Title']) == str:
        titleabstract = row['Title'] + '. ' + row['Abstract']
        titleabstract_list.append(titleabstract)
    elif type(row['Title']) == str:
        titleabstract = row['Title']
        titleabstract_list.append(titleabstract)
        num_blanks += 1
    elif type(row['Abstract']) == str:
        titleabstract = row['Abstract']
        titleabstract_list.append(titleabstract)
        num_blanks += 1


full['titleabstract'] = titleabstract_list

full.to_csv('processed_datasets/all_ref_SD_all_columns.csv')

full = full['titleabstract']

print(len(full))
print(num_blanks)

full.to_csv('processed_datasets/all_ref_SD.csv')
