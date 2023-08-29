from transformers import MarianMTModel, MarianTokenizer
from sklearn.utils import shuffle
import pandas as pd
import torch
from nltk.tokenize import sent_tokenize
import nltk
import re


# In[2]:


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# In[ ]:


# MODEL 1 -> ENGLISH TO FRENCH

# Get the name of the first model
first_model_name = 'Helsinki-NLP/opus-mt-en-fr'

# Get the tokenizer
first_model_tkn = MarianTokenizer.from_pretrained(first_model_name)

# Load the pretrained model based on the name
first_model = MarianMTModel.from_pretrained(first_model_name).to(device)

# In[ ]:


# MODEL 2 -> FRENCH TO ENGLISH

# Get the name of the second model
second_model_name = 'Helsinki-NLP/opus-mt-fr-en'

# Get the tokenizer
second_model_tkn = MarianTokenizer.from_pretrained(second_model_name)

# Load the pretrained model based on the name
second_model = MarianMTModel.from_pretrained(second_model_name).to(device)


# In[ ]:


def format_batch_texts(language_code, batch_texts):
    formated_bach = ">>{}<< {}".format(language_code, batch_texts)
    return formated_bach


# In[8]:


# declare the name of the file you want to use
file_name = "cns_train_new1.csv"
file_path = "data/" + file_name

# In[9]:


train_data = pd.read_csv(file_path)


# In[ ]:


def perform_translation(batch_texts, model, tokenizer, language="fr"):
    # Prepare the text data into appropriate format for the model
    formated_batch_texts = format_batch_texts(language, batch_texts)

    # Generate translation using model
    translated = model.generate(**tokenizer(formated_batch_texts, return_tensors="pt", padding=True, truncation=True).to(device))

    # Convert the generated tokens indices back into text
    translated_texts = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

    return translated_texts


# In[10]:




nltk.download('punkt')

# In[ ]:


# abstracts
new_df = train_data
for index, row in train_data.iterrows():
    if row['decision'] == 'Included':
        print("new included")
        lt = sent_tokenize(row['titleabstract'])
        prev = 0
        for i in range(2, len(lt), 2):
            cur_text = ""
            for sen in lt[prev:i]:
                if len(sen) > 512:
                    cur_text = cur_text + sen
                    continue
                translated_texts = perform_translation(sen, first_model, first_model_tkn)
                back_translated_texts = perform_translation(translated_texts, second_model, second_model_tkn)
                REPLACE_BY_SPACE_RE = re.compile('[{}\[\]|@,;\'"]')
                new_text = REPLACE_BY_SPACE_RE.sub(' ', back_translated_texts[0])
                cur_text += new_text
            new_abstract = ' '.join(lt[0:prev] + [cur_text] + lt[i:])
            if not new_abstract == row['titleabstract']:
                new_df = pd.concat([pd.DataFrame({'titleabstract': [new_abstract], 'decision': ['Included']}), new_df])
            prev = i

# In[ ]:


# # titles
# for index, row in train_data.iterrows():
#     if row['decision'] == 'Included':
#         translated_texts = perform_translation(row['titles'], first_model, first_model_tkn)
#         if len(translated_texts[0]) > 512:
#             continue
#         back_translated_texts = perform_translation(translated_texts, second_model, second_model_tkn)
#         REPLACE_BY_SPACE_RE = re.compile('[{}\[\]\|@,;\'"]')
#         new_text = REPLACE_BY_SPACE_RE.sub('', back_translated_texts[0])
#         new_df = new_df.append({'titles': new_text, 'abstracts': row['abstracts'], 'decision': 'Included'},
#                                ignore_index=True)
#
# # In[ ]:


# new_df['titleabstract'] = new_df['titles'] + '. ' + new_df['abstracts']

# In[ ]:


new_df = new_df.drop_duplicates(subset=['titleabstract'])
new_df = shuffle(new_df)

# In[ ]:


# new_df[new_df.decision == 'Included'].shape[0] / new_df.shape[0]

# In[ ]:


new_df.to_csv('cns_aug_new.csv')
