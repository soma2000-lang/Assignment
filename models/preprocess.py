import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch 
import torch.nn as nn
import torch.optim as optim
from torchtext.vocab import FastText, vocab
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import TensorDataset, DataLoader
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from gensim.models.phrases import Phrases, Phraser
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TreebankWordTokenizer
from string import punctuation
import re, random, pathlib
import matplotlib.pyplot as plt
from tqdm import tqdm
tqdm.pandas()

from transformers import RobertaModel, RobertaTokenizer, get_linear_schedule_with_warmup, AutoTokenizer, AutoModel

nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('vader_lexicon')
nltk.download('punkt')


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

path_files = pathlib.Path('./')

data = pd.read_csv('data/reviews_v1_hiring_task.csv', sep=',', encoding='latin-1').sample(500)
print(data.head())
print(data.shape)
Notkeep = [d for d in data.columns if d not in ['reviews.text','reviews.text','reviews.rating','manufacturer','reviews.date','name','id','brand','categories']]
print(data.drop(columns=Notkeep))
stemmer = WordNetLemmatizer()


nltk.download('stopwords')
en_stop       = set(nltk.corpus.stopwords.words('english'))
to_be_removed = list(en_stop) + list(punctuation)

# Preprocess text for transformers
def preprocess_text(data, full_process=True):
        # Remove all the special characters
        data = re.sub(r'\W', ' ', str(data))
        # remove all single characters
        data= re.sub(r'\s+[a-zA-Z]\s+', ' ', data)
        # Remove single characters from the start
        data = re.sub(r'\^[a-zA-Z]\s+', ' ', data)
        # Substituting multiple spaces with single space
        data = re.sub(r'\s+', ' ', data, flags=re.I)
        # Removing prefixed 'b'
        data= re.sub(r'^b\s+', '', data)
        # Converting to Lowercase
        data = data.lower()
        if full_process:
            # Lemmatization
            tokens = data.split()
            tokens = [stemmer.lemmatize(word) for word in tokens]
            tokens = [word for word in tokens if word not in en_stop]
            tokens = [word for word in tokens if len(word) > 3]
            data = ' '.join(tokens)
        return data
data=preprocess_text(data)
print(data)