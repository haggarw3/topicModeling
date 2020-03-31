import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

import nltk
from nltk import FreqDist
# nltk.download('stopwords')

import spacy

import gensim
from gensim import corpora
from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
from gensim.corpora import Dictionary

import pyLDAvis.gensim
import warnings
warnings.filterwarnings('ignore')

file = open('file_english.txt', 'r')
string = file.readlines()[0]
words = re.findall('[A-z]+',string)
words = words[1:]
print('This is already cleaned using NLTK and regex')
print('This is the number of words we have right now')
print(len(words))


# Example to show how to add other user defined stop words in spacy
from spacy.lang.en import English
nlp = spacy.load('en')  # creating the english language pipeline
stop_words = ['become', 'chose']
for item in stop_words:
    lexeme = nlp.vocab[item]
    lexeme.is_stop = True

cleaned = nlp(' '.join(words)) # Note that for cleaning we are combining words to a string again
# nlp pipeline takes a complete string as an argument
print(type(cleaned)) # This is a spacy object
cleaned = str(cleaned)
words_spacy = cleaned.split(' ')
print('This is the number of words we have after using spacy')
print(len(words_spacy))
