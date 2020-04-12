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
spacy_nlp = spacy.load('en_core_web_sm')  # will try another pipeline later as well

# adding more stop words as per our data
stop_words = ['become', 'chose']
for item in stop_words:
    lexeme = nlp.vocab[item]
    lexeme.is_stop = True

# tokenization using the spacy pipeline
tokenized = nlp(' '.join(words))  # Note that we can use the pipeline to tokenize
# nlp pipeline takes a complete string as an argument
print(type(tokenized))  # This is a spacy object

spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
cleaned = [token for token in tokenized if token.is_stop == False]

print('Number of words after removing stop words from spaCy are', len(cleaned))


# Lemmatization

cleaned_lemma = [token.lemma_ for token in cleaned]

from gensim.models import Phrases

phrases = Phrases(sentences, min_count=1, threshold=1)
