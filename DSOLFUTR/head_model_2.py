# coding : utf-8

import numpy as np
import tensorflow as tf
import nltk
from nltk.util import ngrams

#  input = shape(4, 4, 256)

data = pd.read_csv("word.csv", sep=";", encoding="utf8", index_col=0)

list_letters = []
list_bigrams = []
list_trigrams = []

for tag in data['tag']:

    list_letters += ngrams(tag, 1)
    list_bigrams += ngrams(tag, 2)
    list_trigrams += ngrams(tag, 3)
