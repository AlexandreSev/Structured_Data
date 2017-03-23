# coding : utf-8
from .settings import training_directory

import numpy as np
import tensorflow as tf
import pandas as pd
from nltk.util import ngrams
from os.path import join as pjoin
import unicodedata

#training_directory = "/home/alex/Documents/strutured data/projet/train"


#  input = shape(4, 4, 256)

def get_ngrams(word=None):
    if word is None:
        data = pd.read_csv(pjoin(training_directory, "word.csv"), sep=";", encoding="utf8", index_col=0)
        list_tag = data['tag']
    else:
        list_tag = [word]


    tuple_letters = []
    tuple_bigrams = []
    tuple_trigrams = []
    list_letters = []
    list_bigrams = []
    list_trigrams = []
    elements_to_keep = "abcdefghijklmnopqrstuvwxyz0123456789"


    for tag in list_tag:

        if (0.0, ) in list(ngrams(tag, 1)):
            print(word)
            print(tag)
        tuple_letters += list(ngrams(tag, 1))
        tuple_bigrams += list(ngrams(tag, 2))
        tuple_trigrams += list(ngrams(tag, 3))

    # created list_unique_letter of all the letters and numbers in the text

    for element in tuple_letters :
        element_used = unicodedata.normalize("NFKD", list(element)[0].lower())
        if (element_used in elements_to_keep) and (element_used not in list_letters):
                list_letters.append(element_used)

    list_letters.sort()
    ##### created list_unique_bigrams of all the bigrams in the text

    for element in tuple_bigrams :
        list_element=list(element)
        if (list_element[0].lower() in elements_to_keep) and (list_element[1].lower() in elements_to_keep) :
            l1 = list_element[0].lower()
            l2 = list_element[1].lower()
            element_used = unicodedata.normalize("NFKD", ''.join([l1,l2]))
            if len(element_used)==2 and (element_used not in list_bigrams):
                    list_bigrams.append(element_used)

    list_bigrams.sort()

    # print(list_unique_bigrams)

    ##### created list_unique_trigrams of all the trigrams in the text

    for element in tuple_trigrams :
        list_element=list(element)
        if (list_element[0].lower() in elements_to_keep) and (list_element[1].lower() in elements_to_keep) and  \
            (list_element[2].lower() in elements_to_keep) :
            l1 = list_element[0].lower()
            l2 = list_element[1].lower()
            l3 = list_element[2].lower()
            element_used = unicodedata.normalize("NFKD", ''.join([l1,l2, l3]))
            if len(element_used)==3 and (element_used not in list_trigrams):
                list_trigrams.append(element_used)

    list_trigrams.sort()

    return list_letters + list_bigrams + list_trigrams


def get_dict_ngrams(list_ngrams):
    response = {}
    for i, char in enumerate(list_ngrams):
        response[char] = i
    return response

def reverse_dict(input_dict):
    response = {}
    for key in input_dict:
        response[input_dict[key]] = key
    return response
