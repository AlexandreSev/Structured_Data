# coding : utf-8

import numpy as np
import tensorflow as tf
import pandas as pd
from nltk.util import ngrams

#  input = shape(4, 4, 256)

data = pd.read_csv("word.csv", sep=";", encoding="utf8", index_col=0)

tuple_letters = []
tuple_bigrams = []
tuple_trigrams = []
list_letters = []
list_bigrams = []
list_trigrams = []
elements_to_remove = ['"','-', 'é', '£', ')',"ñ",'?','&','!','(',':',"'"]

for tag in data['tag']:

    tuple_letters += list(ngrams(tag, 1))
    tuple_bigrams += list(ngrams(tag, 2))
    tuple_trigrams += list(ngrams(tag, 3))

# created list_unique_letter of all the letters and numbers in the text

for element in tuple_letters :
    for letter in list(element):
        letter = letter.lower()
        list_letters.append(letter)

list_unique_letter = set()
[x for x in list_letters if x not in list_unique_letter and not list_unique_letter.add(x)]

list_unique_letter = list(list_unique_letter)

for elts in elements_to_remove :
    list_unique_letter.remove(elts)

##### created list_unique_bigrams of all the bigrams in the text

for element in tuple_bigrams :
    list_element=list(element)
    if (list_element[0] in elements_to_remove) or (list_element[1] in elements_to_remove) :
        pass
    else :
        l1 = list_element[0].lower()
        l2 = list_element[1].lower()
        element_used = ''.join([l1,l2])
        if len(element_used)==1:
            pass
        else :
            list_bigrams.append(element_used)

list_unique_bigrams = set()
[x for x in list_bigrams if x not in list_unique_bigrams and not list_unique_bigrams.add(x)]

list_unique_bigrams = list(list_unique_bigrams)

# print(list_unique_bigrams)

##### created list_unique_trigrams of all the trigrams in the text

for element in tuple_trigrams :
    list_element=list(element)
    if (list_element[0] in elements_to_remove) or (list_element[1] in elements_to_remove) or (list_element[2] in elements_to_remove) :
        pass
    else :
        l1 = list_element[0].lower()
        l2 = list_element[1].lower()
        l3 = list_element[2].lower()
        element_used = ''.join([l1,l2,l3])
        if len(element_used)==3:
            list_trigrams.append(element_used)
        else :
            pass

list_unique_trigrams = set()
[x for x in list_trigrams if x not in list_unique_trigrams and not list_unique_trigrams.add(x)]

list_unique_trigrams = list(list_unique_trigrams)

# print(list_unique_trigrams)
