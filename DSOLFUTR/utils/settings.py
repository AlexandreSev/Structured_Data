# coding: utf-8
import os
from os.path import join as pjoin

# Where to find the word.csv. This folder must als contains the folder word of all pictures of ICDAT dataset
training_directory = "/home/alex/Documents/strutured data/projet/train"

# Where to find precomputed representations of ICDAR dataset
representations_directory = pjoin(training_directory, 'representations')

# Where to find the file corresponding to MJSynth
#This folder must contain a folder representations and a folder targets
ox_directory = pjoin(training_directory, "ox_files")