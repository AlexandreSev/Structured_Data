# coding: utf-8
import os
from os.path import join as pjoin

# Where to find the word.csv. This folder must als contains the folder word of all pictures of ICDAT dataset
training_directory = "/Users/antoine/Dropbox/2016-2017 X/structuredData/Structured_Data/DSOLFUTR/data"

# Where to find precomputed representations of ICDAR dataset
representations_directory = pjoin(training_directory, 'representations')

# Where to find the file corresponding to MJSynth
#This folder must contain a folder representations and a folder targets
ox_directory = pjoin(training_directory, "ox_files")