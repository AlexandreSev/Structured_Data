# coding: utf-8
import tensorflow as tf
import numpy as np
from settings import training_directory
import xml.etree.ElementTree as ET
from os.path import join as pjoin


tree = ET.parse(pjoin(training_directory, 'word.xml'))
root = tree.getroot()

for child in root:
	print(ET.tostring(child["image file"]))







