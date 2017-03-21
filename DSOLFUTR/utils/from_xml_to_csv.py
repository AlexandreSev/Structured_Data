# coding: utf-8
from settings import training_directory
import xml.etree.ElementTree as ET
from os.path import join as pjoin
import pandas as pd


tree = ET.parse(pjoin(training_directory, 'word.xml'))
root = tree.getroot()

list_file = []
list_tag = []

for child in root:
	list_file.append(child.attrib["file"])
	list_tag.append(child.attrib["tag"])

data = pd.DataFrame({"file":list_file, "tag": list_tag})

data.to_csv(pjoin(training_directory, "word.csv"), sep=";", encoding="utf8")







