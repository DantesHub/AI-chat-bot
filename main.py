import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy
import tensorflow
import random 
import tflearn
import json

stemmer = LancasterStemmer()
with open("data.json") as file:
	data = json.load(file)

print(data)