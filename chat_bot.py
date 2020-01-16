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

words = []
labels = []
docs = []

for intent in data['intents']:
	#Stemming & tokenizing
	for pattern in intent["patterns"]:
		wrds = nltk.word_tokenize(pattern)
		words.extend(wrds)
		docs.append(pattern)

	if intent['tag'] not in labels:
		labels.append(intent['tag'])