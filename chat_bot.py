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
docs_x = []
docs_y = []

for intent in data['intents']:
	#Stemming & tokenizing
	for pattern in intent["patterns"]:
		wrds = nltk.word_tokenize(pattern)
		words.extend(wrds)
		#classify patten by intent
		docs_x.append(pattern)
		docs_y.append(intent["tag"])

	if intent['tag'] not in labels:
		labels.append(intent['tag'])
words = [stemmer.stem(w.lower()) for w in words]
#get rid of duplicates
words = sorted(list(set(words)))

labels = sorted(labels)

#Create bag of words in order to train our neural net
#how many times each word occurs
training = []
output = []
out_empty = [0 for _ in range(len(classes))]
for x,doc in enumerate(docs_x):
	bag = []
	wrds = [stemmer.stem(w) for w in doc]

	for w in words:
		#word exists
		#1 hot encoded
		if w in wrds: 
			bag.append(1)
		else:
			bag.append(0)

	output_row = out_empty[:]
	output_row[labels.index(docs_y[x])] = 1
	training.append(bag)
	output.append(output_row)







