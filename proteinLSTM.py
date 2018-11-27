# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)
# set_session(sess)  # set this TensorFlow session as the default session for Keras

import keras
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Dropout, TimeDistributed
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np

print('Loading data')

temp = open("pdb_seqres.txt",'r')
temp = temp.read().splitlines()

train = []
validation = []

for i in range(len(temp)):
	if (i+1)% 5 == 0:
		validation.append(temp[i])
	else:
		train.append(temp[i])


classes = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y','\n']
protein_label_encoder = pd.factorize(classes)
encoder = OneHotEncoder()
protein_labels_1hot = encoder.fit_transform(protein_label_encoder[0].reshape(-1,1))
onehot_array = protein_labels_1hot.toarray()
d1 = dict(zip(classes,onehot_array.tolist()))

encodedTrain = []
for line in train:
	for i,aminoacid in enumerate(line):
		encoding = d1[aminoacid]
		encodedTrain.append(encoding)
		if i == len(line)-1:
			encoding = d1['\n']
			encodedTrain.append(encoding)

encodedValidation = []
for line in validation:
	for i,aminoacid in enumerate(line):
		encoding = d1[aminoacid]
		encodedValidation.append(encoding)
		if i == len(line)-1:
			encoding = d1['\n']
			encodedValidation.append(encoding)




encodedTrain = np.array(encodedTrain)
encodedValidation = np.array(encodedValidation)

print(encodedTrain.shape)
print(encodedValidation.shape)


encodedTrain = np.concatenate((encodedTrain,encodedValidation))

print(encodedTrain.shape)


trainX = []
trainY = []

for i in range(0,len(encodedTrain),21):
	trainX.append(encodedTrain[i:i+20])
	trainY.append(encodedTrain[i+1:i+21])

testX = []
testY = []

for i in range(0,len(encodedValidation),21):
	trainX.append(encodedValidation[i:i+20])
	trainY.append(encodedValidation[i+1:i+21])


vocabulary = 21
hidden_size = 128

model = Sequential()
model.add(Embedding(vocabulary,hidden_size,input_length=20))
model.add(LSTM(hidden_size, return_sequences=True))
model.add(LSTM(hidden_size, return_sequences=True))
model.add(Dropout(0.5))
model.add(TimeDistributed(Dense(vocabulary, activation='softmax')))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

result = model.fit(trainX, trainY, epochs=100, verbose=1)

print("score: ", result[0])
print("loss: ", result[1])





