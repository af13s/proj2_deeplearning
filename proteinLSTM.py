import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

import keras
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Dropout, TimeDistributed
from sklearn.preprocessing import OneHotEncoder
from keras.optimizers import Adam
import pandas as pd
import numpy as np
from tqdm import tqdm

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
d1['0']=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]

N = 100

encodedTrain = []
for line in tqdm(train):
	if len(line) < N:
		array = ['0' for _ in range(0,N-len(line))]
		line = array + list(line[0:N])
		line = ''.join(line)
	else:
		line = line[0:N]
	for i,aminoacid in enumerate(line):
		encoding = d1[aminoacid]
		encodedTrain.append(encoding)

encodedTrain = np.array(encodedTrain)
encodedTrain = encodedTrain.reshape(-1,N,21)

print('train Shape: ', encodedTrain.shape)


encodedValidation = []
for line in tqdm(validation):
	if len(line) < N:
		array = ['0' for _ in range(0,N-len(line))]
		line = array + list(line[0:N])
		line = ''.join(line)
	else:
		line = line[0:N]
	for i,aminoacid in enumerate(line):
		encoding = d1[aminoacid]
		encodedValidation.append(encoding)

encodedValidation = np.array(encodedValidation)
encodedValidation = encodedValidation.reshape(-1,N,21)

print('validation Shape: ', encodedValidation.shape)

trainX = []
trainY = []

for sample in encodedTrain:
	trainX.append(sample[0:N-1])
	trainY.append(sample[1:N])

trainX = np.array(trainX)
trainY = np.array(trainY)

print(trainX.shape)
print(trainY.shape)

testX = []
testY = []

for sample in encodedValidation:
	testX.append(sample[0:N-1])
	testY.append(sample[1:N])

testX = np.array(testX)
testY = np.array(testY)

print(testX.shape)
print(testY.shape)


vocabulary = 21
hidden_size = 128

model = Sequential()
#model.add(Embedding(vocabulary,hidden_size,input_length=N-1))
model.add(LSTM(hidden_size, input_shape=(N-1, 21), return_sequences=True))
model.add(LSTM(hidden_size, return_sequences=True))
model.add(Dropout(0.5))
model.add(TimeDistributed(Dense(vocabulary, activation='softmax')))

model.summary()

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=Adam(lr=0.001), metrics=['accuracy'])

model.fit(trainX, trainY, batch_size=512, epochs=10, verbose=1)

# print("score: ", result[0])
# print("loss: ", result[1])





