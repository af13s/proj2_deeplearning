# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)
# set_session(sess)  # set this TensorFlow session as the default session for Keras
from keras.models import load_model
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import random

print('Loading data')

temp = open("pdb_seqres.txt",'r')
temp = temp.read().splitlines()

validation = []

for i in range(len(temp)):
	if (i+1)% 51 == 0:
		validation.append(temp[i])

classes = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y','\n']
protein_label_encoder = pd.factorize(classes)
encoder = OneHotEncoder()
protein_labels_1hot = encoder.fit_transform(protein_label_encoder[0].reshape(-1,1))
onehot_array = protein_labels_1hot.toarray()
d1 = dict(zip(classes,onehot_array.tolist()))
d1['0']=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]

model = load_model('model.h5')
model.summary()


maxCount = 0
for proteinSequence in tqdm(validation[:100]):
	count = 0
	if len(proteinSequence) < 99:
		continue
	sequence = list(proteinSequence[:99])
	trueLabel = proteinSequence[99]
	#print(''.join(sequence))
	w = 0
	while True:

		randomLetter = random.choice(classes)

		while randomLetter == '\n':
			randomLetter = random.choice(classes)
		
		#print(randomLetter)

		index = random.randint(0,len(sequence)-1)

		#print(index)

		sequence[w] = randomLetter

		#print(''.join(sequence))
		
		test = []

		for char in sequence:
			test.append(d1[char])

		test = np.expand_dims(test, axis=0)
		prediction = model.predict_classes(test)

		results = []
		for c in prediction:
			for x in c:
				results.append(classes[x])

		if results[-1] == trueLabel:
			#print('Correct!')
			count+=1
			w+=1
		else:
			#print('Incorrect')
			#print('Count',count)
			if count > maxCount:
				maxCount = count
			break


print(maxCount)
quit()


maxK = [0 for _ in range(20)]
for k in range(20):
	for proteinSequence in tqdm(validation[:1000]):
		numCorrect = 0
		predictionValue = True
		while predictionValue == True:
			trueLabel = proteinSequence[k]
			sequence = list(proteinSequence[numCorrect:k+numCorrect])
			test = []

			for char in sequence:
				test.append(d1[char])

			temp = []
			for _ in range(0,99-len(test)):
				temp.append(d1['0'])

			temp.extend(test)

			test = np.expand_dims(temp, axis=0)
			prediction = model.predict_classes(test)

			results = []
			for c in prediction:
				for x in c:
					results.append(classes[x])

			if results[-1] == trueLabel:
				numCorrect+=1
			else:
				predictionValue = False
		if maxK[k] < numCorrect:
			maxK[k] = numCorrect
	print(k,maxK[k])


print(maxK)
