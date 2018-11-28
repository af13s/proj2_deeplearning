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

def vec2string(prediction):
	results = []
	for c in prediction:
		for x in c:
			results.append(classes[x])

	return ''.join(results)


print(validation[0])

N = 100

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
print(encodedValidation.shape)

encodedValidation = np.array(encodedValidation)
encodedValidation = encodedValidation.reshape(-1,N,21)

model = load_model('model.h5')
model.summary()

total = 0

testX = []
testY = []

for sample in encodedValidation:
	testX.append(sample[0:N-1])
	testY.append(sample[1:N])

testX = np.array(testX)
testY = np.array(testY)


# for sequence100 in testX:
# 	sequence100 = np.expand_dims(sequence100, axis=0)
# 	prediction = model.predict_classes(sequence100)

# 	print([np.array(testY[0].argmax(axis=1))])
# 	print(prediction)

for i in range(len(testX)):
	temp = testX[i][:]
	temp = np.expand_dims(temp, axis=0)
	prediction = model.predict_classes(temp)

	print("Test String: ", vec2string([np.array(testX[i].argmax(axis=1))]))
	print("Actual String: ", vec2string([np.array(testY[i].argmax(axis=1))]))
	print("predicted: ", vec2string(prediction))

	# if vec2string(prediction) != vec2string([np.array(testY[i].argmax(axis=1))]):
	# 	print ("False Prediction")
	# else:
	# 	print("True Prediction")

	
	# for i in range(len(sequence100)):
	# 	seqcopy = sequence100[:]
	# 	# seqcopy[i] = np.array(d1['0'])
	# 	new_prediction = model.predict_classes(seqcopy)

	# 	print(vec2string(new_prediction), vec2string(prediction))
	# 	# if new_prediction[len(prediction)-1] != prediction[len(prediction)-1]:
	# 	# 	total += i
	# 	# 	break




total /= len(encodedValidation)
	

# results = []
# for c in prediction:
# 	for x in c:
# 		results.append(classes[x])

# str1 = ''.join(results)
# valStr = ''.join(validation)

# maxLen = 0
# for i in range(len(str1)):
# 	string = str1[:i]
# 	if string in valStr:
# 		print(string, 'in')
# 		maxLen = len(string)

# print(maxLen)



