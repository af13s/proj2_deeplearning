import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras
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


print(validation[0])

# N = 100

# encodedValidation = []
# for line in tqdm(validation):
# 	if len(line) < N:
# 		array = ['0' for _ in range(0,N-len(line))]
# 		line = array + list(line[0:N])
# 		line = ''.join(line)
# 	else:
# 		line = line[0:N]
# 	for i,aminoacid in enumerate(line):
# 		encoding = d1[aminoacid]
# 		encodedValidation.append(encoding)

# encodedValidation = np.array(encodedValidation)
# print(encodedValidation.shape)

model = load_model('model.h5')
model.summary()

test = []
test.append(d1['A'])

for _ in range(0,99-len(test)):
	test.append(d1['0'])

test = np.expand_dims(test, axis=0)
prediction = model.predict_classes(test)

results = []
for c in prediction:
	for x in c:
		results.append(classes[x])

str1 = ''.join(results)
valStr = ''.join(validation)

maxLen = 0
for i in range(len(str1)):
	string = str1[:i]
	if string in valStr:
		print(string, 'in')
		maxLen = len(string)

print(maxLen)



