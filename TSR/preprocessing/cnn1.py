import numpy as np
import os
import glob
import h5py
import pickle
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras import backend as K
from keras.models import load_model
from keras.models import model_from_json
K.set_image_data_format('channels_first')

from matplotlib import pyplot as plt

from sklearn.utils import shuffle


# # test dataset
# data_dir = '../data/test/'
# training_file = data_dir+'train.p'
# testing_file = data_dir+'test.p'
#
# with open(training_file, mode='rb') as f:
#     train = pickle.load(f)
# with open(testing_file, mode='rb') as f:
#     test = pickle.load(f)
#
# X_train, y_train_temp = train['features'], train['labels']
# X_test, y_temp = test['features'], test['labels']
#
# y_train = []
#
# for i, y in enumerate(y_train_temp):
# 	y_row = np.eye(44, dtype='uint8')[int(y)]
# 	y_train.append(y_row)
#
# y_train = np.asarray(y_train)



data_dir = '../data/'
X_train_file, y_train_file = os.path.join(data_dir, 'X_train.p'), os.path.join(data_dir, 'y_train.p')
X_test_file, y_test_file = os.path.join(data_dir, 'X_test.p'), os.path.join(data_dir, 'y_test.p')

with open(X_train_file, mode='rb') as f:
	X_train = pickle.load(f)
with open(y_train_file, mode='rb') as f:
	y_temp = pickle.load(f)
with open(X_test_file, mode='rb') as f:
	X_test = pickle.load(f)
with open(y_test_file, mode='rb') as f:
	y_test = pickle.load(f)

print('X_train 10')
print(X_train[10])


y_train = []

for i, y in enumerate(y_temp):
	y_row = np.eye(37, dtype='uint8')[int(y)]
	y_train.append(y_row)

y_train = np.asarray(y_train)


def display_train():
	fig, axes = plt.subplots(3, 15, figsize=(15, 3))
	i = 0
	for ax in axes.flatten():
		ax.imshow(X_train[i * 150])
		if i >= 45:
			break
		i += 1
	fig.show()


def AbsNorm(image, a=-.5, b=0.5, col_min=0, col_max=255):
	return (image-col_min)*(b-a)/(col_max-col_min)


def contrast_norm(image) :
	## non adaptive hist equal
	img = image.astype(np.uint8)
	for c in range(3):
		img[:, :, c] = cv2.equalizeHist(img[:, :, c])


	# ## adaptive hist equal
	# lab= cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
	# # apply adative histogram equalization
	# l = lab[:,:,0].astype(np.uint8)
	# clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4,4))
	# cl = clahe.apply(l)
	# lab[:,:,0] = cl
	# # convert back to RGB and scale values
	# img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

	img = np.stack([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)]*3, axis=-1)

	new_img = np.zeros(image.shape)
	for i in range(3):
		#new_img[:,:,i] = AbsNorm(img[:,:,i])
		img[:, :, i] = img[:, :, i] * (255.0 / img[:, :, i].max())

	#print(img.shape)
	new_img = np.ones(img.shape)
	#img = np.rollaxis(img, -1)

	return img

X_train_new, X_test_new = [], []

def preprocess():
	for i, image in enumerate(X_train):
		temp = contrast_norm(X_train[i])
		X_train_new.append(np.moveaxis(temp, 2, 0))
		#X_train[i] = np.rollaxis(X_train[i], -1)

	for i, image in enumerate(X_test):
		X_test[i] = contrast_norm(image)
		X_test_new.append(np.moveaxis(X_test[i], 2, 0))


# display_train()
preprocess()
# display_train()

print(X_train.shape)
print(np.asarray(X_train_new).shape)
print(y_train.shape)
print(X_train[10])
print(X_train_new[2])

def cnn_model():
	model = Sequential()

	model.add(Conv2D(32, (3, 3), padding='same',
					 input_shape=(3, 32, 32),
					 activation='relu'))
	model.add(Conv2D(32, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))

	model.add(Conv2D(64, (3, 3), padding='same',
					 activation='relu'))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))

	model.add(Conv2D(128, (3, 3), padding='same',
					 activation='relu'))
	model.add(Conv2D(128, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))

	model.add(Flatten())
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(37, activation='softmax'))
	return model

def cnn_model1():
	model = Sequential()

	model.add(Conv2D(32, (3, 3), padding='same',
					 input_shape=(3, 32, 32),
					 activation='relu'))
	model.add(Conv2D(32, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))

	model.add(Conv2D(64, (3, 3), padding='same',
					 activation='relu'))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))

	model.add(Conv2D(128, (3, 3), padding='same',
					 activation='relu'))
	model.add(Conv2D(128, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))

	model.add(Flatten())
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(37, activation='softmax'))
	return model

model = cnn_model1()
# let's train the model using SGD + momentum (how original).
lr = 0.000001
lr1 = 0.0001
sgd = SGD(lr=lr1, decay=1e-6, momentum=0.0, nesterov=True)
model.compile(loss='binary_crossentropy',
		  optimizer='adam',
		  metrics=['accuracy'])


def lr_schedule(epoch):
	return lr1*(0.1**int(epoch/10))

batch_size = 32
nb_epoch = 100


X_train_new, y_train = shuffle(X_train_new, y_train)

model = load_model('final_model1.h5')
model.fit(np.asarray(X_train_new), y_train,
		  batch_size=batch_size,
		  epochs=nb_epoch,
		  validation_split=0.2,
		  shuffle=True,
		  callbacks=[LearningRateScheduler(lr_schedule),
					ModelCheckpoint('final_model1.h5',save_best_only=True)]
			)
model.save('final_model1.h5')

y_pred = model.predict(np.asarray(X_test_new), verbose=1)
y_pred_class = []
for i in y_pred:
	y_pred_class.append(np.argmax(i))
print(y_test)
print(y_pred_class)
#score = [int(y_test[i]) == int(y_pred[i]) for i in range(len(y_test))]
#print(score)
#acc = sum(int(y_test[i]) == int(y_pred[i]) for i in range(len(y_test)))/len(y_test)

#print("Test accuracy = {}".format(acc))


## save model to json and h5
model_json = model.to_json()

with open("final_model1_read.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("final_model1_read.h5")
