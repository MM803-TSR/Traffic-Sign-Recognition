import tensorflow as tf
from tensorflow.contrib.layers import flatten
import numpy as np
import pickle
import os
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import cv2


keep_prob = tf.placeholder(tf.float32)  # for fully-connected layers
keep_prob_conv = tf.placeholder(tf.float32)  # for convolutional layers
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))


class VGGnet:

	def __init__(self, n_out=43, mu=0, sigma=0.1, learning_rate=0.0007):
		# Hyperparameters
		self.mu = mu
		self.sigma = sigma

		# Layer 1: Convolutional. Input = 32x32x3. Output = 32x32x32.
		self.conv1_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 3, 32), mean=self.mu, stddev=self.sigma))
		self.conv1_b = tf.Variable(tf.zeros(32))
		self.conv1 = tf.nn.conv2d(x, self.conv1_W, strides=[1, 1, 1, 1], padding='SAME') + self.conv1_b

		# ReLu Activation.
		self.conv1 = tf.nn.relu(self.conv1)

		# Layer 2: Convolutional. Input = 32x32x32. Output = 32x32x32.
		self.conv2_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 32, 32), mean=self.mu, stddev=self.sigma))
		self.conv2_b = tf.Variable(tf.zeros(32))
		self.conv2 = tf.nn.conv2d(self.conv1, self.conv2_W, strides=[1, 1, 1, 1], padding='SAME') + self.conv2_b

		# ReLu Activation.
		self.conv2 = tf.nn.relu(self.conv2)

		# Pooling. Input = 32x32x32. Output = 16x16x32.
		self.conv2 = tf.nn.max_pool(self.conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
		self.conv2 = tf.nn.dropout(self.conv2, keep_prob_conv)  # dropout

		# Layer 3: Convolutional. Input = 16x16x32. Output = 16x16x64.
		self.conv3_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 32, 64), mean=self.mu, stddev=self.sigma))
		self.conv3_b = tf.Variable(tf.zeros(64))
		self.conv3 = tf.nn.conv2d(self.conv2, self.conv3_W, strides=[1, 1, 1, 1], padding='SAME') + self.conv3_b

		# ReLu Activation.
		self.conv3 = tf.nn.relu(self.conv3)

		# Layer 4: Convolutional. Input = 16x16x32. Output = 16x16x64.
		self.conv4_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 64, 64), mean=self.mu, stddev=self.sigma))
		self.conv4_b = tf.Variable(tf.zeros(64))
		self.conv4 = tf.nn.conv2d(self.conv3, self.conv4_W, strides=[1, 1, 1, 1], padding='SAME') + self.conv4_b

		# ReLu Activation.
		self.conv4 = tf.nn.relu(self.conv4)

		# Pooling. Input = 16x16x64. Output = 8x8x64.
		self.conv4 = tf.nn.max_pool(self.conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
		self.conv4 = tf.nn.dropout(self.conv4, keep_prob_conv)  # dropout

		# Layer 5: Convolutional. Input = 8x8x64. Output = 8x8x128.
		self.conv5_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 64, 128), mean=self.mu, stddev=self.sigma))
		self.conv5_b = tf.Variable(tf.zeros(128))
		self.conv5 = tf.nn.conv2d(self.conv4, self.conv5_W, strides=[1, 1, 1, 1], padding='SAME') + self.conv5_b

		# ReLu Activation.
		self.conv5 = tf.nn.relu(self.conv5)

		# Layer 6: Convolutional. Input = 8x8x128. Output = 8x8x128.
		self.conv6_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 128, 128), mean=self.mu, stddev=self.sigma))
		self.conv6_b = tf.Variable(tf.zeros(128))
		self.conv6 = tf.nn.conv2d(self.conv5, self.conv6_W, strides=[1, 1, 1, 1], padding='SAME') + self.conv6_b

		# ReLu Activation.
		self.conv6 = tf.nn.relu(self.conv6)

		# Pooling. Input = 8x8x128. Output = 4x4x128.
		self.conv6 = tf.nn.max_pool(self.conv6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
		self.conv6 = tf.nn.dropout(self.conv6, keep_prob_conv)  # dropout

		# Flatten. Input = 4x4x128. Output = 2048.
		self.fc0 = flatten(self.conv6)

		# Layer 7: Fully Connected. Input = 2048. Output = 128.
		self.fc1_W = tf.Variable(tf.truncated_normal(shape=(2048, 128), mean=self.mu, stddev=self.sigma))
		self.fc1_b = tf.Variable(tf.zeros(128))
		self.fc1 = tf.matmul(self.fc0, self.fc1_W) + self.fc1_b

		# ReLu Activation.
		self.fc1 = tf.nn.relu(self.fc1)
		self.fc1 = tf.nn.dropout(self.fc1, keep_prob)  # dropout

		# Layer 8: Fully Connected. Input = 128. Output = 128.
		self.fc2_W = tf.Variable(tf.truncated_normal(shape=(128, 128), mean=self.mu, stddev=self.sigma))
		self.fc2_b = tf.Variable(tf.zeros(128))
		self.fc2 = tf.matmul(self.fc1, self.fc2_W) + self.fc2_b

		# ReLu Activation.
		self.fc2 = tf.nn.relu(self.fc2)
		self.fc2 = tf.nn.dropout(self.fc2, keep_prob)  # dropout

		# Layer 9: Fully Connected. Input = 128. Output = n_out.
		self.fc3_W = tf.Variable(tf.truncated_normal(shape=(128, n_out), mean=self.mu, stddev=self.sigma))
		self.fc3_b = tf.Variable(tf.zeros(n_out))
		self.logits = tf.matmul(self.fc2, self.fc3_W) + self.fc3_b

		# training operation
		self.one_hot_y = tf.one_hot(y, n_out)
		self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.one_hot_y)
		self.loss_operation = tf.reduce_mean(self.cross_entropy)
		self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
		self.training_operation = self.optimizer.minimize(self.loss_operation)
		self.train_step = tf.train.GradientDescentOptimizer(0.01).minimize(self.cross_entropy)

		# accuracy
		self.correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.one_hot_y, 1))
		self.accuracy_operation = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

		# save all variables
		self.saver = tf.train.Saver()

	def y_predict(self, X_data, BATCH_SIZE=64):
		num_examples = len(X_data)
		y_pred = np.zeros(num_examples, dtype=np.int32)
		sess = tf.get_default_session()
		for offset in range(0, num_examples, BATCH_SIZE):
			batch_x = X_data[offset:offset + BATCH_SIZE]
			y_pred[offset:offset + BATCH_SIZE] = sess.run(tf.argmax(self.logits, 1),
			                                              feed_dict={x: batch_x, keep_prob: 1, keep_prob_conv: 1})
		return y_pred

	def evaluate(self, X_data, y_data, BATCH_SIZE=64):
		num_examples = len(X_data)
		total_accuracy = 0
		sess = tf.get_default_session()
		for offset in range(0, num_examples, BATCH_SIZE):
			batch_x, batch_y = X_data[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
			accuracy = sess.run(self.accuracy_operation,
			                    feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0, keep_prob_conv: 1.0})
			total_accuracy += (accuracy * len(batch_x))
		return total_accuracy / num_examples



data_dir = '../data/'
X_train_file, y_train_file = os.path.join(data_dir, 'X_train.p'), os.path.join(data_dir, 'y_train.p')
X_test_file, y_test_file = os.path.join(data_dir, 'X_test.p'), os.path.join(data_dir, 'y_test.p')

with open(X_train_file, mode='rb') as f:
	X_train = pickle.load(f)
with open(y_train_file, mode='rb') as f:
	y_train = pickle.load(f)
with open(X_test_file, mode='rb') as f:
	X_test = pickle.load(f)
with open(y_test_file, mode='rb') as f:
	y_test = pickle.load(f)


## test dataset
# data_dir = '../data/test/'
# training_file = data_dir+'train.p'
# testing_file = data_dir+'test.p'
#
# with open(training_file, mode='rb') as f:
#     train = pickle.load(f)
# with open(testing_file, mode='rb') as f:
#     test = pickle.load(f)
#
# X_train, y_train = train['features'], train['labels']
# X_test, y_test = test['features'], test['labels']


### Prepare data

# print(y_test)
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


def contrast_norm(image):
	# convert to  lab colorspace
	lab= cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
	# apply adative histogram equalization
	l = lab[:,:,0]
	clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4,4))
	cl = clahe.apply(l)
	lab[:,:,0] = cl
	# convert back to RGB and scale values
	img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
	new_img = np.zeros(image.shape)
	for i in range(3):
		#new_img[:,:,i] = AbsNorm(img[:,:,i])
		img[:, :, i] = img[:, :, i] * (255.0 / img[:, :, i].max())
	return img

X_train_test, y_train_test = [], []

def preprocess():
	for i, image in enumerate(X_train):
		X_train[i] = contrast_norm(image)

	for i, image in enumerate(X_test):
		X_test[i] = contrast_norm(image)

	count = 0

	for i, y in enumerate(y_train):
		if y.tolist() == 2:
			if count > 0:
				continue
			else:
				count += 1
		X_train_test.append(X_train[i])
		y_train_test.append(y)


display_train()
preprocess()
display_train()

print(X_train.shape)
print(y_train.shape)

print(np.asarray(X_train_test).shape)
print(np.asarray(y_train_test).shape)

### Train model

DIR = '../models/'
if not os.path.exists(DIR):
	os.makedirs(DIR)


### train model 1
EPOCHS = 10
BATCH_SIZE = 64
model_1 = VGGnet(n_out=18)
model_name = "model_2"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
	sess.run(tf.global_variables_initializer())
	num_examples = len(y_train_test)
	print("Training...")
	print()
	for i in range(EPOCHS):
		X_train, y_train = shuffle(X_train_test, y_train_test)
		for offset in range(0, num_examples, BATCH_SIZE):
			end = offset + BATCH_SIZE
			batch_x, batch_y = X_train[offset:end], y_train[offset:end]
			_, loss_val = sess.run([model_1.train_step, model_1.training_operation], feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5, keep_prob_conv: 0.7})
			print('loss = ' + loss_val)
		#validation_accuracy = model_1.evaluate(X_validation, y_validation)
		#print("EPOCH {} : Validation Accuracy = {:.3f}".format(i+1, validation_accuracy))
		print("EPOCH {}".format(i + 1))

	model_1.saver.save(sess, os.path.join(DIR, model_name))
	print("Model saved")

with tf.Session() as sess:
	model_1.saver.restore(sess, os.path.join(DIR, "model_2"))
	#test_accuracy = model_1.evaluate(X_test, y_test)
	y_pred = model_1.y_predict(X_test)
	print(y_pred)
	test_accuracy = sum(int(y_test[i]) == int(y_pred[i]) for i in range(len(y_test)))/len(y_test)
	print("Test Accuracy = {:.3f}".format(test_accuracy))

