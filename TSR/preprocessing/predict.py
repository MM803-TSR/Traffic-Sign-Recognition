from keras.models import model_from_json
import cv2
import numpy as np

json_file = open("model_num.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("model_num.h5")


def contrast_norm(image):
	img = cv2.resize(image, (32, 32))
	img = img.astype(np.uint8)
	for c in range(3):
		img[:, :, c] = cv2.equalizeHist(img[:, :, c])
	img = np.stack([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)]*3, axis=-1)
	for i in range(3):
		img[:, :, i] = img[:, :, i] * (255.0 / img[:, :, i].max())
	img = np.moveaxis(img, 2, 0)

	return np.expand_dims(img, axis=0)


def predict(img):
	img = cv2.imread(img)
	img = contrast_norm(img)
	out = model.predict(img)
	pred = np.argmax(out)
	prob = out[0][np.argmax(out)]
	return pred, prob


THREASHOLD = 0.7

impath = '../Real_Images/stop.jpg'
print(predict(impath))

