import os, sys, csv
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle
from PIL import Image

# geometric transformations

# scaling
def scale(img, x_scale=1.9, y_scale=1.5):
	r, c = img.shape[:-1]
	scaled_img = cv2.resize(img, None, fx=x_scale, fy=y_scale)
	new_img = cv2.copyMakeBorder(scaled_img, 5, 5, 5, 5, cv2.BORDER_REPLICATE)
	rows, cols = new_img.shape[:-1]
	r0, c0 = round((rows - r) / 2), round((cols - c) / 2)
	return new_img[r0:r0 + r, c0:c0 + c]


# translation
def translate(img, delta_x=2, delta_y=-2):
	r, c = img.shape[:-1]
	new_img = cv2.copyMakeBorder(img, int(r*0.1), int(r*0.1), int(r*0.1), int(r*0.1), cv2.BORDER_REPLICATE)
	rows, cols = new_img.shape[:-1]
	r0, c0 = round((rows - r) / 2) + delta_y, round((cols - c) / 2) - delta_x
	return new_img[r0:r0 + r, c0:c0 + c]


# rotation
def rotate(img, theta=18):
	r, c = img.shape[:-1]
	new_img = cv2.copyMakeBorder(img, int(r*0.1), int(r*0.1), int(r*0.1), int(r*0.1), cv2.BORDER_REPLICATE)
	rows, cols = new_img.shape[:-1]
	M = cv2.getRotationMatrix2D((cols / 2, rows / 2), theta, 1)
	new_img = cv2.warpAffine(new_img, M, (cols, rows))
	r0, c0 = round((rows - r) / 2), round((cols - c) / 2)
	return new_img[r0:r0 + r, c0:c0 + c]


def warp(img, theta=0.15):
	r, c = img.shape[:-1]
	new_img1 = cv2.resize(img, (int(c * (1+2*theta)), int(r * (1+2*theta))))
	new_img = cv2.copyMakeBorder(img, int(r * 2*theta), int(r * 2*theta), int(c * 2*theta), int(c * 2*theta), cv2.BORDER_REPLICATE)
	plt.imshow(new_img)
	plt.show()
	t1, t2, t3 = theta, (1+theta), (1+2*theta)
	dst1, dst2, dst3, dst4 = np.float32([[0, int(t1*c)], [r, 0], [0, t2*c], [r, t3*c]]), \
							np.float32([[0, 0], [r, t1*c], [0, t3*c], [r, t2*c]]), \
							np.float32([[t1*r, 0], [t2*r, 0], [0, c], [t3*r, c]]), \
							np.float32([[0, 0], [t3*r, 0], [t1*r, c], [t2*r, c]])
	pst = np.float32([[0, 0], [t3*r, 0], [0, t3*c], [t3*r, t3*c]])
	M1, M2, M3, M4 = cv2.getPerspectiveTransform(pst, dst1), \
					cv2.getPerspectiveTransform(pst, dst2), \
					cv2.getPerspectiveTransform(pst, dst3), \
					cv2.getPerspectiveTransform(pst, dst4)

	warped1 = cv2.warpPerspective(new_img, M1, (int(t3*c), int(t3 * r)))[int(r*2*theta):int((1+2*theta)*r), int(c*theta):int((1+theta)*c)]
	warped2 = cv2.warpPerspective(new_img, M2, (int(t3*c), int(t3 * r)))[int(r*2*theta):int((1+2*theta)*r), int(c*theta):int((1+theta)*c)]
	warped3 = cv2.warpPerspective(new_img, M3, (int(t3*c), int(t3 * r)))[int(r*theta):int((1+theta)*r), int(c*2*theta):int((1+2*theta)*c)]
	warped4 = cv2.warpPerspective(new_img, M4, (int(t3*c), int(t3 * r)))[int(r*theta):int((1+theta)*r), int(c*2*theta):int((1+2*theta)*c)]

	return [warped1, warped2, warped3, warped4]


def transform_img(imdir, imname):
	img = cv2.imread(imdir+imname)
	transforms = []
	r, c = img.shape[:-1]

	def helper(image):
		print(image.shape)
		transforms.append(scale(image, 1, 1.5))
		transforms.append(scale(image, 1.5, 1))
		transforms.append(rotate(img, 18))
		transforms.append(rotate(img, -18))
		transforms.append(translate(image, int(r*0.1), int(-r*0.1)))
		transforms.append(translate(image, int(-r*0.1), int(r*0.1)))
		for im in warp(image):
			h, w = im.shape[:-1]
			dr, dc = int(h*0.1), int(w*0.1)
			transforms.append(im[dc:w-dc, dr:h-dr])

	helper(img)

	for _ in range(0):
		temp = transforms[:]
		print(len(temp))
		for image in temp:
			plt.imshow(image)
			plt.show()
			helper(image)

	for i, image in enumerate(transforms):
		cv2.imwrite(imdir+str(i)+imname, image)



# names = os.listdir(directory)
# for name in names:
# 	transform_img(directory, name)

directory = 'D:/Yanxi/MMGRAD/MM803/Project/new dataset1/train/'
categories = os.listdir(directory)
X, y = [], []
for subfolder in categories:
	subdir = directory+subfolder
	imnames = os.listdir(subdir)
	# for imname in imnames:
	# 	if 'png' in imname.split('.'):
	# 		im = Image.open(subdir+'/'+imname)
	# 		rgb_im = im.convert('RGB')
	# 		rgb_im.save(subdir+'/'+imname+'.jpg')

	for imname in imnames:
		imdir = subdir+'/'+imname
		im = cv2.imread(imdir)
		out = cv2.resize(im, (32, 32))
		X.append(np.asarray(out))

		# # for training
		# y_row = np.eye(4, dtype='uint8')[int(subfolder)]
		# y.append(y_row)

		# for testing
		y.append(np.asarray(int(subfolder)))


print(np.asarray(X).shape, len(y))

with open(directory + 'X_train.p', 'wb') as f:
		pickle.dump(np.array(X), f)

with open(directory + 'y_train.p', 'wb') as f:
		pickle.dump(np.array(y), f)

