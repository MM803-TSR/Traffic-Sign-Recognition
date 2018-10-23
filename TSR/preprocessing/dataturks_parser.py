import requests
import json
import os
import cv2

path_temp = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/temp/'
json_file = 'D:\Yanxi\MMGRAD\MM803\Project\Traffic Sign Dataset.json'
# print(path_temp)


def save_cropped_image(url, label, name, x1, y1, x2, y2):
	dir_path = path_temp + label + '/'
	img_temp = requests.get(url)
	img_path = dir_path + name + '.jpg'
	if img_temp.status_code == 200:
		if not os.path.isdir(dir_path):
			os.makedirs(dir_path)
		with open(img_path, 'wb') as f:
			f.write(img_temp.content)
	img = cv2.imread(img_path)
	cv2.imwrite(img_path, img[y1:y2, x1:x2])


def parse_json_object(json_object):
	img_url = json_object['content']
	img_name = json_object['content'][json_object['content'].rfind('_') + 1:json_object['content'].rfind('.')]
	annotation = json_object['annotation'][0]
	label = annotation['label'][0]
	points = annotation['points']
	h = annotation['imageHeight']
	w = annotation['imageWidth']
	x1, y1, x2, y2 = points[0][0] * w, points[0][1] * h, points[3][0] * w, points[3][1] * h

	return img_url, img_name, label, int(x1), int(y1), int(x2), int(y2)


def parse_json(json_file_path):
	f = open(json_file_path)
	train_data = f.readlines()
	train = []
	for line in train_data:
		data = json.loads(line)
		train.append(data)
	for objects in train:
		img_url, img_name, label, x1, y1, x2, y2 = parse_json_object(objects)
		# print(img_url, img_name, label, x1, y1, x2, y2)
		save_cropped_image(img_url, label, img_name, x1, y1, x2, y2)


parse_json(json_file)
