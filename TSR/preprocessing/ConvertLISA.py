import os, sys, csv
from PIL import Image

currdir = os.path.abspath(os.path.join(os.path.abspath(__file__), os.pardir))
pardir = os.path.abspath(os.path.join(currdir, os.pardir))
basedir = 'D:\Yanxi\MMGRAD\MM803\Project\signDatabasePublicFramesOnly'  # change this
toolsdir = basedir + '/tools/'
sys.path.append(toolsdir)
sys.argv.append('80')  # 80% training data - 20% testing
sys.argv.append(basedir + '/allAnnotations.csv')
import splitAnnotationFiles

sign_labels = {}

with open(pardir+'/Sign_labels.csv', 'rt') as csvfile:
	reader = csv.reader(csvfile, delimiter=',')
	for i, row in enumerate(reader):
		if i == 0:
			continue
		if row[2] != 'None':
			sign_labels[row[2]] = row[0]


def parse_csv(csv_file):
	filenames, annotations = [], []
	with open(csv_file, 'rt') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		for i, row in enumerate(reader):
			if i == 0:
				continue
			data = row[0].split(',')[0].split(';')
			tag = data[1]
			if tag not in sign_labels:
				continue
			filenames.append(data[0])
			upper_left_x, upper_left_y, lower_right_x, lower_right_y = int(data[2]), int(data[3]), int(data[4]), int(data[5])
			x, y, width, height = (upper_left_x + lower_right_x) // 2, (upper_left_y + lower_right_y) // 2, \
									lower_right_x - upper_left_x, lower_right_y - upper_left_y
			annotations.append([sign_labels[tag], x, y, width, height])
	return filenames, annotations


def convert(filenames, annotations, type):
	if not os.path.exists(basedir+'/'+type):
		os.makedirs(basedir+'/'+type)
	for i, name in enumerate(filenames):
		im = Image.open(basedir+'/'+name)
		im_width, im_height = im.size
		rgb_im = im.convert('RGB')
		rgb_im.save(basedir+'/'+type+'/'+type+str(i)+'.jpg')

		cls_num, abs_x, abs_y, abs_width, abs_height = annotations[i]

		with open(basedir+'/'+type+'/'+type+str(i)+".txt", "w") as text_file:
			text_file.write("%s %s %s %s %s" % (cls_num, abs_x/im_width, abs_y/im_height, abs_width/im_width, abs_height/im_height))

		print('Generating '+type+str(i))


train_img, train_annot = parse_csv(basedir+'\split1.csv')
convert(train_img, train_annot, 'Train')
test_img, test_annot = parse_csv(basedir+'\split2.csv')
convert(test_img, test_annot, 'Test')
