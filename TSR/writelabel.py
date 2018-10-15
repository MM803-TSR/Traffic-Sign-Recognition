# Convert to labled image to cvs
# Downloading the Sign Images folder from drive first

import sys
import os
import csv
Sign_dict = {}
i = 1
field = ['Class', 'Name']
# Where you need to change based on your directory
Path = '/Users/DXX/Downloads/VideoThresholder-master/Sign Images'

for img in os.listdir(Path):
    if img.endswith('.jpg'):
        Sign_dict[i] = str(img).split('.')[0]
        i += 1 

Sign_lable = open('Sign_labels.csv','w')
writer = csv.writer(Sign_lable)
writer.writerow(field)
for k,v in Sign_dict.items():
    writer.writerow([k,v])
Sign_lable.close()