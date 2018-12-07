
from PIL import Image
from matplotlib import pyplot as plt
import os
import sys
import glob


def paste(bg_dir, sign_dir, labels, x,y):
    global num , _size
    bgs = glob.glob(bg_dir + "/*.jpg")
    for lab in labels:
        cls_num = cls_dict[lab]
        signs = glob.glob(sign_dir + "/" + lab + "/*")
        for si in signs:
            im2 = Image.open(si)
            im2.thumbnail((resize_w,resize_h), Image.ANTIALIAS)
            abs_width,abs_height = im2.size
            abs_x = x+abs_width/2
            abs_y = y+abs_height/2
            for bgi in bgs:
                im = Image.open(bgi)
                im_width, im_height = im.size
                im.paste(im2,(x,y))
                im.save("new_im%d.jpg"%num)
                with open('new_im'+ str(num)+".txt", "w") as text_file:
                    text_file.write("%s %s %s %s %s" % (cls_num, abs_x/im_width, abs_y/im_height, abs_width/im_width, abs_height/im_height))
                num +=1
    

cls_dict = {"speedLimit80":2,"speedLimit70":17,"speedLimit40":3,"speedLimit60":16,"speedLimit50":5,"speedLimit30":11,"guide":18,"constructionahead":9,"60ahead":8,"50ahead":5,"30ahead":11}
num = 0

#resize the sign images
resize_w,resize_h = 100,50

# !!!! Change Here !!!!!
# bg folder contain street view images
# sign folder = new_dataset from drive 

bg_path = "/Users/DXX/Desktop/UACLASS/MM803/Project/Image_concat/bg"
sign_path = "/Users/DXX/Desktop/UACLASS/MM803/Project/Image_concat/sign"

labels = [f for f in os.listdir(sign_path) if not f.startswith('.')]
print(labels)

# 20,200 is the left top coord of the sign in the street image
paste(bg_path,sign_path,labels,20,200)



