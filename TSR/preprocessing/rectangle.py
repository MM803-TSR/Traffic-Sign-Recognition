import numpy as np
import cv2
import math

img=cv2.imread('../Real_Images/speed3.png')
r, c = img.shape[:-1]
print(r, c)
#
# for c in range(3):
# 	img[:, :, c] = cv2.equalizeHist(img[:, :, c])

cv2.imshow('img', img)
cv2.waitKey(0)

kernel = np.ones((3,3),np.uint8)
erosion = cv2.erode(img,kernel,iterations = 3)
dilation = cv2.dilate(erosion,kernel,iterations = 3)
erosion = cv2.erode(dilation,kernel,iterations = 1)
retval, threshold = cv2.threshold(erosion, 120, 240, cv2.THRESH_BINARY)
cv2.imshow('img', threshold)
cv2.waitKey(0)

edges = cv2.Canny(threshold,50,150,apertureSize = 3)
cv2.imshow('img', edges)
cv2.waitKey(0)
minLineLength = min(r, c)*0.001
maxLineGap = min(r, c)*0.1
lines = cv2.HoughLinesP(edges,1,np.pi/180, 100, minLineLength=minLineLength,maxLineGap=maxLineGap)
for line in lines:
	x1,y1,x2,y2 = line[0]
	cv2.line(img,(x1,y1),(x2,y2),(0,255,0),3)

cv2.imshow('image',img)
cv2.waitKey(0)
line_img = img.copy()

for i in range(len(lines)):
	x1, y1, x2, y2 = lines[i][0]
	if x2 == x1:
		continue
	m1 = (y2 - y1) / (x2 - x1)
	A = math.atan(m1) * 180 / math.pi
	if 5 < A < 85 or 95 < A < 175 or -5 > A > -85 or -95 > A > 175:
		continue
	for j in range(i+1, len(lines)):
		x3, y3, x4, y4 = lines[j][0]
		if x4 == x3:
			continue
		I1 = [min(x1, x2)-10, max(x1, x2)+10]
		I2 = [min(x3, x4)-10, max(x3, x4)+10]
		Ia = [max(I1[0], I2[0]),
		      min(I1[1], I2[1])]
		if Ia[1] < Ia[0]:
			continue
		J1 = [min(y1, y2)-10, max(y1, y2)+10]
		J2 = [min(y3, y4)-10, max(y3, y4)+10]
		Ja = [max(J1[0], J2[0]),
		      min(J1[1], J2[1])]
		if Ja[1] < Ja[0]:
			continue


		m2 = (y4 - y3) / (x4 - x3)
		B = math.atan(m2) * 180 / math.pi
		if 5 < B < 85 or 95 < B < 175 or -5 > B > -85 or -95 > B > 175:
			continue
		if abs(A-B) > 85:
			#print(A - B)
			cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
			cv2.line(line_img, (x3, y3), (x4, y4), (0, 255, 0), 2)

cv2.imshow('image', line_img)
cv2.waitKey(0)

mask = cv2.inRange(line_img, (0, 255, 0), (0, 255, 0))
h, w = mask.shape[:2]
fill_mask = np.zeros((h+2, w+2), np.uint8)

cv2.floodFill(mask, fill_mask, (0,0), 255)
mask_inv = cv2.bitwise_not(mask)

cv2.imshow('image', mask_inv)
cv2.waitKey(0)

_, cnts, _ = cv2.findContours(mask_inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
for cnt in cnts:
	x, y, w, h = cv2.boundingRect(cnt)
	if w*h < r*c*0.0007 or w*h > r*c*0.1 or w>h or w<h*0.35:
		continue
	cv2.imshow('img', img[y:y + h, x:x + w])
	cv2.waitKey(0)

#cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


