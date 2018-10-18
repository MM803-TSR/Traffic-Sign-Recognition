import cv2
import matplotlib.pyplot as plt
import numpy as np

# Read stop1.jpg as sample_img and display
img_loc = 'Real_Images/stop1.jpg'
sample_img = cv2.imread(img_loc)

# Convert to RGB color space
orig_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)
plt.imshow(orig_img)
plt.show()
# Convert to HSV color space
hsv_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2HSV)

# Blur image to reduce unrelated noise
blur_img = cv2.bilateralFilter(hsv_img,9,75,75)
plt.imshow(blur_img)
plt.show()

# Get 2 different shades of red and obtain mask to detect any reds in-between
lower_red = (0, 70, 0)
upper_red = (10, 255, 255)
masked_img1 = cv2.inRange(blur_img, lower_red, upper_red)
lower_red = (165, 70, 0)
upper_red = (180, 255, 255)
masked_img2 = cv2.inRange(blur_img, lower_red, upper_red)
color_masked_img = masked_img1 + masked_img2

# Display the color_segmented image
plt.imshow(color_masked_img,cmap='gray')
plt.show()

# Image cleaning by using dilation & erosion
kernel = np.ones((3,3),np.uint8) 
dila_img = cv2.dilate(color_masked_img,kernel,iterations = 1)
eros_img = cv2.erode(dila_img,kernel,iterations = 1)
plt.imshow(eros_img,cmap='gray')
plt.show()

ret,thresh = cv2.threshold(loose_mask,125,255,0)
# above line not necessary, loose_mask is already binary
# to be deleted in next commit

# Draw contour on loose_mask output
_, contours, _ = cv2.findContours(color_masked_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
sorted_contour = sorted(contours, key=cv2.contourArea)[-1:] # let's discuss

size = sample_img.shape
m = np.zeros(size, dtype=np.uint8) 
#Uncomment next line to show contour on the original image ----> 
#m = sample_img

for i, cnt in enumerate(sorted_contour):
    color = (0,255,0)
    cv2.drawContours(m, cnt, -1, color, thickness=5)
    
#Save the output image for a larger view
#cv2.imwrite("contours.jpg", m);

# Display result
plt.imshow(m)
plt.show()

# Strict Segmentation
lower_red = (0, 100, 100)
upper_red = (10, 255, 255)
mask2 = cv2.inRange(hsv_img, lower_red, upper_red)

lower_red = (170, 100, 100)
upper_red = (180, 255, 255)
mask3 = cv2.inRange(hsv_img, lower_red, upper_red)

strict_mask = mask2 + mask3

# Display mask
plt.imshow(strict_mask, cmap='gray')
plt.show()

# Morphological dilation
kernel = np.ones((19, 19), np.uint8)
dilated_mask = cv2.dilate(strict_mask, kernel, iterations=1)
plt.imshow(dilated_mask, cmap='gray')
plt.show()

# Draw contours
_, contours, _ = cv2.findContours(dilated_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
sorted_contour = sorted(contours, key=cv2.contourArea)

size = sample_img.shape
m = np.zeros(size, dtype=np.uint8)

#Uncomment next line to show contour on the original image ---->
#m = sample_img

for i, cnt in enumerate(sorted_contour):
    color = (0, 255, 0)
    cv2.drawContours(m, cnt, -1, color, thickness=5)

plt.imshow(m, cmap='gray')
plt.show()

# Create mask that combines loose and strict after dilation
mask = cv2.bitwise_and(loose_mask, dilated_mask)
target = cv2.bitwise_and(sample_img, sample_img, mask=loose_mask)
plt.imshow(target)
plt.show()
