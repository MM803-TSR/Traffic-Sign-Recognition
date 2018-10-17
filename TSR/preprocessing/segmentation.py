import cv2
import matplotlib.pyplot as plt
import numpy as np

# Read stop1.jpg as sample_img and display
sample_img = cv2.imread('Real_Images/stop1.jpg')
sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)
plt.imshow(sample_img)
plt.show()

# Convert to HSV
hsv_img = cv2.cvtColor(sample_img, cv2.COLOR_RGB2HSV)

# Get 2 different shades of red and obtain mask to detect any reds in-between
lower_red = (0, 50, 50)
upper_red = (10, 255, 255)
mask0 = cv2.inRange(hsv_img, lower_red, upper_red)

lower_red = (170, 50, 50)
upper_red = (180, 255, 255)
mask1 = cv2.inRange(hsv_img, lower_red, upper_red)

loose_mask = mask0 + mask1

# Display mask
plt.imshow(loose_mask, cmap='gray')
plt.show()

# TODO strict segmentation

# ret,thresh = cv2.threshold(loose_mask,125,255,0)
# above line not necessary, loose_mask is already binary
# to be deleted in next commit

# Drawing contour on loose_mask output
_, contours, _ = cv2.findContours(loose_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


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
