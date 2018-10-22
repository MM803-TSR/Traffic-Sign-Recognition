import cv2
import matplotlib.pyplot as plt
import numpy as np

# Read stop1.jpg as sample_img and display
img_loc = 'Real_Images/stop1.jpg'
sample_img = cv2.imread(img_loc)

# Convert to HSV color space
hsv_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2HSV)

# Blur image to reduce unrelated noise
blur_img = cv2.bilateralFilter(hsv_img, 9, 75, 75)
plt.imshow(blur_img)
plt.show()


# Get 2 different shades of color and obtain mask to detect any color in-between
def get_mask(img, lower_color1, upper_color1, lower_color2, upper_color2):
	masked_img1 = cv2.inRange(img, lower_color1, upper_color1)
	masked_img2 = cv2.inRange(img, lower_color2, upper_color2)
	return masked_img1 + masked_img2


def dilate_erode(img, kernel_size):
	kernel = np.ones((kernel_size, kernel_size), np.uint8)
	dila_img = cv2.dilate(img, kernel, iterations=1)
	eros_img = cv2.erode(dila_img, kernel, iterations=1)
	return eros_img


# Default is to draw on original image. Optionally we can pass blank image as 'img_to_draw'
def draw_contour(img, thickness, img_to_draw=sample_img):
	_, contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	# contours = sorted(contours, key=cv2.contourArea)[-1:]
	for i, cnt in enumerate(contours):
		colour = (0, 255, 0)
		cv2.drawContours(img_to_draw, cnt, -1, colour, thickness=thickness)


# Display loose_mask
lower_red1, upper_red1, lower_red2, upper_red2 = (0, 70, 0), (10, 255, 255), (165, 70, 0), (180, 255, 255)
loose_mask = get_mask(blur_img, lower_red1, upper_red1, lower_red2, upper_red2)

# Image cleaning by using dilation & erosion
loose_mask = dilate_erode(loose_mask, 3)
plt.imshow(loose_mask, cmap='gray')
plt.show()

size = sample_img.shape
m = np.zeros(size, dtype=np.uint8)
draw_contour(loose_mask, 3, m)
# Uncomment next block to show contour on the original image ---->
# draw_contour(loose_mask, 3)
# plt.imshow(sample_img)
# plt.show()

# Display result
plt.imshow(m)
plt.show()

# Strict Segmentation
lower_red3, upper_red3, lower_red4, upper_red4 = (0, 100, 100), (10, 255, 255), (170, 100, 100), (180, 255, 255)
strict_mask = get_mask(hsv_img, lower_red3, upper_red3, lower_red4, upper_red4)

# Image cleaning by using dilation & erosion
dilated_mask = dilate_erode(strict_mask, 15)
plt.imshow(dilated_mask, cmap='gray')
plt.show()

# Draw contours
draw_contour(dilated_mask, 5)
plt.imshow(sample_img, cmap='gray')
plt.show()

# Create mask that combines loose and strict after dilation
mask = cv2.bitwise_and(loose_mask, dilated_mask)
target = cv2.bitwise_and(sample_img, sample_img, mask=loose_mask)
plt.imshow(target)
plt.show()
