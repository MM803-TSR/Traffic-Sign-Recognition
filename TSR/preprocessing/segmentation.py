import cv2
import matplotlib.pyplot as plt
import numpy as np


def load_prep_img(img_path):
    sample_img = cv2.imread(img_path)
    size = sample_img.shape
    blank_canvas = np.zeros(size, dtype=np.uint8)
    # Convert to RGB color space
    rgb_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)
    # Convert to HSV color space
    hsv_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2HSV)
    # Blur image to reduce unrelated noise
    blur_img = cv2.bilateralFilter(hsv_img, 9, 75, 75)
    return blank_canvas, sample_img, rgb_img, hsv_img, blur_img


def get_red_mask(img, lower_color1, upper_color1, lower_color2, upper_color2):
    masked_img1 = cv2.inRange(img, lower_color1, upper_color1)
    masked_img2 = cv2.inRange(img, lower_color2, upper_color2)
    red_mask = masked_img1 + masked_img2
    return red_mask


def get_yellow_mask(img, lower_color, upper_color):
    masked_img = cv2.inRange(img, lower_color, upper_color)
    yellow_mask = masked_img
    return yellow_mask


def dilate_erode(img, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_img = cv2.dilate(img, kernel, iterations=1)
    cleaned_img = cv2.erode(dilated_img, kernel, iterations=1)
    return cleaned_img


def find_contour(img):
    _, all_contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    all_sorted_cnt = sorted(all_contours, key=cv2.contourArea)[-1:]
    largest_cnt = all_sorted_cnt[0]
    return largest_cnt, all_sorted_cnt


def draw_contour(img, canvas, contour_thickness):
    largest_cnt, all_sorted_cnt = find_contour(img)
    for i, largest_cnt in enumerate(all_sorted_cnt):
        cv2.drawContours(canvas, largest_cnt, -1, [0, 255, 0], contour_thickness)


def draw_box(img, canvas, contour_thickness):
    pos = []
    _, all_sorted_cnt = find_contour(img)
    for i, cnt in enumerate(all_sorted_cnt):
        colour = (0, 255, 0)
        x, y, w, h = cv2.boundingRect(cnt)
        if abs(w - h) > 1:
            cv2.rectangle(canvas, (x, y), (x + w, y + h), colour, contour_thickness)
            pos.append([x, y, w, h])
    return pos, canvas


def fix_convex_defect(img):
    # Detect convexity defects base on the contour.
    largest_cnt, _ = find_contour(img)
    hulls = cv2.convexHull(largest_cnt, returnPoints=False)
    convexity_defects = cv2.convexityDefects(largest_cnt, hulls)
    for i in range(convexity_defects.shape[0]):
        s_before, e_before, f_before, d_before = convexity_defects[i, 0]
        start_before = tuple(largest_cnt[s_before][0])
        end_before = tuple(largest_cnt[e_before][0])
        far_before = tuple(largest_cnt[f_before][0])
        cv2.line(img, start_before, end_before, [255, 0, 0], 1)
    convex_fix_kernel = np.ones((3, 3), np.uint8)
    convex_dilated_img = cv2.dilate(img, convex_fix_kernel, iterations=1)
    convex_fixed_img = cv2.erode(convex_dilated_img, convex_fix_kernel, iterations=1)
    return convex_fixed_img


def load_template(template_path):
    template = cv2.imread(template_path)
    blurred_template = cv2.bilateralFilter(template,9,75,75)
    gray_scale_template = cv2.cvtColor(blurred_template, cv2.COLOR_BGR2GRAY)
    ret, thresh2 = cv2.threshold(gray_scale_template, 127, 255, 0)
    _, template_contours, _ = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    template_cnt = template_contours[0]
    return gray_scale_template, template_cnt


def shape_compare(contours, template_cnt):
    similarity_level = cv2.matchShapes(contours, template_cnt, 1, 0.0)
    return similarity_level


def expend_border(original_img, region):
    [img_h,img_w] = original_img.shape[:2]
    [x,y,w,h] = r
    # Expending a border around target area
    crop_x = [x-border, x+w+border]
    crop_y = [y-border, y+h+border]
    if crop_x[0]<0 or crop_x[1]>img_w:
        crop_x = [x, x+w+50]
    if crop_y[0]<0 or crop_y[1]>img_h:
        crop_y = [y, y+h+50]
    crop_w = crop_x[1]-crop_x[0]
    crop_h = crop_y[1]-crop_y[0]
    return crop_x[0], crop_y[0], crop_w, crop_h


def crop_and_hist(original_img, region, border):
    x,y,w,h = expend_border(original_img, region)
    crop_mask = np.zeros((h,w,3), dtype=np.uint8)
    target = original_img[y:y+h,x:x+w,:].copy()
    # Do histigram equalization over interested region
    H,S,V = cv2.split(cv2.cvtColor(target, cv2.COLOR_BGR2HSV))
    eq_v = cv2.equalizeHist(V)
    eq_target = cv2.cvtColor(cv2.merge([H, S, eq_v]), cv2.COLOR_HSV2RGB)
    eq_img = original_img.copy()
    eq_img[y:y+h,x:x+w,:] = eq_target
    return target, eq_target


def show_hist(roi_hist):
    color = ['b','g','r']
    for i, c in  enumerate(color):
        hist_target = cv2.calcHist([roi_hist],[i],None,[256],[0,256])
        plt.plot(hist_target,color = c)
        plt.xlim([0,256])
    plt.show()


#------------------------ Main ------------------------------------

# Load in testing images
stop_img_loc = 'Real_Images/stop1.jpg'
blank_canvas_1, stop_sample_img, stop_rgb_img, stop_hsv_img, stop_blur_img = load_prep_img(stop_img_loc)
plt.imshow(stop_blur_img)
plt.show()

# Load in Octagon template
template_loc = 'xxxxxxx'
gray_octagon_template, octagon_template_cnt = load_template(template_loc)
plt.imshow(gray_octagon_template, cmap='gray')
plt.show()

# Get the loose_mask
lower_red1, upper_red1, lower_red2, upper_red2 = (0, 70, 0), (10, 255, 255), (165, 70, 0), (180, 255, 255)
stop_loose_mask = get_red_mask(stop_blur_img, lower_red1, upper_red1, lower_red2, upper_red2)
stop_cleaned_img_1 = dilate_erode(stop_loose_mask, 3)

# Get the strict_mask
lower_red3, upper_red3, lower_red4, upper_red4 = (0, 100, 100), (10, 255, 255), (170, 100, 100), (180, 255, 255)
stop_strict_mask = get_red_mask(stop_blur_img, lower_red3, upper_red3, lower_red4, upper_red4)
stop_cleaned_img_2 = dilate_erode(stop_strict_mask, 15)

# Combine the 2 masks together
final_mask = cv2.bitwise_and(stop_cleaned_img_1, stop_cleaned_img_2)
plt.imshow(final_mask, cmap='gray')
plt.show()



# Draw contour before fixing convex defect
draw_contour(final_mask, blank_canvas_1, 2)
plt.imshow(blank_canvas_1)
plt.show()

# Make shape compare before fixing convexity defects
largest_ori_octagon_cnt, _ = find_contour(final_mask)
similarity_level_ori = shape_compare(largest_ori_octagon_cnt, octagon_template_cnt)
print(similarity_level_ori)

# Fix convexity defects
convex_fixed_img = fix_convex_defect(final_mask)
plt.imshow(convex_fixed_img)
plt.show()

# Re-draw contour after fixing convex defect
blank_canvas_2 = np.zeros(stop_sample_img.shape, dtype=np.uint8)
draw_contour(convex_fixed_img, blank_canvas_2, 2)
plt.imshow(blank_canvas_2)
plt.show()

# Make shape compare after fixing convexity defects
largest_fix_octagon_cnt, _ = find_contour(convex_fixed_img)
similarity_level_fix = shape_compare(largest_fix_octagon_cnt, octagon_template_cnt)
print(similarity_level_fix)

# Check if any of the 2 values is lower than a threshold? If so, draw bounding box on this contour region.
if similarity_level_fix < 0.1 or similarity_level_ori < 0.1:
    target_region_pos, box_on_img = draw_box(final_mask, stop_rgb_img, 1)
    print(target_region_pos)
    plt.imshow(box_on_img)
    plt.show()
    # Draw/ expand the bounding box and crop it.、

    border = 5
    for r in target_region_pos:
        before_eq, after_eq = crop_and_hist(stop_rgb_img, r, border)
        plt.imshow(before_eq)
        plt.show()
else:
    plt.imshow(stop_rgb_img)
    plt.show()
