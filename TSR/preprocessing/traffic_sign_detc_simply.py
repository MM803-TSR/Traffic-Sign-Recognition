import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from preprocessing.predict import predict

def video_to_frames(vedio_path):
	vidcap = cv2.VideoCapture(vedio_path)
	success, image = vidcap.read()
	print("image shape:", image.shape)
	img_set = []
	count = 0
	while success:
		# Save frame as JPEG file
		cv2.imwrite("frame%d.jpg" % count, image)
		# Get 1 frame each 1 second:
		vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 1000))
		success, image = vidcap.read()
		img_set.append(image)
		print('Read a new frame: ', success)
		count += 1
	print("length of img_set", len(img_set))
	return img_set


def frames_to_video(input_img_list, outputpath, fps):
	image_array = []
	for img in input_img_list:
		h = img.shape[1]
		w = img.shape[0]
		size = (h, w)
		img = cv2.resize(img, size)
		image_array.append(img)
	fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
	out = cv2.VideoWriter(outputpath, fourcc, fps, size)
	for i in range(len(image_array)):
		out.write(image_array[i])
	out.release()


def load_prep_img(sample_img):
	size = sample_img.shape
	blank_canvas_red = np.zeros(size, dtype=np.uint8)
	blank_canvas_yellow_orange = np.zeros(size, dtype=np.uint8)
	blank_canvas_white = np.zeros(size, dtype=np.uint8)
	# Convert to RGB color space
	rgb_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)
	# Convert to HSV color space
	hsv_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2HSV)
	# Blur image to reduce unrelated noise
	blur_img = cv2.bilateralFilter(hsv_img, 9, 75, 75)
	return blank_canvas_red, blank_canvas_yellow_orange, blank_canvas_white, sample_img, rgb_img, hsv_img, blur_img


def get_red_mask(img, lower_color1, upper_color1, lower_color2, upper_color2):
	masked_img1 = cv2.inRange(img, lower_color1, upper_color1)
	masked_img2 = cv2.inRange(img, lower_color2, upper_color2)
	loose_red_mask = masked_img1 + masked_img2
	return loose_red_mask


def get_other_colors_mask(img, lower_color, upper_color):
	masked_img = cv2.inRange(img, lower_color, upper_color)
	other_colors_mask = masked_img
	return other_colors_mask


def dilate_erode(img, kernel_size):
	kernel = np.ones((kernel_size, kernel_size), np.uint8)
	dilated_img = cv2.dilate(img, kernel, iterations=1)
	cleaned_img = cv2.erode(dilated_img, kernel, iterations=1)
	return cleaned_img


def find_contour(img):
	_, all_contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	if len(all_contours) != 0:
		all_sorted_cnt = sorted(all_contours, key=cv2.contourArea)[-1:]
		largest_cnt = all_sorted_cnt[0]
	else:
		largest_cnt = None
		all_sorted_cnt = None
		print("no contours")
	return largest_cnt, all_sorted_cnt


def draw_contour_2(img, canvas, contour_thickness):
	largest_cnt, all_sorted_cnt = find_contour(img)
	if all_sorted_cnt is not None:
		for i, largest_cnt in enumerate(all_sorted_cnt):
			cv2.drawContours(canvas, largest_cnt, -1, [0, 255, 0], contour_thickness)


def fix_convex_defect(img):
	# Detect convexity defects base on the contour.
	cnt_before_convex, _ = find_contour(img)
	hull_before = cv2.convexHull(cnt_before_convex, returnPoints=False)
	# print(type(hull_before))
	defects_before = cv2.convexityDefects(cnt_before_convex, hull_before)
	if defects_before is not None:
		for i in range(defects_before.shape[0]):
			s_before, e_before, f_before, d_before = defects_before[i, 0]
			start_before = tuple(cnt_before_convex[s_before][0])
			end_before = tuple(cnt_before_convex[e_before][0])
			far_before = tuple(cnt_before_convex[f_before][0])
			cv2.line(img, start_before, end_before, [255, 0, 0], 1)
		convex_fix_kernel = np.ones((3, 3), np.uint8)
		convex_dilated_img = cv2.dilate(img, convex_fix_kernel, iterations=1)
		convex_fixed_img = cv2.erode(convex_dilated_img, convex_fix_kernel, iterations=1)
	else:
		convex_fixed_img = img
	return convex_fixed_img


def load_template(template_path):
	template = cv2.imread(template_path)
	blurred_template = cv2.bilateralFilter(template, 9, 75, 75)
	gray_scale_template = cv2.cvtColor(blurred_template, cv2.COLOR_BGR2GRAY)
	ret, thresh2 = cv2.threshold(gray_scale_template, 127, 255, 0)
	_, template_contours, _ = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	template_cnt = template_contours[0]
	return gray_scale_template, template_cnt


def shape_compare(contours, template_cnt):
	similarity_level = cv2.matchShapes(contours, template_cnt, 1, 0.0)
	return similarity_level


def draw_box(img, canvas, contour_thickness):
	pos = []
	_, all_sorted_cnt = find_contour(img)
	for i, cnt in enumerate(all_sorted_cnt):
		colour = (0, 255, 0)
		x, y, w, h = cv2.boundingRect(cnt)
		# cv2.rectangle(canvas, (x, y), (x + w, y + h), colour, contour_thickness)
		pos.append([x, y, w, h])
	# print(pos)
	return pos, canvas


def crop_and_hist(original_img, target_region_pos, border):
	[img_h, img_w] = original_img.shape[:2]
	[x, y, w, h] = target_region_pos[0]
	# Original position: (should not chage this!)
	box_x = [x, x + w]
	box_y = [y, y + h]
	crop_x0 = box_x[0] - border
	crop_y0 = box_y[0] - border
	crop_x1 = box_x[1] + border
	crop_y1 = box_y[1] + border
	if crop_x0 < 0:
		crop_x0 = 0
	else:
		crop_x0 = crop_x0
	if crop_y0 < 0:
		crop_y0 = 0
	else:
		crop_y0 = crop_y0
	if crop_x1 > img_w:
		crop_x1 = img_w
	else:
		crop_x1 = crop_x1
	if crop_y1 > img_h:
		crop_y1 = img_h
	else:
		crop_y1 = crop_y1
	# New position:
	x_exp = [crop_x0, crop_x1]
	y_exp = [crop_y0, crop_y1]
	w_exp = x_exp[1] - x_exp[0]
	h_exp = y_exp[1] - y_exp[0]
	if w_exp > h_exp:
		diff = w_exp - h_exp
		h_exp = w_exp
		if y_exp[0] - round(diff / 2) < 0:
			y_exp_new0 = 0
			remains_h = round(diff / 2) - y_exp[0]
			y_exp_new1 = y_exp[1] + round(diff / 2) + remains_h
		else:
			y_exp_new0 = y_exp[0] - round(diff / 2)
			if y_exp[1] + round(diff / 2) > img_h:
				y_exp_new1 = img_h
				remains_h = y_exp[1] + round(diff / 2) - img_h
				y_exp_new0 = y_exp[0] - round(diff / 2) - remains_h
			else:
				y_exp_new1 = y_exp[1] + round(diff / 2)
		y_exp_new = [y_exp_new0, y_exp_new1]
		x_exp_new = x_exp
	else:
		diff = h_exp - w_exp
		w_exp = h_exp
		if x_exp[0] - round(diff / 2) < 0:
			x_exp_new0 = 0
			remains_w = round(diff / 2) - x_exp[0]
			x_exp_new1 = x_exp[1] + round(diff / 2) + remains_w
		else:
			x_exp_new0 = x_exp[0] - round(diff / 2)
			if x_exp[1] + round(diff / 2) > img_w:
				x_exp_new1 = img_w
				remains_w = x_exp[1] + round(diff / 2) - img_w
				x_exp_new0 = x_exp[0] - round(diff / 2) - remains_w
			else:
				x_exp_new1 = x_exp[1] + round(diff / 2)
		y_exp_new = y_exp
		x_exp_new = [x_exp_new0, x_exp_new1]

	target_region_img = original_img[y_exp_new[0]:y_exp_new[1], x_exp_new[0]:x_exp_new[1], :].copy()
	print("target_region_img shape:", target_region_img.shape)
	return target_region_img


def crop_possible_signs(similarity_level_ori, similarity_level_fix, final_mask, rgb_img):
	if similarity_level_ori < 0.8 or similarity_level_fix < 0.8:
		target_region_pos, box_on_img = draw_box(final_mask, rgb_img, 1)
		#print(target_region_pos)
		#         plt.imshow(box_on_img)
		#         plt.show()
		border = 8
		target_region_img_before_eq = crop_and_hist(rgb_img, target_region_pos, border)
		output = target_region_img_before_eq
	else:
		return None, None
	return target_region_pos, output


def draw_box_text(final_mask, img_draw_box, contour_thickness):
	pos = []
	_, all_sorted_cnt = find_contour(final_mask)
	for i, cnt in enumerate(all_sorted_cnt):
		colour = (255, 0, 0)
		x, y, w, h = cv2.boundingRect(cnt)
		cv2.rectangle(img_draw_box, (x, y), (x + w, y + h), colour, contour_thickness)
		pos.append([x, y, w, h])
	return pos, img_draw_box


def text_box_on_img(final_mask, img_draw_box, text):
	target_region_pos, img_draw_box = draw_box_text(final_mask, img_draw_box, contour_thickness=2)
	li = target_region_pos[0]
	x = li[0]
	y = li[1]
	color = (0, 255, 0)
	img_draw_box = Image.fromarray(img_draw_box)
	draw = ImageDraw.Draw(img_draw_box)
	draw.text((x, y), text, fill=color)
	img_draw_box = np.asarray(img_draw_box)
	return img_draw_box


# Load in the Octagon template and get its contours for red canvas
octagon_template_loc = 'red_octagon.jpeg'
gray_octagon_template, octagon_template_cnt = load_template(octagon_template_loc)
# plt.imshow(gray_octagon_template,cmap='gray')
# plt.show()

# Load in the Rectagule template and get its contours for white canvas
rectangle_template_loc = 'rectangle.png'
gray_rectangle_template, rectangle_template_cnt = load_template(rectangle_template_loc)
# plt.imshow(gray_rectangle_template,cmap='gray')
# plt.show()

# Load in the Dimond shape template and get its contours for red canvas
dimond_shape_template_loc = 'dimond_shape.jpeg'
gray_dimond_shape_template, dimond_shape_template_cnt = load_template(dimond_shape_template_loc)
# plt.imshow(gray_dimond_shape_template,cmap='gray')
# plt.show()


# Read image(whatever color) as sample_img and display it.
video_path = '../sample_images/video_1.mov'
input_frames_set = video_to_frames(video_path)
print(len(input_frames_set))
frame_i = 0
processed_frames = []

for frame in input_frames_set[:len(input_frames_set) - 1]:
	output_labeled_img_list = []
	output_red_list = []
	output_yellow_orange_list = []
	output_white_list = []
	output_green_list = []

	print("start processing frame:", frame_i)
	blank_canvas_red, blank_canvas_yellow_orange, blank_canvas_white, sample_img, rgb_img, hsv_img, blur_img = load_prep_img(
		frame)
	w = sample_img.shape[0]
	h = sample_img.shape[1]

	# Red:
	lower_red1, upper_red1, lower_red2, upper_red2 = (0, 70, 0), (10, 255, 255), (165, 70, 0), (180, 255, 255)
	red_loose_mask = get_red_mask(blur_img, lower_red1, upper_red1, lower_red2, upper_red2)
	red_cleaned_img_1 = dilate_erode(red_loose_mask, 3)

	lower_red3, upper_red3, lower_red4, upper_red4 = (0, 100, 50), (10, 255, 255), (170, 100, 50), (180, 255, 255)
	red_strict_mask = get_red_mask(blur_img, lower_red3, upper_red3, lower_red4, upper_red4)
	red_cleaned_img_2 = dilate_erode(red_strict_mask, 15)

	final_red_mask = cv2.bitwise_and(red_cleaned_img_1, red_cleaned_img_2)

	# Yellow:
	lower_yellow_orange, upper_yellow_orange = (11, 70, 0), (34, 255, 255)
	yellow_orange_loose_mask = get_other_colors_mask(blur_img, lower_yellow_orange, upper_yellow_orange)
	yellow_orange_cleaned_img_1 = dilate_erode(yellow_orange_loose_mask, 3)

	lower_yellow_orange2, upper_yellow_orange2 = (11, 100, 50), (34, 255, 255)
	yellow_orange_strict_mask = get_other_colors_mask(blur_img, lower_yellow_orange2, upper_yellow_orange2)
	yellow_orange_cleaned_img_2 = dilate_erode(yellow_orange_strict_mask, 15)

	final_yellow_orange_mask = cv2.bitwise_and(yellow_orange_cleaned_img_1, yellow_orange_cleaned_img_2)

	# Green:
	lower_green, upper_green = (35, 70, 0), (77, 255, 255)
	green_loose_mask = get_other_colors_mask(blur_img, lower_green, upper_green)
	green_cleaned_img_1 = dilate_erode(green_loose_mask, 3)

	lower_green2, upper_green2 = (38, 100, 50), (73, 255, 255)
	green_strict_mask = get_other_colors_mask(blur_img, lower_green2, upper_green2)
	green_cleaned_img_2 = dilate_erode(green_strict_mask, 15)

	final_green_mask = cv2.bitwise_and(green_cleaned_img_1, green_cleaned_img_2)

	# White
	lower_white, upper_white = (0, 0, 60), (180, 30, 255)
	white_loose_mask = get_other_colors_mask(blur_img, lower_white, upper_white)
	white_cleaned_img_1 = dilate_erode(white_loose_mask, 3)
	# plt.imshow(white_cleaned_img_1,cmap='gray')
	# plt.show()
	# Get the strick_white_mask
	lower_white2, upper_white2 = (0, 0, 200), (180, 20, 255)
	white_strict_mask = get_other_colors_mask(blur_img, lower_white2, upper_white2)
	white_cleaned_img_2 = dilate_erode(white_strict_mask, 15)
	# plt.imshow(white_cleaned_img_2,cmap='gray')
	# plt.show()
	# Get the final white mask1(suitable for colorful background):
	final_white_mask1 = cv2.bitwise_and(white_cleaned_img_1, white_cleaned_img_2)
	#     plt.imshow(final_white_mask1,cmap='gray')
	#     plt.show()
	# Get the final white mask2(suitable for white background):
	final_white_mask2 = cv2.bitwise_and(white_loose_mask, white_strict_mask)
	#     plt.imshow(final_white_mask2,cmap='gray')
	#     plt.show()

	# Red:
	# Draw contours on red_canvas before fixing convex defects.
	largest_ori_red_cnt, all_sorted_cnt_red = find_contour(final_red_mask)
	if all_sorted_cnt_red is not None:
		# draw_contour_2(final_red_mask, blank_canvas_red, 2)
		# Compare similarity:
		similarity_level_ori_red = shape_compare(largest_ori_red_cnt, octagon_template_cnt)

		# Re-draw contour after fixing convex defect
		convex_fixed_red_img = fix_convex_defect(final_red_mask)
		blank_canvas_red_2 = np.zeros(sample_img.shape, dtype=np.uint8)
		draw_contour_2(convex_fixed_red_img, blank_canvas_red_2, 2)
		#         plt.imshow(blank_canvas_red_2)
		#         plt.show()

		# Make comparison with the templet
		largest_fix_red_cnt, _ = find_contour(convex_fixed_red_img)
		similarity_level_fix_red = shape_compare(largest_fix_red_cnt, octagon_template_cnt)
		# print(similarity_level_fix_red)

		# Crop out red signs from ori_img:
		pos, output_red = crop_possible_signs(similarity_level_ori_red, similarity_level_fix_red, final_red_mask, rgb_img)
		if output_red is not None:
			#plt.imshow(output_red)
			#plt.show()
			output_red_list.append((pos, output_red))
	#         Image.fromarray(output_red).save('sample_images/cropped_video_4/output_output_red_%d_%d.jpg'%(frame_i,a)

	# Yellow:
	# Draw contours on yellow_orange_canvas before fixing convex defects.
	largest_ori_yellow_orange_cnt, all_sorted_cnt_yellow_orange = find_contour(final_yellow_orange_mask)
	if all_sorted_cnt_yellow_orange is not None:
		# Draw original contour:
		draw_contour_2(final_yellow_orange_mask, blank_canvas_yellow_orange, 2)

		# Try compare similarity level method:
		similarity_level_ori_yellow_orange = shape_compare(largest_ori_yellow_orange_cnt, dimond_shape_template_cnt)

		# Fix convexity defect on yellow_orange canvas & Re-draw contour after fixing convex defect
		convex_fixed_yellow_orange_img = fix_convex_defect(final_yellow_orange_mask)
		blank_canvas_yellow_orange_2 = np.zeros(sample_img.shape, dtype=np.uint8)
		draw_contour_2(convex_fixed_yellow_orange_img, blank_canvas_yellow_orange_2, 2)
		#         plt.imshow(blank_canvas_yellow_orange_2)
		#         plt.show()

		# Within yellow_orange canvas, make comparison with the templet
		largest_fix_yellow_orange_cnt, _ = find_contour(convex_fixed_yellow_orange_img)
		similarity_level_fix_yellow_orange = shape_compare(largest_fix_yellow_orange_cnt, dimond_shape_template_cnt)
		print(similarity_level_fix_yellow_orange)

		# Crop out yellow_orange signs from ori_img according to yellow_orange canvas:
		pos, output_yellow_orange = crop_possible_signs(similarity_level_ori_yellow_orange,
		                                           similarity_level_fix_yellow_orange, final_yellow_orange_mask,
		                                           rgb_img)
		if output_yellow_orange is not None:
			#plt.imshow(output_yellow_orange)
			#plt.show()
			output_yellow_orange_list.append((pos, output_yellow_orange))

	# Green:
	# Draw contours on yellow_orange_canvas before fixing convex defects.
	largest_ori_green_cnt, all_sorted_green_orange = find_contour(final_green_mask)
	if all_sorted_green_orange is not None:
		# Draw original contour:
		# draw_contour_2(final_green_mask, blank_green_orange, 2)
		similarity_level_ori_green = shape_compare(largest_ori_green_cnt, rectangle_template_cnt)

		convex_fixed_green_img = fix_convex_defect(final_green_mask)

		largest_fix_green_cnt, _ = find_contour(convex_fixed_green_img)
		similarity_level_fix_green = shape_compare(largest_fix_green_cnt, rectangle_template_cnt)
		print(similarity_level_fix_green)

		# Crop out yellow_orange signs from ori_img according to yellow_orange canvas:
		pos, output_green = crop_possible_signs(similarity_level_ori_green, similarity_level_fix_green, final_green_mask,
		                                   rgb_img)
		if output_green is not None:
			#plt.imshow(output_green)
			#plt.show()
			output_green_list.append((pos, output_green))

	# Draw contours on white_canvas before fixing convex defects.
	_, all_sorted_cnt_white = find_contour(final_white_mask1)
	largest_fix_white_cnt2, all_sorted_cnt_white2 = find_contour(final_white_mask2)
	if all_sorted_cnt_white or all_sorted_cnt_white2 is not None:
		draw_contour_2(final_white_mask1, blank_canvas_white, 2)
		largest_ori_white_cnt, _ = find_contour(final_white_mask1)
		similarity_level_ori_white = shape_compare(largest_ori_white_cnt, rectangle_template_cnt)
		#         plt.imshow(blank_canvas_white)
		#         plt.show()
		print(similarity_level_ori_white)

		# Fix convexity defect on white canvas & Re-draw contour after fixing convex defect
		convex_fixed_white_img = fix_convex_defect(final_white_mask1)
		blank_canvas_white_2 = np.zeros(sample_img.shape, dtype=np.uint8)
		draw_contour_2(convex_fixed_white_img, blank_canvas_white_2, 2)
		#         plt.imshow(blank_canvas_white_2)
		#         plt.show()

		# Within white canvas, make comparison with the templet
		largest_fix_white_cnt, _ = find_contour(convex_fixed_white_img)
		largest_fix_white_cnt2, _ = find_contour(final_white_mask2)
		similarity_level_fix_white = shape_compare(largest_fix_white_cnt, rectangle_template_cnt)
		similarity_level_fix_white2 = shape_compare(largest_fix_white_cnt2, rectangle_template_cnt)

		if cv2.contourArea(largest_fix_white_cnt) < (w * h) * 0.05 and cv2.contourArea(largest_fix_white_cnt2) < (
				w * h) * 0.05:
			# Crop out white signs from ori_img according to white canvas:
			if similarity_level_fix_white < similarity_level_fix_white2:
				pos, output_white = crop_possible_signs(similarity_level_ori_white, similarity_level_fix_white,
				                                   final_white_mask1, rgb_img)
				if output_white is not None:
					#plt.imshow(output_white)
					#plt.show()
					output_white_list.append((pos, output_white))
			#                 Image.fromarray(output_white).save('sample_images/cropped_video_4/output_output_white_%d_%d.jpg'%(frame_i,c))
			else:
				pos, output_white = crop_possible_signs(similarity_level_ori_white, similarity_level_fix_white2,
				                                   final_white_mask2, rgb_img)
				if output_white is not None:
					#plt.imshow(output_white)
					#plt.show()
					output_white_list.append((pos, output_white))
		#                 Image.fromarray(output_white).save('sample_images/cropped_video_4/output_output_white_%d_%d.jpg'%(frame_i,c))
		else:
			continue
	frame_i += 1

	# print(len(output_red_list))
	# print(len(output_yellow_orange_list))
	# print(len(output_green_list))
	# print(len(output_white_list))

	data_dir = "D:\Yanxi\MMGRAD\MM803\Project/new dataset1/train_negative/"

	cropped_img_input_list = output_red_list + output_yellow_orange_list + output_green_list + output_white_list
	if not cropped_img_input_list:
		pass
	file_i = 0
	classes = []
	for pos, img in cropped_img_input_list:
		pred, prob, prob_std = predict(img)
		if prob_std < 0.1 or prob < 0.7:
			#classes.append((None, None))
			cv2.imwrite(filename=data_dir + 'missed' + '/' + str(video_path.split('/')[-1]) + str(frame_i) + '_' + str(
				file_i) + '.jpg', img=img)
			continue
		#cv2.imwrite(filename=data_dir+str(pred)+'/'+str(video_path.split('/')[-1])+str(frame_i)+'_'+str(file_i)+'.jpg', img=img)
		classes.append((pos, pred))
	# fig, axes = plt.subplots(3, 3, figsize=(3, 3))
	# i = 0
	# for ax in axes.flatten():
	# 	ax.imshow(cropped_img_input_list[i][1])
	# 	if i >= len(cropped_img_input_list) - 1:
	# 		break
	# 	i += 1
	# fig.show()
	#print(classes)

	for pos, pred in classes:
		x, y, w, h = pos[0]
		color, contour_thickness = (0, 255, 0), 3
		rgb_img = np.asarray(rgb_img)
		cv2.rectangle(rgb_img, (x, y), (x + w, y + h), color, contour_thickness)
		rgb_img = Image.fromarray(rgb_img)
		draw = ImageDraw.Draw(rgb_img)
		draw.text((x, y), str(pred), fill=color)
	#processed_frames.append(np.asarray(rgb_img))

print('finish all frames!')
#for f in processed_frames:
	#plt.imshow(f)
	#plt.show()


# if len(output_red_list) != 0:
# 	a = 0
# 	for crop_red_img in output_red_list:
# 		Image.fromarray(crop_red_img).save('sample_images/cropped_video_8/output_output_red_%d.jpg' % a)
# 		a += 1
#
# if len(output_yellow_orange_list) != 0:
# 	b = 0
# 	for crop_yellow_orange_img in output_yellow_orange_list:
# 		Image.fromarray(crop_yellow_orange_img).save('sample_images/cropped_video_8/output_output_yellow_%d.jpg' % b)
# 		b += 1
#
# if len(output_green_list) != 0:
# 	c = 0
# 	for crop_green_img in output_green_list:
# 		Image.fromarray(crop_green_img).save('sample_images/cropped_video_8/output_output_green_%d.jpg' % c)
# 		c += 1
#
# if len(output_white_list) != 0:
# 	d = 0
# 	for crop_white_img in output_white_list:
# 		Image.fromarray(crop_white_img).save('sample_images/cropped_video_8/output_output_white_%d.jpg' % d)
# 		d += 1

