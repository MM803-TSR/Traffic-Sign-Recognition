import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
from PIL import Image, ImageDraw, ImageFont


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
    return pos


def draw_box_text(final_mask, img_draw_box, contour_thickness):
    pos = []
    _, all_sorted_cnt = find_contour(final_mask)
    for i, cnt in enumerate(all_sorted_cnt):
        colour = (255, 0, 0)
        x, y, w, h = cv2.boundingRect(cnt)
        print("type before cv2.rectangle:", type(img_draw_box))
        cv2.rectangle(img_draw_box, (x, y), (x + w, y + h), colour, contour_thickness)
        pos.append([x, y, w, h])
    return pos, img_draw_box


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


def crop_possible_signs(final_mask, rgb_img):
    target_region_pos = draw_box(final_mask, rgb_img, 1)
    print("target_region_pos", target_region_pos)
    border = 8
    target_region_img = crop_and_hist(rgb_img, target_region_pos, border)
    output = target_region_img
    return output, target_region_pos


def text_box_on_img(final_mask, img_draw_box, text):
    # print("input to /text_box_on_img/ func:",type(img_draw_box))
    target_region_pos, img_draw_box = draw_box_text(final_mask, img_draw_box, contour_thickness=2)
    # plt.imshow(img_draw_box)
    # plt.show
    # print("output of func /draw_box_text/:",type(img_draw_box))
    li = target_region_pos[0]
    x = li[0]
    y = li[1]
    color = (0, 255, 0)
    # img_draw_box is a drawed img (contains box, but no text)
    img_draw_box = Image.fromarray(img_draw_box)
    # print('type of img_draw_box after Image.fromarray:',type(img_draw_box))
    draw = ImageDraw.Draw(img_draw_box)
    draw.text((x, y), text, fill=color)
    # plt.imshow(img_draw_box)
    # plt.show
    img_draw_box = np.asarray(img_draw_box)
    # print("output text_box_on_img:",type(img_draw_box))
    return img_draw_box


# Load in the Octagon template and get its contours for red canvas
octagon_template_loc = 'red_octagon.jpeg'
gray_octagon_template, octagon_template_cnt = load_template(octagon_template_loc)
# plt.imshow(gray_octagon_template,cmap='gray')
# plt.show()

# Load in the triangle template and get its contours for red canvas
triangle_template_loc = 'triangle.png'
gray_triangle_template, triangle_template_cnt = load_template(triangle_template_loc)
# plt.imshow(gray_triangle_template,cmap='gray')
# plt.show()

# Load in the Rectagule template and get its contours for white canvas
rectangle_template_loc = 'rectangle.png'
gray_rectangle_template, rectangle_template_cnt = load_template(rectangle_template_loc)
# plt.imshow(gray_rectangle_template,cmap='gray')
# plt.show()

# Load in the Dimond shape template and get its contours for yellow canvas
dimond_shape_template_loc = 'dimond_shape.jpeg'
gray_dimond_shape_template, dimond_shape_template_cnt = load_template(dimond_shape_template_loc)
# plt.imshow(gray_dimond_shape_template,cmap='gray')
# plt.show()


# Read image(whatever color) as sample_img and display it.
video_path = 'sample_images/video_test_5.mov'
input_frames_set = video_to_frames(video_path)
output_labeled_img_list = []
frame_i = 0

for frame in input_frames_set[:len(input_frames_set) - 1]:
    print("start processing frame:", frame_i)
    blank_canvas_red, blank_canvas_yellow_orange, blank_canvas_white, sample_img, rgb_img, hsv_img, blur_img = load_prep_img(
        frame)
    w = sample_img.shape[0]
    h = sample_img.shape[1]

    # Firstly detect whether contains red_signs:
    #  Get the loose_red_mask
    red_masks = []
    lower_red1, upper_red1, lower_red2, upper_red2 = (0, 40, 0), (10, 255, 255), (165, 40, 0), (180, 255, 255)
    # lower_red1, upper_red1, lower_red2, upper_red2 = (0, 60, 20), (8, 255, 230), (165, 60, 20), (180, 255, 230)
    red_loose_mask = get_red_mask(blur_img, lower_red1, upper_red1, lower_red2, upper_red2)
    red_cleaned_img_1 = dilate_erode(red_loose_mask, 3)
    red_masks.append(red_cleaned_img_1)
    # Get the strick_red_mask
    lower_red3, upper_red3, lower_red4, upper_red4 = (0, 100, 50), (10, 255, 255), (170, 100, 50), (180, 255, 255)
    # lower_red3, upper_red3, lower_red4, upper_red4 = (0, 100, 50), (6, 255, 200), (170, 100, 50), (180, 255, 200)
    red_strict_mask = get_red_mask(blur_img, lower_red3, upper_red3, lower_red4, upper_red4)
    red_cleaned_img_2 = dilate_erode(red_strict_mask, 15)
    # Combine 2 red_masks together
    final_red_mask = cv2.bitwise_and(red_cleaned_img_1, red_cleaned_img_2)
    red_masks.append(final_red_mask)
    # plt.imshow(red_cleaned_img_1,cmap='gray')
    # plt.show()
    # plt.imshow(red_cleaned_img_2,cmap='gray')
    # plt.show()
    # plt.imshow(final_red_mask,cmap='gray')
    # plt.show()

    # Get the loose_yellow_orange_mask
    yellow_orange_masks = []
    lower_yellow_orange, upper_yellow_orange = (11, 70, 0), (34, 255, 255)
    # lower_yellow_orange, upper_yellow_orange = (15, 60, 20), (30, 255, 230)
    yellow_orange_loose_mask = get_other_colors_mask(blur_img, lower_yellow_orange, upper_yellow_orange)
    yellow_orange_cleaned_img_1 = dilate_erode(yellow_orange_loose_mask, 3)
    yellow_orange_masks.append(yellow_orange_cleaned_img_1)
    # Get the strick_yellow_orange_mask
    lower_yellow_orange2, upper_yellow_orange2 = (11, 100, 50), (34, 255, 255)
    # lower_yellow_orange2, upper_yellow_orange2 = (15, 100, 50), (30, 255, 200)
    yellow_orange_strict_mask = get_other_colors_mask(blur_img, lower_yellow_orange2, upper_yellow_orange2)
    yellow_orange_cleaned_img_2 = dilate_erode(yellow_orange_strict_mask, 15)
    # Combine 2 yellow_orange_masks together
    final_yellow_orange_mask = cv2.bitwise_and(yellow_orange_cleaned_img_1, yellow_orange_cleaned_img_2)
    yellow_orange_masks.append(final_yellow_orange_mask)
    # plt.imshow(yellow_orange_cleaned_img_2,cmap='gray')
    # plt.show()
    # plt.imshow(yellow_orange_cleaned_img_1,cmap='gray')
    # plt.show()
    # plt.imshow(final_yellow_orange_mask,cmap='gray')
    # plt.show()

    # Get the loose_white_mask
    white_masks = []
    lower_white, upper_white = (0, 0, 60), (180, 30, 255)
    # lower_white, upper_white = (0, 0, 100), (180, 20, 250)
    white_loose_mask = get_other_colors_mask(blur_img, lower_white, upper_white)
    white_cleaned_img_1 = dilate_erode(white_loose_mask, 3)
    white_masks.append(white_loose_mask)
    # Get the strick_white_mask
    lower_white2, upper_white2 = (0, 0, 200), (180, 20, 255)
    # lower_white2, upper_white2 = (0, 0, 200), (180, 12, 243)
    white_strict_mask = get_other_colors_mask(blur_img, lower_white2, upper_white2)
    white_cleaned_img_2 = dilate_erode(white_strict_mask, 15)
    # Get the final white mask1 (suitable for colorful background):
    final_white_mask1 = cv2.bitwise_and(white_cleaned_img_1, white_cleaned_img_2)
    white_masks.append(final_white_mask1)
    # Get the final white mask2 (suitable for white background):
    final_white_mask2 = cv2.bitwise_and(white_loose_mask, white_strict_mask)
    white_masks.append(final_white_mask2)
    #     plt.imshow(white_loose_mask,cmap='gray')
    #     plt.show()
    #     plt.imshow(white_cleaned_img_1,cmap='gray')
    #     plt.show()
    #     plt.imshow(white_cleaned_img_2,cmap='gray')
    #     plt.show()
    #     plt.imshow(final_white_mask1,cmap='gray')
    #     plt.show()
    #     plt.imshow(final_white_mask2,cmap='gray')
    #     plt.show()

    # Red:
    # Draw contours on red_canvas before fixing convex defects.
    blank_canvas_red = [np.zeros(sample_img.shape, dtype=np.uint8), np.zeros(sample_img.shape, dtype=np.uint8)]
    blank_canvas_red_2 = [np.zeros(sample_img.shape, dtype=np.uint8), np.zeros(sample_img.shape, dtype=np.uint8)]
    output_red_list = []
    mask_red_list = []
    i = 0
    for red_mask in red_masks:
        largest_ori_red_cnt, all_sorted_cnt_red = find_contour(red_mask)
        if all_sorted_cnt_red is not None:
            draw_contour_2(red_mask, blank_canvas_red[i], 2)
            # Compare templet similarity:
            similarity_level_ori_red = shape_compare(largest_ori_red_cnt, octagon_template_cnt)
            # plt.imshow(blank_canvas_red[i])
            # plt.show()
            # print("original similarity:", similarity_level_ori_red)
            if similarity_level_ori_red < 0.2:
                if (w * h) / 80 < cv2.contourArea(largest_ori_red_cnt) < (w * h) / 18:
                    output_red, _ = crop_possible_signs(red_mask, rgb_img)
                    # print("output_red shape1:",output_red.shape)
                    # plt.imshow(output_red)
                    # plt.show()
                    output_red_list.append(output_red)
                    mask_red_list.append(red_mask)
                else:
                    print("Region too large")

            elif 0.2 < similarity_level_ori_red < 0.9:  # Compare shape:
                approx_red_1 = cv2.approxPolyDP(largest_ori_red_cnt, 0.04 * cv2.arcLength(largest_ori_red_cnt, True),
                                                True)
                print("length of approx_red_1:", len(approx_red_1))
                if len(approx_red_1) == 3:
                    print("triangle")
                    if (w * h) / 80 < cv2.contourArea(largest_ori_red_cnt) < (w * h) / 18:
                        output_red, _ = crop_possible_signs(red_mask, rgb_img)
                        # print("output_red shape2:",output_red.shape)
                        # plt.imshow(output_red)
                        # plt.show()
                        output_red_list.append(output_red)
                        mask_red_list.append(red_mask)
                    else:
                        print("Region too large")
                elif len(approx_red_1) == 8:
                    print("octagon")
                    if (w * h) / 80 < cv2.contourArea(largest_ori_red_cnt) < (w * h) / 18:
                        output_red, _ = crop_possible_signs(red_mask, rgb_img)
                        # print("output_red shape3:",output_red.shape)
                        # plt.imshow(output_red)
                        # plt.show()
                        output_red_list.append(output_red)
                        mask_red_list.append(red_mask)
                    else:
                        print("Region too large")
                else:
                    # Fix convex defect and re-draw contour:
                    convex_fixed_red_img = fix_convex_defect(red_mask)
                    draw_contour_2(convex_fixed_red_img, blank_canvas_red_2[i], 2)
                    # plt.imshow(blank_canvas_red_2[i])
                    # plt.show()
                    # Compare similarity again:
                    largest_fix_red_cnt, _ = find_contour(convex_fixed_red_img)
                    similarity_level_fix_red = shape_compare(largest_fix_red_cnt, octagon_template_cnt)
                    # print("Fixed similarity:", similarity_level_fix_red)
                    if similarity_level_ori_red < 0.2:
                        if (w * h) / 80 < cv2.contourArea(largest_fix_red_cnt) < (w * h) / 18:
                            output_red, _ = crop_possible_signs(convex_fixed_red_img, rgb_img)
                            # print("output_red shape4:",output_red.shape)
                            # plt.imshow(output_red)
                            # plt.show()
                            output_red_list.append(output_red)
                            mask_red_list.append(convex_fixed_red_img)
                        else:
                            print("Region too large")
                    elif 0.2 < similarity_level_ori_red < 0.9:
                        approx_red_2 = cv2.approxPolyDP(largest_fix_red_cnt,
                                                        0.04 * cv2.arcLength(largest_fix_red_cnt, True), True)
                        print("length of approx_red_2:", len(approx_red_2))
                        if len(approx_red_2) == 3:
                            print("triangle")
                            if (w * h) / 80 < cv2.contourArea(largest_fix_red_cnt) < (w * h) / 18:
                                output_red, _ = crop_possible_signs(convex_fixed_red_img, rgb_img)
                                # print("output_red shape5:",output_red.shape)
                                # plt.imshow(output_red)
                                # plt.show()
                                output_red_list.append(output_red)
                                mask_red_list.append(convex_fixed_red_img)
                            else:
                                print("Region too large")
                        elif len(approx_red_2) == 8:
                            print("octagon")
                            if (w * h) / 80 < cv2.contourArea(largest_fix_red_cnt) < (w * h) / 18:
                                output_red, _ = crop_possible_signs(convex_fixed_red_img, rgb_img)
                                # print("output_red shape6:",output_red.shape)
                                # plt.imshow(output_red)
                                # plt.show()
                                output_red_list.append(output_red)
                                mask_red_list.append(convex_fixed_red_img)
                            else:
                                print("Region too large")
                        else:
                            print("No target signs")
                    else:
                        print("No target signs")
            else:
                # Fix convex defect and re-draw contour:
                convex_fixed_red_img = fix_convex_defect(red_mask)
                draw_contour_2(convex_fixed_red_img, blank_canvas_red_2[i], 2)
                # plt.imshow(blank_canvas_red_2[i])
                # plt.show()
                # Compare similarity again:
                largest_fix_red_cnt, _ = find_contour(convex_fixed_red_img)
                similarity_level_fix_red = shape_compare(largest_fix_red_cnt, octagon_template_cnt)
                # print("Fixed similarity:", similarity_level_fix_red)
                if similarity_level_fix_red < 0.2:
                    if (w * h) / 80 < cv2.contourArea(largest_fix_red_cnt) < (w * h) / 18:
                        output_red, _ = crop_possible_signs(convex_fixed_red_img, rgb_img)
                        # print("output_red shape7:",output_red.shape)
                        # plt.imshow(output_red)
                        # plt.show()
                        output_red_list.append(output_red)
                        mask_red_list.append(convex_fixed_red_img)
                    else:
                        print("Region too large")
                elif 0.2 < similarity_level_fix_red < 0.9:
                    approx_red_2 = cv2.approxPolyDP(largest_fix_red_cnt,
                                                    0.04 * cv2.arcLength(largest_fix_red_cnt, True),
                                                    True)
                    print("length of approx_red_2:", len(approx_red_2))
                    if len(approx_red_2) == 3:
                        print("triangle")
                        if (w * h) / 80 < cv2.contourArea(largest_fix_red_cnt) < (w * h) / 18:
                            output_red, _ = crop_possible_signs(convex_fixed_red_img, rgb_img)
                            # print("output_red shape8:",output_red.shape)
                            # plt.imshow(output_red)
                            # plt.show()
                            output_red_list.append(output_red)
                            mask_red_list.append(convex_fixed_red_img)
                        else:
                            print("Region too large")
                    elif len(approx_red_2) == 8:
                        print("octagon")
                        if (w * h) / 80 < cv2.contourArea(largest_fix_red_cnt) < (w * h) / 18:
                            output_red, _ = crop_possible_signs(convex_fixed_red_img, rgb_img)
                            # print("output_red shape9:",output_red.shape)
                            # plt.imshow(output_red)
                            # plt.show()
                            output_red_list.append(output_red)
                            mask_red_list.append(convex_fixed_red_img)
                        else:
                            print("Region too large")
                    else:
                        print("No target signs")
                else:
                    print("No target signs")
            i += 1
        else:
            print("no contour found")

    # Yellow:
    # Draw contours on red_canvas before fixing convex defects.
    blank_canvas_yellow_orange = [np.zeros(sample_img.shape, dtype=np.uint8),
                                  np.zeros(sample_img.shape, dtype=np.uint8)]
    blank_canvas_yellow_orange_2 = [np.zeros(sample_img.shape, dtype=np.uint8),
                                    np.zeros(sample_img.shape, dtype=np.uint8)]
    output_yellow_orange_list = []
    mask_yellow_orange_list = []
    j = 0
    for yellow_orange_mask in yellow_orange_masks:
        largest_ori_yellow_orange_cnt, all_sorted_cnt_yellow_orange = find_contour(yellow_orange_mask)
        if all_sorted_cnt_red is not None:
            draw_contour_2(yellow_orange_mask, blank_canvas_yellow_orange[j], 2)
            # Compare templet similarity:
            similarity_level_ori_yellow_orange = shape_compare(largest_ori_yellow_orange_cnt, dimond_shape_template_cnt)
            # plt.imshow(blank_canvas_yellow_orange[j])
            # plt.show()
            # print("original similarity:", similarity_level_ori_yellow_orange)
            if similarity_level_ori_yellow_orange < 0.2:  # Directly crop out red signs from ori_img
                output_yellow_orange, _ = crop_possible_signs(yellow_orange_mask, rgb_img)
                # print("output_yellow_orange shape1:",output_yellow_orange.shape)
                # plt.imshow(output_yellow_orange)
                # plt.show()
                output_yellow_orange_list.append(output_yellow_orange)
                mask_yellow_orange_list.append(yellow_orange_mask)
            elif 0.2 < similarity_level_ori_yellow_orange < 0.9:  # Compare shape:
                approx_yellow_orange_1 = cv2.approxPolyDP(largest_ori_yellow_orange_cnt,
                                                          0.04 * cv2.arcLength(largest_ori_yellow_orange_cnt, True),
                                                          True)
                print(len(approx_yellow_orange_1))
                if len(approx_yellow_orange_1) == 4:
                    print("dimond shape")
                    if (w * h) / 80 < cv2.contourArea(largest_ori_yellow_orange_cnt) < (w * h) / 18:
                        output_yellow_orange, _ = crop_possible_signs(yellow_orange_mask, rgb_img)
                        # print("output_yellow_orange shape2:",output_yellow_orange.shape)
                        # plt.imshow(output_yellow_orange)
                        # plt.show()
                        output_yellow_orange_list.append(output_yellow_orange)
                        mask_yellow_orange_list.append(yellow_orange_mask)
                    else:
                        print("Region too large")
                else:
                    # Fix convex defect and re-draw contour:
                    convex_fixed_yellow_orange_img = fix_convex_defect(yellow_orange_mask)
                    draw_contour_2(convex_fixed_yellow_orange_img, blank_canvas_yellow_orange_2[j], 2)
                    # plt.imshow(blank_canvas_yellow_orange_2[j])
                    # plt.show()
                    # Compare similarity again:
                    largest_fix_yellow_orange_cnt, _ = find_contour(convex_fixed_yellow_orange_img)
                    similarity_level_fix_yellow_orange = shape_compare(largest_fix_yellow_orange_cnt,
                                                                       dimond_shape_template_cnt)
                    # print("Fixed similarity:", similarity_level_fix_yellow_orange)
                    if similarity_level_fix_yellow_orange < 0.2:
                        output_yellow_orange, _ = crop_possible_signs(convex_fixed_yellow_orange_img, rgb_img)
                        # print("output_yellow_orange shape3:",output_yellow_orange.shape)
                        # plt.imshow(output_yellow_orange)
                        # plt.show()
                        output_yellow_orange_list.append(output_yellow_orange)
                        mask_yellow_orange_list.append(convex_fixed_yellow_orange_img)
                    elif 0.2 < similarity_level_fix_yellow_orange < 0.9:
                        approx_yellow_orange_2 = cv2.approxPolyDP(largest_fix_yellow_orange_cnt,
                                                                  0.04 * cv2.arcLength(largest_fix_yellow_orange_cnt,
                                                                                       True), True)
                        print("length of approx_yellow_orange_2:", len(approx_yellow_orange_2))
                        if len(approx_yellow_orange_2) == 4:
                            print("dimond shape")
                            if (w * h) / 80 < cv2.contourArea(largest_ori_yellow_orange_cnt) < (w * h) / 18:
                                output_yellow_orange, _ = crop_possible_signs(convex_fixed_yellow_orange_img, rgb_img)
                                # print("output_yellow_orange shape4:",output_yellow_orange.shape)
                                # plt.imshow(output_yellow_orange)
                                # plt.show()
                                output_yellow_orange_list.append(output_yellow_orange)
                                mask_yellow_orange_list.append(convex_fixed_yellow_orange_img)
                            else:
                                print("Region too large")
                        else:
                            print("No target signs")
                    else:
                        print("No target signs")
            else:
                # Fix convex defect and re-draw contour:
                convex_fixed_yellow_orange_img = fix_convex_defect(yellow_orange_mask)
                draw_contour_2(convex_fixed_yellow_orange_img, blank_canvas_yellow_orange_2[j], 2)
                # plt.imshow(blank_canvas_yellow_orange_2[j])
                # plt.show()
                # Compare similarity again:
                largest_fix_yellow_orange_cnt, _ = find_contour(convex_fixed_yellow_orange_img)
                similarity_level_fix_yellow_orange = shape_compare(largest_fix_yellow_orange_cnt,
                                                                   dimond_shape_template_cnt)
                # print("Fixed similarity:", similarity_level_fix_yellow_orange)
                if similarity_level_fix_yellow_orange < 0.2:
                    output_yellow_orange, _ = crop_possible_signs(convex_fixed_yellow_orange_img, rgb_img)
                    # print("output_yellow_orange shape5:",output_yellow_orange.shape)
                    # plt.imshow(output_yellow_orange)
                    # plt.show()
                    output_yellow_orange_list.append(output_yellow_orange)
                    mask_yellow_orange_list.append(convex_fixed_yellow_orange_img)
                elif 0.2 < similarity_level_fix_yellow_orange < 0.9:
                    approx_yellow_orange_2 = cv2.approxPolyDP(largest_fix_yellow_orange_cnt,
                                                              0.04 * cv2.arcLength(largest_fix_yellow_orange_cnt, True),
                                                              True)
                    print("length of approx_yellow_orange_2:", len(approx_yellow_orange_2))
                    if len(approx_yellow_orange_2) == 4:
                        print("dimond shape")
                        if (w * h) / 80 < cv2.contourArea(largest_ori_yellow_orange_cnt) < (w * h) / 18:
                            output_yellow_orange, _ = crop_possible_signs(convex_fixed_yellow_orange_img, rgb_img)
                            # print("output_yellow_orange shape6:",output_yellow_orange.shape)
                            # plt.imshow(output_yellow_orange)
                            # plt.show()
                            output_yellow_orange_list.append(output_yellow_orange)
                            mask_yellow_orange_list.append(convex_fixed_yellow_orange_img)
                        else:
                            print("Region too large")
                    else:
                        print("No target signs")
                else:
                    print("No target signs")
            j += 1
        else:
            print("no contour found")

    blank_canvas_white = [np.zeros(sample_img.shape, dtype=np.uint8), np.zeros(sample_img.shape, dtype=np.uint8),
                          np.zeros(sample_img.shape, dtype=np.uint8)]
    blank_canvas_white_2 = [np.zeros(sample_img.shape, dtype=np.uint8), np.zeros(sample_img.shape, dtype=np.uint8),
                            np.zeros(sample_img.shape, dtype=np.uint8)]
    blank_canvas_white_3 = [np.zeros(sample_img.shape, dtype=np.uint8), np.zeros(sample_img.shape, dtype=np.uint8),
                            np.zeros(sample_img.shape, dtype=np.uint8)]
    blank_canvas_white_4 = [np.zeros(sample_img.shape, dtype=np.uint8), np.zeros(sample_img.shape, dtype=np.uint8),
                            np.zeros(sample_img.shape, dtype=np.uint8)]
    # blank_canvas_white = np.zeros(sample_img.shape, dtype=np.uint8)
    # blank_canvas_white_2 = np.zeros(sample_img.shape, dtype=np.uint8)
    # blank_canvas_white_3 = np.zeros(sample_img.shape, dtype=np.uint8)
    # blank_canvas_white_4 = np.zeros(sample_img.shape, dtype=np.uint8)
    output_white_list = []
    mask_white_list = []
    q = 0
    # White:
    # Draw contours on white_canvas before fixing convex defects.
    # largest_ori_white_cnt, all_sorted_cnt_white = find_contour(final_white_mask1)
    # largest_fix_white_cnt2, all_sorted_cnt_white2 = find_contour(final_white_mask2)
    for white_mask in white_masks:
        largest_ori_white_cnt, all_sorted_cnt_white = find_contour(white_mask)
        # print("this is mask",q)
        if all_sorted_cnt_white is not None:
            draw_contour_2(white_mask, blank_canvas_white[q], 2)
            similarity_level_ori_white = shape_compare(largest_ori_white_cnt, rectangle_template_cnt)
            # plt.imshow(blank_canvas_white[q])
            # plt.show()
            # print(similarity_level_ori_white)
            if similarity_level_ori_white < 0.2:
                if (w * h) / 80 < cv2.contourArea(largest_ori_white_cnt) < (w * h) / 18:
                    output_white, _ = crop_possible_signs(white_mask, rgb_img)
                    # print("output_white shape1:",output_white.shape)
                    # plt.imshow(output_white)
                    # plt.show()
                    output_white_list.append(output_white)
                    mask_white_list.append(white_mask)
                else:
                    print("Region too large1")

            elif 0.2 < similarity_level_ori_white < 0.9:
                approx_white_1 = cv2.approxPolyDP(largest_ori_white_cnt,
                                                  0.04 * cv2.arcLength(largest_ori_white_cnt, True),
                                                  True)
                print("length of approx_white_1:", len(approx_white_1))
                if len(approx_white_1) == 4:
                    print("rectangle")
                    if (w * h) / 80 < cv2.contourArea(largest_ori_white_cnt) < (w * h) / 18:
                        output_white, _ = crop_possible_signs(white_mask, rgb_img)
                        # print("output_white shape2:",output_white.shape)
                        # plt.imshow(output_white)
                        # plt.show()
                        output_white_list.append(output_white)
                        mask_white_list.append(white_mask)
                    else:
                        print("Region too large2")
                else:
                    # Fix convex defect and re-draw contour:
                    convex_fixed_white_img = fix_convex_defect(white_mask)
                    draw_contour_2(convex_fixed_white_img, blank_canvas_white_2[q], 2)
                    # plt.imshow(blank_canvas_white_2[q])
                    # plt.show()
                    largest_fix_white_cnt, _ = find_contour(convex_fixed_white_img)
                    similarity_level_fix_white = shape_compare(largest_fix_white_cnt, rectangle_template_cnt)
                    # print("Fixed similarity:", similarity_level_fix_white)
                    if similarity_level_fix_white < 0.2:
                        if (w * h) / 80 < cv2.contourArea(largest_fix_white_cnt) < (w * h) / 18:
                            output_white, _ = crop_possible_signs(convex_fixed_white_img, rgb_img)
                            # print("output_white shape3:",output_white.shape)
                            # plt.imshow(output_white)
                            # plt.show()
                            output_white_list.append(output_white)
                            mask_white_list.append(convex_fixed_white_img)
                        else:
                            print("Region too large3")
                    elif 0.2 < similarity_level_fix_white < 0.9:
                        approx_white_2 = cv2.approxPolyDP(largest_fix_white_cnt,
                                                          0.04 * cv2.arcLength(largest_fix_white_cnt, True), True)
                        print('length of approx_white_2:', len(approx_white_2))
                        if len(approx_white_2) == 4:
                            print("rectangle")
                            if (w * h) / 80 < cv2.contourArea(largest_fix_white_cnt) < (w * h) / 18:
                                output_white, _ = crop_possible_signs(convex_fixed_white_img, rgb_img)
                                # print("output_white shape4:",output_white.shape)
                                # plt.imshow(output_white)
                                # plt.show()
                                output_white_list.append(output_white)
                                mask_white_list.append(convex_fixed_white_img)
                            else:
                                print("Region too large4")
                        else:
                            print("No target signs")
                    else:
                        print("No target signs")
            else:
                # Fix convex defect and re-draw contour:
                convex_fixed_white_img = fix_convex_defect(white_mask)
                draw_contour_2(convex_fixed_white_img, blank_canvas_white_2[q], 2)
                # plt.imshow(blank_canvas_white_2[q])
                # plt.show()
                largest_fix_white_cnt, _ = find_contour(convex_fixed_white_img)
                similarity_level_fix_white = shape_compare(largest_fix_white_cnt, rectangle_template_cnt)
                # print("Fixed similarity:", similarity_level_fix_white)
                if similarity_level_fix_white < 0.2:
                    if (w * h) / 80 < cv2.contourArea(largest_fix_white_cnt) < (w * h) / 18:
                        output_white, target_region_pos_white = crop_possible_signs(convex_fixed_white_img, rgb_img)
                        # print("output_white shape5:",output_white.shape)
                        # plt.imshow(output_white)
                        # plt.show()
                        output_white_list.append(output_white)
                        mask_white_list.append(convex_fixed_white_img)
                        # store the image
                        # Image.fromarray(output_white).save('output_white.jpg')
                        # rgb_img_img = Image.fromarray(rgb_img)
                        # output_white_labeled = text_box_on_img(convex_fixed_white_img, rgb_img, text='speed limit')
                        # plt.imshow(output_white_labeled)
                        # plt.show()
                    else:
                        print("Region too large5")
                elif 0.2 < similarity_level_fix_white < 0.9:
                    approx_white_2 = cv2.approxPolyDP(largest_fix_white_cnt,
                                                      0.04 * cv2.arcLength(largest_fix_white_cnt, True), True)
                    print("length of approx_white_2:", len(approx_white_2))
                    if len(approx_white_2) == 4:
                        print("rectangle")
                        if (w * h) / 80 < cv2.contourArea(largest_fix_white_cnt) < (w * h) / 18:
                            output_white, _ = crop_possible_signs(convex_fixed_white_img, rgb_img)
                            # print("output_white shape6:",output_white.shape)
                            # plt.imshow(output_white)
                            # plt.show()
                            output_white_list.append(output_white)
                            mask_white_list.append(convex_fixed_white_img)
                        else:
                            print("Region too large6")
                    else:
                        print("No target signs")
                else:
                    print("No target signs")
            q += 1
        else:
            print("no contour found")
    print("length of output_red_list:", len(output_red_list))
    print("length of mask_red_list:", len(mask_red_list))

    print("length of output_yellow_orange_list:", len(output_yellow_orange_list))
    print("length of mask_yellow_orange_list:", len(mask_yellow_orange_list))

    print('length of output_white_list:', len(output_white_list))
    print('length of mask_white_list:', len(mask_white_list))

    # Get all three list for each frame
    rgb_img_label = rgb_img

    if len(mask_red_list) != 0:
        for mask_red in mask_red_list:
            # rgb_img_red = Image.fromarray(rgb_img_red)
            rgb_img_label = text_box_on_img(mask_red, rgb_img_label, text='speed limit')
        plt.imshow(rgb_img_label)
        plt.show()
    #         Image.fromarray(rgb_img_red).save('sample_images/labeled_video_3/1+%d.jpg'%f)

    if len(mask_yellow_orange_list) != 0:
        for mask_yellow_orange in mask_yellow_orange_list:
            #         print('type mask_yellow_orange:',type(mask_yellow_orange))
            #         rgb_img_yellow_orange = Image.fromarray(rgb_img_yellow_orange)
            #         print('type rgb_img_yellow_orange after:',type(rgb_img_yellow_orange))
            rgb_img_label = text_box_on_img(mask_yellow_orange, rgb_img_label, text='speed limit')
        plt.imshow(rgb_img_label)
        plt.show()
    #         Image.fromarray(rgb_img_yellow_orange).save('sample_images/labeled_video_3/2_%d.jpg'%g)

    if len(mask_white_list) != 0:
        for mask_white in mask_white_list:
            #         print('type output_white:',type(output_white))
            #         rgb_img_white = Image.fromarray(rgb_img_white)
            #         print('type rgb_img_white after:',type(rgb_img_white))
            rgb_img_label = text_box_on_img(mask_white, rgb_img_label, text='speed limit')
        plt.imshow(rgb_img_label)
        plt.show()
    #         Image.fromarray(rgb_img_final).save('sample_images/labeled_video_3/3_%d.jpg'%a)
    #         rgb_img_final = np.asarray(rgb_img_final)

    # Store all labeled frames:
    rgb_img_final = rgb_img_label
    output_labeled_img_list.append(rgb_img_final)
    b = 0
    for output_labeled_img in output_labeled_img_list:
        Image.fromarray(output_labeled_img).save('sample_images/labeled_video_5/output_labeled_frames_%d.jpg' % b)
        b += 1

    # Within each frame, store all cropped images:
    p = 0
    for output_crop_red in output_red_list:
        Image.fromarray(output_crop_red).save('sample_images/cropped_video_5/output_crop_red_%d_%d.jpg' % (frame_i, p))
        p += 1
    o = 0
    for output_crop_yellow_orange in output_yellow_orange_list:
        Image.fromarray(output_crop_yellow_orange).save(
            'sample_images/cropped_video_5/output_crop_yellow_orange_%d_%d.jpg' % (frame_i, o))
        o += 1
    u = 0
    for output_crop_white in output_white_list:
        Image.fromarray(output_crop_white).save(
            'sample_images/cropped_video_5/output_crop_white_%d_%d.jpg' % (frame_i, u))
        u += 1
    frame_i += 1
print('finish all frames!')
# len should = # of frames
print("length of output_labeled_img_list", len(output_labeled_img_list))

