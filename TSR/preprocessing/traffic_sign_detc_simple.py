# Read image(whatever color) as sample_img and display it.
video_path = 'sample_images/video_test_8.mov'
input_frames_set = video_to_frames(video_path)
print(len(input_frames_set))

output_labeled_img_list = []
frame_i = 0
output_red_list = []
output_yellow_orange_list = []
output_white_list = []
output_green_list = []

for frame in input_frames_set[:len(input_frames_set)-1]:
    print("start processing frame:",frame_i)
    blank_canvas_red,blank_canvas_yellow_orange,blank_canvas_white,sample_img, rgb_img, hsv_img, blur_img = load_prep_img(frame)
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

    #White
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
        #draw_contour_2(final_red_mask, blank_canvas_red, 2)
        # Compare similarity:
        similarity_level_ori_red = shape_compare(largest_ori_red_cnt, octagon_template_cnt)
#         plt.imshow(blank_canvas_red)
#         plt.show()
#         print(similarity_level_ori_red)

        # Re-draw contour after fixing convex defect
        convex_fixed_red_img = fix_convex_defect(final_red_mask)
        blank_canvas_red_2 = np.zeros(sample_img.shape, dtype=np.uint8)
        draw_contour_2(convex_fixed_red_img, blank_canvas_red_2, 2)
#         plt.imshow(blank_canvas_red_2)
#         plt.show()

        # Make comparison with the templet
        largest_fix_red_cnt, _ = find_contour(convex_fixed_red_img)
        similarity_level_fix_red = shape_compare(largest_fix_red_cnt, octagon_template_cnt)
        #print(similarity_level_fix_red)

        # Crop out red signs from ori_img:
        output_red = crop_possible_signs(similarity_level_ori_red, similarity_level_fix_red, final_red_mask, rgb_img)
        if output_red is not None:
            plt.imshow(output_red)
            plt.show()
            output_red_list.append(output_red)
#         Image.fromarray(output_red).save('sample_images/cropped_video_4/output_output_red_%d_%d.jpg'%(frame_i,a)


    # Yellow:
        # Draw contours on yellow_orange_canvas before fixing convex defects.
    largest_ori_yellow_orange_cnt, all_sorted_cnt_yellow_orange = find_contour(final_yellow_orange_mask)
    if all_sorted_cnt_yellow_orange is not None:
        # Draw original contour:
        draw_contour_2(final_yellow_orange_mask, blank_canvas_yellow_orange, 2)

        # Try compare similarity level method:
        similarity_level_ori_yellow_orange = shape_compare(largest_ori_yellow_orange_cnt, dimond_shape_template_cnt)
#         plt.imshow(blank_canvas_yellow_orange)
#         plt.show()
#         print(similarity_level_ori_yellow_orange)

#         # Try Shape detection method:
#         approx = cv2.approxPolyDP(largest_ori_yellow_orange_cnt,0.04*cv2.arcLength(largest_ori_yellow_orange_cnt,True),True)
#         print(len(approx))
#         if len(approx)==4:
#             print("dimond shape")
#             cv2.drawContours(sample_img,[largest_ori_yellow_orange_cnt],0,(0,0,255),-1)

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
        output_yellow_orange = crop_possible_signs(similarity_level_ori_yellow_orange, similarity_level_fix_yellow_orange, final_yellow_orange_mask, rgb_img)
        if output_yellow_orange is not None:
            plt.imshow(output_yellow_orange)
            plt.show()
            output_yellow_orange_list.append(output_yellow_orange)


    # Green:
        # Draw contours on yellow_orange_canvas before fixing convex defects.
    largest_ori_green_cnt, all_sorted_green_orange = find_contour(final_green_mask)
    if all_sorted_green_orange is not None:
        # Draw original contour:
        #draw_contour_2(final_green_mask, blank_green_orange, 2)
        similarity_level_ori_green = shape_compare(largest_ori_green_cnt, rectangle_template_cnt)

        convex_fixed_green_img = fix_convex_defect(final_green_mask)

        largest_fix_green_cnt, _ = find_contour(convex_fixed_green_img)
        similarity_level_fix_green = shape_compare(largest_fix_green_cnt, rectangle_template_cnt)
        print(similarity_level_fix_green)

        # Crop out yellow_orange signs from ori_img according to yellow_orange canvas:
        output_green = crop_possible_signs(similarity_level_ori_green, similarity_level_fix_green, final_green_mask, rgb_img)
        if output_green is not None:
            plt.imshow(output_green)
            plt.show()
            output_green_list.append(output_green)



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
        largest_fix_white_cnt,_ = find_contour(convex_fixed_white_img)
        largest_fix_white_cnt2, _ = find_contour(final_white_mask2)
        similarity_level_fix_white = shape_compare(largest_fix_white_cnt, rectangle_template_cnt)
        similarity_level_fix_white2 = shape_compare(largest_fix_white_cnt2, rectangle_template_cnt)

        if cv2.contourArea(largest_fix_white_cnt)<(w*h)*0.05 and cv2.contourArea(largest_fix_white_cnt2)<(w*h)*0.05:
            # Crop out white signs from ori_img according to white canvas:
            if similarity_level_fix_white < similarity_level_fix_white2:
                output_white = crop_possible_signs(similarity_level_ori_white, similarity_level_fix_white, final_white_mask1, rgb_img)
                if output_white is not None:
                    plt.imshow(output_white)
                    plt.show()
                    output_white_list.append(output_white)
#                 Image.fromarray(output_white).save('sample_images/cropped_video_4/output_output_white_%d_%d.jpg'%(frame_i,c))
            else:
                output_white = crop_possible_signs(similarity_level_ori_white, similarity_level_fix_white2, final_white_mask2, rgb_img)
                if output_white is not None:
                    plt.imshow(output_white)
                    plt.show()
                    output_white_list.append(output_white)
#                 Image.fromarray(output_white).save('sample_images/cropped_video_4/output_output_white_%d_%d.jpg'%(frame_i,c))
        else: continue
    frame_i += 1
print('finish all frames!')

print(len(output_red_list))
print(len(output_yellow_orange_list))
print(len(output_green_list))
print(len(output_white_list))

if len(output_red_list) != 0:
    a = 0
    for crop_red_img in output_red_list:
        Image.fromarray(crop_red_img).save('sample_images/cropped_video_8/output_output_red_%d.jpg'%a)
        a += 1

if len(output_yellow_orange_list) != 0:
    b = 0
    for crop_yellow_orange_img in output_yellow_orange_list:
        Image.fromarray(crop_yellow_orange_img).save('sample_images/cropped_video_8/output_output_yellow_%d.jpg'%b)
        b += 1

if len(output_green_list) != 0:
    c = 0
    for crop_green_img in output_green_list:
        Image.fromarray(crop_green_img).save('sample_images/cropped_video_8/output_output_green_%d.jpg'%c)
        c += 1

if len(output_white_list) != 0:
    d = 0
    for crop_white_img in output_white_list:
        Image.fromarray(crop_white_img).save('sample_images/cropped_video_8/output_output_white_%d.jpg'%d)
        d += 1

    
