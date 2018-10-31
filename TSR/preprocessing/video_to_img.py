import numpy as np
import cv2
import os

#change the video name 
count = 0
folder_num = 0
# !change the video name
Videos = ['GX010029_m.mp4','GX010033_m.mp4','GX010035_m.mp4']

for v in Videos:
    cap = cv2.VideoCapture(v)
    folder_num += 1
    
    # !change the path you want to save the frames
    Path = "/Users/DXX/Desktop/UACLASS/MM803/Project/img%d" %folder_num 
    
    os.makedirs(Path)
    os.chdir(Path)
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    # Read until video is completed
    while(cap.isOpened()):
        print("Cap is running")
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            count +=1
            if count %15 == 0:
                # Display the resulting frame
                #cv2.imshow('Frame',frame)
                cv2.imwrite("frame %d.jpg" % count,frame)
             
        # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        # Break the loop
        else: 
            break
        
cap.release()
cv2.destroyAllWindows()
