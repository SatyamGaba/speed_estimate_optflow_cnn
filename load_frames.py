"""
Author: Satyam Gaba <satyamgb321@gmaail.com>
"""

import os, sys
import cv2
import numpy as np

def load_frames(mode = "train"):
    """Extracts frames from the video
    NOTE: The function checks if directory exists or not,
    if the directory already exists, it does not extract 
    the frames. Be sure that there is no already existing directory
    and function let the function entirely."""
    dir_path = os.path.join("./data", mode+'_imgs')
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
        vidcap = cv2.VideoCapture('./data/{}.mp4'.format(mode))
        success,image = vidcap.read()
        count = 0
        while success:
            cv2.imwrite("%s/%d.jpg" % (dir_path,count), image)     # save frame as JPEG file      
            success,image = vidcap.read()
            count += 1
            if count % 1000 == 0:
                print("%d frames read" % count)
    else:
        print("Video frames already extracted \nSkipping extraction...")

    frame_cnt = len(os.listdir(dir_path))
    vid_frames = np.empty((frame_cnt,480,640,3),dtype='uint8')
    for i in range(0,frame_cnt):
                frame = cv2.imread(dir_path + '/' + str(i) + ".jpg")
                # print(",",frame.shape)
                vid_frames[i] = frame
                sys.stdout.write("\rLoading frame " + str(i))
    return vid_frames
load_frames()