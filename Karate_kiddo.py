# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 16:09:57 2021

@author: SC
"""

# import keyboard

# keyboard.wait('esc')

import cv2 
# import Image
from PIL import ImageGrab
import numpy as np
from win32api import GetSystemMetrics
import os
import matplotlib.pyplot as plt

from win32api import GetSystemMetrics

width = GetSystemMetrics(0)
height =  GetSystemMetrics(1)


image_size = 600

refPt = []

final_boundaries = []
image = None
image_copy = None
# print("Width =", GetSystemMetrics(0))
# print("Height =", GetSystemMetrics(1))

kman_path = r"D:\Python Projects\Telegram_game\Karate\karate_man.PNG"

def click_and_crop(event, x, y, flags, param):
    global refPt, image
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        # print(refPt)
    elif event == cv2.EVENT_LBUTTONUP:
        refPt.append((x, y))
        final_boundaries.append((refPt[0],refPt[1]))
        cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("Image", image)
        
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        clone = image.copy()
        cv2.rectangle(clone, refPt[0], (x, y), (255, 255, 0), 2)
        cv2.imshow("Image", clone)
        
    
def disp_img():
    global image,image_copy
    img=ImageGrab.grab(bbox=(None))
    img_np = np.array(img)
    img_resize = cv2.resize(img_np,(image_size,image_size))
    image = img_resize
    img_copy = img_resize.copy()
    cv2.imshow('Image',image)
    cv2.setMouseCallback("Image", click_and_crop)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return(final_boundaries)

def save_roi(img,final_boundaries):
    print(final_boundaries)
    if len(final_boundaries)>0:
        for i in range(len(final_boundaries)):
            x1 = final_boundaries[i][0][0]
            x2 = final_boundaries[i][1][0]
            y1 = final_boundaries[i][0][1]
            y2 = final_boundaries[i][1][1]
            print(x1,x2,y1,y2)
            # plt.imshow(img[y1:y2,x1:x2])
            # plt.show()
            crop_img = img[y1:y2,x1:x2]
            file_name = "roi_"+ str(i) +".jpg"
            cv2.imwrite(file_name,crop_img)
    else:
        return None

def search_region(path):
    # fullscreen
    im=ImageGrab.grab()
    im.show()
    
    # part of the screen
    im=ImageGrab.grab(bbox=(10,10,500,500))
    im.show()
    
    # to file
    ImageGrab.grab_to_file('im.png')
    return

if __name__ == "__main__":
    # disp_img()
    # save_roi(image, final_boundaries)
    # print(os.system)
    search_region(kman_path)
    
    