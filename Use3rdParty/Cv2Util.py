import os
import cv2
import datetime
import numpy as np
from math import *

def SHOW_IMAGE(image):
    now = datetime.datetime.now()
    now = str(now)
    cv2.imshow(now, image)
    cv2.waitKey()
    cv2.destroyWindow(now)


def GetImgPaths(folder_path):
    paths = []
    for fpathe,dirs,fs in os.walk(folder_path):
        for f in fs:
            paths.append(f)
    return paths


def ImageProcess(image):
    #  get L channel
    t_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    t_cs = cv2.split(t_image)
    L_chn = t_cs[1]

    # adaptive threshold
    block_size = int(sqrt(image.shape[0]*image.shape[1]/14))
    if block_size%2 != 1:
        block_size += 1
    thre_c = 8

    grad_thre = cv2.adaptiveThreshold(L_chn, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 
    block_size, thre_c
    )

    # get contours
    _, contours, hierachy = cv2.findContours(grad_thre,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    return contours, hierachy
