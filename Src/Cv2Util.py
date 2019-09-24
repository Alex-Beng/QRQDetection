import os
import cv2
import datetime
import numpy as np
from math import *

DEBUGING = True
# DEBUGING = False

def SHOW_IMAGE(image):
    if DEBUGING:
        now = datetime.datetime.now()
        now = str(now)
        cv2.imshow(now, image)
        cv2.waitKey()
        cv2.destroyWindow(now)
    else:
        pass


def CvImageProcess(image):
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

def CvContourCenter(contour):
    M = cv2.moments(contour)
    if M['m00'] == 0:
        M['m00'] = 0.001
    x = M['m10']/M['m00']
    y = M['m01']/M['m00']
    return np.array((x, y))

def CvNest(cont_idx, contours, hierachy, check_layer):
    if check_layer == 1:
        if hierachy[cont_idx][3] != -1:
            return True
        else:
            return False
    elif check_layer > 1:
        if hierachy[cont_idx][3] != -1:
            father_center = CvContourCenter(contours[hierachy[cont_idx][3]])
            curr_center = CvContourCenter(contours[cont_idx])
            # print(father_center, curr_center)
            dist = np.linalg.norm(father_center.reshape(-1,)-curr_center.reshape(-1,))
            if dist < 100:
                return CvNest(hierachy[cont_idx][3], contours, hierachy, check_layer-1)
    return False
