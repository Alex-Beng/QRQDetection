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
# WarpCorrection.py


def BilinearInterpolation(imgSrc:np.ndarray, h, w, sx:float, sy:float)->float:
    """
    对图片的指定位置做双线性插值
    :param imgSrc:源图像
    :param h: src的高度
    :param w: src的宽度
    :param sx: x位置
    :param sy: y位置
    :return: 所插入的值
    """
    intSx, intSy = int(sx), int(sy)
    if 0 <= intSx  < w - 1 and 0 <= intSy < h - 1:
        x1, x2 = intSx, intSx + 1
        y1, y2 = intSy, intSy + 1
        H1 = np.dot(np.array([x2 - sx, sx - x1]), imgSrc[y1: y2 + 1, x1:x2 + 1])
        return H1[0]*(y2 - sy) + H1[1]*(sy - y1)
    else:
        return imgSrc[intSy, intSx]

def WarpCorrection(imgSrc:np.ndarray, dots)->np.ndarray:
    assert len(dots) == 4

    # 四个点的顺序一定要按照左上，右上，右下，左下的顺时针顺序点
    d1, d2, d3, d4 = dots
    x1, x2, x3, x4 = d1[0], d2[0], d3[0], d4[0]
    y1, y2, y3, y4 = d1[1], d2[1], d3[1], d4[1]
    assert x1 < x2
    assert x4 < x3
    assert y1 < y4
    assert y2 < y3

    objW = np.round(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))
    objH = np.round(np.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2))

    # 在这里我简单地设为把所输入的四个点的位置，通过2D变换，变换为长方形的四个顶点的位置（以x1为起点）
    # t1, t2, t3, t4 = (y1, x1), (y1, x1 + objW), (y1 + objH, x1 + objW), (y1 + objH, x1),
    t1, t2, t3, t4 = (0, 0), (0, 300), (300, 300), (300, 0),


    rx1, rx2, rx3, rx4 = t1[1], t2[1], t3[1], t4[1]
    ry1, ry2, ry3, ry4 = t1[0], t2[0], t3[0], t4[0]

    # ================Step 0: 根据 8个点两两对应关系找到Homography矩阵================
    # 把8个约束写成方程组，以矩阵的形式表达
    m = np.array([
                  [y1, x1, 1, 0, 0, 0, -ry1 * y1, -ry1 * x1],
                  [0, 0, 0, y1, x1, 1, -rx1 * y1, -rx1 * x1],
                  [y2, x2, 1, 0, 0, 0, -ry2 * y2, -ry2 * x2],
                  [0, 0, 0, y2, x2, 1, -rx2 * y2, -rx2 * x2],
                  [y3, x3, 1, 0, 0, 0, -ry3 * y3, -ry3 * x3],
                  [0, 0, 0, y3, x3, 1, -rx3 * y3, -rx3 * x3],
                  [y4, x4, 1, 0, 0, 0, -ry4 * y4, -ry4 * x4],
                  [0, 0, 0, y4, x4, 1, -rx4 * y4, -rx4 * x4],
                ])

    vectorSrc = np.array([ry1, rx1, ry2, rx2, ry3, rx3, ry4, rx4])
    vectorSrc.shape = (1, 8)
    HFlat = np.dot(np.linalg.inv(m), np.transpose(vectorSrc))
    a, b, c, d, e, f, g, h = HFlat[0, 0],HFlat[1, 0],HFlat[2, 0],HFlat[3, 0],HFlat[4, 0],HFlat[5, 0],HFlat[6, 0],HFlat[7, 0]

    H = np.array([[a, b, c],
                  [d, e, f],
                  [g, h, 1]], dtype=np.float32)

    # ================Step 1: 通过对原图像四个顶点进行正向投射变换，确定目标图像区域================
    height, width = imgSrc.shape[:2]
    matrixOriginVertex = np.array([[0, 0, 1],
                                   [0, width - 1, 1],
                                   [height - 1, width - 1, 1] ,
                                   [height - 1, 0, 1]])

    result = np.dot(matrixOriginVertex, np.transpose(H))
    # minX = int(min(result[0, 1]/result[0, 2], result[1, 1]/result[1, 2], result[2, 1]/result[2, 2], result[3, 1]/result[3, 2]))
    # maxX = int(max(result[0, 1]/result[0, 2], result[1, 1]/result[1, 2], result[2, 1]/result[2, 2], result[3, 1]/result[3, 2]))
    # minY = int(min(result[0, 0]/result[0, 2], result[1, 0]/result[1, 2], result[2, 0]/result[2, 2], result[3, 0]/result[3, 2]))
    # maxY = int(max(result[0, 0]/result[0, 2], result[1, 0]/result[1, 2], result[2, 0]/result[2, 2], result[3, 0]/result[3, 2]))
    minX = 0
    maxX = 300
    minY = 0
    maxY = 300

    # ================Step 2: 反向变换+双二次插值校正图像================
    vtr = np.empty((0,3),dtype=np.float32) # 对应的是我的xy平面上的点
    for i in range(minY, maxY):
        arr1 = np.arange(minX, maxX)
        arr2 = np.ones(maxX - minX)
        vt1 = np.stack((arr2*i, arr1 , arr2), axis=-1)
        vtr = np.concatenate((vtr, vt1), axis=0)

    # 请注意，因为传进去的是规范化后(Y, X, 1)的值，所以得到的其实是(y/Z, x/Z, 1/Z的值)
    vts = np.dot(vtr,np.linalg.inv(np.transpose(H))) # 这个对应的是uv平面的齐次坐标，但是除了Z
    dstHeight, dstWidth = maxY - minY + 1, maxX - minX + 1
    imgDst = np.zeros((dstHeight, dstWidth, imgSrc.shape[2]), dtype=imgSrc.dtype)

    for (r, s) in zip(vtr, vts):
        ry, rx = int(r[0]), int(r[1])
        iy, ix = s[:2]
        # 需要解 [y, x] = [iy*(g*y + h*x + 1), ix*(g*y + h*x + 1)]这个方程
        TH = np.linalg.inv(np.array([[iy * g - 1, iy * h],
                                     [ix * g, ix * h - 1]]))

        vxy = np.dot(TH, np.array([[-iy], [-ix]]))
        sy, sx = vxy[0, 0], vxy[1, 0]

        if 0 <= round(sy) < height and 0 <= round(sx) < width:
            t_value = BilinearInterpolation(imgSrc, height, width, sx, sy)
            # print(t_value)
            imgDst[ry - minY, rx - minX] = t_value

    return imgDst