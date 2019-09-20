import os
import cv2
import datetime
import numpy as np

def GetImgPaths(folder_path):
    paths = []
    for fpathe,dirs,fs in os.walk(folder_path):
        for f in fs:
            paths.append(f)
    return paths


def MyDrawContours(image, contour,delta_value):
    delta_value = delta_value.astype(np.uint8)
    for pnt in contour:
        image[pnt[0], pnt[1]] += delta_value

# 因为是二值图，所以Fij == 0/1
# 可简化计算
def MyContourCenter(contour):
    sum_x = 0
    sum_y = 0
    for pnt in contour:
        sum_x += pnt[0]
        sum_y += pnt[1]
    sum_x /= len(contour)
    sum_y /= len(contour)
    return (sum_x, sum_y)

def MyNest(cont_idx, contours, hierachy, check_layer):
    if check_layer == 1:
        if hierachy[cont_idx][0] != -1:
            return True
        else:
            return False
    elif check_layer > 1:
        if hierachy[cont_idx][0] != -1:
            father_center = MyContourCenter(contours[hierachy[cont_idx][0]])
            curr_center = MyContourCenter(contours[cont_idx])
            father_center = np.array(father_center).reshape(-1,)
            curr_center = np.array(curr_center).reshape(-1,)
            # print(father_center, curr_center)
            dist = np.linalg.norm(father_center-curr_center)
            if dist < 100:
                return MyNest(hierachy[cont_idx][0], contours, hierachy, check_layer-1)
    return False

def MyVecAngles(points):
    vecs = [ [j-points[(i+1)%3], j-points[(i+2)%3]] for i,j in enumerate(points)]
    angles = []
    for v in vecs:
        lens = [np.sqrt(x.dot(x)) for x in v]
        # print("lens:", lens)
        cos = v[0].dot(v[1])/(lens[0]*lens[1])
        angle_rad = np.arccos(cos)
        # print("angle_:", angle_)
        angle = angle_rad*360/2/np.pi
        # print("angle:", angle)
        angles.append(angle)
    return angles

def MyFitLine(points):
    A = [ [v[0], 1] for v in points]
    B = [v[1] for v in points]
    A = np.array(A, dtype=np.float32).reshape(-1, 2)
    B = np.array(B, dtype=np.float32).reshape(-1, 1)

    # A_T_I = np.linalg.inv(A.T.dot(A))
    A_T_I = np.matrix(A.T.dot(A))
    

    x = A_T_I.I.dot(A.T).dot(B)
    return x

def MyGetCor(x0, y0, z):
    A = [[x0[i], y0[i]] for i in range(2)]
    A = np.array(A, dtype=np.float32).reshape(2, 2)
    A = np.matrix(A)
    B = z.reshape(2, 1)

    return A.I.dot(B)


# 只求L通道，可简化计算
def MyBgr2L(src):
    dst = np.zeros((src.shape[0], src.shape[1]), dtype=np.float32)
    T = np.array(
        [0.072169, 0.715160, 0.212671],
     dtype=np.float32)


    for r in range(src.shape[0]):
        for c in range(src.shape[1]):
            Y = T.dot(src[r, c].astype(np.float32).reshape(-1, 1))
            # print(Y.shape)
            Y = Y[0]/255
            if Y > 0.008856451679035631:
                dst[r, c] =  Y**(1/3)
            else:
                dst[r, c] = 7.787037037037035*Y + 0.13793103448275862
            dst[r, c] = 116*dst[r, c] - 16
            
            if dst[r, c] < 0:
                dst[r, c] = 0 
    return dst