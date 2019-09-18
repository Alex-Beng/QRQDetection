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
    pnt_num = contour.shape[0]
    if delta_value < 255:
        for i in range(pnt_num):
            image[contour[i, 0, 1], contour[i, 0, 0]] += delta_value
    elif delta_value == 255:
        for i in range(pnt_num):
            image[contour[i, 0, 1], contour[i, 0, 0]] += 1
            image[contour[i, 0, 1], contour[i, 0, 0]] = -image[contour[i, 0, 1], contour[i, 0, 0]]
    elif delta_value == 256:    
        for i in range(pnt_num):
            image[contour[i, 0, 1], contour[i, 0, 0]] = -image[contour[i, 0, 1], contour[i, 0, 0]]

def MyContourCenter(contour):
    M = cv2.moments(contour)
    if M['m00'] == 0:
        M['m00'] = 0.001
    x = M['m10']/M['m00']
    y = M['m01']/M['m00']
    return np.array((x, y))

def MyNest(cont_idx, contours, hierachy, check_layer):
    if check_layer == 1:
        if hierachy[cont_idx][3] != -1:
            return True
        else:
            return False
    elif check_layer > 1:
        if hierachy[cont_idx][3] != -1:
            father_center = MyContourCenter(contours[hierachy[cont_idx][3]])
            curr_center = MyContourCenter(contours[cont_idx])
            # print(father_center, curr_center)
            dist = np.linalg.norm(father_center.reshape(-1,)-curr_center.reshape(-1,))
            if dist < 100:
                return MyNest(hierachy[cont_idx][3], contours, hierachy, check_layer-1)
    return False

def MyVecAngles(points):
    vecs = [ [j-points[(i+1)%3], j-points[(i+2)%3]] for i,j in enumerate(points)]
    angles = []
    for v in vecs:
        lens = [np.sqrt(x.dot(x)) for x in v]
        # print("lens:", lens)
        cos = v[0].dot(v[1])/(lens[0]*lens[1])
        angle_ = np.arccos(cos)
        # print("angle_:", angle_)
        angle = angle_*360/2/np.pi
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
    
