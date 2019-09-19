import sys
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass

import cv2
import copy
import pickle
import random
import sklearn
import datetime
import numpy as np
from math import *
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

def SHOW_IMAGE(image):
    now = datetime.datetime.now()
    now = str(now)
    cv2.imshow(now, image)
    key = cv2.waitKey()
    cv2.destroyWindow(now)
    return key

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

def ImageProcessSobel(image):
    #  get L channel
    t_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    t_cs = cv2.split(t_image)
    L_chn = t_cs[1]

    # do the sobel 
    x_g = cv2.Sobel(L_chn, cv2.CV_16S, 1, 0)
    y_g = cv2.Sobel(L_chn, cv2.CV_16S, 0, 1)
    grad_x = cv2.convertScaleAbs(x_g)
    grad_y = cv2.convertScaleAbs(y_g)

    SHOW_IMAGE(grad_x)
    SHOW_IMAGE(grad_y)

    grad = cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)

    SHOW_IMAGE(grad)

    _, grad_thre = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    
    block_size = int(sqrt(image.shape[0]*image.shape[1]/17))
    if block_size%2 != 1:
        block_size += 1
    thre_c = 7

    grad_thre = cv2.adaptiveThreshold(grad_thre, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 
    block_size, thre_c
    )

    SHOW_IMAGE(grad_thre)

    # get contours
    _, contours, hierachy = cv2.findContours(grad_thre,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    print(contours[0].shape)
    return contours, hierachy


def ImageProcess(image):
    #  get L channel
    t_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    t_cs = cv2.split(t_image)
    L_chn = t_cs[1]

    # SHOW_IMAGE(L_chn)

    

    # do the sobel 
    # x_g = cv2.Sobel(L_chn, cv2.CV_16S, 1, 0)
    # y_g = cv2.Sobel(L_chn, cv2.CV_16S, 0, 1)
    # grad_x = cv2.convertScaleAbs(x_g)
    # grad_y = cv2.convertScaleAbs(y_g)

    # SHOW_IMAGE(grad_x)
    # SHOW_IMAGE(grad_y)

    # grad = cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)

    # SHOW_IMAGE(grad)

    # _, grad_thre = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)

    # 使用非锐化掩蔽进行图像增强
    # L_chn_gaus = cv2.GaussianBlur(L_chn, (3, 3), 0.0)
    # L_chn_gaus_mask = L_chn-L_chn_gaus
    # L_chn += L_chn_gaus_mask

    # SHOW_IMAGE(L_chn)


    # lap_kernal = np.ndarray([1]*9).reshape(3, 3, 1)
    # lap_kernal[1, 1, 1] = -8
    # L_chn = cv2.Laplacian(L_chn, cv2.CV_8UC1)
    # SHOW_IMAGE(L_chn)

    # SHOW_IMAGE(L_chn)
    # rows, cols = L_chn.shape
    # C = 1
    # Gam = 4
    # for i in range(rows):
    #     for j in range(cols): ··    
    #         L_chn[i][j]=C*pow(L_chn[i][j], Gam)
    # SHOW_IMAGE(L_chn)

    # 使用look up table进行分段线性变换
    # SHOW_IMAGE(L_chn)

    # look_up_table = [0]*256
    
    # mid_pnt = (128, 64)

    # for i in range(0, mid_pnt[0]):
    #     look_up_table[i] = int(i*(mid_pnt[1]/mid_pnt[0]))
    # for i in range(mid_pnt[0], 256):
    #     look_up_table[i] = int(mid_pnt[1] + (255-mid_pnt[1])/(255-mid_pnt[0])*(i-mid_pnt[0])))
    # for i in range(L_chn.shape[0]):
    #     for j in range(L_chn.shape[1]):
    #         L_chn[i][j] = look_up_table[L_chn[i][j]]
    # SHOW_IMAGE(L_chn)

    # _, grad_thre = cv2.threshold(L_chn, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)

    block_size = int(sqrt(image.shape[0]*image.shape[1]/14))
    if block_size%2 != 1:
        block_size += 1
    thre_c = 8

    grad_thre = cv2.adaptiveThreshold(L_chn, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 
    block_size, thre_c
    )

    SHOW_IMAGE(grad_thre)

    # get contours
    _, contours, hierachy = cv2.findContours(grad_thre,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    # print(contours[0].shape)
    return contours, hierachy



if __name__ == "__main__":
    x = []
    y = []
    mode = input()

    # data_file = open("train_data.bin", 'rb')
    # label_file = open("train_label.bin", 'rb')
    # x = pickle.load(data_file)
    # y = pickle.load(label_file)
    # print(len(x))

    if mode == 'r':
        for i in range(1, 9):
            image_path = "../Pics/PicForMoment/%03d.jpg"%int(i)
            image = cv2.imread(image_path)

            contours = ImageProcess(image)

            for i in contours:
                # print(i, type(i), i.shape, type(contours))
                # break
                t_draw = copy.deepcopy(image)
                t_area = cv2.contourArea(i)
                if t_area < t_draw.shape[0]*t_draw.shape[1]/110:
                # if t_area < 100:
                    continue
                cv2.drawContours(t_draw, i, -1, (0,0,255), 3)

                t_key = SHOW_IMAGE(t_draw)
                if t_key == ord('y'):
                    y.append(1)
                elif t_key == ord('d'):
                    continue
                else:
                    y.append(0)
                
                t_moments = cv2.moments(i)
                t_hu_moments = cv2.HuMoments(t_moments)
                # 进行对数变换
                # 可能出现math domain error
                try: 
                    for i in range(0,7):
                        t_hu_moments[i] = -1* copysign(1.0, t_hu_moments[i]) * log10(abs(t_hu_moments[i]))
                except:
                    del y[-1]
                    continue

                x.append(t_hu_moments.reshape(-1))
    else:
        data_file = open("train_data.bin", 'rb')
        label_file = open("train_label.bin", 'rb')
        x = pickle.load(data_file)
        y = pickle.load(label_file)
        

    print(len(x))
    print(len(y))

    if mode == 'r':
        data_file = open("train_data.bin", 'wb')
        label_file = open("train_label.bin", 'wb')
        pickle.dump(x, data_file)
        pickle.dump(y, label_file)


    train_data, test_data, train_label, test_label = train_test_split(x, y, train_size=0.9, random_state=1)
    classifier = SVC(kernel='linear')
    classifier.fit(train_data, train_label)
    
    print(classifier.score(train_data, train_label))
    print(classifier.score(test_data, test_label))

    print(type(classifier.coef_), classifier.coef_)
    print(type(classifier.intercept_), classifier.intercept_)

    for i in range(3, 22):
        image_path = "../Pics/%03d.jpg"%int(i)
         
        image = cv2.imread(image_path)
        SHOW_IMAGE(image)
        # image = cv2.GaussianBlur(image, (3, 3), 0.0)
        t_draw = np.zeros(image.shape, np.uint8)   


        contours, hierachy = ImageProcess(image)
        # contours, hierachy = ImageProcessSobel(image)
        hierachy = hierachy.reshape(-1, 4)

        filted_once = []

        for idx, j in enumerate(contours):
            
            # 使用面积+长宽比筛
            t_min_rect = cv2.boundingRect(j)
            # if t_min_rect[2]*t_min_rect[3] < image.shape[0]*image.shape[1]/700:
            if t_min_rect[2]*t_min_rect[3] > image.shape[0]*image.shape[1]/17:
                # cv2.drawContours(t_draw, j, -1, (0,0,255), 1)
                continue
            if t_min_rect[2]*t_min_rect[3] < 50:
                # cv2.drawContours(t_draw, j, -1, (0,0,255), 1)
                continue
            if t_min_rect[2]/t_min_rect[3] > 0.5 and t_min_rect[2]/t_min_rect[3] < 2:
                pass
            else:
                # cv2.drawContours(t_draw, j, -1, (0,0,128), 1)
                continue  
            cv2.drawContours(t_draw, j, -1, (0,255,0), 1)
            filted_once.append((idx, j))

        SHOW_IMAGE(t_draw)
        t_draw = np.zeros(image.shape, np.uint8)  
        filted_twice = []

        # 利用不存在儿子轮廓的轮廓 and 三重嵌套关系进行筛选
        # 接下来是轮廓中心距离筛（待定） and 轮廓周长筛
        for j in filted_once:
            # 自己周长 and 爸爸周长
            self_peri = cv2.arcLength(j[1], True)
            father_peri = cv2.arcLength(contours[hierachy[j[0]][3]], True)
            if father_peri/self_peri > 2:
                continue

            
            if hierachy[j[0]][2] == -1 and MyNest(j[0], contours, hierachy, 2):
                # 找轮廓中心
                tt_draw = copy.deepcopy(t_draw)

                t_cont_center = MyContourCenter(j[1])
                t_cont_center = [int(i) for i in t_cont_center]
                t_cont_center = tuple(t_cont_center)
                cv2.circle(tt_draw, t_cont_center, 50, (255, 255, 0))
                cv2.drawContours(tt_draw, j[1], -1, (255, 255, 0))

                t_cont_center = MyContourCenter(contours[hierachy[j[0]][3]])
                t_cont_center = [int(i) for i in t_cont_center]
                t_cont_center = tuple(t_cont_center)
                cv2.circle(tt_draw, t_cont_center, 50, (255, 255, 255))
                cv2.drawContours(tt_draw, contours[hierachy[j[0]][3]], -1, (255, 255, 255))
                SHOW_IMAGE(tt_draw)

                cv2.drawContours(t_draw, j[1], -1, (0,255,0), 1)
                filted_twice.append(j)
            else:
                cv2.drawContours(t_draw, j[1], -1, (0,0,255), 1)
                pass
        SHOW_IMAGE(t_draw)

        t_draw = np.zeros(image.shape, np.uint8)  
        filted_third = []
        result_idxes = []
        print(len(filted_twice))
        if len(filted_twice) > 3:
            # 那就拿爷爷再筛一次
            for j in filted_twice:
                self_peri = cv2.arcLength(j[1], True)
                grafa_peri = cv2.arcLength(contours[hierachy[hierachy[j[0]][3]][3]], True)
                print(self_peri, grafa_peri)
                if grafa_peri/self_peri > 3 or grafa_peri/self_peri < 2:
                    cv2.drawContours(t_draw, j[1], -1, (0,0,255), 1)
                else:
                    cv2.drawContours(t_draw, j[1], -1, (0,255,0), 1)
                    filted_third.append(j)
                SHOW_IMAGE(t_draw)
            print("ya:", len(filted_third))
            # 那就包围矩形最大三个，谢谢
            if len(filted_third) > 3:
                bound_boxes = [(j[0], cv2.boundingRect(j[1])) for j in filted_third]
                bound_boxes = sorted(bound_boxes, key=lambda j: j[1][2]*j[1][3], reverse=True)[:3]

                t_draw = np.zeros(image.shape, np.uint8)  
                for j in bound_boxes:
                    cv2.drawContours(t_draw, contours[j[0]], -1, (0,0,255), 1)
                    SHOW_IMAGE(t_draw)
                result_idxes = [j[0] for j in bound_boxes]
            elif len(filted_third) == 3:
                result_idxes = [j[0] for j in filted_third]

        elif len(filted_twice) == 3:
            result_idxes = [j[0] for j in filted_twice]
        
        # 找中点
        print(result_idxes)
        points = [MyContourCenter(contours[j]).reshape(-1) for j in result_idxes]
        angles = MyVecAngles(points)    
        print(angles)
        
        mid_pnt_idx = angles.index(max(angles))
        min_pnt = points[mid_pnt_idx]
        min_pnt = tuple(min_pnt.astype(np.int16))
        cv2.circle(image, min_pnt, 5, (255, 0, 0))
        SHOW_IMAGE(image)

        # 通过最小二乘获得两个边
        

        # 用nm的分类器
        # t_draw = np.zeros(image.shape, np.uint8)  
        # for j in filted_twice:
        #     t_moments = cv2.moments(j[1])
        #     t_hu_moments = cv2.HuMoments(t_moments)
        #     try: 
        #         for k in range(0,7):
        #             t_hu_moments[k] = -1* copysign(1.0, t_hu_moments[k]) * log10(abs(t_hu_moments[k]))
        #     except:
        #         continue
        #     # print(classifier.predict(t_hu_moments.reshape(1, 7)))
        #     # print(np.matmul(classifier.coef_, t_hu_moments.reshape(7,)) + classifier.intercept_)
        #     if classifier.predict(t_hu_moments.reshape(1, 7))[0] == 1:
        #         cv2.drawContours(t_draw, j[1], -1, (0,255,0), 1)
        #     else:
        #         cv2.drawContours(t_draw, j[1], -1, (0,255,255), 1)
        # SHOW_IMAGE(t_draw)


        

        
