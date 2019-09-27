import os
import cv2
import datetime
import numpy as np
import numba
from math import *
from Cv2Util import *
from Contours import *

# params 
# 正向变换/反向变换
MY_PERSP_TRANS_XY2UV = 0
MY_PERSP_TRANS_UV2XY = 1

# params
# 解方程方法
MY_DEPLOY_LS = 0
MY_DEPLOY_SVD = 1

@numba.jit
def GetImgPaths(folder_path):
    paths = []
    for fpathe,dirs,fs in os.walk(folder_path):
        for f in fs:
            paths.append(f)
    return paths

def MyBoundingBox(contour):
    x_min = min(contour, key=lambda pnt: pnt[0])[0]
    x_max = max(contour, key=lambda pnt: pnt[0])[0]
    y_min = min(contour, key=lambda pnt: pnt[1])[1]
    y_max = max(contour, key=lambda pnt: pnt[1])[1]
    return (x_min, y_min, x_max-x_min, y_max-y_min)

# 因为是四领域，所以 |dx| + |dy| 值域恒为1...
# 所以直接返数组长
def MyArcLen(contour):
    return len(contour)
    # prev_pnt = contour[-1]
    # curve_len = 0.0
    # for curr_pnt in range(contour):
    #     dx = curr_pnt[0] - prev_pnt[0]
    #     dy = curr_pnt[1] - prev_pnt[1]
    #     sum_d = dx + dy
    #     if sum_d == 1:
    #         curve_len += 1.0

        
@numba.jit
def MyDrawContours(image, contour,delta_value):
    delta_value = delta_value.astype(np.uint8)
    for pnt in contour:
        image[pnt[0], pnt[1]] += delta_value

# 因为是二值图，所以Fij == 0/1
# 可简化计算
@numba.jit
def MyContourCenter(contour):
    sum_x = 0
    sum_y = 0
    for pnt in contour:
        sum_x += pnt[0]
        sum_y += pnt[1]
    sum_x /= len(contour)
    sum_y /= len(contour)
    return np.array([sum_y, sum_x])

def MyNest(cont_idx, contours, hierachy, check_layer):
    if check_layer == 1:
        if hierachy[cont_idx][0] != -1:
            return True
        else:
            return False
    elif check_layer > 1:
        if hierachy[cont_idx][0] != -1:            
            father_center = MyContourCenter(contours[hierachy[cont_idx][0]-2])
            curr_center = MyContourCenter(contours[cont_idx])
            father_center = np.array(father_center).reshape(-1,)
            curr_center = np.array(curr_center).reshape(-1,)
            # print(father_center, curr_center)
            dist = np.linalg.norm(father_center-curr_center)
            if dist < 100:
                return MyNest(hierachy[cont_idx][0], contours, hierachy, check_layer-1)
    return False

@numba.jit
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
@numba.jit
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

@numba.jit
def MyIntegralImage(image):
    numrows = image.shape[0]
    numcols = image.shape[1]
    int_image = np.zeros((numrows, numcols), dtype=np.int32)

    for r in range(numrows):
        row_sum = 0
        for c in range(numcols):
            row_sum += image[r, c]
            if r == 0:
                int_image[r, c] = row_sum
            else:
                int_image[r, c] = int_image[r-1, c] + row_sum
    return int_image

@numba.jit
def MyadaptiveThreshold(image, block_size, thre_c):
    numrows = image.shape[0]
    numcols = image.shape[1]
    int_image = MyIntegralImage(image)
    # print(np.min(int_image), np.max(int_image))
    # SHOW_IMAGE((int_image/np.max(int_image)*255).astype(np.uint8) )
    # SHOW_IMAGE(image)
    dst_image = np.zeros((numrows, numcols), dtype=np.uint8)
    block_size //= 2
    print()
    for r in range(numrows):
        for c in range(numcols):
            l_r = r-block_size
            l_c = c-block_size
            r_r = r+block_size
            r_c = c+block_size
            if l_r < 0:
                l_r = 0
            if l_c < 0:
                l_c = 0
            if r_r >= numrows:
                r_r = numrows-1
            if r_c >= numcols:
                r_c = numcols-1
            # 利用积分图加速均值计算
            neighbour_sum = int_image[r_r, r_c] - int_image[r_r, l_c] - int_image[l_r, r_c] + int_image[l_r, l_c]
            # print(neighbour_sum)
            pix_num = (r_r-l_r)*(r_c-l_c)
            # if pix_num < 0:
            #     print(l_r, l_c, r_r, r_c)
            if image[r, c]*pix_num > neighbour_sum + thre_c*pix_num:
                dst_image[r, c] = 255
            else:
                dst_image[r, c] = 0
    return dst_image

def MyImageProcess(image):
    l_chn = MyBgr2L(image).astype(np.uint8)

    # adaptive threshold
    block_size = int(sqrt(image.shape[0]*image.shape[1]/14))
    if block_size%2 != 1:
        block_size += 1
    thre_c = -1

    l_thre = MyadaptiveThreshold(l_chn, block_size, thre_c)

    contours, hierachy = FindContours(l_thre)

    return contours, hierachy, l_thre

# contour is (row col) rather than (x, y)
@numba.jit
def MyLocalCor(x_vec, y_vec, origin, contour):
    corners = [233]*4

    xy_sum_min = 100000
    xy_sum_max = -1
    x_sub_y_max = -1
    y_sub_x_max = -1

    for point in contour:
        point = np.array(point[::-1])
        t_vec = point-origin
        t_cor = MyGetCor(x_vec, y_vec, t_vec)
        x, y = t_cor[0], t_cor[1]
        if x+y < xy_sum_min:
            corners[0] = point
            xy_sum_min = x+y
        if x+y > xy_sum_max:
            corners[3] = point
            xy_sum_max = x+y
        if x-y > x_sub_y_max:
            corners[1] = point
            x_sub_y_max = x-y
        if y-x > y_sub_x_max:
            corners[2] = point
            y_sub_x_max = y-x
    return corners

# 通过method参数指定方法
@numba.jit
def MyGetPerspTrans(all_corners, imagine_width, mode, method):
    # 四个对应点
    imagine_corners = [
        np.array([0, 0]), 
        np.array([0, imagine_width]), 
        np.array([imagine_width, 0]), 
        np.array([imagine_width, imagine_width])
    ]
    pixel_corners = [
        all_corners[0],
        all_corners[6],
        all_corners[9],
        all_corners[12]
    ]
    # print(pixel_corners, imagine_corners)
    # print(corr_corners)
    # print(all_corners)
    # A & B
    A = []
    B = []
    for xy, uv in zip(imagine_corners, pixel_corners):
        if mode == MY_PERSP_TRANS_UV2XY:
            uv, xy = xy, uv
        u = uv[0]
        v = uv[1]
        x = xy[0]
        y = xy[1]
        A.append(
            np.array([
                x, y, 1, 0, 0, 0, -x*u, -y*u
            ])
        )
        A.append(
            np.array([
                0, 0, 0, x, y, 1,-x*v, -y*v
            ])
        )
        B.append(
            np.array([
                u
            ])
        )
        B.append(
            np.array([
                v
            ])
        )
    A = np.array(A, dtype=np.float32).reshape(-1, 8)
    B = np.array(B, dtype=np.float32).reshape(-1, 1)
    print(A.shape, B.shape)

    # print(A)
    # print(B)
    if method == MY_DEPLOY_LS:
        # 计算最小二乘解
        A_T_I = np.matrix(A.T.dot(A))
        x = A_T_I.I.dot(A.T).dot(B)
        yayaya = np.array([1])
        return np.vstack((x, yayaya)).reshape(3, 3)
    elif method == MY_DEPLOY_SVD:
        # x = np.linalg.solve(A, B)
        _, x = cv2.solve(A, B, cv2.DECOMP_SVD)
        # print(x)
        yayaya = np.array([1])
        return np.vstack((x, yayaya)).reshape(3, 3)
        # return np.linalg.solve(A, B).reshape(3, 3)

# def MyGetLostCorner(refer_corners, out_three_corners):
#     delta = []
#     for i in range(1, 4):
#         delta.append(refer_corners[i] - refer_corners[0])
#     scale = [delta[] for i in range(2)]

def MySolveLine(point1, point2):
    x0 = point1[0]
    y0 = point1[1]
    x1 = point2[0]
    y1 = point2[1]
    return y0-y1, x1-x0, x0*y1-x1*y0

def MyGetQrScale(all_corners, pers_trans):
    # if flag:
    #     re_pers_trans = pers_trans.I
    # else:
    #     re_pers_trans = pers_trans
    uv_homo_cors = [
        np.array([i[0], i[1], 1]).reshape(-1, 1) for i in all_corners
    ]
    pixel_corners = [pers_trans.dot(i) for i in uv_homo_cors]
    # print(len(pixel_corners))
    # for i in pixel_corners:
    #     print(i, i.shape)
        # if flag:
        #     cv2.circle(t_draw, (int(i[0, 0])+10, int(i[1, 0])+10), 3, (255, 0, 0))
        # else:
        #     cv2.circle(t_draw, (int(i[0, 0])+10, int(i[1, 0])+10), 3, (0, 0, 255))
    rect_len_avg = 0
    for i in range(0, 9, 4):
        pixel_corners[i][0, 0]
        pixel_corners[i][1, 0]
        rect_len_avg += sqrt( 
            (pixel_corners[i][0, 0] - pixel_corners[i+1][0, 0])**2 + 
            (pixel_corners[i][1, 0] - pixel_corners[i+1][1, 0])**2
        )
        rect_len_avg += sqrt( 
            (pixel_corners[i][0, 0] - pixel_corners[i+2][0, 0])**2 + 
            (pixel_corners[i][1, 0] - pixel_corners[i+2][1, 0])**2
        )
        rect_len_avg += sqrt( 
            (pixel_corners[i+3][0, 0] - pixel_corners[i+2][0, 0])**2 + 
            (pixel_corners[i+3][1, 0] - pixel_corners[i+2][1, 0])**2
        )
        rect_len_avg += sqrt( 
            (pixel_corners[i+3][0, 0] - pixel_corners[i+1][0, 0])**2 + 
            (pixel_corners[i+3][1, 0] - pixel_corners[i+1][1, 0])**2
        )
    rect_len_avg /= 12

    code_len_avg = 0
    code_len_avg += sqrt( 
        (pixel_corners[0][0, 0] - pixel_corners[9][0, 0])**2 + 
        (pixel_corners[0][1, 0] - pixel_corners[9][1, 0])**2
    )
    code_len_avg += sqrt( 
        (pixel_corners[0][0, 0] - pixel_corners[6][0, 0])**2 + 
        (pixel_corners[0][1, 0] - pixel_corners[6][1, 0])**2
    )
    code_len_avg += sqrt( 
        (pixel_corners[6][0, 0] - pixel_corners[12][0, 0])**2 + 
        (pixel_corners[6][1, 0] - pixel_corners[12][1, 0])**2
    )
    code_len_avg += sqrt( 
        (pixel_corners[9][0, 0] - pixel_corners[12][0, 0])**2 + 
        (pixel_corners[9][1, 0] - pixel_corners[12][1, 0])**2
    )
    code_len_avg /= 4

    # print(rect_len_avg, code_len_avg, 7*code_len_avg/rect_len_avg) 
    width = 7*code_len_avg/rect_len_avg
    width = int(width+0.5)
    if width%2 == 0:
        return width+1
    else:
        return width

@numba.jit
def MyFillRect(image, rect, color):
    for r in range(rect[2]):
        for c in range(rect[3]):
            image[rect[0]+r, rect[1]+c] = color

# 解码与重构二维码，abandoned
@numba.jit
def MyDecodeRecon(bin_image, recon_image, xy2uv_h, bit_per_width, bit_rect_width, image):
    half_rect_width = bit_rect_width//2
    for r in range(bit_per_width):
        for c in range(bit_per_width):
            # 采样样
            rc_cor = (
                r*bit_rect_width+half_rect_width, 
                c*bit_rect_width+half_rect_width
            )
            xy_homo_cor = np.array(
                [rc_cor[1], rc_cor[0], 1], dtype=np.float32
            )
            uv_homo_cor = xy2uv_h.dot(xy_homo_cor)
            # print(uv_homo_cor.shape, uv_homo_cor)
            cv2.circle(image, (int(uv_homo_cor[0, 0]+0.5), int(uv_homo_cor[0, 1]+0.5)), 5, (0, 0, 255))
            # print(uv_homo_cor)
            if bin_image[int(uv_homo_cor[0, 0]+0.5), int(uv_homo_cor[0, 1]+0.5)] == 0:
                continue
            else:
                MyFillRect(recon_image, (r*bit_rect_width, c*bit_rect_width, bit_rect_width, bit_rect_width), 255)
            # recon_image[uv_homo_cor[0, 0], uv_homo_cor[0, 1]] = 255
        SHOW_IMAGE(image)
        SHOW_IMAGE(recon_image)

# 双线性插值, ins_x, ins_y为要插入的点
def MyBilinearInter(src_image:np.ndarray, ins_x:float, ins_y:float)->float:
    x1, x2 = int(ins_x), int(ins_x)+1
    y1, y2 = int(ins_y), int(ins_y)+1
    H1 = np.dot(np.array([x2 - ins_x, ins_x - x1]), src_image[y1: y2 + 1, x1:x2 + 1])
    return H1[0]*(y2 - ins_y) + H1[1]*(ins_y - y1)


# 重写一个getPerspective + WarpPerspective二合一函数
# 接受点为参数，返回校正后图
def MyPerspective2in1(src_image, src_dots, dst_dots, dst_width):
    d1, d2, d3, d4 = src_dots
    x1, x2, x3, x4 = d1[0], d2[0], d3[0], d4[0]
    y1, y2, y3, y4 = d1[1], d2[1], d3[1], d4[1]

    t1, t2, t3, t4 = dst_dots
    rx1, rx2, rx3, rx4 = t1[1], t2[1], t3[1], t4[1]
    ry1, ry2, ry3, ry4 = t1[0], t2[0], t3[0], t4[0]

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
    
    minX = 0
    maxX = dst_width
    minY = 0
    maxY = dst_width

    vtr = np.empty((0,3),dtype=np.float32) # 对应的是我的xy平面上的点
    for i in range(minY, maxY):
        arr1 = np.arange(minX, maxX)
        arr2 = np.ones(maxX - minX)
        vt1 = np.stack((arr2*i, arr1 , arr2), axis=-1)
        vtr = np.concatenate((vtr, vt1), axis=0)
    vts = np.dot(vtr,np.linalg.inv(np.transpose(H))) # 这个对应的是uv平面的齐次坐标，但是除了Z
    dstHeight, dstWidth = maxY - minY + 1, maxX - minX + 1
    dst_image = np.zeros((dstHeight, dstWidth), dtype=src_image.dtype)

    for (r, s) in zip(vtr, vts):
        ry, rx = int(r[0]), int(r[1])
        iy, ix = s[:2]
        TH = np.linalg.inv(np.array([[iy * g - 1, iy * h],
                                     [ix * g, ix * h - 1]]))

        vxy = np.dot(TH, np.array([[-iy], [-ix]]))
        sy, sx = vxy[0, 0], vxy[1, 0]

        t_value = MyBilinearInter(src_image, sx, sy)
        dst_image[ry, rx] = t_value

    return dst_image

