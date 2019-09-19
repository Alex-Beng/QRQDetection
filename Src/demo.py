from Cv2Util import *
from MyUtil import *


# just fuck the code !!
if __name__ == "__main__":
    pic_root_path = "../Pics/"
    pics_paths = GetImgPaths(pic_root_path)
    pics_paths = [pic_root_path+i for i in pics_paths]
    
    for pic_path in pics_paths:
        print()
        print(pic_path)

        image = cv2.imread(pic_path)
        image_area = image.shape[0]*image.shape[1]

        contours, hierachy = ImageProcess(image)
        hierachy = hierachy.reshape(-1, 4)

        #初始未筛选的轮廓
        t_draw = np.zeros(image.shape, np.uint8) 
        for cont in contours:
            cv2.drawContours(t_draw, cont, -1, (0,0,255), 1)
        print("raw num: %d"%len(contours))
        SHOW_IMAGE(t_draw)

        # 第一轮用面积加长宽比筛选
        area_rate_thre = 17
        area_pixe_thre = 50
        area_wlrate_thre = 2.0

        filted_once = []
        t_draw = np.zeros(image.shape, np.uint8) 
        for cont_idx, cont in enumerate(contours):
            t_bound_rect = cv2.boundingRect(cont)
            t_area = t_bound_rect[2]*t_bound_rect[3]
            t_wlrate = t_bound_rect[2]/t_bound_rect[3]

            if t_area > image_area/area_rate_thre \
                or t_area < area_pixe_thre \
                    or t_wlrate < 1/area_wlrate_thre \
                        or t_wlrate > 2:
                continue
            else:
                cv2.drawContours(t_draw, cont, -1, (0,255,0), 1)
                filted_once.append((cont_idx, cont))
        print("once num: %d"%len(filted_once))
        SHOW_IMAGE(t_draw)

        # 第二轮使用轮廓关系进行筛选
        filted_twice = []
        t_draw = np.zeros(image.shape, np.uint8)  
        for cont_idx, cont in filted_once:
            # 先康康有没有爸爸，没有爸爸肯定不是
            if hierachy[cont_idx][3] == -1:
                continue
            
            self_peri = cv2.arcLength(cont, True)
            fath_peri = cv2.arcLength(contours[hierachy[cont_idx][3]], True)
            if fath_peri/self_peri < 2 \
                and hierachy[cont_idx][2] == -1 and MyNest(cont_idx, contours, hierachy, 2):
                cv2.drawContours(t_draw, cont, -1, (0,255,0), 1)
                filted_twice.append((cont_idx, cont))
            else:
                continue
                
        print("twice num: %d"%len(filted_twice))
        SHOW_IMAGE(t_draw)

        result_idxes = []
        t_draw = np.zeros(image.shape, np.uint8)  

        if len(filted_twice) > 3:
            filted_third = []
            # 用爷爷轮廓关系再筛一次
            for cont_idx, cont in filted_twice:
                if hierachy[hierachy[cont_idx][3]][3] == -1:
                    continue
                self_peri = cv2.arcLength(cont, True)
                graf_peri = cv2.arcLength(contours[hierachy[hierachy[cont_idx][3]][3]], True)
                if graf_peri/self_peri > 3 \
                    or graf_peri/self_peri < 2:
                    continue
                else:
                    filted_third.append((cont_idx, cont))
                    cv2.drawContours(t_draw, cont, -1, (0,255,0), 1)
            SHOW_IMAGE(t_draw)

            # 还是大于三个那几暴力回最大三个
            if len(filted_third) > 3:
                bound_boxes = [(j[0], cv2.boundingRect(j[1])) for j in filted_third]
                bound_boxes = sorted(bound_boxes, key=lambda j: j[1][2]*j[1][3], reverse=True)[:3]
                result_idxes = [j[0] for j in bound_boxes]
            elif len(filted_third) == 3:
                result_idxes = [j[0] for j in filted_third]
            else: 
                print("yaya，不足三个定位点")    
        elif len(filted_twice) == 3:
            result_idxes = [j[0] for j in filted_twice]
        else:
            print("yayaya，不足三个定位点")
        
        print("showing result")
        t_draw = np.zeros(image.shape, np.uint8) 
        for idx in result_idxes: 
            cv2.drawContours(t_draw,  contours[idx], -1, (0,255,0), 1)
        SHOW_IMAGE(t_draw)

        # 获得了三个定位点 
        # result_idxes = [hierachy[hierachy[j][3]][3] for j in result_idxes]  # 用最外面的轮廓
        cont_points = [contours[j].reshape(-1, 2) for j in result_idxes]   
        points = [MyContourCenter(contours[j]) for j in result_idxes]
        angles = MyVecAngles(points)

        mid_pnt_idx = angles.index(max(angles))
        t_idxes = list(range(3))
        del t_idxes[mid_pnt_idx]
        x_pnt_idx = t_idxes[0]
        y_pnt_idx = t_idxes[1]
        print(x_pnt_idx, mid_pnt_idx, y_pnt_idx)
        print(angles)

        x_vec = points[x_pnt_idx]-points[mid_pnt_idx]
        y_vec = points[y_pnt_idx]-points[mid_pnt_idx]

        # 三个定位区的最外轮廓
        out_cont_idxes = [hierachy[hierachy[j][3]][3] for j in result_idxes]
        out_conts = [contours[j].reshape(-1, 2) for j in out_cont_idxes]
        
        # 首先搞qrcode左上点
        mid_out_cont = out_conts[mid_pnt_idx]
        # cv2.drawContours(image, mid_out_cont.reshape(-1, 1, 2), -1, (0,255,128), 3)
        # SHOW_IMAGE(image)

        t = 10000
        qr_cor_1 = None
        for point in mid_out_cont:
            t_cor = MyGetCor(x_vec, y_vec, point-points[mid_pnt_idx]) 
            if np.sum(t_cor) < t:
                t = np.sum(t_cor)
                qr_cor_1 = point
            # print(t_cor)

        # qr_cor_1 = tuple(qr_cor_1.astype(np.int16))
        cv2.circle(image, tuple(qr_cor_1.astype(np.int16)), 10, (255, 0, 0))
        print("qr code cor1")
        # SHOW_IMAGE(image)            

        # 搞qrcode 2点
        qr_cor_2 = None
        y_out_cont = out_conts[y_pnt_idx]

        t = -10000
        for point in y_out_cont:
            t_cor = MyGetCor(x_vec, y_vec, point-points[mid_pnt_idx])
            if (t_cor[1]-qr_cor_1[1])-(t_cor[0]-qr_cor_1[0]) > t:
                t = (t_cor[1]-qr_cor_1[1])-(t_cor[0]-qr_cor_1[0])
                qr_cor_2 = point
        print(qr_cor_2)
        # qr_cor_2 = tuple(qr_cor_2.astype(np.int16))
        cv2.circle(image, tuple(qr_cor_2.astype(np.int16)), 10, (255, 0, 0))
        print("qr code cor2")
        # SHOW_IMAGE(image)

        # 搞qrcode 3点
        qr_cor_3 = None
        x_out_cont = out_conts[x_pnt_idx]

        t = -10000
        for point in x_out_cont:
            t_cor = MyGetCor(x_vec, y_vec, point-points[mid_pnt_idx])
            if (t_cor[0]-qr_cor_1[0])-(t_cor[1]-qr_cor_1[1]) > t:
                t = (t_cor[0]-qr_cor_1[0])-(t_cor[1]-qr_cor_1[1])
                qr_cor_3 = point
        print(qr_cor_3)
        # qr_cor_3 = tuple(qr_cor_3.astype(np.int16))
        cv2.circle(image, tuple(qr_cor_3.astype(np.int16)), 10, (255, 0, 0))
        print("qr code cor3")
        # SHOW_IMAGE(image)

        # 搞qrcode 4点
        qr_cor_4 = qr_cor_3 + (qr_cor_2-qr_cor_1)

        # 画出包围盒
        cv2.line(image, tuple(qr_cor_1.astype(np.int16)), tuple(qr_cor_2.astype(np.int16)), (0, 255, 0), 3)
        cv2.line(image, tuple(qr_cor_2.astype(np.int16)), tuple(qr_cor_4.astype(np.int16)), (0, 255, 0), 3)
        cv2.line(image, tuple(qr_cor_4.astype(np.int16)), tuple(qr_cor_3.astype(np.int16)), (0, 255, 0), 3)
        cv2.line(image, tuple(qr_cor_3.astype(np.int16)), tuple(qr_cor_1.astype(np.int16)), (0, 255, 0), 3)
        print("final result")
        SHOW_IMAGE(image)

        # mid_pnt = points[mid_pnt_idx]
        # mid_pnt = tuple(mid_pnt.astype(np.int16))
        # cv2.circle(image, mid_pnt, 50, (255, 0, 0))
        # SHOW_IMAGE(image)



        # la_points = [cont_points[mid_pnt_idx]]
        # lb_points = [cont_points[mid_pnt_idx]]

        # cnt = 0
        # for i in range(3):
        #     if i == mid_pnt_idx:
        #         continue
        #     if cnt == 0:
        #         la_points.append(cont_points[i])
        #         cnt += 1
        #     else:
        #         lb_points.append(cont_points[i])
        # # print(la_points[1][:3], lb_points[1][:3])
        # for cont in la_points:
        #     cont = cont.reshape(-1, 1, 2)
        #     cv2.drawContours(image,  cont, -1, (0,255,128), 3)
        # SHOW_IMAGE(image)

        # for cont in lb_points:
        #     cont = cont.reshape(-1, 1, 2)
        #     cv2.drawContours(image,  cont, -1, (255,255,128), 3)
        # SHOW_IMAGE(image)

        # la_points = np.concatenate(la_points, axis=0)
        # lb_points = np.concatenate(lb_points, axis=0)
        
        # xa = MyFitLine(la_points)
        # xb = MyFitLine(lb_points)

        # # print(xa, xb)
        # left_x = np.array([-10000, 1], dtype=np.float32)
        # righ_x = np.array([10000, 1], dtype=np.float32)
        
        # a_l_y = left_x.dot(xa)
        # a_r_y = righ_x.dot(xa)
        # cv2.line(image, (-10000, a_l_y), (10000, a_r_y), (255, 0, 0), 3)
        # SHOW_IMAGE(image)

        # b_l_y = left_x.dot(xb)
        # b_r_y = righ_x.dot(xb)
        # cv2.line(image, (-10000, b_l_y), (10000, b_r_y), (255, 0, 0), 3)
        # SHOW_IMAGE(image)

