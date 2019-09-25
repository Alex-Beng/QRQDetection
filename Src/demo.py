from Cv2Util import *
from MyUtil import *

# just fuck the code !!
if __name__ == "__main__":
    pic_root_path = "../Pics/"
    pic_dst_path = "../Pics/Dst/"
    pics_paths = GetImgPaths(pic_root_path)
    # pics_paths = [pic_root_path+i for i in pics_paths]
    
    for pic_path in pics_paths:
        print()
        print(pic_root_path + pic_path)

        image = cv2.imread(pic_root_path + pic_path)
        image_area = image.shape[0]*image.shape[1]

        contours, hierachy, binary_image = MyImageProcess(image)
        del hierachy[0]
        SHOW_IMAGE(binary_image)

        #初始未筛选的轮廓
        t_draw = np.zeros(image.shape, np.uint8) 
        for cont in contours:
            MyDrawContours(t_draw, cont, np.array([0, 0, 255]))
            # MyDrawContours(t_draw, cont, -1, (0,0,255), 1)
        print("raw num: %d"%len(contours))
        SHOW_IMAGE(t_draw)


        # 第一轮用面积加长宽比筛选
        area_rate_thre = 17
        area_pixe_thre = 50
        area_wlrate_thre = 2.0

        filted_once = []
        t_draw = np.zeros(image.shape, np.uint8) 
        for cont_idx, cont in enumerate(contours):

            if len(cont) < 50:
                # print(cont)
                continue
                
            t_bound_rect = MyBoundingBox(cont)
            # print(t_bound_rect)
            t_area = t_bound_rect[2]*t_bound_rect[3]
            if t_bound_rect[3] != 0:
                t_wlrate = t_bound_rect[2]/t_bound_rect[3]
            else:
                t_wlrate = t_bound_rect[2]

            if t_area > image_area/area_rate_thre \
                or t_area < area_pixe_thre \
                    or t_wlrate < 1/area_wlrate_thre \
                        or t_wlrate > 2:
                continue
            else:
                MyDrawContours(t_draw, cont, np.array([0, 255, 0]))
                # cv2.drawContours(t_draw, cont, -1, (0,255,0), 1)
                filted_once.append(
                    copy.deepcopy((cont_idx, cont))
                    )
        print("once num: %d"%len(filted_once))
        SHOW_IMAGE(t_draw)


        # 第二轮使用轮廓关系进行筛选
        filted_twice = []
        t_draw = np.zeros(image.shape, np.uint8)  
        for cont_idx, cont in filted_once:
            # 先康康有没有爸爸，没有爸爸肯定不是
            if hierachy[cont_idx][0] == 1:
                continue
            
            # print(hierachy[cont_idx], hierachy[hierachy[cont_idx][0]-2])
            self_peri = MyArcLen(cont)
            fath_peri = MyArcLen(contours[hierachy[cont_idx][0]-2])
            
            # t_d = np.zeros(image.shape, np.uint8)  
            # father_center = MyContourCenter(contours[hierachy[cont_idx][0]-2])
            # curr_center = MyContourCenter(contours[cont_idx])
            # father_center = (int(father_center[1]), int(father_center[0]))
            # curr_center = (int(curr_center[1]), int(curr_center[0]))
            
            # MyDrawContours(t_d, contours[cont_idx], np.array([255, 0, 0]))
            # MyDrawContours(t_d, contours[hierachy[cont_idx][0]-2], np.array([255, 255, 0]))
            # cv2.circle(t_d, father_center, 50, (255, 255, 0))
            # cv2.circle(t_d, curr_center, 50, (255, 0, 0))
            # SHOW_IMAGE(t_d)


            if fath_peri/self_peri < 2 \
                and hierachy[cont_idx][1] == -1 and MyNest(cont_idx, contours, hierachy, 2):
                # and hierachy[cont_idx][1] == -1 :
                
                # cv2.drawContours(t_draw, cont, -1, (0,255,0), 1)
                MyDrawContours(t_draw, cont, np.array([0, 255, 0]))
                filted_twice.append(
                    copy.deepcopy((cont_idx, cont))
                    )
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
                if hierachy[ hierachy[cont_idx][0]-2 ][0] == 1:
                    continue
                self_peri = MyArcLen(cont)
                graf_peri = MyArcLen(contours[
                    hierachy[ hierachy[cont_idx][0]-2 ][0]-2
                    ])
                if graf_peri/self_peri > 3 \
                    or graf_peri/self_peri < 2:
                    continue
                else:
                    filted_third.append((cont_idx, cont))
                    MyDrawContours(t_draw, cont, np.array([0, 255, 0]))
            SHOW_IMAGE(t_draw)

            # 还是大于三个那几暴力回最大三个
            if len(filted_third) > 3:
                bound_boxes = [(j[0], MyBoundingBox(j[1])) for j in filted_third]
                bound_boxes = sorted(bound_boxes, key=lambda j: j[1][2]*j[1][3], reverse=True)[:3]
                result_idxes = [j[0] for j in bound_boxes]
            elif len(filted_third) == 3:
                result_idxes = [j[0] for j in filted_third]
            else: 
                print("yaya，不足三个定位点")
                continue
        elif len(filted_twice) == 3:
            result_idxes = [j[0] for j in filted_twice]
        else:
            print("yayaya，不足三个定位点")
            continue
        
        print("showing detection result")
        # t_draw = np.zeros(image.shape, np.uint8) 
        # for idx in result_idxes: 
        #     MyDrawContours(t_draw, contours[idx], np.array([0, 255, 0]))
        # SHOW_IMAGE(t_draw)

        # 获得了三个定位点 
        # result_idxes = [hierachy[hierachy[j][3]][3] for j in result_idxes]  # 用最外面的轮廓
        cont_points = [contours[j] for j in result_idxes]   
        points = [MyContourCenter(contours[j]) for j in result_idxes]
        # for point in points:
        #     # point = np.array(point[::-1])
        #     cv2.circle(image, tuple(point.astype(np.int16)), 50, (0, 0, 255))
        # SHOW_IMAGE(image)

        angles = MyVecAngles(points)

        mid_pnt_idx = angles.index(max(angles))
        t_idxes = list(range(3))
        del t_idxes[mid_pnt_idx]
        x_pnt_idx = t_idxes[0]
        y_pnt_idx = t_idxes[1]
        # print(x_pnt_idx, mid_pnt_idx, y_pnt_idx)
        # print(angles)

        x_vec = points[x_pnt_idx]-points[mid_pnt_idx]
        y_vec = points[y_pnt_idx]-points[mid_pnt_idx]

        
        out_cont_idxes = [hierachy[ hierachy[j][0]-2 ][0]-2 for j in result_idxes]
        out_conts = [contours[j] for j in out_cont_idxes]
        
        # 三个定位区的最外轮廓
        mid_out_cont = out_conts[mid_pnt_idx]
        y_out_cont = out_conts[y_pnt_idx]
        x_out_cont = out_conts[x_pnt_idx]

        t_draw = np.zeros(image.shape, np.uint8) 
        MyDrawContours(t_draw, mid_out_cont, np.array([255, 0, 0]))
        MyDrawContours(t_draw, y_out_cont, np.array([0, 255, 0]))
        MyDrawContours(t_draw, x_out_cont, np.array([0, 0, 255]))
        # SHOW_IMAGE(t_draw)
        # continue

        # 对于定位区外轮廓，获得
        # *_out_cont_corners = [[-x -y], [x, -y], [-x, y], [x, y]]
        mid_out_cont_corners = MyLocalCor(x_vec, y_vec, points[mid_pnt_idx], mid_out_cont)
        y_out_cont_corners = MyLocalCor(x_vec, y_vec, points[y_pnt_idx], y_out_cont)
        x_out_cont_corners = MyLocalCor(x_vec, y_vec, points[x_pnt_idx], x_out_cont)

        # t_draw = np.zeros(image.shape, np.uint8) 
        color = [(128, 255, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
        all_corners = []
        for c_idx, corner in enumerate(mid_out_cont_corners):
            all_corners.append(corner)
            cv2.circle(t_draw, tuple(corner.astype(np.int16)), 10, color[c_idx])
            cv2.circle(image, tuple(corner.astype(np.int16)), 3, color[c_idx])
            # SHOW_IMAGE(t_draw)
        for c_idx, corner in enumerate(y_out_cont_corners):
            all_corners.append(corner)
            cv2.circle(t_draw, tuple(corner.astype(np.int16)), 10, color[c_idx])
            cv2.circle(image, tuple(corner.astype(np.int16)), 3, color[c_idx])
        for c_idx, corner in enumerate(x_out_cont_corners):
            all_corners.append(corner)
            cv2.circle(t_draw, tuple(corner.astype(np.int16)), 10, color[c_idx])
            cv2.circle(image, tuple(corner.astype(np.int16)), 3, color[c_idx])
        SHOW_IMAGE(t_draw)

        # 透视变换求
        # qrc_width = 300
        # persp_trans = MyGetPerspTrans(all_corners, qrc_width)
        # # print(persp_trans)

        # loss_corner_uv = np.array([300, 300, 1], dtype=np.float32)
        # persp_trans = np.matrix(persp_trans)
        # loss_corner_xy = persp_trans.dot(loss_corner_uv).astype(np.int16)
        # # print(loss_corner_xy)

        # cv2.circle(image, (loss_corner_xy[0, 0], loss_corner_xy[0, 1]), 10, (255, 128, 128))
        # SHOW_IMAGE(image)

        # 直接莽平行四边形
        # loss_corner = all_corners[6]+all_corners[9]-all_corners[0]
        # loss_corner = loss_corner.astype(np.int16)


        # 利用交点求

        a0, b0, c0 = MySolveLine((all_corners[6][0], all_corners[6][1]), (all_corners[7][0], all_corners[7][1]))
        a1, b1, c1 = MySolveLine((all_corners[9][0], all_corners[9][1]), (all_corners[11][0], all_corners[11][1]))

        D = a0*b1 - a1*b0
        x = (b0*c1 - b1*c0)/D
        y = (a1*c0 - a0*c1)/D

        loss_corner = (int(x), int(y))
        all_corners.append(np.array([x, y], dtype=np.int16))
        # print(all_corners)
        # print(loss_corner)

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              # cv2.circle(image, (all_corners[0][0], all_corners[0][1]), 10, (255, 0, 0))
        # cv2.circle(image, (all_corners[9][0], all_corners[9][1]), 10, (255, 0, 0))
        # cv2.circle(image, (all_corners[6][0], all_corners[6][1]), 10, (255, 0, 0))
        # cv2.circle(image, (loss_corner[0], loss_corner[1]), 10, (255, 0, 0))

        cv2.line(image, (all_corners[0][0], all_corners[0][1]), (all_corners[9][0], all_corners[9][1]), (0, 0, 255), 1)
        cv2.line(image, (loss_corner[0], loss_corner[1]), (all_corners[9][0], all_corners[9][1]), (0, 0, 255), 1)
        cv2.line(image, (loss_corner[0], loss_corner[1]), (all_corners[6][0], all_corners[6][1]), (0, 0, 255), 1)
        cv2.line(image, (all_corners[0][0], all_corners[0][1]), (all_corners[6][0], all_corners[6][1]), (0, 0, 255), 1)
        SHOW_IMAGE(image)
        if not DEBUGING:
            cv2.imwrite(pic_dst_path+pic_path, image)

        # 2 decode part

        # 2.1 获得透视变换矩阵
        qrcode_width = 600
        persp_trans = MyGetPerspTrans(all_corners, qrcode_width, MY_PERSP_TRANS_XY2UV)
        re_persp_trans = MyGetPerspTrans(all_corners, qrcode_width, MY_PERSP_TRANS_UV2XY)

        # print(persp_trans)

        # 2.2 通过H矩阵获得Scale
        # MyGetQrScale(all_corners, persp_trans)
        # bit_per_width = MyGetQrScale(all_corners, re_persp_trans)
        # bit_rect_width = qrcode_width//bit_per_width
        # recon_qrcode_image = np.zeros((qrcode_width, qrcode_width, 3), dtype=np.uint8)

        # print(bit_per_width, bit_rect_width)
        # t_draw = copy.deepcopy(image)
        # MyDecodeRecon(binary_image, recon_qrcode_image, persp_trans, bit_per_width, bit_rect_width, t_draw)


        
        # 康一下opencv的效果，以及对比
        # pts1 = [
        #     all_corners[0],
        #     all_corners[6],
        #     all_corners[9],
        #     all_corners[12]
        # ]
        # pts1 = np.array(pts1, dtype=np.float32).reshape(-1, 2)
        # pts2 = [
        #     np.array([0, 0]), 
        #     np.array([0, qrcode_width]), 
        #     np.array([qrcode_width, 0]), 
        #     np.array([qrcode_width, qrcode_width])
        # ]
        # pts2 = np.array(pts2, dtype=np.float32).reshape(-1, 2)
        # cv_persp_trans = cv2.getPerspectiveTransform(pts1, pts2)
        # dst = cv2.warpPerspective(image, cv_persp_trans, (qrcode_width, qrcode_width))
        # SHOW_IMAGE(dst)

        # dst = cv2.warpPerspective(image, re_persp_trans, (qrcode_width, qrcode_width))
        # SHOW_IMAGE(dst)

        # dst = cv2.warpPerspective(image, persp_trans.I, (qrcode_width, qrcode_width))
        # SHOW_IMAGE(dst)