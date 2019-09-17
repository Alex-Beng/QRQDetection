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
        
        t_draw = np.zeros(image.shape, np.uint8) 
        for idx in result_idxes: 
            cv2.drawContours(t_draw,  contours[idx], -1, (0,255,0), 1)
        SHOW_IMAGE(t_draw)

        # 获得了三个定位点
        # points = [MyContourCenter(contours[j]) for j in result_idxes]
        # angles = MyVecAngles(points)

        # mid_pnt_idx = angles.index(max(angles))
        # min_pnt = points[mid_pnt_idx]
        # min_pnt = tuple(min_pnt.astype(np.int16))
        # cv2.circle(image, min_pnt, 5, (255, 0, 0))
        # SHOW_IMAGE(image)

