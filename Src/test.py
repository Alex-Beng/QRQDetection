# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from MyUtil import *
from Cv2Util import *
from multiprocessing.dummy import Process
import cv2 as cv

windowCount = int(0)
mainWinName = "source"

def WarpImage(imgSrc:np.ndarray, dots:tuple, count)->None:
    imgBLiner = WarpCorrection(imgSrc, dots)

    winName:str = f"result BLiner {count}"
    cv.namedWindow(winName)
    cv.imshow(winName, imgBLiner)

    # winName:str = f"result nearest {count}"
    # cv.namedWindow(winName)
    # cv.imshow(winName, imgNearest)

    cv.waitKey(0)
    cv.destroyWindow(winName)

class WarpCorrectionMgr:
    def __init__(self, imgSrc):
        self.__clickTime = 0
        self.__imgSrc = imgSrc.copy()
        self.__imgDrawn = imgSrc.copy()
        self.__dots = []

    @property
    def sourceImage(self):
        return self.__imgSrc

    @property
    def drawnImage(self):
        return self.__imgDrawn

    @drawnImage.setter
    def drawnImage(self, newImg):
        self.__imgDrawn = newImg

    @property
    def clickTime(self):
        return self.__clickTime

    @clickTime.setter
    def clickTime(self, v):
        self.__clickTime = v

    @property
    def dots(self):
        return self.__dots

    @staticmethod
    def MouseCallback(event, x, y, flags, param):
        # 四个点的顺序一定要按照左上，右上，右下，左下的顺时针顺序点
        if event == cv.EVENT_LBUTTONDBLCLK:
            clickTime = param.clickTime
            cv.circle(param.drawnImage, (x, y), 8, (0, 0, 255),-1)
            param.dots.append((x, y))
            cv.imshow(mainWinName, param.drawnImage)

            if clickTime + 1 == 4:
                global windowCount
                p = Process(target=WarpImage, args=(param.sourceImage, param.dots.copy(), windowCount))
                p.daemon = True
                p.start()

                param.drawnImage = param.sourceImage.copy()
                cv.imshow(mainWinName,param.sourceImage)
                param.dots.clear()
                windowCount += 1

            param.clickTime = (clickTime + 1) % 4


if __name__ == "__main__":
    cv.namedWindow(mainWinName)
    imgSrc = cv.imread("../Pics/Src/001.jpg")
    # imgSrc = cv.resize(imgSrc, (int(imgSrc.shape[1]/4), int(imgSrc.shape[0]/4)))

    mgr = WarpCorrectionMgr(imgSrc)
    cv.setMouseCallback(mainWinName, WarpCorrectionMgr.MouseCallback, mgr)

    cv.imshow(mainWinName, imgSrc)
    cv.waitKey(0)
    cv.destroyAllWindows()


# from Cv2Util import *
# from MyUtil import *
# from Contours import *



# if __name__ == "__main__":
#     # pic_root_path = "./Pics/"
#     # pics_paths = GetImgPaths(pic_root_path)
#     # pics_paths = [pic_root_path+i for i in pics_paths]
    
#     # for pic_path in pics_paths:
#     #     print()
#     #     print(pic_path)

#     #     image = cv2.imread(pic_path)
#     #     image_area = image.shape[0]*image.shape[1]
#     #     contours, hierachy = MyImageProcess(image)



#     image = np.zeros((10, 11), dtype=np.int16)
#     image[0, 0] = 1
#     image[1, 0] = 1
#     image[2, 0] = 1
#     image[2, 1] = 1
#     image[2, 2] = 1
#     image[1, 2] = 1
#     image[0, 2] = 1
#     image[0, 1] = 1

#     image[0, 3] = 1
#     image[0, 4] = 1
#     image[1, 4] = 1
#     image[2, 4] = 1
#     image[2, 3] = 1

#     image[9, 9] = 1
#     print(image)
    
#     contours, hierachy = FindContours(image)
#     for i in contours:
#         print(i)
#     print()
#     for i in hierachy:
#         print(i)

#     # image = cv2.imread("./Pics/001.jpg")
#     # My_L = MyBgr2L(image)
#     # SHOW_IMAGE(My_L.astype(np.uint8))
#     # L_chn = My_L.astype(np.uint8)

#     # # adaptive threshold
#     # block_size = int(sqrt(image.shape[0]*image.shape[1]/14))
#     # if block_size%2 != 1:
#     #     block_size += 1
#     # thre_c = 0

#     # grad_thre = MyadaptiveThreshold(L_chn, block_size, thre_c)
#     # SHOW_IMAGE(grad_thre)

#     # contours, hierachy = FindContours(grad_thre)

    




    

#     # contours, hierachy = FindContours(grad_thre)
#     # t_draw = np.zeros(image.shape, dtype=np.uint8)
#     # for i in contours:
#     #     MyDrawContours(t_draw, i, np.array([0, 0, 255]))
#     # SHOW_IMAGE(t_draw) 
    

#     # print(hierachy)
    


#     # points = [
#     #     np.array([0, 0]),
#     #     np.array([0, 100]),
#     #     np.array([100, -100])
#     # ]
#     # angles = MyVecAngles(points)
#     # print(angles)


#     # image = np.zeros((100, 100, 3), dtype=np.uint8)
#     # SHOW_IMAGE(image)
#     # image = cv2.rectangle(image, (25, 25), (75, 75), 255)
#     # contours, hierachy = ImageProcess(image)

#     # print(len(contours))

#     # for j in contours:
#     #     t_moments = cv2.moments(j)
#     #     t_hu_moments = cv2.HuMoments(t_moments)
#     #     print(t_hu_moments)
#     #     try: 
#     #         for k in range(0,7):
#     #             t_hu_moments[k] = -1* copysign(1.0, t_hu_moments[k]) * log10(abs(t_hu_moments[k]) + 1)
#     #     except Exception as e:
#     #         print("yayaya")
#     #         print(e)
#     #         continue
#     #     print(t_hu_moments)
#     #     # print(classifier.predict(t_hu_moments.reshape(1, 7)))
#     #     # print(np.matmul(classifier.coef_, t_hu_moments.reshape(7,)) + classifier.intercept_)
#     #     cv2.drawContours(image, j, -1, (0,255,0), 3)
#     #     SHOW_IMAGE(image)
#             # cv2.drawContours(t_draw, j[1], -1, (0,255,255), 1)


#     # SHOW_IMAGE(image)
#     # image = np.zeros((100, 100, 3), dtype=np.int8)
#     # MyDrawContours(image, contours[0], 100)
#     # center = MyContourCenter(contours[0])
#     # print(center.shape)
#     # image[int(center[0, 0]), int(center[0, 1])] += 100

#     # SHOW_IMAGE(image)
#     # MyDrawContours(image, contours[0], -100)
#     # SHOW_IMAGE(image)