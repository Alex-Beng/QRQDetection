from Cv2Util import *
from MyUtil import *
from Contours import *



if __name__ == "__main__":
    image = np.zeros((10, 11), dtype=np.int16)
    image[0, 0] = 1
    image[1, 0] = 1
    image[2, 0] = 1
    image[2, 1] = 1
    image[2, 2] = 1
    image[1, 2] = 1
    image[0, 2] = 1
    image[0, 1] = 1

    image[0, 3] = 1
    image[0, 4] = 1
    image[1, 4] = 1
    image[2, 4] = 1
    image[2, 3] = 1

    image[9, 9] = 1
    


    # image = cv2.imread("../Pics/001.jpg")
    
    # t_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    # t_cs = cv2.split(t_image)
    # L_chn = t_cs[1]

    # # adaptive threshold
    # block_size = int(sqrt(image.shape[0]*image.shape[1]/14))
    # if block_size%2 != 1:
    #     block_size += 1
    # thre_c = 8

    # grad_thre = cv2.adaptiveThreshold(L_chn, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 
    # block_size, thre_c
    # )
    # SHOW_IMAGE(grad_thre)

    print(image)
    contours, hierachy = FindContours(image)
    for i in contours:
        print(i)
    print(hierachy)
    


    # points = [
    #     np.array([0, 0]),
    #     np.array([0, 100]),
    #     np.array([100, -100])
    # ]
    # angles = MyVecAngles(points)
    # print(angles)


    # image = np.zeros((100, 100, 3), dtype=np.uint8)
    # SHOW_IMAGE(image)
    # image = cv2.rectangle(image, (25, 25), (75, 75), 255)
    # contours, hierachy = ImageProcess(image)

    # print(len(contours))

    # for j in contours:
    #     t_moments = cv2.moments(j)
    #     t_hu_moments = cv2.HuMoments(t_moments)
    #     print(t_hu_moments)
    #     try: 
    #         for k in range(0,7):
    #             t_hu_moments[k] = -1* copysign(1.0, t_hu_moments[k]) * log10(abs(t_hu_moments[k]) + 1)
    #     except Exception as e:
    #         print("yayaya")
    #         print(e)
    #         continue
    #     print(t_hu_moments)
    #     # print(classifier.predict(t_hu_moments.reshape(1, 7)))
    #     # print(np.matmul(classifier.coef_, t_hu_moments.reshape(7,)) + classifier.intercept_)
    #     cv2.drawContours(image, j, -1, (0,255,0), 3)
    #     SHOW_IMAGE(image)
            # cv2.drawContours(t_draw, j[1], -1, (0,255,255), 1)


    # SHOW_IMAGE(image)
    # image = np.zeros((100, 100, 3), dtype=np.int8)
    # MyDrawContours(image, contours[0], 100)
    # center = MyContourCenter(contours[0])
    # print(center.shape)
    # image[int(center[0, 0]), int(center[0, 1])] += 100

    # SHOW_IMAGE(image)
    # MyDrawContours(image, contours[0], -100)
    # SHOW_IMAGE(image)