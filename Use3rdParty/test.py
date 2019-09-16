from TrainMomentClf import *
import numpy as np
import cv2
from math import *


if __name__ == "__main__":
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    SHOW_IMAGE(image)
    image = cv2.rectangle(image, (25, 25), (75, 75), 255)
    contours, hierachy = ImageProcess(image)

    print(len(contours))

    for j in contours:
        t_moments = cv2.moments(j)
        t_hu_moments = cv2.HuMoments(t_moments)
        print(t_hu_moments)
        try: 
            for k in range(0,7):
                t_hu_moments[k] = -1* copysign(1.0, t_hu_moments[k]) * log10(abs(t_hu_moments[k]) + 1)
        except Exception as e:
            print("yayaya")
            print(e)
            continue
        print(t_hu_moments)
        # print(classifier.predict(t_hu_moments.reshape(1, 7)))
        # print(np.matmul(classifier.coef_, t_hu_moments.reshape(7,)) + classifier.intercept_)
        cv2.drawContours(image, j, -1, (0,255,0), 3)
        SHOW_IMAGE(image)
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