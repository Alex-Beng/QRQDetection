from TrainMomentClf import *
import numpy as np
import cv2


if __name__ == "__main__":
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    SHOW_IMAGE(image)
    image = cv2.rectangle(image, (25, 25), (75, 75), 255)
    contours = ImageProcess(image)

    # SHOW_IMAGE(image)
    image = np.zeros((100, 100, 3), dtype=np.int8)
    MyDrawContours(image, contours[0], 100)
    center = MyContourCenter(contours[0])
    print(center.shape)
    image[int(center[0, 0]), int(center[0, 1])] += 100

    SHOW_IMAGE(image)
    MyDrawContours(image, contours[0], -100)
    SHOW_IMAGE(image)