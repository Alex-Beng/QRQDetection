import cv2
import datetime
import numpy as np

def SHOW_IMAGE(image):
    now = datetime.datetime.now()
    now = str(now)
    cv2.imshow(now, image)
    cv2.waitKey()
    cv2.destroyWindow(now)


# input int8 3-channel image
def ConvertChannelTo(image, chn_str):
    if chn_str == 'L':
        cv2.cvtColor(image, t_image, cv2.COLOR_BGR2HLS)
        t_channels = cv2.split(t_image)
        return t_channels[1]
    elif chn_str == 'GRAY':
        cv2.cvtColor(image, t_image, cv2.COLOR_BGR2GRAY)
        return t_image
def ThreTo(image, min_thre, max_thre):
    return image>min_thre&image<max_thr

# just fuck the code !!
if __name__ == "__main__":
    for i in range(1, 6):
        image_path = "../Pics/%03d.jpg"%int(i)
        image = cv2.imread(image_path)

        # get L channel
        t_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        t_cs = cv2.split(t_image)
        L_chn = t_cs[1]

        SHOW_IMAGE(L_chn)

        # do the sobel 
        x = cv2.Sobel(L_chn, cv2.CV_16S, 1, 0)
        y = cv2.Sobel(L_chn, cv2.CV_16S, 0, 1)
        grad_x = cv2.convertScaleAbs(x)
        grad_y = cv2.convertScaleAbs(y)

        SHOW_IMAGE(grad_x)
        SHOW_IMAGE(grad_y)

        grad = cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)

        SHOW_IMAGE(grad)

        _, grad_thre = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
        SHOW_IMAGE(grad_thre)
        

    pass