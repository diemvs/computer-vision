import cv2
import matplotlib.pyplot as plt

def convertToBgr(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def show(image, convert_to_bgr=False):
    if(convert_to_bgr):
        image = convertToBgr(image)

    plt.imshow(image)
    plt.axis('off') 
    plt.show()  