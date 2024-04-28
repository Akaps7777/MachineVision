import cv2
import numpy as np
image = cv2.imread('../ImageDirectory/Lenna.png', cv2.IMREAD_COLOR)
if image is None: raise Exception('Image cannot be read')

def cartoon_filter(image):
    img = cv2.imread('../ImageDirectory/Lenna.png', cv2.IMREAD_COLOR)
    Z = img.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    K = 8
    _, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    img_color = res.reshape((img.shape))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    cartoon = cv2.bitwise_and(img_color, img_color, mask=edges)
    return cartoon

cartoon_image = cartoon_filter(image)
cv2.imshow('cartoon_filter', cartoon_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
