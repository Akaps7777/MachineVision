import cv2
import numpy as np
image = cv2.imread('../ImageDirectory/Lenna.png', cv2.IMREAD_COLOR)
if image is None: raise Exception('Image cannot be read')
def light_leak(image, intensity=0.2, color=(0, 150, 255)):
    overlay = np.zeros_like(image, dtype=np.uint8)
    rows, cols, _ = image.shape
    overlay[:rows // 3, 2 * cols // 3:] = color
    cv2.addWeighted(overlay, intensity, image, 1 - intensity, 0, image)
    return image
def dreamy_filter(image):
    blurred_image = cv2.GaussianBlur(image, (21, 21), 0)
    hsv_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    s = cv2.add(s, 50)
    v = cv2.add(v, 50)
    final_hsv = cv2.merge((h, s, v))
    bright_image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    bright_lightleak_image = light_leak(bright_image)

    return bright_lightleak_image


dreamy_image = dreamy_filter(image)
cv2.imshow('dreamy_filter', dreamy_image)
cv2.waitKey(0)

