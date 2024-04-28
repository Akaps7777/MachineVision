import cv2
import numpy as np
image = cv2.imread('../ImageDirectory/Lenna.png', cv2.IMREAD_COLOR)
if image is None: raise Exception('Image cannot be read')

def apply_vintage_effect(image):

    sepia_kernel = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])

    sepia_image = cv2.transform(image, sepia_kernel)
    rows, cols, _ = image.shape
    gaussian_noise = np.random.randn(rows, cols, 3) * 25
    noisy_image = sepia_image + gaussian_noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    mask = np.zeros_like(image)
    rows, cols, _ = mask.shape
    center_x, center_y = int(cols / 2), int(rows / 2)
    cv2.circle(mask, (center_x, center_y), min(center_x, center_y), (255, 255, 255), thickness=-1)
    blurred_mask = cv2.GaussianBlur(mask, (21, 21), sigmaX=0, sigmaY=0)
    vintage_effect = cv2.addWeighted(noisy_image, 1, blurred_mask, -0.5, 0)
    return vintage_effect

vintage_image = apply_vintage_effect(image)

cv2.imshow('Vintage_filter', vintage_image)
cv2.waitKey(0)
