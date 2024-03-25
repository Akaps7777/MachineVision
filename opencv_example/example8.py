import numpy as np, cv2

def calc_histo(image, hsize, ranges=[0,256]):
    hist = np.zeros((hsize, 1), np.float32)
    gap = ranges[1]/hsize

    for i in (image/gap).flat:
        hist[int(i)] +=1
    return hist

image = cv2.imread('../ImageDirectory/pixel.jpg', cv2.IMREAD_GRAYSCALE)
if image is None: raise Exception('Image cannot be read')

hsize,ranges = [32], [0,256]
gap = ranges[1]/hsize[0]
ranges_gap = np.arange(0, ranges[1]+1, gap)
hist1 = calc_histo(image, hsize[0], ranges)
hist2 = cv2.calcHist([image], [0], None, hsize, ranges)
hist3, bins = np.histogram(image, ranges_gap)

print("User 함수: \n", hist1.flatten())
print("OpenCV 함수: \n", hist2.flatten())
print("numpy 함수: \n", hist3)

cv2.imshow("image", image)
cv2.waitKey(0)