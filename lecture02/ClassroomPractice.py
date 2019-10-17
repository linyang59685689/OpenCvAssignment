import cv2
import numpy as np

img = cv2.imread("../res/lena512color.tiff")
# 一阶x方向导数的kernel
kernel1 = np.float32([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
kernel2 = np.float32([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

# 一阶y方向导数的kernel
kernel3 = np.float32([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
kernel4 = np.float32([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

# 二阶导数
kernel5 = np.float32([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
kernel6 = np.float32([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
kernel7 = np.float32([[1, 0, 1], [0, -4, 0], [1, 0, 1]])
kernel8 = np.float32([[-1, 0, -1], [0, 4, 0], [-1, 0, -1]])
kernel9 = np.float32([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
kernel10 = np.float32([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
kernel11 = np.float32([[-1, -2, -1], [-2, 12, -2], [-1, -2, -1]])

# 自己构造了一个高斯kernel
GaussianKernel = np.float32([[1 / 16, 2 / 16, 1 / 16], [2 / 16, 4 / 16, 2 / 16], [1 / 16, 2 / 16, 1 / 16]])
print(np.var(GaussianKernel))
img_float=np.float32(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))
# img_harris = cv2.cornerHarris(img_float, 2, 3, 0.03)
# img_harris=cv2.dilate(img_harris,None)
# threshold=np.max(img_harris)*0.03
# print(np.max(img_harris))
# img[img_harris>threshold]=[0,0,255]
# img2 = cv2.GaussianBlur(img, (3, 3), 7)
# img3 = cv2.filter2D(img, -1, kernel=GaussianKernel)
# sift = cv2.xfeatures2d.SIFT_create()

cv2.imshow("img2", img)
# cv2.imshow("img3", img3)
cv2.waitKey(0)
