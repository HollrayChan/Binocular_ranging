import cv2
import numpy as np
import camera_configs
from scipy import misc

#cv2.namedWindow("depth")
cv2.moveWindow("left", 0, 0)
cv2.moveWindow("right", 640, 0)

cv2.createTrackbar("num", "depth", 2, 10, lambda x: None)
cv2.createTrackbar("blockSize", "depth", 5, 255, lambda x: None)

if __name__ == '__main__':
    frame1 = cv2.imread('./indus_chair/test/recrop_0.bmp')
    frame2 = cv2.imread('./indus_chair/test/resize_0.bmp')
    img1_rectified = cv2.remap(frame1, camera_configs.left_map1, camera_configs.left_map2, cv2.INTER_LINEAR)
    img2_rectified = cv2.remap(frame2, camera_configs.right_map1, camera_configs.right_map2, cv2.INTER_LINEAR)
    path_BM_depth = './depth.png'
 
    # 将图片置为灰度图，为StereoBM作准备
    imgL = cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(img2_rectified, cv2.COLOR_BGR2GRAY)
 
    # 两个trackbar用来调节不同的参数查看效果
    num = cv2.getTrackbarPos("num", "depth")
    blockSize = cv2.getTrackbarPos("blockSize", "depth")
    if blockSize % 2 == 0:
         blockSize += 1
    if blockSize < 5:
         blockSize = 5
 
     # 根据Block Maching方法生成差异图（opencv里也提供了SGBM/Semi-Global Block Matching算法，有兴趣可以试试）
    stereo = cv2.StereoBM_create(numDisparities = 16*num, blockSize = 9)
    
    disparity = stereo.compute(imgL, imgR)
 
    disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # 将图片扩展至3d空间中，其z方向的值则为当前的距离
    threeD = cv2.reprojectImageTo3D(disparity.astype(np.float32)/16., camera_configs.Q)
    print(threeD[0])
# 
#    cv2.imshow("left", img1_rectified)
#    cv2.imshow("right", img2_rectified)
#    cv2.imshow("depth", disp)
# 
#
#    cv2.imwrite(path_BM_left, imgL)
#    cv2.imwrite(path_BM_right, imgR)
    cv2.imwrite(path_BM_depth, disp)
    scipy.misc.imsave('a.jpg', threeD)