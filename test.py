import cv2
import numpy as np
import camera_configs
 

#cv2.namedWindow("depth")
cv2.moveWindow("left", 0, 0)
cv2.moveWindow("right", 640, 0)

cv2.createTrackbar("num", "depth", 2, 10, lambda x: None)
cv2.createTrackbar("blockSize", "depth", 5, 255, lambda x: None)

if __name__ == '__main__':
    imgL = cv2.imread('./left.jpeg')
    imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    
    imgR = cv2.imread('./right.jpeg')
    imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    path_BM_depth = './depth.png'
 
    # 两个trackbar用来调节不同的参数查看效果
    num = cv2.getTrackbarPos("num", "depth")
    blockSize = cv2.getTrackbarPos("blockSize", "depth")
    if blockSize % 2 == 0:
         blockSize += 1
    if blockSize < 5:
         blockSize = 5
 
     # 根据Block Maching方法生成差异图（opencv里也提供了SGBM/Semi-Global Block Matching算法，有兴趣可以试试）
    stereo = cv2.StereoBM_create(numDisparities = 16, blockSize = 7)
    
    disparity = stereo.compute(imgL, imgR)
 
    disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # 将图片扩展至3d空间中，其z方向的值则为当前的距离
    threeD = cv2.reprojectImageTo3D(disparity.astype(np.float32)/16., camera_configs.Q)

    cv2.imwrite(path_BM_depth, disp)
    
    
    # 图片二值化
#    from PIL import Image
#    img = Image.open('depth.png')
#     
#    # 模式L”为灰色图像，它的每个像素用8个bit表示，0表示黑，255表示白，其他数字表示不同的灰度。
#    Img = img.convert('L')
#    Img.save("test1.jpg")
#     
#    # 自定义灰度界限，大于这个值为黑色，小于这个值为白色
#    threshold = 1
#     
#    table = []
#    for i in range(256):
#        if i < threshold:
#            table.append(0)
#        else:
#            table.append(1)
#     
#    # 图片二值化
#    photo = Img.point(table, '1')
#    photo.save("test2.jpg")
