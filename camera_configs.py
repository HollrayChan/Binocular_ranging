import cv2
import numpy as np
 
#左摄像头参数
left_camera_matrix = np.array([[1.475383975722699e+03, 0, 5.490121807230344e+02],
                               [0, 1.481783520961973e+03, -79.462861815048580],
                               [0, 0, 1]])
left_distortion = np.array([[-0.273521825367298, 0.114432926561229, 0, 0, 0]])
 
#右摄像头参数
right_camera_matrix = np.array([[1.419009769669427e+03, 0, 7.835876327811428e+02],
                                [0, 1.411141145704515e+03, -91.617265322940700],
                                [0, 0, 1]])
right_distortion = np.array([[-0.214711298211341, 0.044837155207188, 0, 0, 0]])
 
R = np.array([[0.993368264230316, -0.037004323956130, -0.108858493598798],      # 旋转关系向量
              [0.040818290816555, 0.998620884102595, 0.033018130912664],
              [0.107486551506990, -0.037242581042410, 0.993508747521850]])
T = np.array([-96.349983863951220, -10.922027827511574, -32.995987135842030])      # 平移关系向量
 
size = (1280, 720) # 图像尺寸
 
# 进行立体更正
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,\
                                                                  right_camera_matrix, right_distortion, size, R,T)
# 计算更正map
left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)