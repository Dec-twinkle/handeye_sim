# _*_ coding:utf-8 _*_
# @time: 2020/9/30 上午8:46
# @author: 张新新
# @email: 1262981714@qq.com
import numpy as np
import math
import transforms3d
import cv2

euler_connection = [0,-math.pi/2,0]
euler_rgb = [-math.pi/2,0,math.pi/2]
position_connection = np.array([-1.8495e-1,-1.6655e-7,+9.8655e-1])
position_rgb = np.array([-2.1734e-1,+2.9250e-2,+9.7245e-1])
rotation_connection = transforms3d.euler.euler2mat(euler_connection[0],euler_connection[1],euler_connection[2],'rxyz')
rotation_rgb = transforms3d.euler.euler2mat(euler_rgb[0],euler_rgb[1],euler_rgb[2],'rxyz')
translation_connection = np.append(np.append(rotation_connection,np.transpose([position_connection]),1),np.array([[0,0,0,1]]),0)
translation_rgb = np.append(np.append(rotation_rgb,np.transpose([position_rgb]),1),np.array([[0,0,0,1]]),0)

handeye_gt = np.dot(np.linalg.inv(translation_connection),translation_rgb)
translation_pic = np.array([[-1,0,0,0],
                            [0,1,0,0],
                            [0,0,-1,0],
                            [0,0,0,1]])
handeye_gt = np.dot(translation_pic,handeye_gt)
print("handeye_gt:",handeye_gt)

handeye_gt = np.array([[1,0,0,-0.01410],
                      [0,0,1,0.02925],
                      [0,-1,0,0.03439],
                      [0,0,0,1]])
Hobj2base = np.array([[1,0,0,-0.735],
                      [0,-1,0,0.09],
                      [0,0,-1,0],
                      [0,0,0,1]])


fs = cv2.FileStorage("../config/handineye_gt.yml",cv2.FileStorage_WRITE)
fs.write("Hcamera2end",handeye_gt)
fs.write("Hobj2base",Hobj2base)
fs.release()
