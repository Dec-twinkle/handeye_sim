#-*- coding:utf-8 -*-
# @time: 
# @author:张新新
# @email: 1262981714@qq.com
import cv2
import transforms3d
import numpy as np
from handineye import motion
from handineye import rx
from handineye import rz


import math
from method import tsai
from method import dual
#import handtoeye

class auto_handeye_calibration(object):
    def __init__(self,board,robot,camera,config):
        '''
        初始化，需要指定标定板，机器臂，相机，以及初始化文件
        :param board: 标定板
        :param robot: 机器臂
        :param camera: 相机
        :param config: 配置文件
        '''
        self.board = board
        self.robot = robot
        self.camera = camera
        fs = cv2.FileStorage(config, cv2.FILE_STORAGE_READ)
        self.minZ = fs.getNode("min_Z").real()
        self.maxZ = fs.getNode("max_Z").real()
        self.inter_z = fs.getNode("inter_z").real()
        self.inter_xy = fs.getNode("inter_xy").real()
        self.optic_angle = fs.getNode("optic_angle").real()
        self.cali_type = int(fs.getNode("cali_type").real())  #calibration type： handineye 0 handtoeye 1
        self.picture_number = int(fs.getNode("picture_number").real())
        if self.cali_type==0:
            from handineye import motion
            from handineye import rx
            from handineye import rz
        else:
            from handtoeye import motion
            from handtoeye import rx
            from handtoeye import rz

        init_pose = fs.getNode("init_robot_pose").mat()

        q = init_pose[:4].flatten()
        pose_r = transforms3d.quaternions.quat2mat(q)

        self.init_robot_pose = np.identity(4)

        self.init_robot_pose[:3, :3] = pose_r[:, :]
        self.init_robot_pose[0, 3] = init_pose[4]
        self.init_robot_pose[1, 3] = init_pose[5]
        self.init_robot_pose[2, 3] = init_pose[6]
        fs.release()

        self.next_step_method = 0

    def init_handeye(self):
        '''
        通过旋转机械臂来初始化手眼标定
        :return:
        '''
        flag = self.robot.move(self.init_robot_pose)
        assert flag, "cannot reach init pose"
        rgb_image = self.camera.get_rgb_image()
        flag, objpoint, imgpoint = self.board.getObjImgPointList(rgb_image, verbose=0)
        assert flag, "robot init pose cannot see board"
        self.objpoint_list = []
        self.imgpoint_list = []
        self.Hend2base = []
        self.Hobj2camera = []
        self.image = []
        self.result = []
        self.objpoint_list.append(objpoint)
        self.imgpoint_list.append(imgpoint)
        self.Hend2base.append(self.init_robot_pose)
        camerapose = self.board.extrinsic(imgpoint, objpoint, self.camera.intrinsic, self.camera.dist)
        self.Hobj2camera.append(camerapose)

        ax, ay, az = transforms3d.euler.mat2euler(self.init_robot_pose[:3, :3], 'sxyz')
        euler = [ax, ay, az]
        for i in [0,1]:
            for j in [-1,1]:
                objpoint_temp = None
                imgpoint_temp = None
                robot_pose_temp = None
                euler_temp = euler.copy()
                while (True):
                    euler_temp[i] += j*math.pi / 36
                    pose_r = transforms3d.euler.euler2mat(euler_temp[0], euler_temp[1], euler_temp[2], 'sxyz')
                    robot_pose = self.init_robot_pose.copy()
                    robot_pose[:3, :3] = pose_r[:, :]
                    flag1 = self.robot.move(robot_pose)
                    rgb_image = self.camera.get_rgb_image()
                    flag, objpoint, imgpoint = self.board.getObjImgPointList(rgb_image)
                    if flag and flag1:
                        objpoint_temp = objpoint.copy()
                        imgpoint_temp = imgpoint.copy()
                        robot_pose_temp = robot_pose.copy()
                        image_temp = rgb_image.copy()
                    else:
                        if not objpoint_temp is None:
                            camerapose = self.board.extrinsic(imgpoint_temp, imgpoint_temp, self.camera.intrinsic, self.camera.dist)
                            self.Hobj2camera.append(camerapose)
                            self.objpoint_list.append(objpoint_temp)
                            self.imgpoint_list.append(imgpoint_temp)
                            self.Hend2base.append(robot_pose_temp)
                            self.image.append(image_temp)
                        break
        assert len(self.Hend2base) > 3, "cannot find enough initial data"

        A, B = motion.motion_axxb(self.Hend2base, self.Hobj2camera)
        Hx = dual.calibration(A, B)
        Hx = rx.refine(Hx, self.Hend2base, self.Hobj2camera,
                                           self.board.GetBoardAllPoints())
        q = np.array([])
        t = np.array([])
        for i in range(len(self.Hobj2camera)):
            if self.cali_type==0:
                Hy = np.dot(self.Hend2base[i], np.dot(Hx, self.Hobj2camera[i]))
            else:
                Hy = np.dot(np.linalg.inv(self.Hend2base[i]),np.dot(Hx, self.Hobj2camera[i]))
            q_temp = transforms3d.quaternions.mat2quat(Hy[:3, :3])

            if i == 0:
                q0 = q_temp.copy()
            else:
                if np.linalg.norm(q0 - q_temp) > np.linalg.norm(q0 + q_temp):
                    q_temp = -q_temp
            q = np.append(q, q_temp)
            t = np.append(t, Hy[:3, 3])
        q = q.reshape([-1, 4])
        t = t.reshape([-1, 3])
        q_mean = np.mean(q, 0)
        t_mean = np.mean(t, 0)
        q = q_mean / np.linalg.norm(q)
        Hy_r = transforms3d.quaternions.quat2mat(q)
        Hy = np.identity(4)
        Hy[:3, :3] = Hy_r[:, :]
        Hy[:3, 3] = t_mean[:]
        rme = rz.proj_error(Hx,Hy,self.Hend2base,self.Hobj2camera,self.board.GetBoardAllPoints)
        if self.cali_type==0:
            self.result.append({"image_number": len(self.image), "Hcamera2end":Hx,"Hobj2base":Hy,
                                 "mean_rme":np.mean(np.abs(rme)),"max_rme":np.max(np.abs(rme))})
        else:
            self.result.append({"image_number": len(self.image), "Hcamera2base": Hx, "Hobj2end": Hy,
                                "mean_rme": np.mean(np.abs(rme)), "max_rme": np.max(np.abs(rme))})

    def










