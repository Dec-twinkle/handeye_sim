# _*_ coding:utf-8 _*_
# @time: 2020/9/28 下午4:06
# @author: 张新新
# @email: 1262981714@qq.com
from Vrep import UR5
from Vrep import Kinect
import numpy as np
import transforms3d

def handineye_init(pose,robot,camera,board):
    robot = UR5.robot(0)
    camera = Kinect.camera(0)
    robot.move(pose)
    rgb_image = camera.get_rgb_image()

def get_Expect_robot_pose(know_campose, know_robot_pose,Thand2eye,expect_campose):
    expect_cam_pose_mat = expect_campose
    q = np.array([])
    t = np.array([])
    for j in range(len(know_robot_pose)):
        temp_robot_pose = np.dot(know_robot_pose[j], np.dot(Thand2eye, np.dot(know_campose[j], np.dot(
            np.linalg.inv(expect_cam_pose_mat), np.linalg.inv(Thand2eye)))))
        q = np.append(q, transforms3d.quaternions.mat2quat(temp_robot_pose[:3, :3]))
        t = np.append(t, temp_robot_pose[:3, 3])
    q = q.reshape([-1, 4])
    t = t.reshape([-1, 3])
    for i in range(1, q.shape[0]):
        if abs(np.linalg.norm(q[0, :] - q[i, :])) > abs(np.linalg.norm(q[0, :] + q[i, :])):
            q[i, :] = -q[i, :]
    mean_q = np.mean(q, 0)
    mean_t = np.mean(t, 0)


    expect_robot_pose = np.append(transforms3d.quaternions.quat2mat(mean_q), np.transpose([mean_t]), 1)
    expect_robot_pose = np.append(expect_robot_pose, np.array([[0, 0, 0, 1]]), 0)
    return expect_robot_pose
