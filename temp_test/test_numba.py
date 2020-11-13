# _*_ coding:utf-8 _*_
# @time: 2020/11/13 下午8:49
# @author: 张新新
# @email: 1262981714@qq.com

from numba import jit
import numba as nb
import numpy as np
import random
import math
import transforms3d
import time
@jit(nopython=True)
def score_std_numba(expect_camera_list,Hend2base,Hobj2camera,Hx):
    expect_robot_pose = np.zeros((expect_camera_list.shape[0],4,4),dtype=nb.float32)
    score = np.zeros((expect_camera_list.shape[0],Hend2base.shape[0],4,4),dtype=nb.float32)
    expect_robot_q0 = np.zeros((Hend2base.shape[0],1),dtype=nb.float32)
    expect_robot_q1 = np.zeros((Hend2base.shape[0],1),dtype=nb.float32)
    expect_robot_q2 = np.zeros((Hend2base.shape[0],1),dtype=nb.float32)
    expect_robot_q3 = np.zeros((Hend2base.shape[0],1),dtype=nb.float32)
    expect_robot_t0 = np.zeros((Hend2base.shape[0],1),dtype=nb.float32)
    expect_robot_t1 = np.zeros((Hend2base.shape[0],1),dtype=nb.float32)
    expect_robot_t2 = np.zeros((Hend2base.shape[0],1),dtype=nb.float32)
    for i in range(expect_camera_list.shape[0]):
        for j in range(Hend2base.shape[0]):
            expect_robot_pose= np.dot(Hend2base[j], np.dot(Hx, np.dot(Hobj2camera[j], np.dot(
                    np.linalg.inv(expect_camera_list[i]), np.linalg.inv(Hx)))))
            R = expect_robot_pose[:3,:3]
            if 1 + R[0, 0] + R[1, 1] + R[2, 2] > 0:
                q0 = 0.5 * math.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2])
                expect_robot_q0[j, 0] = q0
                expect_robot_q1[j, 0] = (R[2, 1] - R[1, 2]) / (4 * q0)
                expect_robot_q2[j, 2] = (R[0, 2] - R[2, 0]) / (4 * q0)
                expect_robot_q3[j, 3] = (R[1, 0] - R[0, 1]) / (4 * q0)
            else:
                if max(R[0, 0], R[1, 1], R[2, 2]) == R[0, 0]:
                    t = math.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
                    expect_robot_q0[j, 0] = (R[2, 1] - R[1, 2]) / t
                    expect_robot_q1[j, 0] = t / 4
                    expect_robot_q2[j, 0] = (R[0, 2] + R[2, 0]) / t
                    expect_robot_q3[j, 0] = (R[0, 1] + R[1, 0]) / t
                elif max(R[0, 0], R[1, 1], R[2, 2]) == R[1, 1]:
                    t = math.sqrt(1 - R[0, 0] + R[1, 1] - R[2, 2])
                    expect_robot_q0[j, 0] = (R[0, 2] - R[2, 0]) / t
                    expect_robot_q1[j, 0] = (R[0, 1] + R[1, 0]) / t
                    expect_robot_q2[j, 0] = t / 4
                    expect_robot_q3[j, 0] = (R[1, 2] + R[2, 1]) / t
                else:
                    t = math.sqrt(1 - R[0, 0] - R[1, 1] + R[2, 2])
                    expect_robot_q0[j, 0] = (R[1, 0] - R[0, 1]) / t
                    expect_robot_q1[j, 0] = (R[0, 2] + R[2, 0]) / t
                    expect_robot_q2[j, 0] = (R[1, 2] - R[2, 1]) / t
                    expect_robot_q3[j, 0] = t / 4
            expect_robot_t0[j,0]=expect_robot_pose[0,3]
            expect_robot_t1[j,0]=expect_robot_pose[1,3]
            expect_robot_t2[j,0]=expect_robot_pose[2,3]
        expect_robot_q0_std = np.std(expect_robot_q0)
        expect_robot_q1_std = np.std(expect_robot_q1)
        expect_robot_q2_std = np.std(expect_robot_q2)
        expect_robot_q3_std = np.std(expect_robot_q3)
        expect_robot_t0_std = np.std(expect_robot_t0)
        expect_robot_t1_std = np.std(expect_robot_t1)
        expect_robot_t2_std = np.std(expect_robot_t2)


        #std_euler = np.std(expect_robot_euler,axis=0)
    #return expect_robot_pose




def score_std(expect_camera_list,Hend2base,Hobj2camera,Hx):
    expect_robot_pose = np.empty([len(expect_camera_list),len(Hend2base),4,4])
    for i in range(len(expect_camera_list)):
        for j in range(len(Hend2base)):
            expect_robot_pose[i,j,:,:] = np.dot(Hend2base[j], np.dot(Hx, np.dot(Hobj2camera[j], np.dot(
                    np.linalg.inv(expect_camera_list[i]), np.linalg.inv(Hx)))))
    return expect_robot_pose

if __name__ == '__main__':
    expect_camera_list = []
    for i in range(10000):
        ax = random.random()*math.pi
        ay = random.random()*math.pi
        az = random.random()*math.pi
        r = transforms3d.euler.euler2mat(ax,ay,az)
        RT = np.identity(4)
        RT[:3,:3]=r[:,:]
        RT[0,3] = random.random()
        RT[1,3] = random.random()
        RT[2,3] = random.random()
        expect_camera_list.append(RT)
    Hend2base = []
    for i in range(10):
        ax = random.random()*math.pi
        ay = random.random()*math.pi
        az = random.random()*math.pi
        r = transforms3d.euler.euler2mat(ax,ay,az)
        RT = np.identity(4)
        RT[:3,:3]=r[:,:]
        RT[0,3] = random.random()
        RT[1,3] = random.random()
        RT[2,3] = random.random()
        Hend2base.append(RT)
    Hobj2camera = []
    for i in range(10):
        ax = random.random()*math.pi
        ay = random.random()*math.pi
        az = random.random()*math.pi
        r = transforms3d.euler.euler2mat(ax,ay,az)
        RT = np.identity(4)
        RT[:3,:3]=r[:,:]
        RT[0,3] = random.random()
        RT[1,3] = random.random()
        RT[2,3] = random.random()
        Hobj2camera.append(RT)
    ax = random.random() * math.pi
    ay = random.random() * math.pi
    az = random.random() * math.pi
    r = transforms3d.euler.euler2mat(ax, ay, az)
    Hx = np.identity(4)
    Hx[:3, :3] = r[:, :]
    Hx[0, 3] = random.random()
    Hx[1, 3] = random.random()
    Hx[2, 3] = random.random()

    time1 = time.time()
    #x = score_std(expect_camera_list,Hend2base,Hobj2camera,Hx)
    time2 = time.time()
    print("no numba time:",time2-time1)
    expect_robot_pose = np.empty([len(expect_camera_list), len(Hend2base), 4, 4])
    expect_cameras = np.array(expect_camera_list)
    Hend2bases = np.array(Hend2base)
    Hobj2cameras = np.array(Hobj2camera)
    time3 = time.time()
    score_std_numba(expect_cameras,Hend2bases,Hobj2cameras,Hx)
    time4 = time.time()
    print("numba time:", time4 - time3)





