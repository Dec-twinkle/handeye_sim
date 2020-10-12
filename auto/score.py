import numpy as np
import transforms3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math


def score(objpoint, Trobot2end, Tend2eye, Teye2grid, Trobot2grid):
    proj = np.dot(np.dot(np.dot(np.dot(np.linalg.inv(Trobot2grid), Trobot2end), Tend2eye), Teye2grid), objpoint)
    error = proj[:3, :] - objpoint[:3, :]
    return np.mean(np.abs(np.linalg.norm(error, axis=0)))
def no_local_term(know_robot_pose,expect_robot_pose):
    q0 = transforms3d.quaternions.mat2quat(expect_robot_pose[:3,:3])
    t0 = expect_robot_pose[:3,3]
    min_score = 0
    for j in range(len(know_robot_pose)):
        q = transforms3d.quaternions.mat2quat(know_robot_pose[j][:3,:3])
        if np.linalg.norm(q-q0)>np.linalg.norm(q+q0):
            q = -q
        t = know_robot_pose[j][:3,3]
        q_dis = np.linalg.norm(q-q0)
        t_dis = np.linalg.norm(t-t0)
        score = -(math.pow(math.e,-abs(q_dis))+1 * math.pow(math.e,-abs(t_dis)))
        if score<min_score:
            min_score=score
    return min_score


def score_expect_robot_pose(know_robot_pose, know_cam_Extrinsic, expect_cam_pose, Thand2eye):
    expect_cam_pose_mat = expect_cam_pose
    q = np.array([])
    t = np.array([])
    for j in range(len(know_robot_pose)):
        temp_robot_pose = np.dot(know_robot_pose[j], np.dot(Thand2eye, np.dot(know_cam_Extrinsic[j], np.dot(
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
    std_q = np.std(q, axis=0)
    std_t = np.std(t, axis=0)

    expect_robot_pose = np.append(transforms3d.quaternions.quat2mat(mean_q), np.transpose([mean_t]), 1)
    expect_robot_pose = np.append(expect_robot_pose, np.array([[0, 0, 0, 1]]), 0)
    score = np.mean(std_q) + np.mean(1 * std_t) +no_local_term(know_robot_pose,expect_robot_pose)

    return score, expect_robot_pose


def score_reprojection(know_robot_pose, know_cam_Extrinsic, expect_cam_pose, Thand2eye, Tworld2base, grid):
    expect_cam_pose_mat = expect_cam_pose
    q = np.array([])
    t = np.array([])
    for j in range(len(know_robot_pose)):
        temp_robot_pose = np.dot(know_robot_pose[j], np.dot(Thand2eye, np.dot(know_cam_Extrinsic[j], np.dot(
            np.linalg.inv(expect_cam_pose_mat), np.linalg.inv(Thand2eye)))))
        q = np.append(q, transforms3d.quaternions.mat2quat(temp_robot_pose[:3, :3]))
        t = np.append(t, temp_robot_pose[:3, 3])
    q = q.reshape([-1, 4])
    t = t.reshape([-1, 3])
    for i in range(1,q.shape[0]):
        if abs(np.linalg.norm(q[0, :] - q[i, :])) > abs(np.linalg.norm(q[0, :] + q[i, :])):
            q[i, :] = -q[i, :]
    mean_q = np.mean(q, 0)
    mean_t = np.mean(t, 0)
    Tbase2end = np.append(np.append(transforms3d.quaternions.quat2mat(mean_q), np.transpose([mean_t]), 1),
                          np.array([[0, 0, 0, 1]]), 0)
    if grid.shape[1] == 2:
        n = grid.shape[0]
        grid = np.append(grid, np.zeros([n, 1]), 1)
        grid = np.append(grid, np.ones([n, 1]), 1)
    grid = grid.T
    trans = np.dot(np.linalg.inv(Tworld2base), np.dot(Tbase2end, np.dot(Thand2eye, expect_cam_pose_mat)))
    proj = np.dot(trans, grid)
    error = proj - grid

    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.plot(point[0,:],point[1,:],point[2,:],c='r')
    # ax.plot(pointbase[0,:],pointbase[1,:],pointbase[2,:],c='b')
    # ax.plot(pointCam[0,:],pointCam[1,:],pointCam[2,:],c='y')
    # ax.plot(pointEnd[0,:],pointEnd[1,:],pointEnd[2,:],c='g')
    # ax.set_zlim3d(-1,1)
    # plt.xlim(-1,1)
    # plt.ylim(-1,1)
    #
    #
    # plt.show()
    return np.mean(np.abs(error)), Tbase2end
