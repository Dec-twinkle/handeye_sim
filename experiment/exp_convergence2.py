# _*_ coding:utf-8 _*_
# @time: 2020/10/15 上午9:13
# @author: 张新新
# @email: 1262981714@qq.com
import transforms3d
from auto import auto_handeye_calibration
from auto import init_handeye
from auto import score
from auto import simple_campose
from auto import select_simple_pose
from auto import auto_calibration

from Vrep import vrep_connect
from Vrep import LBR4p
from Vrep import Kinect
from Vrep import Kinect_test

from board import apriltagboard

from handineye import motion
from handineye import rx
from handineye import rz

from method import dual
from method import li
from method import tsai

import cv2
import transforms3d
import numpy as np
import time
if __name__ == '__main__':
    board = apriltagboard.AprilTagBoard("../config/apriltag.yml", "../config/tagId.csv")
    cliend = vrep_connect.getVrep_connect()
    camera = Kinect_test.Camera(cliend,"../config/intrinsic_gt.yml")
    robot = LBR4p.Robot(cliend)

    auto_handeye = auto_handeye_calibration.auto_handeye_calibration(board, robot, camera, "../config/auto_set.yml")
    timestamp = time.time()
    timestruct = time.localtime(timestamp)
    time_str = time.strftime('%m_%d_%H_%M', timestruct)
    auto_handeye.set_select_method(5)
    auto_handeye.run()
        #auto_handeye.save_result("../result/no_local_{}.json".format(time_str))
