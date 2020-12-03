#-*- coding:utf-8 -*-
# @time: 
# @author:张新新
# @email: 1262981714@qq.com
import transforms3d
from auto import auto_handeye_real_world
from auto import auto_handeye_calibration
from auto import utils

import random
from camera import kinect
from robot import aubo

from board import apriltagboard


import time
if __name__ == '__main__':
    board = apriltagboard.AprilTagBoard("../config/apriltag.yml", "../config/tagId.csv")
    camera = kinect.Kinect(port=1024)
    robot = aubo.robot(port=1025)
    auto_handeye = auto_handeye_real_world.auto_handeye_calibration(board, robot, camera,
                                                                            "../config/auto_set_to.yml")
    timestamp = time.time()
    timestruct = time.localtime(timestamp)
    time_str = time.strftime('%m_%d_%H_%M', timestruct)
    auto_handeye.set_select_method(5)
    auto_handeye.init_handeye()
    auto_handeye.handeye_cali()
    auto_handeye.run()
    utils.json_save(random.sample(auto_handeye.Hend2base,10), "../config/init_robot_pose_handtoeye.json")