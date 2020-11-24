# _*_ coding:utf-8 _*_
# @time: 2020/11/23 上午9:21
# @author: 张新新
# @email: 1262981714@qq.com
import os
import cv2
import transforms3d
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from auto import utils
from pylab import mpl
import pandas as pd
import seaborn as sns
colorMap ={
        "no_Local":"black",
        "std":"blue",
        "no_local_std":"gray",
        "rme":"green",
        "no_lock_rme":"red",
        "random":"yellow",
        'ias':"pink"
    }
cali_type = 0
if cali_type==0:
    Hx = "Hcamera2end"
    Hy = "Hobj2base"
else:
    Hx = "Hcamera2base"
    Hy = "Hobj2end"


def getErrorList(fileList,euler_hx_gt,t_hx_gt,euler_hy_gt,t_hy_gt,method=None):
    hx_df_euler = pd.DataFrame(columns=('image number', 'euler_error', 'method'))
    hy_df_euler = pd.DataFrame(columns=('image number', 'euler_error', 'method'))
    hx_df_t = pd.DataFrame(columns=('image number', 't_error', 'method'))
    hy_df_t = pd.DataFrame(columns=('image number', 't_error', 'method'))


    for file in fileList:
        result = utils.json_load(file)
        if not result[0]["simu"]==0.002:
            continue
        for dict in result:
            number = dict["image_number"]
            if number<5:
                continue
            Hx_temp = dict[Hx]
            Hy_temp = dict[Hy]
            euler_hx_temp = transforms3d.euler.mat2euler(Hx_temp[:3, :3])
            t_hx_temp = Hx_temp[:3, 3]
            euler_hx_error_rx = min(abs(euler_hx_temp[0] - euler_hx_gt[0]),
                                    abs(euler_hx_temp[0] - euler_hx_gt[0] + 2 * 3.1415926),
                                    abs(euler_hx_temp[0] - euler_hx_gt[0] - 2 * 3.1415926))
            euler_hx_error_ry = min(abs(euler_hx_temp[1] - euler_hx_gt[1]),
                                    abs(euler_hx_temp[1] - euler_hx_gt[1] + 2 * 3.1415926),
                                    abs(euler_hx_temp[1] - euler_hx_gt[1] - 2 * 3.1415926))
            euler_hx_error_rz = min(abs(euler_hx_temp[2] - euler_hx_gt[2]),
                                    abs(euler_hx_temp[2] - euler_hx_gt[2] + 2 * 3.1415926),
                                    abs(euler_hx_temp[2] - euler_hx_gt[2] - 2 * 3.1415926))
            hx_euler_error = np.mean(np.array([euler_hx_error_rx, euler_hx_error_ry, euler_hx_error_rz]))
            hx_t_error = np.mean(np.abs(t_hx_temp - t_hx_gt))
            hx_df_euler = hx_df_euler.append({'image number':number,"euler_error":hx_euler_error,"method":method},ignore_index=True)
            hx_df_t = hx_df_t.append({'image number':number, 't_error':hx_t_error, 'method':method},ignore_index=True)
            euler_hy_temp = transforms3d.euler.mat2euler(Hy_temp[:3, :3])
            t_hy_temp = Hy_temp[:3, 3]
            euler_hy_error_rx = min(abs(euler_hy_temp[0] - euler_hy_gt[0]),
                                    abs(euler_hy_temp[0] - euler_hy_gt[0] + 2 * 3.1415926),
                                    abs(euler_hy_temp[0] - euler_hy_gt[0] - 2 * 3.1415926))
            euler_hy_error_ry = min(abs(euler_hy_temp[1] - euler_hy_gt[1]),
                                    abs(euler_hy_temp[1] - euler_hy_gt[1] + 2 * 3.1415926),
                                    abs(euler_hy_temp[1] - euler_hy_gt[1] - 2 * 3.1415926))
            euler_hy_error_rz = min(abs(euler_hy_temp[2] - euler_hy_gt[2]),
                                    abs(euler_hy_temp[2] - euler_hy_gt[2] + 2 * 3.1415926),
                                    abs(euler_hy_temp[2] - euler_hy_gt[2] - 2 * 3.1415926))
            hy_euler_error = np.mean(np.array([euler_hy_error_rx, euler_hy_error_ry, euler_hy_error_rz]))
            hy_t_error = np.mean(np.abs(t_hy_temp - t_hy_gt))
            hy_df_euler = hy_df_euler.append({'image number': number, "euler_error": hy_euler_error, "method": method},
                                             ignore_index=True)
            hy_df_t = hy_df_t.append({'image number': number, 't_error': hy_t_error, 'method': method},
                                     ignore_index=True)
    return hx_df_euler,hy_df_euler,hx_df_t,hy_df_t









def draw_error(ax_list,euler_error,t_error,label,x_range):
    rx_error = euler_error[:,0]
    ry_error = euler_error[:,1]
    rz_error = euler_error[:,2]

    tx_error = t_error[:, 0]
    ty_error = t_error[:, 1]
    tz_error = t_error[:, 2]

    ax_list[0].plot(x_range, rx_error,color=colorMap[label], label=label)
    ax_list[0].set_title("rx")
    ax_list[1].plot(x_range, ry_error,color=colorMap[label],  label=label)
    ax_list[1].set_title("ry")
    ax_list[2].plot(x_range, rz_error,color=colorMap[label], label=label)
    ax_list[2].set_title("rz")
    ax_list[3].plot(x_range, tx_error,color=colorMap[label], label=label)
    ax_list[3].set_title("tx")
    ax_list[4].plot(x_range, ty_error,color=colorMap[label], label=label)
    ax_list[4].set_title("ty")
    ax_list[5].plot(x_range, tz_error,color=colorMap[label], label=label)
    ax_list[5].set_title("tz")


def init_set():
    plt.rcParams['figure.figsize'] = (8.0, 4.0)
    # plt.rcParams['image.interpolation'] = 'nearest'  # 设置 interpolation style
    # plt.rcParams['savefig.dpi'] = 1024  # 图片像素
    # plt.rcParams['figure.dpi'] = 300  # 分辨率
    #x_range = np.arange(5, 31, 1)

    fig1, f1_axes = plt.subplots(ncols=2, nrows=1, constrained_layout=True)
    # spec2 = gridspec.GridSpec(ncols=4, nrows=2, figure=fig1)
    ax_list = []

    ax_list.append(f1_axes[0])
    ax_list.append(f1_axes[1])
    ax_list[0].set_ylabel("Absulute Rotation error(rad)")
    ax_list[1].set_ylabel("Absulute Position error(m)")
    # label_ax = fig1.add_subplot(f1_axes[0, 3])
    # plain_ax = fig1.add_subplot(f1_axes[1, 3])
    # plain_ax.get_xaxis().set_visible(False)
    # plain_ax.get_yaxis().set_visible(False)
    # plain_ax.set_frame_on(False)

    # 设置label

    # label_ax.set_frame_on(False)
    # label_ax.get_xaxis().set_visible(False)
    # label_ax.get_yaxis().set_visible(False)
    # lns = []
    # for label in method_list:
    #     lns = lns + plt.plot([], [], color=colorMap[label], label=label)
    # labs = [l.get_label() for l in lns]
    # label_ax.legend(lns, labs, loc="upper right")
    return ax_list



def draw_error_seaborn_ax(ax,error_dic,x_range,fileName=None):
    df = pd.DataFrame(error_dic, index=x_range)
    if not (fileName is None):
        df.to_csv(fileName)
    sns.lineplot(data=df, ax=ax)

# def draw_error_seaborn(ax,error_dic,x_range,file_suffix):
#     error_dic_x = {}
#     error_dic_y = {}
#     error_dic_z = {}
#     for method in error_dic:
#         error_dic_x[method]= error_dic[method][:,0]
#         error_dic_y[method]= error_dic[method][:,1]
#         error_dic_z[method]= error_dic[method][:,2]
#     draw_error_seaborn_ax(ax_list[0],error_dic_x,x_range,file_suffix+"x_error.csv")
#     draw_error_seaborn_ax(ax_list[1],error_dic_y,x_range,file_suffix+"y_error.csv")
#     draw_error_seaborn_ax(ax_list[2],error_dic_z,x_range,file_suffix+"z_error.csv")


if __name__ == '__main__':
    method_list = [ 'std','no_Local','no_local_std', "random",'ias']
    name_list = ['std','no_Local','no_local_std', "random",'ias']
    #文件划分
    method_file_list = []
    sns.set_style("whitegrid")
    for i in range(len(method_list)):
        method_file_list.append([])
    root_dir = "../result/11_33"
    files = os.listdir(root_dir)
    for file in files:
        for i in range(len(method_list)):
            if file.startswith(method_list[i]):
                # if len(method_file_list[i])>20:
                #     continue
                method_file_list[i].append(os.path.join(root_dir,file))
    if cali_type==0:
        fs = cv2.FileStorage("../config/handineye_gt.yml", cv2.FileStorage_READ)
    else:
        fs = cv2.FileStorage("../config/handtoeye_gt.yml", cv2.FileStorage_READ)
    Hx_gt = fs.getNode(Hx).mat()
    Hy_gt = fs.getNode(Hy).mat()
    fs.release()
    euler_hx_gt = transforms3d.euler.mat2euler(Hx_gt[:3, :3])
    t_hx_gt = Hx_gt[:3, 3]
    euler_hy_gt = transforms3d.euler.mat2euler(Hy_gt[:3, :3])
    t_hy_gt = Hy_gt[:3, 3]
    x_range = np.arange(5, 26, 1)
    #error 统计
    method_error = []
    method_std = []
    ## test

    # method_file_list = ["../result/11_8/random_11_12_20_16.json"]
    # euler_hx_error, t_hx_error, euler_hy_error, t_hy_error = \
    #     getErrorList(method_file_list, euler_hx_gt, t_hx_gt, euler_hy_gt, t_hy_gt)
    euler_error_camera2end_mean_dict = {}
    t_error_camera2end_mean_dict = {}
    euler_error_obj2base_mean_dict = {}
    t_error_obj2base_mean_dict = {}
    euler_error_camera2end_std_dict = {}
    t_error_camera2end_std_dict = {}
    euler_error_obj2base_std_dict = {}
    t_error_obj2base_std_dict = {}
    for i in range(len(method_list)):
        euler_hx_error, t_hx_error, euler_hy_error, t_hy_error = \
            getErrorList(method_file_list[i],euler_hx_gt,t_hx_gt,euler_hy_gt,t_hy_gt)

        euler_error_camera2end_mean_dict[name_list[i]]=np.mean(euler_hx_error,axis=0)
        t_error_camera2end_mean_dict[name_list[i]]=np.mean(t_hx_error,axis=0)
        euler_error_obj2base_mean_dict[name_list[i]]=np.mean(euler_hy_error,axis=0)
        t_error_obj2base_mean_dict [name_list[i]]=np.mean(t_hy_error,axis=0)
        euler_error_camera2end_std_dict [name_list[i]]=np.std(euler_hx_error,axis=0)
        t_error_camera2end_std_dict[name_list[i]]=np.std(t_hx_error,axis=0)
        euler_error_obj2base_std_dict[name_list[i]]=np.std(euler_hy_error,axis=0)
        t_error_obj2base_std_dict[name_list[i]]=np.std(t_hy_error,axis=0)
    ax_list = init_set()
    draw_error_seaborn_ax(ax_list[0], euler_error_camera2end_mean_dict, x_range,os.path.join(root_dir, "{0}_mean_r.csv".format(Hx)))
    draw_error_seaborn_ax(ax_list[1], t_error_camera2end_mean_dict, x_range,os.path.join(root_dir, "{0}_mean_t.csv".format(Hx)))
    # for i in range(5):
    #     ax_list[i+1].legend_.remove()
    # ax_list[0].set_ylabel("error(rad)")
    # ax_list[3].set_ylabel("error(m)")
    # ax_list[3].set_xlabel("iter")
    # ax_list[4].set_xlabel("iter")
    # ax_list[5].set_xlabel("iter")
    plt.savefig(os.path.join(root_dir, "E{0}_mean_two.png").format(Hx))
    plt.show()
    ax_list = init_set()
    draw_error_seaborn_ax(ax_list[0], euler_error_camera2end_std_dict, x_range,os.path.join(root_dir, "{0}_std_r.csv".format(Hx)))
    draw_error_seaborn_ax(ax_list[1], t_error_camera2end_std_dict, x_range,os.path.join(root_dir, "{0}_std_t.csv".format(Hx)))
    # for i in range(5):
    #     ax_list[i+1].legend_.remove()
    plt.savefig(os.path.join(root_dir, "E{0}_std_two.png").format(Hx))
    plt.show()
    ax_list = init_set()
    draw_error_seaborn_ax(ax_list[0], euler_error_obj2base_mean_dict, x_range,os.path.join(root_dir, "{0}_mean_r.csv".format(Hy)))
    draw_error_seaborn_ax(ax_list[1], t_error_obj2base_mean_dict, x_range,os.path.join(root_dir, "{0}_mean_t.csv".format(Hy)))
    # for i in range(5):
    #     ax_list[i+1].legend_.remove()
    plt.savefig(os.path.join(root_dir, "E{0}_mean_two.png").format(Hy))
    plt.show()
    ax_list = init_set()
    draw_error_seaborn_ax(ax_list[0], euler_error_obj2base_std_dict, x_range,os.path.join(root_dir, "{0}_std_r.csv".format(Hy)))
    draw_error_seaborn_ax(ax_list[1], t_error_obj2base_std_dict, x_range,os.path.join(root_dir, "{0}_std_r.csv".format(Hy)))
    # for i in range(5):
    #     ax_list[i+1].legend_.remove()
    plt.savefig(os.path.join(root_dir, "E{0}_std_two.png").format(Hy))
    plt.show()