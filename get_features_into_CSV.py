# created at 2018-05-11
# updated at 2018-09-09
# 增加录入多张人脸到CSV的功能

# Author:   coneypo
# Blog:     http://www.cnblogs.com/AdaminXie
# GitHub:   https://github.com/coneypo/Dlib_face_recognition_from_camera

#   return_128d_features()          获取某张图像的128d特征
#   write_into_csv()                将某个文件夹中的图像读取特征并写入csv
#   compute_the_mean()              从csv中读取128d特征，并计算特征均值


import cv2
import os
import dlib
from skimage import io
import csv
import numpy as np
import pandas as pd

path_faces_rd = "data/faces_from_camera/"
path_csv = "data/csvs_from_camera/"

# detector to find the faces
detector = dlib.get_frontal_face_detector()

# shape predictor to find the face landmarks
predictor = dlib.shape_predictor("data/dlib_dat/shape_predictor_5_face_landmarks.dat")

# face recognition model, the object maps human faces into 128D vectors
facerec = dlib.face_recognition_model_v1("data/dlib_dat/dlib_face_recognition_resnet_model_v1.dat")


# 返回单张图像的128D特征
def return_128d_features(path_img):
    img = io.imread(path_img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dets = detector(img_gray, 1)

    print("检测的人脸图像：", path_img, "\n")

    # 因为有可能截下来的人脸再去检测，检测不出来人脸了
    # 所以要确保是 检测到人脸的人脸图像 拿去算特征
    if len(dets) != 0:
        shape = predictor(img_gray, dets[0])
        face_descriptor = facerec.compute_face_descriptor(img_gray, shape)
    else:
        face_descriptor = 0
        print("no face")

    # print(face_descriptor)
    return face_descriptor


# 将文件夹中照片特征提取出来，写入csv
# 输入input:
#   path_faces_personX:     图像文件夹的路径
#   path_csv:               要生成的csv路径

def write_into_csv(path_faces_personX, path_csv):
    dir_pics = os.listdir(path_faces_personX)
    with open(path_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for i in range(len(dir_pics)):
            # 调用return_128d_features()得到128d特征
            print("正在读的人脸图像：", path_faces_personX + "/" + dir_pics[i])
            features_128d = return_128d_features(path_faces_personX + "/" + dir_pics[i])
            #  print(features_128d)
            # 遇到没有检测出人脸的图片跳过
            if features_128d == 0:
                i += 1
            else:
                writer.writerow(features_128d)


# 读取 某人 所有的人脸图像的数据，写入 person_X.csv
faces = os.listdir(path_faces_rd)
for person in faces:
    print(path_csv + person + ".csv")
    write_into_csv(path_faces_rd + person, path_csv + person + ".csv")


# 从csv中读取数据，计算128d特征的均值
def compute_the_mean(path_csv_rd):
    column_names = []

    # 128列特征
    for i in range(128):
        column_names.append("features_" + str(i + 1))

    # 利用pandas读取csv
    rd = pd.read_csv(path_csv_rd, names=column_names)

    # 存放128维特征的均值
    feature_mean = []

    for i in range(128):
        tmp_arr = rd["features_" + str(i + 1)]
        tmp_arr = np.array(tmp_arr)

        # 计算某一个特征的均值
        tmp_mean = np.mean(tmp_arr)
        feature_mean.append(tmp_mean)
    return feature_mean


# 存放所有特征均值的 CSV 的路径
path_csv_feature_all = "data/features_all.csv"
# 存放人脸特征的csv的路径
path_csv_rd = "data/csvs_from_camera/"

with open(path_csv_feature_all, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    csv_rd = os.listdir(path_csv_rd)
    print("特征均值: ")
    for i in range(len(csv_rd)):
        feature_mean = compute_the_mean(path_csv_rd + csv_rd[i])
        # print(feature_mean)
        print(path_csv_rd + csv_rd[i])
        writer.writerow(feature_mean)
