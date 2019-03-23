# 从人脸图像文件中提取人脸特征存入 CSV
# Get features from images and save into features_all.csv

# Author:   coneypo
# Blog:     http://www.cnblogs.com/AdaminXie
# GitHub:   https://github.com/coneypo/Dlib_face_recognition_from_camera
# Mail:     coneypo@foxmail.com

# Created at 2018-05-11
# Updated at 2019-02-25

# 增加录入多张人脸到 CSV 的功能

# return_128d_features()          获取某张图像的 128D 特征
# write_into_csv()                获取某个路径下所有图像的特征，并写入 CSV
# compute_the_mean()              从 CSV 中读取 128D 特征，并计算特征均值

import cv2
import os
import dlib
from skimage import io
import csv
import numpy as np
import pandas as pd

# 要读取人脸图像文件的路径
path_photos_from_camera = "data/data_faces_from_camera/"
# 储存人脸特征 csv 的路径
path_csv_from_photos = "data/data_csvs_from_camera/"

# Dlib 正向人脸检测器
detector = dlib.get_frontal_face_detector()

# Dlib 人脸预测器
predictor = dlib.shape_predictor("data/data_dlib/shape_predictor_5_face_landmarks.dat")

# Dlib 人脸识别模型
# Face recognition model, the object maps human faces into 128D vectors
facerec = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")


# 返回单张图像的 128D 特征
def return_128d_features(path_img):
    img = io.imread(path_img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector(img_gray, 1)

    print("%-40s %-20s" % ("检测到人脸的图像 / image with faces detected:", path_img), '\n')

    # 因为有可能截下来的人脸再去检测，检测不出来人脸了
    # 所以要确保是 检测到人脸的人脸图像 拿去算特征
    if len(faces) != 0:
        shape = predictor(img_gray, faces[0])
        face_descriptor = facerec.compute_face_descriptor(img_gray, shape)
    else:
        face_descriptor = 0
        print("no face")

    # print(face_descriptor)
    return face_descriptor


# 将文件夹中照片特征提取出来, 写入 CSV
#   path_faces_personX:     图像文件夹的路径
#   path_csv_from_photos:   要生成的 CSV 路径

def write_into_csv(path_faces_personX, path_csv_from_photos):
    photos_list = os.listdir(path_faces_personX)
    with open(path_csv_from_photos, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if photos_list:
            for i in range(len(photos_list)):
                # 调用return_128d_features()得到128d特征
                print("%-40s %-20s" % ("正在读的人脸图像 / image to read:", path_faces_personX + "/" + photos_list[i]))
                features_128d = return_128d_features(path_faces_personX + "/" + photos_list[i])
                #  print(features_128d)
                # 遇到没有检测出人脸的图片跳过
                if features_128d == 0:
                    i += 1
                else:
                    writer.writerow(features_128d)
        else:
            print("文件夹内图像文件为空 / Warning: Empty photos in " + path_faces_personX + '/', '\n')
            writer.writerow("")


# 读取某人所有的人脸图像的数据，写入 person_X.csv
faces = os.listdir(path_photos_from_camera)
faces.sort()
for person in faces:
    print("##### " + person + " #####")
    print(path_csv_from_photos + person + ".csv")
    write_into_csv(path_photos_from_camera + person, path_csv_from_photos + person + ".csv")
print('\n')


# 从 CSV 中读取数据，计算 128D 特征的均值
def compute_the_mean(path_csv_from_photos):
    column_names = []

    # 128D 特征
    for feature_num in range(128):
        column_names.append("features_" + str(feature_num + 1))

    # 利用 pandas 读取 csv
    rd = pd.read_csv(path_csv_from_photos, names=column_names)

    if rd.size != 0:
        # 存放 128D 特征的均值
        feature_mean_list = []

        for feature_num in range(128):
            tmp_arr = rd["features_" + str(feature_num + 1)]
            tmp_arr = np.array(tmp_arr)
            # 计算某一个特征的均值
            tmp_mean = np.mean(tmp_arr)
            feature_mean_list.append(tmp_mean)
    else:
        feature_mean_list = []
    return feature_mean_list


# 存放所有特征均值的 CSV 的路径
path_csv_from_photos_feature_all = "data/features_all.csv"

# 存放人脸特征的 CSV 的路径
path_csv_from_photos = "data/data_csvs_from_camera/"

with open(path_csv_from_photos_feature_all, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    csv_rd = os.listdir(path_csv_from_photos)
    csv_rd.sort()
    print("##### 得到的特征均值 / The generated average values of features stored in: #####")
    for i in range(len(csv_rd)):
        feature_mean_list = compute_the_mean(path_csv_from_photos + csv_rd[i])
        print(path_csv_from_photos + csv_rd[i])
        writer.writerow(feature_mean_list)
