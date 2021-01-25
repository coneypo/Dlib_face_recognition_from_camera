# Copyright (C) 2020 coneypo
# SPDX-License-Identifier: MIT

# Author:   coneypo
# Blog:     http://www.cnblogs.com/AdaminXie
# GitHub:   https://github.com/coneypo/Dlib_face_recognition_from_camera
# Mail:     coneypo@foxmail.com

# 摄像头实时人脸识别 / Real-time face detection and recognition

import dlib
import numpy as np
import cv2
import pandas as pd
import os
import time
from PIL import Image, ImageDraw, ImageFont

# Dlib 正向人脸检测器 / Use frontal face detector of Dlib
detector = dlib.get_frontal_face_detector()

# Dlib 人脸 landmark 特征点检测器 / Get face landmarks
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')

# Dlib Resnet 人脸识别模型，提取 128D 的特征矢量 / Use Dlib resnet50 model to get 128D face descriptor
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")


class Face_Recognizer:
    def __init__(self):
        self.feature_known_list = []                # 用来存放所有录入人脸特征的数组 / Save the features of faces in the database
        self.name_known_list = []                   # 存储录入人脸名字 / Save the name of faces in the database

        self.current_frame_face_cnt = 0             # 存储当前摄像头中捕获到的人脸数 / Counter for faces in current frame
        self.current_frame_feature_list = []        # 存储当前摄像头中捕获到的人脸特征 / Features of faces in current frame
        self.current_frame_name_position_list = []  # 存储当前摄像头中捕获到的所有人脸的名字坐标 / Positions of faces in current frame
        self.current_frame_name_list = []           # 存储当前摄像头中捕获到的所有人脸的名字 / Names of faces in current frame

        # Update FPS
        self.fps = 0
        self.frame_start_time = 0

    # 从 "features_all.csv" 读取录入人脸特征 / Get known faces from "features_all.csv"
    def get_face_database(self):
        if os.path.exists("data/features_all.csv"):
            path_features_known_csv = "data/features_all.csv"
            csv_rd = pd.read_csv(path_features_known_csv, header=None)
            for i in range(csv_rd.shape[0]):
                features_someone_arr = []
                for j in range(0, 128):
                    if csv_rd.iloc[i][j] == '':
                        features_someone_arr.append('0')
                    else:
                        features_someone_arr.append(csv_rd.iloc[i][j])
                self.feature_known_list.append(features_someone_arr)
                self.name_known_list.append("Person_"+str(i+1))
            print("Faces in Database：", len(self.feature_known_list))
            return 1
        else:
            print('##### Warning #####', '\n')
            print("'features_all.csv' not found!")
            print(
                "Please run 'get_faces_from_camera.py' and 'features_extraction_to_csv.py' before 'face_reco_from_camera.py'",
                '\n')
            print('##### End Warning #####')
            return 0

    # 计算两个128D向量间的欧式距离 / Compute the e-distance between two 128D features
    @staticmethod
    def return_euclidean_distance(feature_1, feature_2):
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        return dist

    # 更新 FPS / Update FPS of Video stream
    def update_fps(self):
        now = time.time()
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

    def draw_note(self, img_rd):
        font = cv2.FONT_ITALIC

        cv2.putText(img_rd, "Face Recognizer", (20, 40), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "FPS:   " + str(self.fps.__round__(2)), (20, 100), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Faces: " + str(self.current_frame_face_cnt), (20, 140), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Q: Quit", (20, 450), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

    def draw_name(self, img_rd):
        # 在人脸框下面写人脸名字 / Write names under rectangle
        font = ImageFont.truetype("simsun.ttc", 30)
        img = Image.fromarray(cv2.cvtColor(img_rd, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        for i in range(self.current_frame_face_cnt):
            # cv2.putText(img_rd, self.current_frame_name_list[i], self.current_frame_name_position_list[i], font, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
            draw.text(xy=self.current_frame_name_position_list[i], text=self.current_frame_name_list[i], font=font)
            img_with_name = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return img_with_name

    # 修改显示人名 / Show names in chinese
    def show_chinese_name(self):
        # Default known name: person_1, person_2, person_3
        if self.current_frame_face_cnt >= 1:
            self.name_known_list[0] ='张1'.encode('utf-8').decode()
            self.name_known_list[1] ='张2'.encode('utf-8').decode()
            self.name_known_list[2] ='张3'.encode('utf-8').decode()
            self.name_known_list[3] ='张4'.encode('utf-8').decode()
            self.name_known_list[4] ='张5'.encode('utf-8').decode()

    # 处理获取的视频流，进行人脸识别 / Face detection and recognition from input video stream
    def process(self, stream):
        # 1. 读取存放所有人脸特征的 csv / Get faces known from "features.all.csv"
        if self.get_face_database():
            while stream.isOpened():
                print(">>> Frame start")
                flag, img_rd = stream.read()
                faces = detector(img_rd, 0)
                kk = cv2.waitKey(1)
                # 按下 q 键退出 / Press 'q' to quit
                if kk == ord('q'):
                    break
                else:
                    self.draw_note(img_rd)
                    self.current_frame_feature_list = []
                    self.current_frame_face_cnt = 0
                    self.current_frame_name_position_list = []
                    self.current_frame_name_list = []

                    # 2. 检测到人脸 / Face detected in current frame
                    if len(faces) != 0:
                        # 3. 获取当前捕获到的图像的所有人脸的特征 / Compute the face descriptors for faces in current frame
                        for i in range(len(faces)):
                            shape = predictor(img_rd, faces[i])
                            self.current_frame_feature_list.append(face_reco_model.compute_face_descriptor(img_rd, shape))
                        # 4. 遍历捕获到的图像中所有的人脸 / Traversal all the faces in the database
                        for k in range(len(faces)):
                            print(">>>>>> For face", k+1, " in camera")
                            # 先默认所有人不认识，是 unknown / Set the default names of faces with "unknown"
                            self.current_frame_name_list.append("unknown")

                            # 每个捕获人脸的名字坐标 / Positions of faces captured
                            self.current_frame_name_position_list.append(tuple(
                                [faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))

                            # 5. 对于某张人脸，遍历所有存储的人脸特征
                            # For every faces detected, compare the faces in the database
                            current_frame_e_distance_list = []
                            for i in range(len(self.feature_known_list)):
                                # 如果 person_X 数据不为空
                                if str(self.feature_known_list[i][0]) != '0.0':
                                    print("   >>> With person", str(i + 1), ", the e distance: ", end='')
                                    e_distance_tmp = self.return_euclidean_distance(self.current_frame_feature_list[k],
                                                                                    self.feature_known_list[i])
                                    print(e_distance_tmp)
                                    current_frame_e_distance_list.append(e_distance_tmp)
                                else:
                                    # 空数据 person_X
                                    current_frame_e_distance_list.append(999999999)
                            # 6. 寻找出最小的欧式距离匹配 / Find the one with minimum e distance
                            similar_person_num = current_frame_e_distance_list.index(min(current_frame_e_distance_list))
                            print("   >>> Minimum e distance with ", self.name_known_list[similar_person_num], ": ", min(current_frame_e_distance_list))

                            if min(current_frame_e_distance_list) < 0.4:
                                self.current_frame_name_list[k] = self.name_known_list[similar_person_num]
                                print("   >>> Face recognition result:  " + str(self.name_known_list[similar_person_num]))
                            else:
                                print("   >>> Face recognition result: Unknown person")

                            # 矩形框 / Draw rectangle
                            for kk, d in enumerate(faces):
                                # 绘制矩形框
                                cv2.rectangle(img_rd, tuple([d.left(), d.top()]), tuple([d.right(), d.bottom()]),
                                              (0, 255, 255), 2)

                        self.current_frame_face_cnt = len(faces)

                        # 7. 在这里更改显示的人名 / Modify name if needed
                        self.show_chinese_name()

                        # 8. 写名字 / Draw name
                        img_with_name = self.draw_name(img_rd)

                    else:
                        img_with_name = img_rd

                print(">>>>>> Faces in camera now:", self.current_frame_name_list)

                cv2.imshow("camera", img_with_name)

                # 9. 更新 FPS / Update stream FPS
                self.update_fps()
                print(">>> Frame ends\n\n")

    # OpenCV 调用摄像头并进行 process
    def run(self):
        cap = cv2.VideoCapture(0)
        # cap = cv2.VideoCapture("video.mp4")
        cap.set(3, 480)     # 640x480
        self.process(cap)

        cap.release()
        cv2.destroyAllWindows()


def main():
    Face_Recognizer_con = Face_Recognizer()
    Face_Recognizer_con.run()


if __name__ == '__main__':
    main()