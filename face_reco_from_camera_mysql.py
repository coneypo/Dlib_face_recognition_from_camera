# 摄像头实时人脸识别
# Real-time face recognition

# Author:   coneypo
# Blog:     http://www.cnblogs.com/AdaminXie
# GitHub:   https://github.com/coneypo/Dlib_face_recognition_from_camera

# Created at 2018-05-11
# Updated at 2020-05-29

import dlib         # 人脸处理的库 Dlib
import numpy as np  # 数据处理的库 Numpy
import cv2          # 图像处理的库 OpenCV
import pandas as pd # 数据处理的库 Pandas
import os
import time
from PIL import Image, ImageDraw, ImageFont
import pymysql

db = pymysql.connect("localhost", "root", "intel@123", "dlib_database")
cursor = db.cursor()

# 1. Dlib 正向人脸检测器
detector = dlib.get_frontal_face_detector()

# 2. Dlib 人脸 landmark 特征点检测器
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')

# 3. Dlib Resnet 人脸识别模型，提取 128D 的特征矢量
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")


class Face_Recognizer:
    def __init__(self):
        # 用来存放所有录入人脸特征的数组 / Save the features of faces in the database
        self.features_known_list = []

        # 存储录入人脸名字 / Save the name of faces known
        self.name_known_cnt = 0
        self.name_known_list = []

        # 存储当前摄像头中捕获到的所有人脸的坐标名字 / Save the positions and names of current faces captured
        self.pos_camera_list = []
        self.name_camera_list = []
        # 存储当前摄像头中捕获到的人脸数
        self.faces_cnt = 0
        # 存储当前摄像头中捕获到的人脸特征
        self.features_camera_list = []

        # Update FPS
        self.fps = 0
        self.frame_start_time = 0

    # 从 "features_all.csv" 读取录入人脸特征
    def get_face_database(self):
        # 1. get database face numbers
        cmd_rd = "select count(*) from dlib_face_table;"
        cursor.execute(cmd_rd)
        results = cursor.fetchall()
        person_cnt = int(results[0][0])

        # 2. get features for person X
        for person in range(person_cnt):
            # lookup for personX
            cmd_lookup = "select * from dlib_face_table where person_x=\"person_" + str(person + 1) + "\";"
            cursor.execute(cmd_lookup)
            results = cursor.fetchall()
            results = list(results[0][1:])
            self.features_known_list.append(results)
            self.name_known_list.append("Person_" + str(person + 1))
            print(results)
        self.name_known_cnt = len(self.name_known_list)
        print("Faces in Database：", len(self.features_known_list))
        return 1

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
        cv2.putText(img_rd, "Faces: " + str(self.faces_cnt), (20, 140), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Q: Quit", (20, 450), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

    def draw_name(self, img_rd):
        # 在人脸框下面写人脸名字 / Write names under rectangle
        font = ImageFont.truetype("simsun.ttc", 30)
        img = Image.fromarray(cv2.cvtColor(img_rd, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        for i in range(self.faces_cnt):
            # cv2.putText(img_rd, self.name_camera_list[i], self.pos_camera_list[i], font, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
            draw.text(xy=self.pos_camera_list[i], text=self.name_camera_list[i], font=font)
            img_with_name = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return img_with_name

    # 修改显示人名
    def modify_name_camera_list(self):
        # Default known name: person_1, person_2, person_3
        self.name_known_list[0] ='张三'.encode('utf-8').decode()
        self.name_known_list[1] ='李四'.encode('utf-8').decode()
        # self.name_known_list[2] ='xx'.encode('utf-8').decode()
        # self.name_known_list[3] ='xx'.encode('utf-8').decode()
        # self.name_known_list[4] ='xx'.encode('utf-8').decode()

    # 处理获取的视频流，进行人脸识别 / Input video stream and face reco process
    def process(self, stream):
        # 1. 读取存放所有人脸特征的 csv
        if self.get_face_database():
            while stream.isOpened():
                flag, img_rd = stream.read()
                faces = detector(img_rd, 0)
                kk = cv2.waitKey(1)
                # 按下 q 键退出 / Press 'q' to quit
                if kk == ord('q'):
                    break
                else:
                    self.draw_note(img_rd)
                    self.features_camera_list = []
                    self.faces_cnt = 0
                    self.pos_camera_list = []
                    self.name_camera_list = []

                    # 2. 检测到人脸 / when face detected
                    if len(faces) != 0:
                        # 3. 获取当前捕获到的图像的所有人脸的特征，存储到 self.features_camera_list
                        # 3. Get the features captured and save into self.features_camera_list
                        for i in range(len(faces)):
                            shape = predictor(img_rd, faces[i])
                            self.features_camera_list.append(face_reco_model.compute_face_descriptor(img_rd, shape))

                        # 4. 遍历捕获到的图像中所有的人脸 / Traversal all the faces in the database
                        for k in range(len(faces)):
                            print("##### camera person", k + 1, "#####")
                            # 让人名跟随在矩形框的下方
                            # 确定人名的位置坐标
                            # 先默认所有人不认识，是 unknown
                            # Set the default names of faces with "unknown"
                            self.name_camera_list.append("unknown")

                            # 每个捕获人脸的名字坐标 / Positions of faces captured
                            self.pos_camera_list.append(tuple(
                                [faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))

                            # 5. 对于某张人脸，遍历所有存储的人脸特征
                            # For every faces detected, compare the faces in the database
                            e_distance_list = []
                            for i in range(len(self.features_known_list)):
                                # 如果 person_X 数据不为空
                                if str(self.features_known_list[i][0]) != '0.0':
                                    print("with person", str(i + 1), "the e distance: ", end='')
                                    e_distance_tmp = self.return_euclidean_distance(self.features_camera_list[k],
                                                                                    self.features_known_list[i])
                                    print(e_distance_tmp)
                                    e_distance_list.append(e_distance_tmp)
                                else:
                                    # 空数据 person_X
                                    e_distance_list.append(999999999)
                            # 6. 寻找出最小的欧式距离匹配 / Find the one with minimum e distance
                            similar_person_num = e_distance_list.index(min(e_distance_list))
                            print("Minimum e distance with person", self.name_known_list[similar_person_num])

                            if min(e_distance_list) < 0.4:
                                self.name_camera_list[k] = self.name_known_list[similar_person_num]
                                print("May be person " + str(self.name_known_list[similar_person_num]))
                            else:
                                print("Unknown person")

                            # 矩形框 / Draw rectangle
                            for kk, d in enumerate(faces):
                                # 绘制矩形框
                                cv2.rectangle(img_rd, tuple([d.left(), d.top()]), tuple([d.right(), d.bottom()]),
                                              (0, 255, 255), 2)
                            print('\n')

                        self.faces_cnt = len(faces)
                        # 7. 在这里更改显示的人名 / Modify name if needed
                        # self.modify_name_camera_list()
                        # 8. 写名字 / Draw name
                        # self.draw_name(img_rd)
                        img_with_name = self.draw_name(img_rd)
                    else:
                        img_with_name = img_rd

                print("Faces in camera now:", self.name_camera_list, "\n")

                cv2.imshow("camera", img_with_name)

                # 9. 更新 FPS / Update stream FPS
                self.update_fps()

    # OpenCV 调用摄像头并进行 process
    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(3, 480)
        self.process(cap)

        cap.release()
        cv2.destroyAllWindows()


def main():
    Face_Recognizer_con = Face_Recognizer()
    Face_Recognizer_con.run()


if __name__ == '__main__':
    main()