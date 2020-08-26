# Author:   coneypo
# Blog:     http://www.cnblogs.com/AdaminXie
# GitHub:   https://github.com/coneypo/Dlib_face_recognition_from_camera
# Mail:     coneypo@foxmail.com

# Face recognizer with object tracker

import dlib
import numpy as np
import cv2
import os
import pandas as pd
import time

# Dlib 正向人脸检测器
detector = dlib.get_frontal_face_detector()

# 2. Dlib 人脸 landmark 特征点检测器
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')

# 3. Dlib Resnet 人脸识别模型，提取 128D 的特征矢量
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")


class Face_Recognizer:
    def __init__(self):
        self.font = cv2.FONT_ITALIC

        # for FPS
        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0

        # list to save centroid positions of ROI in frame N-1 and N
        self.last_frame_centroid_list = []
        self.current_frame_centroid_list = []

        # list to save names of ROI in frame N-1 and N
        self.last_frame_face_names_list = []
        self.current_frame_face_names_list = []

        # cnt for faces in frame N-1 and N
        self.last_frame_faces_cnt = 0
        self.current_frame_faces_cnt = 0

        # cnt for frame
        self.frame_cnt = 0

        # 用来存放所有录入人脸特征的数组 / Save the features of faces in the database
        self.features_known_list = []

        self.current_frame_face_X_e_distance_list = []

        # 存储录入人脸名字 / Save the name of faces known
        self.name_known_cnt = 0
        self.name_known_list = []

        # 存储当前摄像头中捕获到的所有人脸的坐标名字 / Save the positions and names of current faces captured
        self.current_frame_face_pos_list = []
        # 存储当前摄像头中捕获到的人脸特征
        self.current_frame_face_features_list = []

        self.last_current_frame_centroid_e_distance = 0

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
                self.features_known_list.append(features_someone_arr)
                self.name_known_list.append("Person_" + str(i + 1))
            self.name_known_cnt = len(self.name_known_list)
            print("Faces in Database：", len(self.features_known_list))
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

    # 获取处理之后 stream 的帧数 / Get the fps of video stream
    def update_fps(self):
        now = time.time()
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now
        # 计算两个128D向量间的欧式距离 / Compute the e-distance between two 128D features

    @staticmethod
    def return_euclidean_distance(feature_1, feature_2):
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        return dist

    def centroid_tracker(self):
        for i in range(len(self.current_frame_centroid_list)):
            e_distance_current_frame_person_x_list = []
            # for object 1 in current_frame, compute e-distance with object 1/2/3/4/... in last frame
            for j in range(len(self.last_frame_centroid_list)):
                self.last_current_frame_centroid_e_distance = self.return_euclidean_distance(
                    self.current_frame_centroid_list[i], self.last_frame_centroid_list[j])

                e_distance_current_frame_person_x_list.append(
                    self.last_current_frame_centroid_e_distance)

            last_frame_num = e_distance_current_frame_person_x_list.index(
                min(e_distance_current_frame_person_x_list))
            self.current_frame_face_names_list[i] = self.last_frame_face_names_list[last_frame_num]

    # 生成的 cv2 window 上面添加说明文字 / putText on cv2 window
    def draw_note(self, img_rd):
        # 添加说明 / Add some statements
        cv2.putText(img_rd, "Face recognizer with OT", (20, 40), self.font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "FPS:    " + str(self.fps.__round__(2)), (20, 100), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "Faces:  " + str(self.current_frame_faces_cnt), (20, 130), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "Q: Quit", (20, 450), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

        for i in range(len(self.current_frame_face_names_list)):
            cv2.putText(img_rd, "face_" + str(i + 1), tuple(
                [int(self.current_frame_centroid_list[i][0]), int(self.current_frame_centroid_list[i][1])]), self.font,
                        0.8, (0, 255, 0),
                        1,
                        cv2.LINE_AA)

    # 获取人脸
    def process(self, stream):
        # 1. read data of known faces from csv
        if self.get_face_database():
            while stream.isOpened():
                flag, img_rd = stream.read()
                self.frame_cnt += 1
                kk = cv2.waitKey(1)
                # 2. detect faces for frame X
                faces = detector(img_rd, 0)
                if self.current_frame_face_names_list == ['Person_2', 'Person_2']:
                    break

                # 3. update cnt for faces in frames
                self.last_frame_faces_cnt = self.current_frame_faces_cnt
                self.current_frame_faces_cnt = len(faces)
                # 4. update the face name list in last frame
                self.last_frame_face_names_list = self.current_frame_face_names_list[:]
                # 5. update frame centroid list
                self.last_frame_centroid_list = self.current_frame_centroid_list
                self.current_frame_centroid_list = []

                # 6. if cnt not changes
                if self.current_frame_faces_cnt == self.last_frame_faces_cnt:
                    self.current_frame_face_pos_list = []

                    if self.current_frame_faces_cnt != 0:
                        # 6.1 get ROI positions
                        for k, d in enumerate(faces):
                            self.current_frame_face_pos_list.append(tuple(
                                [faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))
                            self.current_frame_centroid_list.append(
                                [int(faces[k].left() + faces[k].right()) / 2,
                                 int(faces[k].top() + faces[k].bottom()) / 2])

                            # 计算矩形框大小 / Compute the size of rectangle box
                            height = (d.bottom() - d.top())
                            width = (d.right() - d.left())
                            hh = int(height / 2)
                            ww = int(width / 2)
                            cv2.rectangle(img_rd,
                                          tuple([d.left() - ww, d.top() - hh]),
                                          tuple([d.right() + ww, d.bottom() + hh]),
                                          (255, 255, 255), 2)

                    # multi-faces in current frames, use centroid tracker to track
                    if self.current_frame_faces_cnt != 1:
                        self.centroid_tracker()

                    for i in range(self.current_frame_faces_cnt):
                        # 6.2 write names under ROI
                        cv2.putText(img_rd, self.current_frame_face_names_list[i],
                                    self.current_frame_face_pos_list[i], self.font, 0.8, (0, 255, 255), 1,
                                    cv2.LINE_AA)

                # 7. if cnt of faces changes, 0->1 or 1->0 or ...
                else:
                    self.current_frame_face_pos_list = []
                    self.current_frame_face_X_e_distance_list = []

                    # 7.1 face cnt decrease: 1->0, 2->1, ...
                    if self.current_frame_faces_cnt == 0:
                        # clear list of names and features
                        self.current_frame_face_names_list = []
                        self.current_frame_face_features_list = []

                    # 7.2 face cnt increase: 0->1, 0->2, ..., 1->2, ...
                    else:
                        self.current_frame_face_names_list = []

                        for i in range(len(faces)):
                            shape = predictor(img_rd, faces[i])
                            self.current_frame_face_features_list.append(
                                face_reco_model.compute_face_descriptor(img_rd, shape))
                            self.current_frame_face_names_list.append("unknown")

                        # 7.2.1 遍历捕获到的图像中所有的人脸 / Traversal all the faces in the database
                        for k in range(len(faces)):
                            self.current_frame_centroid_list.append(
                                [int(faces[k].left() + faces[k].right()) / 2,
                                 int(faces[k].top() + faces[k].bottom()) / 2])

                            self.current_frame_face_X_e_distance_list = []

                            # 每个捕获人脸的名字坐标 / Positions of faces captured
                            self.current_frame_face_pos_list.append(tuple(
                                [faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))
                            # 7.2.2 对于某张人脸，遍历所有存储的人脸特征
                            # For every faces detected, compare the faces in the database

                            for i in range(len(self.features_known_list)):
                                # 如果 person_X 数据不为空
                                if str(self.features_known_list[i][0]) != '0.0':
                                    e_distance_tmp = self.return_euclidean_distance(
                                        self.current_frame_face_features_list[k],
                                        self.features_known_list[i])
                                    self.current_frame_face_X_e_distance_list.append(e_distance_tmp)
                                else:
                                    # 空数据 person_X
                                    self.current_frame_face_X_e_distance_list.append(999999999)

                            # 7.2.3. 寻找出最小的欧式距离匹配 / Find the one with minimum e distance
                            similar_person_num = self.current_frame_face_X_e_distance_list.index(
                                min(self.current_frame_face_X_e_distance_list))

                            if min(self.current_frame_face_X_e_distance_list) < 0.4:
                                self.current_frame_face_names_list[k] = self.name_known_list[similar_person_num]

                # 8. 生成的窗口添加说明文字 / Add note on cv2 window
                self.draw_note(img_rd)

                # 9. 按下 'q' 键退出 / Press 'q' to exit
                if kk == ord('q'):
                    break

                self.update_fps()
                cv2.namedWindow("camera", 1)
                cv2.imshow("camera", img_rd)

    def run(self):
        # cap = cv2.VideoCapture("head-pose-face-detection-female-and-male.mp4")
        cap = cv2.VideoCapture(0)
        self.process(cap)

        cap.release()
        cv2.destroyAllWindows()


def main():
    Face_Recognizer_con = Face_Recognizer()
    Face_Recognizer_con.run()


if __name__ == '__main__':
    main()