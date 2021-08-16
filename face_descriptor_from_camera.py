# Copyright (C) 2018-2021 coneypo
# SPDX-License-Identifier: MIT

# 摄像头实时人脸特征描述子计算 / Real-time face descriptor computing

import dlib         # 人脸识别的库 Dlib
import cv2          # 图像处理的库 OpenCV
import time

# 1. Dlib 正向人脸检测器
detector = dlib.get_frontal_face_detector()

# 2. Dlib 人脸 landmark 特征点检测器
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')

# 3. Dlib Resnet 人脸识别模型，提取 128D 的特征矢量
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")


class Face_Descriptor:
    def __init__(self):
        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0
        self.frame_cnt = 0

    def update_fps(self):
        now = time.time()
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(3, 480)
        self.process(cap)
        cap.release()
        cv2.destroyAllWindows()

    def process(self, stream):
        while stream.isOpened():
            flag, img_rd = stream.read()
            self.frame_cnt+=1
            k = cv2.waitKey(1)

            print('- Frame ', self.frame_cnt, " starts:")

            timestamp1 = time.time()
            faces = detector(img_rd, 0)
            timestamp2 = time.time()
            print("--- Time used to `detector`:                  %s seconds ---" % (timestamp2 - timestamp1))

            font = cv2.FONT_HERSHEY_SIMPLEX

            # 检测到人脸
            if len(faces) != 0:
                for face in faces:
                    timestamp3 = time.time()
                    face_shape = predictor(img_rd, face)
                    timestamp4 = time.time()
                    print("--- Time used to `predictor`:                 %s seconds ---" % (timestamp4 - timestamp3))

                    timestamp5 = time.time()
                    face_desc = face_reco_model.compute_face_descriptor(img_rd, face_shape)
                    timestamp6 = time.time()
                    print("--- Time used to `compute_face_descriptor：`   %s seconds ---" % (timestamp6 - timestamp5))

            # 添加说明
            cv2.putText(img_rd, "Face descriptor", (20, 40), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img_rd, "FPS:   " + str(self.fps.__round__(2)), (20, 100), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(img_rd, "Faces: " + str(len(faces)), (20, 140), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(img_rd, "S: Save current face", (20, 400), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img_rd, "Q: Quit", (20, 450), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

            # 按下 'q' 键退出
            if k == ord('q'):
                break

            self.update_fps()

            cv2.namedWindow("camera", 1)
            cv2.imshow("camera", img_rd)
            print('\n')


def main():
    Face_Descriptor_con = Face_Descriptor()
    Face_Descriptor_con.run()


if __name__ == '__main__':
    main()