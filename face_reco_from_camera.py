# 摄像头实时人脸识别
# Real-time face recognition

# Author:   coneypo
# Blog:     http://www.cnblogs.com/AdaminXie
# GitHub:   https://github.com/coneypo/Dlib_face_recognition_from_camera

# Created at 2018-05-11
# Updated at 2020-02-27

import dlib          # 人脸处理的库 Dlib
import numpy as np   # 数据处理的库 numpy
import cv2           # 图像处理的库 OpenCv
import pandas as pd  # 数据处理的库 Pandas
import os

# 人脸识别模型，提取128D的特征矢量
# face recognition model, the object maps human faces into 128D vectors
# Refer this tutorial: http://dlib.net/python/index.html#dlib.face_recognition_model_v1
facerec = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")


# 计算两个128D向量间的欧式距离
# Compute the e-distance between two 128D features
def return_euclidean_distance(feature_1, feature_2):
    feature_1 = np.array(feature_1)
    feature_2 = np.array(feature_2)
    dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
    return dist


# 1. Check 存放所有人脸特征的 csv
if os.path.exists("data/features_all.csv"):
    path_features_known_csv = "data/features_all.csv"
    csv_rd = pd.read_csv(path_features_known_csv, header=None)

    # 用来存放所有录入人脸特征的数组
    # The array to save the features of faces in the database
    features_known_arr = []

    # 2. 读取已知人脸数据
    # Print known faces
    for i in range(csv_rd.shape[0]):
        features_someone_arr = []
        for j in range(0, len(csv_rd.iloc[i])):
            features_someone_arr.append(csv_rd.iloc[i][j])
        features_known_arr.append(features_someone_arr)
    print("Faces in Database：", len(features_known_arr))

    # Dlib 检测器和预测器
    # The detector and predictor will be used
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')

    # 创建 cv2 摄像头对象
    cap = cv2.VideoCapture(0)

    # 3. When the camera is open
    while cap.isOpened():

        flag, img_rd = cap.read()
        img_gray = cv2.cvtColor(img_rd, cv2.COLOR_RGB2GRAY)
        faces = detector(img_gray, 0)

        # 待会要写的字体 font to write later
        font = cv2.FONT_ITALIC

        # 存储当前摄像头中捕获到的所有人脸的坐标/名字
        # The list to save the positions and names of current faces captured
        pos_namelist = []
        name_namelist = []

        kk = cv2.waitKey(1)

        # 按下 q 键退出
        # press 'q' to exit
        if kk == ord('q'):
            break
        else:
            # 检测到人脸 when face detected
            if len(faces) != 0:
                # 4. 获取当前捕获到的图像的所有人脸的特征，存储到 features_cap_arr
                # 4. Get the features captured and save into features_cap_arr
                features_cap_arr = []
                for i in range(len(faces)):
                    shape = predictor(img_rd, faces[i])
                    features_cap_arr.append(facerec.compute_face_descriptor(img_rd, shape))

                # 5. 遍历捕获到的图像中所有的人脸
                # 5. Traversal all the faces in the database
                for k in range(len(faces)):
                    print("##### camera person", k+1, "#####")
                    # 让人名跟随在矩形框的下方
                    # 确定人名的位置坐标
                    # 先默认所有人不认识，是 unknown
                    # Set the default names of faces with "unknown"
                    name_namelist.append("unknown")

                    # 每个捕获人脸的名字坐标 the positions of faces captured
                    pos_namelist.append(tuple([faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top())/4)]))

                    # 对于某张人脸，遍历所有存储的人脸特征
                    # For every faces detected, compare the faces in the database
                    e_distance_list = []
                    for i in range(len(features_known_arr)):
                        # 如果 person_X 数据不为空
                        if str(features_known_arr[i][0]) != '0.0':
                            print("with person", str(i + 1), "the e distance: ", end='')
                            e_distance_tmp = return_euclidean_distance(features_cap_arr[k], features_known_arr[i])
                            print(e_distance_tmp)
                            e_distance_list.append(e_distance_tmp)
                        else:
                            # 空数据 person_X
                            e_distance_list.append(999999999)
                    # Find the one with minimum e distance
                    similar_person_num = e_distance_list.index(min(e_distance_list))
                    print("Minimum e distance with person", int(similar_person_num)+1)

                    if min(e_distance_list) < 0.4:
                        ####### 在这里修改 person_1, person_2 ... 的名字 ########
                        # 可以在这里改称 Jack, Tom and others
                        # Here you can modify the names shown on the camera
                        name_namelist[k] = "Person "+str(int(similar_person_num)+1)
                        print("May be person "+str(int(similar_person_num)+1))
                    else:
                        print("Unknown person")

                    # 矩形框
                    # draw rectangle
                    for kk, d in enumerate(faces):
                        # 绘制矩形框
                        cv2.rectangle(img_rd, tuple([d.left(), d.top()]), tuple([d.right(), d.bottom()]), (0, 255, 255), 2)
                    print('\n')

                # 6. 在人脸框下面写人脸名字
                # 6. write names under rectangle
                for i in range(len(faces)):
                    cv2.putText(img_rd, name_namelist[i], pos_namelist[i], font, 0.8, (0, 255, 255), 1, cv2.LINE_AA)

        print("Faces in camera now:", name_namelist, "\n")

        cv2.putText(img_rd, "Press 'q': Quit", (20, 450), font, 0.8, (84, 255, 159), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Face Recognition", (20, 40), font, 1, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Faces: " + str(len(faces)), (20, 100), font, 1, (0, 0, 255), 1, cv2.LINE_AA)

        cv2.imshow("camera", img_rd)

    cap.release()
    cv2.destroyAllWindows()

else:
    print('##### Warning #####', '\n')
    print("'features_all.py' not found!")
    print("Please run 'get_faces_from_camera.py' and 'features_extraction_to_csv.py' before 'face_reco_from_camera.py'", '\n')
    print('##### Warning #####')