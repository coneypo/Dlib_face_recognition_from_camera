# created at 2018-05-11
# updated at 2018-05-21

# By coneypo
# Blog: http://www.cnblogs.com/AdaminXie
# GitHub: https://github.com/coneypo/Dlib_face_recognition_from_camera

# 进行人脸录入
# 录入多张人脸
import dlib         # 人脸识别的库 Dlib
import numpy as np  # 数据处理的库 Numpy
import cv2          # 图像处理的库 OpenCv
import os
import time

# Dlib 预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# 创建 cv2 摄像头对象
cap = cv2.VideoCapture(0)

# cap.set(propId, value)
# 设置视频参数，propId 设置的视频参数，value 设置的参数值
cap.set(3, 480)

# 截图 screenshoot 的计数器
cnt_ss = 0

# 人脸截图的计数器
cnt_p = 0

# 存储人脸的文件夹
current_face_dir = 0

# 保存
path_save = "F:/code/python/P_dlib_face_reco/data/get_from_camera/"
path_make_dir = "F:/code/python/P_dlib_face_reco/data/faces_from_camera/"

# cap.isOpened（） 返回 true/false 检查初始化是否成功

while cap.isOpened():

    # cap.read()
    # 返回两个值：
    #    一个布尔值 true/false，用来判断读取视频是否成功/是否到视频末尾
    #    图像对象，图像的三维矩阵q
    flag, im_rd = cap.read()

    # 每帧数据延时 1ms，延时为 0 读取的是静态帧
    kk = cv2.waitKey(1)

    # 取灰度
    img_gray = cv2.cvtColor(im_rd, cv2.COLOR_RGB2GRAY)

    # 人脸数 rects
    rects = detector(img_gray, 0)

    # print(len(rects))

    # 待会要写的字体
    font = cv2.FONT_HERSHEY_SIMPLEX

    # 按下 'n' 新建存储人脸的文件夹
    if kk == ord('n'):
        current_face_dir = path_make_dir + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        os.makedirs(current_face_dir)
        print("新建的人脸文件夹: ", current_face_dir)

        # 将人脸计数器清零
        cnt_p = 0

    if len(rects) != 0:
        # 检测到人脸

        # 矩形框
        for k, d in enumerate(rects):

            # 计算矩形大小
            # (x,y), (宽度width, 高度height)
            pos_start = tuple([d.left(), d.top()])
            pos_end = tuple([d.right(), d.bottom()])

            # 计算矩形框大小
            height = d.bottom() - d.top()
            width = d.right() - d.left()

            # 根据人脸大小生成空的图像
            cv2.rectangle(im_rd, tuple([d.left(), d.top()]), tuple([d.right(), d.bottom()]), (0, 255, 255), 2)
            im_blank = np.zeros((height, width, 3), np.uint8)

            # 按下 's' 保存摄像头中的人脸到本地
            if kk == ord('s'):
                cnt_p += 1
                for ii in range(height):
                    for jj in range(width):
                        im_blank[ii][jj] = im_rd[d.top() + ii][d.left() + jj]
                # 存储人脸图像文件
                #cv2.imwrite(path_save + "img_face_" + str(cnt_p) + ".jpg", im_blank)
                #print("写入本地：", path_save + "img_face_" + str(cnt_p) + ".jpg")
                cv2.imwrite(current_face_dir + "/img_face_" + str(cnt_p) + ".jpg", im_blank)
                print("写入本地：", str(current_face_dir) + "/img_face_" + str(cnt_p) + ".jpg")

        # 显示人脸数
    cv2.putText(im_rd, "Faces: " + str(len(rects)), (20, 100), font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)

    # 添加说明
    cv2.putText(im_rd, "Face Register", (20, 40), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(im_rd, "N: new face folder", (20, 350), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(im_rd, "S: save face", (20, 400), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(im_rd, "Q: quit", (20, 450), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

    # 按下 'q' 键退出
    if kk == ord('q'):
        break

    # 窗口显示
    # cv2.namedWindow("camera", 0) # 如果需要摄像头窗口大小可调
    cv2.imshow("camera", im_rd)

# 释放摄像头
cap.release()

# 删除建立的窗口
cv2.destroyAllWindows()
