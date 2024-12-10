# Copyright (C) 2018-2021 coneypo
# SPDX-License-Identifier: MIT

# Author:   coneypo
# Blog:     http://www.cnblogs.com/AdaminXie
# GitHub:   https://github.com/coneypo/Dlib_face_recognition_from_camera
# Mail:     coneypo@foxmail.com

# 人脸录入 Tkinter GUI / Face register GUI with tkinter

import dlib
import numpy as np
import cv2
import os
import shutil
import time
import logging
import tkinter as tk
from tkinter import font as tkFont
from tkinter import messagebox
from PIL import Image, ImageTk

# Dlib 正向人脸检测器 / Use frontal face detector of Dlib
detector = dlib.get_frontal_face_detector()


class Face_Register:
    def __init__(self):

        self.current_frame_faces_cnt = 0  # 当前帧中人脸计数器 / cnt for counting faces in current frame
        self.existing_faces = 0  # 已录入的人脸数 / cnt for counting saved faces
        self.ss_cnt = 0  # 录入 person_n 人脸时图片计数器 / cnt for screen shots
        self.registered_names = []  # 已录入的人脸名字 / names of registered faces
        
        self.path_photos_from_camera = "data/data_faces_from_camera/"
        self.current_face_dir = ""
        self.font = cv2.FONT_ITALIC

        if os.listdir(self.path_photos_from_camera):
            self.existing_faces = len(os.listdir(self.path_photos_from_camera))

        # Tkinter GUI
        self.win = tk.Tk()
        self.win.title("人脸录入")

        # PLease modify window size here if needed
        self.win.geometry("1300x550")

        # GUI left part
        self.frame_left_camera = tk.Frame(self.win)
        self.label = tk.Label(self.win)
        self.label.pack(side=tk.LEFT)
        self.frame_left_camera.pack()

        # GUI right part
        self.frame_right_info = tk.Frame(self.win)
        self.label_cnt_face_in_database = tk.Label(self.frame_right_info, text=str(self.existing_faces))
        self.label_fps_info = tk.Label(self.frame_right_info, text="")
        self.input_name = tk.Entry(self.frame_right_info, width=25)
        self.input_name_char = ""
        self.label_warning = tk.Label(self.frame_right_info)
        self.label_face_cnt = tk.Label(self.frame_right_info, text="Faces in current frame: ")
        self.log_all = tk.Label(self.frame_right_info)

        self.font_title = tkFont.Font(family='Helvetica', size=20, weight='bold')
        self.font_step_title = tkFont.Font(family='Helvetica', size=15, weight='bold')
        self.font_warning = tkFont.Font(family='Helvetica', size=15, weight='bold')

        # Current frame and face ROI position
        self.current_frame = np.ndarray
        self.face_ROI_image = np.ndarray
        self.face_ROI_width_start = 0
        self.face_ROI_height_start = 0
        self.face_ROI_width = 0
        self.face_ROI_height = 0
        self.ww = 0
        self.hh = 0

        self.out_of_range_flag = False
        self.face_folder_created_flag = False

        # FPS
        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0
        self.fps_show = 0
        self.start_time = time.time()

        self.cap = cv2.VideoCapture(0)  # Get video stream from camera
        # self.cap = cv2.VideoCapture("test.mp4")   # Input local video

    # 删除之前存的人脸数据文件夹 / Delete old face folders
    def GUI_clear_data(self):
        # 删除之前存的人脸数据文件夹, 删除 "/data_faces_from_camera/person_x/"...
        folders_rd = os.listdir(self.path_photos_from_camera)
        for i in range(len(folders_rd)):
            shutil.rmtree(self.path_photos_from_camera + folders_rd[i])
        if os.path.isfile("./data/features_all.csv"):
            os.remove("./data/features_all.csv")
        self.label_cnt_face_in_database['text'] = "0"
        self.registered_names.clear()
        self.log_all["text"] = "全部图片和`features_all.csv`已全部移除!"
        self.log_all["fg"] = "green"

    def GUI_get_input_name(self):
        self.input_name_char = self.input_name.get()
        if self.input_name_char:
            if self.input_name_char not in self.registered_names:

                self.create_face_folder()
                self.registered_names.append(self.input_name_char)
                self.label_cnt_face_in_database['text'] = str(self.registered_names.__len__())
            else:
                self.log_all["text"] = "此名字已被录入，请输入新的名字!"
                self.log_all["fg"] = "red"
        else:
            self.log_all["text"] = "请输入姓名"
            self.log_all["fg"] = "red"

    def delete_name(self):
        self.input_name_char = self.input_name.get()
        if self.input_name_char:
            if self.input_name_char in self.registered_names:
                self.remove_face_dir(self.path_photos_from_camera + "person_" + self.input_name_char)
                self.log_all["text"] = "'" + self.input_name_char + "'" + "已移除!"
                self.log_all["fg"] = "green"
                self.registered_names.remove(self.input_name_char)
                self.label_cnt_face_in_database['text'] = str(self.registered_names.__len__())
            else:
                self.log_all["text"] = "此名字不存在，请输入正确的名字!"
                self.log_all["fg"] = "red"
        else:
            self.log_all["text"] = "请先输入要删除的姓名"
            self.log_all["fg"] = "red"

    def change_name(self):
        self.input_name_char = self.input_name.get()
        if self.input_name_char:
            if self.input_name_char in self.registered_names:
                self.current_face_dir = self.path_photos_from_camera + \
                                    "person_" + \
                                    self.input_name_char
                pecturt_list = os.listdir(self.current_face_dir)
                self.ss_cnt = len(pecturt_list)  # 将人脸计数器置为原来的 / Clear the cnt of screen shots
                self.face_folder_created_flag = True  # Face folder already created
                self.label_cnt_face_in_database['text'] = str(self.registered_names.__len__())
                self.log_all["text"] = "可以添加新照片了!"
                self.log_all["fg"] = "green"
            else:
                self.log_all["text"] = "此名字不存在，请输入正确的名字!"
                self.log_all["fg"] = "red"
        else:
            self.log_all["text"] = "请先输入要更改的姓名"
            self.log_all["fg"] = "red"      

    def GUI_info(self):
        tk.Label(self.frame_right_info,
                 text="Face register",
                 font=self.font_title).grid(row=0, column=0, columnspan=3, sticky=tk.W, padx=2, pady=20)

        tk.Label(self.frame_right_info,
                 text="FPS: ").grid(row=1, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
        self.label_fps_info.grid(row=1, column=2, sticky=tk.W, padx=5, pady=2)

        tk.Label(self.frame_right_info,
                 text="数据库中已有的人脸: ").grid(row=2, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
        self.label_cnt_face_in_database.grid(row=2, column=2, columnspan=3, sticky=tk.W, padx=5, pady=2)

        tk.Label(self.frame_right_info,
                 text="当前帧中的人脸: ").grid(row=3, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
        self.label_face_cnt.grid(row=3, column=2, columnspan=3, sticky=tk.W, padx=5, pady=2)

        self.label_warning.grid(row=4, column=0, columnspan=3, sticky=tk.W, padx=5, pady=2)

        # Step 1: Clear old data
        tk.Label(self.frame_right_info,
                 font=self.font_step_title,
                 text="删除之前存的人脸数据文件夹").grid(row=5, column=0, columnspan=2, sticky=tk.W, padx=5, pady=20)
        tk.Button(self.frame_right_info,
                  text='删除全部',
                  command=self.GUI_clear_data).grid(row=6, column=0, columnspan=3, sticky=tk.W, padx=5, pady=2)

        # Step 2: Input name and create folders for face
        tk.Label(self.frame_right_info,
                 font=self.font_step_title,
                 text="Step 1: 输入姓名").grid(row=7, column=0, columnspan=2, sticky=tk.W, padx=5, pady=20)

        tk.Label(self.frame_right_info, text="姓名: ").grid(row=8, column=0, sticky=tk.W, padx=5, pady=0)
        self.input_name.grid(row=8, column=1, sticky=tk.W, padx=0, pady=2)

        tk.Button(self.frame_right_info,
                  text='录入',
                  command=self.GUI_get_input_name).grid(row=8, column=2, padx=5)
        
        tk.Button(self.frame_right_info,
                  text='更改',
                  command=self.change_name).grid(row=8, column=3, padx=5)
        
        tk.Button(self.frame_right_info,
                  text='删除',
                  command=self.delete_name).grid(row=8, column=4, padx=5)

        # Step 3: Save current face in frame
        tk.Label(self.frame_right_info,
                 font=self.font_step_title,
                 text="Step 2: 保存当前人脸图片").grid(row=9, column=0, columnspan=2, sticky=tk.W, padx=5, pady=20)

        tk.Button(self.frame_right_info,
                  text='保存',
                  command=self.save_current_face).grid(row=10, column=0, columnspan=3, sticky=tk.W)

        # Show log in GUI
        self.log_all.grid(row=11, column=0, columnspan=20, sticky=tk.W, padx=5, pady=20)

        self.frame_right_info.pack()

    # 新建保存人脸图像文件和数据 CSV 文件夹 / Mkdir for saving photos and csv
    def pre_work_mkdir(self):
        # 新建文件夹 / Create folders to save face images and csv
        if os.path.isdir(self.path_photos_from_camera):
            pass
        else:
            os.makedirs(self.path_photos_from_camera)

    # 如果有之前录入的人脸, 在之前 person_x 的序号按照 person_x+1 开始录入 / Start from person_x+1
    def check_existing_faces(self):
        if os.listdir(self.path_photos_from_camera):
            # 获取已录入的最后一个人脸序号 / Get the order of latest person
            person_list = os.listdir(self.path_photos_from_camera)
            for person in person_list:
                name = person.split('_')[1]
                self.registered_names.append(name)
            self.existing_faces = len(person_list)

        # 如果第一次存储或者没有之前录入的人脸, 按照 person_1 开始录入 / Start from person_1
        else:
            self.registered_names.clear()
            print("No previous data.")

    # 更新 FPS / Update FPS of Video stream
    def update_fps(self):
        now = time.time()
        # 每秒刷新 fps / Refresh fps per second
        if str(self.start_time).split(".")[0] != str(now).split(".")[0]:
            self.fps_show = self.fps
        self.start_time = now
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now
        formatted_fps = "{:.2f}".format(self.fps)
        self.label_fps_info["text"] = str(formatted_fps)

    def create_face_folder(self):
        # 新建存储人脸的文件夹 / Create the folders for saving faces
        self.current_face_dir = self.path_photos_from_camera + \
                                    "person_" + \
                                    self.input_name_char
        os.makedirs(self.current_face_dir)
        self.log_all["text"] = "\"" + self.current_face_dir + "/\" created!"
        self.log_all["fg"] = "green"
        logging.info("\n%-40s %s", "新建的人脸文件夹 / Create folders:", self.current_face_dir)

        self.ss_cnt = 0  # 将人脸计数器清零 / Clear the cnt of screen shots
        self.face_folder_created_flag = True  # Face folder already created

    def remove_face_dir(self, folder_path):
        try:
            shutil.rmtree(folder_path)
            print(f"Folder '{folder_path}' has been deleted successfully.")
        except Exception as e:
            print(f"Failed to delete folder '{folder_path}'. Error: {e}")

    def save_current_face(self):
        if self.face_folder_created_flag:
            if self.current_frame_faces_cnt == 1:
                if not self.out_of_range_flag:
                    self.ss_cnt += 1
                    # 根据人脸大小生成空的图像 / Create blank image according to the size of face detected
                    self.face_ROI_image = np.zeros((int(self.face_ROI_height * 2), self.face_ROI_width * 2, 3),
                                                   np.uint8)
                    for ii in range(self.face_ROI_height * 2):
                        for jj in range(self.face_ROI_width * 2):
                            self.face_ROI_image[ii][jj] = self.current_frame[self.face_ROI_height_start - self.hh + ii][
                                self.face_ROI_width_start - self.ww + jj]
                    self.log_all["text"] = "\"" + self.current_face_dir + "/img_face_" + str(
                        self.ss_cnt) + ".jpg\"" + " 保存成功!"
                    self.log_all["fg"] = "green"

                    # 使用Pillow保存图像
                    img_pil = Image.fromarray(self.face_ROI_image)
                    img_pil.save(self.current_face_dir + "/img_face_" + str(self.ss_cnt) + ".jpg")
                    
                    logging.info("%-40s %s/img_face_%s.jpg", "写入本地 / Save into：",
                                 str(self.current_face_dir), str(self.ss_cnt) + ".jpg")
                else:
                    self.log_all["text"] = "人脸不在范围内（人脸框白色才能保存）!"
                    self.log_all["fg"] = "red"
            else:
                self.log_all["text"] = "没找到人脸或者找到多个人脸"
                self.log_all["fg"] = "red"
        else:
            self.log_all["text"] = "请先执行step 1"
            self.log_all["fg"] = "red"

    def get_frame(self):
        try:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    return ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    raise Exception("Unable to open the camera")
        except Exception as e:
            messagebox.showerror("Error", f"没有找到摄像头!!!{e}\n")
            print("Error: No video input!!!{e}")

    # 获取人脸 / Main process of face detection and saving
    def process(self):
        ret, self.current_frame = self.get_frame()
        faces = detector(self.current_frame, 0)
        # Get frame
        if ret:
            self.update_fps()
            self.label_face_cnt["text"] = str(len(faces))
            # 检测到人脸 / Face detected
            if len(faces) != 0:
                # 矩形框 / Show the ROI of faces
                for k, d in enumerate(faces):
                    self.face_ROI_width_start = d.left()
                    self.face_ROI_height_start = d.top()
                    # 计算矩形框大小 / Compute the size of rectangle box
                    self.face_ROI_height = (d.bottom() - d.top())
                    self.face_ROI_width = (d.right() - d.left())
                    self.hh = int(self.face_ROI_height / 2)
                    self.ww = int(self.face_ROI_width / 2)

                    # 判断人脸矩形框是否超出 480x640 / If the size of ROI > 480x640
                    if (d.right() + self.ww) > 640 or (d.bottom() + self.hh > 480) or (d.left() - self.ww < 0) or (
                            d.top() - self.hh < 0):
                        self.label_warning["text"] = "OUT OF RANGE"
                        self.label_warning['fg'] = 'red'
                        self.out_of_range_flag = True
                        color_rectangle = (255, 0, 0)
                    else:
                        self.out_of_range_flag = False
                        self.label_warning["text"] = ""
                        color_rectangle = (255, 255, 255)
                    self.current_frame = cv2.rectangle(self.current_frame,
                                                       tuple([d.left() - self.ww, d.top() - self.hh]),
                                                       tuple([d.right() + self.ww, d.bottom() + self.hh]),
                                                       color_rectangle, 2)
            self.current_frame_faces_cnt = len(faces)

            # Convert PIL.Image.Image to PIL.Image.PhotoImage
            img_Image = Image.fromarray(self.current_frame)
            img_PhotoImage = ImageTk.PhotoImage(image=img_Image)
            self.label.img_tk = img_PhotoImage
            self.label.configure(image=img_PhotoImage)

        # Refresh frame
        self.win.after(20, self.process)

    def run(self):
        self.pre_work_mkdir()
        self.check_existing_faces()
        self.GUI_info()
        self.process()
        self.win.mainloop()


def main():
    logging.basicConfig(level=logging.INFO)
    Face_Register_con = Face_Register()
    Face_Register_con.run()


if __name__ == '__main__':
    main()
