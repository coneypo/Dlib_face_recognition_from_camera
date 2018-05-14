# 测试
# 输出单张照片的128D特征face_descriptor

import cv2
import dlib
from skimage import io

# detector to find the faces
detector = dlib.get_frontal_face_detector()
# shape predictor to find the face landmarks
predictor = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")
# face recognition model, the object maps human faces into 128D vectors
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

path_img = "F:/code/python/P_dlib_face_reco/data/get_from_camera/"

# 返回单张图像的128D特征
img = io.imread(path_img+"img_face_2.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
dets = detector(img_gray, 1)

if len(dets) != 0:
    shape = predictor(img_gray, dets[0])
    face_descriptor = facerec.compute_face_descriptor(img_gray, shape)
    print(face_descriptor)
else:
    print("no face")