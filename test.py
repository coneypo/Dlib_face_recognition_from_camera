import os
import dlib
from skimage import io

# detector to find the faces
detector = dlib.get_frontal_face_detector()

# shape predictor to find the face landmarks
predictor = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")

# the object maps human faces into 128D vectors
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

path_img = "F:/code/python/P_dlib_face_reco/data/get_from_camera/"
img = io.imread(path_img+"img_face_2.jpg")
dets = detector(img, 1)

shape = predictor(img, dets[0])

face_descriptor = facerec.compute_face_descriptor(img, shape)

print("face_descriptor:", face_descriptor)
