import dlib
import cv2
# Dlib 预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

img_gray = cv2.imread("2008_002470.jpg")

dets = detector(img_gray, 1)


if len(dets) != 0:
    face_des = []
    for i in range(len(dets)):
        shape = predictor(img_gray, dets[i])
        face_des.append(facerec.compute_face_descriptor(img_gray, shape))
else:
    face_des = []

print(len(face_des))
# for i in range(len(face_des)):
#     print(face_des[i])
#     print("\n")

print(face_des)