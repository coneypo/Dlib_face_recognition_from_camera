Face recognition from camera
############################

Introduction
************

Detect and recognize single/multi-faces from camera by Dlib;

调用摄像头进行人脸识别，支持多张人脸同时识别;


#. Face register / 人脸录入 

   .. image:: introduction/get_face_from_camera.png
      align: center

#. Generate database / 建立人脸数据库 
#. Face recognizer / 利用摄像头进行人脸识别
   
   当单张人脸:
   
   .. image:: introduction/face_reco_single_person.png
      :align: center

   当多张人脸:
   
   .. image:: introduction/face_reco_two_people.png
      :align: center


About Source Code
*****************

Python 源码介绍如下:

#. get_face_from_camera.py: 

   
   进行 Face register / 人脸信息采集录入

   请注意存储人脸图片时，矩形框不要超出摄像头范围，要不然无法保存到本地;
   
   超出会有 "out of range" 的提醒;


#. get_features_into_CSV.py: 
     
   从上一步存下来的图像文件中，提取人脸数据存入CSV;
  
   会生成一个存储所有特征人脸数据的 "features_all.csv"；
   ( size: n*128 , n means n people you registered and
   128 means 128D features of the face)


#. face_reco_from_camera.py: 

   This part will implement real-time face recognition;

   这一步将调用摄像头进行实时人脸识别;
  
   Compare the faces captured from camera with the 
   faces you have registered which are saved in "features_all.csv"
   
   将捕获到的人脸数据和之前存的人脸数据进行对比计算欧式距离,
   由此判断是否是同一个人;


For more details, please refer to my blog (in chinese) or contact me by e-mail;

可以访问我的博客获取本项目的更详细介绍，如有问题欢迎邮件联系我:

  Blog: https://www.cnblogs.com/AdaminXie/p/9010298.html  
  
  Mail: coneypo@foxmail.com


仅限于交流学习, 商业合作勿扰；

Author : coneypo
Thanks for your support.
