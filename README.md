# Face recognition from camera

Using Dlib to detect and recognize faces from camera (support multi-faces) ;

调用摄像头进行人脸识别，支持多张人脸同时识别;

Three steps:

	face register >> generate datebase >> face recognition
  	人脸录入 >> 建立人脸数据库 >> 利用摄像头进行人脸识别

1. get_face_from_camera.py : 
	
	# face register / 人脸录入

2. get_features_into_CSV.py: 
	
	# generate the features from the photos you captured and write the datas into CSV / 将图像文件中人脸数据提取出来存入CSV
 	# will generate a "features_all.csv" ( size: n*128 , n means n people you registered and 128 means 128D features of the face)

3. face_reco_from_camera.py: 
	
	# face recognition from camera (support multi-faces) / 实时进行人脸识别
  	# compare the faces captured from camera with the faces you have registered which are saved in "features_all.csv"
  	# 将捕获到的人脸数据和之前存的人脸数据进行对比计算欧式距离
  
For more details, pls refer to my blog (in chinese) or contact me by e-mail:
	
	Blog: https://www.cnblogs.com/AdaminXie/p/9010298.html  
	Mail: coneypo@foxmail.com

Thanks for your support.

Author: coneypo

Last Update: 7 Sep
