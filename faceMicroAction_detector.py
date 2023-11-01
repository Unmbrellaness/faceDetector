import time
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 调用关键点检测模型
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True,
                                  max_num_faces=3,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)


# 读取图像
img1 = cv2.imread(r"C:\Users\10481\Desktop\faceDetection_v2\image.jpg")
# 将BGR图像转为RGB图像
_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
# 使用模型获取关键点
results = face_mesh.process(_img1)


# 输出关键点
landmarks = results.multi_face_landmarks
print(landmarks)
# 提取关键点坐标
print(landmarks[0].landmark)

# 获取每个人脸的关键点的数量
print(len(landmarks[0].landmark))


# mediapipe提供的绘制模块
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 利用mp_drawing绘制面网
annotated_image = img1.copy()
# 循环获取每一个人脸的关键点
for face_landmarks in results.multi_face_landmarks:
    mp_drawing.draw_landmarks(image=annotated_image, landmark_list=face_landmarks,
                              # 选取关键点
                              connections=mp_face_mesh.FACEMESH_TESSELATION,
                              # 绘制关键点，若为None，表示不绘制关键点，也可以指定点的颜色、粗细、半径
                              landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2,
                                                                           circle_radius=2),
                              # 绘制连接线的格式
                              connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

# # 利用opencv显示图像，在jupyter中显示效果一般
# cv2.imshow("image",annotated_image)
# if cv2.waitKey(0) == ord("q"):
#     cv2.destroyAllWindows()

# 利用matplotlib展示
plt.figure(figsize=(50, 20))
plt.imshow(annotated_image[:, :, ::-1])

# 获取摄像头，0/1为摄像头编号
cap = cv2.VideoCapture(0)
# 循环读取视频每一帧
while True:
    success, frame = cap.read()
    if success:
        start = time.time()
        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # 复制图像
        annotated_image = frame.copy()
        # 解封此行代码，可以将关键点绘制在黑色图像上，该代码能生成一张黑色图像
        # annotated_image = np.zeros(annotated_image.shape, dtype='uint8')
        # 如果检测到关键点
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 利用mp_drawing绘制图像
                mp_drawing.draw_landmarks(image=annotated_image, landmark_list=face_landmarks,
                                          connections=mp_face_mesh.FACEMESH_TESSELATION,
                                          landmark_drawing_spec=None,
                                          connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

                mp_drawing.draw_landmarks(image=annotated_image, landmark_list=face_landmarks,
                                          connections=mp_face_mesh.FACEMESH_IRISES,
                                          landmark_drawing_spec=None,
                                          connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())

                mp_drawing.draw_landmarks(image=annotated_image, landmark_list=face_landmarks,
                                          connections=mp_face_mesh.FACEMESH_CONTOURS,
                                          landmark_drawing_spec=None,
                                          connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
            end = time.time()
            fps = 1 / (end - start)

            annotated_image = cv2.putText(annotated_image, str(int(fps)), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,
                                          (0, 0, 255), 1)
        # 如果没有检测到关键点，在黑色背景上显示“NO FACE TO DETECT”
        else:
            annotated_image = np.zeros(annotated_image.shape, dtype='uint8')
            annotated_image = cv2.putText(annotated_image, str("NO FACE TO DETECT"), (300, 400),
                                          cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 1)

        cv2.imshow("image", annotated_image)
        if cv2.waitKey(30) == ord("q"):
            cv2.destroyAllWindows()
            break
# 释放摄像头
cap.release()
