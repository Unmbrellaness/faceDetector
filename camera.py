
import cv2
import numpy as np
import os
# import mediapipe as mp
import shutil
import threading
import tkinter as tk
# from PIL import Image, ImageTk

id_dict = {}  # 字典里存的是id——name键值对
Total_face_num = 1  # 已经被识别有用户名的人脸个数

camera = cv2.VideoCapture(0)
cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# recognizer = cv2.face.LBPHFaceRecognizer_create()
photoName = 'unknown'
userNumber = 1

def Get_new_face(pictur_num,user_name,userNameId):
    print("正在从摄像头录入新人脸信息 \n")

    # 存在目录data就清空，不存在就创建，确保最后存在空的data目录
    # filepath = "data"
    # if not os.path.exists(filepath):
    #     os.mkdir(filepath)
    # else:
    #     shutil.rmtree(filepath)
    #     os.mkdir(filepath)

    # 创建用户的文件夹
    filepath = "data/"+str(user_name)
    if not os.path.exists(filepath):
        os.mkdir(filepath)

    sample_num = 0  # 已经获得的样本数
    user_name_id = userNameId

    while True:  # 从摄像头读取图片

        success, img = camera.read()

        # 转为灰度图片
        if success is True:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            break

        # 检测人脸，将每一帧摄像头记录的数据带入OpenCv中，让Classifier判断人脸
        # 其中gray为要检测的灰度图像，1.3为每次图像尺寸减小的比例，5为minNeighbors
        face_detector = cascade
        faces = face_detector.detectMultiScale(gray, 1.2, 5)

        # 框选人脸，for循环保证一个能检测的实时动态视频流
        check_box_color = (255,0,0) #蓝色
        text_color = (255, 0, 255)  #紫色
        for (x, y, w, h) in faces:
            # xy为左上角的坐标,w为宽，h为高，用rectangle为人脸标记画框
            cv2.rectangle(img, (x, y), (x + w, y + w), check_box_color,2)
            cv2.putText(img,'{}'.format(photoName),(x, y),cv2.FONT_HERSHEY_SIMPLEX,1,text_color,1)
            # 样本数加1
            sample_num += 1
            # 保存图像，把灰度图片看成二维数组来检测人脸区域，这里是保存在data缓冲文件夹内
            cv2.imwrite("./data/" + str(user_name)+ "/"+ str(user_name_id) +'.'+ str(sample_num)+ '.jpg', gray[y:y + h, x:x + w])

        # pictur_num = 30  # 表示摄像头拍摄取样的数量,越多效果越好，但获取以及训练的越慢
        cv2.imshow("camera", img)
        cv2.waitKey(10)
        if sample_num > pictur_num:
            break
        else:  # 控制台内输出进度条
            l = int(sample_num / pictur_num * 50)
            r = int((pictur_num - sample_num) / pictur_num * 50)
            print("\r" + "%{:.1f}".format(sample_num / pictur_num * 100) + "=" * l + "->" + "_" * r, end="")

# 创建一个函数，用于从数据集文件夹中获取训练图片,并获取id
def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    print(image_paths)
    # 新建2个list用于存放
    face_samples = []
    ids = []

    # 遍历图片路径，导入图片和id添加到list中
    for image_path in image_paths:

        # 通过图片路径将其转换为灰度图片
        img = Image.open(image_path).convert('L')

        # 将图片转化为数组
        img_np = np.array(img, 'uint8')

        if os.path.split(image_path)[-1].split(".")[-1] != 'jpg':
            continue

        # 为了获取id，将图片和路径分裂并获取
        id = int(os.path.split(image_path)[-1].split(".")[0])
        # id = str(os.path.split(image_path)[-1].split(".")[1])
        ids.append(id)
        face_samples.append(img_np)
        # # 调用熟悉的人脸分类器
        # detector = cascade
        # faces = detector.detectMultiScale(img_np)
        #
        # # 将获取的图片和id添加到list中
        # for (x, y, w, h) in faces:
        #     face_samples.append(img_np[y:y + h, x:x + w])
        #     ids.append(id)
    return face_samples, ids

def Train_new_face():
    print("\n正在训练")
    # cv2.destroyAllWindows()
    trainpath = "data"

    # 初始化识别的方法
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    # 调用函数并将数据喂给识别器训练
    faces, ids = get_images_and_labels(trainpath)
    print('本次用于训练的组别为:')  # 调试信息
    print(ids)  # 输出组别

    # print(np.array(ids))

    # 训练模型  #将输入的所有图片转成四维数组
    recognizer.train(faces, np.array(ids))
    # 保存模型

    yml = "model.yml"
    rec_f = open(yml, "w+")
    rec_f.close()
    recognizer.save(yml)


def scan_face(nameTable):
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    recognizer.read('model.yml')

    while True:  # 从摄像头读取图片
        success, img = camera.read()

        # 转为灰度图片
        if success is True:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            break

        # 检测人脸，将每一帧摄像头记录的数据带入OpenCv中，让Classifier判断人脸
        # 其中gray为要检测的灰度图像，1.3为每次图像尺寸减小的比例，5为minNeighbors
        face_detector = cascade
        faces = face_detector.detectMultiScale(gray, 1.2, 5)

        # 框选人脸，for循环保证一个能检测的实时动态视频流
        check_box_color = (255, 0, 0)  # 蓝色
        text_color = (255, 0, 255)  # 紫色
        for (x, y, w, h) in faces:
            # 识别
            idnum, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            if confidence > 85:
                idnum = 0

            # xy为左上角的坐标,w为宽，h为高，用rectangle为人脸标记画框
            cv2.rectangle(img, (x, y), (x + w, y + h), check_box_color, 2)
            cv2.putText(img, 'id:{} conf:{:.3f}'.format(nameTable[idnum],confidence), (x, y), cv2.FONT_HERSHEY_PLAIN, 2, text_color, 1)

        cv2.imshow("camera", img)
        cv2.waitKey(10)

# while True:

    # ret,frame=camera.read()
    # gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # faces = cascade.detectMultiScale(gray,1.15)
    # for (x,y,w,h) in faces:
    #     cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),5)
    #     # cv2.imwrite("./data/User." + str(T) + '.' + str(sample_num) + '.jpg', gray[y:y + h, x:x + w])
    # cv2.imshow("camera",frame)
    # cv2.waitKey(10)
    # if cv2.waitKey(10) & 0xFF == ord('q'):
    #     break

# Get_new_face(100)
if __name__ == '__main__':
    people = ['unknown','cy','zbc','dyf','gpl']
    # pictur_num = 100
    # user_name = 'gpl'
    # Get_new_face(pictur_num, user_name,4)
    # Train_new_face()
    scan_face(people)

