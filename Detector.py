import numpy as np
import mediapipe as mp
import math
import cv2

# 0 左眉左上角 46
# 1 左眉右角 55
# 2 右眉左角 285
# 3 右眉右上角 276
# 4 左眼皮上 159
# 5 左眼皮下 144
# 6 左瞳孔上 470
# 7 左瞳孔下 472
# 8 左眼左上角 130
# 9 左眼右上角 133
# 10 右眼皮上 386
# 11 右眼皮下 373
# 12 右瞳孔上 475
# 13 右瞳孔下 477
# 14 右眼左上角 362
# 15 右眼右上角 359
# 16 鼻子左上角 209
# 17 鼻子右上角 429
# 18 嘴左上角 61
# 19 嘴右上角 292
# 20 嘴中央下角 17
# 21 嘴中央上角 11
# 22 下巴角 152

# 左上点和右下点截图
def crop(image, lu, rd):
    return image[lu[1]:rd[1], lu[0]:rd[0]]

# 通过 mediapipe 获取人脸特征点并筛选出需要使用的点
class FaceDetector:
    def __init__(self):
        self.__detector = mp.solutions.face_mesh.FaceMesh(max_num_faces=1,
                                                          refine_landmarks=True,
                                                          min_detection_confidence=0.75,
                                                          min_tracking_confidence=0.5)
        self.__POINT_NEEDED = (46, 55, 285, 276, 159, 144, 470, 472, 130, 133, 386,
                               373, 475, 477, 362, 359, 209, 429, 61, 292, 17, 11, 152, 50, 280)
        self.__INDEX_DICT = {46: 0, 55: 1, 285: 2, 276: 3, 159: 4, 144: 5, 470: 6, 472: 7, 130: 8, 133: 9, 386: 10,
                             373: 11, 475: 12, 477: 13, 362: 14, 359: 15, 209: 16, 429: 17, 61: 18, 292: 19, 17: 20,
                             11: 21, 152: 22, 50: 23, 280: 24}

    def run(self, image):
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.__detector.process(image)

        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        h, w, c = image.shape

        pointDict = {}
        time = cv2.getTickCount()
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for pointID, lm in enumerate(face_landmarks.landmark):
                    if pointID not in self.__POINT_NEEDED:
                        continue
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    pointDict[self.__INDEX_DICT[pointID]] = (cx, cy)
        else:
            return None
        return pointDict, time


def getDistance(pt1, pt2):
    return math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])


class PoseJudge:
    class PoseParam:  # 一些参数，时间统一单位 s
        def __init__(self):
            self.EYE_CLOSE_RATIO = 0.5
            self.BLINK_INTERVAL_THRESH = 1

            self.MOUTH_OPEN_RATIO = 0.7
            self.YAWN_INTERVAL_THRESH = 1

            # 头部角度
            self.HEAD_PITCH_ANGLE = 15.0   # 疲劳
            self.HEAD_DOWN_INTERVAL_THRESH = 1

            self.HEAD_YAW_ANGLE = 20.0      # 分心需要根据摄像头摆放位置设定
            self.HEAD_YAW_INTERVAL_THRESH = 1

            self.HEAD_ROLL_ANGLE = 20.0     # 疲劳
            self.HEAD_SIDE_INTERVAL_THRESH = 1

            # PnP 解算参数
            self.HEAD_3D_POINTS = np.float32([[6.825897, 6.760612, 4.402142],  # 0左眉左上角
                                             [1.330353, 7.122144, 6.903745],  # 1左眉右角
                                             [-1.330353, 7.122144, 6.903745],  # 2右眉左角
                                             [-6.825897, 6.760612, 4.402142],  # 3右眉右上角
                                             [5.311432, 5.485328, 3.987654],  # 8左眼左上角
                                             [1.789930, 5.393625, 4.413414],  # 9左眼右上角
                                             [-1.789930, 5.393625, 4.413414],  # 14右眼左上角
                                             [-5.311432, 5.485328, 3.987654],  # 15右眼右上角
                                             [2.005628, 1.409845, 6.165652],  # 16鼻子左上角
                                             [-2.005628, 1.409845, 6.165652],  # 17鼻子右上角
                                             [2.774015, -2.080775, 5.048531],  # 18嘴左上角
                                             [-2.774015, -2.080775, 5.048531],  # 19嘴右上角
                                             [0.000000, -3.116408, 6.097667],  # 20嘴中央下角
                                             [0.000000, -7.415691, 4.070434]])  # 22下巴角
            self.CAMERA_MATRIX = np.float32([[6.5308391993466671e+002, 0.0, 3.1950000000000000e+002],
                                             [0.0, 6.5308391993466671e+002, 2.3950000000000000e+002],
                                             [0.0, 0.0, 1.0]])
            self.DIST = np.float32([7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000])

    def __init__(self):
        self.__param = self.PoseParam()

        self.__eye_closed = {}  # count: time
        self.__eye_closed_count = 0
        self.__mouth_opened = {}  # count: time
        self.__mouth_opened_count = 0
        self.__head_down = {}  # count: time
        self.__head_down_count = 0
        self.__head_side = {}  # count: time
        self.__head_side_count = 0
        self.__not_front = {}   # 未注视前方
        self.__not_front_count = 0

    # 判断闭眼
    def __isEyeClosed(self, pointDict):
        left_ratio = getDistance(pointDict[4], pointDict[5]) / getDistance(pointDict[6], pointDict[7])
        right_ratio = getDistance(pointDict[10], pointDict[11]) / getDistance(pointDict[12], pointDict[13])
        return left_ratio < self.__param.EYE_CLOSE_RATIO and right_ratio < self.__param.EYE_CLOSE_RATIO

    # 判断张嘴
    def __isMouthOpened(self, pointDict):
        return (getDistance(pointDict[20], pointDict[21]) / getDistance(pointDict[18],
                                                                        pointDict[19])) > self.__param.MOUTH_OPEN_RATIO

    def __get_HEAD_2D(self, pointDict):
        return np.float32([pointDict[0], pointDict[1], pointDict[2], pointDict[3],
                           pointDict[8], pointDict[9], pointDict[14], pointDict[15],
                           pointDict[16], pointDict[17], pointDict[18], pointDict[19],
                           pointDict[20], pointDict[22]])

    def getHeadState(self, pointDict):
        head_2d_pts = self.__get_HEAD_2D(pointDict)

        _, rVec, tVec = cv2.solvePnP(self.__param.HEAD_3D_POINTS, head_2d_pts, self.__param.CAMERA_MATRIX, self.__param.DIST)
        rotation_mat, _ = cv2.Rodrigues(rVec)
        pose_mat = cv2.hconcat((rotation_mat, tVec))
        _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)
        pitch, yaw, roll = [math.radians(_) for _ in euler_angle]

        pitch = math.degrees(math.asin(math.sin(pitch)))
        roll = -math.degrees(math.asin(math.sin(roll)))
        yaw = math.degrees(math.asin(math.sin(yaw)))
        # print('pitch:{}, yaw:{}, roll:{}'.format(pitch, yaw, roll))

        reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                                   [10.0, 10.0, -10.0],
                                   [10.0, -10.0, -10.0],
                                   [10.0, -10.0, 10.0],
                                   [-10.0, 10.0, 10.0],
                                   [-10.0, 10.0, -10.0],
                                   [-10.0, -10.0, -10.0],
                                   [-10.0, -10.0, 10.0]])

        reprojectdst, _ = cv2.projectPoints(reprojectsrc, rVec, tVec, self.__param.CAMERA_MATRIX, self.__param.DIST)

        reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))  # 以8行2列显示

        # 计算欧拉角calc euler angle
        # 参考https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#decomposeprojectionmatrix
        rotation_mat, _ = cv2.Rodrigues(rVec)  # 罗德里格斯公式（将旋转矩阵转换为旋转向量）
        pose_mat = cv2.hconcat((rotation_mat, tVec))  # 水平拼接，vconcat垂直拼接
        # eulerAngles –可选的三元素矢量，包含三个以度为单位的欧拉旋转角度
        _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)  # 将投影矩阵分解为旋转矩阵和相机矩阵

        return reprojectdst, euler_angle

    # 通过判断动作状态的频率来确定状态
    # 检测状态 -> 时间间隔判断 -> 存入或重置序列 -> 判断序列
    def __normal_judge_function(self, Bool, state_dict, count, time, thresh):
        if Bool:
            if count == 0:
                state_dict[count] = time
                count += 1
            elif (time - state_dict[count - 1]) / cv2.getTickFrequency() < thresh:
                state_dict[count] = time
                count += 1
            else:
                state_dict.clear()
                state_dict[0] = time
                count = 1
        elif count != 0 and (time - state_dict[count - 1]) / cv2.getTickFrequency() > thresh:
            state_dict.clear()
            count = 0

        return state_dict, count

    # 疲劳判断，最后返回可更改为字典或元组
    def judge(self, pointDict, time):
        _, pyr = self.getHeadState(pointDict)

        self.__eye_closed, self.__eye_closed_count = self.__normal_judge_function(self.__isEyeClosed(pointDict),
                                                                                  self.__eye_closed, self.__eye_closed_count,
                                                                                  time, self.__param.BLINK_INTERVAL_THRESH)
        self.__mouth_opened, self.__mouth_opened_count = self.__normal_judge_function(self.__isMouthOpened(pointDict),
                                                                                      self.__mouth_opened, self.__mouth_opened_count,
                                                                                      time, self.__param.YAWN_INTERVAL_THRESH)
        self.__head_down, self.__head_down_count = self.__normal_judge_function(pyr[0] > self.__param.HEAD_PITCH_ANGLE,
                                                                                self.__head_down, self.__head_down_count,
                                                                                time, self.__param.HEAD_DOWN_INTERVAL_THRESH)
        self.__head_side, self.__head_side_count = self.__normal_judge_function(abs(pyr[2]) > self.__param.HEAD_ROLL_ANGLE,
                                                                                self.__head_side, self.__head_side_count,
                                                                                time, self.__param.HEAD_SIDE_INTERVAL_THRESH)
        self.__not_front, self.__not_front_count = self.__normal_judge_function(abs(pyr[1]) > self.__param.HEAD_YAW_ANGLE,
                                                                                self.__not_front, self.__not_front_count,
                                                                                time, self.__param.HEAD_YAW_INTERVAL_THRESH)

        return {'eye_closed': self.__eye_closed_count > 15, 'yawn': self.__mouth_opened_count > 20,
                'head_down': self.__head_down_count > 15, 'head_side': self.__head_side_count > 15,
                'not_facing_front': self.__not_front_count > 15}


DEBUG = False