import cv2 as cv
import numpy as np
import mediapipe as mp
import math


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer


mp_face_mesh = mp.solutions.face_mesh
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390,
            249, 263, 466, 388, 387, 386, 385, 384, 398]

RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154,
             155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_IRIS = [469, 470, 471, 472]
RIGHT_IRIS = [474, 475, 476, 477]

L_H_LEFT = [33]  # right eye right most landmark
L_H_RIGHT = [133]  # right eye left most landmark
R_H_LEFT = [362]  # left eye right most landmark
R_H_RIGHT = [263]  # left eye left most landmark

R_UP = [159]
R_DOWN = [145]


class Ui_MainWindow(object):

    def setupUi(self, MainWindow):
        self.frame = None
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        MainWindow.setLayoutDirection(QtCore.Qt.LeftToRight)
        # MainWindow.setAnimated(True)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # Font
        font = QtGui.QFont()
        font.setPointSize(16)
        # frame
        self.imgLabel = QtWidgets.QLabel(self.centralwidget)
        self.imgLabel.setGeometry(QtCore.QRect(150, 30, 521, 381))
        self.imgLabel.setMinimumSize(QtCore.QSize(4, 0))
        self.imgLabel.setMaximumSize(QtCore.QSize(640, 480))
        self.imgLabel.setText("")
        self.imgLabel.setObjectName("imgLabel")

        # Start Button
        self.pushButton1 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton1.setGeometry(QtCore.QRect(400, 500, 111, 41))
        self.pushButton1.setObjectName("Start camera")
        self.pushButton1.clicked.connect(self.imdisplay)

        # Stop button
        self.pushButton2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton2.setGeometry(QtCore.QRect(550, 500, 111, 41))
        self.pushButton2.setObjectName("Stop camera")
        self.pushButton2.clicked.connect(self.exit_app)

        # text
        self.text_label = QtWidgets.QLabel(self.centralwidget)
        self.text_label.setFont(font)
        self.text_label.setGeometry(QtCore.QRect(200, 500, 200, 41))
        self.text_label.setLayoutDirection(QtCore.Qt.LeftToRight)

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        self._translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(self._translate("MainWindow", "MainWindow"))
        self.pushButton1.setText(self._translate("MainWindow", "Start Cam"))
        self.pushButton2.setText(self._translate("MainWindow", "Stop Cam"))

    def imdisplay(self):
        self.capture = cv.VideoCapture(0)
        self.capture.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
        self.capture.set(cv.CAP_PROP_FRAME_WIDTH, 640)
        self.timer = QTimer(None)  # None(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(5)  # 5 milli seconds refresh

    def exit_app(self):
        app = QtWidgets.QApplication(sys.argv)
        sys.exit(app.exec_())

    def update_frame(self):
        ret, self.frame = self.capture.read()
        self.frame = cv.flip(self.frame, 1)

        # algorithm
        with mp_face_mesh.FaceMesh(max_num_faces=1,
                                   refine_landmarks=True,
                                   min_detection_confidence=0.5,
                                   min_tracking_confidence=0.5
                                   ) as face_mesh:
            self.rgb_frame = cv.cvtColor(self.frame, cv.COLOR_BGR2RGB)
            img_h, img_w = self.frame.shape[:2]

            results = face_mesh.process(self.rgb_frame)
            if results.multi_face_landmarks:
                mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(
                    int) for p in results.multi_face_landmarks[0].landmark])
            (l_cx, l_cy), l_radius = cv.minEnclosingCircle(
                mesh_points[LEFT_IRIS])
            (r_cx, r_cy), r_radius = cv.minEnclosingCircle(
                mesh_points[RIGHT_IRIS])
            center_left = np.array([l_cx, l_cy], dtype=np.int32)
            center_right = np.array([r_cx, r_cy], dtype=np.int32)
            cv.circle(self.frame, center_left, int(l_radius),
                      (255, 0, 255), 1, cv.LINE_AA)
            cv.circle(self.frame, center_right, int(r_radius),
                      (255, 0, 255), 1, cv.LINE_AA)

            iris_pos, ratio = self.iris_position(
                center_right, mesh_points[R_UP], mesh_points[R_DOWN], mesh_points[R_H_RIGHT], mesh_points[R_H_LEFT][0])

            # text shows the iris position
            self.text_label.setText(iris_pos)

            cv.putText(self.frame, f"Iris pos: {iris_pos}", (
                30, 30), cv.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 1, cv.LINE_AA)

        self.displayImage(self.frame, 1)

    def displayImage(self, img, window=1):
        qformat = QImage.Format_Indexed8
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        outImage = QImage(img, img.shape[1],
                          img.shape[0], img.strides[0], qformat)
        outImage = outImage.rgbSwapped()

        if window == 1:
            self.imgLabel.setPixmap(QPixmap.fromImage(outImage))
            self.imgLabel.setScaledContents(True)

    def euclidean_distance(self, point1, point2):
        x1, y1, = point1.ravel()
        x2, y2, = point2.ravel()
        self.distance = math.sqrt((x2 - x1)**2 + (y2-y1)**2)
        return self.distance

    def iris_position(self, iris_center, up_point, down_point, right_point, left_point):
        center_to_right_dist = self.euclidean_distance(
            iris_center, right_point)
        total_distance = self.euclidean_distance(right_point, left_point)

        ratio = center_to_right_dist/total_distance

        diff = down_point-up_point
        if diff[0][1] < 9:
            iris_position = "down"
        elif diff[0][1] > 12:
            iris_position = "up"
        elif ratio > 0.42 and ratio <= 0.57:
            iris_position = "center"
        elif ratio <= 0.42:
            iris_position = "right"
        else:
            iris_position = "left"
        ratio_vertical = diff[0][1]
        return iris_position, ratio_vertical


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
