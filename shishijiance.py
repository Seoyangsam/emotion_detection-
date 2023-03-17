import cv2
import numpy as np
import tensorflow as tf
from Classification.CNN_Models import *


def emotion_detection(Model,weight_path, size  = (128,128)):
    model = Model
    model.load_weights(weight_path)

    # 打开摄像头
    cap = cv2.VideoCapture(0)

    while True:
        # 读取视频流中的帧
        ret, frame = cap.read()

        # 将帧转换为灰度图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = frame[:,:,::-1]

        # 使用OpenCV的人脸检测器检测人脸区域
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # 对于每个检测到的人脸区域
        for (x,y,w,h) in faces:
            # 提取人脸图像区域并将其调整为模型的输入大小
            if size[0] == 128:
                face_img = rgb[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, size)
                face_img = np.reshape(face_img, (1, 128, 128, 3))
            else:
                face_img = gray[y:y + h, x:x + w]
                face_img = cv2.resize(face_img, size)
                face_img = np.reshape(face_img, (1, 48, 48, 1))

            face_img = face_img * 127.5 + 127.5

            # 使用ResNet模型对表情进行分类
            predictions = model.predict(face_img)
            print('np.argmax(predictions):',np.argmax(predictions))
            # 显示识别出的表情
            labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
            label = labels[np.argmax(predictions)]
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # 在人脸区域周围绘制边框
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # 显示帧
        cv2.imshow('Facial Expression Recognition', frame)

        # 按'q'键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # model = ResNet18([2, 2, 2, 2])
    # weight_path = './checkpoint/ResNet18.ckpt'
    # emotion_detection(model,weight_path,size=(128,128))

    model = simple_net()
    weight_path = 'model.h5'
    emotion_detection(model, weight_path, size=(48, 48))