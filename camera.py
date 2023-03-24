import cv2
import numpy as np
from CNN_Models import *


def emotion_detection(Model, weight_path, size=(128, 128)):
    model = Model
    model.load_weights(weight_path)

    # open camera
    cap = cv2.VideoCapture(0)

    while True:
        # read
        ret, frame = cap.read()

        # convert into gray images
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = frame[:, :, ::-1]

        # detect face
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # For each detected face region extract the face image region and resize it to the input size of the model
        for (x, y, w, h) in faces:
            if size[0] == 128:
                face_img = rgb[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, size)
                face_img = np.reshape(face_img, (1, 128, 128, 3))
            else:
                face_img = gray[y:y + h, x:x + w]
                face_img = cv2.resize(face_img, size)
                face_img = np.reshape(face_img, (1, 48, 48, 1))

            face_img = face_img * 127.5 + 127.5

            # classify via ResNet model
            predictions = model.predict(face_img)
            print('np.argmax(predictions):', np.argmax(predictions))
            # print emotions
            labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
            label = labels[np.argmax(predictions)]
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Draw a border around the face area
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # display frame
        cv2.imshow('Facial Expression Recognition', frame)

        # press q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # release
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    model = ResNet18([2, 2, 2, 2])
    weight_path = './checkpoint/ResNet18.ckpt'
    emotion_detection(model, weight_path, size=(128, 128))

    # model = simple_net()
    # weight_path = 'model.h5'
    # emotion_detection(model, weight_path, size=(48, 48))
