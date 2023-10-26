import numpy as np
import cv2
from CNN_Models import *

# Load the model
model = ResNet18([2, 2, 2, 2])
model.load_weights('./checkpointResNet18/ResNet18.ckpt')

# Prevent OpenCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)

# Dictionary which assigns each label an emotion (alphabetical order)
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Define the video file path
vidFile = r'./video.MOV'

# Open the video file
cap = cv2.VideoCapture(vidFile)
cv2.namedWindow('video', cv2.WINDOW_NORMAL)

while True:
    # Find haar cascade to draw a bounding box around the face
    ret, frame = cap.read()
    if not ret:
        break

    # Load the face cascade classifier
    facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]

        # Preprocess the face image
        cropped_img = cv2.resize(roi_gray, (48, 48))
        cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_GRAY2BGR)  # Convert to 3 channels (assuming RGB)
        cropped_img = cropped_img.astype(np.float32) / 255.0  # Convert to float and normalize to [0, 1]
        cropped_img = np.expand_dims(cropped_img, 0)  # Add batch dimension

        # Perform the prediction
        prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))

        # Display the emotion label
        cv2.putText(frame, emotion_dict[maxindex], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)

        cv2.imshow('Video', cv2.resize(frame, (1600, 960), interpolation=cv2.INTER_CUBIC))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()