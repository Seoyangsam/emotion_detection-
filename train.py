import os
import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
image_dir = "./data"
valid_extensions = ['.jpg']
folders = os.listdir(image_dir)
image_paths = []

for folder in folders:
    folder_path = os.path.join(image_dir, folder)
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            ext = os.path.splitext(file)[1]
            if ext.lower() in valid_extensions:
                image_paths.append(os.path.join(folder_path, file))

# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for image_path in image_paths:
        # Read the image
        image = cv2.imread(image_path)

        # Check if the image is loaded correctly.
        if image is None:
            raise ValueError('Failed to load the input image.')

        # Make Detections
        results = holistic.process(image)

        # Draw landmarks
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                                  mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                  mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                                  )

        cv2.imshow('Image', image)

        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
import csv
import os
import numpy as np

num_coords = len(results.face_landmarks.landmark)
landmarks = ['class']
for val in range(1, num_coords+1):
    landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]

with open('coords.csv', mode='w', newline='') as f:
    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(landmarks)

class_name = "angry"

image_dir = "./data/train_data/angry"
valid_extensions = ['.jpg']
folders = os.listdir(image_dir)
image_paths = []

for folder in folders:
    folder_path = os.path.join(image_dir, folder)
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            ext = os.path.splitext(file)[1]
            if ext.lower() in valid_extensions:
                image_paths.append(os.path.join(folder_path, file))

# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for image_path in image_paths:
        # Read the image
        image = cv2.imread(image_path)

        # Check if the image is loaded correctly.
        if image is None:
            raise ValueError('Failed to load the input image.')

        # Make Detections
        results = holistic.process(image)

        # Draw landmarks
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                                  mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                  mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                                  )

        cv2.imshow('Image', image)

        if cv2.waitKey(0) & 0xFF == ord('q'):
            break