import cv2
import os

# Define the desired image size
new_size = (256, 256)

# Loop through the two main folders (train_data and test_data)
for main_folder in ["train_data", "test_data"]:
    # Loop through the subfolders (angry, disgust, fearful, happy, neutral, sad, surprise)
    for emotion_folder in ["angry", "disgust", "fearful", "happy", "neutral", "sad", "surprise"]:
        # Build the path to the current subfolder
        current_folder = os.path.join("~/PycharmProjects/emotion_detection-", main_folder, emotion_folder)

        # Loop through the images in the current subfolder
        for filename in os.listdir(current_folder):
            # Read in the current image
            img = cv2.imread(os.path.join(current_folder, filename))

            # Resize the image to the desired size
            resized_img = cv2.resize(img, new_size)

            # Build the path to the resized image
            resized_filename = os.path.join(current_folder, "resized_" + filename)

            # Save the resized image
            cv2.imwrite(resized_filename, resized_img)
