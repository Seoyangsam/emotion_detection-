import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# specify the directory where the images are located
data_dir = "path/to/cafe/images/sessions"

# create empty lists to hold the images and labels
images = []
labels = []

# create a dictionary to hold the filenames and labels
filenames = {}

# loop through the directory and add the filenames and labels to the dictionary
for file in os.listdir(data_dir):
    if file.endswith(".jpg"):
        # extract the label from the filename
        label = file.split("_")[1]
        # add the filename and label to the dictionary
        if label in filenames:
            filenames[label].append(file)
        else:
            filenames[label] = [file]

# loop through the dictionary and add the images and labels to the lists
for label, files in filenames.items():
    for file in files:
        # load the image
        img = load_img(os.path.join(data_dir, file))
        # convert the image to a numpy array
        img = img_to_array(img)
        # add the image and label to the lists
        images.append(img)
        labels.append(label)

# convert the lists to numpy arrays
images = np.array(images)
labels = np.array(labels)

# split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
