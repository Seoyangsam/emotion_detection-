import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

def make_dataset(data_path, save_path):
# all picture = []
    for root, dirs, files in os.walk(data_path, 'r'):
        for i in range(len(files)):
            file_path = root +  "\\" + files[i]
            print ('file path:',file_path)
            img = cv.imread(file_path)
            resize_img = cv.resize(img, (128, 128))

            cv.imwrite(save_path+ '\\' + files[i], resize_img)
#all picture.append (resize img[:, : ::-1])

# all picture = np.array(all picture)# print('all picture.shape:',all picture.shape)
# np.save (save path,all picture)
if __name__ == '__main__':
    path = './data'
    save_path = './data/test'
    make_dataset(path, save_path)