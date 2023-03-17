import cv2
import os
import random
import shutil
from Classification.CNN_Models import *
import numpy as np

def labeling(data_path,save_path,use_resize = False):
    model = ResNet18([2,2,2,2])
    model.load_weights('./checkpoint_1/ResNet18.ckpt')

    for root, dirs, files in os.walk(data_path, 'r'):
        for i in range(len(files)):
            file_path = root + '\\' + files[i]
            print('file_path:', file_path)

            img = cv2.imread(file_path)[:, :, ::-1]

            if use_resize:
                img = cv2.resize(img, (128, 128))
            img = (img - 127.5) / 127.5
            img = img.reshape(1,img.shape[0],img.shape[1],img.shape[2])
            result = model.predict(img)
            pred = tf.argmax(result, axis=1)
            np_pred = np.array(pred)
            print(np_pred[0])

            img = img.reshape(img.shape[1], img.shape[2], img.shape[3])
            img = img  * 127.5 + 127.5
            img = img[:,:,::-1]

            # cv2.imwrite(file_path[:-4] + '_' + str(np_pred[0]) + '.png',img)
            cv2.imwrite(save_path + files[i][:-4] + '_' + str(np_pred[0]) + '.png', img)

def moveFile(fileDir, trainDir):
    pathDir = os.listdir(fileDir)
    filenumber = len(pathDir)
    rate1 = 0.9
    picknumber1 = int(filenumber * rate1)
    sample1 = random.sample(pathDir, picknumber1)
    print(sample1)
    for name in sample1:
        shutil.move(fileDir + name, trainDir + "\\" + name)

if __name__ == '__main__':
    data_path = '../ProGAN/generate_picture_128'
    save_path = '../ProGAN/generate_picture_128/'
    labeling(data_path,save_path,use_resize=False)

    # fileDir = 'Dataset_test/'
    # trainDir = 'Dataset_Train'
    # moveFile(fileDir, trainDir)

