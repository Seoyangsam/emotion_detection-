import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
from keras.optimizers import Adam
from CNN_Models import ResNet18
from keras.losses import SparseCategoricalCrossentropy
import cv2 as cv
from keras.preprocessing.image import ImageDataGenerator


def generate_dataset(data_path, use_generate_picture=False):
    all_img = []
    all_label = []
    for root, dirs, files in os.walk(data_path, 'r'):

        for i in range(len(files)):
            print('===================')
            file_path = root + '/' + files[i]
            print('loading:', file_path)

            img = cv.imread(file_path)
            print('img.shape', img.shape)
            all_img.append(img[:, :, ::-1])

            if use_generate_picture:
                label = int(files[i][-5])
                all_label.append(label)
            else:
                path_list = file_path.split('/')
                label = path_list[-1].split('-')[1][:3]
                if label == 'ang':
                    all_label.append(0)
                elif label == 'dis':
                    all_label.append(1)
                elif label == 'fea':
                    all_label.append(2)
                elif label == 'hap':
                    all_label.append(3)
                elif label == 'neu':
                    all_label.append(4)
                elif label == 'sad':
                    all_label.append(5)
                elif label == 'sur':
                    all_label.append(6)
    x = np.array(all_img)
    y = np.array(all_label)
    print('x.shape:', x.shape)
    print('y.shape:', y.shape)
    return x, y


def train(x_train_savepath, y_train_savepath, save_weight_path, out_put_path,
          batch_size=8, epochs=300, use_data_enhance=True, use_generate_picture=False):
    if os.path.exists(x_train_savepath) and os.path.exists(y_train_savepath):
        print('-------------Load Datasets--------------')
        x_train = np.load(x_train_savepath)
        x_train = (x_train - 127.5) / 127.5
        y_train = np.load(y_train_savepath)

        print('x_train.shape:', x_train.shape)
        print('y_train.shape:', y_train.shape)

        x_train_size = x_train.shape[0]
        arr = np.arange(x_train_size)
        np.random.shuffle(arr)
        x_train = x_train[arr]
        y_train = y_train[arr]

        rate = 0.05
        x_train = x_train[:-int(rate * x_train_size)]
        y_train = y_train[:-int(rate * x_train_size)]
        print('x_train.shape:', x_train.shape)
        print('y_train.shape:', y_train.shape)
        x_test = x_train[-int(rate * x_train_size):]
        y_test = y_train[-int(rate * x_train_size):]
        print('x_test.shape:', x_test.shape)
        print('y_test.shape:', y_test.shape)

    else:
        print('--------------Generate Datasets---------------')
        data_path = './data/train_data'
        x_train, y_train = generate_dataset(data_path, use_generate_picture)
        print('--------------Save Datasets------------')
        print('x_train.shape,y_train.shape:', x_train.shape, y_train.shape)

        np.save(x_train_savepath, x_train)
        np.save(y_train_savepath, y_train)

    model = ResNet18([2, 2, 2, 2])

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss=SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['sparse_categorical_accuracy'])

    checkpoint_save_path = save_weight_path

    if os.path.exists(checkpoint_save_path + '.index'):
        print('-------------load the model-----------------')
        model.load_weights(checkpoint_save_path)

    # Save model
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                     save_weights_only=True,
                                                     save_best_only=True)
    if use_data_enhance:
        image_gen_train = ImageDataGenerator(
            rescale=1. / 1.,
            rotation_range=45,
            width_shift_range=.15,
            height_shift_range=.15,
            horizontal_flip=True,
        )
        image_gen_train.fit(x_train)

        history = model.fit(image_gen_train.flow(x_train, y_train, batch_size=batch_size), epochs=epochs,
                            validation_data=(x_test, y_test), validation_freq=1, callbacks=[cp_callback])
    else:
        history = model.fit(x_train, y_train, batch_size=2, epochs=100,
                            validation_data=(x_test, y_test), validation_freq=1, callbacks=[cp_callback])

    model.summary()

    acc = history.history['sparse_categorical_accuracy']
    val_acc = history.history['val_sparse_categorical_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(out_put_path + '/acc.png')


if __name__ == '__main__':
    x_train_savepath = 'x_train.npy'
    y_train_savepath = 'y_train.npy'
    save_weight_path = './checkpointResNet18/ResNet18.ckpt'
    out_put_path = './Ouput'
    train(x_train_savepath, y_train_savepath, save_weight_path, out_put_path, batch_size=8, epochs=30,
          use_data_enhance=True)
