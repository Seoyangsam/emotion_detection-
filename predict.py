import tensorflow as tf
import os
import numpy as np
import cv2

from CNN_Models import *


def predict(Model, weight_path,data_path):
	model = Model
	model.load_weights(weight_path)
	num_true = 0
	num_files = 0
	for root, dirs, files in os.walk(data_path, 'r'):
		num_files += len(files)
		print('num_files:', num_files)
		for i in range(len(files)):
			file_path = root + '/' + files[i]
			print('file_path:', file_path)

			img = cv2.imread(file_path)[:, :, ::-1]
			img = (img - 127.5) / 127.5
			img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
			result = model.predict(img)
			pred = tf.argmax(result, axis=1)
			np_pred = np.array(pred)

			path_list = file_path.split('/')
			label = path_list[-1].split('-')[1][:3]
			if label == 'ang':
				label = 0
			elif label == 'dis':
				label = 1
			elif label == 'fea':
				label = 2
			elif label == 'hap':
				label = 3
			elif label == 'neu':
				label = 4
			elif label == 'sad':
				label = 5
			elif label == 'sur':
				label = 6

			# path_list = file_path.split('\\')
			# label = path_list[-1].split('-')[1][:3]
			# if label == 'ang':
			# 	label = 0
			# elif label == 'dis':
			# 	label = 1
			# elif label == 'fea':
			# 	label = 2
			# elif label == 'hap':
			# 	label = 3
			# elif label == 'neu':
			# 	label = 4
			# elif label == 'sad':
			# 	label = 5
			# elif label == 'sur':
			# 	label = 6

			if np_pred[0] == label :
				num_true += 1
			else:
				print(label, '======>', np_pred[0])

	return num_true / num_files



if __name__ == '__main__':


	model = ResNet18([2, 2, 2, 2])
	checkpoint_save_path = './checkpointResNet18/ResNet18.ckpt.index'

	data_path = './data/test_data'
	acc = predict(model, checkpoint_save_path, data_path)

	print(acc)



