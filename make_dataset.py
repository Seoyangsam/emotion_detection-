
import cv2 as cv
import os
import shutil
import random

# img = cv.imread('D:\PycharmProjects\A_Jiedan\picture\sessions\\6280\9977-angry_F-AA-01.jpg')
# resize_img = cv.resize(img,(128,128))
# plt.figure(1)
# plt.imshow(img[:, :, ::-1], )
# plt.figure(2)
# plt.imshow(resize_img[:, :, ::-1], )
# plt.show()


def make_dataset(source_folder_path,destination_folder_path, train_path,test_path):
	source_folder = source_folder_path
	destination_folder = destination_folder_path
	if not os.path.exists(destination_folder):
		os.makedirs(destination_folder)
	for folder_name in os.listdir(source_folder):
		folder_path = os.path.join(source_folder, folder_name)
		if os.path.isdir(folder_path):
			target_folder_path = os.path.join(destination_folder, folder_name)
			shutil.move(folder_path, target_folder_path)

	# all_picture = []
	if not os.path.exists(train_path):
		os.makedirs(train_path)
	for root, dirs, files in os.walk(destination_folder, 'r'):
		for i in range(len(files)):
			file_path = root + '\\' + files[i]
			print('file_path:',file_path)
			img = cv.imread(file_path)
			resize_img = cv.resize(img,(128,128))
			#
			cv.imwrite(train_path+'\\'+files[i],resize_img)

	# 		all_picture.append(resize_img[:, :, ::-1])
	#
	# all_picture = np.array(all_picture)
	# print('all_picture.shape:',all_picture.shape)
	# np.save(save_path,all_picture)
	if not os.path.exists(test_path):
		os.makedirs(test_path)
	pathDir = os.listdir(train_path)
	filenumber = len(pathDir)
	rate1 = 0.1
	picknumber1 = int(filenumber * rate1)
	sample1 = random.sample(pathDir, picknumber1)
	print(sample1)
	for name in sample1:
		shutil.move(train_path + name, test_path + "\\" + name)

if __name__ =='__main__':
	source_folder_path = './data'
	destination_folder_path = './all_datasets'
	train_path = './data/train'
	test_path = './data/test'
	make_dataset(source_folder_path,destination_folder_path,train_path,test_path)
