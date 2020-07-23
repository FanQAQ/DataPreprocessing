import shutil
import os
from PIL import Image
from sklearn import preprocessing
import numpy as np
import cv2
# import matplotlib.pyplot as plt


Figure_dir = 'D:\\PreprocessingRadationPatten\\Image\\'
Label_dir = 'D:\\PreprocessingRadationPatten\\Label\\'
Selected_dir = 'D:\\PreprocessingRadationPatten\\Selected\\'
Split_Selected_dir = 'D:\\PreprocessingRadationPatten\\Split_Selected\\'
Selected_Label_dir = 'D:\\PreprocessingRadationPatten\\Selected_Label\\'
Split_Selected_Label_dir = 'D:\\PreprocessingRadationPatten\\Split_Selected_Label\\'
file_dir_0 = 'D:\\PreprocessingRadationPatten\\Training_Label\\'
file_dir_1 = 'D:\\PreprocessingRadationPatten\\Training_Data\\'
file_dir_2 = 'D:\\PreprocessingRadationPatten\\Training_Data\\'
new_file_dir_0 = 'D:\\PreprocessingRadationPatten\\Validation_Label\\'
new_file_dir_1 = 'D:\\PreprocessingRadationPatten\\Validation_Data\\'
new_file_dir_2 = 'D:\\PreprocessingRadationPatten\\Validation_Data\\'

Split_rate = 0.9
countA = 0
A = []
# -------------------------- 筛选数据 --------------------------------
for i in range(1, 1001):
    img = cv2.imread(Figure_dir + str(i) + '(2).png')
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, th1 = cv2.threshold(gray_img, 254, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(th1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    count = len(contours)
    print(f'第{i}个方向图的主瓣共有：{count}个')
    if count <= 5:
        countA += 1
        A.append(i)
print(f'符合条件的方向图共有：{countA}个')
print(A)

# -------------------------- 转移筛选出的数据 --------------------------------
for i in A:
    file_0 = Label_dir + str(i) + '.txt'
    file_1 = Figure_dir + str(i) + '(1).png'
    file_2 = Figure_dir + str(i) + '(2).png'
    shutil.copy(file_0, Selected_Label_dir)
    shutil.copy(file_1, Selected_dir)
    shutil.copy(file_2, Selected_dir)

# -------------------------- 重新将筛选出的数据排序 --------------------------------
current_folder = os.listdir(Selected_Label_dir)

for index, name in enumerate(current_folder):
    os.rename(Selected_Label_dir + name[:-4] + '.txt', Selected_Label_dir + str(index + 100000) + '.txt')
    os.rename(Selected_dir + name[:-4] + '(1).png', Selected_dir + str(index + 100000) + '(1).png')
    os.rename(Selected_dir + name[:-4] + '(2).png', Selected_dir + str(index + 100000) + '(2).png')

current_folder = os.listdir(Selected_Label_dir)

for index, name in enumerate(current_folder):
    os.rename(Selected_Label_dir + name[:-4] + '.txt', Selected_Label_dir + str(index) + '.txt')
    os.rename(Selected_dir + name[:-4] + '(1).png', Selected_dir + str(index) + '(1).png')
    os.rename(Selected_dir + name[:-4] + '(2).png', Selected_dir + str(index) + '(2).png')

# -------------------------- 划分训练集、测试集 --------------------------------
tra_num = len(current_folder) * Split_rate
for index, value in enumerate(range(int(tra_num), len(current_folder))):
    shutil.move(Selected_Label_dir + str(value) + '.txt', Split_Selected_Label_dir + str(index) + '.txt')
    shutil.move(Selected_dir + str(value) + '(1).png', Split_Selected_dir + str(index) + '(1).png')
    shutil.move(Selected_dir + str(value) + '(2).png', Split_Selected_dir + str(index) + '(2).png')

# -------------------------- 保存为NPY文件 --------------------------------
Training_data_folder = os.listdir(Selected_dir)
Training_label_folder = os.listdir(Selected_Label_dir)
Validation_data_folder = os.listdir(Split_Selected_dir)
Validation_label_folder = os.listdir(Split_Selected_Label_dir)
Folder = [Training_data_folder, Training_label_folder, Validation_data_folder, Validation_label_folder]
DIR = [Selected_dir, Selected_Label_dir, Split_Selected_dir, Split_Selected_Label_dir]
Dir = [file_dir_1, file_dir_0, new_file_dir_1, new_file_dir_0]
lb = preprocessing.LabelBinarizer()

countF = 0
for folder in Folder:
    countF += 1
    A = lambda aa: len(aa) if countF % 2 == 0 else len(aa)/2
    print(int(A(folder)))
    for num in range((int(A(folder)))):
        if countF % 2 == 1:
            img1 = np.array(Image.open(DIR[countF - 1] + str(num) + '(1).png'))
            img1 = img1[:, :, np.newaxis]
            img2 = np.array(Image.open(DIR[countF - 1] + str(num) + '(2).png'))
            img2 = img2[:, :, np.newaxis]
            img = np.concatenate((img1, img2), axis=-1).reshape((100, 100, 2))
            np.save(Dir[countF - 1] + str(num) + '.npy', img)
        if countF % 2 == 0:
            label = np.loadtxt(DIR[countF - 1] + str(num) + '.txt')
            lb.fit(label)
            labels = lb.transform(label)
            try:
                labels = labels.reshape(100, 4)
            except Exception:
                F = np.zeros((100, 1))
                labels = np.concatenate((F, labels), axis=1)
            np.save(Dir[countF - 1] + str(num) + '.npy', labels)












