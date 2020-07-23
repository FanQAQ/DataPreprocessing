import cv2
import shutil
import os

countA = 0
A = []
for i in range(1, 1001):
    img = cv2.imread('D:\\EMLab\\Image_processing\\IMAGE\\' + str(i) + '(2).png')
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度处理
    ret, th1 = cv2.threshold(gray_img, 254, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(th1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    count = len(contours)
    print(f'第{i}个图像的玉米粒共有：{count}个')
    if count <= 5:
        countA += 1
        A.append(i)

print(f'合格的玉米共有：{countA}个')
print(A)


for i in A:
    file = 'D:\\EMLab\\Image_processing\\IMAGE\\' + str(i) + '(2).png'
    file_dir = 'D:\\EMLab\\Image_processing\\'
    shutil.copy(file, file_dir)


current_folder = os.listdir('D:\\EMLab\\Image_processing\\')

for index, name in enumerate(current_folder):
    os.rename(name, str(index) + '.png')




























