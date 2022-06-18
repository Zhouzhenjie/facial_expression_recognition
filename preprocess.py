import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.preprocessing import StandardScaler


def splitSenses(img):
    # plt.imshow(img)
    # plt.show()
    img_eyes = img[50:65, 45:95]
    img_leye = img[50:65, 45:65]
    img_reye = img[50:65, 75:95]
    img_nose = img[65:80, 60:80]
    img_mouth = img[82:98, 55:85]
    img = img[50:100, 45:95]
    img_ = [img, img_eyes, img_leye, img_reye, img_nose, img_mouth]
    # plt.imshow(img_eyes)
    # plt.show()
    # plt.imshow(img_leye)
    # plt.show()
    # plt.imshow(img_reye)
    # plt.show()
    # plt.imshow(img_nose)
    # plt.show()
    # plt.imshow(img_mouth)
    # plt.show()
    # plt.imshow(img)
    # plt.show()
    return img_


def getImge(file_path, id_):
    img_path = "/photo/" + str(id_ + 1223)
    f = open(file_path + img_path, "rb")
    img = np.fromfile(f, dtype=np.ubyte)
    if img.size == 128 * 128:
        img = img.reshape([128, 128])
        # plt.imshow(img)
        # plt.show()
        # sc = StandardScaler()
        # sc.fit(img)
        img = splitSenses(img)

    else:
        if img.size == 512 * 512:
            print("512:", id_)
            img = img.reshape([512, 512])
            return 0
        else:
            print("no_img:", id_)
            return 0
    # plt.imshow(img)
    # plt.show()
    return img


def loadData(file_path):
    label_path = "/label.txt"
    f = open(file_path + label_path, "r")
    label = f.read().splitlines()
    data = []
    for i in range(len(label)):
        temp = label[i].split()
        # print(temp)
        if len(temp) > 10:
            sex = temp[temp.index("(_sex") + 1]
            sex = sex.strip(")")
            age = temp[temp.index("(_age") + 1]
            age = age.rstrip(")")
            race = temp[temp.index("(_race") + 1]
            race = race.rstrip(")")
            face = temp[temp.index("(_face") + 1]
            face = face.rstrip(")")
            prop = temp[(temp.index("(_prop") + 1):]
            if len(prop) > 1:
                prop[0] = prop[0][2:]
                prop.pop(-1)
            else:
                prop = 0
            img = getImge(file_path, i)
            data.append([sex, age, race, face, prop, img])
        else:
            data.append(0)
            print("miss:", i)
    return data


data_ = loadData("/home/jg/FR/face")
# print(data_[5])
# plt.imshow(data_[1225 - 1223][5][4])
# plt.show()

img = cv2.imread("/home/jg/image_left.png")
img2 = cv2.imread("/home/jg/image_right.png")
img3 = img[0:1080, 0:1200]
img4 = img2[0:1080, 240:1440]
plt.subplot(221)
plt.imshow(img)
plt.subplot(222)
plt.imshow(img2)
plt.subplot(223)
plt.imshow(img3)
plt.subplot(224)
plt.imshow(img4)
plt.show()
