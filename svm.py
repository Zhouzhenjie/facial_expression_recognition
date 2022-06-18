import os
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score

# 进行图片的分割
def splitSenses(img):
    img_eyes = img[50:65, 45:95]
    img_leye = img[50:65, 45:65]
    img_reye = img[50:65, 75:95]
    img_nose = img[65:80, 60:80]
    img_mouth = img[82:98, 55:85]
    img_face2 = img[50:100,45:95]

    # 0：原图 1：眼睛 2：左眼 3：右眼 4：鼻子 5：嘴巴 6:抠全脸
    img_ = [img, img_eyes, img_leye, img_reye, img_nose, img_mouth,img_face2]
    #打印图片
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
    return img_

# 获取图片
def getImge(file_path, id_):
    img_path = "/rawdata/" + str(id_ + 1223)
    f = open(file_path + img_path, "rb")
    img = np.fromfile(f, dtype=np.ubyte)

    if img.size == 128 * 128:
        img = img.reshape([128, 128])

        # 标准化处理
        sc = StandardScaler()
        sc.fit(img)
        img = sc.transform(img)

        #  # 归一化处理
        # mm = make_pipeline(MinMaxScaler(feature_range=(0,1)), Normalizer())
        # img = mm.fit_transform(img)

        img = splitSenses(img)
        # plt.imshow(img)
        # plt.show()
    else:
        if img.size == 512 * 512:
            # print("512:", id_)
            img = img.reshape([512, 512])
            return 0
        else:
            # print("no_img:", id_)
            return 0
    # plt.imshow(img)
    # plt.show()

    return img

# 读取标签
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
            # print("miss:", i)
    return data

np.set_printoptions(threshold=np.inf)

# 存储各种图像
X = []        #原始图片
Y = []        #标签
X_eyes = []   #眼睛
X_le = []     #左眼
X_re = []     #右眼
X_nose = []    #鼻子
X_mouth = []   #嘴巴
X_face2 = []   #处理后的全脸
X_lbp = []    #特征提取后结果
X_eyes_lbp = []   #
X_le_lbp = []    #
X_re_lbp = []    #
X_nose_lbp = []   #
X_mouth_lbp = []   #
X_face2_lbp = []   #

# 获取图像数据
data = loadData(os.getcwd())

# 统计各表情的个数
print('统计各表情的个数')
funny = 0
smiling = 0
serious = 0
other = 0

# 进行表情统计
for i in range(len(data)):
    temp = data[i]
    if temp != 0:
        if temp[5] != 0:
            X.append(temp[5][0])
            X_eyes.append(temp[5][1])
            X_le.append(temp[5][2])
            X_re.append(temp[5][3])
            X_nose.append(temp[5][4])
            X_mouth.append(temp[5][5])
            X_face2.append(temp[5][6])

            if temp[3] == "funny":
                Y.append(0)
                funny += 1
            elif temp[3] == "smiling":
                Y.append(0)
                smiling += 1
            elif temp[3] == "serious":
                Y.append(1)
                serious += 1
            else:
                Y.append(3)
                other += 1

print("funny", funny / len(Y))
print("smiling", smiling / len(Y))
print("serious", serious / len(Y))
print("other", other / len(Y))
#######################################################
# #取部分数据测试
X1 = X[:]
X2 = X_eyes[:]
X3 = X_le[:]
X4 = X_re[:]
X5 = X_nose[:]
X6 = X_mouth[:]
X7 = X_face2[:]
Y1 = Y[:]

##数据降维
x_size = len(X1)
X1 = np.array(X1)
X1 = X1.reshape(x_size, 128*128)

x_eyes_size = len(X2)
X2 = np.array(X2)
X2 = X2.reshape(x_eyes_size, 15 * 50)
# X_eyes_dct = np.array(X_eyes_dct)
# X_eyes_dct = X_eyes_dct.reshape(x_eyes_size, 15 * 50)

x_le_size = len(X3)
X3 = np.array(X3)
X3 = X3.reshape(x_le_size, 15 * 20)
# X_le_dct = np.array(X_le_dct)
# X_le_dct = X_le_dct.reshape(x_le_size, 15 * 20)

x_re_size = len(X4)
X4 = np.array(X4)
X4 = X4.reshape(x_re_size, 15 * 20)
# X_re_dct = np.array(X_re_dct)
# X_re_dct = X_re_dct.reshape(x_re_size, 15 * 20)

x_nose_size = len(X5)
X5 = np.array(X5)
X5 = X5.reshape(x_nose_size, 15 * 20)
# X_nose_dct = np.array(X_nose_dct)
# X_nose_dct = X_nose_dct.reshape(x_nose_size, 15 * 20)

x_mouth_size = len(X6)
X6 = np.array(X6)
X6 = X6.reshape(x_mouth_size, 16 * 30)
# X_mouth_dct = np.array(X_mouth_dct)
# X_mouth_dct = X_mouth_dct.reshape(x_mouth_size, 16 * 30)

x_face2_size = len(X7)
X7 = np.array(X7)
X7 = X7.reshape(x_face2_size, 50 * 50)
# X_mouth_dct = np.array(X_mouth_dct)
# X_mouth_dct = X_mouth_dct.reshape(x_mouth_size, 16 * 30)
# #################################################################
#SVM算法分类和交叉验证
from sklearn.svm import SVC
svc = SVC(kernel='rbf', C=1E6)

print('svm分类准确率')

Y_pred = cross_val_predict(svc, X1, Y1, cv=5)
print("face_score:", accuracy_score(Y1, Y_pred))

Y_pred = cross_val_predict(svc, X2, Y1, cv=5)
print("eyes_score:", accuracy_score(Y1, Y_pred))

Y_pred = cross_val_predict(svc, X3, Y1, cv=5)
print("le_score:", accuracy_score(Y1, Y_pred))

Y_pred = cross_val_predict(svc, X4, Y1, cv=5)
print("re_score:", accuracy_score(Y1, Y_pred))

Y_pred = cross_val_predict(svc, X5, Y1, cv=5)
print("nose_score:", accuracy_score(Y1, Y_pred))

Y_pred = cross_val_predict(svc, X6, Y1, cv=5)
print("mouth_score:", accuracy_score(Y1, Y_pred))

Y_pred = cross_val_predict(svc, X7, Y1, cv=5)
print("face2_score:", accuracy_score(Y1, Y_pred))
################################################################
#圆LBP算法特征提取
def yuan_LBP(img, r=3, p=8):
    h, w = img.shape
    dst = np.zeros((h, w), dtype=img.dtype)
    for i in range(r, h - r):
        for j in range(r, w - r):
            LBP_str = []
            for k in range(p):
                rx = i + r * np.cos(2 * np.pi * k / p)
                ry = j - r * np.sin(2 * np.pi * k / p)
                # print(rx, ry)
                x0 = int(np.floor(rx))
                x1 = int(np.ceil(rx))
                y0 = int(np.floor(ry))
                y1 = int(np.ceil(ry))

                f00 = img[x0, y0]
                f01 = img[x0, y1]
                f10 = img[x1, y0]
                f11 = img[x1, y1]
                w1 = x1 - rx
                w2 = rx - x0
                w3 = y1 - ry
                w4 = ry - y0
                fxy = w3 * (w1 * f00 + w2 * f10) + w4 * (w1 * f01 + w2 * f11)
                if fxy >= img[i, j]:
                    LBP_str.append(1)
                else:
                    LBP_str.append(0)
            LBP_str = ''.join('%s' % id for id in LBP_str)
            dst[i, j] = int(LBP_str, 2)
    return dst

#LBP特征提取
def extract_lbp_features(x):
    image_descriptions = []  # 用来存储LBP处理后的数据

    for i in range(len(x)):
        result = yuan_LBP(x[i], r=3, p=8)

        result = np.array(x[i])

        image_descriptions.append(result)
        # print(result.shape)

    return image_descriptions
################################################################
# #重新取部分值测试
X11 = X[:]
X22 = X_eyes[:]
X33 = X_le[:]
X44 = X_re[:]
X55 = X_nose[:]
X66 = X_mouth[:]
X77 = X_face2[:]

#进行LBP特征提取并降维
x_lbp_size = len(X11)
X_lbp = extract_lbp_features(X11)
X_lbp = np.array(X_lbp)
# print(X_lbp.shape)
X_lbp = X_lbp.reshape(x_lbp_size, 128*128)

x_eyes_size = len(X22)
X_eyes_lbp = extract_lbp_features(X22)
X_eyes_lbp = np.array(X_eyes_lbp)
# print(X_lbp.shape)
X_eyes_lbp = X_eyes_lbp.reshape(x_eyes_size, 15 * 50)

x_le_size = len(X33)
X_le_lbp = extract_lbp_features(X33)
X_le_lbp = np.array(X_le_lbp)
# print(X_lbp.shape)
X_le_lbp = X_le_lbp.reshape(x_le_size, 15 * 20)

x_re_size = len(X44)
X_re_lbp = extract_lbp_features(X44)
X_re_lbp = np.array(X_re_lbp)
# print(X_lbp.shape)
X_re_lbp = X_re_lbp.reshape(x_re_size, 15 * 20)

x_nose_size = len(X55)
X_nose_lbp = extract_lbp_features(X55)
X_nose_lbp = np.array(X_nose_lbp)
# print(X_lbp.shape)
X_nose_lbp = X_nose_lbp.reshape(x_nose_size, 15 * 20)

x_mouth_size = len(X66)
X_mouth_lbp = extract_lbp_features(X66)
X_mouth_lbp = np.array(X_mouth_lbp)
# print(X_lbp.shape)
X_mouth_lbp = X_mouth_lbp.reshape(x_mouth_size, 16*30)

x_face2_size = len(X77)
X_face2_lbp = extract_lbp_features(X77)
X_face2_lbp = np.array(X_face2_lbp)
# print(X_lbp.shape)
X_face2_lbp = X_face2_lbp.reshape(x_face2_size, 50*50)
################################################################################
#特征提取后再SVM分类

print('LBP特征提取后再SVM分类')

Y_pred = cross_val_predict(svc, X_lbp, Y1, cv=5)
print("face_lbp_score:", accuracy_score(Y1, Y_pred))

Y_pred = cross_val_predict(svc, X_eyes_lbp, Y1, cv=5)
print("eyes_lbp_score:", accuracy_score(Y1, Y_pred))

Y_pred = cross_val_predict(svc, X_le_lbp, Y1, cv=5)
print("le_lbp_score:", accuracy_score(Y1, Y_pred))

Y_pred = cross_val_predict(svc, X_re_lbp, Y1, cv=5)
print("re_lbp_score:", accuracy_score(Y1, Y_pred))

Y_pred = cross_val_predict(svc, X_nose_lbp, Y1, cv=5)
print("nose_lbp_score:", accuracy_score(Y1, Y_pred))

Y_pred = cross_val_predict(svc, X_mouth_lbp, Y1, cv=5)
print("mouth_lbp_score:", accuracy_score(Y1, Y_pred))

Y_pred = cross_val_predict(svc, X_face2_lbp, Y1, cv=5)
print("face2_lbp_score:", accuracy_score(Y1, Y_pred))

#集成分类器，mouth分类准确率最高，这里只进行mouth和face2测试
from sklearn.ensemble import BaggingClassifier

print('确定max_features')
#确定max_features
k = 0.8
best_score = 0
best_max_features = 0
for i in range(1,21):
    clf = BaggingClassifier(svc, n_estimators=10, max_samples=0.5, max_features=k, n_jobs=-1, random_state=42)
    k += 0.01
    Y_pred = cross_val_predict(clf, X6, Y1, cv=5)
    scores = cross_val_score(clf, X6, Y1, cv=5, scoring="accuracy")
    print("k=",k, "scores=",scores.mean())
    if scores.mean() > best_score:
        best_score = scores.mean()
        best_max_features = k

print("best_score=", best_score) #0.807
print("best_max_features=", best_max_features) #k=0.93

print('确定max_samples')
# 确定max_samples
k = 0.4
best_score = 0
best_max_samples = 0
for i in range(1,15):
    clf = BaggingClassifier(svc, n_estimators=10, max_samples=k, max_features=0.92, n_jobs=-1, random_state=42)
    k += 0.01
    Y_pred = cross_val_predict(clf, X6, Y1, cv=5)
    scores = cross_val_score(clf, X6, Y1, cv=5, scoring="accuracy")
    print("k=",k, "scores=",scores.mean())
    if scores.mean() > best_score:
        best_score = scores.mean()
        best_max_samples = k

print("best_score=", best_score) #0.81
print("best_max_samples=", best_max_samples) #k=0.42

print('集成分类器准确率 ')

clf = BaggingClassifier(svc, n_estimators=10, max_samples=0.42, max_features=0.93, n_jobs=-1, random_state=42)
Y_mouth_pred = cross_val_predict(clf, X6, Y1, cv=5)
print("mouth_Bagging_score:", accuracy_score(Y1, Y_mouth_pred))
#
clf = BaggingClassifier(svc, n_estimators=10, max_samples=0.42, max_features=0.93, n_jobs=-1, random_state=42)
Y_mouth_pred = cross_val_predict(clf, X_mouth_lbp, Y1, cv=5)
print("mouth_lbp_Bagging_score:", accuracy_score(Y1, Y_mouth_pred))
#
clf = BaggingClassifier(svc, n_estimators=10, max_samples=0.42, max_features=0.93, n_jobs=-1, random_state=42)
Y_mouth_pred = cross_val_predict(clf, X7, Y1, cv=5)
print("face2_Bagging_score:", accuracy_score(Y1, Y_mouth_pred))
#
clf = BaggingClassifier(svc, n_estimators=10, max_samples=0.42, max_features=0.93, n_jobs=-1, random_state=42)
Y_mouth_pred = cross_val_predict(clf, X_face2_lbp, Y1, cv=5)
print("face2_lbp_Bagging_score:", accuracy_score(Y1, Y_mouth_pred))
#
# clf = BaggingClassifier(svc, n_estimators=10, max_samples=0.5, max_features=0.93, n_jobs=-1, random_state=42)
# Y_mouth_pred = cross_val_predict(clf, X5, Y1, cv=5)
# print("mouth_Bagging_score:", accuracy_score(Y1, Y_mouth_pred))
#
# clf = BaggingClassifier(svc, n_estimators=10, max_samples=0.5, max_features=0.93, n_jobs=-1, random_state=42)
# Y_mouth_pred = cross_val_predict(clf, X6, Y1, cv=5)
# print("mouth_Bagging_score:", accuracy_score(Y1, Y_mouth_pred))
#
# clf = BaggingClassifier(svc, n_estimators=10, max_samples=0.5, max_features=0.93, n_jobs=-1, random_state=42)
# Y_mouth_pred = cross_val_predict(clf, X_mouth_lbp, Y1, cv=5)
# print("mouth_lbp_Bagging_score:", accuracy_score(Y1, Y_mouth_pred))
#
# clf = BaggingClassifier(svc, n_estimators=10, max_samples=0.5, max_features=0.93, n_jobs=-1, random_state=42)
# Y_mouth_pred = cross_val_predict(clf, X_mouth_lbp, Y1, cv=5)
# print("mouth_lbp_Bagging_score:", accuracy_score(Y1, Y_mouth_pred))
#
# clf = BaggingClassifier(svc, n_estimators=10, max_samples=0.5, max_features=0.93, n_jobs=-1, random_state=42)
# Y_mouth_pred = cross_val_predict(clf, X_mouth_lbp, Y1, cv=5)
# print("mouth_lbp_Bagging_score:", accuracy_score(Y1, Y_mouth_pred))
#
# clf = BaggingClassifier(svc, n_estimators=10, max_samples=0.5, max_features=0.93, n_jobs=-1, random_state=42)
# Y_mouth_pred = cross_val_predict(clf, X_mouth_lbp, Y1, cv=5)
# print("mouth_lbp_Bagging_score:", accuracy_score(Y1, Y_mouth_pred))
#
# clf = BaggingClassifier(svc, n_estimators=10, max_samples=0.5, max_features=0.93, n_jobs=-1, random_state=42)
# Y_mouth_pred = cross_val_predict(clf, X_mouth_lbp, Y1, cv=5)
# print("mouth_lbp_Bagging_score:", accuracy_score(Y1, Y_mouth_pred))
#
# clf = BaggingClassifier(svc, n_estimators=10, max_samples=0.5, max_features=0.93, n_jobs=-1, random_state=42)
# Y_mouth_pred = cross_val_predict(clf, X_mouth_lbp, Y1, cv=5)
# print("mouth_lbp_Bagging_score:", accuracy_score(Y1, Y_mouth_pred))
