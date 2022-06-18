import cv2
import matplotlib.pyplot as plt
import numpy as np
import preprocess as pp
from sklearn.neighbors import KNeighborsClassifier as Knn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.ensemble import BaggingClassifier

np.set_printoptions(threshold=np.inf)

X = []
X_dct = []
Y = []
X_eyes = []
X_le = []
X_re = []
X_nose = []
X_mouth = []
X_eyes_dct = []
X_le_dct = []
X_re_dct = []
X_nose_dct = []
X_mouth_dct = []

data = pp.data_

funny = 0
smiling = 0
serious = 0
other = 0

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

            img_ = temp[5][0].astype('float32')  # 将unit8类型转换为float32类型
            img_dct = cv2.dct(img_)  # 进行离散余弦变换
            img_dct_log = np.log(abs(img_dct) + 1e-5)  # 进行log 处理
            X_dct.append(img_dct_log)

            img_ = temp[5][1].astype('float32')  # 将unit8类型转换为float32类型
            img_dct = cv2.dct(img_)  # 进行离散余弦变换
            img_dct_log = np.log(abs(img_dct) + 1e-5)  # 进行log 处理
            X_eyes_dct.append(img_dct_log)

            img_ = temp[5][2].astype('float32')  # 将unit8类型转换为float32类型
            img_dct = cv2.dct(img_)  # 进行离散余弦变换
            img_dct_log = np.log(abs(img_dct) + 1e-5)  # 进行log 处理
            X_le_dct.append(img_dct_log)

            img_ = temp[5][3].astype('float32')  # 将unit8类型转换为float32类型
            img_dct = cv2.dct(img_)  # 进行离散余弦变换
            img_dct_log = np.log(abs(img_dct) + 1e-5)  # 进行log 处理
            X_re_dct.append(img_dct_log)

            img_ = temp[5][4].astype('float32')  # 将unit8类型转换为float32类型
            img_dct = cv2.dct(img_)  # 进行离散余弦变换
            img_dct_log = np.log(abs(img_dct) + 1e-5)  # 进行log 处理
            X_nose_dct.append(img_dct_log)

            img_ = temp[5][5].astype('float32')  # 将unit8类型转换为float32类型
            img_dct = cv2.dct(img_)  # 进行离散余弦变换
            img_dct_log = np.log(abs(img_dct) + 1e-5)  # 进行log 处理
            X_mouth_dct.append(img_dct_log)

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

kfold = KFold(n_splits=5, random_state=1, shuffle=True)

# img = X[0].astype('float32')  # 将unit8类型转换为float32类型
# print(img)
# img_dct = cv2.dct(img)  # 进行离散余弦变换
# img_dct_log = np.log(abs(img_dct) + 1e-5)  # 进行log 处理
# print(img_dct_log)
# # 进行离散余弦反变换
# img_recor = cv2.idct(img_dct)
# print(img_recor)
# # print(img_dct.shape)
# zip_len = 90
# # 图片压缩，只保留90*90的数据,就是图片的shape变为90*90
# recor_temp = img_dct[0:zip_len, 0:zip_len]
# # print(recor_temp.shape)
# # 建立一个143*93的0数组
# recor_temp2 = np.zeros(X[0].shape)
# # 将压缩后的离散余弦变换图覆盖0数组，也就是将压缩图像恢复为原来图像大小，其余填充为0
# recor_temp2[0:zip_len, 0:zip_len] = recor_temp
# # print(recor_temp2)
# recor_temp2_float = recor_temp2.astype('float32')
# img_recor_recor = cv2.idct(recor_temp2_float)
# # 显示
# print(img_recor_recor)
#
# plt.subplot(221)
# plt.imshow(X[0])
# plt.title('original')
# print(X[0])
#
# plt.subplot(222)
# plt.imshow(img_dct_log)
# plt.title("dct_log")
#
# plt.subplot(223)
# plt.imshow(img_recor)
# plt.title("img_recor")
#
# plt.subplot(224)
# plt.imshow(img_recor_recor)
# plt.title("img_recor_recor")
# plt.show()

x_size = len(X)
print(x_size)
X = np.array(X)
X = X.reshape(x_size, 50 * 50)
X_dct = np.array(X_dct)
X_dct = X_dct.reshape(x_size, 50 * 50)

x_eyes_size = len(X_eyes)
X_eyes = np.array(X_eyes)
X_eyes = X_eyes.reshape(x_eyes_size, 15 * 50)
X_eyes_dct = np.array(X_eyes_dct)
X_eyes_dct = X_eyes_dct.reshape(x_eyes_size, 15 * 50)

x_le_size = len(X_le)
X_le = np.array(X_le)
X_le = X_le.reshape(x_le_size, 15 * 20)
X_le_dct = np.array(X_le_dct)
X_le_dct = X_le_dct.reshape(x_le_size, 15 * 20)

x_re_size = len(X_re)
X_re = np.array(X_re)
X_re = X_re.reshape(x_re_size, 15 * 20)
X_re_dct = np.array(X_re_dct)
X_re_dct = X_re_dct.reshape(x_re_size, 15 * 20)

x_nose_size = len(X_nose)
X_nose = np.array(X_nose)
X_nose = X_nose.reshape(x_nose_size, 15 * 20)
X_nose_dct = np.array(X_nose_dct)
X_nose_dct = X_nose_dct.reshape(x_nose_size, 15 * 20)

x_mouth_size = len(X_mouth)
X_mouth = np.array(X_mouth)
X_mouth = X_mouth.reshape(x_mouth_size, 16 * 30)
X_mouth_dct = np.array(X_mouth_dct)
X_mouth_dct = X_mouth_dct.reshape(x_mouth_size, 16 * 30)

bag = Knn()

Y_pred = cross_val_predict(bag, X, Y, cv=kfold)
print("face_score:", metrics.accuracy_score(Y, Y_pred))
Y_dct_pred = cross_val_predict(bag, X_dct, Y, cv=kfold)
print("face_dct_score:", metrics.accuracy_score(Y, Y_dct_pred))

Y_eyes_pred = cross_val_predict(bag, X_eyes, Y, cv=kfold)
print("eyes_score:", metrics.accuracy_score(Y, Y_eyes_pred))
Y_eyes_dct_pred = cross_val_predict(bag, X_eyes_dct, Y, cv=kfold)
print("eyes_dct_score:", metrics.accuracy_score(Y, Y_eyes_dct_pred))

Y_le_pred = cross_val_predict(bag, X_le, Y, cv=kfold)
print("le_score:", metrics.accuracy_score(Y, Y_le_pred))
Y_le_dct_pred = cross_val_predict(bag, X_le_dct, Y, cv=kfold)
print("le_dct_score:", metrics.accuracy_score(Y, Y_le_dct_pred))

Y_re_pred = cross_val_predict(bag, X_re, Y, cv=kfold)
print("re_score:", metrics.accuracy_score(Y, Y_re_pred))
Y_re_dct_pred = cross_val_predict(bag, X_re_dct, Y, cv=kfold)
print("re_dct_score:", metrics.accuracy_score(Y, Y_re_dct_pred))

Y_nose_pred = cross_val_predict(bag, X_nose, Y, cv=kfold)
print("nose_score:", metrics.accuracy_score(Y, Y_nose_pred))
Y_nose_dct_pred = cross_val_predict(bag, X_nose_dct, Y, cv=kfold)
print("nose_dct_score:", metrics.accuracy_score(Y, Y_nose_dct_pred))

Y_mouth_pred = cross_val_predict(bag, X_mouth, Y, cv=kfold)
print("mouth_score:", metrics.accuracy_score(Y, Y_mouth_pred))
Y_mouth_dct_pred = cross_val_predict(bag, X_mouth_dct, Y, cv=kfold)
print("mouth_dct_score:", metrics.accuracy_score(Y, Y_mouth_dct_pred))

# best_score = 0
# best_k = -1
# best_weight = ""
# best_p = 1
# for k in range(1, 10):
#     clf = Knn(n_neighbors=k, weights="uniform", n_jobs=-1)
#     Y_pred = cross_val_predict(clf, X, Y, cv=kfold)
#     score = metrics.accuracy_score(Y, Y_pred)
#     print("face_score:", score)
#     if score > best_score:
#         best_score = score
#         best_k = k
#         best_weight = "uniform"
#
#     # weight==distance时
# for k in range(1, 10):
#     for p in range(1, 7):
#         clf = Knn(n_neighbors=k, weights="distance", p=p, n_jobs=-1)
#         Y_pred = cross_val_predict(clf, X, Y, cv=kfold)
#         score = metrics.accuracy_score(Y, Y_pred)
#         print("face_score:", score)
#         if score > best_score:
#             best_score = score
#             best_k = k
#             best_weight = "distance"
#             best_p = p
#
# print("the best n_neighbors", best_k)
# print("the best weights", best_weight)
# print("the best p", best_p)
# print(best_score)

# knn_best = Knn(n_neighbors=9, weights="distance", p=2)
# bagging = BaggingClassifier(base_estimator=knn_best, n_estimators=100, n_jobs=-1, random_state=1)
# Y_pred = cross_val_predict(bagging, X, Y, cv=kfold)
# print("face_score:", metrics.accuracy_score(Y, Y_pred))
# Y_mouth_pred = cross_val_predict(bagging, X_mouth, Y, cv=kfold)
# print("mouth_score:", metrics.accuracy_score(Y, Y_mouth_pred))
