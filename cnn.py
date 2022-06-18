import numpy as np
import cv2
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import time
import preprocess as pp
from sklearn.model_selection import train_test_split

torch.set_num_threads(12)

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

            # img_ = temp[5][0].astype('float32')  # 将unit8类型转换为float32类型
            # img_dct = cv2.dct(img_)  # 进行离散余弦变换
            # img_dct_log = np.log(abs(img_dct) + 1e-5)  # 进行log 处理
            # X_dct.append(img_dct_log)
            #
            # img_ = temp[5][1].astype('float32')  # 将unit8类型转换为float32类型
            # img_dct = cv2.dct(img_)  # 进行离散余弦变换
            # img_dct_log = np.log(abs(img_dct) + 1e-5)  # 进行log 处理
            # X_eyes_dct.append(img_dct_log)
            #
            # img_ = temp[5][2].astype('float32')  # 将unit8类型转换为float32类型
            # img_dct = cv2.dct(img_)  # 进行离散余弦变换
            # img_dct_log = np.log(abs(img_dct) + 1e-5)  # 进行log 处理
            # X_le_dct.append(img_dct_log)
            #
            # img_ = temp[5][3].astype('float32')  # 将unit8类型转换为float32类型
            # img_dct = cv2.dct(img_)  # 进行离散余弦变换
            # img_dct_log = np.log(abs(img_dct) + 1e-5)  # 进行log 处理
            # X_re_dct.append(img_dct_log)
            #
            # img_ = temp[5][4].astype('float32')  # 将unit8类型转换为float32类型
            # img_dct = cv2.dct(img_)  # 进行离散余弦变换
            # img_dct_log = np.log(abs(img_dct) + 1e-5)  # 进行log 处理
            # X_nose_dct.append(img_dct_log)
            #
            # img_ = temp[5][5].astype('float32')  # 将unit8类型转换为float32类型
            # img_dct = cv2.dct(img_)  # 进行离散余弦变换
            # img_dct_log = np.log(abs(img_dct) + 1e-5)  # 进行log 处理
            # X_mouth_dct.append(img_dct_log)

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

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=1)


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input 维度 [1, 128, 128]
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [64, 64, 64]

            nn.Conv2d(64, 128, 3, 1, 1),  # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [128, 32, 32]

            nn.Conv2d(128, 256, 3, 1, 1),  # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [256, 16, 16]

            nn.Conv2d(256, 512, 3, 1, 1),  # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [512, 8, 8]

            nn.Conv2d(512, 512, 3, 1, 1),  # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [512, 4, 4]
        )
        self.fc = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 11)

        )

    def forward(self, x):
        out = self.cnn(x)
        print(out.shape)
        out = out.view(out.size()[0], -1)
        print(out.shape)
        return self.fc(out)


class ImgDataset(Dataset):  # 父类继承,注意继承dataset父类必须重写getitem,len否则报错.
    # 初始化
    def __init__(self, x, y=None):
        self.x = x
        # label is required to be a LongTensor
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)

    # 读取某幅图片, item为索引号
    def __getitem__(self, index):
        # 读取单通道灰度图
        img_gray = self.x[index]

        # 高斯模糊
        # img_Gus = cv2.GaussianBlur(img_gray, (3,3), 0)

        # 直方图均衡化
        # img_hist = cv2.equalizeHist(img_gray)

        # 像素值标准化,0-255的像素范围转成0-1范围来描述
        img = np.array(img_gray)
        img = img.reshape(1, 50, 50)

        # 用于训练的数据需要为tensor类型
        img_tensor = torch.from_numpy(img)  # 将numpy中的ndarray转换成pytorch中的tensor
        img_tensor = img_tensor.type('torch.FloatTensor')  # Tensor转FloatTensor
        if self.y is not None:
            label = self.y[index]
            return img_tensor, label
        else:
            return img_tensor

    # 获取数据集样本个数
    def __len__(self):
        return len(self.x)


batch_size = 32
train_set = ImgDataset(X_train, Y_train)
val_set = ImgDataset(X_val, Y_val)
test_set = ImgDataset(X_test)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
print("Dataset complicated")

print("Training")
print("...")
# 使用training set訓練，並使用validation set尋找好的參數
model = Classifier()
loss = nn.CrossEntropyLoss()  # 因為是 classification task，所以 loss 使用 CrossEntropyLoss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # optimizer 使用 Adam
num_epoch = 30  # 迭代30次

for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    model.train()  # 確保 model 是在 train model (開啟 Dropout 等...)
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()  # 用 optimizer 將 model 參數的 gradient 歸零
        train_pred = model(data[0])  # 利用 model 得到預測的機率分佈 這邊實際上就是去呼叫 model 的 forward 函數
        batch_loss = loss(train_pred, data[1])  # 計算 loss （注意 prediction 跟 label 必須同時在 CPU 或是 GPU 上）
        batch_loss.backward()  # 利用 back propagation 算出每個參數的 gradient
        optimizer.step()  # 以 optimizer 用 gradient 更新參數值

        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        train_loss += batch_loss.item()

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            val_pred = model(data[0])
            batch_loss = loss(val_pred, data[1])

            val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            val_loss += batch_loss.item()

        # 將結果 print 出來
        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
              (epoch + 1, num_epoch, time.time() - epoch_start_time, \
               train_acc / train_set.__len__(), train_loss / train_set.__len__(), val_acc / val_set.__len__(),
               val_loss / val_set.__len__()))

train_val_x = np.concatenate((X_train, X_val), axis=0)
train_val_y = np.concatenate((Y_train, Y_val), axis=0)
train_val_set = ImgDataset(train_val_x, train_val_y)
train_val_loader = DataLoader(train_val_set, batch_size=batch_size, shuffle=True)

model_best = Classifier()
loss = nn.CrossEntropyLoss()  # 因為是 classification task，所以 loss 使用 CrossEntropyLoss
optimizer = torch.optim.Adam(model_best.parameters(), lr=0.001)  # optimizer 使用 Adam
num_epoch = 30

for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0

    model_best.train()
    for i, data in enumerate(train_val_loader):
        optimizer.zero_grad()
        train_pred = model_best(data[0])
        batch_loss = loss(train_pred, data[1])
        batch_loss.backward()
        optimizer.step()

        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        train_loss += batch_loss.item()

        # 將結果 print 出來
    print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f' % \
          (epoch + 1, num_epoch, time.time() - epoch_start_time, \
           train_acc / train_val_set.__len__(), train_loss / train_val_set.__len__()))

print("Training complicated")

print("Testing")
print("...")
model_best.eval()
prediction = []
with torch.no_grad():
    for i, data in enumerate(test_loader):
        test_pred = model_best(data)
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        for y in test_label:
            prediction.append(y)
print(prediction)
from sklearn import metrics

print("mouth_score:", metrics.accuracy_score(Y_test, prediction))
print("Testing complicated")
