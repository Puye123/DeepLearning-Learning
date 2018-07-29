#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras.datasets import mnist
import matplotlib.pyplot as plt

# 訓練データセットとテストデータセットのロード
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# データセットをのぞいてみる
print("train_images.shape: " + str(train_images.shape)) #(60000, 28, 28)
print("train_labels: " + str(train_labels)) #[5 0 4 ... 5 6 8]
print("test_images.shape: " + str(test_images.shape)) #(10000, 28, 28)
print("test_labels: " + str(test_labels)) #[7 2 1 ... 4 5 6]

# 画像を表示してみる
train_image = train_images[0] # expect '5'
test_image = test_images[0] # expect '7'

figure, ax = plt.subplots(1, 2)
ax[0].imshow(train_image, cmap=plt.cm.binary)
ax[1].imshow(test_image, cmap=plt.cm.binary)
plt.show()