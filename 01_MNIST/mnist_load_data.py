#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras.datasets import mnist

# 訓練データセットとテストデータセットのロード
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# データセットをのぞいてみる
print("train_images.shape: " + str(train_images.shape)) #(60000, 28, 28)
print("train_labels: " + str(train_labels)) #[5 0 4 ... 5 6 8]
print("test_images.shape: " + str(test_images.shape)) #(10000, 28, 28)
print("test_labels: " + str(test_labels)) #[7 2 1 ... 4 5 6]