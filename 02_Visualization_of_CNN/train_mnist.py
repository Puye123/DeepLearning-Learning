# -*- coding: utf-8 -*-

from keras import models
from keras import layers
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import backend as K

# 画像データの準備
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
if K.image_data_format() == 'channels_first':
    train_images = train_images.reshape(60000, 1, 28, 28)
    test_images = test_images.reshape(10000, 1, 28, 28)
    input_shape = (1, 28, 28)
else:
    train_images = train_images.reshape(60000, 28, 28, 1)
    test_images = test_images.reshape(10000, 28, 28, 1)
    input_shape = (28, 28, 1)
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# ラベルの準備
train_labels = to_categorical(train_labels, 10)
print("train_labels[0] : " , train_labels[0])
test_labels = to_categorical(test_labels, 10)
print("test_labels[0]  : " ,test_labels[0])

# ニューラルネットワークの構築とコンパイル
network = models.Sequential()
network.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
network.add(layers.Conv2D(64, (3, 3), activation='relu'))
network.add(layers.MaxPooling2D(pool_size=(2, 2)))
network.add(layers.Dropout(0.25))
network.add(layers.Flatten())
network.add(layers.Dense(128, activation='relu'))
network.add(layers.Dropout(0.5))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
network.summary()

# 学習
network.fit(train_images, train_labels, epochs=20, batch_size=128)
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc  : ' , test_acc)
print('test_loss : ' , test_loss)

# 学習結果を保存
network.save('mnist.h5')