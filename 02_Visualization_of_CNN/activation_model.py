# -*- coding: utf-8 -*-

from keras import models
from keras import layers
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import gc

# モデルのロード
model = models.load_model('./mnist.h5')
model.summary()

# 画像データの準備
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
if K.image_data_format() == 'channels_first':
    test_images = test_images.reshape(10000, 1, 28, 28)
else:
    test_images = test_images.reshape(10000, 28, 28, 1)
test_images = test_images.astype('float32') / 255.0
test_image =  np.expand_dims(test_images[1], axis=0)

# 活性化を返すモデルを新たに作成
layer_outputs = [layer.output for layer in model.layers[:4]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(test_image)

# 各層の名前を取得
layer_names = []
for layer in model.layers[:4]:
    layer_names.append(layer.name)
    print(layer.name)

images_per_row = 8

# 特徴マップを表示
for layer_name, layer_activation in zip(layer_names, activations):
    # 特徴マップに含まれている特徴量の数
    n_features = layer_activation.shape[-1]
    # 特徴マップの形状
    size = layer_activation.shape[1]

    # この行列で活性化のチャネルをタイル表示
    n_cols = n_features // images_per_row
    print(size + n_cols)
    print(images_per_row * size)
    display_grid = np.zeros((size * n_cols, images_per_row * size))

    # 各フィルタを1つの大きな水平グリッドでタイル表示
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0, :, :, col * images_per_row + row]
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image
    
    # グリッドを表示
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')

plt.show()