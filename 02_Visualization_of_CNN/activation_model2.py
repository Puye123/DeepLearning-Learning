# -*- coding: utf-8 -*-

from keras import models
from keras import layers
from keras.utils import to_categorical
from keras import backend as K
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np

# モデルのロード
model = models.load_model('./mnist.h5')
model.summary()

# オリジナル画像の読み込みと表示
tmp_image = image.load_img('./02_Visualization_of_CNN/soft_cream.png', target_size=(28, 28), grayscale=True)
image_array = image.img_to_array(tmp_image)
plt.imshow(image_array.reshape(28,28), cmap=plt.cm.binary)
plt.show()

# モデルのinputに合わせて成形
image_array = np.expand_dims(image_array, axis=0)

# 推論結果を確認
result = model.predict_proba(image_array)
i = 0
for r in result[0]:
    print(i, ": ", r)
    i += 1

# 活性化を返すモデルを新たに作成
layer_outputs = [layer.output for layer in model.layers[:4]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(image_array)

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