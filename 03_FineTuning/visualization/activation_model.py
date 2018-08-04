# -*- coding: utf-8 -*-

from keras import models
from keras import layers
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import backend as K
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import gc
import PIL

data_dir = '../DeepLearning-Learning_DATAS/03_FineTuning/'
h5_name = 'janken_small_cnn_01.h5'
#image_name = 'validation/gu/20180801_211326790991.jpg'
image_name = 'validation/tyoki/20180801_211605459033.jpg'

# モデルのロード
model = models.load_model(data_dir + h5_name)
model.summary()

# 画像データの準備
img = PIL.Image.open(data_dir + image_name)
img = img.resize((150, 150))
x = image.img_to_array(img)
x = x.astype('float32')/255.0
x = np.expand_dims(x, axis=0)

# 答え合わせ
hand_name = ['グー', '手が写っていない', 'その他', 'パー', 'チョキ' ]
result = model.predict(x)
your_hand_index = np.argmax(result[0])
print('あなたの出した手は... 「' + hand_name[your_hand_index] + '!」')
index = 0
for r in result[0]:
    print(hand_name[index], ' : ', r)
    index = index + 1

# 活性化を返すモデルを新たに作成
layer_outputs = [layer.output for layer in model.layers[:8]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(x)

# 各層の名前を取得
layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)

images_per_row = 8

# 特徴マップを表示
for layer_name, layer_activation in zip(layer_names, activations):
    # 特徴マップに含まれている特徴量の数
    n_features = layer_activation.shape[-1]
    # 特徴マップの形状
    size = layer_activation.shape[1]

    # この行列で活性化のチャネルをタイル表示
    n_cols = n_features // images_per_row
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