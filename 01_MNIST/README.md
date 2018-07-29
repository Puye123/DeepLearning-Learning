# MNIST

## 1 データセットのロード

### 1.1 ロード
MNIST用データセットは`keras.datasets`の中にある`mnist`をインポートする.  
([mnist_load_data.py](https://github.com/Puye123/DeepLearning-Learning/blob/master/01_MNIST/mnist_load_data.py)参照)
```python
from keras.datasets import mnist

# 訓練データセットとテストデータセットのロード
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
```

### 1.2 データセットの確認
#### 1.2.1 データサイズを確認  
([mnist_load_data.py](https://github.com/Puye123/DeepLearning-Learning/blob/master/01_MNIST/mnist_load_data.py)参照)
```python
print("train_images.shape: " + str(train_images.shape))
print("train_labels: " + str(train_labels))
print("test_images.shape: " + str(test_images.shape))
print("test_labels: " + str(test_labels))
```
実行結果
```
train_images.shape: (60000, 28, 28)
train_labels: [5 0 4 ... 5 6 8]
test_images.shape: (10000, 28, 28)
test_labels: [7 2 1 ... 4 5 6]
```
訓練用データセットは28x28pixelの画像が60000枚であることがわかる.  
テストデータセットは28x28pixelの画像が10000枚であることがわかる.

#### 1.2.2 データセット画像を確認  
([mnist_load_data.py](https://github.com/Puye123/DeepLearning-Learning/blob/master/01_MNIST/mnist_load_data.py)参照)
```python
train_image = train_images[0] # expect '5'
test_image = test_images[0] # expect '7'

figure, ax = plt.subplots(1, 2)
ax[0].imshow(train_image, cmap=plt.cm.binary)
ax[1].imshow(test_image, cmap=plt.cm.binary)
plt.show()
```
実行結果  
![データセット画像の表示](https://user-images.githubusercontent.com/32557553/43364053-951edb28-934d-11e8-926f-06c3d435e592.png)  
train_labelsを確認したところ, `train_labels: [5 0 4 ... 5 6 8]` であった.   
即ち, train_imagesの0番目の数字は'5'である.  
同様に, test_imagesの0番目の数字は'7'である.  
実行結果を見ると, '5'と'7'がそれぞれ表示されていることがわかる.

## 2 ニューラルネットワークの構築と学習

### 2.1 ニューラルネットワークの構築とコンパイル
`models`と`layers`をインポートする.
```python
from keras import models
from keras import layers
```
ニューラルネットワークの構築を構築する.  
- 1層目 : 512出力, 28*28入力の全結合層
- 2層目 : 10出力のソフトマックス層  

```python
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))
```
コンパイルする.  
```python
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
```
ニューラルネットワークを確認する.
```python
network.summary()
```

確認結果  
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense_1 (Dense)              (None, 512)               401920
_________________________________________________________________
dense_2 (Dense)              (None, 10)                5130
=================================================================
Total params: 407,050
Trainable params: 407,050
Non-trainable params: 0
```

### 2.2 画像データの準備
データ構造を(60000, 28, 28)から(60000, 28*28)に変更.  
さらに, 各画素値を[0, 1]の区間に収まるように変換(float32型).  
データのロードは "1.1 ロード" の通り.  
```python
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255.0
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255.0
```

### 2.3 ラベルの準備
ラベルをone-hot-encodingする.  
```python
train_labels = to_categorical(train_labels)
print("train_labels[0] : " , train_labels[0])
test_labels = to_categorical(test_labels)
print("test_labels[0]  : " ,test_labels[0])
```
エンコード後の中身をのぞいてみる.  
```
train_labels[0] :  [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
test_labels[0]  :  [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
```
先程確認した画像データの'5'と'7'に対応するビットがそれぞれ1になっており, one-hot化できていることがわかる.  

### 2.4 学習
fitメソッドで学習を行う.  
エポック数は10とする
evalateメソッドでテストデータを用いて評価する.  
```python
network.fit(train_images, train_labels, epochs=10, batch_size=128)
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc  : ' , test_acc)
print('test_loss : ' , test_loss)
```
実行結果(GPUを使用)  
```
Epoch 1/10
2018-07-29 17:50:53.575372: I T:\src\github\tensorflow\tensorflow\core\platform\cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
2018-07-29 17:50:53.852575: I T:\src\github\tensorflow\tensorflow\core\common_runtime\gpu\gpu_device.cc:1356] Found device 0 with properties:
name: GeForce GTX 1060 6GB major: 6 minor: 1 memoryClockRate(GHz): 1.7085
pciBusID: 0000:01:00.0
totalMemory: 6.00GiB freeMemory: 4.96GiB
2018-07-29 17:50:53.858010: I T:\src\github\tensorflow\tensorflow\core\common_runtime\gpu\gpu_device.cc:1435] Adding visible gpu devices: 0
2018-07-29 17:50:54.414242: I T:\src\github\tensorflow\tensorflow\core\common_runtime\gpu\gpu_device.cc:923] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-07-29 17:50:54.417375: I T:\src\github\tensorflow\tensorflow\core\common_runtime\gpu\gpu_device.cc:929]      0
2018-07-29 17:50:54.419579: I T:\src\github\tensorflow\tensorflow\core\common_runtime\gpu\gpu_device.cc:942] 0:   N
2018-07-29 17:50:54.422128: I T:\src\github\tensorflow\tensorflow\core\common_runtime\gpu\gpu_device.cc:1053] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 4732 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1060 6GB, pci bus id: 0000:01:00.0, compute capability: 6.1)
60000/60000 [==============================] - 3s 49us/step - loss: 0.2588 - acc: 0.9259
Epoch 2/10
60000/60000 [==============================] - 2s 26us/step - loss: 0.1047 - acc: 0.9694
Epoch 3/10
60000/60000 [==============================] - 1s 25us/step - loss: 0.0688 - acc: 0.9789
Epoch 4/10
60000/60000 [==============================] - 1s 24us/step - loss: 0.0501 - acc: 0.9848
Epoch 5/10
60000/60000 [==============================] - 1s 24us/step - loss: 0.0381 - acc: 0.9884
Epoch 6/10
60000/60000 [==============================] - 1s 24us/step - loss: 0.0293 - acc: 0.9910
Epoch 7/10
60000/60000 [==============================] - 1s 24us/step - loss: 0.0228 - acc: 0.9935
Epoch 8/10
60000/60000 [==============================] - 1s 24us/step - loss: 0.0171 - acc: 0.9949
Epoch 9/10
60000/60000 [==============================] - 1s 24us/step - loss: 0.0137 - acc: 0.9960
Epoch 10/10
60000/60000 [==============================] - 1s 24us/step - loss: 0.0102 - acc: 0.9973
10000/10000 [==============================] - 0s 37us/step
test_acc  :  0.981
test_loss :  0.07306255485720212
```
