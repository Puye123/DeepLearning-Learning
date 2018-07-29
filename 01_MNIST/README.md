# MNIST

## データセットのロード
MNIST用データセットは`keras.datasets`の中にある`mnist`をインポートする.  
([mnist_load_data.py](https://github.com/Puye123/DeepLearning-Learning/blob/master/01_MNIST/mnist_load_data.py)参照)
```python
from keras.datasets import mnist

# 訓練データセットとテストデータセットのロード
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
```
中身を確認してみる.  
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
