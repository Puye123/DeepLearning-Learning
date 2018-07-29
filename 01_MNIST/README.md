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