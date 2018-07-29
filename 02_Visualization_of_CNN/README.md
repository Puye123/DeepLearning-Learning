# Visualization_of_CNN

## 1 前準備
MNISTのCNNを可視化したいが, 毎回学習させるのは時間がかかるので学習結果を保存して使い回す.  

### 1.1 モデル
以下のモデルを使用.  
[kerasのサンプルコード](https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py)と同じモデル
```python
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
```

### 1.2 学習結果を保存
```python
network.save('mnist.h5')
```
### 1.3 学習結果のロード
```python
model = load_model('mnist.h5')
```
### 1.4 [補足] Kerasでのモデルの保存方法
Kerasでのモデルの保存方法については[Keras Documentation](https://keras.io/ja/getting-started/faq/#keras_1)を見て学んだ.  
これでモデル全体(アーキテクチャ + 重み + オプティマイザの状態)の保存/読み込みができる.  
ちなみに, モデルのアーキテクチャのみを保存/読み込みする場合は以下のように実装する.  
```python
# save as JSON
json_string = model.to_json()
# save as YAML
yaml_string = model.to_yaml()
```
``` python
# model reconstruction from JSON:
from keras.models import model_from_json
model = model_from_json(json_string)

# model reconstruction from YAML
from keras.models import model_from_yaml
model = model_from_yaml(yaml_string)
```
また, モデルの重みのみを保存/読み込みする場合は, 
```python
# 保存
model.save_weights('mnist_weights.h5')
# 読み込み
model.load_weights('mnist_weights.h5')
```
とする.  

## 2. 中間層の出力を可視化する
### 2.1 ポイント
Keras APIを用いて活性化状態を返すモデルを新たに生成する.  
```python
layer_outputs = [layer.output for layer in model.layers[:4]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(test_image)
```
あとは, `activations`の中身をプロットしていけば良い.  

### 2.2 ソースコード
- kerasに入っているデータセットを使うソースコード([リンク](https://github.com/Puye123/DeepLearning-Learning/blob/master/02_Visualization_of_CNN/activation_model.py))
- オリジナル画像を使うソースコード([リンク](https://github.com/Puye123/DeepLearning-Learning/blob/master/02_Visualization_of_CNN/activation_model2.py))

### 2.3 可視化の結果
以下のIssue参照.  
[中間層を可視化してみる](https://github.com/Puye123/DeepLearning-Learning/issues/11)

# 参考資料
- https://github.com/fchollet/deep-learning-with-python-notebooks
- [PythonとKerasによるディープラーニング](https://book.mynavi.jp/ec/products/detail/id=90124) 5.4章 CNNが学習した内容を可視化する