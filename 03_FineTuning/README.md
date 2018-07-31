# Fine Tuning
じゃんけんシステムを作りながらファインチューニングを学ぶ

# 1　前準備
## 1.1　画像を集める
Webカメラを使って手をひたすら撮影するスクリプトを作る.

### 1.1.1　撮影スクリプトの使い方
#### help
```
usage: python capture.py FILE [--dir <dir_name>] [--num <num_of_pictures>] [--size <size_of_pictures>] [--help]

optional arguments:
  -h, --help            show this help message and exit
  -d <dir_name>, --dir <dir_name>
                        directory name
  -n <num_of_pictures>, --num <num_of_pictures>
                        number of pictures
  -s <size_of_pictures>, --size <size_of_pictures>
                        size of pictures
```
#### デフォルト値
- 出力ディレクトリ名 : images
- 撮影枚数 : 10
- 画像サイズ : 224  

※ 画像は正方形になるようにリサイズされます

#### 使用例
- 'gu' ディレクトリに1000枚画像を保存する ※サイズはデフォルト値 (224, 224)
```
> python capture.py -d gu -n 1000
```