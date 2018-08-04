# -*- coding: utf-8 -*-
from keras import models
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import PIL
import cv2

data_dir = '../DeepLearning-Learning_DATAS/03_FineTuning/'
h5_name = 'janken_small_cnn_01.h5'

def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = PIL.Image.fromarray(img)
    img = img.resize((150, 150))
    x = image.img_to_array(img)
    x = x.astype('float32')/255.0
    x = np.expand_dims(x, axis=0)
    return x

def make_heatmap(model, input_img):
    # ヒートマップの生成
    preds = model.predict(input_img)
    
    output_vector = model.output[:, np.argmax(preds[0])]
    last_conv_layer = model.get_layer('conv2d_4')
    grads = K.gradients(output_vector, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([input_img])
    for i in range(128):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    # 推論結果を取得
    hand_name = ['Rock', 'Not found', 'Other', 'Paper', 'Scissor' ]
    result = model.predict(input_img)
    your_hand_index = np.argmax(result[0])

    index = 0
    print('----------------------')
    for r in result[0]:
        print(hand_name[index], ' : ', r)
        index = index + 1

    return heatmap, hand_name[your_hand_index]


def synthotic_image(heatmap, img):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    synthotic = (heatmap/3 + img).astype('uint8')
    
    return synthotic


def main():
    cap = cv2.VideoCapture(0)
    model = models.load_model(data_dir + h5_name)
    model.summary()

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (400, 300)) 
        heatmap, result = make_heatmap(model, preprocess_image(frame))
        synthotic = synthotic_image(heatmap, frame)
        cv2.putText(synthotic, str(result), (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255))

        cv2.imshow('Grad-CAM', synthotic)

        k = cv2.waitKey(100)
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    main()