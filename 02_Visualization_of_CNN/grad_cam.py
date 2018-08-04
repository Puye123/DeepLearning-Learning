# -*- coding: utf-8 -*-
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import PIL
import cv2


def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = PIL.Image.fromarray(img)
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def make_heatmap(model, input_img):
    preds = model.predict(input_img)
    
    output_vector = model.output[:, np.argmax(preds[0])]
    last_conv_layer = model.get_layer('block5_conv3')
    grads = K.gradients(output_vector, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([input_img])
    for i in range(512):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    return heatmap, decode_predictions(preds, top=1)[0]


def synthotic_image(heatmap, img):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    synthotic = heatmap + img
    
    return synthotic


def main():
    cap = cv2.VideoCapture(0)
    model = VGG16(weights='imagenet')

    while True:
        ret, frame = cap.read()
        #frame = cv2.resize(frame, (224, 224)) 
        heatmap, result = make_heatmap(model, preprocess_image(frame))
        synthotic = synthotic_image(heatmap, frame)
        cv2.putText(frame, str(result), (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 0))

        cv2.imshow('camera capture', frame)
        cv2.imshow('grad-CAM', synthotic)

        k = cv2.waitKey(1)
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    main()