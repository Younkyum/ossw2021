from PIL import Image
import numpy as np
from os import listdir
import cv2
import tensorflow as tf
from pytesseract import *


def plot_pred(img, p):
    p = np.asarray(p)
    p *= IMAGE_SIZE
    p = p.astype(int)
    img = img[p[1]:p[1] + p[2], p[0]:p[0] + p[2]]
    # rect = Rectangle(xy=(p[0], p[1]), width=p[2], height=p[2], linewidth=2, edgecolor='g',
    #                 facecolor='none')
    # ax.add_patch(rect)

    return img


path = "dataset/test/"
IMAGE_SIZE = 200

image_set = [cv2.imread(path + img) for img in listdir(path)]
image_set = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in image_set]
image_set = [np.stack((img,) * 3, axis=-1) for img in image_set]
image_set = [cv2.resize(img, dsize=(IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA) for img in image_set]
image_set = [img / 255. for img in image_set]

image_set = np.asarray(image_set)
print(image_set.shape)

model = tf.keras.models.load_model(
            'E:/Coding/Projects/ossw2021_COSMITH/git/symbol_detection/models/saved_symbol_detection_model_VGG16')

pred = model.predict(image_set)

cnt = 0
for img in image_set:
    image = Image.fromarray(plot_pred(img, pred[cnt]))
    text = image_to_string(image, lang="kor")
    print(text)
    cnt += 1
