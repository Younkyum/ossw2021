from PIL import Image
import numpy as np
from os import listdir
from os.path import isfile
import cv2
import tensorflow as tf
import io
from google.cloud import vision
from google.cloud.vision_v1 import types


def detect_text():
    with io.open("prediction/out.jpg", 'rb') as image_file:
        content = image_file.read()

    client = vision.ImageAnnotatorClient.from_service_account_json("western-verve-325505-3fda9294d9f6.json")

    image = types.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    print('Texts:')

    for text in texts:
        print('\n"{}"'.format(text.description))

        vertices = (['({},{})'.format(vertex.x, vertex.y)
                     for vertex in text.bounding_poly.vertices])

        print('bounds: {}'.format(','.join(vertices)))


def plot_pred(img, p):
    p = np.asarray(p)
    p *= IMAGE_SIZE
    p = p.astype(int)
    img = img[p[1]:p[1] + p[2], p[0]:p[0] + p[2]]

    cv2.imshow("Output", img)
    cv2.waitKey()

    img *= 255
    cv2.imwrite("prediction/out.jpg", img)

    return Image.fromarray(img.astype(np.uint8))


path = "test/"
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
    plot_pred(img, pred[cnt])
    detect_text()
    cnt += 1
