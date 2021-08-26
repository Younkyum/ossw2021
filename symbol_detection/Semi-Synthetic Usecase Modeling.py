from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.patches import Rectangle
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

BATCH_SIZE = 64
EPOCH_SIZE = 64


def birb_gen(batch_size=32):
    birb_pil = Image.open("birb.png")
    birb_pil = birb_pil.resize((64, 64))
    birb = np.asarray(birb_pil)

    while True:
        X = np.zeros((batch_size, 128, 128, 3))
        Y = np.zeros((batch_size, 3))

        for i in range(batch_size):
            size = np.random.randint(32, 64)
            temp_birb = birb_pil.resize((size, size))
            birb = np.asarray(temp_birb) / 255.0
            birb_x, birb_y, _ = birb.shape

            bg = Image.new('RGB', (128, 128))

            x1 = np.random.randint(1, 128 - birb_x)
            y1 = np.random.randint(1, 128 - birb_y)

            bg.paste(temp_birb, (x1, y1))
            birb = np.asarray(bg) / 255.0

            X[i] = birb
            Y[i, 0] = x1 / 128.0
            Y[i, 1] = y1 / 128.0
            Y[i, 2] = birb_x / 128.

        yield X, Y


def plot_pred(img, p):
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    rect = Rectangle(xy=(p[0]*128, p[1]*128), width=p[2]*128, height=p[2]*128, linewidth=1, edgecolor='g', facecolor='none')
    ax.add_patch(rect)
    plt.show()


try:
    model = tf.keras.models.load_model(
        'E:/Coding/Projects/ossw2021_COSMITH/git/symbol_detection/models/saved_semi_synthetic_model')   # 경로는 적절히 바꿔서 사용할 것

except OSError:
    print("Model not found! Making a new one...")

    vgg = tf.keras.applications.VGG16(input_shape=[128, 128, 3], include_top=False, weights='imagenet')
    x = Flatten()(vgg.output)
    x = Dense(3, activation='sigmoid')(x)
    model = Model(vgg.input, x)
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001))

    model.fit_generator(birb_gen(), steps_per_epoch=EPOCH_SIZE, epochs=10)
    model.save('models/saved_semi_synthetic_model')

else:
    x, _ = next(birb_gen())

    pred = model.predict(x)

    cnt = 0
    for img in x:
        plot_pred(img, pred[cnt])
        cnt += 1

