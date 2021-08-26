import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras.utils import plot_model
import cv2

IS_MODELLED = True
BATCH_SIZE = 64
EPOCH_SIZE = 64


def synthetic_gen(batch_size=48):
    while True:
        X = np.zeros((batch_size, 128, 128, 3))
        Y = np.zeros((batch_size, 3))

        for i in range(batch_size):
            x = np.random.randint(8, 120)
            y = np.random.randint(8, 120)
            a = min(128 - max(x, y), min(x, y))
            r = np.random.randint(4, a)

            for x_i in range(128):
                for y_i in range(128):
                    if ((x_i - x) ** 2) + ((y_i - y) ** 2) < r ** 2:
                        X[i, x_i, y_i, :] = 1
            Y[i, 0] = (x - r) / 128.
            Y[i, 1] = (y - r) / 128.
            Y[i, 2] = 2 * r / 128.

        yield X, Y


def plot_pred(img, p):
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    rect = Rectangle(xy=(p[1]*128, p[0]*128), width=p[2]*128, height=p[2]*128, linewidth=1, edgecolor='g', facecolor='none')
    ax.add_patch(rect)
    plt.show()


try:
    model = tf.keras.models.load_model(
        'E:/Coding/Projects/ossw2021_COSMITH/git/symbol_detection/models/saved_synthetic_model')

except OSError:
    vgg = tf.keras.applications.VGG16(input_shape=[128, 128, 3], include_top=False, weights='imagenet')
    x = Flatten()(vgg.output)
    x = Dense(3, activation='sigmoid')(x)
    model = Model(vgg.input, x)
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001))

    model.fit_generator(synthetic_gen(), steps_per_epoch=EPOCH_SIZE, epochs=10)
    model.save('models/saved_synthetic_model')

else:
    x, _ = next(synthetic_gen())

    pred = model.predict(x)

    for i in range(5):
        im = x[i]
        p = pred[i]
        plot_pred(im, p)
