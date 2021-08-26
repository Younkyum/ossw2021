from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

EPOCH_SIZE = 64


def natural_birb_gen(batch_size=32):
    birb_pil = Image.open("birb.png")

    while True:
        X = np.zeros((batch_size, 128, 128, 3))
        Y = np.zeros((batch_size, 3))

        for i in range(batch_size):
            size = np.random.randint(24, 84)
            birb_temp = birb_pil.resize((size, size))
            bg = Image.open("background/{0}.jpg".format(np.random.randint(1, 5)))
            bg = bg.resize((128, 128))

            xPos = np.random.randint(0, 128 - size)
            yPos = np.random.randint(0, 128 - size)

            bg.paste(birb_temp, (xPos, yPos), mask=birb_temp)

            X[i] = np.asarray(bg) / 255.
            Y[i, 0] = xPos / 128.
            Y[i, 1] = yPos / 128.
            Y[i, 2] = size / 128.

        yield X, Y


def plot_pred(img, p):
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    rect = Rectangle(xy=(p[0]*128, p[1]*128), width=p[2]*128, height=p[2]*128, linewidth=1, edgecolor='g', facecolor='none')
    ax.add_patch(rect)

    plt.show()


try:
    model = tf.keras.models.load_model(
        'E:/Coding/Projects/ossw2021_COSMITH/git/symbol_detection/models/saved_natural_synthetic_model')   # 경로는 적절히 바꿔서 사용할 것

except OSError:
    print("Model not found! Making a new one...")

    vgg = tf.keras.applications.VGG16(input_shape=[128, 128, 3], include_top=False, weights='imagenet')
    x = Flatten()(vgg.output)
    x = Dense(3, activation='sigmoid')(x)
    model = Model(vgg.input, x)
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001))

    model.fit_generator(natural_birb_gen(), steps_per_epoch=EPOCH_SIZE, epochs=10)
    model.save('models/saved_natural_synthetic_model')

else:
    x, _ = next(natural_birb_gen())

    pred = model.predict(x)

    cnt = 0
    for img in x:
        plot_pred(img, pred[cnt])
        cnt += 1

