from PIL import Image
import numpy as np
import dask.dataframe as dd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

IMAGE_SIZE = 300


class CirclingIter:     # 마지막 숫자에 다다르면 다시 처음으로 돌아가는 이터레이터
    def __init__(self, last, start=0):
        self.start = start
        self.last = last
        self.current = start

    def __iter__(self):
        return self

    def __next__(self):
        r = self.current

        if self.current < self.last - 1:
            self.current += 1

        else:
            self.current = self.start

        return r


def make_square(img):   # 이미지를 정사각형으로 바꾸는 함수
    size = (lambda: img.width if img.width > img.height else img.height)()
    squaredImg = Image.fromarray(np.uint8(np.zeros((size, size, 3))))
    squaredImg.paste(img)

    return squaredImg


def train_data_gen(batch_size=8):
    df = dd.read_csv('dataset/train/_annotations.csv')  # 데이터셋에 대한 정보가 저장된 csv 파일
    dataset_size = df.filename.size.compute()
    cntIter = CirclingIter(dataset_size)

    while True:
        X = np.zeros((batch_size, IMAGE_SIZE, IMAGE_SIZE, 3))
        Y = np.zeros((batch_size, 4))
        for i in range(batch_size):
            cnt = next(cntIter)
            img = Image.open('dataset/train/{0}'.format(df.filename.compute()[cnt]))
            img = make_square(img)
            original_size = float(img.height)
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE))

            X[i] = np.asarray(img) / 255.
            Y[i][0] = df.xmin.compute()[cnt] / original_size
            Y[i][1] = df.ymin.compute()[cnt] / original_size
            Y[i][2] = df.xmax.compute()[cnt] / original_size
            Y[i][3] = df.ymax.compute()[cnt] / original_size

        yield X, Y


def plot_pred(img, p):
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    p *= IMAGE_SIZE
    width = p[2] - p[0]
    height = p[3] - p[1]

    rect = Rectangle(xy=(p[0], p[1]), width=width, height=height, linewidth=2, edgecolor='g',
                     facecolor='none')
    ax.add_patch(rect)

    plt.show()


try:
    model = tf.keras.models.load_model(
        'E:/Coding/Projects/ossw2021_COSMITH/git/symbol_detection/models/saved_symbol_detection_model')   # 모델 경로는 적절히 바꿔서 사용할 것

except OSError:
    print("Model not found! Making a new one...")

    vgg = tf.keras.applications.VGG16(input_shape=[IMAGE_SIZE, IMAGE_SIZE, 3], include_top=False, weights='imagenet')
    x = Flatten()(vgg.output)
    x = Dense(4, activation='sigmoid')(x)
    model = Model(vgg.input, x)
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001))

    model.fit_generator(train_data_gen(), steps_per_epoch=64, epochs=10)
    model.save('models/saved_symbol_detection_model')   # 모델을 저장할 위치

else:
    x, _ = next(train_data_gen())

    pred = model.predict(x)

    cnt = 0
    for img in x:
        plot_pred(img, pred[cnt])
        cnt += 1
