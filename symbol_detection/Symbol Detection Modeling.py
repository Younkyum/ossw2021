from PIL import Image
import numpy as np
import dask.dataframe as dd
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Conv2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import cv2

IMAGE_SIZE = 200
raw_train_x = []
raw_train_y = []
test_x = []
test_y = []


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


def load_train_images(num=0):   # 훈련에 사용할 사진들과 표기의 위치와 크기를 기록한 파일을 불러옴
    global raw_train_x, raw_train_y

    raw_train_x = []
    raw_train_y = []
    dataset_path = 'dataset/train/'
    print("Loading dataset information file...")
    df = dd.read_csv(dataset_path + '_annotations.csv')

    print("Loading dataset files...")
    if num == 0:
        file_num = df.filename.size.compute()

    else:
        file_num = num

    for i in range(file_num):
        line = df.loc[i].compute()

        raw_train_x.append(cv2.imread(dataset_path + line.filename[i]))
        raw_train_y.append([line.xmin[i], line.ymin[i], line.side[i]])
        """
        raw_train_y[i][0]: 경계 정사각형의 최소 x좌표
        raw_train_y[i][1]: 경계 정사각형의 최소 y좌표
        raw_train_y[i][2]: 경계 정사각형의 변 길이
        """


def crop_randomly(np_img, label):   # 표기가 포함되도록 무작위의 위치와 크기로 사진을 잘라냄
    pil_img = Image.fromarray(np_img)
    width = pil_img.width
    height = pil_img.height
    min_length = label[2]
    max_length = min(height, width)

    side = np.random.randint(min_length, max_length)
    x_lower_bound = max(label[0] + label[2] - side, 0)
    y_lower_bound = max(label[1] + label[2] - side, 0)
    x_upper_bound = min(label[0], width - side)
    y_upper_bound = min(label[1], height - side)
    crop_x_min = np.random.randint(x_lower_bound, x_upper_bound + 1)
    crop_y_min = np.random.randint(y_lower_bound, y_upper_bound + 1)
    crop_x_max = crop_x_min + side
    crop_y_max = crop_y_min + side

    cropped_label = (label[0] - crop_x_min, label[1] - crop_y_min, label[2])
    cropped_img = pil_img.crop((crop_x_min, crop_y_min, crop_x_max, crop_y_max))

    return np.asarray(cropped_img), cropped_label


def invert_grayscale(grayscale_img):
    for y in range(grayscale_img.shape[0]):
        for x in range(grayscale_img.shape[1]):
            grayscale_img[y, x] = float(255 - grayscale_img[y, x])

    return grayscale_img


def normalize_img(np_square_img, label, input_size=IMAGE_SIZE):  # 훈련에 사용할 수 있도록 색 값, 사진의 크기 등을 정규화
    pil_img = Image.fromarray(np_square_img)
    original_size = pil_img.width

    pil_img = pil_img.resize((input_size, input_size))
    np_square_img = np.asarray(pil_img) / 255.

    label = np.asarray(label) / original_size
    label = tuple(label)

    return np_square_img, label


def test_data_gen(set_size=50, amp_ratio=2):  # 모델의 정확도 평가를 위한 데이터 생성
    load_train_images(set_size)

    raw_data_number = len(raw_train_x)

    global test_x
    global test_y

    for i in range(raw_data_number):
        img = raw_train_x[i]
        label = raw_train_y[i]
        print("Picture " + str(i))
        for j in range(amp_ratio):
            processed_img, processed_label = crop_randomly(img, label)

            # if np.random.randint(0, 2):
            #    processed_img = invert_grayscale(processed_img)     # 랜덤으로 흑백반전 적용

            processed_img, processed_label = normalize_img(processed_img, processed_label)

            test_x.append(processed_img)
            test_y.append(processed_label)

    test_x = np.asarray(test_x)
    test_y = np.asarray(test_y)
    print("Validation set loaded!")


def train_data_gen(set_size=0, batch_size=16):  # 훈련에 사용할 데이터 생성
    raw_data_number = len(raw_train_x)
    cntIter = CirclingIter(raw_data_number)

    while True:
        X = np.zeros((batch_size, IMAGE_SIZE, IMAGE_SIZE, 3))
        Y = np.zeros((batch_size, 3))

        for num in range(batch_size):
            cnt = next(cntIter)
            img = raw_train_x[cnt]
            label = raw_train_y[cnt]
            processed_img, processed_label = crop_randomly(img, label)
            X[num], Y[num] = normalize_img(processed_img, processed_label)

        yield X, Y


with tf.device('/GPU:0'):
    try:
        test_data_gen()
        load_train_images()

        model = tf.keras.models.load_model(
            'E:/Coding/Projects/ossw2021_COSMITH/git/symbol_detection/models/saved_symbol_detection_model_ResNet50')   # 모델 경로는 적절히 바꿔서 사용할 것

    except OSError:
        print("Model not found! Making a new one...")

        start = VGG16(input_shape=[IMAGE_SIZE, IMAGE_SIZE, 3], include_top=False, weights='imagenet')
        # start = ResNet50(input_shape=[IMAGE_SIZE, IMAGE_SIZE, 3], include_top=False, weights='imagenet')
        x = Flatten()(start.output)
        out = Dense(3, activation='sigmoid')(x)
        model = Model(start.input, out)
        model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001))

    finally:
        checkpoint_path = 'E:/Coding/Projects/ossw2021_COSMITH/git/symbol_detection/models/checkpoints'
        checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=True,
            monitor='val_loss',
            mode='max',
            save_best_only=True
        )

        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

        # model.fit(train_x, train_y, batch_size=6, epochs=10, shuffle=True)
        model.fit_generator(train_data_gen(batch_size=10),
                            steps_per_epoch=len(raw_train_x),
                            epochs=100,
                            validation_data=(test_x, test_y),
                            callbacks=[checkpoint_callback, early_stopping_callback])
        model.save('models/saved_symbol_detection_model_ResNet50')  # 모델을 저장할 위치
