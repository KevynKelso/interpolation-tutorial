import pathlib

import tensorflow as tf
import tensorflow_datasets as tfds
from matplotlib import pyplot as plt

## create train and validation datasets
DB_PATH = "./archive/images"
BUFFER_SIZE = 10000
BATCH_SIZE = 32
IMG_WIDTH = 128
IMG_HEIGHT = 128


def my_data():
    def load(image_file):
        image = tf.io.read_file(image_file)
        image = tf.image.decode_jpeg(image, channels=3)

        input_image = tf.cast(image, tf.float32)
        return input_image

    def random_crop(input_image):
        cropped_image = tf.image.random_crop(
            input_image, size=[IMG_HEIGHT, IMG_WIDTH, 3]
        )

        return cropped_image

    def resize(input_image):
        input_image = tf.image.resize(
            input_image,
            [IMG_HEIGHT, IMG_WIDTH],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        )
        return input_image

    def normalize(input_image):
        input_image = input_image / 255
        return input_image

    @tf.function()
    def random_jitter(input_image):
        input_image = random_crop(input_image)

        if tf.random.uniform(()) > 0.5:
            input_image = tf.image.flip_left_right(input_image)

        return input_image

    def load_image_train(image_file):
        input_image = load(image_file)
        # input_image = random_jitter(input_image)
        input_image = resize(input_image)
        input_image = normalize(input_image)

        return input_image, input_image

    def load_image_test(image_file):
        input_image = load(image_file)
        # input_image = random_jitter(input_image)
        input_image = resize(input_image)
        input_image = normalize(input_image)

        return input_image, input_image

    data_dir = pathlib.Path(DB_PATH)
    image_count = len(list(data_dir.glob("*.jpg")))
    dataset = tf.data.Dataset.list_files(DB_PATH + "/*.jpg")

    val_size = int(image_count * 0.2)
    train_ds = dataset.skip(val_size)
    val_ds = dataset.take(val_size)

    # print(tf.data.experimental.cardinality(train_ds).numpy())
    # print(tf.data.experimental.cardinality(val_ds).numpy())

    train_ds = train_ds.map(
        load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    train_ds = train_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    val_ds = val_ds.map(load_image_test)
    val_ds = val_ds.batch(BATCH_SIZE)

    return train_ds, val_ds
