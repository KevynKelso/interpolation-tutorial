import glob
import os
import sys

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model, load_model

from data import BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, my_data

autoencoder = load_model("my_model")

encoder = Model(autoencoder.input, autoencoder.layers[10].output)
encoder.summary()
decoder_input = Input(shape=(256,))
decoder = Model(decoder_input, autoencoder.layers[-1](decoder_input))
decoder.summary()

train_ds, val_ds = my_data()

files = glob.glob("./archive/images/*.jpg")
num_batches = int(len(files) / BATCH_SIZE)
if len(files) % BATCH_SIZE != 0:
    print("Source and Target Files should be a multiple of Batch Size")


target_batches = []
mid_batches = []
src_batch = np.zeros((BATCH_SIZE, IMG_WIDTH, IMG_HEIGHT, 3))
target_batch = np.zeros((BATCH_SIZE, IMG_WIDTH, IMG_HEIGHT, 3))
src_latent = []
target_latent = []
img_nr = 0

for i in range(len(files)):
    source = Image.open(files[i])
    source = source.resize((IMG_WIDTH, IMG_HEIGHT))
    src_arr = np.array(source) / 255

    target = Image.open("./archive/images/" + os.path.basename(files[i]))
    target = target.resize((IMG_WIDTH, IMG_HEIGHT))
    target_arr = np.array(target) / 255

    src_batch[img_nr, :, :, :] = src_arr
    target_batch[img_nr, :, :, :] = target_arr
    img_nr = img_nr + 1

    if i % (BATCH_SIZE - 1) == 0 and i > 0:
        src_latent.append(encoder(src_batch))
        target_latent.append(encoder(target_batch))
        src_batch = np.zeros((BATCH_SIZE, IMG_WIDTH, IMG_HEIGHT, 3))
        target_batch = np.zeros((BATCH_SIZE, IMG_WIDTH, IMG_HEIGHT, 3))
        img_nr = 0

num_imgs = int(BATCH_SIZE * num_batches)
for i in range(1):
    mid_latent = (src_latent[i] + target_latent[i]) / 2

    src_img = decoder.predict(src_latent[i])
    target_img = decoder.predict(target_latent[i])
    mid_img = decoder.predict(mid_latent)
    for j in range(BATCH_SIZE):
        # ax = fig.add_subplot(num_imgs, 3, i * BATCH_SIZE + 3 * j + 1)
        # plt.imshow((src_img[j] * 255).astype(np.uint8))
        plt.imsave(f"src-{j}", (src_img[j] * 255).astype(np.uint8))
        # ax.axis("off")
        # ax = fig.add_subplot(num_imgs, 3, i * BATCH_SIZE + 3 * j + 2)
        # plt.imshow((mid_img[j] * 255).astype(np.uint8))
        plt.imsave(f"mid-{j}", (mid_img[j] * 255).astype(np.uint8))
        # ax.axis("off")
        # ax = fig.add_subplot(num_imgs, 3, i * BATCH_SIZE + 3 * j + 3)
        # plt.imshow((target_img[j] * 255).astype(np.uint8))
        plt.imsave(f"target-{j}", (target_img[j] * 255).astype(np.uint8))
        # ax.axis("off")
        if j == 50:
            break
