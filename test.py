import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from data import my_data

model = tf.keras.models.load_model("my_model")
train_ds, val_ds = my_data()

for input_images, output_images in val_ds.take(1):
    reconst_vec = model.predict(input_images)
    for i in range(25):
        reconst = reconst_vec[i, :, :, :] * 255
        reconst = np.array(reconst)
        reconst = reconst.astype(np.uint8)
        plt.imsave(f"im{i}.png", np.concatenate((input_images[i], reconst), axis=1)
