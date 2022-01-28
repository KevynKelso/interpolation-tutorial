import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from scipy.stats import norm

from data import my_data

fig = plt.figure(figsize=(15, 5))

model = tf.keras.models.load_model("my_vae")
train_ds, val_ds = my_data()

for input_images, output_images in val_ds.take(1):
    reconst_vec = model.predict(input_images)
    for i in range(25):
        ax = fig.add_subplot(5, 5, i + 1)
        ax.axis("off")
        reconst = reconst_vec[i, :, :, :] * 255
        reconst = np.array(reconst)
        reconst = reconst.astype(np.uint8)
        ax.imshow(reconst)
