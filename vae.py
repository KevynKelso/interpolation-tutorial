from os import path

import tensorflow.keras.backend as K
from matplotlib import pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (BatchNormalization, Conv2D,
                                     Conv2DTranspose, Dense, Flatten, Input,
                                     Lambda, Reshape)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.preprocessing import image

from data import my_data


def vae():
    latent_dim = 256  # Number of latent dimension parameters

    input_img = Input(shape=(128, 128, 3))

    x = Conv2D(32, (3, 3), activation="relu", padding="same", strides=2)(input_img)
    x = BatchNormalization()(x)

    x = Conv2D(16, (3, 3), activation="relu", padding="same", strides=2)(x)
    x = BatchNormalization()(x)

    x = Conv2D(4, (3, 3), activation="relu", padding="same", strides=2)(x)
    x = BatchNormalization()(x)

    shape_before_flattening = K.int_shape(x)
    print(shape_before_flattening)
    x = Flatten()(x)

    z_mu = Dense(latent_dim)(x)
    z_log_sigma = Dense(
        latent_dim, kernel_initializer="zeros", bias_initializer="zeros"
    )(x)

    # sampling function
    def sampling(args):
        z_mu, z_log_sigma = args
        epsilon = K.random_normal(
            shape=(K.shape(z_mu)[0], latent_dim), mean=0.0, stddev=1.0
        )
        return z_mu + K.exp(z_log_sigma) * epsilon

    # sample vector from the latent distribution
    z = Lambda(sampling)([z_mu, z_log_sigma])

    # encoder = Model(input_img, z)
    # decoder takes the latent distribution sample as input
    decoder_input = Input(K.int_shape(z)[1:])
    x = Dense(
        4096, activation="relu", name="intermediate_decoder", input_shape=(latent_dim,)
    )(decoder_input)
    x = Reshape((8, 8, 64))(x)

    x = Conv2DTranspose(32, (3, 3), strides=2, padding="same")(x)
    x = BatchNormalization()(x)

    x = Conv2DTranspose(16, (3, 3), strides=2, padding="same")(x)
    x = BatchNormalization()(x)

    x = Conv2DTranspose(8, (3, 3), strides=2, padding="same")(x)
    x = BatchNormalization()(x)

    x = Conv2DTranspose(3, (3, 3), strides=2, padding="same", activation="sigmoid")(x)

    # decoder model statement
    decoder = Model(decoder_input, x)

    # apply the decoder to the sample from the latent distribution
    pred = decoder(z)

    def vae_loss(x, pred):
        x = K.flatten(x)
        pred = K.flatten(pred)
        # Reconstruction loss
        reconst_loss = 1000 * K.mean(K.square(x - pred))

        # KL divergence
        kl_loss = -0.5 * K.mean(
            1 + z_log_sigma - K.square(z_mu) - K.exp(z_log_sigma), axis=-1
        )

        return reconst_loss + kl_loss

    # VAE model statement
    vae = Model(input_img, pred)
    vae.add_loss(vae_loss(input_img, pred))
    optimizer = Adam(learning_rate=0.0005)
    vae.compile(optimizer=optimizer, loss=None)

    vae.summary()

    train_ds, val_ds = my_data()
    # run the model
    early_stopping = EarlyStopping(
        monitor="val_loss", min_delta=0, patience=10, verbose=5, mode="auto"
    )

    vae.fit(
        train_ds,
        epochs=50,
        validation_data=val_ds,
        callbacks=[early_stopping],
        verbose=1,
    )
    vae.save("my_model")


if __name__ == "__main__":
    vae()
