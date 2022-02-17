from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (BatchNormalization, Conv2D,
                                     Conv2DTranspose, Input)
from tensorflow.keras.models import Model

from data import my_data


def accordion_vae():
    model_name = "convAE_v2"
    # Encoder
    input_img = Input(shape=(128, 128, 3))

    x = Conv2D(128, (3, 3), activation="relu", padding="same", strides=2)(input_img)
    # output image = (64, 64, 128)
    x = Conv2D(128, (3, 3), activation="relu", padding="same", strides=2)(x)
    # output image = (32, 32, 128)
    x = Conv2D(256, (3, 3), activation="relu", padding="same", strides=2)(x)
    # output image = (16, 16, 256)
    x = BatchNormalization()(x)  # No effect on output image
    # output image = (16, 16, 256)

    # Decoder
    x = Conv2DTranspose(128, (3, 3), activation="relu", padding="same", strides=2)(x)
    # output image = (32,32,512)
    x = Conv2DTranspose(64, (3, 3), activation="relu", padding="same", strides=2)(x)
    # ouptup image = (64, 64, 256)
    x = Conv2DTranspose(3, (3, 3), activation="sigmoid", padding="same", strides=2)(x)
    # ouptup image = (128, 128, 3)

    ae = Model(input_img, x)
    ae.compile(optimizer="adam", loss="mse")
    ae.summary()

    train_ds, val_ds = my_data()
    early_stopping = EarlyStopping(
        monitor="val_loss", min_delta=0, patience=10, verbose=5, mode="auto"
    )
    history = ae.fit(
        train_ds,
        epochs=50,
        validation_data=val_ds,
        callbacks=[early_stopping],
        verbose=1,
    )
    ae.save(f"models/{model_name}.h5")

    with open(f"histories/{model_name}.txt", "w") as f:
        f.write(f"{history}")


def main():
    accordion_vae()


if __name__ == "__main__":
    main()
