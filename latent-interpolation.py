import glob
import random
from tkinter import Button, Label, Scale, Tk

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageTk
from tensorflow.keras.models import Model, load_model

from data import IMG_HEIGHT, IMG_WIDTH

g_latent = []


def get_vae():
    autoencoder = load_model("./bigVAE_128")
    z = autoencoder.layers[10]

    encoder = Model(autoencoder.input, z.output)
    decoder = autoencoder.layers[11]

    return encoder, decoder


def encode_decode(encoder, decoder, img):
    # input image preprocessing
    # img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_arr = np.array(img) / 255
    batch = np.zeros((1, IMG_WIDTH, IMG_HEIGHT, 3))
    batch[0, :, :, :] = img_arr

    # actual ML
    latent = encoder(batch)
    output = decoder.predict(latent)

    return latent[0], (output * 255).astype(np.uint8)[0]


def run_input_output_imgs(encoder, decoder, file):
    img_pil = Image.open(file).resize((IMG_WIDTH, IMG_HEIGHT))
    latent, output_img = encode_decode(encoder, decoder, img_pil)

    tk_img_input = ImageTk.PhotoImage(img_pil)
    tk_img_output = ImageTk.PhotoImage(Image.fromarray(output_img))

    return tk_img_input, tk_img_output, latent


def set_scales_to_latent_vec(scales, latent_vec):
    latent_arr = np.array(latent_vec)
    for scale, latent_val in zip(scales, latent_arr):
        scale.set(latent_val)


def gui(files, encoder, decoder):
    ws = Tk()

    def change_img(input_img_label, output_img_label, scales, length):
        global g_latent
        input_img, output_img, latent = run_input_output_imgs(
            encoder, decoder, files[random.randint(0, length - 1)]
        )

        g_latent = latent

        input_img_label.configure(image=input_img)
        output_img_label.configure(image=output_img)
        input_img_label.image = input_img  # prevents garbage collection
        output_img_label.image = output_img

        set_scales_to_latent_vec(scales, g_latent)

    def decode_new_latent(output_img, scales):
        # need to build latent vector from scale values
        new_latent = [scale.get() for scale in scales]
        output = decoder.predict([new_latent])[0]
        tk_img_output = ImageTk.PhotoImage(
            Image.fromarray((output * 255).astype(np.uint8))
        )

        output_img.configure(image=tk_img_output)
        output_img.image = tk_img_output

    img_default_input, img_default_output, g_latent = run_input_output_imgs(
        encoder, decoder, files[0]
    )

    ws.title("Latent Space Interpolation GUI")

    input_image = Label(ws, image=img_default_input)
    output_image = Label(ws, image=img_default_output)

    scales = [
        Scale(
            ws,
            from_=-5,
            to=5,
            resolution=1e-10,
            showvalue=0,
            command=lambda _: decode_new_latent(output_image, scales),
        )
        for _ in range(g_latent.shape[0])
    ]

    set_scales_to_latent_vec(scales, g_latent)

    button = Button(
        ws,
        text="Change source image",
        command=lambda: change_img(input_image, output_image, scales, len(files)),
    )
    button2 = Button(
        ws,
        text="Reset sliders",
        command=lambda: set_scales_to_latent_vec(scales, g_latent),
    )

    button.grid(row=1, column=0)
    button2.grid(row=3, column=0)
    input_image.grid(row=0, column=0)
    output_image.grid(row=2, column=0)

    num_cols = 50
    for i, scale in enumerate(scales):
        scale.grid(row=(int(i / num_cols) % num_cols), column=((i + 1) % num_cols) + 1)

    ws.mainloop()


def main():
    encoder, decoder = get_vae()
    files = glob.glob("./archive/images/*.jpg")[:100]
    gui(files, encoder, decoder)


if __name__ == "__main__":
    main()
