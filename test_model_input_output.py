from glob import glob

from PIL import Image, ImageDraw, ImageFont

from latent_interpolation import (get_ae, get_vae, run_input_output_img_ae,
                                  run_input_output_imgs)


def test_normal_ae():
    files = glob("./archive/images/*.jpg")[:5]
    model_name = "./models/convAE_v1.h5"
    ae = get_ae(model_name)
    in_img, out_img = run_input_output_img_ae(ae, files[0])
    in_img.save("test1.jpg")
    out_img.save("test2.jpg")


def test_vae():
    files = glob("./archive/images/*.jpg")[:5]
    model_names = [
        "./models/bigVAE_256.h5",
        "./models/bigVAE_128.h5",
        "./models/bigVAE_64.h5",
    ]
    texts = ["256 latent variables", "128 latent variables", "64 latent variables"]
    new_im = Image.new("RGB", (768, 675))

    x_col_offset = 0
    for text, model_name in zip(texts, model_names):
        encoder, decoder = get_vae(model_name)

        y_offset = 20
        for i, file in enumerate(files):
            input_img, output_img, latent = run_input_output_imgs(
                encoder, decoder, file
            )

            x_offset = x_col_offset
            y_size = 0
            for im in [input_img, output_img]:
                new_im.paste(im, (x_offset, y_offset))
                x_offset += im.size[0]
                y_size = im.size[1]

            y_offset += y_size
        x_col_offset += 256 + 5

    font = ImageFont.truetype("/Library/Fonts/Arial Unicode.ttf", 16)
    draw = ImageDraw.Draw(new_im)
    draw.text((20, 0), texts[0], (255, 255, 255), font=font)

    draw2 = ImageDraw.Draw(new_im)
    draw2.text((281, 0), texts[1], (255, 255, 255), font=font)

    draw3 = ImageDraw.Draw(new_im)
    draw3.text((542, 0), texts[2], (255, 255, 255), font=font)

    new_im.save("blury_test.jpg")


def main():
    test_normal_ae()


if __name__ == "__main__":
    main()
