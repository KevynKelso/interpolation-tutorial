from glob import glob

from PIL import Image, ImageDraw, ImageFont

from gui_ml import (get_ae, get_vae, run_image_ae, run_input_output_img_ae,
                    run_input_output_imgs)

IMG_WIDTH = 128
IMG_HEIGHT = 128
TEXT_SIZE = 20
COL_WIDTH = 5


def get_blank_img_table(rows, cols):
    return Image.new(
        "RGB", ((IMG_WIDTH + COL_WIDTH) * cols, IMG_HEIGHT * rows + TEXT_SIZE)
    )


def test_stacked():
    num_imgs = 5
    files = glob("./archive/images/*.jpg")[:num_imgs]
    vae_model_name = "./models/bigVAE_256.h5"
    ae_model_name = "./models/convAE_v2.h5"

    ae = get_ae(ae_model_name)
    vae = get_vae(vae_model_name)

    new_im = get_blank_img_table(num_imgs, 7)

    left_col = [Image.open(f).resize((IMG_WIDTH, IMG_HEIGHT)) for f in files]
    mid_col = [run_input_output_imgs(vae[0], vae[1], f)[1] for f in files]
    right_col = [run_image_ae(ae, img) for img in mid_col]
    right2_col = [run_image_ae(ae, img) for img in right_col]
    right3_col = [run_image_ae(ae, img) for img in right2_col]
    right4_col = [run_image_ae(ae, img) for img in right3_col]
    right5_col = [run_image_ae(ae, img) for img in right4_col]

    write_image_column(new_im, left_col)
    write_image_column(new_im, mid_col, col_num=1)
    write_image_column(new_im, right_col, col_num=2)
    write_image_column(new_im, right2_col, col_num=3)
    write_image_column(new_im, right3_col, col_num=4)
    write_image_column(new_im, right4_col, col_num=5)
    write_image_column(new_im, right5_col, col_num=6)

    font = ImageFont.truetype("/Library/Fonts/Arial Unicode.ttf", 16)
    draw = ImageDraw.Draw(new_im)
    draw.text((10, 0), "Input", (255, 255, 255), font=font)

    draw2 = ImageDraw.Draw(new_im)
    draw2.text((138, 0), "OutVAE/InAE", (255, 255, 255), font=font)

    draw3 = ImageDraw.Draw(new_im)
    draw3.text((266, 0), "Output AE", (255, 255, 255), font=font)

    new_im.save("vae_ae.jpg")


def test_normal_ae():
    num_imgs = 5
    files = glob("./archive/images/*.jpg")[:num_imgs]
    model_name = "./models/convAE_v2.h5"
    ae = get_ae(model_name)

    new_im = get_blank_img_table(num_imgs, 2)

    left_col = [run_input_output_img_ae(ae, f)[0] for f in files]
    right_col = [run_input_output_img_ae(ae, f)[1] for f in files]

    write_image_column(new_im, left_col)
    write_image_column(new_im, right_col, col_num=1)

    font = ImageFont.truetype("/Library/Fonts/Arial Unicode.ttf", 16)
    draw = ImageDraw.Draw(new_im)
    draw.text((20, 0), "Input", (255, 255, 255), font=font)

    draw2 = ImageDraw.Draw(new_im)
    draw2.text((133, 0), "Output", (255, 255, 255), font=font)

    new_im.save("convAE_v2_inout.jpg")


def write_image_column(new_image, images, col_num=0):
    y_offset = TEXT_SIZE
    for im in images:
        new_image.paste(im, (IMG_WIDTH * col_num + COL_WIDTH * col_num, y_offset))
        y_offset += IMG_HEIGHT

    return new_image


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
    test_stacked()
    # test_normal_ae()


if __name__ == "__main__":
    main()
