from PIL import Image
import numpy as np
import gcm
import otsu
import time


def apply_gamma_correction(image_matrix, gamma) -> np.array:
    """
    Compute matrix of new gamma-corrected
    image with the given value
    :param np.array image_matrix: matrix of image
    :param float gamma : given gamma
    :return: np.array
    """

    # cf https://en.wikipedia.org/wiki/Gamma_correction
    gamma_image = Image.fromarray(image_matrix)
    height, width = gamma_image.size
    for row in range(height):
        for column in range(width):
            red = min(round(pow(gamma_image.getpixel((row, column))[0] / 255, gamma) * 255), 255)
            green = min(round(pow(gamma_image.getpixel((row, column))[1] / 255, gamma) * 255), 255)
            blue = min(round(pow(gamma_image.getpixel((row, column))[2] / 255, gamma) * 255), 255)
            gamma_image.putpixel((row, column), (red, green, blue))
    return np.array(gamma_image)


def binarization(file_name):
    name = str(file_name[:-4])
    image_matrix = np.array(Image.open(file_name))
    gamma = gcm.gamma_correction_method(image_matrix)

    gamma_image_matrix = apply_gamma_correction(image_matrix, gamma)
    Image.fromarray(gamma_image_matrix).save(name + "_gamma.png")

    binary_image_matrix = otsu.apply_otsu_threshold(gamma_image_matrix)
    Image.fromarray(binary_image_matrix).save(name + "_binary.png")


Image.fromarray(apply_gamma_correction(np.array(Image.open("22.png")), 0.1)).save("Collins_1.png")
Image.fromarray(apply_gamma_correction(np.array(Image.open("22.png")), 1.0)).save("Collins_2.png")
Image.fromarray(apply_gamma_correction(np.array(Image.open("22.png")), 2.0)).save("Collins_3.png")
Image.fromarray(apply_gamma_correction(np.array(Image.open("22.png")), 2.5)).save("Collins_4.png")
Image.fromarray(apply_gamma_correction(np.array(Image.open("22.png")), 5.0)).save("Collins_5.png")
Image.fromarray(apply_gamma_correction(np.array(Image.open("22.png")), 7.5)).save("Collins_6.png")
Image.fromarray(apply_gamma_correction(np.array(Image.open("22.png")), 10.0)).save("Collins_7.png")


start_time = time.time()
binarization("Collins.png")
print("-----%s seconds------" % (time.time() - start_time))
