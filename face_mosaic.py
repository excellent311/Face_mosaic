import cv2 as cv
import numpy as np
import os
import math

import logging
LOG_FORMAT = "************%(asctime)s - %(levelname)s- %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def mosaic(image, coordinate, kernel_size=3):
    """
    mosaic for the specified area which definition by coordinate

    input: image , corrdinate [x1, y1, x2, y2]
    output: mosaiced image
    """
    x1, y1, x2, y2 = coordinate
    image_h, image_w, image_channels = image.shape
    
    num_h = math.floor(abs(y2 - y1) / kernel_size)
    mod_h = abs(y2 - y1) % kernel_size
    num_w = math.floor(abs(x2 - x1) / kernel_size)
    mod_w = abs(x2 - x1) % kernel_size

    for y in range(num_h):
        for x in range(num_w):
            image[y*kernel_size:y*kernel_size+kernel_size, x*kernel_size:x*kernel_size+kernel_size,:] = \
                np.mean(
                    image[y*kernel_size:y*kernel_size+kernel_size, x*kernel_size:x*kernel_size+kernel_size,:],
                    axis=(0, 1)
                )
    return image

def load_image(image_path):
    """
    load image from image_path

    input: image_path
    output: image
    """
    if os.path.exists(image_path):
        image = cv.imread(image_path)
        print("loading image：{}".format(image_path))
        logging.info("image type{}".format(type(image)))
        return image
    else:
        print("image file not exists")
if __name__ == "__main__":
    image_path = "test.jpg"
    image = load_image(image_path)
    coor = [50,100,500,800]
    image = mosaic(image, coor, kernel_size=10)
    cv.imshow('test', image)
    cv.waitKey()
