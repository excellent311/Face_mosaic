import cv2 as cv
import numpy as np
import os
import math

import logging
LOG_FORMAT = "************%(asctime)s - %(levelname)s- %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

def face_detection(image, classifier_path="haarcascade_frontalface_default.xml"):
    """
    find all face in input image and return face by coordinate(list)

    input: image, classifier file path (.xml) and have default path
    output: list coordinate [x1, y1, x2, y2]
    """
    classifier_path = os.path.join('data', classifier_path)
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    detector = cv.CascadeClassifier(classifier_path)
    faces = detector.detectMultiScale(gray_image)
    for i, face in enumerate(faces):
        faces[i] = [face[0], face[1], face[0]+face[2], face[1]+ face[3]]
    return faces
    

def mosaic(image, coordinate, kernel_size=3):
    """
    mosaic for the specified area which definition by coordinate

    input: image , corrdinate [x1, y1, x2, y2]
    output: mosaiced image
    """
    x1, y1, x2, y2 = coordinate
    image_h, image_w, image_channels = image.shape
    print(x1, y1, x2, y2)
    num_h = math.floor(abs(y2 - y1) / kernel_size)
    mod_h = abs(y2 - y1) % kernel_size
    num_w = math.floor(abs(x2 - x1) / kernel_size)
    mod_w = abs(x2 - x1) % kernel_size
    print(num_h,mod_h,num_w,mod_w)
    for y in range(num_h):
        for x in range(num_w):
            image[y1+y*kernel_size:y1+y*kernel_size+kernel_size, x1+x*kernel_size:x1+x*kernel_size+kernel_size,:] = \
                np.mean(
                    image[y1+y*kernel_size:y1+y*kernel_size+kernel_size, x1+x*kernel_size:x1+x*kernel_size+kernel_size,:],
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
    image_path = "3.jpg"
    image = load_image(image_path)

    faces = face_detection(image)
    for face in faces:
        image = mosaic(image, face, kernel_size=3)
    cv.imshow('test', image)
    cv.waitKey()
