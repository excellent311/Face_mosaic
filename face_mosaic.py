import cv2 as cv
import numpy as np
import os
import math
import random

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
    

def mosaic(image, coordinate, kernel_size=10):
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
            image[y1+y*kernel_size:y1+y*kernel_size+kernel_size, x1+x*kernel_size:x1+x*kernel_size+kernel_size,:] = \
                np.mean(
                    image[y1+y*kernel_size:y1+y*kernel_size+kernel_size, x1+x*kernel_size:x1+x*kernel_size+kernel_size,:],
                    axis=(0, 1)
                )
    return image


def load_image(image_path, is_png=False):
    """
    load image from image_path

    input: image_path
    output: image
    """
    if os.path.exists(image_path):
        if is_png:
            image = cv.imread(image_path, cv.IMREAD_UNCHANGED)
        else:
            image = cv.imread(image_path)
        print("loading image：{}".format(image_path))
        return image
    else:
        print("image file not exists")


def mosaic_videocapture():
    """
    make mosaic for your videocapture
    input, output: none
    """
    cap = cv.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        faces = face_detection(frame)
        for face in faces:
            frame = mosaic(frame, face, kernel_size=15)
        cv.imshow('frame', frame)
        if cv.waitKey(1) &0xFF == ord(' '):
    	    break
    cap.release()
    cv.destroyAllWindows()


def mosaic_image(image_path):
    image = load_image(image_path)
    faces = face_detection(image)
    for face in faces:
        image = mosaic(image, face, kernel_size=10)
    cv.imshow('{}'.format(image_path), image)
    cv.waitKey()

def load_fake_face_png(index):
    """
    retrurn fake_face_png
    """
    fake_face_png_path = os.path.join('data', 'fake_png', '{}.png'.format(index))
    return load_image(fake_face_png_path, is_png=True)


def change_face_image(image, face_image, face_coor):
    #TODO:
    
    def change_rate(rate, face_coor):
        x1, y1, x2, y2 = face_coor
        x_c = int((x1+x2)/2);y_c = int((y1+y2)/2)
        x_2 = int(abs(x2-x1)/2*rate);y_2 = int(abs(y1-y2)/2*rate)
        x1 = x_c - x_2
        y1 = y_c - y_2
        x2 = x_c + x_2
        y2 = y_c + y_2
        return x1, y1, x2, y2
    
    x1, y1, x2, y2 = change_rate(1, face_coor)
    # change dtype
    face_h = abs(y2-y1);face_w = abs(x2-x1)
    face_image = cv.resize(face_image, (face_h, face_w)).astype('uint8')
    face_image_rgb = face_image[:,:,:3]
    face_iamge_alpha = face_image[:,:,-1]
    print(np.max(face_iamge_alpha))
    face_boolen_Flase = face_iamge_alpha == np.zeros((face_h, face_w))
    face_boolen_Flase = np.stack([face_boolen_Flase, face_boolen_Flase, face_boolen_Flase],axis=2)
    print(image[y1:y1+face_h,x1:x1+face_w,:].shape,)
    image[y1:y1+face_h,x1:x1+face_w,:] = image[y1:y1+face_h,x1:x1+face_w,:]*face_boolen_Flase+face_image_rgb
    return image


def change_face(image):
    #image_path = "test.jpg"
    image = load_image(image_path)
    faces = face_detection(image)
    faces_random_index = random.sample(range(15),2)
    for i, face in enumerate(faces):
        face_image = face_image = load_fake_face_png(faces_random_index[i])
        image = change_face_image(image,face_image,face)
    return image

def change_face_videocapture():
    cap = cv.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        frame = change_face(frame)
        if cv.waitKey(1) &0xFF == ord(' '):
    	    break
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    image_path = "test.jpg"
    #mosaic_videocapture()
    image = change_face(image_path)
    cv.imshow('t', image)
    cv.waitKey()

    
    

