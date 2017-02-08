import numpy as np
import skimage.transform as skt
import cv2
from random import randrange

def normalize_image(image):
  '''
  Normalize image in the range (-0.5, 0.5)
  :param image: original image
  :return: normalized image
  '''
  return (image - 127.5)/255.0


def crop_image(image):
  '''
  Crop the image Y: rows 60 to 120
  :param image: image array to crop
  :return: cropped image rows 60 to 120.
  '''
  return image[60:120,...]


def modify_brightness(image):
  '''
  Randomly modify the brightness of the image.
  :param image: image to change the brightness
  :return: randomly modified brightness image
  '''
  image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
  brightness_factor = np.random.uniform(low=0.3, high=1.3)
  image_hsv[...,2] = np.minimum(image_hsv[...,2] * brightness_factor, 255)
  image_bgr = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)
  return image_bgr


def translate_image_random(image, steering_angle):
  '''
  Randomly translate the image up to 40 pixels left or right.
  :param image: image to translate
  :param steering_angle: initial steering angle for the input image
  :return: tuple with the translated image and the adjusted steering angle for the translation
  '''
  max_translation = 40.0
  x_translation = max_translation * np.random.uniform(low=-1.0, high=1.0)
  new_steering_angle = steering_angle + (x_translation/max_translation) * 4/25.0
  transform = skt.SimilarityTransform(translation=[x_translation, 0])
  translated_image = np.array(skt.warp(image, transform) * 255, dtype=np.uint8)
  return translated_image, new_steering_angle


def preprocess_image(image_path, steering_angle):
  '''
  Load the image from image_path and preprocess:
  (1) crop;
  (2) randomly modify brightness
  (3) randomly translate image left/right
  (4) randomly flip image towards y-axis
  (5) convert image from BGR to YUV color space
  (6) normalize image between (-0.5, 0.5)
  :param image_path: path to image file
  :param steering_angle: original steering angle
  :return: preprocessed image and adjusted steering angle
  '''
  image = cv2.imread(image_path)
  image = crop_image(image)
  image = modify_brightness(image)
  image, steering_angle = translate_image_random(image, steering_angle)

  flip = randrange(2)
  if (flip == 1):
    image = cv2.flip(image, 1)
    steering_angle *= -1.0

  image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
  image = normalize_image(image)
  return image, steering_angle