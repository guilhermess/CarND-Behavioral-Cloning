import tensorflow as tf
import numpy as np
import argparse
import os
import csv
import cv2
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.utils import shuffle
import random
from collections import namedtuple
tf.python.control_flow_ops = tf
import frame_processing as fp

ImageAngle = namedtuple('ImageRecord', 'image, steering_angle')

def model():
  '''
  The model implemented by this method is heavily based on the following paper from NVIDIA:
  Mariusz Bojarski et al., End to End Learning for Self-Driving Cars

  The following changes were made to the NVIDIA architecture:
  1) Input is 60x320x3 instead of 66x200x3
  2) Added 3 dropout layers between the fully connected layers.

  :return: NVIDIA-based Keras deep neural network model for training a self driving car
  '''
  model = Sequential()
  model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2,2), activation='relu', input_shape=(60,320,3)))
  model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2,2), activation='relu'))
  model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2,2), activation='relu'))
  model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1,1), activation='relu'))
  model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1,1), activation='relu'))
  model.add(Flatten())
  model.add(Dense(1164, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(100, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(50, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(10, activation='relu'))
  model.add(Dense(1))
  return model


def split_training_and_validation_data(csv_filename, validation_rate=0.2):
  '''
  This method reads the specified CSV file and splits the data in validation and training data.
  The validation images are read into memory since and returned as numpy arrays. The training data is stored as
  path to the images and associated steering angles to be used by the generator.
  :param csv_filename: path to CSV file containing images and steering angle information
  :param validation_rate: split for validation, should be a float between (0.0, 1.0), default 0.2
  :return: tuple with (1) list of ImageAngle records for training, (2) validation x numpy array, (3) validation y
  '''
  with open(csv_filename) as csvfile:
    path = os.path.dirname(csv_filename)
    csv_reader = csv.reader(csvfile, delimiter=',')
    iter_csv = iter(csv_reader)
    next(iter_csv)
    image_path_angle = []

    for row in iter_csv:
      center_img_name = row[0].strip()
      left_img_name = row[1].strip()
      right_img_name = row[2].strip()
      steering_angle = float(row[3])
      path_to_center_img = path + '/' + center_img_name
      path_to_left_img = path + '/' + left_img_name
      path_to_right_img = path + '/' + right_img_name

      image_angle_record = [ImageAngle(path_to_center_img, steering_angle),
                            ImageAngle(path_to_left_img, steering_angle + 0.25),
                            ImageAngle(path_to_right_img, steering_angle - 0.25)]
      image_path_angle.append(image_angle_record)

    image_path_angle = shuffle(image_path_angle)
    num_validation = int(validation_rate * len(image_path_angle))
    validation_image_paths = image_path_angle[:num_validation]
    training_image_path_angle = image_path_angle[num_validation:]

    validation_x = []
    validation_y = []
    for i in range(num_validation):
      center_left_right_index = random.randrange(3)
      image_path = validation_image_paths[i][center_left_right_index].image
      steering_angle = validation_image_paths[i][center_left_right_index].steering_angle
      image, steering_angle = fp.preprocess_image(image_path, steering_angle)
      validation_x.append(image)
      validation_y.append(steering_angle)

    np_validation_x = np.array(validation_x)
    np_validation_y = np.array(validation_y)

    return training_image_path_angle,  np_validation_x, np_validation_y

def image_steering_angle_generator(training_image_paths, batch_size):
  '''
  Method used by the Keras generator to read the data from persistent storage with a given batch size, preprocess the
  data and feed to the model for traning.
  :param training_image_paths: training dataset records containing path to image files and steering angles
  :param batch_size: batch size to be returned by the generator
  :return: numpy batch with training image data and steering angles
  '''

  while(1):
    batch_x = []
    batch_y = []
    for batch_index in range(batch_size):
      image_index = random.randrange(len(training_image_paths))
      center_left_right_index = random.randrange(3)
      image_path = training_image_paths[image_index][center_left_right_index].image
      steering_angle = training_image_paths[image_index][center_left_right_index].steering_angle
      image, steering_angle = fp.preprocess_image(image_path, steering_angle)
      batch_x.append(image)
      batch_y.append(steering_angle)

    np_batch_x = np.array(batch_x, dtype=np.float32)
    np_batch_y = np.array(batch_y, dtype=np.float32)
    yield np_batch_x, np_batch_y


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Self-Driving Simulator Model Training')
    parser.add_argument('-csv', help='Path to csv file', action='store', default='')
    parser.add_argument('-batch_size', help='', action='store', default=64)
    parser.add_argument('-nb_epoch', help='', action='store', default=40)
    parser.add_argument('-learning_rate', help='', action='store', default=1e-4)
    parser.add_argument('-samples_per_epoch', help='', action='store', default=13632)
    parser.add_argument('-validation_split', help='', action='store', default=0.2)
    parser.add_argument('-model_name', help='', action='store', default='model')
    args = parser.parse_args()

    model = model()
    model.compile(Adam(args.learning_rate), 'mse')
    model.summary()

    train_images_angles, x_validation, y_validation = split_training_and_validation_data(args.csv, args.validation_split)

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')

    history = model.fit_generator(image_steering_angle_generator(train_images_angles, args.batch_size),
                                  validation_data=(x_validation, y_validation),
                                  callbacks=[early_stopping],
                                  nb_epoch=args.nb_epoch, samples_per_epoch=args.samples_per_epoch)

    json_file = open(args.model_name + ".json", 'w')
    json_file.write(model.to_json())
    json_file.close()

    model.save_weights(args.model_name + ".h5")

