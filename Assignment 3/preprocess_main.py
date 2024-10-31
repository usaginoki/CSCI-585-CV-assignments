import cv2 as cv
import regex as re
import os
import numpy as np

import json

from utils import (
    log_print,
    load_images,
    preprocess,
    preprocess_and_save_images,
)

from config import (
    RESULTS_FOLDER,
    FIGURES_FOLDER,
    HYPERPARAMETERS_FOLDER,
    TRAIN_PATH,
    TEST_PATH,
    PREPROCESSED_TRAIN_PATH,
    PREPROCESSED_TEST_PATH,
    PREPROCESSED_FOLDER,
    CLASSES,
    IMG_SIZE,
    KERNEL_SIZE,
    CANNY_LOW_THRESHOLD,
    CANNY_HIGH_THRESHOLD,
)

# define folder for saving the results
if not os.path.exists(RESULTS_FOLDER):
    os.makedirs(RESULTS_FOLDER)
# define figures folder inside results folder
if not os.path.exists(FIGURES_FOLDER):
    os.makedirs(FIGURES_FOLDER)
# define hyperparameters folder inside results folder
if not os.path.exists(HYPERPARAMETERS_FOLDER):
    os.makedirs(HYPERPARAMETERS_FOLDER)
# define preprocessed folder inside results folder
if not os.path.exists(PREPROCESSED_FOLDER):
    os.makedirs(PREPROCESSED_FOLDER)
if not os.path.exists(PREPROCESSED_TRAIN_PATH):
    os.makedirs(PREPROCESSED_TRAIN_PATH)
if not os.path.exists(PREPROCESSED_TEST_PATH):
    os.makedirs(PREPROCESSED_TEST_PATH)

log_print("Starting the script")
# define the path to the dataset
train_path = TRAIN_PATH
test_path = TEST_PATH
# define the list of classes
classes = CLASSES


log_print("Loading the images")
train_images, train_labels = load_images(train_path, classes)
test_images, test_labels = load_images(test_path, classes)

log_print("Images loaded")

# explore the old (lame) feature and classification methods
log_print("Exploring the old (lame) feature and classification methods")
log_print(
    "Defining the preprocess function: converting to grayscale, resizingand blurring"
)

log_print("Preprocessing for train and test images")
img_size = IMG_SIZE
kernel_size = KERNEL_SIZE
canny_low_threshold = CANNY_LOW_THRESHOLD
canny_high_threshold = CANNY_HIGH_THRESHOLD

# preprocess images and save them
log_print("Preprocessing images and saving them")
preprocessed_train = preprocess_and_save_images(
    train_images,
    train_labels,
    PREPROCESSED_TRAIN_PATH,
    img_size=img_size,
    kernel_size=kernel_size,
)
preprocessed_test = preprocess_and_save_images(
    test_images,
    test_labels,
    PREPROCESSED_TEST_PATH,
    img_size=img_size,
    kernel_size=kernel_size,
)
