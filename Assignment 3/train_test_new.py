import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import normalize, StandardScaler

import json

from utils import (
    log_print,
    preprocess_and_extract_features_new,
    load_preprocessed_images,
)

from config import (
    RESULTS_FOLDER,
    FIGURES_FOLDER,
    HYPERPARAMETERS_FOLDER,
    PREPROCESSED_TRAIN_PATH,
    PREPROCESSED_TEST_PATH,
    PREPROCESSED_FOLDER,
    IMG_SIZE,
    KERNEL_SIZE,
    CANNY_LOW_THRESHOLD,
    CANNY_HIGH_THRESHOLD,
    NEW_PARAMS,
    LOG_FOLDER,
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
# define logs folder inside results folder
if not os.path.exists(LOG_FOLDER):
    os.makedirs(LOG_FOLDER)

log_file = os.path.join(LOG_FOLDER, "output_new.log")
# load preprocessed images
log_print("Loading preprocessed images", log_file)
train_images, train_labels = load_preprocessed_images(PREPROCESSED_TRAIN_PATH)
test_images, test_labels = load_preprocessed_images(PREPROCESSED_TEST_PATH)

log_print("Preprocessed images loaded", log_file)

features = ["hog", "sift", "dtcwt"]

# extract features
log_print("Extracting features", log_file)
train_features = preprocess_and_extract_features_new(
    train_images,
    img_size=IMG_SIZE,
    kernel_size=KERNEL_SIZE,
    canny_low_threshold=CANNY_LOW_THRESHOLD,
    canny_high_threshold=CANNY_HIGH_THRESHOLD,
    preprocessed=True,
    features=features,
)
test_features = preprocess_and_extract_features_new(
    test_images,
    img_size=IMG_SIZE,
    kernel_size=KERNEL_SIZE,
    canny_low_threshold=CANNY_LOW_THRESHOLD,
    canny_high_threshold=CANNY_HIGH_THRESHOLD,
    preprocessed=True,
    features=features,
)
# normalize the features
# train_features = normalize(train_features)
# test_features = normalize(test_features)

log_print("Features extracted", log_file)
# define the pipeline
pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("pca", PCA()),
        ("classifier", SGDClassifier(loss="hinge")),
    ]
)

# perform grid search
log_print("Performing grid search with the preprocessed features", log_file)
grid_search = GridSearchCV(pipeline, NEW_PARAMS, cv=5, scoring="f1_macro")
grid_search.fit(train_features, train_labels)


log_print("Evaluating the classifier", log_file)
log_print(f"Best parameters: {grid_search.best_params_}", log_file)
log_print(f"Best score: {grid_search.best_score_}", log_file)

# # save the best parameters to a file
with open(os.path.join(HYPERPARAMETERS_FOLDER, "new_best_params.json"), "w") as f:
    json.dump(grid_search.best_params_, f)

# evaluate the classifier on the test set
test_predictions = grid_search.predict(test_features)
log_print(f"Test accuracy: {accuracy_score(test_labels, test_predictions)}", log_file)
log_print(
    f"Test F1 score: {f1_score(test_labels, test_predictions, average='macro')}",
    log_file,
)
log_print(classification_report(test_labels, test_predictions), log_file)
log_print(confusion_matrix(test_labels, test_predictions), log_file)
