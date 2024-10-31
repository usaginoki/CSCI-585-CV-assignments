import os

DATASET_PATH = "Dataset"
TRAIN_PATH = os.path.join(DATASET_PATH, "Training")
TEST_PATH = os.path.join(DATASET_PATH, "Testing")
# define the list of classes
CLASSES = os.listdir(TRAIN_PATH)

RESULTS_FOLDER = "results"
FIGURES_FOLDER = os.path.join(RESULTS_FOLDER, "figures")
HYPERPARAMETERS_FOLDER = os.path.join(RESULTS_FOLDER, "hyperparameters")
LOG_FOLDER = os.path.join(RESULTS_FOLDER, "logs")

PREPROCESSED_FOLDER = os.path.join(RESULTS_FOLDER, "preprocessed")
PREPROCESSED_TRAIN_PATH = os.path.join(PREPROCESSED_FOLDER, "Training")
PREPROCESSED_TEST_PATH = os.path.join(PREPROCESSED_FOLDER, "Testing")

IMG_SIZE = (80, 80)
KERNEL_SIZE = (5, 5)
CANNY_LOW_THRESHOLD = 100
CANNY_HIGH_THRESHOLD = 200

OLD_PARAMS = {
    "pca__n_components": [50, 100],
    "classifier__n_neighbors": [3, 9],
    "classifier__weights": ["uniform", "distance"],
    "classifier__metric": ["euclidean"],
}

NEW_PARAMS = {
    "pca__n_components": [50, 100],
    "classifier__penalty": ["l2"],
    "classifier__alpha": [0.0001],
    "classifier__learning_rate": ["optimal"],
}
