# performs feature importance evaluation
# for the model produced by train_and_evaluate.py

import os
import numpy as np
from config import LOG_FOLDER, FIGURES_FOLDER
import pickle
from cosfire_dtcwt_sift_lssvm import (
    load_and_preprocess_images,
    extract_log_gabor_features,
    extract_dtcwt_features,
    extract_dense_sift_features,
    extract_all_features,
)
import json
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd


def log_time(message, path=LOG_FOLDER):
    """Print message with current timestamp"""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}] {message}")
    with open(os.path.join(path, "output_feature_importance.log"), "a") as f:
        f.write(f"[{current_time}] {message}\n")


def main():
    # load the model
    with open("gender_classifier_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("feature_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    original_accuracy = 0.92
    num_trials = 5

    # load the data
    log_time("Loading data...")
    male_images, male_labels = load_and_preprocess_images("Dataset/Testing/male", 1)
    female_images, female_labels = load_and_preprocess_images(
        "Dataset/Testing/female", 0
    )
    log_time("Data loaded")

    X = male_images + female_images
    y = male_labels + female_labels

    feature_types = ["Log-Gabor", "DTCWT", "SIFT"]

    importances_mean = {}
    importances_full = {}

    log_time("Evaluating feature importance...", path=LOG_FOLDER)

    # ! method 1
    # features = {"Log-Gabor": [], "DTCWT": [], "SIFT": []}
    # log_time("Extracting features...")
    # features["Log-Gabor"] = np.array([extract_log_gabor_features(img) for img in X])
    # features["DTCWT"] = np.array([extract_dtcwt_features(img) for img in X])
    # features["SIFT"] = np.array([extract_dense_sift_features(img) for img in X])
    # log_time("Features extracted")

    # # # scale features
    # # log_time("Scaling features...")
    # # features["Log-Gabor"] = scaler.transform(features["Log-Gabor"])
    # # features["DTCWT"] = scaler.transform(features["DTCWT"])
    # # features["SIFT"] = scaler.transform(features["SIFT"])
    # # log_time("Features scaled")

    # log_time("Shuffling features...")
    # features["Log-Gabor_shuffled"] = np.random.shuffle(features["Log-Gabor"].copy())
    # features["DTCWT_shuffled"] = np.random.shuffle(features["DTCWT"].copy())
    # features["SIFT_shuffled"] = np.random.shuffle(features["SIFT"].copy())
    # log_time("Features shuffled")
    # for feature_type in feature_types:
    #     log_time(f"Feature type: {feature_type}", path=LOG_FOLDER)
    #     running_features = []
    #     for feature_name in feature_types:
    #         if feature_name == feature_type:
    #             running_features.append(feature_name + "_shuffled")
    #         else:
    #             running_features.append(feature_name)
    #     X = np.array(
    #         [
    #             np.concatenate([features[feature][i] for feature in running_features])
    #             for i in range(len(X))
    #         ]
    #     )
    #     # scale features
    #     X_scaled = scaler.transform(X)
    #     accuracy = model.score(X_scaled, y)
    #     log_time(f"Accuracy: {accuracy}", path=LOG_FOLDER)
    #     # find the relative importance of the feature type
    #     importance = accuracy - original_accuracy
    #     log_time(f"Importance: {importance}", path=LOG_FOLDER)
    #     importances[feature_type] = importance

    # ! method 2

    log_gabor_dim = extract_log_gabor_features(X[0]).shape[0]
    dt_cwt_dim = extract_dtcwt_features(X[0]).shape[0]
    sift_dim = extract_dense_sift_features(X[0]).shape[0]

    log_gabor_indices = np.arange(log_gabor_dim)
    dt_cwt_indices = np.arange(log_gabor_dim, log_gabor_dim + dt_cwt_dim)
    sift_indices = np.arange(
        log_gabor_dim + dt_cwt_dim, log_gabor_dim + dt_cwt_dim + sift_dim
    )
    feature_indices = {
        "Log-Gabor": log_gabor_indices,
        "DTCWT": dt_cwt_indices,
        "SIFT": sift_indices,
    }

    log_time("Extracting features...")
    X = np.array([extract_all_features(img) for img in X])
    log_time("Features extracted")

    log_time("Scaling features...")
    X_scaled = scaler.transform(X)
    log_time("Features scaled")

    for feature_type in feature_types:
        log_time(f"Feature type: {feature_type}", path=LOG_FOLDER)
        running_accuracies = []
        for trial in range(num_trials):
            # shuffle rows of X in designated feature_indices
            log_time(f"Shuffling features... trial {trial}", path=LOG_FOLDER)
            X_shuffled = X_scaled.copy()
            X_shuffled[:, feature_indices[feature_type]] = np.random.permutation(
                X_shuffled[:, feature_indices[feature_type]]
            )
            log_time("Features shuffled", path=LOG_FOLDER)
            accuracy = model.score(X_shuffled, y)
            log_time(f"Accuracy: {accuracy}", path=LOG_FOLDER)
            running_accuracies.append(accuracy)
        importances_mean[feature_type] = original_accuracy - np.mean(running_accuracies)
        importances_full[feature_type] = original_accuracy - np.array(
            running_accuracies
        )
        log_time(f"Importance: {importances_mean[feature_type]}", path=LOG_FOLDER)

    log_time("Feature importance evaluation completed", path=LOG_FOLDER)

    # draw and save barchart for importances
    log_time("Drawing barchart...", path=LOG_FOLDER)
    fig, ax = plt.subplots()
    ax.bar(importances_mean.keys(), importances_mean.values())
    if not os.path.exists(FIGURES_FOLDER):
        os.makedirs(FIGURES_FOLDER)
    fig.savefig(os.path.join(FIGURES_FOLDER, "feature_importances.png"))
    log_time("Barchart saved", path=LOG_FOLDER)

    # draw and save boxplot for importances_full
    log_time("Drawing boxplot...", path=LOG_FOLDER)
    fig, ax = plt.subplots()
    ax.boxplot(list(importances_full.values()), labels=list(importances_full.keys()))
    fig.savefig(os.path.join(FIGURES_FOLDER, "feature_importances_full.png"))
    log_time("Boxplot saved", path=LOG_FOLDER)

    # save importances
    log_time("Saving importances...", path=LOG_FOLDER)
    with open("feature_importances.json", "w") as f:
        json.dump(importances_mean, f)
    log_time("Importances saved", path=LOG_FOLDER)

    importances_full_df = pd.DataFrame(importances_full)
    importances_full_df.to_csv("feature_importances_full.csv", index=False)
    log_time("Importances full saved", path=LOG_FOLDER)


if __name__ == "__main__":
    main()
