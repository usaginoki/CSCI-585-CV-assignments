import os
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pickle
from cosfire_dtcwt_sift_lssvm import (
    load_and_preprocess_images,
    extract_all_features,
)
import time
from datetime import datetime
from config import LOG_FOLDER
from sklearn.linear_model import SGDClassifier


def log_time(message, path=LOG_FOLDER):
    """Print message with current timestamp"""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}] {message}")
    with open(os.path.join(path, "output.log"), "a") as f:
        f.write(f"[{current_time}] {message}\n")


def train_model():
    """Train the model and save it along with the scaler"""
    start_time = time.time()

    # Load training data
    log_time("Loading training data...")
    male_images, male_labels = load_and_preprocess_images("Dataset/Training/male", 1)
    female_images, female_labels = load_and_preprocess_images(
        "Dataset/Training/female", 0
    )
    log_time(f"Loaded {len(male_images)} male and {len(female_images)} female images")

    # Combine datasets
    X = male_images + female_images
    y = male_labels + female_labels

    # Extract features
    log_time("Extracting features from training images...")
    features = [extract_all_features(img) for img in X]
    X_features = np.array(features)
    log_time(f"Features extracted. Shape: {X_features.shape}")

    # Scale features
    log_time("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)

    # Train SGD Classifier
    log_time("Training SGD Classifier...")
    clf = SGDClassifier(
        loss="hinge",  # SVM-like loss function
        penalty="l2",
        alpha=0.0001,  # regularization strength
        max_iter=1000,
        random_state=42,
        learning_rate="optimal",
    )
    clf.fit(X_scaled, y)

    # Save model and scaler
    log_time("Saving model and scaler...")
    with open("gender_classifier_model.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open("feature_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    training_time = time.time() - start_time
    log_time(f"Total training time: {training_time:.2f} seconds")

    return clf, scaler


def evaluate_model(model, scaler):
    """Evaluate the model on test data"""
    start_time = time.time()

    log_time("Loading test data...")
    test_male_images, test_male_labels = load_and_preprocess_images(
        "Dataset/Testing/male", 1
    )
    test_female_images, test_female_labels = load_and_preprocess_images(
        "Dataset/Testing/female", 0
    )
    log_time(
        f"Loaded {len(test_male_images)} male and {len(test_female_images)} female test images"
    )

    # Combine test datasets
    X_test = test_male_images + test_female_images
    y_test = test_male_labels + test_female_labels

    log_time("Extracting features from test images...")
    test_features = [extract_all_features(img) for img in X_test]
    X_test_features = np.array(test_features)
    log_time(f"Test features extracted. Shape: {X_test_features.shape}")

    # Scale features using the same scaler
    X_test_scaled = scaler.transform(X_test_features)

    # Make predictions
    log_time("Making predictions...")
    y_pred = model.predict(X_test_scaled)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    log_time("\nTest Set Results:")
    log_time(f"Accuracy: {accuracy:.2f}")
    log_time("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Female", "Male"]))

    evaluation_time = time.time() - start_time
    log_time(f"Total evaluation time: {evaluation_time:.2f} seconds")

    return accuracy


def main():
    total_start_time = time.time()
    log_time("Starting gender classification process...")

    # Check if model and scaler already exist
    if os.path.exists("gender_classifier_model.pkl") and os.path.exists(
        "feature_scaler.pkl"
    ):
        log_time("Loading existing model and scaler...")
        with open("gender_classifier_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("feature_scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
    else:
        log_time("Training new model...")
        model, scaler = train_model()

    # Evaluate on test set
    evaluate_model(model, scaler)

    total_time = time.time() - total_start_time
    log_time(f"Total execution time: {total_time:.2f} seconds")


if __name__ == "__main__":
    main()
