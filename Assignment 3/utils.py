import cv2 as cv
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import datetime
from scipy.fft import dctn
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFECV


def log_print(text, path="log.txt"):
    print(text)
    with open(path, "a") as f:
        f.write(f"{datetime.datetime.now()}: {text}\n")


def load_images(path, classes):
    images = []
    labels = []
    for class_name in classes:
        class_path = os.path.join(path, class_name)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv.imread(img_path)
            images.append(img)
            labels.append(class_name)
    return images, labels


def preprocess(img, img_size, kernel_size):
    # Check if image is already grayscale
    if len(img.shape) == 2 or img.shape[2] == 1:
        gray_img = img
    else:
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # resize to 100x100
    img = cv.resize(gray_img, img_size)

    # apply gaussian blur
    img = cv.GaussianBlur(img, kernel_size, 0)

    return img


def preprocess_and_save_images(images, labels, path, img_size, kernel_size):
    preprocessed_images = []
    # Save labels to txt file
    with open(os.path.join(path, "labels.txt"), "w") as f:
        for i, label in enumerate(labels):
            f.write(f"{i}.png {label}\n")

    # Preprocess and save images
    for i, img in enumerate(images):
        preprocessed_img = preprocess(img, img_size, kernel_size)
        cv.imwrite(os.path.join(path, f"{i}.png"), preprocessed_img)
        preprocessed_images.append(preprocessed_img)
    return preprocessed_images


def load_preprocessed_images(path):
    preprocessed_images = []
    labels = []
    # Load labels from labels.txt
    with open(os.path.join(path, "labels.txt"), "r") as f:
        for line in f:
            img_name, label = line.strip().split()
            img_path = os.path.join(path, img_name)
            img = cv.imread(img_path)
            preprocessed_images.append(img)
            labels.append(label)
    return preprocessed_images, labels


def feature_extractor_old(
    img,
    img_size=(100, 100),
    kernel_size=(5, 5),
    canny_low_threshold=100,
    canny_high_threshold=200,
):
    processed_img = preprocess(img, img_size, kernel_size)

    # canny edge detection
    edges = cv.Canny(processed_img, canny_low_threshold, canny_high_threshold)

    return edges


def preprocess_and_extract_features(
    images,
    img_size=(100, 100),
    kernel_size=(5, 5),
    canny_low_threshold=100,
    canny_high_threshold=200,
    preprocessed=False,
):
    features = []
    for img in images:
        processed_img = (
            preprocess(img, img_size, kernel_size) if not preprocessed else img
        )
        feature = feature_extractor_old(
            processed_img,
            img_size,
            kernel_size,
            canny_low_threshold,
            canny_high_threshold,
        )
        features.append(feature.flatten())  # Flatten the feature array
    return np.array(features)


def hog_feature_extractor(img):
    # apply HoG to the image
    hog = cv.HOGDescriptor()
    hog_desc = hog.compute(img)
    return hog_desc


def sift_feature_extractor(img):
    # apply SIFT to the image
    sift = cv.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    return keypoints, descriptors


def dt_cwt_feature_extractor(img):
    # apply DTCWT to the image
    coeffs = dctn(img, type=2)
    return coeffs


def edge_feature_extractor(img, canny_low_threshold=100, canny_high_threshold=200):
    # apply edge detection to the image
    edges = cv.Canny(img, canny_low_threshold, canny_high_threshold)
    return edges


def feature_extractor_new(img):
    # apply the feature extractor to the image
    return (
        edge_feature_extractor(img),
        hog_feature_extractor(img),
        sift_feature_extractor(img),
        dt_cwt_feature_extractor(img),
    )


def preprocess_and_extract_features_new(
    images,
    img_size=(51, 51),
    kernel_size=(5, 5),
    canny_low_threshold=100,
    canny_high_threshold=200,
    preprocessed=False,
    concatenate=True,
    features=["edges", "hog", "sift", "dtcwt"],
):
    all_features = []
    for img in images:
        # Preprocess image
        processed_img = (
            preprocess(img, img_size, kernel_size) if not preprocessed else img
        )
        running_features = {}
        # Extract all features
        if "edges" in features:
            edge_features = edge_feature_extractor(
                processed_img, canny_low_threshold, canny_high_threshold
            ).flatten()
            running_features["edges"] = edge_features
        if "hog" in features:
            hog_features = hog_feature_extractor(processed_img)
            running_features["hog"] = hog_features
        if "sift" in features:
            _, sift_features = sift_feature_extractor(processed_img)
            if sift_features is not None:
                # Take mean of SIFT descriptors to get fixed length
                sift_features = (
                    np.mean(sift_features, axis=0)
                    if sift_features.shape[0] > 0
                    else np.zeros(128)
                )
            else:
                sift_features = np.zeros(128)  # SIFT descriptor length is 128
            running_features["sift"] = sift_features
        if "dtcwt" in features:
            dtcwt_features = dt_cwt_feature_extractor(processed_img).flatten()
            running_features["dtcwt"] = dtcwt_features

        # Concatenate all features if required
        combined_features = (
            np.concatenate(list(running_features.values()))
            if concatenate
            else running_features
        )

        all_features.append(combined_features)
    if concatenate:
        return np.array(all_features)
    else:
        return all_features


def analyze_feature_importance_sklearn(
    features, labels, feature_names, train_images, figures_folder, new_grid_search
):
    # Get feature dimensions for segmenting the feature vector
    edge_dim = edge_feature_extractor(preprocess(train_images[0])).flatten().shape[0]
    hog_dim = hog_feature_extractor(preprocess(train_images[0])).flatten().shape[0]
    sift_dim = 128
    dtcwt_dim = dt_cwt_feature_extractor(preprocess(train_images[0])).flatten().shape[0]
    feature_dims = [edge_dim, hog_dim, sift_dim, dtcwt_dim]

    # Create and train SVM model with best parameters
    best_params = new_grid_search.best_params_
    svm_model = SVC(
        C=best_params["classifier__C"],
        kernel=best_params["classifier__kernel"],
        gamma=best_params["classifier__gamma"],
    )

    # Apply PCA first (since we used it in our pipeline)
    pca = PCA(n_components=best_params["pca__n_components"])
    X_pca = pca.fit_transform(features)

    # Fit the model
    svm_model.fit(X_pca, labels)

    # Method 1: Permutation Importance
    log_print("\nPermutation Importance Analysis:")
    r = permutation_importance(svm_model, X_pca, labels, n_repeats=10, random_state=42)

    # Create feature importance plot
    plt.figure(figsize=(10, 6))
    importances = r.importances_mean
    std = r.importances_std

    feature_indices = range(len(importances))
    plt.bar(feature_indices, importances, yerr=std, align="center")
    plt.title("Feature Importance (Permutation)")
    plt.xlabel("PCA Components")
    plt.ylabel("Mean Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_folder, "permutation_importance.png"))
    plt.close()

    # Print top important features
    sorted_idx = importances.argsort()
    log_print("Top 10 most important PCA components:")
    for idx in sorted_idx[-10:]:
        log_print(f"PCA component {idx}: {importances[idx]:.4f} ± {std[idx]:.4f}")

    # Method 2: Feature Importance by Feature Type
    log_print("\nFeature Type Importance Analysis:")

    # Analyze each feature type separately
    feature_type_scores = {}
    start_idx = 0

    for feature_name, dim in zip(feature_names, feature_dims):
        # Extract features for this type
        end_idx = start_idx + dim
        X_type = features[:, start_idx:end_idx]

        # Apply PCA to these features
        pca_type = PCA(
            n_components=min(best_params["pca__n_components"], X_type.shape[1])
        )
        X_type_pca = pca_type.fit_transform(X_type)

        # Evaluate using cross validation
        cv_scores = cross_val_score(svm_model, X_type_pca, labels, cv=5)
        feature_type_scores[feature_name] = (cv_scores.mean(), cv_scores.std())

        log_print(f"{feature_name}: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        start_idx = end_idx

    # Plot feature type importance
    plt.figure(figsize=(10, 6))
    names = list(feature_type_scores.keys())
    means = [score[0] for score in feature_type_scores.values()]
    stds = [score[1] for score in feature_type_scores.values()]

    plt.bar(names, means, yerr=stds)
    plt.title("Feature Type Importance")
    plt.xlabel("Feature Type")
    plt.ylabel("Cross-validation Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_folder, "feature_type_importance.png"))
    plt.close()

    # Method 3: Recursive Feature Elimination

    log_print("\nRecursive Feature Elimination Analysis:")

    # Use a linear SVM for feature selection
    linear_svm = SVC(kernel="linear", C=best_params["classifier__C"])

    # Perform recursive feature elimination with cross-validation
    rfe = RFECV(estimator=linear_svm, step=1, cv=5, scoring="accuracy", n_jobs=-1)

    # Fit RFE
    rfe.fit(X_pca, labels)

    # Plot number of features vs. cross-validation scores
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(rfe.grid_scores_) + 1), rfe.grid_scores_)
    plt.xlabel("Number of Features Selected")
    plt.ylabel("Cross-validation Score")
    plt.title("Recursive Feature Elimination")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_folder, "rfe_scores.png"))
    plt.close()

    log_print(f"Optimal number of features: {rfe.n_features_}")
    log_print(f"Best cross-validation score: {max(rfe.grid_scores_):.4f}")


def get_top_combinations(
    features, labels, feature_names, feature_dims, best_params, top_k=3
):
    from itertools import combinations

    results = []
    start_indices = [0] + list(np.cumsum(feature_dims[:-1]))

    # Try different combinations of features
    for r in range(2, len(feature_names) + 1):
        for combo in combinations(range(len(feature_names)), r):
            # Combine selected features
            selected_features = []
            combo_names = []
            for idx in combo:
                start = start_indices[idx]
                end = start + feature_dims[idx]
                selected_features.append(features[:, start:end])
                combo_names.append(feature_names[idx])

            X = np.hstack(selected_features)

            # Evaluate combination
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            accuracies = []

            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = labels[train_idx], labels[val_idx]

                pca = PCA(
                    n_components=min(best_params["pca__n_components"], X_train.shape[1])
                )
                X_train_pca = pca.fit_transform(X_train)
                X_val_pca = pca.transform(X_val)
                svm_model = SVC(
                    kernel=best_params["classifier__kernel"],
                    C=best_params["classifier__C"],
                    gamma=best_params["classifier__gamma"],
                )
                svm_model.fit(X_train_pca, y_train)
                accuracy = accuracy_score(y_val, svm_model.predict(X_val_pca))
                accuracies.append(accuracy)

            mean_accuracy = np.mean(accuracies)
            results.append((combo_names, mean_accuracy))

    # Sort and get top k combinations
    results.sort(key=lambda x: x[1], reverse=True)
    for names, acc in results[:top_k]:
        log_print(f"Combination {' + '.join(names)}: {acc:.4f}")
