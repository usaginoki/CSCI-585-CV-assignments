# Assignment 3
# full code and results available at
# https://github.com/usaginoki/CSCI-585-CV-assignments

#! this file does not contain any code
#! instead it explains what scripts do what and answers the questions in the assignment

# * Part 1: gender recognition with old features
# This part is implemented in train_test_old.py
# The results are in the results/logs/output_old.log file

# Here I used the edges as features (as in assignment 1)
# The pipeline consists of PCA and KNN
# After hyperparameter tuning I got the following results:
# ! accuracy: 0.80

# * Part 2.1: gender recognition with new features
# This part is implemented in train_test_new.py
# The results are in the results/logs/output_new.log file

# Added features and classificator change were based on the paper:
# "Comparison of Recent Machine Learning Techniques for Gender Recognition from Facial Images"
# by Lemley J. and Andonie R. (2016)
# Added features are: HOG, SIFT, DTCWT
#   the new features can better capture the general shape of the face, texture and landmarks
# New classifier is SVM
#   (actually SGDClassifier, which is also linear but has a different learning process)
# The pipeline consists of PCA and SVM
# After hyperparameter tuning I got the following results:
# ! accuracy: 0.83

# * Part 2.2: another method from another paper:
# This part is implemented in cosfire_dtcwt_sift_lssvm.py and train_and_evaluate.py
# "Recognizing gender from images with facial makeup" by Micheal A. and Palanisamy G. (2024)

# Added features are: Log-Gabor, DTCWT, SIFT
# New classifier is SGDClassifier (paper used LS-SVM, but I used SGDClassifier for faster training)
# no hyperparameter tuning was done here
# The results are in the results/logs/output.log file
# ! accuracy: 0.92


# * Part 3: feature importance evaluation
# This part is implemented in feature_importance_evaluation.py
# The results are in the results/logs/output_feature_importance.log file
# The graphical representation of the feature importance is in the results/figures/ folder

# I performed feature importance with permutation importance algorithm with modifications:
# Instead of using in-built scikit function I implemented my own version in the following way:
#   - I trained a model with all features
#   - I calculated the accuracy
#   - For each feature I calculated the accuracy when the feature vectors (of this feature) were shuffled
#   - (i.e. when analyzing importance of the DTCWT, the vectors for DTCWT between same images were shuffled)
#   - The feature importance is the difference between the accuracy when the feature vectors were shuffled and the original accuracy
#   - The feature importance is then normalized by the sum of absolute values of feature importances
# This was done in order to evaluate the relative importance of each feature type, instead of each single feature
# The mean results are the following:
# ! Log-Gabor: 0.001
# ! DTCWT: 0.281
# ! SIFT: 0.157
# As the boxplot suggests, there is little to no deviation from the mean importances.


# ! notes for the future:
#   - feature extraction takes a lot of time, so it is better to save the extracted features
#   and load them instead of extracting them every time
#   - better manage the folders as they file system here is a mess


# ! The scripts:

# ? config.py:
#    - contains the experiment configurations

# ? utils.py:
#    - contains utility functions (basically all functions used in the other scripts)

# ? preprocess_main.py:
#    - preprocesses the train and test images
#    - saves them in the PREPROCESSED_TRAIN_PATH and PREPROCESSED_TEST_PATH folders

# ? train_test_old.py:
#    - loads the preprocessed train and test images
#    - extracts features as in the first assignment (here they were edges)
#    - performs a grid search with k-nearest neighbors
#    - evaluates the classifier on the test set

# ? train_test_new.py:
#    - loads the preprocessed train and test images
#    - extracts features as in the second assignment (here they were edges, HOG, SIFT, DTCWT)
#    - performs a grid search with a support vector machine
#    - evaluates the classifier on the test set

# ? feature_importance_evaluation.py:
#    - loads and preprocesses test images
#    - extracts features: Log-Gabor, DTCWT, SIFT
#    - evaluates the feature importance with permutation importance (5 permuatations per feature type)
#    - permutation is done across feature types, not within them
#    - draws and saves the barchart and boxplot of the feature importances
#    - saves the importances as JSON and CSV files

# ? cosfire_dtcwt_sift_lssvm.py:
#    - contains the implementation of the method from the paper
#      "Recognizing gender from images with facial makeup"
#      by Micheal A. and Palanisamy G. (2024)

# ? train_and_evaluate.py:
#    - loads and preprocesses images (new preprocessing)
#    - extracts features: Log-Gabor, DTCWT, SIFT
#    - trains a classifier: SGDClassifier
#    - evaluates the classifier on the test set
