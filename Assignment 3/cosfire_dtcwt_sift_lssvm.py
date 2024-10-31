import os
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from dtcwt.numpy import Transform2d
from lssvm import LSSVM
import glob


def load_and_preprocess_images(folder_path, label):
    """Load and preprocess images from a folder."""
    images = []
    labels = []

    for img_path in glob.glob(os.path.join(folder_path, "*")):
        # Read image
        img = cv2.imread(img_path)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Resize to standard size (e.g., 80x80)
        resized = cv2.resize(gray, (80, 80))

        # Histogram equalization
        equalized = cv2.equalizeHist(resized)

        images.append(equalized)
        labels.append(label)

    return images, labels


def extract_dtcwt_features(image):
    """Extract Dual-Tree Complex Wavelet Transform features."""
    transform = Transform2d()
    coeffs = transform.forward(image, nlevels=4)

    # Extract magnitude of complex coefficients
    features = []
    for level in coeffs.highpasses:
        features.extend(np.abs(level).flatten())

    return np.array(features)


def extract_dense_sift_features(image):
    """Extract Dense SIFT features."""
    sift = cv2.SIFT_create()

    # Create dense grid of keypoints
    step_size = 8
    kp = [
        cv2.KeyPoint(x, y, step_size)
        for y in range(0, image.shape[0], step_size)
        for x in range(0, image.shape[1], step_size)
    ]

    # Compute SIFT descriptors
    _, desc = sift.compute(image, kp)

    return (
        desc.flatten()
        if desc is not None
        else np.zeros(
            128 * (image.shape[0] // step_size) * (image.shape[1] // step_size)
        )
    )


def create_log_gabor_filter(size, wavelength, orientation, sigma):
    """Create a Log-Gabor filter in the frequency domain."""
    rows, cols = size
    center_x = cols // 2
    center_y = rows // 2

    # Create meshgrid of frequencies
    y, x = np.meshgrid(np.arange(rows) - center_y, np.arange(cols) - center_x)

    # Convert to polar coordinates
    radius = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    # Avoid division by zero
    radius[center_y, center_x] = 1

    # Log-Gabor radial component
    log_rad = np.log(radius / wavelength)
    radial = np.exp(-(log_rad**2) / (2 * np.log(sigma) ** 2))

    # Angular component
    angular = np.exp((-theta - orientation) ** 2 / (2 * (np.pi / 8) ** 2))

    # Combine components
    gabor_filter = radial * angular

    # Set DC to 0
    gabor_filter[center_y, center_x] = 0

    return gabor_filter


def extract_log_gabor_features(image):
    """Extract Log-Gabor features from the image."""
    # Parameters for Log-Gabor filters
    wavelengths = [3, 6, 12]  # Different scales
    orientations = np.arange(0, np.pi, np.pi / 6)  # 6 orientations
    sigma = 0.65  # Filter bandwidth

    features = []
    img_fft = np.fft.fft2(image)

    for wavelength in wavelengths:
        for orientation in orientations:
            # Create filter
            lg_filter = create_log_gabor_filter(
                image.shape, wavelength, orientation, sigma
            )

            # Apply filter in frequency domain
            filtered = np.fft.ifft2(img_fft * lg_filter)
            magnitude = np.abs(filtered)

            # Extract statistical features from magnitude
            features.extend([np.mean(magnitude), np.std(magnitude), np.max(magnitude)])

    return np.array(features)


def extract_all_features(image):
    """Combine all features."""
    log_gabor_feat = extract_log_gabor_features(image)
    dtcwt_feat = extract_dtcwt_features(image)
    sift_feat = extract_dense_sift_features(image)

    return np.concatenate([log_gabor_feat, dtcwt_feat, sift_feat])
