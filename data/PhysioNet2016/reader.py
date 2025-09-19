import numpy as np
import scipy.io
import os

# features.mat contains the extracted MFCC and DWT features for each heart sound sample.
# labels.mat contains the corresponding labels: 0 for abnormal and 1 for normal heart sounds.
#
# Feature extraction follows the method by Yaseen et al.:
# https://github.com/yaseen21khan/Classification-of-Heart-Sound-Signal-Using-Multiple-Features-


def load_data():
    """
    Load and preprocess the PhysioNet 2016 heart sound dataset.

    This function reads MFCC + DWT features and their corresponding labels 
    from .mat files, handles missing values, normalizes feature values to [0, 1], 
    and splits the dataset into training and test sets.

    Returns
    - train_features (np.ndarray): Training features, shape [num_train, num_features]
    - train_labels (np.ndarray): Training labels, shape [num_train]
    - test_features (np.ndarray): Test features, shape [num_test, num_features]
    - test_labels (np.ndarray): Test labels, shape [num_test]
    """
    
    train_ratio = 0.9  # Ratio of samples to use for training.

    current_dir = os.path.dirname(__file__)
    feature_file = os.path.join(current_dir, "features.mat")
    label_file = os.path.join(current_dir, "labels.mat")
    print(f"Loading features from {feature_file};")
    print(f"Loading labels from {label_file}...")

    # Load features and labels from .mat files
    features = scipy.io.loadmat(feature_file)["features"]
    labels = scipy.io.loadmat(label_file)["data"]

    # Fill any missing values with the mean of their respective feature column
    features = np.where(np.isnan(features), np.nanmean(features, axis=0), features)

    # Normalize features to [0, 1] using 2nd and 98th percentiles
    lower_percentile = np.percentile(features, 2, axis=0)
    upper_percentile = np.percentile(features, 98, axis=0)
    norm_features = (features - lower_percentile) / (upper_percentile - lower_percentile)

    # Clip values outside the normalized range
    norm_features[norm_features >= 1] = 1
    norm_features[norm_features <= 0] = 0

    # Combine features and labels for shuffling and splitting
    sample_data = np.hstack((norm_features, labels))
    np.random.shuffle(sample_data)
    num_trains = int(len(sample_data) * train_ratio)
    train_data = sample_data[:num_trains]
    test_data = sample_data[num_trains:]

    # Separate features and labels for train/test sets
    train_features = train_data[:, :-1]
    train_labels = train_data[:, -1].astype(int)
    test_features = test_data[:, :-1]
    test_labels = test_data[:, -1].astype(int)


    print(f"\t- Number of train samples: {len(train_features)}")
    print(f"\t- Number of test samples: {len(test_features)}\n")

    return train_features, train_labels, test_features, test_labels