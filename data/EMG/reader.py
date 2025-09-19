import numpy as np
import scipy.io
import os
import pandas as pd
import pdb


def load_data():
    """
    Load and preprocess the sEMG muscle fatigue dataset.

    Returns
    - train_features (np.ndarray): Training features, shape [num_train, num_features]
    - train_labels (np.ndarray): Training labels, shape [num_train]
    - test_features (np.ndarray): Test features, shape [num_test, num_features]
    - test_labels (np.ndarray): Test labels, shape [num_test]
    """

    # Load CSV into pandas DataFrame
    train_data = pd.read_csv(f'data/EMG/train.csv').to_numpy()
    test_data = pd.read_csv(f'data/EMG/test.csv').to_numpy()

    num_trains = len(train_data)
    
    features = np.concatenate([train_data[..., :-1], test_data[..., :-1]], axis=0)
    labels = np.concatenate([train_data[..., -1:], test_data[..., -1:]], axis=0)

    # Fill any missing values with the mean of their respective feature column
    features = np.where(np.isnan(features), np.nanmean(features, axis=0), features)

    # Normalize features to [0, 1] using 2nd and 98th percentiles
    lower_percentile = np.percentile(features, 2, axis=0)
    upper_percentile = np.percentile(features, 98, axis=0)
    norm_features = (features - lower_percentile) / (upper_percentile - lower_percentile)

    # Clip values outside the normalized range
    norm_features[norm_features >= 1] = 1
    norm_features[norm_features <= 0] = 0

    # Separate features and labels for train/test sets
    train_features = norm_features[:num_trains]
    train_labels = labels[:num_trains].astype(int).squeeze(1)
    test_features = norm_features[num_trains:]
    test_labels = labels[num_trains:].astype(int).squeeze(1)


    print(f"\t- Number of train samples: {len(train_features)}")
    print(f"\t- Number of test samples: {len(test_features)}\n")

    return train_features, train_labels, test_features, test_labels