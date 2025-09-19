import numpy as np
import scipy.io
import os
import pandas as pd
import pdb

def load_data():
    """
    Load and preprocess the WBCD dataset.

    This function reads WBCD features and their corresponding labels 
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
    filename = os.path.join(current_dir, "data.csv")


    # Load features and labels from .mat files
    df = pd.read_csv(filename)


    # Drop the ID column
    df = df.drop(columns=["id", "Unnamed: 32"])

    # Target: diagnosis column
    labels = df['diagnosis'].map({'M': 1, 'B': 0}).to_numpy()  # Encode as 1 = Malignant, 0 = Benign

    # Features
    features = df.drop(columns=['diagnosis']).to_numpy()


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
    sample_data = np.hstack((norm_features, labels.reshape(-1, 1)))
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