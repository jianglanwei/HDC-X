import numpy as np
from tqdm import tqdm

# This python file implements:
# Hyperdimensional Computing with In-Class Clustering (HD3C) core functions.

def setup(cfg_):
    """
    Initialize global configuration and generate ID-HV and Level-HV dictionaries.

    This function must be called before using any function in this module.
    It sets the global `cfg` variable and precomputes the identity and level
    hypervector dictionaries based on the loaded configuration. These are stored
    in `cfg.id_hv_dict` and `cfg.level_hv_dict`.

    Args
    - cfg_ (SimpleNamespace): Configuration object containing model hyperparameters.
    """
    global cfg
    cfg = cfg_
    cfg.id_hv_dict = id_hv_dict()
    cfg.level_hv_dict = level_hv_dict()

class HVSet:
    """
    Represents a set of hypervectors using their accumulated sum and count.

    This class supports efficient bundling of a large number of hypervectors
    without storing them individually.
    """

    def __init__(self):
        """Initialize an empty HVSet with zero sum and zero count."""
        self.sum = np.zeros(cfg.dim, dtype=int)  # Accumulated sum of hypervectors.
        self.count = 0  # Number of hypervectors added to the set.

    def add(self, hv: np.ndarray):
        """Add a hypervector to the set."""
        self.sum += hv
        self.count += 1

    def sub(self, hv: np.ndarray):
        """Remove a hypervector from the set."""
        self.sum -= hv
        self.count -= 1

    def add_set(self, other: 'HVSet'):
        """Merge another HVSet into this one by adding all of its hypervectors."""
        if other.count == 0:
            return
        self.sum += other.sum
        self.count += other.count

    def bundle(self):
        """
        Compute the bundled hypervector using majority voting over all added 
        hypervectors.

        Bundle is implemented by point-wise addition and binarization with a 
        majority function (majority voting).

        If any dimension sums to zero, the dimension will be randomly assigned 
        -1 or 1 using a noise hypervector.

        Returns
        - np.ndarray: The bundled output as a bipolar hypervector.
        """
        if self.count == 0:
            return hv()
        elif self.count % 2 == 0:
            noise = hv()
            avg = (self.sum + noise) / (self.count + 1)
        else:
            avg = self.sum / self.count
        return np.where(avg > 0, 1, -1)

def hv():
    """
    Generate a random bipolar hypervector.
    
    Returns
    - np.ndarray: The random bipolar hypervector.
    """
    return np.random.choice([-1, 1], cfg.dim)

def id_hv_dict():
    """
    Generate the identity hypervector dictionary (ID-HV Dictionary).

    Each identity hypervector is randomly generated and uniquely assigned to
    a feature index. Each ID-HV labels one feature in the HD3C framework.

    Returns
    - np.ndarray: A 2D array of shape (num_features, dim). Each ID-HV is a row 
    in the 2D array.
    """
    return np.array([hv() for _ in range(cfg.num_features)])

def level_hv_dict():
    """
    Generate the level hypervector dictionary (Level-HV Dictionary).

    Each Level-HV represents a quantized level of a continuous feature.
    The first Level-HV is randomly initialized. Each subsequent Level-HV
    is created by flipping a fixed number of dimensions from the previous one.
    Each dimension will be flipped no more than once across all levels.

    Returns
    - np.ndarray: A 2D array of shape (num_levels, dim). Each Level-HV is a row
    in the 2D array.
    """
    level_hv_dict_ = np.zeros((cfg.num_levels, cfg.dim), dtype=int)
    level_hv_dict_[0] = hv()  # Initialize level 0 with a random HV
    rand_order = np.random.permutation(cfg.dim)  # Random flip order
    flip_dims_per_level = cfg.dim // (cfg.num_levels - 1)
    for level in range(1, cfg.num_levels):
        # Copy previous Level-HV and flip a subset of dimensions
        level_hv_dict_[level] = level_hv_dict_[level - 1].copy()
        flip_dims = rand_order[
            (level - 1) * flip_dims_per_level : level * flip_dims_per_level
        ]
        level_hv_dict_[level, flip_dims] *= -1
    return level_hv_dict_

def get_level(value: float):
    """
    Compute the quantized level index for a given normalized feature value.

    The normalized feature value is assumed to be within cfg.value_range and is 
    mapped to one of cfg.num_levels discrete levels.

    If the value is exactly equal to the upper boundary, it is assigned to
    the last level (index cfg.num_levels - 1).

    Args
    - value (float): A normalized feature value within cfg.value_range.

    Returns
    - int: The level index corresponding to the quantized bin.
    """
    level_length = (cfg.value_range[1] - cfg.value_range[0]) / cfg.num_levels
    level_idx = int((value - cfg.value_range[0]) // level_length)
    if level_idx == cfg.num_levels:
        level_idx -= 1
    return level_idx

def encode_sample_hvs(features: np.ndarray):
    """
    Encode a batch of samples into their corresponding Sample-HVs.

    For each sample, each feature is represented by binding its identity
    hypervector (ID-HV) with a level hypervector (Level-HV) determined by
    quantizing the feature value. All bound feature-value pairs are then 
    bundled into one Sample-HV using majority function (majority voting).

    Args
    - features (np.ndarray): A 2D array of shape [num_samples, num_features], 
    containing normalized feature values.

    Returns
    - np.ndarray: A 2D array of shape [num_samples, dim] where each row is a 
    Sample-HV representing the corresponding input sample.
    """
    sample_hvs = np.zeros((len(features), cfg.dim), dtype=int)
    for sample_idx, sample_features in enumerate(tqdm(features)):
        bound_feature_set = HVSet()
        for feature_idx, feature_value in enumerate(sample_features):
            id_hv = cfg.id_hv_dict[feature_idx]
            level_hv = cfg.level_hv_dict[get_level(feature_value)]
            bound_feature_set.add(id_hv * level_hv)
        sample_hvs[sample_idx] = bound_feature_set.bundle()
    return sample_hvs

def classify(sample_hv: np.ndarray, cluster_hvs: np.ndarray):
    """
    Classify a Sample-HV by finding the most similar Cluster-HV.

    Similarity is computed using Hamming distance, the fraction of differing
    bits between the Sample-HV and each Cluster-HV. The function returns the 
    index of the closest Cluster-HV.

    Args
    - sample_hv (np.ndarray): A hypervector representing the input sample.
    - cluster_hvs (np.ndarray): A 2D array of shape 
    [num_classes * num_clusters_per_class, dim] containing candidate Cluster-HVs.

    Returns
    - int: The index of the Cluster-HV with the smallest Hamming distance.
    """
    hamming_distances = np.sum(sample_hv != cluster_hvs, axis=1) / cfg.dim
    return int(np.argmin(hamming_distances))

def hyperspace_clustering(sample_hvs: np.ndarray):
    """
    Perform unsupervised clustering of Sample-HVs into Cluster-HVs in hyperspace.

    This function follows a K-means-inspired procedure. It begins with random 
    assignments of samples to clusters, then iteratively refines the assignments 
    by computing new Cluster-HVs (bundled HVs per cluster) and reassigning each 
    sample to the nearest cluster using Hamming distance.

    Args
    - sample_hvs (np.ndarray): A 2D array of shape [num_samples, dim], 
    containing Sample-HVs to be clustered.

    Returns
    - list[HVSet]: A list of HVSet objects, each representing one cluster.
    - np.ndarray: An array of shape [num_samples], mapping each sample to its
    assigned cluster's index.
    """
    # Initialize clusters by randomly assigning each sample to a cluster.
    num_samples = len(sample_hvs)
    clusters = [HVSet() for _ in range(cfg.num_clusters_per_class)]
    init_assigns = np.random.randint(0, cfg.num_clusters_per_class, num_samples)
    for sample_hv, assigned_cluster in zip(sample_hvs, init_assigns):
        clusters[assigned_cluster].add(sample_hv)
    cluster_hvs = np.array([cluster.bundle() for cluster in clusters])

    # Prepare to store final cluster assignments for each sample.
    sample_to_cluster = np.zeros(num_samples, dtype=int)

    # Iteratively reassign samples to nearest clusters and update Cluster-HVs.
    for iter in range(cfg.num_clustering_iters):
        clusters = [HVSet() for _ in range(cfg.num_clusters_per_class)]
        for sample_idx, sample_hv in enumerate(sample_hvs):
            pred_cluster = classify(sample_hv, cluster_hvs)
            if iter == cfg.num_clustering_iters - 1:
                sample_to_cluster[sample_idx] = pred_cluster
            clusters[pred_cluster].add(sample_hv)
        cluster_hvs = np.array([cluster.bundle() for cluster in clusters])

    return clusters, sample_to_cluster

def generate_clusters(sample_hvs: np.ndarray, labels: np.ndarray):
    """
    Cluster samples within each class using hyperspace clustering.

    This function performs class-wise unsupervised clustering using the 
    `hyperspace_clustering` function, and translates local cluster indices into 
    global cluster indices.

    Args
    - sample_hvs (np.ndarray): A 2D array of shape [num_samples, dim], containing
    Sample-HVs to be clustered.
    - labels (np.ndarray): A 1D array of shape [num_samples], containing the class
    label for each sample.

    Returns
    - list[HVSet]: List of HVSet objects, each representing one cluster.
    - np.ndarray: A 1D array of shape [num_samples], mapping each sample to its 
    assigned cluster's index.
    """
    # Group samples by class
    sample_by_class = [
        {"sample_hvs": [], "sample_idxs": []} for _ in range(cfg.num_classes)
    ]
    for sample_idx, (sample_hv, label) in enumerate(zip(sample_hvs, labels)):
        sample_by_class[label]["sample_hvs"].append(sample_hv)
        sample_by_class[label]["sample_idxs"].append(sample_idx)
    
    all_clusters = []
    sample_to_cluster_global = np.zeros(len(sample_hvs), dtype=int)

    # Perform clustering within each class
    for class_idx, class_entry in enumerate(tqdm(sample_by_class)):
        class_clusters, sample_to_cluster_local = (
            hyperspace_clustering(class_entry["sample_hvs"])
        )
        all_clusters.extend(class_clusters)

        # Translate local cluster indices into global cluster indices
        sample_to_cluster_global[class_entry["sample_idxs"]] = (
            class_idx * cfg.num_clusters_per_class + sample_to_cluster_local
        )
    return all_clusters, sample_to_cluster_global


def retrain_clusters(
        sample_hvs: np.ndarray, sample_to_cluster: np.ndarray, clusters: list[HVSet]
    ):
    """
    Refine Cluster-HVs using misclassified samples and compute class-level 
    training accuracy.

    Each sample is reclassified using the current Cluster-HVs. If the predicted 
    cluster differs from the true one, the sample is subtracted from the incorrect 
    cluster and added to its correct cluster. Class-level accuracy is computed by 
    comparing predicted and true class labels.

    Args
    - sample_hvs (np.ndarray): Array of shape [num_samples, dim] containing 
    Sample-HVs.
    - sample_to_cluster (np.ndarray): Array of shape [num_samples], mapping each 
    sample to its true cluster index.
    - clusters (list[HVSet]): List of HVSet objects, each representing a cluster 
    from training.

    Returns
    - float: Class-level training accuracy, as a value between 0 and 1.
    """
    num_errors = 0

    # Bundle each cluster's hypervectors into Cluster-HVs
    cluster_hvs = np.array([cluster.bundle() for cluster in clusters])

    for sample_hv, true_cluster in zip(sample_hvs, sample_to_cluster):
        # Predict cluster
        pred_cluster = classify(sample_hv, cluster_hvs)

        if pred_cluster != true_cluster:
            # Refine Cluster-HVs by the reference of misclassified samples.
            clusters[true_cluster].add(sample_hv)
            clusters[pred_cluster].sub(sample_hv)

            # Check if prediction is wrong at the class level
            pred_class = pred_cluster // cfg.num_clusters_per_class
            true_class = true_cluster // cfg.num_clusters_per_class
            if pred_class != true_class:
                num_errors += 1

    return 1 - num_errors / len(sample_hvs)

def accuracy(sample_hvs: np.ndarray, labels: np.ndarray, clusters: list[HVSet]):
    """
    Evaluate classification accuracy using Cluster-HVs.

    Each sample is classified to the nearest cluster using the `classify` 
    function, which compares the sample's hypervector (Sample-HV) to each 
    bundled Cluster-HV using Hamming distance. The predicted label is the class 
    where the cluster belongs.

    Args
    - sample_hvs (np.ndarray): Array of shape [num_samples, dim] containing 
    Sample-HVs to be evaluated.
    - labels (np.ndarray): Array of shape [num_samples] containing the true 
    class labels.
    - clusters (list[HVSet]): List of HVSet objects, each representing a cluster 
    from training.

    Returns
    - float: Classification accuracy as a value between 0 and 1.
    """
    num_errors = 0

    # Bundle each cluster's hypervectors into Cluster-HVs
    cluster_hv = np.array([cluster.bundle() for cluster in clusters])

    # Classify each sample and compare against the true label
    for sample_hv, true_class in zip(sample_hvs, labels):
        pred_class = classify(sample_hv, cluster_hv) // cfg.num_clusters_per_class
        if pred_class != true_class:
            num_errors += 1
    
    return 1 - num_errors / len(sample_hvs)