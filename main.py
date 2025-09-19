import hd3c_core as hd3c
import argparse
from types import SimpleNamespace
import os
import yaml
import importlib.util


parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="PhysioNet2016")
args = parser.parse_args()


# Load YAML config file for the selected task
cfg_path = os.path.join("config", f"{args.task}.yaml")
assert os.path.exists(cfg_path), f"config file not found: {cfg_path}"
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)
cfg = SimpleNamespace(**cfg)


# Initialize HD3C module with config
hd3c.setup(cfg)


# Load dataset using reader module
reader_path = os.path.join("data", args.task, "reader.py")
assert os.path.exists(reader_path), f"data reader not found: {reader_path}"
spec = importlib.util.spec_from_file_location("reader", reader_path)
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)

train_features, train_labels, test_features, test_labels = reader.load_data()
assert train_features.shape[1] == cfg.num_features, (
    f"expected {cfg.num_features} features, got {train_features.shape[1]}"
)
assert train_labels.max() == cfg.num_classes - 1, (
    f"expected {cfg.num_classes} classes, got max label {train_labels.max()}"
)


# Encode training data into Sample-HVs
print("Encoding training Sample-HVs...")
train_sample_hvs = hd3c.encode_sample_hvs(train_features)
print("\t- Training Sample-HVs encoded.\n")


# Generate clusters from training Sample-HVs and labels
print("Clustering training samples...")
clusters, train_cluster_labels = hd3c.generate_clusters(train_sample_hvs, train_labels)
print("\t- Clustering complete.\n")
print("Initial training complete.\n")


# Encode test data into Sample-HVs
print("Encoding test Sample-HVs...")
test_sample_hvs = hd3c.encode_sample_hvs(test_features)
print("\tTest Sample-HVs encoded.\n")


# Evaluate classification accuracy on train and test sets
print("Evaluating classification accuracy...")
train_accuracy = hd3c.accuracy(train_sample_hvs, train_labels, clusters)
test_accuracy = hd3c.accuracy(test_sample_hvs, test_labels, clusters)
print(f"\tAccuracy: {train_accuracy * 100:.2f}% (train), {test_accuracy * 100:.2f}% (test)\n")


# Retrain Cluster-HVs with misclassified samples for specified epochs
for epoch in range(cfg.num_retrain_epochs):
    print(f"Epoch {epoch}: Retraining cluster representations...")
    train_accuracy = hd3c.retrain_clusters(train_sample_hvs, train_cluster_labels, clusters)
    test_accuracy = hd3c.accuracy(test_sample_hvs, test_labels, clusters)
    print(f"\tAccuracy: {train_accuracy * 100:.2f}% (train), {test_accuracy * 100:.2f}% (test)\n")

print(f"Retraining complete.")