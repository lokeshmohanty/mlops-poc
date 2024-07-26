import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from pathlib import Path
from clearml import Dataset

class FmnistDataset:
    name = "Fashion MNIST"
    project = "Image Classification"

    def __init__(self):
        self.path = Path(Dataset.get(
            dataset_name=self.name,
            dataset_project=self.project,
            only_completed=False
        ).get_local_copy())

    def get_data(self):
        train = pd.read_csv(self.path / "fashion-mnist_train.csv")
        test = pd.read_csv(self.path / "fashion-mnist_test.csv")
        return train, test

class SyntheticDataset:
    name = "Synthetic Data"
    project = "Moon Classification"

    def __init__(self):
        self.path = Path(Dataset.get(
            dataset_name=self.name,
            dataset_project=self.project,
            only_completed=False
        ).get_local_copy())

        self.X = np.loadtxt(self.path / "X.csv", delimiter=",")
        self.y = np.loadtxt(self.path / "y.csv", delimiter=",")

    def get_data(self, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X,
            self.y,
            test_size=test_size,
            random_state=random_state
        )
        return X_train, X_test, y_train, y_test

