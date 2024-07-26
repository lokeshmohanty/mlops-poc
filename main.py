import typer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

app = typer.Typer()


@app.command()
def generate_synthetic_data(n_samples: int = 500,
                            noise: float = 0.30,
                            random_state: int = 42):
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    dataset = Dataset.create(
        dataset_name="Synthetic Data",
        dataset_project="Moon Classification"
    )
    np.savetxt("X.csv", X, delimiter=",")
    np.savetxt("y.csv", y, delimiter=",")
    dataset.add_files(path="X.csv")
    dataset.add_files(path="y.csv")
    dataset.upload()
    dataset.finalize()


@app.command()
def run(model: str = "adaboost", dataset: str = "synthetic"):
    if dataset == "synthetic":
        from datasets import SyntheticDataset
        data = SyntheticDataset().get_data()
    else:
        raise ValueError("Invalid dataset")
    if model == "adaboost":
        from adaboost import Adaboost
        model = Adaboost("Moon Classification", "Adaboost Classifier", data)
    else:
        raise ValueError("Invalid model")


if __name__ == "__main__":
    app()
