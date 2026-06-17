import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.model_factory import ModelFactory
from src.evaluator import Evaluator




class Trainer:

    def __init__(self, dataset_path):
        
        self.dataset_path = dataset_path

        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)

        self.dataset = None

        self.X_train = None
        self.X_test = None

        self.y_train = None
        self.y_test = None

        self.scaler = StandardScaler()


    def load_dataset(self):
        self.dataset = pd.read_csv(
            self.dataset_path
        )

    def prepare_data(self):
        ignore_columns = ["segment_id", "target", "source_file"]

        X = self.dataset.drop(columns=ignore_columns)

        y = self.dataset["target"]

        X_scaled = self.scaler.fit_transform(X)
        sample_count = len(y)
        class_count = len(set(y))

        # adaptive test size for tiny datasets

        if sample_count < 10:
            test_size = 0.5
        else:
            test_size = 0.2

        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test
        ) = train_test_split(
            X_scaled,
            y,
            test_size=test_size,
            random_state=42,
            stratify=y
        )


    def run(self):

        models = ModelFactory.create_models()

        results = []

        for name, model in models.items():
            print()
            print("=" * 50)
            print(name)
            print("=" * 50)

            model.fit(
                self.X_train,
                self.y_train
            )

            model_path = self.models_dir / f"{name}.pkl"

            joblib.dump(model, model_path)

            print("Saved:", model_path)

            y_pred = model.predict(
                self.X_test
            )

            metrics = Evaluator.evaluate(
                self.y_test,
                y_pred
            )

            print(
                "Accuracy:",
                round(metrics["accuracy"], 4)
            )

            print(
                "F1:",
                round(metrics["f1"], 4)
            )

            print(
                "Precision:",
                round(metrics["precision"], 4)
            )

            print(
                "Recall:",
                round(metrics["recall"], 4)
            )

            print(
                "Confusion Matrix:"
            )

            print(metrics["confusion_matrix"])

            results.append({
                "model": name,

                "accuracy":
                    metrics["accuracy"],

                "f1":
                    metrics["f1"],

                "precision":
                    metrics["precision"],

                "recall":
                    metrics["recall"]
            })

        return pd.DataFrame(results)