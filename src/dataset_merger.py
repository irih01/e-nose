from pathlib import Path

import pandas as pd


class DatasetMerger:

    def __init__(self, dataset_dir):

        self.dataset_dir = Path(dataset_dir)

    def merge(self):

        csv_files = sorted(
            self.dataset_dir.glob("*.csv")
        )

        datasets = []

        print()
        print("Found datasets:")
        print()

        for file in csv_files:

            print(file.name)

            df = pd.read_csv(file)

            # optional:
            # keep track of source session

            df["source_file"] = file.stem

            datasets.append(df)

        merged = pd.concat(
            datasets,
            ignore_index=True
        )

        return merged