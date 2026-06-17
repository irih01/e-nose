from dataset_merger import DatasetMerger

merger = DatasetMerger(
    dataset_dir="../data/processed"
)

dataset = merger.merge()

print()
print("Merged shape:")
print(dataset.shape)

print()
print(dataset["target"].value_counts())

dataset.to_csv(
    "../data/processed/dataset_master.csv",
    index=False
)

print()
print("Saved master dataset.")