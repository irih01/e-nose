from src.trainer import Trainer

trainer = Trainer(
    dataset_path="data/processed/dataset_master.csv"
)

trainer.load_dataset()

trainer.prepare_data()

results = trainer.run()

print()
print(results)

results.to_csv(
    "model_results.csv",
    index=False
)

print()
print("Results saved.")