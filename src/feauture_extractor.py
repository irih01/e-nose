import numpy as np
import pandas as pd


class FeatureExtractor:

    def __init__(self):
        self.stats = [
            "mean",
            "std",
            "min",
            "max",
            "median"
        ]

    
    def extract(self, segments):
        rows = []

        for seg_id, seg in enumerate(segments):
            feat = {}

            feat["segment_id"] = seg_id
            feat["target"] = int(seg["target"].iloc[0])

            feat["temp_mean"] = seg["temperature"].mean()
            feat["humidity_mean"] = seg["relative_humidity"].mean()
            feat["pressure_mean"] = seg["pressure"].mean()

            grouped = seg.groupby(
                [
                    "sensor_index",
                    "heater_profile_step_index"
                ]
            )

            for (sensor, heater), g in grouped:

                vals = np.log(g["resistance_gassensor"].values + 1)

                prefix = f"s{sensor}_h{heater}"

                feat[f"{prefix}_mean"] = np.mean(vals)
                feat[f"{prefix}_std"] = np.std(vals)
                feat[f"{prefix}_min"] = np.min(vals)
                feat[f"{prefix}_max"] = np.max(vals)
                feat[f"{prefix}_median"] = np.median(vals)

            rows.append(feat)

        dataset = pd.DataFrame(rows)

        return dataset.fillna(0)