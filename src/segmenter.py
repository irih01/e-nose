import pandas as pd


class Segmenter:

    def segment_by_label(self, df):
        df = df.sort_values("timestamp_since_poweron")

        segments = []

        current = []
        current_target = None

        for _, row in df.iterrows():
            target = row["target"]

            if current_target is None:
                current_target = target

            if target != current_target:
                if len(current) > 0:
                    segments.append(pd.DataFrame(current))

                current = []
                current_target = target

            current.append(row)

        if len(current) > 0:
            segments.append(pd.DataFrame(current))

        return segments
    