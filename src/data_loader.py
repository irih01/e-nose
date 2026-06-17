import json
import pandas as pd

class BMEDataLoader:
    
    def __init__(self, raw_path, label_path):
        self.raw_path = raw_path
        self.label_path = label_path

        self.label_map = {}


    def load_label_info(self):
        with open(self.label_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data["labelInformation"]:
            tag = item["labelTag"]
            name = item["labelName"]

            self.label_map[tag] = name

    
    def load_raw_dataframe(self):
        with open(self.raw_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        columns = [
            c["key"] for c in raw["rawDataBody"]["dataColumns"]
        ]

        rows = raw["rawDataBody"]["dataBlock"]

        df = pd.DataFrame(rows, columns=columns)
        
        df["label_name"] = df["label_tag"].map(self.label_map)

        return df
    

    def load(self):
        self.load_label_info()

        return self.load_raw_dataframe()
        