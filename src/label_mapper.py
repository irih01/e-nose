import pandas as pd

class LabelMapper:

    def __init__(self):
        self.coffee_keywords = [
            "cafea",
            "coffee"
        ]


    def to_binary(self, label_name):
        if pd.isna(label_name):
            return None
        
        label_name = str(label_name).lower()

        for kw in self.coffee_keywords:
            if kw in label_name:
                return 1
            
        return 0
    

    def apply(self, df):
        df["target"] = df["label_name"].apply(self.to_binary)

        return df