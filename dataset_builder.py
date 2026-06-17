from src.data_loader import BMEDataLoader
from src.label_mapper import LabelMapper
from src.segmenter import Segmenter
from src.feauture_extractor import FeatureExtractor




# ==========================================
# CONFIG
# ==========================================

RAW_FILE = "data/cofee/cafea_dulap.bmerawdata"
LABEL_FILE = "data/cofee/cafea_dulap.bmelabelinfo"

OUTPUT_CSV = "dataset_cafea_dulap.csv"


# ===================================
# LOAD
# ===================================

loader = BMEDataLoader(
    raw_path=RAW_FILE,
    label_path=LABEL_FILE
)

df = loader.load()

# ===================================
# LABELS
# ===================================

mapper = LabelMapper()

df = mapper.apply(df)

# ===================================
# SEGMENTATION
# ===================================

segmenter = Segmenter()

segments = segmenter.segment_by_label(df)

print("Segments:", len(segments))

# ===================================
# FEATURES
# ===================================

extractor = FeatureExtractor()

dataset = extractor.extract(segments)

print(dataset.shape)

dataset.to_csv(
    f"data/processed/{OUTPUT_CSV}",
    index=False
)

print("Dataset saved.")