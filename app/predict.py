from pathlib import Path

import joblib
from sklearn.datasets import load_iris

MODEL_PATH = Path(__file__).resolve().parent.parent / "model.joblib"
model = joblib.load(MODEL_PATH)

SPECIES_NAMES = load_iris().target_names

def predict(features: int):
    if len(features) != 4:
        raise ValueError("Expected exactly 4 features.")

    sample = [list(features)]
    label = model.predict(sample)[0]
    return str(SPECIES_NAMES[label])

if __name__ == "__main__":
    predict([5.1, 3.5, 1.4, 0.2])
