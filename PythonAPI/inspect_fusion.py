from pathlib import Path
import joblib
import tensorflow as tf
import pandas as pd
import numpy as np

root = Path(__file__).resolve().parent.parent

for name in ["model_shared_moredata.keras", "model_shared_experience.keras"]:
    path = root / "models" / name
    if not path.exists():
        continue

    model = tf.keras.models.load_model(path, compile=False)
    scaler = joblib.load(str(path) + ".scaler")

    print(f"\n{name}")
    print("Model input:", model.input_shape)
    print("Model output:", model.output_shape)
    print("Scaler features:", len(scaler.mean_))

emb_path = root / "video_features_ce" / "visual_embeddings.npy"
emb = np.load(emb_path, mmap_mode="r")
print("\nVisual embeddings:", emb.shape)

csv_path = root / "windowed_dataset_experience.csv"
if csv_path.exists():
    df = pd.read_csv(csv_path, nrows=3)
    print("Experience dataset columns:", len(df.columns))
    print("Columns:", df.columns.tolist())
else:
    print("Experience CSV not found at:", csv_path)
