from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import ast

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "video_features_ce" / "frame_features.csv"
OUT = ROOT / "video_features_ce" / "processed_frames.npz"

df = pd.read_csv(CSV)
images = []
valid_rows = []

for i, row in df.iterrows():
    path = ROOT / Path(str(row["frame_path"]))
    try:
        with Image.open(path) as im:
            im = im.convert("L").resize((96, 64))
            images.append(np.asarray(im, dtype=np.uint8))
            valid_rows.append(i)
    except Exception as e:
        print("Skipping:", path, e)

if not images:
    raise RuntimeError("No frames could be processed.")

frames = np.stack(images)
meta = df.iloc[valid_rows].reset_index(drop=True)

np.savez_compressed(
    OUT,
    frames=frames,
    frame_indices=meta["frame_index"].to_numpy(dtype=np.int32),
    brightness=meta["brightness"].to_numpy(dtype=np.float32),
    edge_density=meta["edge_density"].to_numpy(dtype=np.float32),
    candidate_count=meta["candidate_count"].to_numpy(dtype=np.int16),
    frame_paths=meta["frame_path"].astype(str).to_numpy()
)

print("Saved:", OUT)
print("Frames:", len(frames))
print("Tensor shape:", frames.shape)
print("Pixel range:", int(frames.min()), int(frames.max()))
print("Size MB:", round(OUT.stat().st_size / 1024**2, 1))
