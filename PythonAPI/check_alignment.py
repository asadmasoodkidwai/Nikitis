from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parent.parent
for name in [
    "flattened_window_datasets/windowed_dataset_0.csv",
    "flattened_window_datasets/windowed_dataset_experience.csv",
]:
    path = root / name
    df = pd.read_csv(path)
    print(f"\n{name}: {len(df):,} rows")
    print("Action label counts:")
    print(df.iloc[:, -10:].sum().to_string())
    print("Any frame/path columns:", [
        c for c in df.columns
        if any(k in c.lower() for k in ("frame", "image", "path", "timestamp"))
    ])
