from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parent.parent
folder = root / "flattened_window_datasets"

for path in sorted(folder.glob("windowed_dataset*.csv")):
    df = pd.read_csv(path, nrows=2)
    print(f"\n{path.name}")
    print("Columns:", len(df.columns))
    print("First columns:", df.columns.tolist()[:12])
    print("Last columns:", df.columns.tolist()[-12:])
