
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = Path(".")
DATA = ROOT / "flattened_window_datasets"
MODEL = ROOT / "models/model_shared_attackfocused.keras"
SCALER = joblib.load(str(MODEL) + ".scaler")
model = tf.keras.models.load_model(MODEL, compile=False)

buttons = ["up", "down", "right", "left", "y", "b", "x", "a", "l", "r"]
features = [f"{f}_t-{t}" for t in range(5, -1, -1)
            for f in [
                "timer", "fight_result", "has_round_started",
                "is_round_over", "player1_id", "p1_health", "p1_x",
                "p1_y", "p1_jumping", "p1_crouching", "p1_in_move",
                "p1_move_id", "player2_id", "p2_health", "p2_x",
                "p2_y", "p2_jumping", "p2_crouching", "p2_in_move",
                "p2_move_id", "diff_x", "diff_y", "diff_health"]]

f = DATA / "windowed_dataset_experience.csv"
df = pd.read_csv(f)
X = df[features].apply(pd.to_numeric, errors="coerce")
valid = np.isfinite(X.to_numpy()).all(axis=1)
X = X.loc[valid].to_numpy(np.float32)

pred = model.predict(SCALER.transform(X[:500]), verbose=0)
for i, b in enumerate(buttons):
    print(f"{b:>5}: mean={pred[:, i].mean():.3f}, "
          f">.05={(pred[:, i] > .05).mean():.1%}, "
          f">.20={(pred[:, i] > .20).mean():.1%}, "
          f">.50={(pred[:, i] > .50).mean():.1%}")
