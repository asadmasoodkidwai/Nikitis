from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models, callbacks

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "flattened_window_datasets"
OUT = ROOT / "models"
OUT.mkdir(exist_ok=True)

WINDOW = 6
STATE = [
    "timer", "fight_result", "has_round_started", "is_round_over",
    "player1_id", "p1_health", "p1_x", "p1_y", "p1_jumping",
    "p1_crouching", "p1_in_move", "p1_move_id", "player2_id",
    "p2_health", "p2_x", "p2_y", "p2_jumping", "p2_crouching",
    "p2_in_move", "p2_move_id", "diff_x", "diff_y", "diff_health"
]
FEATURES = [
    f"{f}_t-{t}"
    for t in range(WINDOW - 1, -1, -1)
    for f in STATE
]
BUTTONS = ["up", "down", "right", "left", "y", "b", "x", "a", "l", "r"]
LABELS = [f"player1_buttons_{b}" for b in BUTTONS]

files = [DATA / "windowed_dataset_experience.csv"]
if not files:
    raise SystemExit(f"No windowed datasets found in {DATA}")

frames = []
for file in files:
    df = pd.read_csv(file)
    missing = set(FEATURES + LABELS) - set(df.columns)
    if missing:
        print(f"SKIP {file.name}: missing {sorted(missing)[:5]}")
        continue
    df = df[FEATURES + LABELS].copy()
    df["_source"] = file.stem
    frames.append(df)
    print(f"Loaded {file.name}: {len(df):,} rows")

if not frames:
    raise SystemExit("No compatible datasets found.")

df = pd.concat(frames, ignore_index=True)
print(f"\nCombined rows: {len(df):,}")
print(f"Datasets combined: {len(frames)}")

# Encode categorical/boolean fields consistently.
fight_map = {"NOT_OVER": 0, "P1": 1, "P2": 2}
bool_cols = [
    "has_round_started", "is_round_over", "p1_jumping",
    "p1_crouching", "p1_in_move", "p2_jumping",
    "p2_crouching", "p2_in_move"
]
for t in range(WINDOW):
    c = f"fight_result_t-{t}"
    df[c] = df[c].replace(fight_map)
    for b in bool_cols:
        c = f"{b}_t-{t}"
        df[c] = df[c].replace({
            True: 1, False: 0, "True": 1, "False": 0,
            "true": 1, "false": 0
        })

X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
y = df[LABELS].apply(pd.to_numeric, errors="coerce").fillna(0)
y = (y > 0).astype(np.float32)


# Remove invalid feature rows.
valid = np.isfinite(X.to_numpy()).all(axis=1)
X, y = X.loc[valid], y.loc[valid]
sources = df.loc[X.index, "_source"].reset_index(drop=True)

# Preserve source and row order for chronological validation.
combined = pd.concat(
    [
        X.reset_index(drop=True),
        y.reset_index(drop=True),
        sources.rename("_source"),
    ],
    axis=1,
)

sources = combined["_source"].to_numpy()
X = combined[FEATURES].to_numpy(dtype=np.float32)
y = combined[LABELS].to_numpy(dtype=np.float32)

active = (y.sum(axis=1) > 0).astype(int)

if len(X) < 100:
    raise SystemExit(f"Too few usable rows: {len(X)}")

# Validate on the later portion of each source.
# Leave a 5-row gap to reduce overlapping-window leakage.
train_parts, val_parts = [], []

for source in pd.unique(sources):
    idx = np.flatnonzero(sources == source)
    split = int(len(idx) * 0.80)
    gap = WINDOW - 1

    if split <= gap or len(idx) - split < 1:
        raise SystemExit(f"Not enough rows for split: {source}")

    train_parts.extend(idx[:split - gap])
    val_parts.extend(idx[split:])

train_idx = np.asarray(train_parts, dtype=int)
val_idx = np.asarray(val_parts, dtype=int)

print(f"Training rows before balancing: {len(train_idx):,}")
print(f"Validation rows: {len(val_idx):,}")
print("Validation uses later rows from each source.")


# Keep the natural training distribution.
scaler = StandardScaler()
X_train = scaler.fit_transform(X[train_idx])
X_val = scaler.transform(X[val_idx])
y_train, y_val = y[train_idx], y[val_idx]
sample_weights = np.ones(len(y_train), dtype=np.float32)

path = OUT / "model_shared_candidate.keras"
joblib.dump(scaler, str(path) + '.scaler')


model = models.Sequential([
    layers.Input(shape=(len(FEATURES),)),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.25),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.15),
    layers.Dense(64, activation="relu"),
    layers.Dense(len(LABELS), activation="sigmoid")
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
        tf.keras.metrics.Precision(name="precision", thresholds=0.5),
        tf.keras.metrics.Recall(name="recall", thresholds=0.5)
    ]
)

cbs = [
    callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),
    callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6
    )
]

print(f"\nTraining rows: {len(train_idx):,}")
print(f"Validation rows: {len(val_idx):,}")

model.fit(
    X_train,
    y_train,
    sample_weight=sample_weights,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=128,
    callbacks=cbs,
    verbose=1
)

# The custom loss is only needed for training.
# Recompile with standard BCE so normal model loading still works.
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
        tf.keras.metrics.Precision(name="precision", thresholds=0.5),
        tf.keras.metrics.Recall(name="recall", thresholds=0.5)
    ]
)

model.save(path)

print("\n=== VALIDATION: BUTTON-BY-BUTTON ===")
pred = model.predict(X_val, verbose=0)
for i, name in enumerate(BUTTONS):
    truth = y_val[:, i] > 0.5
    predicted = pred[:, i] >= 0.5

    tp = np.sum(truth & predicted)
    fp = np.sum(~truth & predicted)
    fn = np.sum(truth & ~predicted)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    print(
        f"{name:>5}: threshold=.50 "
        f"precision={precision:.3f} recall={recall:.3f} "
        f"predicted_positive={predicted.mean():.3%}"
    )

print(f"\nCandidate saved: {path}")
print(f"Candidate scaler: {path}.scaler")
print("Original model_shared.keras remains untouched.")