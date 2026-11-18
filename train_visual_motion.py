from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf

ROOT = Path("frames_CE")
OUT = Path("models")
OUT.mkdir(exist_ok=True)

VIDEO_DIRS = sorted(p for p in ROOT.iterdir() if p.is_dir())
if not VIDEO_DIRS:
    raise SystemExit("No video folders found.")

# Compact grayscale visual representation; resize to keep CPU training practical.
SIZE = (96, 64)
SEQUENCE = 4
STRIDE = 3
X, Y = [], []

for video_dir in VIDEO_DIRS:
    files = sorted(video_dir.glob("*.jpg"))
    if len(files) < SEQUENCE + 1:
        continue

    previous = []
    for idx, path in enumerate(files):
        if idx % STRIDE != 0:
            continue
        im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        im = cv2.resize(im, SIZE, interpolation=cv2.INTER_AREA)
        previous.append(im.astype(np.float32) / 255.0)

    # Preserve the boundary between videos.
    for i in range(len(previous) - SEQUENCE):
        X.append(np.stack(previous[i:i + SEQUENCE], axis=-1))
        Y.append(previous[i + SEQUENCE][..., None])

    print(f"{video_dir.name}: {len(previous):,} sampled frames")

if len(X) < 100:
    raise SystemExit(f"Not enough sequences: {len(X)}")

X = np.asarray(X, dtype=np.float32)
Y = np.asarray(Y, dtype=np.float32)
print("Training sequences:", len(X), "input:", X.shape, "target:", Y.shape)

inputs = tf.keras.Input(shape=(SIZE[1], SIZE[0], SEQUENCE))
z = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
z = tf.keras.layers.MaxPooling2D()(z)
z = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(z)
z = tf.keras.layers.UpSampling2D()(z)
z = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(z)
outputs = tf.keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid")(z)
model = tf.keras.Model(inputs, outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mae")

model.fit(
    X, Y,
    epochs=8,
    batch_size=32,
    validation_split=0.1,
    shuffle=True,
    callbacks=[tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=2, restore_best_weights=True
    )],
)
path = OUT / "visual_motion_ce.keras"
model.save(path)
print("Saved:", path.resolve())
print("This is visual representation learning, not an action policy.")
