
from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

FRAME_DIR = Path("attack_video_frames")
ENCODER_PATH = Path("models/sf2_visual_encoder_only.keras")
OUT = Path("models/attack_visual_classifier.keras")

encoder = tf.keras.models.load_model(ENCODER_PATH, compile=False)
paths = sorted(FRAME_DIR.glob("*.jpg"))

if len(paths) < 20:
    raise SystemExit(f"Need more frames; found {len(paths)}")

frames = []
for p in paths:
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    img = cv2.resize(img, (96, 64)).astype(np.float32) / 255.0
    frames.append(img[..., None])

x = np.asarray(frames, dtype=np.float32)
emb = encoder.predict(x, batch_size=64, verbose=1)

# Unsupervised proxy: large visual changes between adjacent frames.
change = np.zeros(len(emb), dtype=np.float32)
change[1:] = np.mean(np.abs(emb[1:] - emb[:-1]), axis=1)

# Mark the top 20% of visual-change frames as "high motion".
cutoff = np.quantile(change, 0.80)
y = (change >= cutoff).astype(np.float32)

classifier = models.Sequential([
    layers.Input(shape=(emb.shape[1],)),
    layers.Dense(64, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])
classifier.compile(optimizer="adam", loss="binary_crossentropy",
                   metrics=["accuracy"])

classifier.fit(emb, y, epochs=10, batch_size=64, validation_split=0.2)
classifier.save(OUT)

print("Saved:", OUT)
print("Frames:", len(x), "Embedding shape:", emb.shape)
print("High-motion cutoff:", float(cutoff))
print("NOTE: high motion is only a proxy, not a confirmed attack label.")
