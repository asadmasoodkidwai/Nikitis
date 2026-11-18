from pathlib import Path
import numpy as np
import tensorflow as tf

# Script lives in PythonAPI; datasets/models live in the project root.
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "video_features_ce" / "processed_frames.npz"
MODEL_OUT = ROOT / "models" / "sf2_visual_encoder.keras"
EMBED_OUT = ROOT / "video_features_ce" / "visual_embeddings.npy"

if not SRC.is_file():
    raise FileNotFoundError(f"Processed frames not found: {SRC}")

data = np.load(SRC)
x = data["frames"].astype(np.float32)[..., None] / 255.0
print("Training frames:", x.shape)

# Compact convolutional autoencoder.
inp = tf.keras.Input(shape=(64, 96, 1), name="frame")
z = tf.keras.layers.Conv2D(
    16, 3, strides=2, padding="same", activation="relu"
)(inp)
z = tf.keras.layers.Conv2D(
    32, 3, strides=2, padding="same", activation="relu"
)(z)
z = tf.keras.layers.Conv2D(
    64, 3, strides=2, padding="same", activation="relu"
)(z)

latent_shape = tuple(int(v) for v in z.shape[1:])
flat = tf.keras.layers.Flatten()(z)
embedding = tf.keras.layers.Dense(
    128, activation="relu", name="visual_embedding"
)(flat)

d = tf.keras.layers.Dense(int(np.prod(latent_shape)), activation="relu")(embedding)
d = tf.keras.layers.Reshape(latent_shape)(d)
d = tf.keras.layers.Conv2DTranspose(
    64, 3, strides=2, padding="same", activation="relu"
)(d)
d = tf.keras.layers.Conv2DTranspose(
    32, 3, strides=2, padding="same", activation="relu"
)(d)
d = tf.keras.layers.Conv2DTranspose(
    16, 3, strides=2, padding="same", activation="relu"
)(d)
reconstruction = tf.keras.layers.Conv2D(
    1, 3, padding="same", activation="sigmoid"
)(d)

autoencoder = tf.keras.Model(inp, reconstruction, name="sf2_autoencoder")
encoder = tf.keras.Model(inp, embedding, name="sf2_visual_encoder")
autoencoder.compile(optimizer="adam", loss="mse")

autoencoder.summary()
autoencoder.fit(
    x,
    x,
    epochs=5,
    batch_size=64,
    validation_split=0.05,
    shuffle=True,
)

MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
autoencoder.save(MODEL_OUT)

embeddings = encoder.predict(x, batch_size=128, verbose=1)
np.save(EMBED_OUT, embeddings)

print("Encoder saved:", MODEL_OUT)
print("Embeddings saved:", EMBED_OUT)
print("Embedding shape:", embeddings.shape)
print("Reconstruction loss:", autoencoder.evaluate(x[:512], x[:512], verbose=0))
