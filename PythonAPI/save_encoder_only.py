from pathlib import Path
import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent
model = tf.keras.models.load_model(
    ROOT / "models" / "sf2_visual_encoder.keras",
    compile=False
)

encoder = tf.keras.Model(
    inputs=model.input,
    outputs=model.get_layer("visual_embedding").output,
    name="sf2_visual_encoder_only"
)

out = ROOT / "models" / "sf2_visual_encoder_only.keras"
encoder.save(out)

print("Saved:", out)
print("Input:", encoder.input_shape)
print("Output:", encoder.output_shape)
