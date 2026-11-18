import sys
import numpy as np
sys.path.insert(0, "PythonAPI")
from bot import Bot

b = Bot(player_id=0)
print("Model loaded:", type(b).__name__)

# Inspect the model's raw outputs on a sample input.
model = b.model if hasattr(b, "model") else None
if model is None:
    print("Find model attribute in bot.py before proceeding.")
else:
    x = np.zeros((1, 138), dtype=np.float32)
    y = model.predict(x, verbose=0).flatten()
    buttons = ["up", "down", "right", "left", "y", "b", "x", "a", "l", "r"]
    print("Raw predictions:")
    for name, value in zip(buttons, y):
        print(f"{name:>5}: {value:.4f}")
