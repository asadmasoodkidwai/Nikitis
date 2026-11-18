from pathlib import Path
from PIL import Image, ImageDraw
import pandas as pd
import math

root = Path(__file__).resolve().parent
df = pd.read_csv(root / "video_features_ce" / "frame_features.csv")

# Spread samples across all six matches.
n = min(36, len(df))
indices = [round(i * (len(df)-1) / (n-1)) for i in range(n)]
thumbs = []

for idx in indices:
    row = df.iloc[idx]
    p = root / Path(str(row["frame_path"]))
    with Image.open(p) as im:
        im = im.convert("RGB")
        im.thumbnail((240, 140))
        thumbs.append((im.copy(), str(row["frame_path"]).split("\\")[0], int(row["frame_index"])))

cols, cell_w, cell_h = 4, 250, 170
sheet = Image.new("RGB", (cols*cell_w, math.ceil(len(thumbs)/cols)*cell_h), "white")
draw = ImageDraw.Draw(sheet)

for i, (im, folder, frame) in enumerate(thumbs):
    x, y = (i % cols)*cell_w, (i // cols)*cell_h
    sheet.paste(im, (x, y))
    draw.text((x+4, y+142), f"{folder[:22]} | frame {frame}", fill="black")

out = root / "video_features_ce" / "contact_sheet.jpg"
sheet.save(out, quality=90)
print("Saved:", out)
