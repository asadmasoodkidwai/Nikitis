from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path("frames_CE")
OUT = Path("video_features_ce")
OUT.mkdir(exist_ok=True)

files = sorted(ROOT.rglob("*.jpg"))
if not files:
    raise SystemExit("ERROR: No JPG frames found in frames_CE")

rows = []
sample_every = 3  # process every third extracted frame

for i in tqdm(range(0, len(files), sample_every), desc="Scanning frames"):
    path = files[i]
    im = cv2.imread(str(path))
    if im is None:
        continue

    h, w = im.shape[:2]
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    edges = float(cv2.Canny(gray, 60, 140).mean())

    # SF2 gameplay generally has substantial color and edge detail.
    saturation = float(hsv[:, :, 1].mean())
    valid = brightness > 12 and edges > 1.5 and saturation > 12

    # Record colored connected regions as candidate fighter/effect regions.
    # These are proposals only, not guaranteed player detections.
    mask = cv2.inRange(hsv, np.array([0, 45, 45]), np.array([179, 255, 255]))
    mask[:int(h * 0.12), :] = 0
    mask[int(h * 0.90):, :] = 0
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)

    candidates = []
    for j in range(1, count):
        x, y, bw, bh, area = stats[j]
        if 35 <= area <= 12000 and bw >= 4 and bh >= 8:
            candidates.append((int(x), int(y), int(bw), int(bh), int(area)))

    candidates.sort(key=lambda z: z[4], reverse=True)
    candidates = candidates[:12]

    row = {
        "frame_path": str(path),
        "frame_index": i,
        "width": w,
        "height": h,
        "brightness": round(brightness, 2),
        "saturation": round(saturation, 2),
        "edge_density": round(edges, 2),
        "likely_gameplay": valid,
        "candidate_count": len(candidates),
    }

    # Store up to 6 largest candidate regions.
    for k in range(6):
        if k < len(candidates):
            x, y, bw, bh, area = candidates[k]
            row.update({
                f"r{k+1}_x": x, f"r{k+1}_y": y,
                f"r{k+1}_w": bw, f"r{k+1}_h": bh,
                f"r{k+1}_area": area,
            })
        else:
            row.update({
                f"r{k+1}_x": -1, f"r{k+1}_y": -1,
                f"r{k+1}_w": 0, f"r{k+1}_h": 0,
                f"r{k+1}_area": 0,
            })

    rows.append(row)

df = pd.DataFrame(rows)
csv_path = OUT / "frame_features.csv"
df.to_csv(csv_path, index=False)

print("\nDONE")
print(f"Frames scanned: {len(df):,}")
print(f"Likely gameplay candidates: {int(df['likely_gameplay'].sum()):,}")
print(f"CSV: {csv_path.resolve()}")
print("Note: colored regions are candidates, not verified fighter identities.")
