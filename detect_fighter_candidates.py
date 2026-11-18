from pathlib import Path
import cv2
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent
src = pd.read_csv(root / "video_features_ce" / "frame_features.csv")
out = root / "video_features_ce" / "fighter_candidates.csv"
rows = []

for _, r in src.iterrows():
    p = root / Path(str(r["frame_path"]))
    im = cv2.imread(str(p))
    if im is None:
        continue

    h, w = im.shape[:2]
    # Work only in the gameplay area, excluding HUD and lower border.
    y0, y1 = int(.14*h), int(.91*h)
    roi = im[y0:y1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Saturated colored pixels are candidate foreground, not guaranteed fighters.
    mask = cv2.inRange(hsv, np.array([0, 65, 45]), np.array([179, 255, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if 180 <= area <= .12*w*h and 18 <= bh <= .65*h and 8 <= bw <= .25*w:
            boxes.append((x, y+y0, bw, bh, area))

    boxes.sort(key=lambda b: b[4], reverse=True)
    top = boxes[:8]
    # Save candidate boxes for inspection; don't label them as fighters yet.
    rows.append({
        "frame_path": str(r["frame_path"]),
        "frame_index": int(r["frame_index"]),
        "candidate_boxes": repr(top),
        "candidate_box_count": len(boxes),
        "largest_box_area": float(top[0][4]) if top else 0,
    })

pd.DataFrame(rows).to_csv(out, index=False)
print("Saved:", out)
print("Rows:", len(rows))
print("Candidate count distribution:")
print(pd.Series([r["candidate_box_count"] for r in rows]).value_counts().sort_index().to_string())
