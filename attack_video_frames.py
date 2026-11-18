
from pathlib import Path
import cv2
import csv

VIDEO = Path("attack_reference.mp4")
OUT = Path("attack_video_frames")
FPS_SAMPLE = 5

OUT.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(VIDEO))
if not cap.isOpened():
    raise SystemExit(f"Cannot open video: {VIDEO.resolve()}")

source_fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
duration = total_frames / source_fps if source_fps else 0

print(f"Video FPS: {source_fps:.2f}")
print(f"Duration: {duration:.1f}s")
print(f"Extracting at {FPS_SAMPLE} frames/sec")

interval = 1.0 / FPS_SAMPLE
next_time = 0.0
index = 0

with open(OUT / "timestamps.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "timestamp_seconds"])

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        if timestamp + 1e-6 >= next_time:
            filename = f"frame_{index:06d}.jpg"
            cv2.imwrite(str(OUT / filename), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            writer.writerow([filename, round(timestamp, 4)])
            index += 1
            next_time += interval

cap.release()
print(f"Done: {index:,} frames saved to {OUT.resolve()}")
print(f"Timestamp index: {(OUT / 'timestamps.csv').resolve()}")
