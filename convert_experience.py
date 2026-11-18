import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "experience.jsonl"
OUT = ROOT / "flattened_window_datasets" / "windowed_dataset_experience.csv"

WINDOW = 6
STATE = [
    "timer", "fight_result", "has_round_started", "is_round_over",
    "player1_id", "p1_health", "p1_x", "p1_y", "p1_jumping",
    "p1_crouching", "p1_in_move", "p1_move_id", "player2_id",
    "p2_health", "p2_x", "p2_y", "p2_jumping", "p2_crouching",
    "p2_in_move", "p2_move_id", "diff_x", "diff_y", "diff_health"
]
BUTTONS = ["up", "down", "right", "left", "y", "b", "x", "a", "l", "r"]
FEATURES = [
    f"{name}_t-{t}"
    for t in range(WINDOW - 1, -1, -1)
    for name in STATE + [f"player1_buttons_{b}" for b in BUTTONS]
]
LABELS = [f"player1_buttons_{b}" for b in BUTTONS]

def make_row(record):
    s = record["state"]
    p1, p2 = s["p1"], s["p2"]
    action = record["action"]["p1"]

    result = s.get("result", "NOT_OVER")
    result = {"NOT_OVER": 0, "P1": 1, "P2": 2}.get(result, 0)

    row = {
        "timer": s.get("timer", 0),
        "fight_result": result,
        "has_round_started": int(bool(s.get("round_started", False))),
        "is_round_over": int(bool(s.get("round_over", False))),
        "player1_id": p1.get("character", 0),
        "p1_health": p1.get("health", 0),
        "p1_x": p1.get("x", 0),
        "p1_y": p1.get("y", 0),
        "p1_jumping": int(bool(p1.get("jumping", False))),
        "p1_crouching": int(bool(p1.get("crouching", False))),
        "p1_in_move": int(bool(p1.get("in_move", False))),
        "p1_move_id": p1.get("move", 0),
        "player2_id": p2.get("character", 0),
        "p2_health": p2.get("health", 0),
        "p2_x": p2.get("x", 0),
        "p2_y": p2.get("y", 0),
        "p2_jumping": int(bool(p2.get("jumping", False))),
        "p2_crouching": int(bool(p2.get("crouching", False))),
        "p2_in_move": int(bool(p2.get("in_move", False))),
        "p2_move_id": p2.get("move", 0),
        "diff_x": p2.get("x", 0) - p1.get("x", 0),
        "diff_y": p2.get("y", 0) - p1.get("y", 0),
        "diff_health": p1.get("health", 0) - p2.get("health", 0),
    }

    for b in BUTTONS:
        row[f"player1_buttons_{b}"] = int(bool(action.get(b.capitalize(), False)))
    return row

records = []
with SOURCE.open(encoding="utf-8") as f:
    for line in f:
        if line.strip():
            try:
                records.append(make_row(json.loads(line)))
            except (KeyError, TypeError, ValueError) as e:
                print("Skipping malformed record:", e)

rows = []
history = []

for r in records:
    history.append(r)
    if len(history) > WINDOW:
        history.pop(0)
    if len(history) != WINDOW:
        continue

    # Historical states/actions are features; current action is the label.
    out = {}
    for t, past in enumerate(history):
        lag = WINDOW - 1 - t
        for name in STATE:
            out[f"{name}_t-{lag}"] = past[name]
        for b in BUTTONS:
            out[f"player1_buttons_{b}_t-{lag}"] = past[f"player1_buttons_{b}"]

    for b in BUTTONS:
        out[f"player1_buttons_{b}"] = history[-1][f"player1_buttons_{b}"]
    rows.append(out)

df = pd.DataFrame(rows, columns=FEATURES + LABELS)
df.to_csv(OUT, index=False)
print("Saved:", OUT)
print("Rows:", len(df), "| Columns:", len(df.columns))
print("Expected columns: 208")
print("Action positives:")
print(df[LABELS].sum().to_string())

