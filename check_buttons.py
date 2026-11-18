import pandas as pd

d = pd.read_csv("normalized_character_datasets/normalized_dataset_0.csv")
buttons = ["up", "down", "right", "left", "y", "b", "x", "a", "l", "r"]

for b in buttons:
    pct = d[f"player1_buttons_{b}"].mean() * 100
    print(f"{b:>5}: {pct:5.1f}% pressed")

cols = [f"player1_buttons_{b}" for b in buttons]
print("Unique button combinations:", d[cols].drop_duplicates().shape[0])
