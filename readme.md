# Model Kombat

An AI agent for Model Combact (Street Fighter II (SNES) gamebot competition). It plays through the BizHawk emulator using the competition's Python API.

This repo contains my training scripts, frame-processing tools, and Champion Edition reference data, built on top of the competition starter code (`gamebot-competition-master`).

## Repository contents

| Path | Purpose |
|------|---------|
| `train_shared_model.py`, `train_shared_candidate.py`, `train_shared_aggressive.py` | Training scripts for the shared policy model and its variants |
| `train_attack_visual.py`, `train_visual_motion.py` | Training scripts using visual and motion features |
| `process_visual_frames.py`, `track_ce_frames.py`, `inspect_frames.py`, `attack_video_frames.py` | Frame extraction, tracking, and inspection tools |
| `detect_fighter_candidates.py`, `make_contact_sheet.py`, `convert_experience.py` | Fighter detection, contact sheets, and experience conversion |
| `check_model.py`, `check_buttons.py` | Quick sanity checks for models and button mappings |
| `SF2_CE_DATA/` | Champion Edition reference data |
| `PythonAPI/`, `single-player/`, `two-players/` | Competition API and example bots |
| `file_inventory.txt` | Inventory of project files |

## Setup

1. Install BizHawk (the competition ships `bizhawk_prereqs.exe` for the prerequisites).
2. Provide your own Street Fighter II ROM. **The ROM is not included in this repo.**
3. Install Python 3 and the dependencies your scripts need (for example `numpy` and your ML library of choice).
4. Start BizHawk with the competition's Lua/Python bridge, then run a bot from `single-player/` or `two-players/`.

## Training

Run a training script from the project root, for example:

```bash
python train_shared_model.py
```

Adjust paths inside the scripts to point to your local data.

## Not included

- Game ROMs (copyrighted, provide your own)
- Large datasets and trained model checkpoints (too big for GitHub)
- Local editor files (`.vs/`)

## Credits

Based on the Street Fighter II gamebot competition starter code and reference data from community sources. See `README` in the competition folder for the original instructions.
