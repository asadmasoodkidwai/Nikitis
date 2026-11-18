import os
import joblib
import tensorflow as tf
from collections import deque
from command import Command
from buttons import Buttons
from device_utils import get_device
from rl_agent import TabularQLearner

# Per-frame console printing is expensive (each print() is effectively a
# flush/syscall) and this file was doing 10+ of them every single frame.
# Off by default - set SF2_DEBUG=1 to see prediction probabilities per frame.
DEBUG = os.environ.get('SF2_DEBUG', '0') == '1'

# define constants the same way as done when training
WINDOW_SIZE = 6
STATE_FEATURES = ['timer', 'fight_result', 'has_round_started', 'is_round_over','player1_id', 'p1_health', 'p1_x', 'p1_y', 'p1_jumping', 'p1_crouching', 'p1_in_move', 'p1_move_id','player2_id', 'p2_health', 'p2_x', 'p2_y', 'p2_jumping', 'p2_crouching', 'p2_in_move', 'p2_move_id','diff_x', 'diff_y', 'diff_health']
# making feature columns just like during training time - order must stay
# in lockstep with how the scaler was originally fit
FEATURE_COLS = []
for t in range(WINDOW_SIZE-1, -1, -1):
    suffix = f"_t-{t}"
    for feat in STATE_FEATURES:
        FEATURE_COLS.append(feat + suffix)
BUTTONS = ['UP', 'DOWN', 'RIGHT', 'LEFT', 'Y', 'B', 'X', 'A', 'L', 'R']
P1_BUTTON_COLS = [f'player1_buttons_{b}' for b in BUTTONS]

FIGHT_MAP = {'NOT_OVER': 0, 'P1': 1, 'P2': 2}

class SpecialMoveDetector:
    """Thresholded special-move classifier output with a short confidence window."""

    def __init__(self, threshold=0.75, confirm_frames=2):
        self.threshold = threshold
        self.confirm_frames = confirm_frames
        self.streak = 0

    def update(self, probability):
        if probability >= self.threshold:
            self.streak += 1
        else:
            self.streak = 0

        return self.streak >= self.confirm_frames

class Bot:
    def __init__(self,player_id=0, model_path=None):
        self.buttons = Buttons()
        self.cmd = Command()
        # locate model & scaler
        if model_path is None:
            base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
            model_path = os.path.join(base, 'model_shared_moredata.keras')

        # ---- auto-detect GPU, fall back to CPU automatically ----
        self.device = get_device()

        with tf.device(self.device):
            self.model = tf.keras.models.load_model(model_path)

        scaler = joblib.load(model_path + '.scaler')
        # Move the scaler's mean/scale into the graph as constants so scaling
        # happens on-device inside one compiled call instead of building a
        # fresh pandas DataFrame + running sklearn's transform() every frame.
        self.scaler_mean_ = tf.constant(scaler.mean_, dtype=tf.float32)
        self.scaler_scale_ = tf.constant(scaler.scale_, dtype=tf.float32)

        self.special_detector = SpecialMoveDetector()

        n_features = len(FEATURE_COLS)

        self.use_rl = os.environ.get("SF2_RL", "0") == "1"

        rl_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), "..", "models", "rl_qtable.json"
            )
        )
        self.rl_agent = TabularQLearner(rl_path)

        print(f"[Bot] RL mode: {self.use_rl}")

        def _make_compiled_predict(use_xla):
            @tf.function(
                input_signature=[tf.TensorSpec(shape=[1, n_features], dtype=tf.float32)],
                jit_compile=use_xla,
            )
            def compiled_predict(x):
                x_scaled = (x - self.scaler_mean_) / self.scaler_scale_
                return self.model(x_scaled, training=False)
            return compiled_predict

        with tf.device(self.device):
            dummy = tf.zeros((1, n_features), dtype=tf.float32)
            try:
                # Try XLA first - fastest option when the model's ops support it.
                self._compiled_predict = _make_compiled_predict(use_xla=True)
                for _ in range(3):
                    _ = self._compiled_predict(dummy)
            except Exception as e:
                # Some layer/op in the model may not be XLA-compatible - fall
                # back to a plain compiled graph rather than crashing the bot.
                print(f"[Bot] XLA compilation failed ({e}); falling back to standard @tf.function.")
                self._compiled_predict = _make_compiled_predict(use_xla=False)
                for _ in range(3):
                    _ = self._compiled_predict(dummy)

        print(f"[Bot] Model loaded and warmed up on {self.device}.")

        # init frame buffer
        self.buffer = deque(maxlen=WINDOW_SIZE)
        empty = {feat: 0 for feat in STATE_FEATURES}
        empty['fight_result'] = 'NOT_OVER'
        for _ in range(WINDOW_SIZE):
            self.buffer.append(empty.copy())

    def _frame_to_dict(self, gs):
        # map GameState to raw feature dict (no suffix)
        p1, p2 = gs.player1, gs.player2
        d = {
            'timer': gs.timer,
            'fight_result': gs.fight_result,
            'has_round_started': int(gs.has_round_started),
            'is_round_over': int(gs.is_round_over),
            'player1_id': p1.player_id,
            'p1_health': p1.health,
            'p1_x': p1.x_coord,
            'p1_y': p1.y_coord,
            'p1_jumping': int(p1.is_jumping),
            'p1_crouching': int(p1.is_crouching),
            'p1_in_move': int(p1.is_player_in_move),
            'p1_move_id': p1.move_id,
            'player2_id': p2.player_id,
            'p2_health': p2.health,
            'p2_x': p2.x_coord,
            'p2_y': p2.y_coord,
            'p2_jumping': int(p2.is_jumping),
            'p2_crouching': int(p2.is_crouching),
            'p2_in_move': int(p2.is_player_in_move),
            'p2_move_id': p2.move_id,
            'diff_x': p1.x_coord - p2.x_coord,
            'diff_y': p1.y_coord - p2.y_coord,
            'diff_health': p1.health - p2.health,
        }
        return d

    def fight(self, gs, player_id):
        # 1. append new frame
        if self.use_rl:
            return self.rl_agent.fight(gs, player_id)
        
        raw = self._frame_to_dict(gs)
        self.buffer.append(raw)


        # 2. build flattened feature list (same column order as training)
        flat = []
        for t in range(WINDOW_SIZE-1, -1, -1):
            frame = self.buffer[WINDOW_SIZE-1 - t]
            for feat in STATE_FEATURES:
                val = frame[feat]
                flat.append(FIGHT_MAP[val] if feat == 'fight_result' else int(val))

        # 3. scale + predict in one compiled, on-device call
        #    (replaces the old per-frame pandas DataFrame + scaler.transform()
        #    + model.predict() - model.predict() in particular carries large
        #    fixed overhead per call that dwarfs actual compute for a single
        #    sample; calling the compiled function directly avoids all of it)
        x = tf.constant([flat], dtype=tf.float32)


        preds = self._compiled_predict(x).numpy()[0]

        if DEBUG:
            print("MODEL OUTPUT:", {
                b: round(float(preds[i]), 4)
                for i, b in enumerate(BUTTONS)
            })

        if DEBUG:
            print("\nPrediction probabilities for each button:")
            for button, prob in zip(BUTTONS, preds):
                if prob > 0.5:
                    print(f"{button}: {prob:.2%}")



               # Use the trained model's predictions.
        p1, p2 = gs.player1, gs.player2
        is_p1 = str(player_id) == "1"
        me, opponent = (p1, p2) if is_p1 else (p2, p1)
        
        THRESHOLDS = {
            "UP":    0.25,  # More opportunities for jump attacks
            "DOWN":  0.35,  # Keep crouching, reduce crouch-locking
            "LEFT":  0.48,  # Slightly more responsive movement
            "RIGHT": 0.48,
            "Y":     0.11,  # More punch attempts
            "B":     0.12,  # More punch attempts
            "X":     0.14,  # More attack attempts
            "A":     0.12,  # More attack attempts
            "L":     0.50,
            "R":     0.50,
        }

        probs = {b: float(preds[i]) for i, b in enumerate(BUTTONS)}
        btn_map = {b: probs[b] > THRESHOLDS[b] for b in BUTTONS}

        if DEBUG:
            print("Button probabilities:", {
                b: round(float(preds[i]), 3)
                for i, b in enumerate(BUTTONS)
            })

        # Use one attack button at a time; don't force Y and A together.
        # Preserve the model's attack predictions.

        cmd = Command()
        if player_id == "1":
            cmd.player_buttons = Buttons(btn_map)
        else:
            cmd.player2_buttons = Buttons(btn_map)

        if DEBUG:
            active_buttons = [btn for btn, state in btn_map.items() if state]
            print(f"\nActive buttons for Player {player_id}:", ", ".join(active_buttons) if active_buttons else "None")

        return cmd


