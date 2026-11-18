import json
import os
import random

from command import Command
from buttons import Buttons


class TabularQLearner:
    # Actions are available choices, not a scripted combat sequence.
    ACTIONS = [
        (),                         # 0: neutral
        ("FWD",),                   # 1: approach
        ("BACK",),                  # 2: retreat / stand defense
        ("UP", "FWD"),              # 3: jump forward
        ("UP", "BACK"),             # 4: jump backward
        ("DOWN",),                  # 5: crouch
        ("Y",),                     # 6: punch
        ("B",),                     # 7: punch
        ("X",),                     # 8: punch
        ("A",),                     # 9: kick
        ("L",),                     # 10: kick
        ("R",),                     # 11: kick
        ("UP", "Y"),                # 12: jump punch
        ("UP", "A"),                # 13: jump kick
        ("DOWN", "B"),              # 14: crouch punch
        ("DOWN", "A"),              # 15: crouch kick
        ("FWD", "Y"),               # 16: moving punch
        ("FWD", "A"),               # 17: moving kick
        ("DOWN", "BACK"),           # 18: crouch defense
    ]

    BUTTON_NAMES = ("UP", "DOWN", "LEFT", "RIGHT",
                    "Y", "B", "X", "A", "L", "R")


    def __init__(self, save_path):
        self.save_path = save_path
        self.q = {}

        self.alpha = 0.15
        self.gamma = 0.90
        self.steps = 0
        self.epsilon = 0.25

        self.prev_state = None
        self.prev_action = None
        self.prev_my_health = None
        self.prev_opp_health = None

        try:
            with open(save_path, "r", encoding="utf-8") as f:
                self.q = json.load(f)
            print(f"[RL] Loaded {len(self.q)} learned states.")
        except (OSError, ValueError):
            print("[RL] Starting a new Q-table.")

    def row(self, state):
        if state not in self.q or len(self.q[state]) != len(self.ACTIONS):
            self.q[state] = [0.0] * len(self.ACTIONS)
        return self.q[state]

    def get_state(self, gs, player_id):
        p1, p2 = gs.player1, gs.player2
        if str(player_id) == "1":
            me, opp = p1, p2
        else:
            me, opp = p2, p1

        distance = abs(float(me.x_coord) - float(opp.x_coord))
        distance_bin = 0 if distance < 40 else 1 if distance < 100 else 2

        health_diff = float(me.health) - float(opp.health)
        health_bin = 0 if health_diff < -15 else 2 if health_diff > 15 else 1

        # Compact state representation keeps learning tractable.
        values = (
            distance_bin,
            health_bin,
            int(bool(opp.is_jumping)),
            int(bool(opp.is_crouching)),
            int(bool(opp.is_player_in_move)),
            int(bool(me.is_jumping)),
            int(bool(me.is_crouching)),
        )

        return ",".join(map(str, values)), me, opp

    def save(self):
        try:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            temp_path = self.save_path + ".tmp"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(self.q, f)
            os.replace(temp_path, self.save_path)
        except OSError as exc:
            print(f"[RL] Could not save Q-table: {exc}")

    def make_command(self, action_index, me, opp, player_id):
        forward = (
            "RIGHT" if float(me.x_coord) < float(opp.x_coord)
            else "LEFT"
        )
        backward = "LEFT" if forward == "RIGHT" else "RIGHT"

        names = set(self.ACTIONS[action_index])
        resolved = set()

        for name in names:
            if name == "FWD":
                resolved.add(forward)
            elif name == "BACK":
                resolved.add(backward)
            else:
                resolved.add(name)

        button_map = {
            name: name in resolved for name in self.BUTTON_NAMES
        }

        cmd = Command()
        if str(player_id) == "1":
            cmd.player_buttons = Buttons(button_map)
        else:
            cmd.player2_buttons = Buttons(button_map)
        return cmd

    def fight(self, gs, player_id):
        state, me, opp = self.get_state(gs, player_id)
        result = str(gs.fight_result)

        terminal = result in ("P1", "P2")
        done = bool(gs.is_round_over) or terminal

        # Reward actual damage dealt and penalize damage received.
        if self.prev_state is not None:
            my_health = float(me.health)
            opp_health = float(opp.health)

            my_loss = self.prev_my_health - my_health
            opp_loss = self.prev_opp_health - opp_health

            # Avoid treating round-reset health restoration as a reward.
            health_reset = my_loss < -20 or opp_loss < -20
            reward = 0.0 if health_reset else opp_loss - 0.8 * my_loss

            if terminal:
                won = (
                    (result == "P1" and str(player_id) == "1")
                    or (result == "P2" and str(player_id) == "2")
                )
                reward += 80.0 if won else -80.0

            old_row = self.row(self.prev_state)
            next_max = 0.0 if done else max(self.row(state))
            target = reward + self.gamma * next_max

            old_row[self.prev_action] += (
                self.alpha * (target - old_row[self.prev_action])
            )

        if done:
            self.prev_state = None
            self.prev_action = None
            self.save()

            # Send neutral input while the round is over.
            cmd = Command()
            if str(player_id) == "1":
                cmd.player_buttons = Buttons({})
            else:
                cmd.player2_buttons = Buttons({})
            return cmd

        # Gradually reduce exploration while retaining some experimentation.
        epsilon = max(0.08, 0.25 * (0.9997 ** self.steps))
        row = self.row(state)

        if random.random() < epsilon:
            action = random.randrange(len(self.ACTIONS))
        else:
            best_value = max(row)
            best_actions = [
                i for i, value in enumerate(row)
                if value == best_value
            ]
            action = random.choice(best_actions)

        self.steps += 1

        self.prev_state = state
        self.prev_action = action
        self.prev_my_health = float(me.health)
        self.prev_opp_health = float(opp.health)

        if self.steps % 25 == 0:
            self.save()

        if self.steps % 250 == 0:
            print(
                f"[RL] Steps={self.steps}, "
                f"states={len(self.q)}, epsilon={epsilon:.3f}"
            )

        return self.make_command(action, me, opp, player_id)