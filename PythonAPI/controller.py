import socket
import json
import sys
import time
from game_state import GameState
from command import Command
from buttons import Buttons
from listen_to_key import get_current_keypress
from make_dataset import record_frame
import os

EXPERIENCE_FILE = "experience.jsonl"


def snapshot(gs):
    def player(p):
        return {
            "character": p.player_id,
            "health": p.health,
            "x": p.x_coord,
            "y": p.y_coord,
            "jumping": p.is_jumping,
            "crouching": p.is_crouching,
            "in_move": p.is_player_in_move,
            "move": p.move_id,
        }

    return {
        "p1": player(gs.player1),
        "p2": player(gs.player2),
        "timer": gs.timer,
        "result": gs.fight_result,
        "round_started": gs.has_round_started,
        "round_over": gs.is_round_over,
    }


def calculate_reward(previous, current, player_id):
    # Ignore health differences across round resets.
    p1_before = previous["p1"]["health"]
    p2_before = previous["p2"]["health"]
    p1_after = current["p1"]["health"]
    p2_after = current["p2"]["health"]

    reward = 0.0

    if not current["round_over"]:
        if p1_after <= p1_before and p2_after <= p2_before:
            damage_dealt = (
                p2_before - p2_after if player_id == 1
                else p1_before - p1_after
            )
            damage_taken = (
                p1_before - p1_after if player_id == 1
                else p2_before - p2_after
            )

            # Reward damage dealt; penalize damage received.
            reward += (damage_dealt - damage_taken) / 10.0

    result = str(current["result"]).upper()
    if result in ("P1", "P2"):
        reward += 10.0 if result == f"P{player_id}" else -10.0

    return reward

player_id = sys.argv[1]
MODE = 'record' if len(sys.argv) > 2 and sys.argv[2] == 'record' else 'bot'
MODEL_TYPE = 'rnn' if len(sys.argv) > 2 and sys.argv[2] == 'rnn' else 'standard'
port = 9999 if player_id == '1' else 10000

def connect(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('127.0.0.1', port))
    s.listen(1)
    client, _ = s.accept()
    print('Connected to game!')
    return client

def send(sock, cmd):
    payload = json.dumps(cmd.object_to_dict())
    print("[Controller] Sending:", payload)
    print("\n[Controller] Detailed Debug:")
    print("1. Command object button states:", vars(cmd.player_buttons))
    print("2. Serialized command:", payload)
    print("3. Socket info:", sock.getsockname(), "->", sock.getpeername())
    sock.sendall(payload.encode())

def receive(sock):
    data = json.loads(sock.recv(4096).decode())
    return GameState(data)

def main():
    sock = connect(port)
    cmd = Command()
    previous_state = None
    previous_action = None
    pid = int(player_id)

    log_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        EXPERIENCE_FILE
    )
    log_file = open(log_path, "a", encoding="utf-8")
    print(f"[Logger] Saving experience to: {log_path}")
    player_id_set = False
    while True:
        gs = receive(sock)

        current_state = snapshot(gs)

        # The new observation completes the previous action's transition.
        if previous_state is not None and previous_action is not None:
            result = str(current_state["result"]).upper()
            done = bool(current_state["round_over"]) or result in ("P1", "P2")

            record = {
                "state": previous_state,
                "action": previous_action,
                "reward": calculate_reward(
                    previous_state, current_state, pid
                ),
                "next_state": current_state,
                "done": done,
                "player_id": pid,
            }

            log_file.write(json.dumps(record) + "\n")
            log_file.flush()

        if MODE != 'record' and not player_id_set:
            from bot import Bot
            print("[Controller] Using standard model bot")
                
            bot = Bot(player_id=gs.player1.player_id)
            player_id_set = True

        if MODE == 'record':
            keys = get_current_keypress()
            # Forward human input
            cmd.player_buttons = Buttons({k: True for k in keys})
            record_frame(gs, keys)
        else:
            print("\n[Controller] Getting bot command...")
            cmd = bot.fight(gs, player_id)
            print(f"[Controller] Bot command received: {cmd.object_to_dict()}")
        
        
        send(sock, cmd)
        
        previous_state = current_state
        previous_action = cmd.object_to_dict()

        
if __name__ == '__main__':
    main()