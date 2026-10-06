#!/usr/bin/env python3
"""
solver_client.py — Interactive player for local ARC-AGI-3 games.

Usage:
    python solver_client.py                      # list public API + local games/
    python solver_client.py <game_name>          # local game in games/<game_name>/
    python solver_client.py maze
    python solver_client.py maze --level 3       # start at level 3 (0-indexed)
    python solver_client.py maze --seed 0        # deterministic env generation
    python solver_client.py ls20                 # falls back to arcade remote

    # Troubleshooting mode — headless, saves each frame as PNG:
    python solver_client.py ps:absorb --troubleshoot
    python solver_client.py ps:absorb --troubleshoot --actions "uldr" --level 2
    python solver_client.py ps:absorb --troubleshoot --num-steps 30

Controls:
    w / ↑        ACTION1 — up
    s / ↓        ACTION2 — down
    a / ←        ACTION3 — left
    d / →        ACTION4 — right
    e / Space    ACTION5 — interact / select / execute
    z / u        ACTION7 — undo
    r            RESET current level
    q / Escape   quit
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import queue
import random
from dataclasses import dataclass
import matplotlib
matplotlib.use("TkAgg")  # persistent GUI backend
# Clear all built-in key bindings so they don't intercept game keys
# (e.g. 's' saves figure, 'q' closes window, 'f' fullscreens, 'r' resets view)
import matplotlib.pyplot as plt
for _k in list(plt.rcParams):
    if _k.startswith("keymap."):
        plt.rcParams[_k] = []

import arc_agi
from arc_agi import OperationMode
from arc_agi.rendering import frame_to_rgb_array
from arcengine import ActionInput, FrameDataRaw, GameAction, GameState
from adapters import GymGridworldsAdapter, MinigridAdapter, ProcGenAdapter, PuzzleScriptAdapter
# Environment resolution (the adapter wrappers + native arc_agi loader + the cascade) lives in
# the shared game_envs module, used verbatim by solver.py too. solver_client owns only the
# interactive concerns (GUI renderer, key handling, listing/validation).
import game_envs
from game_envs import resolve_env, NOOP_RENDERER, _get_underlying_game, _jump_to_level


# ---------------------------------------------------------------------------
# Persistent GUI renderer — keeps one window open for the whole session
# ---------------------------------------------------------------------------

class _PersistentRenderer:
    """Callable renderer that keeps a single matplotlib window open."""

    def __init__(self, key_queue: queue.Queue, scale: int = 4, frame_delay_s: float = 0.08) -> None:
        self._scale = scale
        self._key_queue = key_queue
        self._frame_delay_s = frame_delay_s
        self.fig: plt.Figure | None = None
        self._ax_status = None
        self._ax_bar = None
        self._ax_game = None
        self._status_text = None
        self._im = None
        self._bar_fill = None
        self._steps_remaining: int | None = None
        self._max_steps: int | None = None

    def __call__(self, steps: int, frame_data: FrameDataRaw) -> None:
        frames = frame_data.frame
        if not frames:
            return
        if self.fig is None or not plt.fignum_exists(self.fig.number):
            self.fig, (self._ax_status, self._ax_bar, self._ax_game) = plt.subplots(
                3, 1,
                figsize=(6, 6.5),
                gridspec_kw={"height_ratios": [1, 1, 22], "hspace": 0.0},
            )
            # --- Status text (top strip) ---
            self._ax_status.set_xlim(0, 1)
            self._ax_status.set_ylim(0, 1)
            self._ax_status.axis("off")
            self._status_text = self._ax_status.text(
                0.5, 0.5,
                "ARC-AGI-3  |  w/a/s/d=move  r=reset  q=quit",
                ha="center", va="center", fontsize=9,
                transform=self._ax_status.transAxes,
            )
            # --- Progress bar (middle strip) ---
            self._ax_bar.set_xlim(0, 1)
            self._ax_bar.set_ylim(0, 1)
            self._ax_bar.axis("off")
            # Dark background track
            self._ax_bar.add_patch(
                plt.Rectangle((0, 0.1), 1, 0.8, color="#333333", zorder=1)
            )
            # Coloured fill — width updated by update_step_budget()
            self._bar_fill = plt.Rectangle((0, 0.1), 0, 0.8, color="#44cc44", zorder=2)
            self._ax_bar.add_patch(self._bar_fill)
            self._update_bar_visual()
            # --- Game image (main area) ---
            self._ax_game.axis("off")
            rgb = frame_to_rgb_array(steps, frames[0], scale=self._scale)
            self._im = self._ax_game.imshow(rgb, interpolation="nearest")
            plt.tight_layout(pad=0.5)
            plt.ion()
            self.fig.canvas.mpl_connect("key_press_event", self._on_key)
            self.fig.canvas.mpl_connect("button_press_event", self._on_click)
            plt.show(block=False)
        for idx, frame in enumerate(frames):
            rgb = frame_to_rgb_array(steps, frame, scale=self._scale)
            self._im.set_data(rgb)
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            if idx < len(frames) - 1:
                plt.pause(self._frame_delay_s)

    def _update_bar_visual(self) -> None:
        if self._bar_fill is None:
            return
        if self._steps_remaining is None or self._max_steps is None or self._max_steps == 0:
            self._bar_fill.set_width(0)
            return
        fraction = max(0.0, min(1.0, self._steps_remaining / self._max_steps))
        self._bar_fill.set_width(fraction)
        if fraction > 0.5:
            color = "#44cc44"   # green — plenty of steps left
        elif fraction > 0.25:
            color = "#ffcc00"   # yellow — getting low
        else:
            color = "#cc2222"   # red — close to LOSE
        self._bar_fill.set_color(color)

    def update_step_budget(self, steps_remaining: int, max_steps: int) -> None:
        """Refresh the steps-remaining progress bar shown above the game."""
        self._steps_remaining = steps_remaining
        self._max_steps = max_steps
        if self.fig is not None and plt.fignum_exists(self.fig.number):
            self._update_bar_visual()
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

    def set_status(self, text: str) -> None:
        if self.fig is not None and plt.fignum_exists(self.fig.number):
            if self._status_text is not None:
                self._status_text.set_text(text)
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

    def _on_key(self, event) -> None:
        self._key_queue.put(event.key)

    def _on_click(self, event) -> None:
        if event.button != 1 or event.xdata is None or event.ydata is None:
            return
        # Ignore clicks on the status text or progress bar strips
        if self._ax_game is not None and event.inaxes is not self._ax_game:
            return
        # Map the clicked image pixel back into camera display space [0, 63].
        # This works whether frame_to_rgb_array outputs 64x64 or pre-scaled frames.
        h = 64
        w = 64
        if self._im is not None:
            arr = self._im.get_array()
            if hasattr(arr, "shape") and len(arr.shape) >= 2:
                h = int(arr.shape[0]) or 64
                w = int(arr.shape[1]) or 64
        click_x = int(float(event.xdata) * 64.0 / float(w))
        click_y = int(float(event.ydata) * 64.0 / float(h))
        click_x = max(0, min(63, click_x))
        click_y = max(0, min(63, click_y))
        self._key_queue.put(_MouseClick(x=click_x, y=click_y))


@dataclass(frozen=True)
class _MouseClick:
    x: int
    y: int


# Matplotlib key name → GameAction
_KEY_MAP: dict[str, GameAction] = {
    "w":     GameAction.ACTION1,
    "up":    GameAction.ACTION1,
    "s":     GameAction.ACTION2,
    "down":  GameAction.ACTION2,
    "a":     GameAction.ACTION3,
    "left":  GameAction.ACTION3,
    "d":     GameAction.ACTION4,
    "right": GameAction.ACTION4,
    "e":     GameAction.ACTION5,
    " ":     GameAction.ACTION5,
    "f":     GameAction.ACTION6,
    "z":     GameAction.ACTION7,
    "u":     GameAction.ACTION7,
}

# ---------------------------------------------------------------------------
# Validation database
# ---------------------------------------------------------------------------

_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "games_db.json")

def _load_db() -> dict:
    if os.path.exists(_DB_PATH):
        with open(_DB_PATH) as f:
            return json.load(f)
    return {}

def _save_db(db: dict) -> None:
    with open(_DB_PATH, "w") as f:
        json.dump(db, f, indent=2, sort_keys=True)

def _validate_game(game_name: str) -> None:
    db = _load_db()
    db[game_name] = True
    _save_db(db)
    print(f"\033[32m✓\033[0m  '{game_name}' marked as validated.")


#: Games `resolve_env` serves from code rather than a games/<id>/<id>.py folder, so a
#: directory scan can never see them (`synthetic_geodesic` lives in utils/synthetic.py).
_BUILTIN_GAME_NAMES = ("synthetic_geodesic",)


def _local_game_names() -> list[str]:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.join(here, "games")
    out: list[str] = list(_BUILTIN_GAME_NAMES)
    if not os.path.isdir(root):
        return sorted(out)
    for name in sorted(os.listdir(root)):
        # Skip PuzzleScript adapter games — they appear in their own section
        if name.startswith("ps:"):
            continue
        d = os.path.join(root, name)
        if os.path.isdir(d) and os.path.isfile(os.path.join(d, f"{name}.py")):
            out.append(name)
    return sorted(out)


def _puzzlescript_game_names() -> list[str]:
    """Return sorted list of integrated PuzzleScript game names (ps:xxx folders)."""
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.join(here, "games")
    if not os.path.isdir(root):
        return []
    out: list[str] = []
    for name in sorted(os.listdir(root)):
        if not name.startswith("ps:") or name == "ps:":
            continue
        d = os.path.join(root, name)
        if os.path.isdir(d):
            out.append(name)
    return out


def _print_available_games() -> None:
    db = _load_db()
    total_games = 0
    validated_games = 0

    print("Public (ARC-AGI-3 API)")
    print("-" * 44)
    api = arc_agi.Arcade(operation_mode=OperationMode.ONLINE)
    public = api.get_environments()
    if not public:
        print("  (none)")
    else:
        for info in public:
            # Public API games are listed for reference only — they are not
            # counted in the validated/total stats below.
            title = getattr(info, "title", None)
            suffix = f"  —  {title}" if title else ""
            print(f"  {info.game_id}{suffix}")

    print()
    print("Local (games/ + built-in envs)")
    print("-" * 44)
    local = _local_game_names()
    if not local:
        print("  (none)")
    else:
        for name in local:
            total_games += 1
            if db.get(name):
                marker = "\033[32m✓\033[0m"
                validated_games += 1
            else:
                marker = "\033[31m✗\033[0m"
            print(f"  {marker}  {name}")

    print()
    print("Minigrid full-obs (via adapter)")
    print("-" * 44)
    for env_id in MinigridAdapter.list_available_envs():
        total_games += 1
        if db.get(env_id):
            marker = "\033[32m✓\033[0m"
            validated_games += 1
        else:
            marker = "\033[31m✗\033[0m"
        print(f"  {marker}  {env_id}")

    print()
    print("Minigrid partial-obs (via adapter)  —  prefix env_id with 'partial:'")
    print("-" * 44)
    for env_id in MinigridAdapter.list_available_partial_envs():
        total_games += 1
        partial_key = f"partial:{env_id}"
        if db.get(partial_key):
            marker = "\033[32m✓\033[0m"
            validated_games += 1
        else:
            marker = "\033[31m✗\033[0m"
        print(f"  {marker}  partial:{env_id}")

    print()
    print("ProcGen (via adapter)  —  use 'procgen_<name>'")
    print("-" * 44)
    for game_name in ProcGenAdapter.list_available_games():
        total_games += 1
        client_name = f"procgen_{game_name}"
        if db.get(client_name):
            marker = "\033[32m✓\033[0m"
            validated_games += 1
        else:
            marker = "\033[31m✗\033[0m"
        print(f"  {marker}  {client_name}")

    print()
    print("Gym-Gridworlds (via adapter)  —  prefix env_id with 'gymgw:'")
    print("-" * 44)
    for env_id in GymGridworldsAdapter.list_available_envs():
        total_games += 1
        client_name = f"gymgw:{env_id}"
        if db.get(client_name):
            marker = "\033[32m✓\033[0m"
            validated_games += 1
        else:
            marker = "\033[31m✗\033[0m"
        print(f"  {marker}  {client_name}")

    print()
    print("PuzzleScript (via adapter)  —  prefix game name with 'ps:'")
    print("-" * 44)
    for client_name in _puzzlescript_game_names():
        total_games += 1
        if db.get(client_name):
            marker = "\033[32m✓\033[0m"
            validated_games += 1
        else:
            marker = "\033[31m✗\033[0m"
        print(f"  {marker}  {client_name}")

    print()
    print("=" * 44)
    pct = (validated_games / total_games * 100) if total_games > 0 else 0.0
    print(f"Total games: {total_games}")
    print(f"Validated:   {validated_games}/{total_games} ({pct:.1f}%)")


def _refresh_step_budget(env, renderer: _PersistentRenderer) -> None:
    """Read the game's step counter and push an update to the renderer's progress bar."""
    game = _get_underlying_game(env)
    if game is None:
        return
    sc = getattr(game, "_step_counter", None)
    if sc is None:
        return
    steps_remaining = getattr(sc, "steps_remaining", None)
    max_steps = getattr(sc, "max_steps", None)
    if steps_remaining is not None and max_steps is not None:
        renderer.update_step_budget(steps_remaining, max_steps)


# ---------------------------------------------------------------------------
# Troubleshooting mode — headless frame capture
# ---------------------------------------------------------------------------

# Action characters for troubleshoot mode
_TROUBLESHOOT_ACTION_MAP: dict[str, GameAction] = {
    "u": GameAction.ACTION1,  # up
    "d": GameAction.ACTION2,  # down
    "l": GameAction.ACTION3,  # left
    "r": GameAction.ACTION4,  # right
    "a": GameAction.ACTION5,  # action
}

_ACTION_NAMES: dict[GameAction, str] = {
    GameAction.ACTION1: "UP",
    GameAction.ACTION2: "DOWN",
    GameAction.ACTION3: "LEFT",
    GameAction.ACTION4: "RIGHT",
    GameAction.ACTION5: "ACTION",
}


def _frame_to_png(frame: "np.ndarray", path: str, scale: int = 4) -> None:
    """Save a 64×64 ARC palette frame as an RGB PNG image."""
    from arc_agi.rendering import COLOR_MAP
    import numpy as np
    from PIL import Image

    height, width = frame.shape
    rgb = np.zeros((height * scale, width * scale, 3), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            val = int(frame[y, x])
            hex_color = COLOR_MAP.get(val, "#000000FF")
            r_val = int(hex_color[1:3], 16)
            g_val = int(hex_color[3:5], 16)
            b_val = int(hex_color[5:7], 16)
            rgb[y * scale:(y + 1) * scale, x * scale:(x + 1) * scale] = (r_val, g_val, b_val)

    img = Image.fromarray(rgb)
    img.save(path)


def _run_troubleshoot(args) -> None:
    """Run a game headlessly, saving each rendered frame as a PNG for inspection.

    Output goes to debug_frames/<game_name>/ with files:
      - frame_0000_reset.png          (initial state)
      - frame_0001_UP.png             (after action UP)
      - frame_0002_DOWN.png           ...
      - summary.txt                   (text log of all steps)
    """
    import numpy as np

    game_name = args.game
    if game_name is None:
        print("[troubleshoot] Error: must specify a game name.")
        return

    # Determine action sequence
    if args.actions:
        actions = []
        for ch in args.actions.lower():
            act = _TROUBLESHOOT_ACTION_MAP.get(ch)
            if act is None:
                print(f"[troubleshoot] Warning: unknown action char '{ch}', skipping.")
                continue
            actions.append(act)
    else:
        # Random actions
        choices = [GameAction.ACTION1, GameAction.ACTION2,
                   GameAction.ACTION3, GameAction.ACTION4, GameAction.ACTION5]
        actions = [random.choice(choices) for _ in range(args.num_steps)]

    # Create output directory
    safe_name = game_name.replace(":", "_").replace("/", "_")
    out_dir = pathlib.Path("debug_frames") / safe_name / f"level_{args.level}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load the game headlessly (we access frames directly). The shared cascade resolves
    # every backend; NOOP_RENDERER satisfies the wrappers' renderer contract.
    try:
        env = resolve_env(game_name, NOOP_RENDERER, level=args.level, seed=args.seed)
    except Exception as e:  # noqa: BLE001
        print(f"[troubleshoot] Error: could not load game '{game_name}': {e}")
        return
    if env is None:
        print(f"[troubleshoot] Error: could not load game '{game_name}'.")
        return

    print(f"[troubleshoot] Game: {game_name}, level: {args.level}")
    print(f"[troubleshoot] Actions: {len(actions)} steps")
    print(f"[troubleshoot] Output: {out_dir}/")
    print()

    summary_lines: list[str] = []
    summary_lines.append(f"Game: {game_name}")
    summary_lines.append(f"Level: {args.level}")
    summary_lines.append(f"Actions: {len(actions)} steps")
    summary_lines.append("")

    # Reset and capture initial frame
    obs = env.reset()
    frame_idx = 0

    if obs and obs.frame:
        frame = obs.frame[0] if isinstance(obs.frame, list) else obs.frame
        png_path = out_dir / f"frame_{frame_idx:04d}_RESET.png"
        _frame_to_png(frame, str(png_path))
        summary_lines.append(f"[{frame_idx:04d}] RESET  state={obs.state}")
        print(f"  [{frame_idx:04d}] RESET  → state={obs.state}")

    # Execute each action
    for action in actions:
        frame_idx += 1
        obs = env.step(action)

        if obs is None:
            summary_lines.append(f"[{frame_idx:04d}] {_ACTION_NAMES.get(action, '?')}  → None (no response)")
            print(f"  [{frame_idx:04d}] {_ACTION_NAMES.get(action, '?')}  → None")
            continue

        action_name = _ACTION_NAMES.get(action, "UNKNOWN")
        state_str = str(obs.state) if obs.state else "?"
        levels_done = getattr(obs, "levels_completed", 0)

        if obs.frame:
            frame = obs.frame[0] if isinstance(obs.frame, list) else obs.frame
            png_path = out_dir / f"frame_{frame_idx:04d}_{action_name}.png"
            _frame_to_png(frame, str(png_path))

        line = f"[{frame_idx:04d}] {action_name}  → state={state_str}  levels_completed={levels_done}"
        summary_lines.append(line)
        print(f"  {line}")

        # Stop early on terminal states
        if obs.state == GameState.WIN:
            print(f"\n[troubleshoot] WIN at step {frame_idx}!")
            summary_lines.append(f"\nWIN at step {frame_idx}")
            break
        elif obs.state == GameState.GAME_OVER:
            print(f"\n[troubleshoot] GAME_OVER at step {frame_idx}.")
            summary_lines.append(f"\nGAME_OVER at step {frame_idx}")
            break

    # Write summary
    summary_path = out_dir / "summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n")

    if hasattr(env, "close"):
        env.close()

    print(f"\n[troubleshoot] Done. {frame_idx + 1} frames saved to {out_dir}/")
    print(f"[troubleshoot] Summary: {summary_path}")


# ---------------------------------------------------------------------------
# Main game loop
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive ARC-AGI-3 game player.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "game",
        nargs="?",
        default=None,
        help="Game to play (e.g. 'maze', 'ls20'); omit to list games.",
    )
    parser.add_argument(
        "--validate",
        metavar="GAME",
        default=None,
        help="Mark a local game as manually validated.",
    )
    parser.add_argument(
        "--level", "-l",
        type=int,
        default=0,
        metavar="N",
        help="Level index to start at (0-indexed, default: 0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="N",
        help="Deterministic seed for env generation (and troubleshoot random actions). "
             "Omit for a random seed.",
    )
    parser.add_argument(
        "--troubleshoot",
        action="store_true",
        default=False,
        help="Run in headless troubleshooting mode: save each frame as PNG to debug_frames/.",
    )
    parser.add_argument(
        "--actions",
        type=str,
        default=None,
        metavar="SEQ",
        help="Action sequence for troubleshoot mode (u=up, d=down, l=left, r=right, a=action). "
             "E.g. 'uulddrr'. If omitted, random actions are used.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=20,
        metavar="N",
        help="Number of random steps in troubleshoot mode (default: 20, ignored if --actions given).",
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    if args.validate is not None:
        _validate_game(args.validate)
        return

    if args.game is None:
        _print_available_games()
        return

    if args.troubleshoot:
        _run_troubleshoot(args)
        return

    arc = arc_agi.Arcade()
    key_queue: queue.Queue[str | _MouseClick] = queue.Queue()
    renderer = _PersistentRenderer(key_queue)

    # Shared cascade: procgen_ -> gymgw: -> ps: -> [partial:]MiniGrid -> native
    # games/<id>/<id>.py -> arcade remote (arc.make). Same resolver solver.py uses.
    env = resolve_env(args.game, renderer, level=args.level, seed=args.seed, arcade=arc)

    level_suffix = f"  level {args.level}" if args.level != 0 else ""
    print(f"\n[client] Playing: {args.game}{level_suffix}  (focus the game window and use keys)")

    env.reset()
    _jump_to_level(env, args.level, renderer)
    _refresh_step_budget(env, renderer)

    total_actions = 0
    level_actions = 0
    levels_done = args.level

    def _status(extra: str = "") -> str:
        return (
            f"Level {levels_done + 1}  |  moves this level: {level_actions}"
            f"  |  total moves: {total_actions}"
            + (f"  |  {extra}" if extra else "")
        )

    def _step_mouse(x: int, y: int):
        mouse_action = ActionInput(id=GameAction.ACTION6, data={"x": x, "y": y})

        if hasattr(env, "step_mouse"):
            return env.step_mouse(x, y)

        if hasattr(env, "perform_action"):
            try:
                return env.perform_action(mouse_action, raw=True)
            except TypeError:
                return env.perform_action(mouse_action)

        if hasattr(env, "step"):
            # arc_agi wrappers expose step(action, data=...), not step(ActionInput).
            try:
                return env.step(GameAction.ACTION6, data={"x": x, "y": y})
            except TypeError:
                try:
                    return env.step(GameAction.ACTION6, {"x": x, "y": y})
                except TypeError:
                    return None

        return None

    renderer.set_status(_status())

    running = True
    terminal_state: GameState | None = None
    while running:
        # Process GUI events and wait briefly for a keypress
        plt.pause(0.05)
        try:
            event = key_queue.get_nowait()
        except queue.Empty:
            continue

        if isinstance(event, _MouseClick):
            if terminal_state is not None:
                continue
            obs = _step_mouse(event.x, event.y)
            if obs is None:
                continue
            total_actions += 1
            level_actions += 1
            if obs.levels_completed > levels_done:
                print(f"[client] Level {levels_done + 1} done in {level_actions} moves.")
                levels_done = obs.levels_completed
                level_actions = 0
            _refresh_step_budget(env, renderer)
            if obs.state == GameState.WIN:
                renderer.set_status(_status("YOU WIN!"))
                if terminal_state != GameState.WIN:
                    print(f"\n[client] You win!  Levels: {levels_done}  |  Total moves: {total_actions}")
                    print("[client] Press q to quit or r to replay.")
                terminal_state = GameState.WIN
            elif obs.state == GameState.GAME_OVER:
                renderer.set_status(_status("GAME OVER"))
                if terminal_state != GameState.GAME_OVER:
                    print("\n[client] Game over — press r to reset or q to quit.")
                terminal_state = GameState.GAME_OVER
            else:
                renderer.set_status(_status())
            continue

        key = event
        if key in ("q", "escape"):
            print("\n[client] Quit.")
            break

        if key == "r":
            env.reset()
            _jump_to_level(env, args.level, renderer)
            levels_done = args.level
            level_actions = 0
            terminal_state = None
            _refresh_step_budget(env, renderer)
            renderer.set_status(_status())
            continue

        if terminal_state is not None:
            continue

        action = _KEY_MAP.get(key)
        if action is None:
            continue

        obs = env.step(action)
        if obs is None:
            continue

        total_actions += 1
        level_actions += 1

        if obs.levels_completed > levels_done:
            # Completed a level
            print(f"[client] Level {levels_done + 1} done in {level_actions} moves.")
            levels_done = obs.levels_completed
            level_actions = 0

        _refresh_step_budget(env, renderer)
        if obs.state == GameState.WIN:
            renderer.set_status(_status("YOU WIN!"))
            if terminal_state != GameState.WIN:
                print(f"\n[client] You win!  Levels: {levels_done}  |  Total moves: {total_actions}")
                print("[client] Press q to quit or r to replay.")
            terminal_state = GameState.WIN
        elif obs.state == GameState.GAME_OVER:
            renderer.set_status(_status("GAME OVER"))
            if terminal_state != GameState.GAME_OVER:
                print("\n[client] Game over — press r to reset or q to quit.")
            terminal_state = GameState.GAME_OVER
        else:
            renderer.set_status(_status())

    if hasattr(env, "close"):
        env.close()
    plt.close("all")

    scorecard = arc.get_scorecard()
    if scorecard:
        print(
            f"\n--- Final scorecard ---"
            f"\n  Levels completed : {scorecard.total_levels_completed}"
            f"\n  Environments won : {scorecard.total_environments_completed}"
            f"\n  Total moves      : {scorecard.total_actions}"
            f"\n  Score            : {scorecard.score:.1f}"
        )
        print(scorecard)


if __name__ == "__main__":
    main()
