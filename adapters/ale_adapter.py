"""ale_adapter.py — Wraps Atari Learning Environment games as ARCBaseGame-compatible objects.

Exposes the same duck-type interface as ARCBaseGame so the solver codebase
can treat ALE games identically to native arcengine games.

Frame pipeline:
  ALE obs (210×160×3 RGB)
  → resize to 64×64 RGB  (PIL.Image.LANCZOS)
  → quantize to 16-color ARC palette  (nearest-neighbour in RGB space)
  → 64×64 uint8 ndarray of palette indices

Action mapping:
  GameAction.ACTION1–7 are mapped to ALE action names (UP, DOWN, LEFT, RIGHT,
  FIRE, etc.) via get_action_meanings(). Unmapped actions fall back to NOOP.
  Per-game overrides are provided for the 5 Phase 1 games; any other ALE game
  uses the DEFAULT_ACTION_MAP.

WIN / GAME_OVER semantics:
  terminated=True  →  GameState.WIN    (natural episode end — the solver
                                        learns from the full episode structure)
  truncated=True   →  GameState.GAME_OVER  (timeout / max-frames exceeded)

Level concept:
  ALE has no multi-level structure. set_level(idx) seeds the environment with
  a deterministic seed derived from (base_seed + idx), giving different episode
  starts per "level". This allows the training pipeline to request varied
  starting conditions without changing game identity.

Usage:
    from adapters import ALEAdapter
    from arcengine import GameAction, ActionInput, GameState

    game = ALEAdapter("Freeway", seed=42)
    game.set_level(0)
    result = game.perform_action(ActionInput(id=GameAction.ACTION1), raw=True)
    print(result.state)          # GameState.NOT_FINISHED
    print(result.frame[0].shape) # (64, 64)
"""

from __future__ import annotations

import numpy as np
from PIL import Image

import gymnasium as gym
import ale_py

from arcengine import ActionInput, GameAction, GameState, FrameDataRaw

from adapters.base import BaseAdapter

# ---------------------------------------------------------------------------
# Register ALE environments with gymnasium (idempotent)
# ---------------------------------------------------------------------------
gym.register_envs(ale_py)

# ---------------------------------------------------------------------------
# ARC 16-color palette  (index → (R, G, B))
# Mirrors solver_viewer.ARC_PALETTE exactly.
# ---------------------------------------------------------------------------
_ARC_PALETTE_RGB: np.ndarray = np.array([
    (255, 255, 255),  # 0  white
    ( 30, 147, 255),  # 1  blue
    (232,  16,  16),  # 2  red
    ( 79, 204,  48),  # 3  green
    ( 51,  51,  51),  # 4  dark gray
    (  0,   0,   0),  # 5  black
    (227, 110, 246),  # 6  pink
    (230, 230, 230),  # 7  light gray
    (249,  60,  49),  # 8  red (cursor)
    ( 30, 147, 255),  # 9  blue (player)
    (170, 170, 170),  # 10 mid gray
    (255, 220,   0),  # 11 yellow
    (255, 133,  27),  # 12 orange
    (160,  96,   0),  # 13 brown
    ( 79, 204,  48),  # 14 green (target)
    ( 96,  48, 160),  # 15 purple
], dtype=np.float32)  # float for distance computation

# ---------------------------------------------------------------------------
# Default GameAction → ALE action-name mapping
# Used for any game whose action meanings include these names.
# ---------------------------------------------------------------------------
_DEFAULT_ACTION_MAP: dict[GameAction, str] = {
    GameAction.ACTION1: "UP",
    GameAction.ACTION2: "DOWN",
    GameAction.ACTION3: "LEFT",
    GameAction.ACTION4: "RIGHT",
    GameAction.ACTION5: "FIRE",
    GameAction.ACTION6: "UPFIRE",
    GameAction.ACTION7: "NOOP",   # undo has no ALE equivalent
}

# Per-game overrides where the default mapping doesn't fit well.
# Keys are the game name strings passed to gym.make ("Freeway", etc.).
_GAME_ACTION_OVERRIDES: dict[str, dict[GameAction, str]] = {
    "Freeway": {
        # Freeway only has NOOP / UP / DOWN — chicken crosses road vertically
        GameAction.ACTION1: "UP",
        GameAction.ACTION2: "DOWN",
        GameAction.ACTION3: "NOOP",
        GameAction.ACTION4: "NOOP",
        GameAction.ACTION5: "NOOP",
        GameAction.ACTION6: "NOOP",
        GameAction.ACTION7: "NOOP",
    },
    "SpaceInvaders": {
        # SpaceInvaders: NOOP / FIRE / RIGHT / LEFT / RIGHTFIRE / LEFTFIRE
        GameAction.ACTION1: "NOOP",
        GameAction.ACTION2: "NOOP",
        GameAction.ACTION3: "LEFT",
        GameAction.ACTION4: "RIGHT",
        GameAction.ACTION5: "FIRE",
        GameAction.ACTION6: "RIGHTFIRE",
        GameAction.ACTION7: "NOOP",
    },
}

# Hardcoded from ALE upstream registration source:
# https://github.com/Farama-Foundation/Arcade-Learning-Environment
# (roms/md5.json + registration.py single-player exclusions)
_AVAILABLE_ALE_GAMES: tuple[str, ...] = (
    "Adventure",
    "AirRaid",
    "Alien",
    "Amidar",
    "Assault",
    "Asterix",
    "Asteroids",
    "Atlantis",
    "Atlantis2",
    "Backgammon",
    "BankHeist",
    "BasicMath",
    "BattleZone",
    "BeamRider",
    "Berzerk",
    "Blackjack",
    "Bowling",
    "Boxing",
    "Breakout",
    "Carnival",
    "Casino",
    "Centipede",
    "ChopperCommand",
    "CrazyClimber",
    "Crossbow",
    "Darkchambers",
    "Defender",
    "DemonAttack",
    "DonkeyKong",
    "DoubleDunk",
    "Earthworld",
    "ElevatorAction",
    "Enduro",
    "Entombed",
    "Et",
    "FishingDerby",
    "FlagCapture",
    "Freeway",
    "Frogger",
    "Frostbite",
    "Galaxian",
    "Gopher",
    "Gravitar",
    "Hangman",
    "HauntedHouse",
    "Hero",
    "HumanCannonball",
    "IceHockey",
    "Jamesbond",
    "JourneyEscape",
    "Kaboom",
    "Kangaroo",
    "KeystoneKapers",
    "KingKong",
    "Klax",
    "Koolaid",
    "Krull",
    "KungFuMaster",
    "LaserGates",
    "LostLuggage",
    "MarioBros",
    "MiniatureGolf",
    "MontezumaRevenge",
    "MrDo",
    "MsPacman",
    "NameThisGame",
    "Othello",
    "Pacman",
    "Phoenix",
    "Pitfall",
    "Pitfall2",
    "Pong",
    "Pooyan",
    "PrivateEye",
    "Qbert",
    "Riverraid",
    "RoadRunner",
    "Robotank",
    "Seaquest",
    "SirLancelot",
    "Skiing",
    "Solaris",
    "SpaceInvaders",
    "SpaceWar",
    "StarGunner",
    "Superman",
    "Surround",
    "Tennis",
    "Tetris",
    "TicTacToe3D",
    "TimePilot",
    "Trondead",
    "Turmoil",
    "Tutankham",
    "UpNDown",
    "Venture",
    "VideoCheckers",
    "VideoChess",
    "VideoCube",
    "VideoPinball",
    "WizardOfWor",
    "WordZapper",
    "YarsRevenge",
    "Zaxxon",
)


# ---------------------------------------------------------------------------
# Frame utilities
# ---------------------------------------------------------------------------

def _resize_rgb(obs: np.ndarray, size: int = 64) -> np.ndarray:
    """Resize a (H, W, 3) uint8 RGB array to (size, size, 3)."""
    img = Image.fromarray(obs, mode="RGB")
    img = img.resize((size, size), Image.LANCZOS)
    return np.asarray(img, dtype=np.uint8)


def _quantize_to_arc_palette(rgb_frame: np.ndarray) -> np.ndarray:
    """Convert a (64, 64, 3) uint8 RGB frame to a (64, 64) uint8 palette-index frame.

    For each pixel, finds the nearest ARC palette color by squared Euclidean
    distance in RGB space.
    """
    # (64*64, 3) float
    pixels = rgb_frame.reshape(-1, 3).astype(np.float32)
    # (64*64, 16) — squared distances to each palette entry
    diff = pixels[:, None, :] - _ARC_PALETTE_RGB[None, :, :]   # (N, 16, 3)
    sq_dist = (diff ** 2).sum(axis=2)                            # (N, 16)
    indices = sq_dist.argmin(axis=1).astype(np.uint8)           # (N,)
    return indices.reshape(64, 64)


def _process_obs(obs: np.ndarray) -> np.ndarray:
    """Full pipeline: ALE RGB obs → 64×64 ARC palette index frame."""
    resized = _resize_rgb(obs, size=64)
    return _quantize_to_arc_palette(resized)


# ---------------------------------------------------------------------------
# ALEAdapter
# ---------------------------------------------------------------------------

class ALEAdapter(BaseAdapter):
    """Wraps an ALE/Atari game as an ARCBaseGame-compatible object.

    Args:
        game_name: ALE game name, e.g. "Freeway", "MontezumaRevenge".
                   Used to construct "ALE/{game_name}-v5".
        seed:      Base random seed. set_level(idx) uses seed + idx.
        frameskip: Number of ALE frames to skip per action (default 4).
    """

    def __init__(
        self,
        game_name: str,
        seed: int = 0,
        frameskip: int = 4,
    ) -> None:
        self._game_name = game_name
        self._game_id = f"ale_{game_name.lower()}"
        self._seed = seed
        self._frameskip = frameskip

        # Create gymnasium ALE environment
        self._env = gym.make(
            f"ALE/{game_name}-v5",
            obs_type="rgb",
            frameskip=frameskip,
            repeat_action_probability=0.0,   # deterministic
            full_action_space=False,          # minimal action set per game
        )

        # Build action mapping: GameAction → env action integer
        self._action_index_map: dict[GameAction, int] = self._build_action_map()

        # Shared adapter scaffolding (state, step counter, undo stack) + 1st frame.
        # The step counter is cosmetic — ALE episodes end via terminated/truncated,
        # not the HUD counter.
        self._init_base(self._game_id,
                        max_steps=self._env.spec.max_episode_steps or 100_000)
        self._reset_env(self._seed)

    # ------------------------------------------------------------------
    # Action mapping
    # ------------------------------------------------------------------

    def _build_action_map(self) -> dict[GameAction, int]:
        """Map GameAction → ALE action index using the env's action meanings."""
        meanings: list[str] = self._env.unwrapped.get_action_meanings()
        # index in meanings list == integer to pass to env.step()
        name_to_idx = {name: idx for idx, name in enumerate(meanings)}
        noop_idx = name_to_idx.get("NOOP", 0)

        name_map = _GAME_ACTION_OVERRIDES.get(self._game_name, _DEFAULT_ACTION_MAP)

        result: dict[GameAction, int] = {}
        for action, ale_name in name_map.items():
            result[action] = name_to_idx.get(ale_name, noop_idx)
        return result

    # ------------------------------------------------------------------
    # Internal reset
    # ------------------------------------------------------------------

    def _reset_env(self, seed: int) -> None:
        obs, _info = self._env.reset(seed=seed)
        self._current_frame = _process_obs(obs)
        self._state = GameState.NOT_FINISHED
        self._action_count = 0

    # ------------------------------------------------------------------
    # BaseAdapter hooks (game_id/set_level/perform_action/_make_frame_data +
    # RESET + ACTION7-undo + terminal short-circuit come from BaseAdapter).
    # ------------------------------------------------------------------

    def _apply(self, action_input: ActionInput) -> None:
        """Step the ALE env once.

        terminated → WIN (natural episode end); truncated → GAME_OVER (timeout)."""
        ale_action = self._action_index_map.get(action_input.id, 0)
        obs, _reward, terminated, truncated, _info = self._env.step(ale_action)

        self._current_frame = _process_obs(obs)
        self._action_count += 1

        if terminated:
            self._state = GameState.WIN
        elif truncated:
            self._state = GameState.GAME_OVER
        else:
            self._state = GameState.NOT_FINISHED

    # ── snapshot / restore for generic UNDO ──────────────────────────────────
    def _snapshot(self):
        """Snapshot the full ALE machine state via cloneState, plus adapter
        bookkeeping. cloneState captures RAM + registers, so restore is exact."""
        return (self._env.unwrapped.ale.cloneState(),
                self._action_count, self._step_counter.steps_remaining,
                self._state, self._current_frame)

    def _restore(self, snap) -> None:
        (ale_state, self._action_count, self._step_counter.steps_remaining,
         self._state, self._current_frame) = snap
        self._env.unwrapped.ale.restoreState(ale_state)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @staticmethod
    def list_available_games() -> list[str]:
        """Return hardcoded ALE game names supported by this adapter."""
        return list(_AVAILABLE_ALE_GAMES)

    def get_action_meanings(self) -> list[str]:
        """Return ALE action meanings for this game (diagnostic use)."""
        return self._env.unwrapped.get_action_meanings()

    def close(self) -> None:
        """Release the underlying gymnasium environment."""
        self._env.close()

    def __repr__(self) -> str:
        return (
            f"ALEAdapter(game={self._game_name!r}, "
            f"level={self._current_level_index}, state={self._state})"
        )
