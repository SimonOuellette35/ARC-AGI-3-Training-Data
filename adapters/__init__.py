"""adapters/ — Wrappers that expose external game libraries via the ARCBaseGame interface.

Each adapter implements the same duck-type interface as ARCBaseGame:
  - perform_action(ActionInput, raw=False) → FrameDataRaw
  - _state: GameState
  - set_level(idx: int)
  - full_reset() / level_reset()
  - _current_level_index: int
  - _seed: int

The solver codebase uses these attributes and methods directly; no isinstance
checks against ARCBaseGame are required.
"""

from .ale_adapter import ALEAdapter
from .gymgridworlds_adapter import GymGridworldsAdapter
from .minigrid_adapter import MinigridAdapter
from .procgen_adapter import ProcGenAdapter
from .puzzlescript_adapter import PuzzleScriptAdapter

__all__ = ["ALEAdapter", "GymGridworldsAdapter", "MinigridAdapter", "ProcGenAdapter", "PuzzleScriptAdapter"]
