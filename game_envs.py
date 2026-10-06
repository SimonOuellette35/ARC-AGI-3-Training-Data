"""game_envs.py -- Shared game-environment resolution for the ARC-AGI-3 stack.

ONE source of truth for turning a game-string (the same identifiers the interactive
client and the planner both accept) into a runnable environment wrapper. Both
solver_client.py (interactive player) and solver.py (model-based planner) import
from here, so the resolution cascade and the env wrappers live in a single place
instead of being copy-pasted across the two entry points.

Resolution cascade (first match wins) -- see resolve_env():
    procgen_<name>        -> ProcGenAdapter        (via _ProcGenEnvironmentWrapper)
    gymgw:<env_id>        -> GymGridworldsAdapter   (via _GymGridworldsEnvironmentWrapper)
    ps:<game_name>        -> PuzzleScriptAdapter    (via _PuzzleScriptEnvironmentWrapper)
    [partial:]MiniGrid-*  -> MinigridAdapter        (via _MinigridEnvironmentWrapper)
    games/<id>/<id>.py    -> native ARCBaseGame     (via arc_agi LocalEnvironmentWrapper)
    <anything else>       -> arcade remote          (Arcade.make)

Every wrapper exposes the same duck-typed loop API:
    reset()                 -> FrameDataRaw
    step(action: GameAction) -> FrameDataRaw
    step_mouse(x, y)        -> FrameDataRaw   (adapter wrappers only)
    close()                                   (adapter wrappers only)

The renderer is any callable (steps:int, frame_data:FrameDataRaw) -> None. Pass
NOOP_RENDERER for headless use (the planner); solver_client passes its live GUI
renderer. The native (arc_agi) path additionally needs an Arcade instance; callers
may pass one or let resolve_env lazily build a shared singleton.
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import logging
import os
import random
import re
import sys
from typing import Optional

import arc_agi
from arcengine import ActionInput, FrameDataRaw, GameAction, GameState
from adapters import GymGridworldsAdapter, MinigridAdapter, ProcGenAdapter, PuzzleScriptAdapter
from utils import arc_game

_HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Headless renderer
# ---------------------------------------------------------------------------

class _NoOpRenderer:
    """Renderer that draws nothing -- used by headless callers (the planner). The
    adapter wrappers require *some* renderer callable; this satisfies that contract.
    update_step_budget() is a no-op so the wrappers' progress-bar refreshes are safe."""

    def __call__(self, steps: int, frame_data: FrameDataRaw) -> None:
        pass

    def update_step_budget(self, *args, **kwargs) -> None:
        pass


NOOP_RENDERER = _NoOpRenderer()


# ---------------------------------------------------------------------------
# Minigrid environment wrapper
# ---------------------------------------------------------------------------

class _MinigridEnvironmentWrapper:
    """Small runtime wrapper so MinigridAdapter fits the loop API."""

    def __init__(
        self,
        env_id: str,
        renderer,
        seed: Optional[int] = None,
        level: int = 0,
        partial_obs: bool = False,
    ) -> None:
        self._renderer = renderer
        self._steps = 0
        self._adapter = MinigridAdapter(
            env_id=env_id,
            seed=seed if seed is not None else random.randint(0, 2**31 - 1),
            partial_obs=partial_obs,
        )
        self._adapter.set_level(level)

    def reset(self) -> FrameDataRaw:
        self._adapter.full_reset()
        self._steps = 0
        obs = self._adapter.perform_action(ActionInput(id=GameAction.RESET), raw=True)
        self._renderer(self._steps, obs)
        return obs

    def step(self, action: GameAction) -> FrameDataRaw:
        self._steps += 1
        obs = self._adapter.perform_action(ActionInput(id=action), raw=True)
        self._renderer(self._steps, obs)
        return obs

    def step_mouse(self, x: int, y: int) -> FrameDataRaw:
        self._steps += 1
        obs = self._adapter.perform_action(
            ActionInput(id=GameAction.ACTION6, data={"x": x, "y": y}),
            raw=True,
        )
        self._renderer(self._steps, obs)
        return obs

    def close(self) -> None:
        self._adapter.close()


def _load_minigrid_env(env_id: str, renderer, level: int = 0, seed: Optional[int] = None):
    # Support "partial:<env_id>" syntax to enable the 7x7 egocentric (partial-obs) mode.
    partial = env_id.startswith("partial:")
    bare_env_id = env_id[len("partial:"):] if partial else env_id
    if bare_env_id not in MinigridAdapter.list_available_envs():
        return None
    return _MinigridEnvironmentWrapper(
        env_id=bare_env_id, renderer=renderer, level=level, partial_obs=partial, seed=seed
    )


# ---------------------------------------------------------------------------
# ProcGen environment wrapper
# ---------------------------------------------------------------------------

class _ProcGenEnvironmentWrapper:
    """Small runtime wrapper so ProcGenAdapter fits the loop API."""

    def __init__(
        self,
        game_name: str,
        renderer,
        seed: Optional[int] = None,
        level: int = 0,
    ) -> None:
        self._renderer = renderer
        self._steps = 0
        self._adapter = ProcGenAdapter(
            game_name=game_name,
            seed=seed if seed is not None else random.randint(0, 2**31 - 1),
        )
        self._adapter.set_level(level)

    def reset(self) -> FrameDataRaw:
        # Use level_reset() to stay at the current level rather than full_reset()
        # which would reinitialize to level 0 (wrong distribution_mode for heist/leaper).
        self._adapter.level_reset()
        self._steps = 0
        obs = self._adapter.perform_action(ActionInput(id=GameAction.RESET), raw=True)
        self._renderer(self._steps, obs)
        return obs

    def step(self, action: GameAction) -> FrameDataRaw:
        self._steps += 1
        obs = self._adapter.perform_action(ActionInput(id=action), raw=True)
        self._renderer(self._steps, obs)

        if obs.state == GameState.WIN:
            next_level = self._adapter._current_level_index + 1
            if next_level < self._adapter.n_levels:
                # Brief pause so the WIN frame is visible, then load the next level.
                _gui_pause(0.8)
                self._adapter.set_level(next_level)
                self._steps = 0
                obs = self._adapter.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                obs.levels_completed = next_level
                self._renderer(self._steps, obs)

        return obs

    def step_mouse(self, x: int, y: int) -> FrameDataRaw:
        # ProcGen games don't use mouse clicks -- treat as no-op.
        return self._adapter.perform_action(ActionInput(id=GameAction.ACTION6), raw=True)

    def close(self) -> None:
        self._adapter.close()


def _load_procgen_env(name: str, renderer, level: int = 0, seed: Optional[int] = None):
    """Load a ProcGen game by its full client name (e.g. 'procgen_maze')."""
    prefix = "procgen_"
    if not name.startswith(prefix):
        return None
    game_name = name[len(prefix):]
    if game_name not in ProcGenAdapter.list_available_games():
        return None
    return _ProcGenEnvironmentWrapper(game_name=game_name, renderer=renderer, level=level, seed=seed)


# ---------------------------------------------------------------------------
# Gym-Gridworlds environment wrapper
# ---------------------------------------------------------------------------

class _GymGridworldsEnvironmentWrapper:
    """Small runtime wrapper so GymGridworldsAdapter fits the loop API."""

    def __init__(
        self,
        env_id: str,
        renderer,
        seed: Optional[int] = None,
        level: int = 0,
    ) -> None:
        self._renderer = renderer
        self._steps = 0
        self._adapter = GymGridworldsAdapter(
            env_id=env_id,
            seed=seed if seed is not None else random.randint(0, 2**31 - 1),
        )
        self._adapter.set_level(level)

    def reset(self) -> FrameDataRaw:
        self._adapter.level_reset()
        self._steps = 0
        obs = self._adapter.perform_action(ActionInput(id=GameAction.RESET), raw=True)
        self._renderer(self._steps, obs)
        return obs

    def step(self, action: GameAction) -> FrameDataRaw:
        self._steps += 1
        obs = self._adapter.perform_action(ActionInput(id=action), raw=True)
        self._renderer(self._steps, obs)
        return obs

    def step_mouse(self, x: int, y: int) -> FrameDataRaw:
        # Gym-Gridworlds doesn't use mouse clicks -- treat as no-op (STAY).
        return self._adapter.perform_action(ActionInput(id=GameAction.ACTION5), raw=True)

    def close(self) -> None:
        self._adapter.close()


def _load_gymgridworlds_env(name: str, renderer, level: int = 0, seed: Optional[int] = None):
    """Load a Gym-Gridworlds env from a 'gymgw:<env_id>' client name."""
    prefix = "gymgw:"
    if not name.startswith(prefix):
        return None
    env_id = name[len(prefix):]
    if env_id not in GymGridworldsAdapter.list_available_envs():
        return None
    return _GymGridworldsEnvironmentWrapper(env_id=env_id, renderer=renderer, level=level, seed=seed)


# ---------------------------------------------------------------------------
# PuzzleScript environment wrapper
# ---------------------------------------------------------------------------

class _PuzzleScriptEnvironmentWrapper:
    """Small runtime wrapper so PuzzleScriptAdapter fits the loop API."""

    def __init__(
        self,
        game_name: str,
        renderer,
        seed: Optional[int] = None,
        level: int = 0,
        adapter: Optional[PuzzleScriptAdapter] = None,
    ) -> None:
        self._renderer = renderer
        self._steps = 0
        if adapter is not None:
            self._adapter = adapter
        else:
            self._adapter = PuzzleScriptAdapter(
                game_name=game_name,
                seed=seed if seed is not None else random.randint(0, 2**31 - 1),
            )
        self._adapter.set_level(level)

    def reset(self) -> FrameDataRaw:
        self._adapter.level_reset()
        self._steps = 0
        obs = self._adapter.perform_action(ActionInput(id=GameAction.RESET), raw=True)
        self._renderer(self._steps, obs)
        return obs

    def step(self, action: GameAction) -> FrameDataRaw:
        self._steps += 1
        obs = self._adapter.perform_action(ActionInput(id=action), raw=True)
        self._renderer(self._steps, obs)

        if obs.state == GameState.WIN:
            next_level = self._adapter._current_level_index + 1
            if next_level < self._adapter.n_levels:
                # Brief pause so the WIN frame is visible, then load the next level.
                _gui_pause(0.8)
                self._adapter.set_level(next_level)
                self._steps = 0
                obs = self._adapter.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                obs.levels_completed = next_level
                self._renderer(self._steps, obs)

        return obs

    def step_mouse(self, x: int, y: int) -> FrameDataRaw:
        # PuzzleScript doesn't use mouse clicks -- treat as ACTION5.
        return self._adapter.perform_action(ActionInput(id=GameAction.ACTION5), raw=True)

    def close(self) -> None:
        self._adapter.close()


def _load_puzzlescript_env(name: str, renderer, level: int = 0, seed: Optional[int] = None):
    """Load a PuzzleScript env from a 'ps:<game_name>' client name.

    Accepts either the raw file stem (e.g. ps:Atlas_Shrank) or the normalized name
    used in game folders (e.g. ps:atlas_shrank). Falls back to checking
    games/ps:<name>/metadata.json for the adapter_game_name field.

    If games/ps:<name>/ps:<name>.py defines `make_game(seed)`, the returned adapter is
    used as-is (this is how per-game subclasses customize behavior such as per-level
    step limits)."""
    prefix = "ps:"
    if not name.startswith(prefix):
        return None
    game_name = name[len(prefix):]

    # Per-game adapter customization via games/ps:<name>/ps:<name>.py:make_game.
    custom_adapter: Optional[PuzzleScriptAdapter] = None
    game_file = os.path.join(_HERE, "games", f"ps:{game_name}", f"ps:{game_name}.py")
    if os.path.isfile(game_file):
        module_key = f"ps_game_{game_name}"
        spec = importlib.util.spec_from_file_location(module_key, game_file)
        module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        sys.modules[module_key] = module
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        if hasattr(module, "make_game"):
            custom_adapter = module.make_game(seed=seed if seed is not None else random.randint(0, 2**31 - 1))

    if custom_adapter is not None:
        return _PuzzleScriptEnvironmentWrapper(
            game_name=custom_adapter._game_name,
            renderer=renderer,
            level=level,
            adapter=custom_adapter,
        )

    available = PuzzleScriptAdapter.list_available_games()
    if game_name in available:
        return _PuzzleScriptEnvironmentWrapper(game_name=game_name, renderer=renderer, level=level, seed=seed)
    # Try loading via metadata.json in the game folder.
    meta_path = os.path.join(_HERE, "games", f"ps:{game_name}", "metadata.json")
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        adapter_name = meta.get("adapter_game_name", game_name)
        if adapter_name in available:
            return _PuzzleScriptEnvironmentWrapper(game_name=adapter_name, renderer=renderer, level=level, seed=seed)
    return None


# ---------------------------------------------------------------------------
# Synthetic geodesic maze (utils/synthetic.py -- an in-memory env, no game dir)
# ---------------------------------------------------------------------------

class _SyntheticGeodesicEnvironmentWrapper:
    """Loop-API wrapper over ``utils.synthetic.SyntheticMazeGame``.

    The synthetic maze is the one env with no ``games/<id>/`` directory and no
    adapter, so nothing in the cascade below used to resolve it -- every caller that
    went through ``resolve_env`` (generate_goal_input.py, evidence_log.py) fell all
    the way through to the remote Arcade and got None back. This wrapper closes that
    hole by speaking FrameDataRaw like every other wrapper, over the SAME game object
    the corpus generator records from (so a (seed, level) here is the (seed, level)
    the corpus holds).

    Keyboard-only: ACTION1 up, 2 down, 3 left, 4 right. Reaching a maze's goal
    auto-advances to the next, exactly as an arcengine game does; WIN lands when the
    last maze falls."""

    def __init__(self, renderer, level: int = 0, seed: Optional[int] = None) -> None:
        from utils.synthetic import SyntheticMazeGame, render_game, step_agent

        self._render_game, self._step_agent = render_game, step_agent
        self._seed = seed if seed is not None else random.randint(0, 2**31 - 1)
        self._renderer = renderer
        self._start_level = int(level)
        self._steps = 0
        self.levels_completed = 0
        # Named _maze, NOT _game: _get_underlying_game()/_jump_to_level() key off
        # `_game` and would then try to drive this through a camera it doesn't have.
        # reset() already seats the requested level, so _jump_to_level is a no-op here.
        self._maze = SyntheticMazeGame(self._seed)
        n = len(self._maze.levels)
        if not 0 <= self._start_level < n:
            # Same wording the engine uses, so master_goal_input_generator's
            # LEVEL_RANGE_RE folds out-of-range levels instead of logging a failure.
            raise ValueError(f"Level index {self._start_level} out of range [0, {n})")

    def _obs(self, state: GameState) -> FrameDataRaw:
        fdr = FrameDataRaw()
        fdr.frame = [self._render_game(self._maze)]
        fdr.state = state
        fdr.levels_completed = self.levels_completed
        fdr.win_levels = len(self._maze.levels)
        fdr.available_actions = [1, 2, 3, 4]        # keyboard-only; no click
        return fdr

    def reset(self) -> FrameDataRaw:
        self._steps = 0
        self.levels_completed = 0
        self._maze.set_level(self._start_level)
        obs = self._obs(GameState.NOT_FINISHED)
        self._renderer(self._steps, obs)
        return obs

    def step(self, action: GameAction) -> FrameDataRaw:
        self._steps += 1
        g = self._maze
        g.agent = self._step_agent(g.level, g.agent, int(action.value))
        state = GameState.NOT_FINISHED
        if g.solved:
            self.levels_completed += 1
            if not g.advance():
                state = GameState.WIN
        obs = self._obs(state)
        self._renderer(self._steps, obs)
        return obs

    def step_mouse(self, x: int, y: int) -> FrameDataRaw:
        # The maze has no click action -- a click is a legal no-op, like a blocked move.
        return self._obs(GameState.NOT_FINISHED)

    def close(self) -> None:
        pass


def _load_synthetic_env(name: str, renderer, level: int = 0, seed: Optional[int] = None):
    """Load the in-memory synthetic maze from the 'synthetic_geodesic' name."""
    if name not in ("synthetic_geodesic", "synthetic"):
        return None
    return _SyntheticGeodesicEnvironmentWrapper(renderer=renderer, level=level, seed=seed)


# ---------------------------------------------------------------------------
# Native ARC-AGI-3 game loader (games/<id>/<id>.py via the arc_agi framework)
# ---------------------------------------------------------------------------

_DEFAULT_ARCADE: Optional["arc_agi.Arcade"] = None


def default_arcade() -> "arc_agi.Arcade":
    """Lazily build (and cache) a shared Arcade. Native games run fully locally, so we
    use OFFLINE mode (no anon-key fetch, no /fetch_from_api, no scorecard POST) and a
    WARNING-level logger to suppress the per-level INFO spam. To talk to the remote API
    instead, set OPERATION_MODE=online (or normal) in the environment."""
    global _DEFAULT_ARCADE
    if _DEFAULT_ARCADE is None:
        quiet_logger = logging.getLogger("arc_agi.base")
        quiet_logger.setLevel(logging.WARNING)
        op_mode = arc_agi.base.OperationMode.OFFLINE if not os.getenv("OPERATION_MODE") else arc_agi.base.OperationMode.NORMAL
        _DEFAULT_ARCADE = arc_agi.Arcade(operation_mode=op_mode, logger=quiet_logger)
    return _DEFAULT_ARCADE


def is_native_game(game_name: str) -> bool:
    """A native local game is a non-PuzzleScript folder games/<id>/<id>.py."""
    if game_name.startswith("ps:"):
        return False
    return os.path.isfile(os.path.join(_HERE, "games", game_name, f"{game_name}.py"))


def _load_local_env(arcade: "arc_agi.Arcade", game_name: str, renderer, seed: Optional[int] = None):
    """Load a native local game (games/<game_name>/<game_name>.py) through the arc_agi
    framework. The class is the PascalCase of the snake_case id (ar25 -> Ar25,
    maze -> Maze, alloy_furnace -> AlloyFurnace)."""
    game_file = os.path.join(_HERE, "games", game_name, f"{game_name}.py")
    if not os.path.exists(game_file):
        return None

    spec = importlib.util.spec_from_file_location(game_name, game_file)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[game_name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]

    # Derive PascalCase class name (e.g. alloy_furnace -> AlloyFurnace). A ':' prefix
    # namespace is also a word boundary (mc:lamelightsout -> McLamelightsout), so
    # mouse-click native games can share a stem with their PuzzleScript sibling.
    pascal_name = "".join(part.capitalize() for part in re.split(r"[_:]", game_name))
    game_cls = getattr(module, pascal_name, None)
    if game_cls is None:
        raise ImportError(f"Class '{pascal_name}' not found in {game_file}")

    game_dir = os.path.dirname(game_file)
    # Pass game_name (snake_case) as class_name so the framework finds the file via
    # class_name.lower(). It can't look up a class named "<game_name>" (the real class is
    # PascalCase), so it logs two expected ERRORs; we inject the loaded instance below.
    env_info = arc_agi.base.EnvironmentInfo(
        game_id=game_name,
        class_name=game_name,
        local_dir=game_dir,
    )
    scorecard_id = arcade.create_scorecard()
    arcade._default_scorecard_id = scorecard_id  # so get_scorecard() finds it
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    _prev_level = arcade.logger.level
    arcade.logger.setLevel(logging.CRITICAL)
    try:
        wrapper = arc_agi.base.LocalEnvironmentWrapper(
            environment_info=env_info,
            logger=arcade.logger,
            scorecard_id=scorecard_id,
            scorecard_manager=arcade.scorecard_manager,
            renderer=renderer,
            seed=seed,
        )
    finally:
        arcade.logger.setLevel(_prev_level)
    # The framework finds the file but can't look up a class named "<game_name>" --
    # inject the pre-loaded instance and reset properly.
    if wrapper._game is None:
        kwargs = {"seed": seed} if "seed" in inspect.signature(game_cls).parameters else {}
        wrapper._game = game_cls(**kwargs)
        # A game whose __init__ does not declare `seed` was just built UNSEEDED, so
        # its display rotation came from the global fallback draw -- unpinnable by
        # the harness, hence not reproducible from (game, seed, level). Install the
        # seed after the fact so the orientation is a pure function of (seed, level)
        # for every native game, not only the ones that accept a seed argument.
        # See utils.arc_game.seed_game / AugmentedGame.adopt_seed.
        if not kwargs:
            arc_game.seed_game(wrapper._game, seed)
        wrapper.reset()
    return wrapper


# ---------------------------------------------------------------------------
# Unified resolution cascade
# ---------------------------------------------------------------------------

def resolve_env(game_name: str, renderer=None, level: int = 0, seed: Optional[int] = None,
                arcade: Optional["arc_agi.Arcade"] = None):
    """Resolve a game-string to a runnable env wrapper (the loop API above).

    First match wins: procgen_ -> gymgw: -> ps: -> [partial:]MiniGrid ->
    synthetic_geodesic -> native games/<id>/<id>.py -> arcade remote. Returns the
    wrapper; never None (the remote Arcade.make is the final fallback and raises on a
    truly unknown id)."""
    if renderer is None:
        renderer = NOOP_RENDERER
    env = _load_procgen_env(game_name, renderer, level=level, seed=seed)
    if env is None:
        env = _load_gymgridworlds_env(game_name, renderer, level=level, seed=seed)
    if env is None:
        env = _load_puzzlescript_env(game_name, renderer, level=level, seed=seed)
    if env is None:
        env = _load_minigrid_env(game_name, renderer, level=level, seed=seed)
    if env is None:
        env = _load_synthetic_env(game_name, renderer, level=level, seed=seed)
    if env is None:
        env = _load_local_env(arcade or default_arcade(), game_name, renderer, seed=seed)
    if env is None:
        env = (arcade or default_arcade()).make(game_name, renderer=renderer)
    return env


# Wrapper class -> short kind label (cosmetic; the planner prints it).
_KIND_BY_WRAPPER = {
    "_MinigridEnvironmentWrapper": "minigrid",
    "_ProcGenEnvironmentWrapper": "procgen",
    "_GymGridworldsEnvironmentWrapper": "gymgw",
    "_PuzzleScriptEnvironmentWrapper": "puzzlescript",
    "_SyntheticGeodesicEnvironmentWrapper": "synthetic",
    "LocalEnvironmentWrapper": "arc",
}


def env_kind(env) -> str:
    return _KIND_BY_WRAPPER.get(type(env).__name__, "arc")


# ---------------------------------------------------------------------------
# Shared introspection / level helpers
# ---------------------------------------------------------------------------

def _get_underlying_game(env):
    """Return the ARCBaseGame instance buried inside any env wrapper, or None."""
    game = getattr(env, "_game", None)
    if game is not None:
        return game
    adapter = getattr(env, "_adapter", None)
    if adapter is not None:
        game = getattr(adapter, "_game", None)
        if game is not None:
            return game
        # Adapters (Minigrid, ProcGen) expose _step_counter directly.
        if hasattr(adapter, "_step_counter"):
            return adapter
    return None


def _jump_to_level(env, level: int, renderer) -> None:
    """Move env to *level* and re-render the opening frame via the renderer."""
    if level == 0:
        return

    # Adapter wrappers (Minigrid, ProcGen, ...) expose set_level on their adapter and
    # render via perform_action, not a camera. Handle them first: _get_underlying_game
    # returns the adapter itself (it has _step_counter), so the camera path below would
    # otherwise be taken and fail with AttributeError: no 'camera' attribute.
    _adapter = getattr(env, "_adapter", None)
    if (
        _adapter is not None
        and getattr(_adapter, "_game", None) is None
        and hasattr(_adapter, "set_level")
    ):
        adapter = _adapter
        # Skip set_level if already at the right level (avoids expensive gym env
        # recreation in ProcGenAdapter where set_level calls _make_good_env).
        already_there = (
            hasattr(adapter, "_current_level_index")
            and adapter._current_level_index == level
        )
        if not already_there:
            adapter.set_level(level)
        obs = adapter.perform_action(ActionInput(id=GameAction.RESET), raw=True)
        renderer(0, obs)
        return

    game = _get_underlying_game(env)
    if game is None or not hasattr(game, "set_level"):
        return

    game.set_level(level)

    # Re-render the new level so the display is correct immediately. Use camera.render()
    # (not _raw_render) so camera interfaces (e.g. RotationDisplay) are applied.
    rendered_frame = game.camera.render(game.current_level.get_sprites())
    fdr = FrameDataRaw()
    fdr.frame = [rendered_frame]
    fdr.state = game._state
    # This observation also initializes live solvers, so retain the engine's
    # action mask and progress metadata after the level change.
    fdr.game_id = game._game_id
    fdr.levels_completed = game._score
    fdr.win_levels = game._win_score
    fdr.available_actions = list(game._available_actions)
    renderer(0, fdr)


def _gui_pause(seconds: float) -> None:
    """Visual dwell on level-win transitions. Uses matplotlib's event-loop pause when a
    GUI session is active; a harmless no-op otherwise (headless planner runs)."""
    try:
        import matplotlib.pyplot as plt
        if plt.get_fignums():
            plt.pause(seconds)
    except Exception:  # noqa: BLE001  -- never let a cosmetic pause break a step
        pass
