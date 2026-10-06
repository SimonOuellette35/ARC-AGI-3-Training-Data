"""BaseAdapter -- shared base for the external-environment adapters.

The five adapters (`MinigridAdapter`, `GymGridworldsAdapter`, `ProcGenAdapter`,
`ALEAdapter`, `PuzzleScriptAdapter`) each wrap a different external engine as an
ARCBaseGame-compatible object, and each independently re-implemented the same
scaffolding: a ``_StepCounter``, the ``_state`` bookkeeping, the ``perform_action``
skeleton (RESET handling + WIN/GAME_OVER short-circuit + step accounting +
FrameData construction), and -- until now -- NO working UNDO.

``BaseAdapter`` factors that scaffolding out and, crucially, provides ONE correct
generic UNDO. Native arcengine games get undo from the engine; the adapter-wrapped
games (minigrid/procgen/puzzlescript/gymgw/ale) got it nowhere, which is exactly
the gap this closes. UNDO here is snapshot-based, so it reverses ANY action
(clicks included) and is identical at train and test time:

  * before applying a normal action, ``perform_action`` pushes ``self._snapshot()``
    onto a bounded undo stack;
  * ``ACTION7`` pops the stack and ``self._restore()``s it -- and this works even
    from GAME_OVER / WIN, so a perturbation burst that strands or kills the agent
    can be undone step-for-step back to the exact pre-burst state.

A subclass supplies the env-specific pieces:

  * ``_apply(action_input)`` -- perform ONE non-RESET/non-ACTION7 action against
    the real env, updating ``self._current_frame`` / ``self._state`` /
    ``self._action_count`` / ``self._step_counter`` (this is the body the adapter's
    old ``perform_action`` had, minus the RESET/terminal/undo plumbing).
  * ``_snapshot()`` / ``_restore(snap)`` -- capture / restore the full mutable
    state (env + adapter bookkeeping). Cheap per env: grid-copy for the
    gridworlds/puzzlescript, ``get_state``/``set_state`` for procgen,
    ``cloneState``/``restoreState`` for ALE.
  * ``_reset_env(seed)`` -- (re)initialise the wrapped env to ``seed`` and set the
    initial ``self._current_frame`` / ``self._state``.

The public API each adapter exposed (``perform_action``, ``set_level`` /
``full_reset`` / ``level_reset``, ``.game_id``, ``._state``, ``._current_frame``,
``._step_counter``, ``._make_frame_data``) is preserved, so the generators that
drive these adapters need no changes.
"""
from __future__ import annotations

from arcengine import ActionInput, GameAction, GameState, FrameDataRaw

_ACTION7 = GameAction.ACTION7
_RESET = GameAction.RESET


class StepCounter:
    """Remaining step budget for the progress-bar HUD (shared by all adapters)."""

    def __init__(self, max_steps: int) -> None:
        self.max_steps = max_steps
        self.steps_remaining = max_steps

    def reset(self, max_steps: int | None = None) -> None:
        if max_steps is not None:
            self.max_steps = max_steps
        self.steps_remaining = self.max_steps


class BaseAdapter:
    """Common scaffolding + generic snapshot-based UNDO for external-env adapters."""

    #: how many past states are kept undoable (bounded so per-step snapshotting
    #: stays cheap; bursts are short, so a small window is plenty).
    max_undo: int = 64

    # ── construction helpers (subclass __init__ calls these) ─────────────────
    def _init_base(self, game_id: str, *, max_steps: int,
                   available_actions: list[int] | None = None) -> None:
        """Initialise the shared adapter state. Call from the subclass __init__
        AFTER the wrapped env exists, then call ``self.set_level`` (or
        ``_reset_env``) to produce the first frame."""
        self._game_id = game_id
        self._state: GameState = GameState.NOT_PLAYED
        self._current_level_index: int = 0
        self._action_count: int = 0
        self._available_actions: list[int] = (
            available_actions if available_actions is not None
            else [1, 2, 3, 4, 5, 6, 7])
        self._current_frame = None
        self._step_counter = StepCounter(max_steps)
        self._undo_stack: list = []

    # ── subclass hooks (env-specific) ────────────────────────────────────────
    def _reset_env(self, seed: int) -> None:
        """(Re)initialise the wrapped env to ``seed`` and set the initial
        ``self._current_frame`` and ``self._state`` (NOT_FINISHED)."""
        raise NotImplementedError

    def _apply(self, action_input: ActionInput) -> None:
        """Apply ONE non-RESET/non-ACTION7 action to the real env, updating
        ``self._current_frame`` / ``self._state`` / ``self._action_count`` /
        ``self._step_counter``. (The adapter's old perform_action body.)"""
        raise NotImplementedError

    def _snapshot(self):
        """Return a restorable snapshot of the FULL mutable state (env + adapter
        bookkeeping like ``_action_count`` / ``_step_counter.steps_remaining`` /
        ``_state`` / ``_current_frame``)."""
        raise NotImplementedError

    def _restore(self, snap) -> None:
        """Restore a snapshot produced by ``_snapshot``."""
        raise NotImplementedError

    # ── shared ARCBaseGame-compatible interface ──────────────────────────────
    @property
    def game_id(self) -> str:
        return self._game_id

    def set_level(self, idx: int) -> None:
        """Select a 'level' by resetting the env with seed = base_seed + idx."""
        self._current_level_index = idx
        self._undo_stack.clear()
        self._reset_env(self._seed + idx)

    def full_reset(self) -> None:
        self._current_level_index = 0
        self._undo_stack.clear()
        self._reset_env(self._seed)

    def level_reset(self) -> None:
        self._undo_stack.clear()
        self._reset_env(self._seed + self._current_level_index)

    def perform_action(self, action_input: ActionInput, raw: bool = False) -> FrameDataRaw:
        """Drive one action, with RESET + generic UNDO handled here and the
        env-specific step delegated to ``_apply``."""
        aid = action_input.id
        if aid == _RESET:
            self.level_reset()
            return self._make_frame_data(action_input)
        # UNDO is handled BEFORE the terminal short-circuit so it can revive the
        # agent from a GAME_OVER / WIN reached during a perturbation burst.
        if aid == _ACTION7:
            if self._undo_stack:
                self._restore(self._undo_stack.pop())
            return self._make_frame_data(action_input)
        if self._state in (GameState.WIN, GameState.GAME_OVER):
            return self._make_frame_data(action_input)
        # snapshot the pre-action state (bounded), then apply the real action
        self._undo_stack.append(self._snapshot())
        if len(self._undo_stack) > self.max_undo:
            self._undo_stack.pop(0)
        self._apply(action_input)
        return self._make_frame_data(action_input)

    def _make_frame_data(self, action_input: ActionInput) -> FrameDataRaw:
        fd = FrameDataRaw()
        fd.game_id = self._game_id
        fd.state = self._state
        fd.levels_completed = self._current_level_index
        fd.win_levels = 1
        fd.action_input = action_input
        fd.full_reset = False
        fd.available_actions = self._available_actions
        if self._current_frame is not None:
            fd.frame = [self._current_frame]
        return fd
