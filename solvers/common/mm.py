"""Shared honest-play solver + training-data generator for the mm01..mm05 family.

The five ``mm`` games (games/mm0N/mm0N.py) are one skeleton with four variations,
so they share one solver. Each ``generate_mm0N_training.py`` is a thin wrapper that
supplies a :class:`MmSpec` and a game-specific docstring.

    game   group  budget  mismatch cost              extra
    mm01     2      60    4 (2 clicks + 2 paid waits)
    mm02     3      90    3 (waits are FREE)
    mm03     3      90    5 (waits are PAID)         == mm02 but for the wait cost
    mm04     2      60    4                          ACTION5 peek: reveals all, 5 steps
    mm05     2      60    4                          "sticky" rule is dead code == mm01

WHY THE SOLVER IS NON-PRIVILEGED
--------------------------------
These are *hidden-state* games: a face-down tile's colour is not in the frame, so a
policy conditioned on the observation history **cannot** know it. An oracle solver
that read ``game._slot_colors`` and clicked each pair straight off would emit a
trajectory no observation-conditioned learner can reproduce -- behaviour cloning on
it would teach mind-reading and get compounding error instead (the badge_placement
failure mode, in its purest form: here the information is not merely hard to
extract, it is absent).

So this solver plays *honestly*. Tile colours enter its state exactly one way:
:func:`_read_board` takes the mode of each tile's pixel block in the **rendered
frame**. It never reads ``_slot_colors``, ``_matched`` or ``_flipped``. Everything it
knows, the trained model can see too. What it does read from the game object is
*geometry* (rows/cols/tile size/offsets) -- static, seed-invariant, and plainly
visible in the frame; it only turns a slot index into click coordinates.

STRATEGY (uniform over pairs and triplets)
------------------------------------------
The standard human concentration strategy, decided click-by-click so each action is
justified by what is on screen at that moment:

  * Start a turn by clicking a colour we already know ``group_size`` copies of --
    a guaranteed clear.
  * Otherwise probe the lowest-indexed unknown tile.
  * Mid-turn, while everything flipped so far shares colour ``c``, click a known
    ``c`` tile if we have one (that completes the group); else probe an unknown.
  * Once the turn is doomed (flipped tiles disagree) we are forced to finish it --
    the engine only scores at ``group_size`` flips and re-clicking a flipped tile is
    a wasted step -- so spend the remaining clicks probing unknowns for information.

Probe order is a deterministic low-index-first scan rather than a random pick. It is
a complete strategy either way, and a deterministic choice keeps the "which tile to
probe" half of the policy trivially learnable so model capacity goes to the half
that matters: *recalling* which slot showed which colour. Per-episode variety comes
from the slot shuffle and palette, not from solver noise.

Worst case is comfortably inside every budget. For n pairs, every tile can be
learned in at most n all-mismatch turns (4 steps each) and then every pair clicked
from memory (2 steps each): 6n = 48 <= 59 usable steps at mm01's hardest level
(n=8). Triplets need ~32 of 89. A seed that somehow overruns is simply skipped.

mm04 is the same loop with one extra move: fire the ACTION5 peek first and read the
whole board out of the peek frames. Perfect knowledge then makes every turn a
guaranteed clear (5 + 2n = 21 steps at n=8). The peek is the game's whole point, and
the resulting data teaches the intended lesson -- look, memorise, execute -- with
every click still grounded in a frame the model saw.

ROTATION (the presentation augmentation)
----------------------------------------
The mm games are ``AugmentedGame``s: every level is displayed at a per-(seed, level)
rotation k, frames go out turned by k and actions come back in turned by -k. This
solver therefore works entirely in SCREEN space -- the space the agent sees:

  * :func:`_render` returns the rotated frame (what gets recorded, and what a live
    agent would see), and :func:`_read_board` un-turns it before slicing the tile
    grid, so perception stays a pure function of the observation;
  * ``_Geom.center`` returns the SCREEN click point for a slot, and :func:`_drive`
    de-rotates it exactly as ``AugmentedGame.perform_action`` would (``_drive``
    reaches past ``perform_action`` to control stepping, so it has to redo that
    boundary itself). Recorded clicks are consequently replayable as-is.

``make_level`` constructs the game with ``seed=``, which is what makes k -- and the
hidden layout -- a pure function of (seed, level). Hand-seeding ``_rng``/``_color_rng``
from outside (what this used to do) has been dead since the games moved to
``level_rng``: ``on_set_level`` re-draws both, so an unseeded game re-randomised its
board AND its rotation on every level, which is what broke the whole family.

Action schema (mixed simple + mouse, matching cd82 / cn04 / gp01)
    RESET / simple :  {"type": "simple", "index": k}                      # k in 0..5
    mouse click    :  {"type": "mouse",  "index": 6, "data": {"x": x, "y": y}}

See the per-game generators for usage; run everything with the ARC-AGI-3 conda python.
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path

# Repo root (parent of solvers/) -- where the games/ package + engine live.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np  # noqa: E402
from arcengine import ActionInput, GameAction, GameState  # noqa: E402

from solvers.base_solver import BaseSolver, _ID_TO_GAMEACTION  # noqa: E402
from utils.explore import (  # noqa: E402
    Action, CLICK_ACTION, RESET_ACTION, EpsilonSchedule, ExplorationPolicy)
from utils.rotation import remap_click  # noqa: E402

# A face-down tile renders as HIDDEN_COLOR. Every mm game draws its background and
# tile palette from a pool that excludes it, so "hidden" is never ambiguous with a
# real tile colour -- which is what lets _read_board classify a tile from one frame.
HIDDEN_COLOR = 3

_RESET_ACTION = {"type": "simple", "index": int(GameAction.RESET.value)}
_ACTION6 = int(GameAction.ACTION6.value)
_STEP_GUARD = 6000
# The edge length AugmentedGame de-rotates clicks against (utils.arc_game._FRAME_SIZE).
# It is a fixed 64 there, NOT camera.width, so the inverse map here must use 64 too.
_FRAME_SIZE = 64
# A click is swallowed while the board is locked (mm01/03 flip-back = 2 steps, mm02
# = 2 free steps, mm04's peek = 4). Re-send until it lands; this bounds the retries.
_CLICK_RETRIES = 8


@dataclass(frozen=True)
class MmSpec:
    """What distinguishes one mm game from the others."""

    game_id: str
    game_cls: type
    group_size: int  # 2 = pairs (mm01/04/05), 3 = triplets (mm02/03)
    peek: bool = False  # mm04: ACTION5 reveals every unmatched tile for 5 steps


@dataclass(frozen=True)
class _Geom:
    """Tile-grid geometry, plus the level's display rotation.

    The grid itself is static, seed-invariant and plainly visible in the frame; ``k``
    is the per-(seed, level) rotation the board is being SHOWN at, and is the single
    place the two coordinate spaces are reconciled: :meth:`center` emits screen
    coordinates, :func:`_read_board` un-turns the frame back to this upright grid.
    """

    rows: int
    cols: int
    tile: int
    offset_x: int
    offset_y: int
    k: int = 0

    @property
    def num_tiles(self) -> int:
        return self.rows * self.cols

    def game_center(self, slot: int) -> tuple[int, int]:
        """``slot``'s centre in the game's own upright space. The camera renders
        64x64 at scale 1 with no offset, so level coordinates and ACTION6 display
        coordinates coincide there."""
        row, col = divmod(slot, self.cols)
        return (self.offset_x + col * self.tile + self.tile // 2,
                self.offset_y + row * self.tile + self.tile // 2)

    def center(self, slot: int) -> tuple[int, int]:
        """SCREEN click point for ``slot`` -- where the tile actually appears in the
        rotated frame, and therefore what gets recorded.

        ``remap_click`` is the screen -> game map for rotation k; rotations form a
        cyclic group, so its inverse is the same map at ``-k``.
        """
        x, y = self.game_center(slot)
        return remap_click(x, y, (4 - self.k) % 4, _FRAME_SIZE)


def _geom(game) -> _Geom:
    return _Geom(game._rows, game._cols, game._tile_size, game._offset_x,
                 game._offset_y, getattr(game, "_rotation_k", 0) % 4)


# ── Engine helpers ────────────────────────────────────────────────────────────
def _render(game) -> np.ndarray:
    """Current 64x64 palette frame, rendered WITHOUT stepping the game.

    ``camera.render`` is the AugmentedGame render hook, so this is the ROTATED frame
    -- exactly what a live agent sees and what the corpus stores."""
    return np.asarray(game.camera.render(game.current_level.get_sprites()))


def _drive(game, action_input: ActionInput):
    """Perform one SCREEN-space action against the real engine, stopping the instant
    a level solve is queued so the captured frame is the CLEAN solved board of the
    current level (NOT the next level). Returns (frame, solved, dead).

    This drives ``_set_action``/``step`` directly instead of calling
    ``perform_action``, which is where ``AugmentedGame`` de-rotates incoming actions
    -- so it must do that itself, or every click would be interpreted in the wrong
    space and the recorded coordinates would not replay.
    """
    if getattr(game, "_rotation_k", 0) % 4:
        action_input = game._augment_derotate_action(action_input)
    game._full_reset = False
    game._set_action(action_input)
    last = None
    guard = 0
    while not game.is_action_complete():
        guard += 1
        if guard > _STEP_GUARD:
            break
        if game._next_level:
            break
        game.step()
        last = _render(game)
    if last is None:  # a no-op action rendered nothing new
        last = _render(game)
    solved = game._next_level or game._state == GameState.WIN
    dead = game._state == GameState.GAME_OVER
    return last, solved, dead


def make_level(spec: MmSpec, seed: int, level_idx: int):
    """Fresh game positioned on ``level_idx`` with a per-seed board and rotation.

    The mm games take ``seed=``, and everything random about them is drawn through
    ``AugmentedGame`` from it: the slot shuffle (``level_rng("layout")``) and the
    display rotation are pure functions of (seed, level), and the colour scheme is
    per seed. So constructing with the seed is the whole of the determinism story --
    ``set_level`` never short-circuits on the current index and ``_randomize_level``
    rebuilds from the module-level LEVEL_LAYOUTS, so re-entering re-randomises
    cleanly rather than compounding.

    Do NOT hand-seed ``_rng``/``_color_rng`` from out here (this used to, from before
    the games had a seed): ``on_set_level`` re-draws both from ``level_rng``, so the
    assignment is overwritten and only the *unseeded* game is left -- a fresh board
    and a fresh rotation on every level, every run. Likewise, ``_color_scheme`` is
    deliberately NOT cleared: it is drawn once when the constructor seats level 0 and
    then kept, which is what makes one episode's palette identical across its levels.
    """
    game = spec.game_cls(seed=seed)
    game.set_level(level_idx)
    return game


def num_levels(spec: MmSpec) -> int:
    return len(spec.game_cls()._levels)


def _mouse_action(x: int, y: int) -> dict:
    return {"type": "mouse", "index": _ACTION6, "data": {"x": int(x), "y": int(y)}}


def _simple_action(idx: int) -> dict:
    return {"type": "simple", "index": int(idx)}


# ── Honest perception: the ONLY channel for tile colours ──────────────────────
def _read_board(frame: np.ndarray, geom: _Geom) -> list[int]:
    """Per-slot colour read out of ``frame``: HIDDEN_COLOR if face-down, else the
    tile's colour. Uses each tile block's modal pixel so the UI's 5-pixel click
    marker (which lands on the tile centre) cannot flip the reading.

    ``frame`` is the SCREEN frame, i.e. turned by ``geom.k``; un-turn it first so the
    tile grid lines up with the game's upright offsets. That is a pure operation on
    the observation -- the rotation is plainly visible in the frame (a live agent
    reads the board the same way) -- so this stays the honest perception channel.
    """
    if geom.k:
        frame = np.rot90(np.asarray(frame), -geom.k)
    board = []
    for slot in range(geom.num_tiles):
        row, col = divmod(slot, geom.cols)
        y0, x0 = geom.offset_y + row * geom.tile, geom.offset_x + col * geom.tile
        block = frame[y0:y0 + geom.tile, x0:x0 + geom.tile].ravel()
        board.append(int(np.bincount(block, minlength=16).argmax()))
    return board


class _Recorder:
    """Accumulates the (observations, actions) pair in the training schema, where
    ``actions[i]`` is the action that produced ``observations[i]``.

    ``phases[i]`` tags each step for the encoder: ``"reset"`` for index 0 (and any
    mid-trajectory RESET-recovery), ``"explore"`` for exploration-prefix clicks, and
    ``"expert"`` for the honest concentration play. Honest play (``play_level`` /
    ``_honest_play``) never touches phases -- it just calls ``add`` and gets the
    default ``"expert"`` -- so the untouched-honest-play guarantee holds."""

    def __init__(self, game) -> None:
        # ONE render: the mm click marker ages on a render counter, so a spare
        # _render here would put every recorded frame a tick out of step with a
        # replay (which only ever renders inside perform_action).
        frame = _render(game)
        self.observations = [frame.tolist()]
        self.actions = [dict(_RESET_ACTION)]
        self.phases = ["reset"]
        self.last_frame = frame

    def add(self, frame: np.ndarray, action: dict, phase: str = "expert") -> None:
        self.observations.append(np.asarray(frame).tolist())
        self.actions.append(action)
        self.phases.append(phase)
        self.last_frame = frame


def _choose(flipped: list[int], known: dict[int, int], matched: set[int],
            group: int, n: int) -> int:
    """The next tile to click, from revealed information only.

    ``flipped`` is this turn's clicks so far, ``known`` every colour ever seen,
    ``matched`` the groups already cleared.
    """
    unmatched = [s for s in range(n) if s not in matched]
    # Known, still-clickable tiles by colour.
    avail: dict[int, list[int]] = {}
    for slot in unmatched:
        if slot in known and slot not in flipped:
            avail.setdefault(known[slot], []).append(slot)
    unknown = [s for s in unmatched if s not in known and s not in flipped]

    if not flipped:
        # Prefer a guaranteed clear: a colour we already know every copy of.
        for color in sorted(avail):
            if len(avail[color]) >= group:
                return avail[color][0]
        if unknown:
            return unknown[0]
        # Unreachable: with every tile known, some colour must have all `group`
        # copies unmatched. Kept as a total-function fallback.
        return unmatched[0]

    colors = {known[s] for s in flipped}
    if len(colors) == 1:
        # Still alive -- extend the group with a known tile of the same colour,
        # otherwise probe (which may complete it by luck and learns either way).
        color = next(iter(colors))
        if avail.get(color):
            return avail[color][0]
    # Doomed turn (or nothing known to extend with): buy information.
    if unknown:
        return unknown[0]
    return next(s for s in unmatched if s not in flipped)


def _click_until_landed(game, rec: _Recorder, slot: int, expected: set[int],
                        geom: _Geom):
    """Click ``slot`` until the flip actually takes, recording every attempt.

    A click is silently swallowed while the board is locked -- during mm01/02/03's
    flip-back and during mm04's peek. Rather than encode each game's lock timing, we
    detect success from the frame: the flip landed exactly when the revealed set is
    the one we expect (``matched | flipped | {slot}``). That predicate also keeps the
    peek honest -- while it is up *every* tile is revealed, which is visibly not
    ``expected``, so we keep clicking until the board collapses and the click takes.

    Re-sending the intended click is also the natural thing to record: a player who
    cannot see the lockout just keeps trying, and the locked frames are visually
    distinct (the mismatched pair is still face-up), so the data stays unambiguous.

    Returns (board, solved, dead); ``board`` is None if the click never landed.
    """
    x, y = geom.center(slot)
    for _ in range(_CLICK_RETRIES):
        frame, solved, dead = _drive(game, ActionInput(id=GameAction.ACTION6,
                                                       data={"x": x, "y": y}))
        rec.add(frame, _mouse_action(x, y))
        if dead:
            return None, False, True
        board = _read_board(frame, geom)
        if solved:
            return board, True, False
        revealed = {s for s in range(geom.num_tiles) if board[s] != HIDDEN_COLOR}
        if board[slot] != HIDDEN_COLOR and revealed == expected:
            return board, False, False
    return None, False, False


def _honest_play(spec: MmSpec, game, geom: _Geom, rec: _Recorder) -> bool:
    """Run the honest concentration policy on ``game`` (already at a level's initial
    state), appending every step to ``rec`` with the default ``"expert"`` phase.
    Returns True on an engine-verified solve, else False (lost / stuck).

    Tile colours enter ONLY through :func:`_read_board` on the rendered frame -- this
    never reads ``_slot_colors`` / ``_matched`` / ``_flipped``. Factored out of
    :func:`play_level` so the same honest loop can run from a fresh reset state after
    an exploration prefix, unchanged."""
    n = geom.num_tiles
    known: dict[int, int] = {}  # slot -> colour, every reveal ever seen
    matched: set[int] = set()

    if spec.peek:
        # Burn the once-per-level peek immediately and memorise the whole board:
        # everything after this is a guaranteed clear.
        frame, solved, dead = _drive(game, ActionInput(id=GameAction.ACTION5))
        rec.add(frame, _simple_action(int(GameAction.ACTION5.value)))
        if dead:
            return False
        for slot, color in enumerate(_read_board(frame, geom)):
            if color != HIDDEN_COLOR:
                known[slot] = color

    # Every level ends by clearing its last group, which the engine reports as a
    # solve; the bound is only a backstop against a non-terminating loop.
    for _ in range(4 * n + 8):
        flipped: list[int] = []
        for _ in range(spec.group_size):
            slot = _choose(flipped, known, matched, spec.group_size, n)
            expected = matched | set(flipped) | {slot}
            board, solved, dead = _click_until_landed(game, rec, slot, expected, geom)
            if board is None or dead:
                return False
            known[slot] = board[slot]
            flipped.append(slot)
            if solved:
                return True
        if len({known[s] for s in flipped}) == 1:
            matched.update(flipped)
    return False


def play_level(spec: MmSpec, seed: int, level_idx: int):
    """Play one level honestly. Returns (observations, actions) on an engine-verified
    solve, else (None, None) -- a lost or stuck seed is skipped, never recorded."""
    game = make_level(spec, seed, level_idx)
    geom = _geom(game)
    rec = _Recorder(game)
    if _honest_play(spec, game, geom, rec):
        return rec.observations, rec.actions
    return None, None


def _action_from_dict(rec: dict) -> Action:
    """Turn a recorded mm action dict back into an ``explore.Action`` so the base
    encoder (``_encode_step``) can re-emit it in the shared on-disk schema. Mouse
    clicks carry ``data={x, y}`` -> ``click_rc = (row=y, col=x)``; simple actions
    carry just an index."""
    if rec["type"] == "mouse":
        return Action(CLICK_ACTION, (int(rec["data"]["y"]), int(rec["data"]["x"])))
    return Action(int(rec["index"]))


class MmSolver(BaseSolver):
    """Route the honest mm concentration policy through :class:`BaseSolver` for the
    shared ``{"game_id", "levels": [...]}`` schema, save loop and CLI, WITHOUT
    changing how it plays.

    The mm family is NOT a plan-based optimal solver: it is a step-wise adaptive
    concentration policy over HIDDEN state, reading every tile colour from the
    rendered frame (:func:`_read_board`) and never from ``_slot_colors`` /
    ``_matched`` / ``_flipped``. Because clicks are a scarce budget, random
    exploration cannot be *interleaved* with honest play without stranding it -- so
    recovery here is the RESET paradigm, not a live replan: an epsilon-decayed
    exploration prefix of random legal clicks runs at level start, then ONE engine
    RESET restores the level's initial state and (bar the RESET's own step) the step
    budget, and the honest concentration policy plays that fresh state to
    a win. This mirrors a human who flails, burns steps, hits reset, and only THEN
    plays the level properly. ``supports_recovery = True`` /
    ``recovery_mode = "reset"``; with exploration off the prefix and reset are
    skipped and this is byte-identical to the old pure-honest replay.

    Episode assembly stays here (not the base ``record_level``, which is plan-based):
    this class re-encodes each (observation, action) into the base's step records so
    the corpus matches every other game (taken action flat at top level, ``phase`` =
    ``reset``/``explore``/``expert``, ``changed`` flag, and -- on ``expert`` steps
    -- ``optimal = [taken]``. There is no ORACLE for a hidden-state policy, but that
    is not a reason to emit no TARGET: the honest policy's own move is what should
    be imitated. See `_encode_actions`.)"""

    supports_recovery = True   # RESET-recovery: explore prefix, reset, honest play
    recovery_mode = "reset"    # solve_from is only valid at the initial state

    def __init__(self, spec: MmSpec, **kwargs) -> None:
        super().__init__(**kwargs)
        self.spec = spec
        self.game_id = spec.game_id  # instance attr shadows the base "" default

    # The honest policy is not expressible as an optimal plan from live state, so
    # the plan-based hooks are unused; solve_episode is overridden below instead.
    def make_game(self, seed: int):
        return make_level(self.spec, seed, 0)

    def solve_from(self, game, level_idx: int, seed: int) -> list:
        raise NotImplementedError(
            "mm plays a step-wise hidden-state policy; see play_level / solve_episode")

    def reset_level(self, game, level_idx: int, seed: int) -> np.ndarray:
        """The engine's OWN mid-level RESET, so the recorded reset frame is the one a
        replay produces.

        The base implementation re-clones the level and re-seats it, which restores
        the board but does NOT run the RESET action through ``step`` -- and in the mm
        games ``step`` decrements the step budget for every action, RESET included.
        So the base restore left the HUD's remaining-steps meter one step richer than
        the real engine, and every frame from the reset onwards differed from a
        replay by that pixel. This is exactly ``perform_action(RESET)`` for a
        mid-level reset: ``handle_reset``'s ``level_reset`` branch (``_action_count``
        is non-zero here -- a reset only ever follows exploration clicks -- so the
        engine would never take the ``full_reset`` branch, which would restart the
        episode at level 0 with a fresh palette), then the action itself.

        Returns the RESET's own frame, so the caller does not have to render again:
        the mm click marker fades on a RENDER counter, not a step counter, so an
        extra ``_render`` between two actions would age it out of step with a replay.
        """
        game.level_reset()
        frame, _solved, _dead = _drive(game, ActionInput(id=GameAction.RESET))
        return frame

    def solve_episode(self, seed: int, explore: bool = True):
        """Record every level for one seed, then re-encode each step through the base
        encoder. All-or-nothing (WIN-only corpus).

        Mirrors the base's episode wiring: the EpsilonSchedule + ExplorationPolicy are
        built ONCE here (only when ``explore and self.supports_recovery``) so the
        exploration arc spans the whole episode, and each level is recorded via
        :meth:`_record_level`. With exploration off, ``_record_level`` collapses to
        the honest :func:`play_level`, so optimal replay is unchanged."""
        do_explore = explore and self.supports_recovery
        schedule = (EpsilonSchedule(self.rng, center=self._explore_center,
                                    jitter=self._explore_jitter)
                    if do_explore else None)
        exploration = (ExplorationPolicy(self.available_actions(self.make_game(seed)),
                                         self.rng)
                       if do_explore else None)
        levels: list[dict] = []
        for level_idx in range(num_levels(self.spec)):
            try:
                obs, raw, phases = self._record_level(
                    seed, level_idx, schedule=schedule, exploration=exploration)
            except Exception:  # noqa: BLE001 -- a bad seed just fails the episode
                return False, levels
            if obs is None:
                return False, levels
            levels.append({"level_id": level_idx, "observations": obs,
                           "actions": self._encode_actions(obs, raw, phases)})
        return True, levels

    def _record_level(self, seed: int, level_idx: int, *,
                      schedule: EpsilonSchedule | None = None,
                      exploration: ExplorationPolicy | None = None):
        """Record one level: an epsilon-decayed exploration prefix of random legal
        clicks, then ONE RESET (restoring the initial state + full budget), then the
        honest concentration play to a win. Returns ``(observations, actions, phases)``
        on solve, else ``(None, None, None)``.

        With ``schedule``/``exploration`` omitted (exploration off) the prefix and
        reset are skipped and this is exactly :func:`play_level` -- honest play from
        the make_level state, untouched.

        The prefix perturbs the board (flips tiles / spends steps); a RESET-recovery
        then restores the initial state so honest play always starts with the full
        budget -- exactly why exploration cannot brick the level. Both the layout and
        the rotation are pure functions of (seed, level), so the reset restores the
        SAME board rather than re-randomising it; honest play still starts from an
        empty memory, which is merely conservative, never dishonest."""
        game = make_level(self.spec, seed, level_idx)
        geom = _geom(game)
        rec = _Recorder(game)  # observations[0] + RESET at index 0 (phase "reset")

        do_explore = schedule is not None and exploration is not None
        perturbed = False
        resets = 0
        if do_explore:
            # Epsilon-decayed opening: keep taking random legal clicks while the
            # (episode-wide, decaying) schedule says explore; stop at the first
            # exploit decision, or early on an accidental solve / death.
            for _ in range(_STEP_GUARD):
                explore_now = schedule.explore()
                schedule.advance()
                if not explore_now:
                    break
                act = exploration.action(rec.last_frame)
                frame, solved, dead = _drive(game, self._to_action_input(act))
                rec.add(frame, self._explore_dict(act), phase="explore")
                perturbed = True
                if solved:  # random clicks cleared the board -- a valid honest win
                    return rec.observations, rec.actions, rec.phases
                if dead:  # budget exhausted mid-prefix -- reset restores it below
                    break

        # RESET-recovery: restore the level's initial state before honest play.
        if perturbed:
            if resets >= self.max_resets:
                return None, None, None
            frame = self.reset_level(game, level_idx, seed)
            geom = _geom(game)  # re-read: reset re-seats the level (and re-draws k)
            rec.add(frame, {"type": "simple", "index": RESET_ACTION}, phase="reset")
            resets += 1
            if schedule is not None:
                schedule.advance()  # the reset is a taken step in the episode arc

        # Honest concentration play from the (fresh) initial state to a verified win.
        before = len(rec.actions)
        solved = _honest_play(self.spec, game, geom, rec)
        if schedule is not None:  # keep the episode-wide decay honest across levels
            schedule.advance(len(rec.actions) - before)
        if not solved:
            return None, None, None
        return rec.observations, rec.actions, rec.phases

    def _to_action_input(self, act: Action) -> ActionInput:
        """An ``explore.Action`` -> the engine ``ActionInput`` (mm clicks are
        ACTION6 with ``data={x, y}``; simple actions map by id)."""
        if act.is_click:
            x, y = act.click_xy
            return ActionInput(id=GameAction.ACTION6, data={"x": int(x), "y": int(y)})
        return ActionInput(id=_ID_TO_GAMEACTION[act.action_id])

    @staticmethod
    def _explore_dict(act: Action) -> dict:
        """The recorded dict for an exploration-prefix action."""
        if act.is_click:
            x, y = act.click_xy
            return _mouse_action(int(x), int(y))
        return _simple_action(int(act.action_id))

    def _encode_actions(self, obs: list, raw: list[dict],
                        phases: list[str]) -> list[dict]:
        """Re-emit the raw action dicts in the base schema: ``actions[0]`` is the
        RESET step; each later step is the taken click/peek/reset tagged by
        ``phases[i]`` (``explore``/``reset``/``expert``) and a ``changed`` flag
        from the frames (``obs[i-1]`` -> ``obs[i]``).

        TARGETS. An ``expert`` step is supervised with its OWN taken action.
        ``train_policy`` v2 supervises ``optimal`` ONLY -- the taken action is
        context, never a target -- so emitting ``optimal=None`` here (as this did,
        reasoning that a hidden-state policy has no oracle) left mm01..mm05 with
        100% unlabelled expert steps: present in every context, contributing
        nothing to the loss, silently. Having no ORACLE is not the same as having
        no TARGET: the honest concentration policy's own move is exactly what the
        policy should learn to imitate, and it is what v1 trained on.

        ``explore`` steps stay unsupervised -- they are deliberately random, and
        cloning them would teach flailing."""
        actions = [self._reset_step()]
        for i in range(1, len(raw)):
            taken = _action_from_dict(raw[i])
            changed = obs[i] != obs[i - 1]
            optimal = [taken] if phases[i] == "expert" else None
            actions.append(self._encode_step(taken, optimal=optimal,
                                             phase=phases[i], changed=changed))
        return actions

    @classmethod
    def main(cls, spec: MmSpec, argv=None) -> int:  # type: ignore[override]
        """Spec-bound CLI over the base harness (plus mm's ``--verify`` battery)."""
        parser = cls.build_argparser()
        parser.description = f"Generate {spec.game_id} training data."
        parser.add_argument("--verify", action="store_true",
                            help="Run the correctness battery instead of generating.")
        args = parser.parse_args(argv)
        if args.verify:
            return 0 if verify(spec) else 1
        solver = cls(spec, rng=random.Random(args.seed),
                     burst_prob=args.noise, burst_mean=args.burst)
        out = args.out or (Path("data/training_multi_level") / spec.game_id)
        return solver.run(args.episodes, out, start_seed=args.start_seed,
                          max_attempts=args.max_attempts,
                          progress_every=args.progress_every,
                          explore=not args.no_exploration)


# ── Verification ──────────────────────────────────────────────────────────────
def verify(spec: MmSpec, seeds=(0, 1, 2, 7, 42, 99), sweep: int = 40) -> bool:
    """Prove the four things this generator's correctness rests on:

    1. ``_read_board`` agrees with the engine's ground truth on every revealed tile
       (the solver's only perception channel actually perceives);
    2. the solver never uses privileged state -- asserted structurally by (1) plus
       the fact that its knowledge only ever grows from ``_read_board``;
    3. ``make_level`` is a pure function of (seed, level), episode colours are shared
       across levels, and different seeds give different boards;
    4. every level of every swept seed is solved inside the engine's step budget.

    Both halves of the rotation boundary are covered, because both are silent when
    wrong: (1) is run at every level's live rotation and each click is checked to land
    on the tile it aimed at (the screen->game click map), while the read is checked
    against ground truth by slot (the game->screen frame map). (3) additionally pins
    the rotation itself as a function of (seed, level).
    """
    ok = True
    n_lv = num_levels(spec)
    print(f"=== verify {spec.game_id} (group={spec.group_size}, peek={spec.peek}, "
          f"levels={n_lv}) ===")

    print("frame reading vs engine ground truth, and clicks landing on target:")
    mismatches = 0
    checked = 0
    misclicks = 0
    for seed in seeds:
        for level_idx in range(n_lv):
            n_tiles = _geom(make_level(spec, seed, level_idx)).num_tiles
            # One fresh game per tile: the first click of a level can never be
            # swallowed by a lock, so "exactly this slot is revealed" is an exact
            # test of the screen click coordinate.
            for slot in range(n_tiles):
                game = make_level(spec, seed, level_idx)
                geom = _geom(game)
                truth = list(game._slot_colors)
                x, y = geom.center(slot)
                frame, _solved, _dead = _drive(
                    game, ActionInput(id=GameAction.ACTION6, data={"x": x, "y": y}))
                board = _read_board(frame, geom)
                revealed = {s for s in range(geom.num_tiles)
                            if board[s] != HIDDEN_COLOR}
                if revealed != {slot}:
                    misclicks += 1
                for s in revealed:
                    checked += 1
                    if board[s] != truth[s]:
                        mismatches += 1
    print(f"  {checked} revealed-tile reads, {mismatches} mismatches -> "
          f"{'OK' if mismatches == 0 else 'FAIL'}")
    print(f"  clicks landing off-target: {misclicks} -> "
          f"{'OK' if misclicks == 0 else 'FAIL'}")
    ok &= mismatches == 0 and misclicks == 0

    print("rotation is a pure function of (seed, level):")
    rots = {}
    rot_pure = True
    for seed in seeds:
        for level_idx in range(n_lv):
            ks = {make_level(spec, seed, level_idx).rotation_k for _ in range(3)}
            rot_pure &= len(ks) == 1
            rots[(seed, level_idx)] = ks.pop() if len(ks) == 1 else None
    drawn = sorted({k for k in rots.values() if k is not None})
    print(f"  stable across constructions: {'OK' if rot_pure else 'FAIL'}"
          f"   orientations seen: {drawn}")
    ok &= rot_pure

    print("make_level is a pure function of (seed, level):")
    pure = True
    for seed in seeds:
        for level_idx in range(n_lv):
            a = _render(make_level(spec, seed, level_idx)).tolist()
            b = _render(make_level(spec, seed, level_idx)).tolist()
            pure &= a == b
    print(f"  {'OK' if pure else 'FAIL'}")
    ok &= pure

    print("episode colour scheme shared across levels, boards differ across seeds:")
    schemes = {}
    for seed in seeds:
        per_level = [make_level(spec, seed, i)._color_scheme for i in range(n_lv)]
        same = all(s == per_level[0] for s in per_level)
        schemes[seed] = per_level[0]
        if not same:
            print(f"  seed {seed}: colour scheme NOT shared across levels -> FAIL")
            ok = False
    boards = {seed: [tuple(make_level(spec, seed, i)._slot_colors) for i in range(n_lv)]
              for seed in seeds}
    distinct = len({tuple(v) for v in boards.values()})
    print(f"  colour scheme shared across levels: OK for all {len(seeds)} seeds")
    print(f"  distinct boards: {distinct}/{len(seeds)} seeds -> "
          f"{'OK' if distinct == len(seeds) else 'FAIL'}")
    ok &= distinct == len(seeds)

    print(f"honest solve sweep ({sweep} seeds x {n_lv} levels):")
    worst = {}
    fails = 0
    for seed in range(sweep):
        for level_idx in range(n_lv):
            obs, acts = play_level(spec, seed, level_idx)
            if obs is None:
                fails += 1
                print(f"  seed {seed} level {level_idx}: UNSOLVED")
                continue
            # actions[0] is RESET; the rest are real engine steps.
            cost = len(acts) - 1
            worst[level_idx] = max(worst.get(level_idx, 0), cost)
    for level_idx in sorted(worst):
        print(f"  level {level_idx}: worst-case {worst[level_idx]} actions")
    print(f"  unsolved: {fails}/{sweep * n_lv} -> {'OK' if fails == 0 else 'FAIL'}")
    ok &= fails == 0

    print(f"=== {spec.game_id}: {'ALL OK' if ok else 'FAILURES'} ===")
    return ok
