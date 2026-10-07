"""BaseSolver -- the shared harness behind every ``generate_<id>_training.py``.

Every generator in this directory re-implements the same skeleton: solve each
level with a game-specific expert, replay the plan against the rendering engine
to capture frames, bundle the levels into one episode JSON, and write one file
per WIN seed under a ``main()`` that filters seeds and prints progress. Only the
`solve()` search and a couple of engine quirks are actually game-specific; the
rest was copy-pasted ~58 times (see the module-level analysis). ``common/mm.py``
already proved the factoring for the mm01..mm05 family -- ``BaseSolver`` is that
generalised to every game.

What the base owns
------------------
* the episode/level record+replay loop (`record_level`, `solve_episode`),
* the canonical output schema and the WIN-filter CLI (`run`, `main`),
* action encoding (`_encode_step`) -- the single source of truth for the on-disk
  step format (the flat taken action plus the `optimal`/`phase`/`changed` labels),
* the default engine-driving template (`drive`), and
* the optional **exploration prefix**, wired to `utils.explore`.

What a subclass supplies
------------------------
Minimum: `game_id`, `make_game(seed)`, and `solve_from(game, level_idx, seed)`
returning an optimal plan FROM THE GAME'S CURRENT STATE (a list of `GameAction` /
ints for simple actions, or `explore.Action` for clicks). Override `num_levels`,
`set_level`, `render`, `available_actions`, or `drive` for engines that don't
match the native ``ARCBaseGame`` template; set `supports_recovery = True` to
enable the exploration prefix + bursts, and override `optimal_set_from` to emit
the full optimal tie set as the training target (see `record_level`).

The on-disk schema is exactly what the encoder/dynamics stack expects and what
all existing generators already emit::

    {"game_id": "<id>",
     "levels": [{"level_id": 0,
                 "observations": [[[c, ...], ...], ...],   # [T, 64, 64] palette idx
                 "actions":      [a_0, ..., a_{T-1}]},      # length T, a_0 == RESET
                ...]}
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
import random
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

# Repo root (parent of solvers/) -- where arcengine, games/, utils/ resolve.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Mute the legacy-gym banner the same way solver.py does, so generators that
# pull in procgen/minigrid adapters stay quiet.
try:
    with open(os.devnull, "w") as _devnull, contextlib.redirect_stderr(_devnull):
        import gym  # noqa: F401
except Exception:  # noqa: BLE001
    pass
warnings.filterwarnings("ignore", message=r".*old step API.*")
warnings.filterwarnings("ignore", message=r".*np\.bool8.*")

import numpy as np  # noqa: E402

from arcengine import ActionInput, GameAction, GameState  # noqa: E402
from utils.explore import (  # noqa: E402
    Action, ExplorationPolicy, ExplorationPrefix, CLICK_ACTION, RESET_ACTION)
from adapters.base import BaseAdapter  # noqa: E402  (burst-undo snapshot dispatch)
from utils.rotation import (  # noqa: E402
    inverse_remap_action_full, remap_click)

_STEP_GUARD = 4000

# Frames are 64x64 and clicks are addressed in that space -- NOT ``camera.width``,
# which is the camera's GRID size. Same constant `AugmentedGame` de-rotates with.
_AUGMENT_FRAME = 64

# GameAction is not value-indexable (``GameAction(4)`` raises), so map ids -> the
# enum member explicitly, exactly as solver.py's _IDX_TO_GAMEACTION does.
_ID_TO_GAMEACTION = {int(a.value): a for a in GameAction}


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------
@dataclass
class DriveResult:
    """Outcome of driving one action against the engine.

    ``frames`` is the WHOLE animation one action produced, not just its settled
    result. This mirrors the real interface: ``ARCBaseGame.perform_action``
    renders after every internal ``step()`` and returns ``FrameData.frame`` as a
    LIST of grids, so an agent submits one action and receives the entire
    sequence without acting in between. For any game whose actions animate -- a
    multi-column program run, a slide, a physics settle -- that sequence is
    where the mechanic is actually visible, and keeping only the last frame
    makes the corpus strictly less informative than the environment it is meant
    to teach. (TN36 is the extreme case: a failed program run animates the block
    across 8 frames and then snaps it back to its checkpoint, so the settled
    frame differs from the pre-action one by a single timer pixel -- the
    experiment is invisible unless the animation is kept.)

    A bare ``(H,W)`` frame is promoted to ``(1,H,W)``, so the many ``drive``
    overrides that return one settled frame keep working unchanged.
    """

    frames: np.ndarray         # (k,H,W) palette frames for this action, k >= 1
    solved: bool               # this action solved the current level
    dead: bool = False         # the action ended the episode in GAME_OVER

    def __post_init__(self) -> None:
        arr = np.asarray(self.frames)
        if arr.ndim == 2:                      # single settled frame
            arr = arr[None]
        elif arr.ndim != 3:
            raise ValueError(
                f"DriveResult.frames must be (H,W) or (k,H,W), got {arr.shape}")
        self.frames = arr

    @property
    def frame(self) -> np.ndarray:
        """The SETTLED frame -- the last of the animation. This is the state the
        NEXT action is decided from, so `changed` / `prev` tracking uses it."""
        return self.frames[-1]


def _as_action(a) -> Action:
    """Coerce a plan entry into an `explore.Action`.

    Accepts an `Action` (returned as-is), a `GameAction` or int (a simple
    action), or an `ActionInput` (simple, or ACTION6 with ``data={x, y}`` ->
    a click). Clicks store the target as ``click_rc = (y, x)`` so that
    ``Action.click_xy == (x, y)`` matches the engine's mouse convention.
    """
    if isinstance(a, Action):
        return a
    if isinstance(a, GameAction):
        return Action(int(a.value))
    if isinstance(a, int):
        return Action(a)
    if isinstance(a, ActionInput):
        if int(a.id.value) == CLICK_ACTION and a.data:
            return Action(CLICK_ACTION, (int(a.data["y"]), int(a.data["x"])))
        return Action(int(a.id.value))
    raise TypeError(f"unsupported plan action: {a!r}")


# ---------------------------------------------------------------------------
# BaseSolver
# ---------------------------------------------------------------------------
class BaseSolver:
    """Base class for the per-game training-data generators. Subclass, set
    ``game_id``, and implement ``make_game`` + ``solve_from``; call
    ``YourSolver.main()`` from ``if __name__ == '__main__'``. Set
    ``supports_recovery = True`` once ``solve_from`` is verified to plan from any
    live state, which switches on the exploration prefix + perturbation bursts."""

    game_id: str = ""
    step_guard: int = _STEP_GUARD

    #: Whether ``solve_from`` can plan from an ARBITRARY reachable state, not just
    #: a level's initial one. This is the gate for the exploration
    #: prefix: after an exploratory action perturbs the state, the base re-invokes
    #: ``solve_from`` on the LIVE game to recover, which is only correct if the
    #: solver reads that live state (and the game isn't bricked by the detour).
    #: Left False so a not-yet-audited solver degrades to pure optimal replay even
    #: with exploration switched on; flip to True once ``solve_from`` is verified
    #: to recover from off-plan states (see DESIGN.md's ``solve_from`` refactor).
    supports_recovery: bool = False

    #: HOW recovery works when exploration bricks the state (see `record_level`):
    #:   "replan" -- ``solve_from`` reads live state and re-plans from wherever the
    #:              detour left us; a RESET is only used if the state is dead or
    #:              genuinely unsolvable. For reversible games (the natural True).
    #:   "reset"  -- ``solve_from`` is only valid from the level's INITIAL state
    #:              (cached/precomputed plan), so recovery is: explore the prefix,
    #:              then apply RESET to restore the initial state, then replay the
    #:              plan. This makes even IRREVERSIBLE games recoverable -- exactly
    #:              like a human who flails, dies/gets stuck, hits reset, and only
    #:              THEN solves. RESET is recorded as a real action in the buffer.
    recovery_mode: str = "replan"
    #: Max RESET-recoveries recorded per level before the episode is abandoned.
    max_resets: int = 3
    #: Default for ``--noise``, so a game whose post-burst solvability check is
    #: ruinously expensive can ship with bursting OFF without every caller having
    #: to remember the flag. Overridden by an explicit ``--noise`` on the command
    #: line. Set it to 0.0 only with a measurement to point at: bursts are where
    #: the recovery data comes from, and a game with none of it teaches the policy
    #: nothing about getting back on track.
    default_noise: float = 0.1

    #: STOCHASTIC OPTIMAL PLAY (DESIGN.md): at each expert step, sample the taken
    #: action uniformly among the equally-optimal actions rather than always
    #: replaying one canonical BFS route -- so a given board yields many distinct,
    #: still-perfectly-optimal trajectories across episodes. The optimal SET comes
    #: from `optimal_set_from` if the subclass provides it; otherwise, for a
    #: "replan" solver, the base DERIVES it by one-step lookahead over the simple
    #: actions (see `_optimal_set`). Where neither is available (cached-plan /
    #: click-optimum / non-copyable engine) it degrades gracefully to the single
    #: canonical path. Set False to force the canonical route (or where the probe
    #: is too costly).
    stochastic_optimal: bool = True

    def __init__(self, *, rng: random.Random | None = None,
                 explore_center: int = 10, explore_jitter: int = 5,
                 burst_prob: float = 0.0, burst_mean: int = 3) -> None:
        self.rng = rng or random.Random()
        self._explore_center = explore_center      # exploration-prefix length centre
        self._explore_jitter = explore_jitter      # +/- jitter on that length
        self._burst_prob = burst_prob              # P(start a burst) per optimal step
        self._burst_mean = burst_mean              # mean burst length (steps)

    # ── subclass API: the ONE required hook ──────────────────────────────────
    def make_game(self, seed: int):
        """Return a fresh game instance for ``seed`` (positioned at level 0)."""
        raise NotImplementedError

    def solve_from(self, game, level_idx: int, seed: int) -> list:
        """Return an optimal action plan **from the current state of ``game``**.

        This is the single expression of the game's solver logic (DESIGN.md's
        ``solve_from`` prerequisite). Everything else derives from it:

          * plain replay follows the plan returned at the level's initial state;
          * the exploration prefix follows the plan too, but each time it takes
            an exploratory action instead it discards the stale plan and calls
            ``solve_from`` again on the perturbed live game.

        So the solver must read the LIVE ``game`` (its grid / engine state), not
        rebuild the board from ``(seed, level_idx)`` -- ``level_idx`` and ``seed``
        are passed only for seed-derived constants and logging. Return a list of
        ``GameAction`` / ints (simple actions) and/or ``explore.Action`` (clicks);
        an empty list means "no progress possible from here" and fails the episode.
        """
        raise NotImplementedError

    # ── subclass API: OVERRIDABLE (native ARCBaseGame defaults) ───────────────
    def num_levels(self, game) -> int:
        return len(game._levels)

    def set_level(self, game, level_idx: int) -> None:
        game.set_level(level_idx)
        # Clear the engine's pending level-advance flag. When a level is solved
        # the engine raises ``_next_level`` and the native ``drive`` stops on it;
        # if it is still set when the NEXT level begins, the first action of that
        # level is mis-read as an instant solve (the whole level collapses to a
        # bogus 2-frame record). The engine's own ``_really_set_next_level``
        # clears it; mirror that here so every native-drive multi-level game
        # records real trajectories without needing a per-subclass workaround.
        if getattr(game, "_next_level", False):
            game._next_level = False

    def reset_level(self, game, level_idx: int, seed: int) -> None:
        """Restore ``level_idx`` to its INITIAL state -- the engine's RESET
        action, used by `record_level` to recover after exploration bricks or
        kills the level.

        The real engine RESET (``perform_action(RESET)`` -> ``handle_reset`` ->
        ``level_reset``) re-clones the current level from ``_clean_levels``; this
        mirrors that (re-clone the clean template to undo any in-place mutation)
        and then re-runs the solver's ``set_level`` so per-(seed, level)
        reseeding is re-applied and the restored frame matches ``observations[0]``
        exactly. For adapter games (no ``_clean_levels``) it falls back to
        ``set_level``, whose adapter reset re-seats the level. Override for games
        whose ``set_level`` ADVANCES in place (``_really_set_next_level``) rather
        than restores -- there, rebuild a fresh instance seated at ``level_idx``.

        It also CLEARS ``GAME_OVER``, as the engine's ``level_reset`` does (it ends
        with ``self._state = GameState.NOT_FINISHED``). Without that, a RESET taken
        to recover from a death restores the board but leaves the game dead, so the
        very next `drive` reports ``dead`` again -- the recorder then burns through
        ``max_resets`` and discards the episode, i.e. death recovery silently never
        worked for the lethal games. Only GAME_OVER is cleared: a WIN is left alone,
        since the level is over either way and the caller decides."""
        clean = getattr(game, "_clean_levels", None)
        levels = getattr(game, "_levels", None)
        if clean is not None and levels is not None and 0 <= level_idx < len(clean):
            try:
                levels[level_idx] = clean[level_idx].clone()
            except Exception:                    # noqa: BLE001 -- fall back to set_level
                pass
        self.set_level(game, level_idx)
        if getattr(game, "_state", None) == GameState.GAME_OVER:
            game._state = GameState.NOT_FINISHED

    # ── burst UNDO: snapshot / restore the live game (recorder-internal) ──────
    # A perturbation burst that strands the game in an unsolvable / dead state is
    # ROLLED BACK rather than recovered from: the game state is restored to the
    # pre-burst snapshot and the tentatively-recorded burst frames are dropped
    # from the buffer, so the burst never happened. UNDO is NOT a real competition
    # action, so it must leave NO trace in the trajectory (no ACTION7, no undo
    # frames) -- see `record_level`. Reversible bursts are kept as recovery data.
    def _game_snapshot(self, game):
        """Capture the full mutable game state. Adapter games use their BaseAdapter
        ``_snapshot`` ([[base-adapter]]); native arcengine games are captured by
        deep-copying the WHOLE game object (bursts are rare, so the cost is
        amortised).

        The clone MUST be rooted at ``game`` (``deepcopy(game)``), NOT at
        ``game.__dict__``: several native games hang sub-objects that back-reference
        the game itself (e.g. sb26's camera HUD interface reads energy off the game),
        and deep-copying ``__dict__`` in isolation copies that back-reference into an
        ORPHAN game clone -- so a later render/step dereferences a half-built object
        and raises ``AttributeError``. Rooting the copy at ``game`` keeps every
        internal back-reference pointing inside the single clone (via deepcopy's
        memo), and `_game_restore` re-points them at the live game.

        The discriminator MUST be ``isinstance(game, BaseAdapter)`` -- NOT
        ``hasattr(game, "_snapshot")`` -- because several native games (sokoban,
        aquarium_sorter, ice_rink_stars, frog_crossing, badge_placement, ...) define
        their own unrelated ``_snapshot`` with no matching ``_restore``; duck-typing
        misroutes them and corrupts the restore."""
        if isinstance(game, BaseAdapter):
            return ("adapter", game._snapshot())
        return ("native", copy.deepcopy(game))

    def _game_restore(self, game, snap) -> None:
        """Restore a snapshot produced by ``_game_snapshot`` IN PLACE (the same
        game object keeps its identity, so the driving loop's reference stays
        valid)."""
        kind, data = snap
        if kind == "adapter":
            game._restore(data)
            return
        # ``data`` is a pristine whole-game clone. Rebuild ``game``'s __dict__ from
        # it, deep-copying each VALUE with a memo that maps the clone back to the
        # LIVE ``game`` -- so any sub-object that back-references its game (sb26's
        # camera HUD interface, etc.) re-points to ``game``, not the orphan clone.
        # (Memo the clone ROOT, not each value, and deepcopy would short-circuit to
        # ``game`` and copy nothing; hence the per-value loop.)
        memo = {id(data): game}
        new_dict = {k: copy.deepcopy(v, memo) for k, v in data.__dict__.items()}
        game.__dict__.clear()
        game.__dict__.update(new_dict)

    def _burst_recovery_wins(self, game, plan: list) -> bool:
        """Would replaying ``plan`` from the LIVE (post-burst) state actually WIN?

        ``solve_from`` returning a non-empty plan only proves the *model* believes
        the post-burst state is solvable. A model false-positive -- a confident plan
        the real engine will NOT win -- would otherwise be committed at burst-close
        and only exposed many frames later, when the doomed plan finally strands us
        and RESET is the sole recourse (the pre-burst anchor is long gone). So DRIVE
        the recovery on a throwaway copy of the live state and report whether it
        reaches a solve. Restores the live state afterwards, so this leaves NO trace
        -- neither on the game nor on the recording buffer. Bursts are rare, so the
        one extra replay per burst is amortised."""
        snap = self._game_snapshot(game)
        try:
            for act in plan:
                res = self.drive(game, act)
                if res.solved:
                    return True
                if res.dead:
                    return False
            return False                       # plan ran out without a win
        except Exception:                      # noqa: BLE001 -- a crash is not a win
            return False
        finally:
            self._game_restore(game, snap)

    # ── data-generation: lift the per-game step limit ────────────────────────
    def _step_counter_probe(self, game):
        """A zero-arg callable giving the game's live steps-remaining, or None if
        the game exposes no step budget. Re-reads the live attribute each call so
        it survives the counter being reset at a level/RESET boundary."""
        for attr in ("_step_counter", "step_counter"):
            sc = getattr(game, attr, None)
            if sc is not None and hasattr(sc, "steps_remaining"):
                return lambda a=attr: getattr(getattr(game, a, None),
                                              "steps_remaining", None)
        for attr in ("_steps_remaining", "steps_remaining"):
            if hasattr(game, attr):
                return lambda a=attr: getattr(game, a, None)
        cam = getattr(game, "camera", None)
        for iattr in ("interfaces", "_interfaces"):
            for c in (getattr(cam, iattr, None) or []):
                if hasattr(c, "steps_remaining"):
                    return lambda c=c: getattr(c, "steps_remaining", None)
        return None

    def _lift_step_limit(self, game) -> None:
        """Stop the game's step-budget exhaustion from ending the episode during
        DATA GENERATION -- exploration overhead must never discard a trajectory.

        The step counter still ticks down normally (so the HUD bar depletes as
        usual and simply floors at zero); we only suppress the *step-exhaustion*
        loss by patching ``game.lose()`` to be a no-op WHEN the counter is at zero,
        leaving every real loss (hazards, wrong moves -- which fire while steps
        remain) fully intact. Idempotent per game; always on during generation."""
        if getattr(game, "_gen_step_death_off", False):
            return
        probe = self._step_counter_probe(game)
        if probe is None:
            return
        # Native arcengine games end the episode on step exhaustion via
        # ``game.lose()``; this patch neutralises THAT path. The external-env
        # adapters (BaseAdapter: minigrid/gymgw/procgen/ale/puzzlescript) have no
        # ``lose()`` -- they flip ``_state`` to GAME_OVER directly in ``_apply``,
        # and the record loop already absorbs that via RESET recovery. So with no
        # ``lose()`` to patch there is nothing to lift here; return (don't crash).
        orig_lose = getattr(game, "lose", None)
        if not callable(orig_lose):
            return
        def _lose_unless_out_of_steps():
            try:
                remaining = probe()
                if remaining is not None and remaining <= 0:
                    return                 # step exhaustion -> keep playing
            except Exception:              # noqa: BLE001 -- never let this mask a real lose
                pass
            orig_lose()
        game.lose = _lose_unless_out_of_steps
        game._gen_step_death_off = True

    def render(self, game) -> np.ndarray:
        """Current 64x64 palette frame WITHOUT stepping the game.

        Exactly what the agent sees -- i.e. AT the level's rotation. Frames are never
        un-rotated, here or anywhere else: a solver that plans off pixels plans on the
        board as shown, and therefore produces SCREEN actions (see
        `plans_in_screen_space`).
        """
        return np.asarray(game.camera.render(game.current_level.get_sprites()))

    def available_actions(self, game) -> list[int]:
        """Valid action ids for the exploration policy. Best-effort: reads a
        common attribute, else the four movement keys."""
        for attr in ("available_actions", "_available_actions"):
            av = getattr(game, attr, None)
            if av:
                return [int(a) for a in av]
        return [1, 2, 3, 4]

    def optimal_set_from(self, game, level_idx: int, seed: int) -> list | None:
        """The set of equally-optimal next actions at the current state, used as
        the *target* label for policy training (DESIGN.md's optimal-action sets).

        Default ``None`` -> `record_level` uses the head of the already-computed
        ``solve_from`` plan (a single-element target), so no extra solve is done
        per step. A solver with a distance oracle can override this to return
        every ``a`` with ``dist(succ(a)) == dist - 1``, turning the target loss
        into a soft target over the whole tie set rather than one arbitrary
        tie-break."""
        return None

    def _copy_game(self, game):
        """A restorable copy of ``game`` for the one-step optimal-set probe, or
        ``None`` to disable the probe (the default).

        The probe derives the optimal action set generically but costs one game
        copy + `solve_from` PER candidate action PER step, so with a full
        ``deepcopy`` it makes a fast BFS game ~|A|x slower -- too slow to be a
        silent default. So it is OFF unless a subclass opts in by returning a
        (ideally cheap) snapshot here, e.g. ``return copy.deepcopy(game)``. The
        FAST way to get stochastic optimal is to override `optimal_set_from`
        instead: a search solver already knows its optimal set (the distance
        field), so returning it is O(1) and needs no copy."""
        return None

    def _optimal_set(self, game, level_idx: int, seed: int, plan: list) -> list:
        """The equally-optimal next actions at the current state, for stochastic
        optimal sampling. Order of preference:

          1. the subclass's `optimal_set_from` (a distance oracle knows the set);
          2. else, for a ``replan`` solver, DERIVE it by one-step lookahead -- an
             action is optimal iff taking it leaves a `solve_from` plan exactly one
             step shorter -- enumerating the SIMPLE actions (clicks aren't
             enumerable, so click optima need `optimal_set_from`);
          3. else the single canonical head ``plan[:1]`` (graceful degradation).

        The probe drives each candidate on a `_copy_game` snapshot, so it never
        disturbs the live game."""
        given = self.optimal_set_from(game, level_idx, seed)
        if given is not None:
            given = [_as_action(a) for a in given]
            return given or plan[:1]
        if (not self.stochastic_optimal or self.recovery_mode != "replan"
                or len(plan) < 2 or plan[0].is_click):
            return plan[:1]
        head = plan[0]
        target = len(plan) - 1                     # steps-to-go after an optimal move
        opt = [head]
        for aid in self.available_actions(game):
            if aid in (RESET_ACTION, CLICK_ACTION) or aid == head.action_id:
                continue
            g2 = self._copy_game(game)
            if g2 is None:
                return plan[:1]                    # can't probe -> canonical path
            try:
                res = self.drive(g2, Action(int(aid)))
                if res.dead:
                    continue
                if res.solved:
                    if target == 0:
                        opt.append(Action(int(aid)))
                    continue
                p2 = [_as_action(x) for x in self.solve_from(g2, level_idx, seed)]
                if p2 and len(p2) == target:
                    opt.append(Action(int(aid)))
            except Exception:                      # noqa: BLE001 -- skip a bad candidate
                continue
        return opt

    def drive(self, game, action: Action) -> DriveResult:
        """Perform one action against the real (rendering) engine and return
        EVERY frame it rendered + solve/death flags. Stops the instant a level
        solve is queued (``_next_level``) so the captured frames end on the
        CLEAN solved board of the current level, not the next one. Override for
        non-native engines (adapters, procgen, ...).

        This is deliberately the same loop as the engine's own ``perform_action``
        (render after each internal ``step()``, keep them all) -- see
        `DriveResult`. Overrides that return a single settled frame stay valid;
        they simply contribute a one-frame animation."""
        self._note_rotation(game)
        core = self._to_core(action)       # no-op unless the solver planned on-screen
        if core.is_click:
            ai = ActionInput(id=GameAction.ACTION6,
                             data={"x": core.click_xy[0], "y": core.click_xy[1]})
        else:
            ai = ActionInput(id=_ID_TO_GAMEACTION[core.action_id])
        game._full_reset = False
        game._set_action(ai)
        frames: list = []
        guard = 0
        while not game.is_action_complete():
            guard += 1
            if guard > self.step_guard or game._next_level:
                break
            game.step()
            frames.append(game.camera.render(game.current_level.get_sprites()))
        if not frames:                         # no-op action rendered nothing new
            frames.append(game.camera.render(game.current_level.get_sprites()))
        solved = game._next_level or game._state == GameState.WIN
        dead = game._state == GameState.GAME_OVER
        return DriveResult(np.asarray(frames), solved, dead)

    # ── rotation: ONE convention for every solver ────────────────────────────
    #
    #   a solver PLANS in the core game's UPRIGHT space; the base SUBMITS and RECORDS
    #   the screen action.
    #
    # `drive` submits the upright action through ``game._set_action``, which bypasses
    # `AugmentedGame.perform_action` and so reaches the core game unrotated -- the game
    # moves exactly as the plan intended. The frames come back from ``camera.render``,
    # which the wrapper hooks, so they are ROTATED; the recorded action therefore has to
    # be the SCREEN action, or the demo pairs a turned frame with an upright label and
    # at evaluation the wrapper de-rotates the policy's press into the wrong move.
    #
    # A solver that reads the FRAME instead sees the board as shown and solves it in
    # that space; it sets `plans_in_screen_space` and no conversion happens at all.
    #
    # THERE IS NO WAY TO GET AN UN-ROTATED FRAME, deliberately. Frames are only ever
    # rendered at the level's rotation -- the same thing solver_client.py shows a human.
    # An un-rotate helper would inevitably get used to record with, and the demo would
    # then hold a frame nobody ever saw. Read the sprites, or read the frame as shown.
    _augment_k: int = 0

    #: Set True by a solver that PLANS ON THE RENDERED FRAME. It sees the board at the
    #: level's rotation and solves it there, so the plan it returns is already in SCREEN
    #: space: no conversion when recording, and `drive` de-rotates before handing the
    #: action to the core game (the same thing `AugmentedGame.perform_action` does for a
    #: real agent). Solvers that read SPRITES instead see the core game's upright state,
    #: so they leave this False and the base converts their plan on the way out.
    plans_in_screen_space: bool = False

    def _note_rotation(self, game) -> None:
        """Remember the orientation the frames being recorded are rendered at."""
        self._augment_k = int(getattr(game, "_rotation_k", 0)) % 4

    def _to_core(self, action: Action) -> Action:
        """Screen action -> the core game's upright action (`drive` submits this)."""
        k = self._augment_k % 4
        if not k or not self.plans_in_screen_space:
            return action
        if action.is_click:
            x, y = action.click_xy
            gx, gy = remap_click(int(x), int(y), k, _AUGMENT_FRAME)
            return Action(action.action_id, click_rc=(gy, gx))
        game_a = _ID_TO_GAMEACTION.get(action.action_id)
        if game_a is None:
            return action
        from utils.rotation import remap_action
        return Action(int(remap_action(game_a, k).value), click_rc=action.click_rc)

    def _to_screen(self, action: Action) -> Action:
        """Upright (game-space) action -> the SCREEN action that produces it.

        A no-op for a `plans_in_screen_space` solver: its plan is already screen."""
        k = self._augment_k % 4
        if not k or self.plans_in_screen_space:
            return action
        if action.is_click:
            x, y = action.click_xy
            # ``remap_click`` is screen->game (a -k turn); its inverse is a +k turn.
            sx, sy = remap_click(int(x), int(y), (4 - k) % 4, _AUGMENT_FRAME)
            return Action(action.action_id, click_rc=(sy, sx))
        game_a = _ID_TO_GAMEACTION.get(action.action_id)
        if game_a is None:
            return action
        return Action(int(inverse_remap_action_full(game_a, k).value),
                      click_rc=action.click_rc)

    # ── action encoding (the single source of truth for the on-disk format) ──
    def _action_dict(self, action: Action) -> dict:
        """The flat ``{"type", "index"[, "data"]}`` form of one action, in SCREEN space."""
        action = self._to_screen(action)
        if action.is_click:
            x, y = action.click_xy
            return {"type": "mouse", "index": CLICK_ACTION,
                    "data": {"x": int(x), "y": int(y)}}
        return {"type": "simple", "index": int(action.action_id)}

    def _encode_step(self, taken: Action, *, optimal: list | None, phase: str,
                     changed: bool, n_obs: int = 1) -> dict:
        """One on-disk step record.

        TWO ACTION STREAMS (DESIGN.md): the *taken* action is stored flat at the
        top level -- so the legacy dataloader, which reads ``type``/``index``,
        keeps working unchanged -- and the training *target* is the additive
        ``optimal`` list (the optimal action SET at the pre-step state). They
        differ on exploration/burst steps: ``taken`` is the random detour,
        ``optimal`` is what the expert would have done, so the policy trains on
        recovery targets without being taught the mistake. ``phase`` is the loss
        mask (``explore``/``burst`` steps are excluded from the policy loss);
        ``changed`` flags null transitions for the dynamics model.

        MULTI-FRAME OBSERVATIONS: ``n_obs`` is how many frames of the level's
        flat ``observations`` stream this action produced (its *span*; see
        `record_level`). It is >1 whenever the action animated. Absent on legacy
        records, where it defaults to 1 -- so every episode written before this
        change still reads back exactly as it did."""
        rec = dict(self._action_dict(taken))
        rec["phase"] = phase
        rec["changed"] = bool(changed)
        rec["n_obs"] = int(n_obs)
        rec["optimal"] = None if optimal is None else [self._action_dict(a)
                                                       for a in optimal]
        return rec

    @classmethod
    def _reset_step(cls) -> dict:
        return {"type": "simple", "index": RESET_ACTION, "phase": "reset",
                "changed": False, "n_obs": 1, "optimal": None}

    # ── reset-mode prefix for CUSTOM record loops ────────────────────────────
    def _reset_prefix(self, *, schedule: ExplorationPrefix | None,
                      exploration: ExplorationPolicy | None, prev: np.ndarray,
                      drive_explore, reset_to_level, record_obs, record_act):
        """RESET-recovery exploration prefix for generators that keep a CUSTOM
        record loop (they override ``solve_episode`` / ``record_level`` and so the
        base RESET-recovery in `record_level` does not apply). It reproduces the
        ``recovery_mode == "reset"`` arc for that custom-loop case: run the
        episode-wide exploration prefix (taking exploratory actions, recorded with
        ``phase="explore"``), then -- unless an exploratory action already won the
        level -- perform ONE RESET back to the level's initial state (recorded with
        ``phase="reset"``, action index 0) so the generator's own optimal replay
        that follows starts from a valid state. Exactly like a human who flails,
        gets stuck/dies, hits reset, and only then solves.

        The caller supplies the engine-specific glue as callbacks:

          * ``drive_explore(action)`` -> ``(frames, terminal)`` -- perform one
            exploratory ``explore.Action`` against the real engine and return the
            frames it rendered plus ``"win"`` / ``"over"`` / ``None``. ``frames``
            may be a single ``(H,W)`` frame or the whole ``(k,H,W)`` animation.
          * ``reset_to_level()`` -> ``frames`` -- restore the level's INITIAL
            state (and clear any GAME_OVER) and return the restored frame(s),
            same two shapes.
          * ``record_obs(frame_ndarray)`` -- append ONE observation (and whatever
            per-frame labels the generator keeps). Called once per frame of the
            span, so this contract is unchanged from the single-frame era.
          * ``record_act(step_dict)`` -- append one encoded action record. Its
            ``n_obs`` is filled in from the span length automatically.

        MULTI-FRAME: returning a ``(k,H,W)`` span from ``drive_explore`` /
        ``reset_to_level`` is how a custom-loop generator records an action's whole
        animation -- which is what `solver.py` feeds the policy live, so a
        generator that keeps only the settled frame trains a token layout inference
        never sees. Returning one frame still works and is byte-identical to the
        old behaviour (``k == 1``), so generators migrate one at a time. See
        `record_level` for the alignment rule and `DriveResult` for why the
        animation matters.

        ``prev`` is the current frame (``observations[0]``). Returns the new
        ``prev`` frame. A no-op returning ``prev`` unchanged when exploration is
        off (``exploration is None``), so callers can invoke it unconditionally.
        Only the exploratory steps advance the (episode-wide) schedule, so the
        opening exploration budget decays across the whole episode."""
        if exploration is None:
            return prev
        def emit(returned, act_record_fn, prev_frame):
            """Record one action's whole animation and return its settled frame.

            ``drive_explore`` / ``reset_to_level`` may hand back either a single
            ``(H,W)`` frame or a ``(k,H,W)`` SPAN. Both are normalised to a span
            here, ``record_obs`` is called once PER FRAME (so its existing
            one-frame-at-a-time contract is unchanged), and the action record
            carries ``n_obs = k`` so `record_level`'s alignment rule still holds:
            ``sum(n_obs) == len(observations)``.

            A caller that still returns one frame gets ``k == 1`` and is
            byte-identical to the old behaviour -- which is what lets generators
            migrate to spans one at a time."""
            span = np.asarray(returned)
            if span.ndim == 2:                      # (H,W) -> a 1-frame span
                span = span[None]
            for f in span:
                record_obs(f)
            settled = span[-1]
            act_record_fn(len(span), settled, prev_frame)
            return settled

        perturbed = False
        last_terminal = None
        for _ in range(self.step_guard):
            if schedule is None or not schedule.explore():
                break
            schedule.advance()
            act = exploration.action(prev)
            frames, terminal = drive_explore(act)
            prev = emit(frames, lambda k, settled, was: record_act(
                self._encode_step(act, optimal=None, phase="explore",
                                  changed=not np.array_equal(settled, was),
                                  n_obs=k)), prev)
            perturbed = True
            last_terminal = terminal
            if terminal is not None:            # WIN or GAME_OVER ends the prefix
                break
        if perturbed and last_terminal != "win":
            prev = emit(reset_to_level(), lambda k, settled, was: record_act(
                self._encode_step(Action(RESET_ACTION),
                                  optimal=[Action(RESET_ACTION)], phase="reset",
                                  changed=not np.array_equal(settled, was),
                                  n_obs=k)), prev)
        return prev

    # ── level / episode recording ────────────────────────────────────────────
    def record_level(self, game, level_idx: int, seed: int, *,
                     schedule: ExplorationPrefix | None = None,
                     exploration: ExplorationPolicy | None = None):
        """Solve one level from ``game``'s current position and record it.

        Returns ``(observations, actions)`` on a verified solve, else
        ``(None, None)``.

        FRAME/ACTION ALIGNMENT (multi-frame). ``observations`` is a FLAT stream of
        frames and ``actions[i]`` owns a contiguous *span* of it, ``actions[i]
        ["n_obs"]`` frames long -- the whole animation that action produced (see
        `DriveResult`). So::

            start(i) = sum(a["n_obs"] for a in actions[:i])
            actions[i]  produced  observations[start(i) : start(i) + n_obs(i)]
            sum(a["n_obs"] for a in actions) == len(observations)

        ``actions[0]`` is the leading RESET producing ``observations[0]``. The
        state action ``i`` was DECIDED from is the last frame of span ``i-1``;
        the earlier frames of a span are the in-flight animation, which is what
        carries the mechanics an agent has to infer. With every ``n_obs == 1``
        (the common case: non-animating actions, and every episode recorded
        before this change) this collapses to the old 1:1 invariant, where
        ``actions[i]`` took the agent from ``observations[i-1]`` to
        ``observations[i]``.

        Driving is a single plan-cache loop over `solve_from`:

          * hold an optimal plan for the current state; the next optimal action is
            its head, and (via `optimal_set_from`) the recorded target;
          * normally take that optimal action (STOCHASTIC OPTIMAL: `solve_from`'s
            own tie-breaking gives a different valid route each episode);
          * when the exploration prefix or a perturbation BURST
            fires, take an exploratory action instead, still record the optimal
            target, and DISCARD the plan so the next optimal step re-`solve_from`s
            the perturbed live state -- transparent recovery.

        With ``schedule``/``exploration`` omitted this is pure optimal replay.
        They are created once per EPISODE in `solve_episode`, so the exploration
        arc spans the whole episode rather than restarting each level.

        RESET RECOVERY: when exploration bricks the level (dead / unsolvable) the
        loop records a RESET (restoring the initial state via `reset_level`) and
        resumes solving -- so exploration can safely reach GAME_OVER or a dead end.
        In ``recovery_mode == "reset"`` the whole exploration prefix runs first,
        then ONE RESET, then the plan replays from the (valid) initial state; in
        ``"replan"`` mode `solve_from` re-plans from the live perturbed state and
        RESET is only the last resort. See `recovery_mode`."""
        self.set_level(game, level_idx)
        self._lift_step_limit(game)                # step exhaustion != discard
        prev = self.render(game)
        observations = [prev.tolist()]
        actions = [self._reset_step()]
        recovering = exploration is not None      # exploration => RESET allowed
        reset_mode = self.recovery_mode == "reset"

        def do_reset() -> None:
            """Restore the initial state and record it as a RESET action."""
            nonlocal prev, perturbed
            self.reset_level(game, level_idx, seed)
            frame = self.render(game)
            observations.append(frame.tolist())
            actions.append(self._encode_step(
                Action(RESET_ACTION), optimal=[Action(RESET_ACTION)],
                phase="reset", changed=not np.array_equal(frame, prev)))
            prev = frame
            perturbed = False                      # back at a clean initial state

        def do_undo_burst() -> None:
            """Roll the game AND the recorded buffer back to the pre-burst anchor,
            discarding the stranding burst entirely -- the state is restored and
            the tentatively-recorded burst frames are dropped, so it never
            happened. UNDO is not a real competition action, so it must leave NO
            trace in the trajectory (no ACTION7, no undo frames): the buffer is
            simply truncated. Only ever fires when a burst put us in an
            unsolvable / dead state; reversible bursts are kept as recovery data."""
            nonlocal prev, perturbed, plan, burst_left, burst_anchor
            snap, obs_len, act_len, anchor_prev = burst_anchor
            self._game_restore(game, snap)
            del observations[obs_len:]
            del actions[act_len:]
            prev = anchor_prev
            perturbed = False
            plan = []
            burst_left = 0
            burst_anchor = None

        plan: list = []
        burst_left = 0
        resets = 0
        perturbed = False        # has an exploratory action moved us off-plan?
        burst_anchor = None      # (snapshot, obs_len, act_len, prev) of an active burst
        for _ in range(self.step_guard):
            # ── decide explore vs exploit ────────────────────────────────────
            exploring, via_burst = False, False
            if recovering:
                if burst_left > 0:                 # continuing an active burst
                    exploring, via_burst = True, True
                    burst_left -= 1
                elif schedule is not None and schedule.explore():
                    exploring = True               # opening exploration prefix
                elif (not reset_mode and burst_anchor is None
                      and self._burst_prob > 0
                      and self.rng.random() < self._burst_prob):
                    # ``burst_anchor is None`` guard: never start a new burst while
                    # a finished one is still pending its solvability check -- the
                    # post-burst exploit MUST run first (to roll the burst back or
                    # commit it), else a second burst would overwrite the anchor and
                    # lose the pre-burst snapshot.
                    # Bursts fire ONLY in replan mode: a genuine solve_from is
                    # needed both to LABEL each burst step's optimal target and to
                    # judge whether the burst is recoverable. (Reset-mode solvers
                    # are state-blind, so there is no point bursting them.) A burst
                    # that strands the game is UNDONE (rolled back, below) rather
                    # than RESET -- so snapshot the pre-burst state + buffer here.
                    burst_left = self._sample_burst() - 1
                    exploring, via_burst = True, True
                    burst_anchor = (self._game_snapshot(game),
                                    len(observations), len(actions), prev)
            if schedule is not None:
                schedule.advance()

            # A perturbation step is only worth taking if we can LABEL it. Its
            # target is `solve_from`'s head re-planned from the LIVE state, so an
            # empty plan means the state is unplannable and the step would land in
            # the data with `optimal=None` -- and v2 supervises `optimal` ONLY, so
            # such a step sits in every context contributing nothing to the loss.
            # (The EXPERT-phase taken-action fallback deliberately does not rescue
            # it: cloning a random detour as if it were expert is worse than no
            # target at all.) So "cannot label" is treated as "cannot perturb":
            #   * mid-burst -> roll the burst back to its anchor, exactly as for a
            #     burst that strands us. The state is restored and the tentative
            #     frames are dropped, so the unlabelled step never happened.
            #   * otherwise -> drop the perturbation and fall through to EXPLOIT,
            #     whose `not plan` branch already owns unsolvable-state recovery.
            # Reset-mode prefixes are exempt: their explore steps are deliberately
            # unlabelled and are undone by the RESET that closes the prefix.
            relabelled: list | None = None
            if exploring and not reset_mode:
                p = [_as_action(a) for a in self.solve_from(game, level_idx, seed)]
                if p:
                    relabelled = p[:1]
                elif burst_anchor is not None:
                    do_undo_burst()
                    continue
                else:
                    exploring, via_burst = False, False
                    burst_left = 0

            if exploring:
                taken = exploration.action(prev)
                phase = "burst" if via_burst else "explore"
                perturbed = True
                plan = []
                # A relabeled optimal target is only valid when solve_from re-plans
                # from the live state; reset-mode plans are stale off-initial.
                optimal = relabelled
            else:
                # EXPLOIT. Reset-mode's solve_from is only valid at the INITIAL
                # state, so ANY perturbation (the opening prefix) forces a RESET
                # before we can (re)solve. Replan-mode skips this and just re-plans
                # from the live state below (undoing a burst / resetting the prefix
                # only if the state turns out unsolvable).
                if reset_mode and perturbed:
                    if resets >= self.max_resets:
                        return None, None
                    do_reset(); resets += 1; plan = []
                    continue
                if not plan:
                    plan = [_as_action(a)
                            for a in self.solve_from(game, level_idx, seed)]
                    if not plan:                   # unsolvable from here
                        if burst_anchor is not None:   # a burst stranded us -> UNDO it
                            do_undo_burst(); continue
                        if recovering and resets < self.max_resets:
                            do_reset(); resets += 1; plan = []
                            continue
                        return None, None
                # A burst is pending its solvability verdict. solve_from returned a
                # plan, but that only proves the MODEL believes the post-burst state
                # winnable -- ENGINE-VERIFY the recovery on a throwaway copy before
                # committing. A model false-positive (a confident plan the engine
                # won't actually win) is caught HERE, one step after the burst, and
                # rolled back -- instead of surfacing many frames later when the
                # doomed plan strands us and RESET is the only recourse left.
                if burst_anchor is not None:
                    if not self._burst_recovery_wins(game, plan):
                        do_undo_burst(); continue
                # Reached a verified-solvable state: the burst was recoverable, so
                # KEEP it as recovery data (drop the rollback anchor).
                burst_anchor = None
                # STOCHASTIC OPTIMAL: sample the taken action uniformly among the
                # equally-optimal ones (records the whole set as the target). With
                # a single optimal action this is just the canonical replay.
                opts = self._optimal_set(game, level_idx, seed, plan)
                optimal = opts
                if len(opts) > 1:
                    taken = self.rng.choice(opts)
                    plan = []                      # deviated -> re-solve next step
                elif self._in_optimal_set(plan[0], opts):
                    taken = plan.pop(0)
                else:
                    # STALE PLAN. `optimal` was recomputed from the LIVE state,
                    # while the head came from a plan cached at an earlier one --
                    # so following the head would record an expert step acting
                    # against its own label, and `episode_is_clean` would throw
                    # the whole (winning) episode away. Common in partial-obs
                    # solvers, where every step reveals something that can move
                    # the optimum. Take the FRESH target instead and drop the
                    # cache, so the next step re-`solve_from`s the live state.
                    # Byte-for-byte a no-op whenever the head agrees, which is
                    # every step of a fully-observed replay.
                    taken = opts[0]
                    plan = []
                phase = "expert"

            res = self.drive(game, taken)
            changed = not np.array_equal(res.frame, prev)
            # Append the action's WHOLE animation; `n_obs` records the span so
            # the flat stream stays re-alignable with the action stream.
            observations.extend(f.tolist() for f in res.frames)
            actions.append(self._encode_step(taken, optimal=optimal, phase=phase,
                                             changed=changed,
                                             n_obs=len(res.frames)))
            prev = res.frame
            if res.solved:
                return observations, actions
            if res.dead:
                if burst_anchor is not None:       # a burst killed us -> UNDO it
                    do_undo_burst(); continue
                if recovering and resets < self.max_resets:  # GAME_OVER -> RESET
                    do_reset(); resets += 1; plan = []   # do_reset clears `perturbed`
                    continue
                return None, None
        return None, None

    # ── post-recording validation ────────────────────────────────────────────
    #: Throw away a recorded episode whose EXPERT steps drifted off their own
    #: target (see `episode_is_clean`). Rejection only -- it never edits an
    #: episode, so an accepted one is byte-for-byte what it was before.
    validate_episodes: bool = True

    @staticmethod
    def _act_identity(a: dict):
        """Identity of a recorded action: type + index + click payload. The taken
        action and the `optimal` entries are both written through `_action_dict`,
        so they compare directly."""
        return (a.get("type"), a.get("index"),
                json.dumps(a.get("data"), sort_keys=True))

    def _in_optimal_set(self, action: Action, optimal: list) -> bool:
        """Is ``action`` one of the ``optimal`` actions, compared exactly as
        `episode_is_clean` will compare them on disk? Both sides go through
        `_action_dict`, so screen-space remapping and click payloads are folded
        in and a live check can never disagree with the recorded one."""
        me = self._act_identity(self._action_dict(action))
        return any(me == self._act_identity(self._action_dict(o)) for o in optimal)

    def episode_is_clean(self, levels: list) -> bool:
        """Is every EXPERT step's taken action inside its own recorded target?

        It should be, by construction: `record_level` samples the taken action
        FROM the optimal set whenever that set has ties. The one window where it
        cannot be taken for granted is a run of forced-SINGLE-optimal steps,
        where the taken action is the head of a plan cached from an earlier state
        (`taken = plan.pop(0)`) while the recorded target is recomputed fresh --
        so a plan that has gone stale shows up here as an expert step acting
        against its own label. Measured on the shipped corpora that is rare and
        concentrated: 2 dungeon_key_hunt seeds in 1000 (dither loops ~500 steps
        long), 4 in mx01, 23 single steps in nu01, and nothing at all in the
        other 26 corpora.

        Explore and burst steps are exempt -- their taken action is a deliberate
        detour and differing from the target is the entire point of the recovery
        data. An expert step with NO target fails too: v2 supervises `optimal`
        only, so an unlabelled expert step is a step that teaches nothing.

        Pure bookkeeping over a list already in memory -- no game copy, no
        search -- so it costs nothing next to recording the episode."""
        for lvl in levels:
            for act in lvl.get("actions", ()):
                if act.get("phase") != "expert":
                    continue
                optimal = act.get("optimal")
                if not optimal:
                    return False
                if self._act_identity(act) not in {
                        self._act_identity(o) for o in optimal}:
                    return False
        return True

    def _sample_burst(self) -> int:
        """Geometric-ish burst length (>=1), mean ~``_burst_mean``."""
        n = 1
        p = 1.0 / max(1, self._burst_mean)
        while self.rng.random() > p:
            n += 1
        return n

    def solve_episode(self, seed: int, explore: bool = True):
        """Record every level for one seed. Returns ``(ok, levels)``,
        all-or-nothing: a single unsolved level fails the whole episode (the
        WIN-only corpus contract).

        Exploration (the exploration prefix + bursts) is enabled only when requested
        AND the solver declares `supports_recovery` -- otherwise a perturbation
        could strand a plan-only expert that can't re-solve from the detour. The
        exploration policy and prefix are built ONCE here so the arc is
        episode-wide, not re-injected at every level."""
        game = self.make_game(seed)
        n = self.num_levels(game)
        do_explore = explore and self.supports_recovery
        schedule = (ExplorationPrefix(self.rng, center=self._explore_center,
                                    jitter=self._explore_jitter)
                    if do_explore else None)
        exploration = (ExplorationPolicy(self.available_actions(game), self.rng)
                       if do_explore else None)
        levels: list[dict] = []
        for level_idx in range(n):
            try:
                obs, acts = self.record_level(game, level_idx, seed,
                                              schedule=schedule,
                                              exploration=exploration)
            except Exception:                    # noqa: BLE001 -- a bad seed just fails
                return False, levels
            if obs is None:
                return False, levels
            levels.append({"level_id": level_idx, "observations": obs,
                           "actions": acts})
        return True, levels

    # ── output normalisation (schema is enforced here, at the write path) ────
    @classmethod
    def normalize_levels(cls, levels: list) -> list:
        """Force every level onto the canonical schema, whatever produced it.

        This is the single guarantee that ALL generators emit identical output --
        including ones that override ``solve_episode`` and assemble trajectories by
        hand. It (a) DROPS the obsolete per-pixel encoder ``labels`` block (a dead
        schema kept by a few old generators), and (b) upgrades any bare
        ``{"type","index"}`` action to the two-stream form with ``phase`` /
        ``changed`` / ``optimal``. For a hand-built (pure-optimal replay) step the
        taken action is the optimal one, so ``optimal`` defaults to the taken
        action; the RESET at index 0 gets ``phase="reset"`` and ``optimal=None``.

        It also carries ``n_obs`` -- the number of frames the action produced (see
        `record_level`). This is a WHITELISTING normaliser, so a new field has to be
        named here or it is silently dropped on the way to disk; losing ``n_obs``
        leaves the frames on disk but destroys the alignment that makes them
        readable. When it is absent the level is assumed 1:1, and a level whose
        spans then fail to tile its frame stream is a hard error rather than a
        truncated demonstration.

        Idempotent: levels already in canonical form pass through unchanged."""
        out = []
        for lvl in levels:
            obs = lvl.get("observations", [])
            acts = lvl.get("actions", [])
            norm_acts = []
            at = 0
            for i, a in enumerate(acts):
                d = {k: v for k, v in a.items() if k in ("type", "index", "data")}
                d["phase"] = a.get("phase", "reset" if i == 0 else "expert")
                k = max(1, int(a.get("n_obs", 1) or 1))
                d["n_obs"] = k
                if "changed" in a:
                    d["changed"] = bool(a["changed"])
                else:
                    # Compare this span's SETTLED frame with the previous one's.
                    j, prev_j = at + k - 1, at - 1
                    d["changed"] = bool(i > 0 and j < len(obs)
                                        and obs[j] != obs[prev_j])
                at += k
                if "optimal" in a:
                    d["optimal"] = a["optimal"]
                elif d["phase"] == "reset":
                    d["optimal"] = None
                else:
                    d["optimal"] = [{k2: v for k2, v in d.items()
                                     if k2 in ("type", "index", "data")}]
                norm_acts.append(d)
            if acts and at != len(obs):
                raise ValueError(
                    f"level {lvl.get('level_id')}: action spans cover {at} frames "
                    f"but observations has {len(obs)} -- n_obs lost or corrupted")
            out.append({"level_id": lvl.get("level_id", len(out)),
                        "observations": obs, "actions": norm_acts})
        return out

    # ── CLI harness ──────────────────────────────────────────────────────────
    def run(self, episodes: int, out: Path, *, start_seed: int = 0,
            max_attempts: int = 0, progress_every: int = 25,
            explore: bool = True) -> int:
        """Write ``episodes`` WIN episodes, one JSON per solvable seed."""
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        print(f"Generating {episodes} {self.game_id} episodes "
              f"(one multi-level game per WIN seed)")
        print(f"Output -> {out.resolve()}")

        max_attempts = max_attempts or (10 * episodes)
        written, attempts, seed, rejected = 0, 0, start_seed, 0
        while written < episodes and attempts < max_attempts:
            ok, levels = self.solve_episode(seed, explore=explore)
            attempts += 1
            cur_seed, seed = seed, seed + 1
            if not ok:
                continue
            if self.validate_episodes and not self.episode_is_clean(levels):
                # A WIN that is not clean training data: an expert step acted
                # against its own recorded target. Drop the SEED, not the run --
                # `written` is untouched, so asking for N still delivers N.
                rejected += 1
                print(f"  seed {cur_seed}: discarded (expert step off its own "
                      f"optimal target)")
                continue
            episode = {"game_id": self.game_id,
                       "levels": self.normalize_levels(levels)}
            out_path = out / f"episode_{written:05d}_seed{cur_seed}.json"
            with out_path.open("w") as f:
                json.dump(episode, f)
            written += 1
            if written % progress_every == 0 or written == episodes:
                total_T = sum(len(lvl["observations"]) for lvl in levels)
                print(f"  {written}/{episodes}  attempts={attempts}  "
                      f"levels={len(levels)}  totalT={total_T}  -> {out_path.name}")

        if rejected:
            print(f"Discarded {rejected} winning seed(s) that failed validation "
                  f"(expert step off its own optimal target)")
        if written < episodes:
            print(f"WARN: only {written}/{episodes} after {attempts} attempts")
            return 1
        print(f"Done: {written} episodes written to {out}")
        return 0

    # ── entry point ──────────────────────────────────────────────────────────
    @classmethod
    def build_argparser(cls) -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(
            description=f"Generate {cls.game_id} training data.")
        p.add_argument("--episodes", type=int, default=2000,
                       help="Number of WIN episodes (one per solvable seed).")
        p.add_argument("--start-seed", type=int, default=0)
        p.add_argument("--out", type=Path, default=None,
                       help="Output dir (default data/training_multi_level/<id>).")
        p.add_argument("--max-attempts", type=int, default=0,
                       help="Hard cap on seeds to try (0 = 10x episodes).")
        p.add_argument("--progress-every", type=int, default=25)
        p.add_argument("--seed", type=int, default=0,
                       help="RNG seed for exploration/stochastic-optimal choices.")
        p.add_argument("--no-exploration", action="store_true",
                       help="Disable the fixed-length exploration prefix "
                            "(ON by default). Only engages for solvers that "
                            "declare supports_recovery; others replay pure optimal.")
        p.add_argument("--noise", type=float, default=cls.default_noise,
                       help="P(start a perturbation burst) per optimal step "
                            "(0 = no bursts). Needs supports_recovery. "
                            f"Default for {cls.game_id}: {cls.default_noise}.")
        p.add_argument("--burst", type=int, default=5,
                       help="Mean length (steps) of a perturbation burst.")
        return p

    @classmethod
    def main(cls, argv=None) -> int:
        args = cls.build_argparser().parse_args(argv)
        solver = cls(rng=random.Random(args.seed),
                     burst_prob=args.noise, burst_mean=args.burst)
        out = args.out or (Path("data/training_multi_level") / cls.game_id)
        return solver.run(args.episodes, out, start_seed=args.start_seed,
                          max_attempts=args.max_attempts,
                          progress_every=args.progress_every,
                          explore=not args.no_exploration)
