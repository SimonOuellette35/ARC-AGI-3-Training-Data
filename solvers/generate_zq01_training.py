"""Generate Phase-1 training data for zq01 ("Zone Timer").

zq01 (games/zq01/zq01.py) is a pure TIMING puzzle on an open grid: ACTION1..4 step
the player up/down/left/right, and a fixed set of hazard cells toggles between
blocking (visible) and harmless (invisible) every ``period`` actions. Reach the goal
cell to clear the level; each *seed* is a 5-level game, so every WIN seed yields one
multi-level episode.

Nothing here is lethal and there are no walls. A move into an on-hazard, or off the
board, simply consumes the action -- so stalling is a first-class move and the whole
game is "go round, or wait for the blink and go through".

THE EXPERT IS A DISTANCE FIELD, NOT A SEARCH PER QUERY. The state is exactly
``(player cell, timer phase)`` with the phase in ``[0, 2 * period)``, which is a few
thousand states for any board this game draws, so ``utils.zone_timer.ZoneTimerField``
builds the successor table once and runs ONE reverse BFS from the goal. Then
``solve_from`` is a greedy descent of that field and ``optimal_set_from`` is a 4-way
lookup -- the exact co-optimal tie set, at no extra cost. That same field is what the
GAME's level sampler uses as its solvability certificate, so every board zq01 can
draw is one this expert solves optimally (see utils/zone_timer.py).

WHAT THE EXPERT IS ALLOWED TO KNOW (no privileged reads). The two level constants the
field needs -- the blink PERIOD and WHICH cells are hazards -- are not visible in the
level's opening frame, because the hazards start off and therefore invisible. They are
not read off the level data; they are LEARNED by probing the environment as a black
box, exactly as the tt02 / dc22 / tu93 experts do: ``_probe`` clones the game, parks
the clone's player off the board so nothing it does can change the level, steps that
clone, and watches the hazards blink. Two toggles name the period; any on-frame names
the hazard cells. The probe reads only what its own frames show.

The third input, the phase, needs no probe in the normal case: the timer starts at
zero when a level is seated and advances by exactly one per action, so the solver
keeps its own tally of the actions IT has taken (``_zq_clock``, kept on the game object
so snapshots and burst rollbacks carry it). That is an agent counting its own moves,
not a peek at ``game._ticks``. The tally is cross-checked against the observable
on/off state every query, and a probe re-derives the phase from scratch (steps until
the next toggle) if it ever disagrees or is missing -- so ``solve_from`` still works on
a game this solver did not seat.

Recovery (supports_recovery = True, recovery_mode = "replan")
------------------------------------------------------------
The field covers EVERY ``(cell, phase)``, so replanning after the exploration prefix
or mid-burst is the same lookup as planning from the start -- and because the game has
no walls, no death and no irreversible state, every reachable state is winnable, so a
burst can wander anywhere and the expert simply re-solves from wherever it was left.
Off-plan detours are recorded as recovery data rather than rolled back.

Usage (run from the repo root):
    python solvers/generate_zq01_training.py --episodes 1000 \
        --out data/training_multi_level/zq01
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arcengine import ActionInput, GameAction                   # noqa: E402
from games.zq01.zq01 import Zq01                                # noqa: E402
from solvers.base_solver import BaseSolver, DriveResult          # noqa: E402
from utils.rotation import inverse_remap_action_full            # noqa: E402
from utils.zone_timer import ZoneTimerField                     # noqa: E402

#: Game-space directions, index-aligned with ``utils.zone_timer.DELTAS`` and with the
#: game's own ACTION1..4 -> (dx, dy) mapping.
_ACTIONS = (GameAction.ACTION1, GameAction.ACTION2,
            GameAction.ACTION3, GameAction.ACTION4)

#: Probe budget, in actions. Two toggles are needed to name the period, so this covers
#: any period up to ~31 -- an order of magnitude past what the game draws (3..7).
_MAX_PROBE = 64


class Zq01Solver(BaseSolver):
    game_id = "zq01"
    # ``solve_from`` answers from an exact field over every (cell, phase) of the LIVE
    # board, so it re-plans optimally from any state an exploratory detour leaves
    # behind. zq01 has no walls, no death and no irreversible state, so there is
    # nothing a burst can do that the next replan cannot undo.
    supports_recovery = True
    recovery_mode = "replan"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._models: dict = {}          # (seed, level) -> (period, hazard cells)
        self._fields: dict = {}          # full board signature -> ZoneTimerField

    def make_game(self, seed: int):
        # Zq01 is an AugmentedGame: the seed fixes every level's board size, timer
        # period, hazard pattern, layout, colours and display rotation, so a recorded
        # episode replays byte-for-byte.
        return Zq01(seed=seed)

    def available_actions(self, game) -> list[int]:
        return [1, 2, 3, 4]

    # ── the action tally (the phase, derived from the solver's own moves) ────
    def set_level(self, game, level_idx: int) -> None:
        """Seat a level and restart the action tally -- the engine's timer starts at
        zero whenever a level is seated, and so does the count of actions taken since.
        (`BaseSolver.reset_level` routes through here, so a recorded RESET re-syncs
        the tally exactly the way it re-syncs the board.)"""
        super().set_level(game, level_idx)
        game._zq_clock = 0

    def drive(self, game, action) -> DriveResult:
        """Perform one action and advance the tally. Every action the harness takes --
        expert, exploration, burst, and the throwaway replays used to verify a burst --
        goes through here, so the tally cannot drift from the number of actions the
        board has actually seen."""
        res = super().drive(game, action)
        game._zq_clock = int(getattr(game, "_zq_clock", 0)) + 1
        return res

    # ── observable reads ────────────────────────────────────────────────────
    @staticmethod
    def _hazards_visible(game) -> bool:
        """Are the hazards currently blocking? Drawn == blocking in this game, so this
        is exactly what the frame shows (and what the corner indicator echoes)."""
        return any(s.is_visible
                   for s in game.current_level.get_sprites_by_tag("zq_hazard"))

    @staticmethod
    def _live_board(game):
        """``(gw, gh, player cell, goal cell)`` read off the visible board."""
        level = game.current_level
        gw, gh = (int(v) for v in level.grid_size)
        p = level.get_sprites_by_tag("player")[0]
        t = level.get_sprites_by_tag("target")[0]
        return gw, gh, (int(p.x), int(p.y)), (int(t.x), int(t.y))

    # ── the level constants, learned by black-box probing ────────────────────
    def _probe(self, game):
        """Watch the hazards blink on a throwaway clone; return
        ``(period, hazard cells, steps until the next toggle)`` or ``None``.

        The clone's player is parked off the board, so every probe action is a no-op
        for the player -- it can neither reach the goal nor disturb the level -- while
        the timer keeps ticking. Only what the probe's own frames show is read: which
        hazard tiles are DRAWN, and when the blink flips. Two flips name the period
        (the gap between them is exactly one phase), any on-frame names the cells, and
        the first flip names how far into the current phase the live board is."""
        clone = copy.deepcopy(game)
        clone.current_level.get_sprites_by_tag("player")[0].set_position(-3, -3)
        seq = [self._visible_hazards(clone)]          # t=0: the live configuration
        flips: list[int] = []
        for i in range(1, _MAX_PROBE + 1):
            clone._full_reset = False
            clone._set_action(ActionInput(id=GameAction.ACTION1))
            guard = 0
            while not clone.is_action_complete() and guard < 8:
                guard += 1
                clone.step()
            seq.append(self._visible_hazards(clone))
            if bool(seq[i]) != bool(seq[i - 1]):
                flips.append(i)
                if len(flips) == 2:
                    break
        if len(flips) < 2:
            return None
        period = flips[1] - flips[0]
        cells = next((c for c in seq if c), None)
        if not cells or period < 1:
            return None
        return period, frozenset(cells), flips[0]

    @staticmethod
    def _visible_hazards(game) -> tuple:
        return tuple(sorted((int(s.x), int(s.y))
                            for s in game.current_level.get_sprites_by_tag("zq_hazard")
                            if s.is_visible))

    def _model(self, game, level_idx: int, seed: int):
        """``(period, hazard cells)`` for the live level -- probed once and memoised,
        since both are constants of the (seed, level) board."""
        key = (seed, level_idx)
        model = self._models.get(key)
        if model is None:
            probed = self._probe(game)
            if probed is None:
                return None
            period, cells, _to_flip = probed
            model = (period, cells)
            self._models[key] = model
        return model

    def _phase(self, game, level_idx: int, seed: int, period: int):
        """The live timer phase in ``[0, 2 * period)``.

        Fast path: the action tally. ``_zq_clock`` is written in exactly two places --
        `set_level` (which zeroes it, because seating a level restarts the timer) and
        `drive` (which increments it, because every action advances the timer by one).
        So its presence means "this solver has been counting every action this board
        has seen", and it is then exact by construction. It is cross-checked against
        the one thing the board does show, whether the hazards are up -- note that
        this pins the phase only to its HALF (on vs off), which is all a single frame
        can say, so the tally is the authority and the check is a guard, not a
        derivation.

        Slow path: no tally (a game this solver never seated -- an external caller
        handing us a live board) or a tally the board contradicts. Then the phase is
        re-derived from scratch by probing, and deliberately NOT written back: a board
        we are not driving can advance behind our back, so every query on it re-probes
        rather than trusting a stale number."""
        on_now = self._hazards_visible(game)
        clock = getattr(game, "_zq_clock", None)
        if clock is not None:
            ph = int(clock) % (2 * period)
            if (ph // period) % 2 == (1 if on_now else 0):
                return ph
        probed = self._probe(game)                    # re-derive from scratch
        if probed is None:
            return None
        p2, cells, to_flip = probed
        if p2 != period:                              # the memoised model was stale
            self._models[(seed, level_idx)] = (p2, cells)
            return None                               # caller re-enters with the new one
        return (2 * period - to_flip) if on_now else (period - to_flip)

    # ── the field ───────────────────────────────────────────────────────────
    def _field(self, gw, gh, period, hazard, target) -> ZoneTimerField:
        key = (gw, gh, period, hazard, target)
        field = self._fields.get(key)
        if field is None:
            field = ZoneTimerField(gw, gh, period, hazard, target)
            self._fields[key] = field
        return field

    def _state(self, game, level_idx: int, seed: int):
        """``(field, player cell, phase)`` for the live board, or ``None`` when it
        cannot be planned from (an unlearnable timer, or a phase the board denies)."""
        for _attempt in range(2):                     # one retry after a stale model
            model = self._model(game, level_idx, seed)
            if model is None:
                return None
            period, hazard = model
            phase = self._phase(game, level_idx, seed, period)
            if phase is None:
                continue                              # model was refreshed -> retry
            gw, gh, player, target = self._live_board(game)
            field = self._field(gw, gh, period, hazard, target)
            return field, field.cell(*player), phase
        return None

    def _screen(self, game, dirs) -> list:
        """Game-space direction indices -> the SCREEN actions to press, through this
        level's live rotation (the field is built in game space; the agent presses in
        screen space, and the game's ``screen_action_to_game`` maps it back)."""
        k = game.rotation_k
        return [_ACTIONS[d] for d in dirs]

    # ── the solver API ──────────────────────────────────────────────────────
    def solve_from(self, game, level_idx: int, seed: int):
        """An optimal plan from the game's LIVE state: a greedy descent of the
        distance field, tie-broken at random so repeated episodes of the same seed
        take different (equally optimal) routes -- including different choices of
        where to stall while the barrier is up."""
        st = self._state(game, level_idx, seed)
        if st is None:
            return []
        field, cell, phase = st
        dirs = field.plan(cell, phase, self.rng)
        return self._screen(game, dirs)

    def optimal_set_from(self, game, level_idx: int, seed: int):
        """Every equally-optimal next action at the LIVE state -- a 4-way lookup, so
        the full tie set is the recorded training target at no extra cost. It is
        usually more than one move: with the phase in the state, stalling against the
        board edge is frequently as good as advancing."""
        st = self._state(game, level_idx, seed)
        if st is None:
            return None
        field, cell, phase = st
        return self._screen(game, field.optimal_dirs(cell, phase)) or None


if __name__ == "__main__":
    sys.exit(Zq01Solver.main())
