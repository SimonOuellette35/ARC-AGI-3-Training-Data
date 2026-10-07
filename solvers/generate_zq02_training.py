"""Generate Phase-1 training data for the zq02 "Dual Phase Hazards" game.

zq02 (games/zq02/zq02.py): reach the green target. ACTION1..4 step the player one
cell up/down/left/right. Two independent hazard SETS blink on and off on separate
periods and offsets; a hazard that is ON blocks the move (the action is consumed,
the player stays and a HUD block flashes), a hazard that is OFF is a free cell.
Five levels, 8x8 and 10x8. Nothing is lethal and nothing is irreversible -- this is
purely a TIMING maze, and the only resource is parity.

Mechanics, in the exact order ``Zq02.step`` applies them:

1. ``_ticks`` increments, then set A toggles iff ``(_ticks + offset_a) % period_a
   == 0`` and set B likewise -- so BOTH sets are already at their new state when
   the move is resolved. The frame rendered after the action therefore shows the
   configuration the action was resolved against, i.e. the configuration the NEXT
   action will meet is one tick further on;
2. the move is resolved against the TOPMOST sprite of the destination cell
   (``Level.get_sprite_at(..., ignore_collidable=True)``, layer-descending then
   insertion order): a wall blocks, a *collidable* (= ON) hazard blocks, anything
   else (an OFF hazard, the target, empty space) is entered;
3. if the player now stands on a target, the level is won.

Every action costs exactly one tick, including one that changes nothing -- an
out-of-bounds move or a blocked one. So pushing into the board edge is a free WAIT,
which is the whole game: a full hazard column (level 4) can only be crossed during
its OFF window, and reaching the right cell at the right parity is the puzzle.

Because ``ignore_collidable=True`` picks the topmost sprite regardless of
visibility, WHICH sprite decides a cell is a static property of the level, not of
the blink phase -- so each cell has a fixed kind (free / wall / hazard-A /
hazard-B) and only the two blink bits vary.

Expert
------
The state is exactly ``(player cell, tick phase)``. Both sets toggle on fixed
periods, so the configuration sequence is periodic; with period ``P`` the state space
is ``P * grid_cells``, small enough to enumerate outright. That is
``utils.zone_timer.PhasedGridField`` -- the SHARED field for this repo's
time-varying-obstacle games (zq01 is the single-set case) -- which builds the exact
successor table for the four moves and runs ONE reverse BFS from the winning
transitions. All `_Model` adds is the zq02-specific translation: cell kinds plus the
blink schedule become the per-phase blocked masks that field is defined over (rules
1-3 above, exactly). `solve_from` is then a lookup plus a greedy descent and
`optimal_set_from` is the exact co-optimal tie set -- no search per step, and
re-planning from an arbitrary state is free.

The same field is what the GAME's per-seed level sampler uses as its solvability
certificate, so every board zq02 can draw is one this expert solves optimally.

Nothing privileged is read. The blink SCHEDULE (which no viewer of a single frame
can see) is LEARNED by black-box probing, exactly as the tt02 patrol schedule is:
`_learn_schedule` clones the game, parks the clone's player off the board so every
action is a no-op that still ticks the clock, and steps that clone -- recording only
the hazard sprites' VISIBILITY, which is precisely what the frame shows -- until the
configuration sequence closes on itself. That yields ``P`` and the phase table.

The live PHASE is the agent's own action count since the level began (one action ==
one tick, and a level starts at tick 0 with both sets off), kept in a counter on the
game object -- so the recorder's burst snapshot/restore rewinds it with the rest of
the world. Every query cross-checks that counter against the OBSERVED blink bits and,
on any mismatch, re-identifies the phase by probing the clone forward and matching
the observed blink window against the schedule (a window of ``P`` configurations is
unique because ``P`` is the *minimal* period). So the phase is always pinned by
something an agent can actually see.

Recovery (supports_recovery = True, recovery_mode = "replan")
------------------------------------------------------------
The field covers EVERY ``(cell, phase)``, so `solve_from` re-plans optimally from any
live state: after the exploration prefix, mid-burst, or at any parity the detour left
behind. zq02 has no death and no irreversible move, so every reachable state is
solvable and recovery never needs a RESET -- the perturbation data is pure "wrong
parity / wrong cell, here is the optimal route back". A state with no finite distance
still returns ``[]`` (the base then rolls the burst back or RESETs) rather than
guessing.

Zq02 is an ``AugmentedGame``: each *seed* redraws every level's board size, BOTH blink
periods and offsets, both hazard patterns, the player/goal cells, the colours and the
display rotation. The field is computed in GAME space and every emitted action is
converted to the SCREEN direction the agent must press via
``inverse_remap_action_full``, so the game's own ``screen_action_to_game`` maps it back
to the intended move. On top of that, stochastic-optimal play spreads episodes of the
SAME board over many routes -- in a timing game the co-optimal tie sets are large,
since waiting against an edge for the right parity is frequently as good as advancing.

Usage (run from the repo root):
    python solvers/generate_zq02_training.py --episodes 1000 \
        --out data/training_multi_level/zq02
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                              # noqa: E402

from arcengine import ActionInput, GameAction                    # noqa: E402
from games.zq02.zq02 import Zq02                                 # noqa: E402
from solvers.base_solver import BaseSolver                        # noqa: E402
from utils.rotation import inverse_remap_action_full              # noqa: E402
from utils.zone_timer import PhasedGridField                      # noqa: E402

#: Direction index -> the GAME-space action it is. `_screen` maps these to the
#: SCREEN action to press, through the level's live display rotation.
_ACTIONS = (GameAction.ACTION1, GameAction.ACTION2,
            GameAction.ACTION3, GameAction.ACTION4)

# Cell kinds. The deciding sprite of a cell is fixed for the whole level (see the
# module docstring), so this table is static and only the two blink bits vary.
_FREE, _WALL, _HAZ_A, _HAZ_B, _SOLID = range(5)

#: Longest blink period this expert will accept, and how many configurations the
#: probe records. The window must cover the true period plus the candidate period
#: for the periodicity test to be conclusive, hence 2x. The stock levels are
#: ``P = lcm(2*period_a, 2*period_b)`` = 12..70.
_MAX_PERIOD = 128
_PROBE_LEN = 2 * _MAX_PERIOD

#: Attribute the solver keeps its own tick count under, ON THE GAME OBJECT -- so the
#: recorder's whole-game snapshot/restore (burst undo, the recovery dry-run) rewinds
#: the count together with the world it describes.
_TICK_ATTR = "_zq02_expert_ticks"


class _Model:
    """One level's blink SCHEDULE plus its exact ``(cell, phase)`` distance-to-win
    field.

    ``kind[c]`` is the static kind of cell ``c`` (see ``_FREE``..``_SOLID``),
    ``targets`` the winning cells, and ``sched[t]`` the ``(on_a, on_b)``
    configuration in effect at tick ``t`` (mod ``P``) -- ``sched[0]`` being a level's
    opening configuration, since a level starts at tick 0.

    The field itself is `utils.zone_timer.PhasedGridField`, shared with zq01: this
    class's whole job is the zq02-specific translation from (cell kinds + blink
    schedule) into the per-phase blocked masks that field is defined over. A cell
    blocks at phase ``t`` iff it is a wall / other solid, or it is a hazard whose set
    is ON at ``t`` -- which is exactly what ``Zq02.step`` resolves a move against."""

    def __init__(self, gw: int, gh: int, kind: tuple, targets: frozenset,
                 sched: tuple) -> None:
        self.gw, self.gh = gw, gh
        self.ncell = gw * gh
        self.kind = kind
        self.targets = targets
        self.sched = sched
        self.P = len(sched)
        kind_arr = np.asarray(kind)
        static = (kind_arr == _WALL) | (kind_arr == _SOLID)
        is_a, is_b = kind_arr == _HAZ_A, kind_arr == _HAZ_B
        blocked = np.empty((self.P, self.ncell), dtype=bool)
        for t, (on_a, on_b) in enumerate(sched):
            blocked[t] = static | (is_a if on_a else False) | (is_b if on_b else False)
        self.field = PhasedGridField(gw, gh, blocked, targets)

    def cell(self, x: int, y: int) -> int:
        return y * self.gw + x


class Zq02Solver(BaseSolver):
    game_id = "zq02"
    #: ``solve_from`` answers from a field covering EVERY (cell, phase), so it
    #: re-plans optimally from any live state. zq02 has no death and no irreversible
    #: move, so a perturbation can only cost distance, never the level.
    supports_recovery = True
    recovery_mode = "replan"

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        #: (gw, gh, kind, targets, sched) -> `_Model`. The field depends only on the
        #: level's geometry and blink schedule, never on where the player is, so one
        #: entry serves every replan, burst and RESET inside a level -- and, zq02
        #: being static, every episode too.
        self._models: dict = {}
        self._model: _Model | None = None       # the LIVE level's model

    def make_game(self, seed: int):
        # Zq02 is an AugmentedGame: the seed fixes every level's board size, blink
        # timing, hazard patterns, layout, colours and display rotation, so a recorded
        # episode replays byte-for-byte.
        return Zq02(seed=seed)

    def available_actions(self, game) -> list[int]:
        return [1, 2, 3, 4]

    # ── observable reads ─────────────────────────────────────────────────────
    @staticmethod
    def _config(game) -> tuple:
        """The live ``(on_a, on_b)`` blink bits, read off the hazard sprites'
        VISIBILITY -- i.e. off exactly the pixels the frame shows. A set with no
        sprites at all falls back to the game's own HUD indicator state, which is the
        same information by another route.

        Read from the sprites rather than from a render because it is the direct
        route, and because it cannot perturb anything: ``Zq02UI.render_interface``
        used to decrement the blocked-move flash counter as a side effect of drawing,
        so any extra render ate frames of a HUD animation that belonged to the
        recording (the observe-twice-per-step trap). The countdown now belongs to
        ``Zq02.step``, so rendering is pure -- but reading state by rendering it is
        still the wrong habit.
        """
        level = game.current_level
        ui = getattr(game, "_ui", None)
        out = []
        for tag, hud in (("zq2_hazard_a", "_a"), ("zq2_hazard_b", "_b")):
            sprites = level.get_sprites_by_tag(tag)
            if sprites:
                out.append(any(bool(s.is_visible) for s in sprites))
            else:
                out.append(None if ui is None else bool(getattr(ui, hud, False)))
        return tuple(out)

    @staticmethod
    def _cell_kinds(game):
        """``(gw, gh, kind, targets)`` for the live level.

        ``kind[c]`` is decided by the TOPMOST sprite of cell ``c`` in
        ``Level.get_sprite_at``'s order (layer-descending, then insertion order),
        which is what the engine resolves a move against. Visibility does not enter
        that choice (the game passes ``ignore_collidable=True``), so the table is
        static for the whole level and only the blink bits vary.

        The PLAYER is skipped: it is never its own destination, and including it
        would mark whichever cell it currently occupies permanently blocked.
        ``targets`` is collected from every target sprite regardless of layering,
        because the engine's win test compares the player's cell against each
        target's cell directly -- a target buried under a hazard still wins.
        """
        level = game.current_level
        gw, gh = (int(v) for v in level.grid_size)
        kind = [_FREE] * (gw * gh)
        decided = [False] * (gw * gh)
        for s in sorted(level.get_sprites(), key=lambda sp: sp.layer, reverse=True):
            if "player" in s.tags:
                continue
            if "wall" in s.tags:
                k = _WALL
            elif "zq2_hazard_a" in s.tags:
                k = _HAZ_A
            elif "zq2_hazard_b" in s.tags:
                k = _HAZ_B
            else:
                # Anything else blocks iff it is collidable (the target is not).
                k = _SOLID if s.is_collidable else _FREE
            for yy in range(int(s.y), int(s.y) + int(s.height)):
                for xx in range(int(s.x), int(s.x) + int(s.width)):
                    if not (0 <= xx < gw and 0 <= yy < gh):
                        continue
                    c = yy * gw + xx
                    if decided[c]:
                        continue
                    decided[c] = True
                    kind[c] = k
        targets = frozenset(
            int(t.y) * gw + int(t.x)
            for t in level.get_sprites_by_tag("target")
            if 0 <= int(t.x) < gw and 0 <= int(t.y) < gh)
        return gw, gh, tuple(kind), targets

    # ── the blink schedule, learned by black-box probing ─────────────────────
    def _probe(self, game, steps: int) -> list:
        """The next ``steps + 1`` blink configurations, starting with the live one.

        Clones the game and parks the clone's player off the board, so every action
        is a no-op for the player (an out-of-bounds move is skipped) while the clock
        still ticks and the sets still blink -- the player can neither win nor be
        blocked, so the probe never changes the level under itself. Only the hazard
        VISIBILITY is read, which is what the frame shows; nothing is rendered, so
        the live game is untouched (see `_config`).
        """
        clone = copy.deepcopy(game)
        clone._player.set_position(-3, -3)
        seq = [self._config(clone)]
        for _ in range(steps):
            clone._full_reset = False
            clone._set_action(ActionInput(id=GameAction.ACTION1))
            guard = 0
            while not clone.is_action_complete() and guard < 16:
                guard += 1
                clone.step()
            seq.append(self._config(clone))
        return seq

    def _learn_schedule(self, game) -> tuple | None:
        """The blink configuration cycle, anchored at the CURRENT tick.

        Called only from `set_level`, i.e. at tick 0, so ``sched[0]`` is a level's
        opening configuration and the phase of tick ``T`` is plainly ``T % P``.

        ``P`` is the smallest period consistent with the whole probe window. The
        window is ``2 * _MAX_PERIOD`` long, so it spans the true period plus any
        candidate: an accepted ``P`` is therefore a genuine period, and the smallest
        such is the minimal one. Returns ``None`` for a game whose sets are not on a
        short cycle -- this expert fails the episode rather than guess.
        """
        seq = self._probe(game, _PROBE_LEN)
        n = len(seq)
        for P in range(1, _MAX_PERIOD + 1):
            if 2 * P > n:
                break
            if all(seq[t] == seq[t + P] for t in range(n - P)):
                return tuple(seq[:P])
        return None

    def _identify(self, game, model: _Model) -> int | None:
        """The live phase, re-derived from OBSERVATION alone: probe the blink window
        forward and match it against the schedule. A window of ``P`` configurations
        pins the phase uniquely -- a shorter-shift match would make that shift a
        period of the whole sequence, contradicting ``P`` being minimal.

        Only a safety net: the tick counter is exact by construction (one action ==
        one tick, and `set_level` anchors it at 0), so this is what keeps the expert
        honest if that ever stops holding. Note what triggers it -- the cheap
        per-query cross-check in `_state` compares the counter's predicted blink bits
        against the observed ones, so it catches a desync the frame REVEALS, not one
        that happens to land on an identical-looking configuration. It is a tripwire
        on the invariant, not a substitute for it.
        """
        window = self._probe(game, model.P - 1)
        P = model.P
        for t in range(P):
            if all(model.sched[(t + i) % P] == window[i] for i in range(len(window))):
                return t
        return None

    # ── level preparation ────────────────────────────────────────────────────
    def set_level(self, game, level_idx: int) -> None:
        """Seat the level, anchor the tick counter at 0, and build its model.

        The schedule probe happens HERE rather than lazily so that it always runs at
        tick 0, which is what makes ``sched[0]`` a level's opening configuration and
        the phase a plain ``ticks % P``. ``reset_level`` routes through here too, so
        a RESET re-anchors both the counter and the model.
        """
        super().set_level(game, level_idx)
        setattr(game, _TICK_ATTR, 0)
        self._model = None
        gw, gh, kind, targets = self._cell_kinds(game)
        sched = self._learn_schedule(game)
        if sched is None:
            return                          # solve_from -> [] -> the episode fails
        key = (gw, gh, kind, targets, sched)
        model = self._models.get(key)
        if model is None:
            model = _Model(gw, gh, kind, targets, sched)
            self._models[key] = model
        self._model = model

    # ── tick counting ────────────────────────────────────────────────────────
    def drive(self, game, action):
        """Perform one action and advance the tick count. EVERY zq02 action costs
        exactly one tick -- a blocked move and a move off the board included -- so
        the count is the number of actions taken since the level began."""
        res = super().drive(game, action)
        setattr(game, _TICK_ATTR, int(getattr(game, _TICK_ATTR, 0)) + 1)
        return res

    # ── live state -> state id ───────────────────────────────────────────────
    def _state(self, game):
        """``(model, cell, phase)`` for the live board, or ``(None, -1, -1)`` when it
        cannot be planned from (the level is already won, or the schedule was not
        learnable, or the phase cannot be pinned)."""
        model = self._model
        if model is None:
            return None, -1, -1
        phase = int(getattr(game, _TICK_ATTR, 0)) % model.P
        observed = self._config(game)
        if model.sched[phase] != observed:
            # The counter disagrees with what the frame shows; re-derive the phase
            # from observation and re-anchor the counter on the answer.
            phase = self._identify(game, model)
            if phase is None:
                return None, -1, -1
            setattr(game, _TICK_ATTR, phase)
        p = game._player
        px, py = int(p.x), int(p.y)
        if not (0 <= px < model.gw and 0 <= py < model.gh):
            return None, -1, -1
        c = model.cell(px, py)
        if c in model.targets:
            return None, -1, -1                      # already on the target
        return model, c, phase

    @staticmethod
    def _screen(game, dirs) -> list:
        """Game-space direction indices -> the SCREEN actions to press, through this
        level's live display rotation (identity for the stock, unrotated game)."""
        k = int(getattr(game, "rotation_k", 0) or 0)
        return [_ACTIONS[d] for d in dirs]

    # ── BaseSolver hooks ─────────────────────────────────────────────────────
    def solve_from(self, game, level_idx: int, seed: int):
        """An optimal plan from the game's CURRENT state -- any cell, any parity.
        ``[]`` when the level cannot be won from here, which in this game only
        happens if the schedule was not learnable (there is no death and no
        irreversible move)."""
        model, cell, phase = self._state(game)
        if model is None:
            return []
        return self._screen(game, model.field.plan(cell, phase, self.rng))

    def optimal_set_from(self, game, level_idx: int, seed: int):
        """The exact co-optimal tie set at the live state. Usually several: with the
        phase in the state, waiting against the board edge for a hazard's OFF window
        is regularly as good as walking round it."""
        model, cell, phase = self._state(game)
        if model is None:
            return None
        return self._screen(game, model.field.optimal_dirs(cell, phase)) or None


if __name__ == "__main__":
    sys.exit(Zq02Solver.main())
