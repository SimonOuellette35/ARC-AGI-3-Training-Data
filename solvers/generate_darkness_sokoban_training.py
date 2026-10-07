"""Generate Phase-1 training data for the PuzzleScript game ps:darkness_sokoban
("Darkness Sokoban" by Stephen Lavelle) -- a PARTIALLY OBSERVABLE Sokoban.

Each solved seed yields one multi-level episode JSON in the shared schema::

    {
      "game_id": "puzzlescript_darkness_sokoban",
      "levels": [
        {"level_id": 0,
         "observations": [[[c, ...], ...], ...],   # [T, 64, 64] palette idx
         "actions":      [a_0, ..., a_{T-1}]},     # length T
        ...
      ]
    }

``actions[0]`` is the RESET that produced ``observations[0]``; ``actions[i]`` for
i>=1 took the agent from ``observations[i-1]`` to ``observations[i]``. The
recorded index is the *screen* action (post rotation/flip remap), so replaying
the recorded actions reproduces the recorded frames exactly.


THE GAME
========
Mechanically this is textbook Sokoban -- walk, push one crate at a time, win when
every Target holds a Crate -- played with the lights off. Every object the board
is made of (``Background``, ``Target``, ``Wall``, ``Crate``) is painted BLACK,
which is also the frame's background colour, so the whole board renders as an
empty black square. What you can see is drawn by a second, parallel set of
"memory" objects (``BackgroundM`` green, ``TargetM`` blue ring, ``WallM`` brown
checker, ``CrateM`` orange ring) which the rules paint onto the player's cell and
its four orthogonal neighbours, and *wipe off everything else*, every single
turn::

    [Background Backgroundm|No Player] -> [Background No Backgroundm|No Player]
    ... (one per object; these run BEFORE movement, so the board goes dark)
    LATE[Background No Backgroundm|Player] -> [Background Backgroundm|Player]
    ... (one per object; these run AFTER, so the plus around the new cell lights)

So the observation is a **five-cell plus centred on the player** -- nothing else.
Verified exact against the interpreter (`--fuzz`): over random play on both
levels the lit set is always ``{player} | neighbours(player)``, clipped to the
grid, and the crate/player dynamics are exactly the plain-Sokoban model.

Three consequences that shape the expert:

  * **The player's own cell is ambiguous.** ``TargetM``'s blue ring occupies
    exactly the 3x3 the player sprite fills, so a player standing on a target
    renders pixel-identically to a player standing on floor. You only learn that
    the cell you were on was a target once you step OFF it and look back. The
    reader models this honestly (it returns the two-element possibility set), and
    the expert simply keeps exploring until nothing is left ambiguous.
  * **Terrain is static, so knowledge only grows.** Walls and targets never move,
    and a crate moves only when the player pushes it -- which the player is
    standing next to by construction, so every crate move is observed. The
    hidden state is therefore not a probability distribution (as in
    [[blind-maze-a1-solver]]) but a partial MAP, and the "filter" is a set union.
  * **A bump is free.** ``perform_action`` only increments the adapter's action
    counter when the engine grid actually changed, so walking into a wall costs
    nothing against the 200-action budget -- but it still runs a turn, so it
    still lights the plus. That is why the very first press of a level is always
    informative whichever direction it is.

At level load nothing has run yet, so frame 0 of every level shows the player
alone on a black field: not even the cell under it is lit.


THE EXPERT: MAP FIRST, THEN SOLVE
=================================
This is a partially observable game and it is solved as one -- see
[[no-privileged-solvers]]. The expert never reads the interpreter's grid for a
cell it has not lit; `CellReader` composites each cell through the adapter's own
`_render_cell_sprite` and looks the resulting pixel block up in a table built
from every cell state the game can produce, so two states the FRAME cannot tell
apart come back as an ambiguous pair rather than as an answer. A cell whose block
equals the empty (all-black) block is simply not in the observation.

The policy has two phases and a distance oracle for each, which is what makes
every recorded step come with a real optimal SET (see [[always-emit-optimal-targets]]):

  1. **Explore.** A cell is *resolved* by standing on one of its neighbours, so
     the sensing positions are the standable cells that still have an unresolved
     neighbour. The expert walks to the nearest one over the pushless walk graph
     (crates are obstacles: a push is the only irreversible act in Sokoban and
     there is nothing to gain by making one while mapping). The optimal set is
     every direction that strictly decreases the distance to the *nearest*
     sensing position, so all shortest routes to all equally-near frontier cells
     are labelled -- and because that distance strictly decreases every step, the
     walk can never dither.
     Exploration must be exhaustive: the win condition is "all targets covered",
     so an unvisited pocket could always hold one more target. It costs ~21 steps
     on level 0 and ~22 on level 1, against solves of 11 and 17.
  2. **Solve.** With the map complete, the remaining unknown cells are treated as
     wall and the level is plain Sokoban on <=20 free cells. That assumption is
     exact for anything the player can walk to -- a reachable cell would still be
     a frontier -- and it is an assumption only about cells that can be reached
     *by pushing a crate out of the way*, which neither shipped level has. If one
     ever did hide a target there, the expert would solve its known board, the
     engine would not declare a win, and the level would end in RESET and then be
     abandoned: a dropped seed, not a mislabelled one.
     A backward BFS from every winning state over reversed moves gives the EXACT
     distance-to-win field for the whole state space (~20k states, tens of
     milliseconds), so the plan is optimal, the optimal set is
     ``{a : dist(succ(a)) == dist - 1}`` in O(1), and "is this position still
     winnable?" is a dict lookup -- which is what makes burst recovery decidable.

If the field says the current state cannot win (a crate pushed into a corner by
an exploration press), the expert's answer is RESET, and that is what it records.


RECOVERY  (``recovery_mode = "replan"``)
========================================
A map that is built from observations does not care who chose the action, so a
detour is not damage: the expert absorbs whatever the detour revealed and
re-plans from wherever it left off. The two perturbation channels differ only in
what they are allowed to do to the crates:

  * the episode-wide exploration PREFIX runs with the full action set (1-5;
    ACTION5 is a dead key in this game and records as a null transition). It may
    well shove a crate somewhere useless. That is fine and it is the point: the
    expert finishes mapping, finds the position unwinnable, records a RESET and
    solves the level properly -- the human "flail, get stuck, hit reset" arc. A
    cheap corner-deadlock test on the KNOWN map short-circuits the obvious cases
    so the reset does not cost a full exploration first.
  * perturbation BURSTS are gated on the phase, because deadlock is only
    decidable once the map is complete. During the SOLVE phase a burst may press
    anything: the distance field afterwards is an exact solvability oracle, so an
    unrecoverable burst is rolled back exactly as `BaseSolver` does it (engine
    snapshot restored, the tentatively recorded frames truncated, the expert's
    map rewound with them so it cannot act on knowledge the recording no longer
    shows). During the EXPLORE phase, where no such oracle exists yet, bursts are
    restricted to directions that do not push a crate -- always survivable,
    always informative, and no rollback needed.

Both prefix and burst steps carry the expert's own optimal set as ``optimal``, so
every recorded step has a target to train on.


AUGMENTATION
============
The engine state after reset is identical for every seed (the levels are fixed
ASCII maps), so the only per-(seed, level) variable is presentation: the frame
rotation plus an independent horizontal and vertical flip with the matching
directional action remap. Darkness Sokoban is added to
`PuzzleScriptAdapter._FLIP_GAMES` on the same grounds as Bad Example and Count
Mover -- gravity-free, screen-relative input, every rule stated over the relative
force ``>`` and therefore symmetric in both axes, a positional-in-no-way
``All Target on Crate`` win, and no sprite whose mirror image is another object's
art. That takes 2 levels from 8 presentations to 32. No colour augmentation:
green floor / blue target / orange crate / brown wall is the entire readout of a
game whose whole point is what you can and cannot see, and a flattening recolor
could only take information away.

Usage (run from the repo root):
    python solvers/generate_darkness_sokoban_training.py --episodes 200 \
        --out data/training_multi_level/darkness_sokoban

    python solvers/generate_darkness_sokoban_training.py --plans   # per-level report
    python solvers/generate_darkness_sokoban_training.py --fuzz    # model vs interpreter
    python solvers/generate_darkness_sokoban_training.py --audit   # render legibility
"""

from __future__ import annotations

import random
import sys
from collections import deque
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arcengine import ActionInput, GameAction, GameState            # noqa: E402
from adapters.puzzlescript_adapter import (PuzzleScriptAdapter,     # noqa: E402
                                           _render_cell_sprite,
                                           _render_frame)
from solvers.base_solver import BaseSolver, _ID_TO_GAMEACTION       # noqa: E402
from solvers.common.ps_astar import screen_action                   # noqa: E402
from utils.explore import (EpsilonSchedule, ExplorationPolicy,       # noqa: E402
                           RESET_ACTION)
from utils.rotation import remap_action_full                        # noqa: E402

GAME_NAME = "Darkness_Sokoban"

DIRECTIONS = ("up", "down", "left", "right")
DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
ACTION_TO_DIR = {GameAction.ACTION1: "up", GameAction.ACTION2: "down",
                 GameAction.ACTION3: "left", GameAction.ACTION4: "right",
                 GameAction.ACTION5: "action"}

#: The frame's background / letterbox palette index. Every object of the *board*
#: (Background, Target, Wall, Crate) is painted this colour, which is exactly why
#: an unlit cell is unreadable.
DARK = 5

#: What a cell can be. ``terrain`` is permanent, ``occupant`` is what sits on it,
#: ``lit`` is whether this turn's rules painted the memory sprites onto it. Note
#: the level parser puts ``Background`` under EVERY cell, targets included, so a
#: lit target carries both BackgroundM (green) and TargetM (blue ring).
TERRAINS = ("floor", "target")
OCCUPANTS = ("none", "wall", "crate", "player")


def combo_objects(terrain: str, occupant: str, lit: bool) -> list[str]:
    """The PuzzleScript objects present in a cell in state (terrain, occupant, lit)."""
    names = ["background"]
    if terrain == "target":
        names.append("target")
    if occupant != "none":
        names.append(occupant)
    if lit:
        names.append("backgroundm")
        if terrain == "target":
            names.append("targetm")
        if occupant in ("wall", "crate"):
            names.append(occupant + "m")
    return names


# ---------------------------------------------------------------------------
# Observation: what the 64x64 frame actually shows, and nothing else
# ---------------------------------------------------------------------------

class CellReader:
    """Reads the board the way a viewer reads it: by the pixels a cell renders.

    For every one of the game's cell states this composites the cell through the
    adapter's own `_render_cell_sprite` -- the exact function the frame renderer
    calls, at the exact ``cell_px`` the level renders at -- and keys a table by
    the resulting pixel block. `read` then classifies each cell of the live grid
    by ITS block. Two states that render the same block therefore come back as
    the same (multi-element) possibility set, which is the whole honesty
    contract: the reader cannot resolve anything the frame does not.

    The rotation/flip augmentation is a bijection on the frame, so it merges no
    two blocks and is left to `screen_action` on the way out.
    """

    def __init__(self, game: PuzzleScriptAdapter) -> None:
        self.game = game
        self.ps = game._game
        engine = game._engine
        self.cell_px = max(1, min(64 // engine.height, 64 // engine.width))
        self._players = set(engine._player_indices)
        self._layers = engine._obj_layers
        self._blocks: dict[frozenset, np.ndarray] = {}

        idx = self.ps.obj_name_to_idx
        missing = [n for n in ("background", "target", "wall", "crate", "player",
                               "backgroundm", "targetm", "wallm", "cratem")
                   if n not in idx]
        if missing:
            raise RuntimeError(f"{GAME_NAME}: objects {missing} are gone -- "
                               "re-audit CellReader before generating")

        self.table: dict[bytes, frozenset] = {}
        self._states: dict[tuple, bytes] = {}
        for terrain in TERRAINS:
            for occupant in OCCUPANTS:
                for lit in (False, True):
                    names = combo_objects(terrain, occupant, lit)
                    key = self.block(frozenset(idx[n] for n in names)).tobytes()
                    self._states[(terrain, occupant, lit)] = key
                    self.table.setdefault(key, set()).add((terrain, occupant))
        self.table = {k: frozenset(v) for k, v in self.table.items()}
        self.empty = self._states[("floor", "none", False)]
        self._check_legibility()

    def _check_legibility(self) -> None:
        """Assert the three properties the expert's honesty rests on. Each is a
        statement about the RENDERER, so a future sprite or palette edit stops
        this generator instead of quietly turning it into a privileged solver
        (or into a blind one)."""
        # 1. Darkness is total: an unlit cell with no player in it renders
        #    exactly like the empty frame, whatever it is hiding.
        leaks = [(t, o) for t in TERRAINS for o in OCCUPANTS if o != "player"
                 and self._states[(t, o, False)] != self.empty]
        if leaks:
            raise RuntimeError(f"{GAME_NAME}: unlit {leaks} render now -- the "
                               "board is no longer dark, re-audit CellReader")
        # 2. A lit cell is fully legible: the five things the plus can show are
        #    pairwise distinct and none of them is the empty block.
        lit = {(t, o): self._states[(t, o, True)]
               for t, o in [("floor", "none"), ("target", "none"),
                            ("floor", "wall"), ("floor", "crate"),
                            ("target", "crate")]}
        if len(set(lit.values())) != len(lit) or self.empty in lit.values():
            raise RuntimeError(
                f"{GAME_NAME}: lit cell states collide "
                f"({[k for k, v in lit.items()]} -> {len(set(lit.values()))} "
                "distinct blocks) -- the plus is no longer readable")
        # 3. The one thing that IS hidden stays hidden, and stays SO: the player
        #    sprite covers TargetM exactly, so the cell under the player is
        #    ambiguous. The expert resolves it by stepping off; if a sprite edit
        #    ever exposed it the expert would still be correct, just pessimistic,
        #    so this is a documentation check rather than a hard requirement.
        for lit_flag in (False, True):
            poss = self.table[self._states[("floor", "player", lit_flag)]]
            if poss != frozenset({("floor", "player"), ("target", "player")}):
                raise RuntimeError(
                    f"{GAME_NAME}: the cell under the player now reads as "
                    f"{sorted(poss)} -- update CellReader's docs and Knowledge")

    def block(self, cellset: frozenset) -> np.ndarray:
        """The cell_px x cell_px pixel block this set of objects renders as."""
        cached = self._blocks.get(cellset)
        if cached is not None:
            return cached
        others, players = [], []
        for i in cellset:
            obj = self.ps.objects.get(self.ps.obj_idx_to_name.get(i, ""))
            if obj is None:
                continue
            (players if i in self._players else others).append(
                (self._layers.get(i, 0), obj))
        others.sort(key=lambda x: x[0])
        players.sort(key=lambda x: x[0])
        out = _render_cell_sprite(others + players, self.cell_px, None, DARK)
        self._blocks[cellset] = out
        return out

    def read(self) -> dict:
        """``{cell: possibility set}`` for every cell that renders something.
        Cells absent from the result are cells the frame leaves black."""
        out = {}
        for r, row in enumerate(self.game._engine.grid):
            for c, cell in enumerate(row):
                key = self.block(frozenset(cell)).tobytes()
                if key == self.empty:
                    continue
                poss = self.table.get(key)
                if poss is None:
                    raise RuntimeError(
                        f"{GAME_NAME}: cell {(r, c)} renders a block no cell "
                        "state of this game produces -- re-audit CellReader")
                out[(r, c)] = poss
        return out


# ---------------------------------------------------------------------------
# The map the expert builds out of those observations
# ---------------------------------------------------------------------------

@dataclass
class Knowledge:
    """Everything the expert has seen. Terrain is permanent and crates only move
    when pushed (which is always observed), so absorbing an observation is a set
    union -- nothing here is ever revised, only added to."""

    #: cell -> "wall" / "floor" / "target", for cells resolved from a neighbour.
    terrain: dict = field(default_factory=dict)
    #: cells currently holding a crate.
    crates: set = field(default_factory=set)
    #: the player's cell (its sprite is always drawn, so this is never hidden).
    player: object = None

    def copy(self) -> "Knowledge":
        return Knowledge(dict(self.terrain), set(self.crates), self.player)

    def absorb(self, view: dict) -> None:
        for cell, poss in view.items():
            terrains = {t for t, _ in poss}
            occupants = {o for _, o in poss}
            if occupants == {"wall"}:
                self.terrain[cell] = "wall"
            elif len(terrains) == 1:
                # Unambiguous terrain: a lit floor/target with nothing on it, or
                # with a crate on it (CrateM's hollow centre shows the ring or
                # the green underneath).
                self.terrain[cell] = next(iter(terrains))
            if occupants == {"crate"}:
                self.crates.add(cell)
            elif "crate" not in occupants:
                self.crates.discard(cell)
            if occupants == {"player"}:
                self.player = cell

    # -- geometry ---------------------------------------------------------
    def standable(self, cell) -> bool:
        """Can the player be on ``cell``? The player's own cell counts even when
        its terrain is still ambiguous -- it is standing on it."""
        if cell in self.crates:
            return False
        return self.terrain.get(cell) in ("floor", "target") or cell == self.player

    def free_cells(self) -> set:
        return {c for c, t in self.terrain.items() if t in ("floor", "target")}

    def targets(self) -> frozenset:
        return frozenset(c for c, t in self.terrain.items() if t == "target")

    def unresolved(self, cell) -> bool:
        return cell not in self.terrain

    def frontier(self) -> set:
        """The sensing positions: standable cells with an unresolved neighbour.

        Standing on a cell resolves all four of its neighbours at once, so this
        is exactly the set of places still worth walking to. The player's own
        cell is included as a source (it may be the only way through) but never
        as a destination -- being there is what left its neighbours resolved."""
        out = set()
        cells = [c for c, t in self.terrain.items() if t != "wall"]
        if self.player is not None:
            cells.append(self.player)
        for cell in cells:
            if cell in self.crates:
                continue
            if any(self.unresolved((cell[0] + dr, cell[1] + dc))
                   for dr, dc in DELTA.values()):
                out.add(cell)
        return out

    def deadlocked(self) -> bool:
        """Is a crate wedged into a corner of KNOWN walls, off a target?

        The classic Sokoban corner test, restricted to walls we have actually
        seen (an unresolved neighbour is never assumed to be a wall). It is only
        a short-circuit: once the map is complete the distance field decides
        solvability exactly. Its job is to catch a prefix that shoved a crate
        into a corner before the expert spends twenty steps mapping."""
        for crate in self.crates:
            if self.terrain.get(crate) == "target":
                continue
            vert = any(self.terrain.get((crate[0] + dr, crate[1])) == "wall"
                       for dr in (-1, 1))
            horiz = any(self.terrain.get((crate[0], crate[1] + dc)) == "wall"
                        for dc in (-1, 1))
            if vert and horiz:
                return True
        return False


def walk_distances(know: Knowledge, sources) -> dict:
    """BFS distances to the nearest of ``sources`` over the PUSHLESS walk graph
    (crates are obstacles). Symmetric, so one sweep from the sources answers
    "how far is the player from the nearest one" and "which step shortens it"."""
    dist = {s: 0 for s in sources}
    queue = deque(dist)
    while queue:
        cell = queue.popleft()
        for dr, dc in DELTA.values():
            nxt = (cell[0] + dr, cell[1] + dc)
            if nxt in dist or not know.standable(nxt):
                continue
            dist[nxt] = dist[cell] + 1
            queue.append(nxt)
    return dist


# ---------------------------------------------------------------------------
# Sokoban over the mapped part of the board
# ---------------------------------------------------------------------------

#: Cap on the number of winning crate configurations enumerated to seed the
#: backward BFS. Both shipped levels have exactly as many crates as targets, so
#: there is precisely ONE winning configuration; the cap only exists so a level
#: with many spare crates degrades to "no field" (keep exploring / reset) instead
#: of hanging.
WIN_STATE_CAP = 20_000


def push_successor(state, direction: str, free: frozenset):
    """One Sokoban move, or None when the press would change nothing (a bump into
    a wall, or a push blocked by a wall or a second crate)."""
    (pos, crates) = state
    dr, dc = DELTA[direction]
    ahead = (pos[0] + dr, pos[1] + dc)
    if ahead not in free:
        return None
    if ahead in crates:
        beyond = (ahead[0] + dr, ahead[1] + dc)
        if beyond not in free or beyond in crates:
            return None
        return (ahead, frozenset((crates - {ahead}) | {beyond}))
    return (ahead, crates)


class WinField:
    """Exact distance-to-win over the whole reachable Sokoban state space.

    Built by a backward BFS from every winning state: the predecessor of
    ``(pos, crates)`` under ``direction`` is the player one cell back, either
    having plain-walked (crates unchanged) or having pushed the crate that now
    sits one cell ahead. The state space is |free| x C(|free|, k) -- ~20k for
    these 7x7 boards -- so the whole field costs tens of milliseconds and is
    built once per level, not once per step.

    It gives three things a forward search would each have to re-derive: an
    optimal plan, the optimal SET at every state in O(1), and an exact
    "is this position still winnable?" test, which is what makes burst rollback
    decidable."""

    def __init__(self, free: frozenset, targets: frozenset, n_crates: int) -> None:
        self.free = free
        self.targets = targets
        self.dist: dict = {}
        spare = n_crates - len(targets)
        if spare < 0 or not targets or not targets <= free:
            return                      # not enough crates found yet (or none)
        rest = sorted(free - targets)
        if len(rest) < spare:
            return
        seeds = []
        for extra in combinations(rest, spare):
            crates = frozenset(targets | set(extra))
            for pos in free - crates:
                seeds.append((pos, crates))
            if len(seeds) > WIN_STATE_CAP:
                return
        self.dist = {s: 0 for s in seeds}
        queue = deque(seeds)
        while queue:
            state = queue.popleft()
            (pos, crates) = state
            here = self.dist[state]
            for dr, dc in DELTA.values():
                prev = (pos[0] - dr, pos[1] - dc)
                if prev not in free or prev in crates:
                    continue
                back = (prev, crates)               # the player just walked
                if back not in self.dist:
                    self.dist[back] = here + 1
                    queue.append(back)
                ahead = (pos[0] + dr, pos[1] + dc)  # ... or just pushed
                if ahead in crates and pos not in crates:
                    moved = frozenset((crates - {ahead}) | {pos})
                    back = (prev, moved)
                    if back not in self.dist:
                        self.dist[back] = here + 1
                        queue.append(back)

    def winnable(self, state) -> bool:
        return state in self.dist

    def optimal(self, state):
        """``(directions, distance)`` -- every move that gets one step closer to
        a win. ``([], 0)`` when the state already wins, ``(None, None)`` when it
        cannot."""
        here = self.dist.get(state)
        if here is None:
            return None, None
        if here == 0:
            return [], 0
        opts = [d for d in DIRECTIONS
                if self.dist.get(push_successor(state, d, self.free), -1) == here - 1]
        return opts, here


# ---------------------------------------------------------------------------
# The expert
# ---------------------------------------------------------------------------

class DarknessExpert:
    """Map-then-solve policy over the partial map. `decide` returns
    ``(phase, directions)`` where ``phase`` is "explore", "solve" or "reset" and
    ``directions`` is the full set of equally-optimal choices at this state."""

    def __init__(self, game: PuzzleScriptAdapter) -> None:
        self.game = game
        self.reader = CellReader(game)
        self.know = Knowledge()
        self._field: WinField | None = None
        self._field_key = None

    # -- lifecycle ---------------------------------------------------------
    def start_level(self) -> None:
        """(Re)start from a blacked-out board, at a level start or after RESET."""
        self.know = Knowledge()
        self._field = None
        self._field_key = None
        self.observe()

    def observe(self) -> None:
        self.know.absorb(self.reader.read())

    def checkpoint(self):
        """Everything a rolled-back burst has to put back. The field is keyed by
        the terrain it was built from, so it rebuilds itself on demand."""
        return self.know.copy()

    def restore(self, snap) -> None:
        self.know = snap

    # -- the two distance oracles -----------------------------------------
    def field(self) -> WinField:
        """The distance-to-win field for the CURRENT map, rebuilt whenever the
        map has grown (which during a level is at most a handful of times)."""
        free = frozenset(self.know.free_cells())
        targets = self.know.targets()
        key = (free, targets, len(self.know.crates))
        if self._field_key != key:
            self._field = WinField(free, targets, len(self.know.crates))
            self._field_key = key
        return self._field

    def state(self):
        return (self.know.player, frozenset(self.know.crates))

    def explore_options(self):
        """``(directions, distance)`` toward the nearest sensing position, or
        ``(None, None)`` when there is nothing left to map (or nothing reachable
        without pushing)."""
        if not self.know.terrain:
            # Level start: the board is black, not even our own cell is lit. The
            # first press turns the lights on whichever way it goes -- and with
            # zero observations there is nothing to prefer -- so all four are
            # equally optimal.
            return list(DIRECTIONS), 0
        targets = self.know.frontier() - {self.know.player}
        if not targets:
            return None, None
        dist = walk_distances(self.know, targets)
        here = dist.get(self.know.player)
        if here is None:
            return None, None           # only reachable by pushing; give up on it
        opts = [d for d in DIRECTIONS
                if dist.get((self.know.player[0] + DELTA[d][0],
                             self.know.player[1] + DELTA[d][1]), here) == here - 1]
        return (opts, here) if opts else (None, None)

    # -- policy ------------------------------------------------------------
    def decide(self):
        """``(phase, directions)``. An empty direction list never comes back:
        when the expert has nothing useful to do the phase is "reset" and the
        target is the RESET action, which is genuinely the best move available."""
        if self.know.deadlocked():
            return "reset", []
        opts, _ = self.explore_options()
        if opts:
            return "explore", opts
        opts, _ = self.field().optimal(self.state())
        if opts:
            return "solve", opts
        return "reset", []

    def recoverable(self) -> bool:
        """Can the level still be won from here? Exact once the map is complete
        (the field decides); during exploration only the corner test can speak,
        so this is the conservative "nothing says otherwise" answer."""
        if self.know.deadlocked():
            return False
        opts, _ = self.explore_options()
        if opts:
            return True
        return self.field().winnable(self.state())

    def pushless_directions(self) -> list:
        """Directions that move the player without touching a crate -- the
        reversible half of the action space, which is what a burst is allowed to
        do while the map is too incomplete to judge a deadlock."""
        pos = self.know.player
        if pos is None:
            return []
        out = []
        for d in DIRECTIONS:
            nxt = (pos[0] + DELTA[d][0], pos[1] + DELTA[d][1])
            if nxt not in self.know.crates and self.know.standable(nxt):
                out.append(d)
        return out


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def _frame_to_list(frame) -> list:
    return np.asarray(frame).tolist()


class DarknessSokobanSolver(BaseSolver):
    """`BaseSolver` for ps:darkness_sokoban.

    Like the other ps: generators this drives an external interpreter rather than
    a native `ARCBaseGame`, and the expert is a policy over a partial map rather
    than a plan over a visible board -- so `solve_episode` is overridden and only
    the CLI (`main`/`run`), the validation gate and the
    ``{"game_id", "levels": [...]}`` save schema come from the base."""

    game_id = "puzzlescript_darkness_sokoban"
    game_name = GAME_NAME

    #: Levels never attempted. Empty: the expert wins both.
    skip_levels: frozenset = frozenset()

    #: Hard cap on recorded steps per level (across resets). Mapping + solving
    #: costs 32 and 39 steps on the two levels, so this is a runaway guard, not a
    #: budget. The adapter's own 200 *effective* actions is the real ceiling, and
    #: a RESET puts that counter back to zero.
    max_steps: int = 160

    supports_recovery = True
    recovery_mode = "replan"
    max_resets = 2

    # -- plumbing ----------------------------------------------------------
    def make_game(self, seed: int):
        return PuzzleScriptAdapter(self.game_name, seed=seed)

    def solve_from(self, game, level_idx: int, seed: int) -> list:
        """Not used: this generator drives its own map-then-solve loop (the
        expert acts on a partial map that only the recorded action/observation
        stream builds), so `solve_episode` is overridden and the base record loop
        never runs."""
        raise NotImplementedError(
            "darkness_sokoban records through its own mapping loop; "
            "see DarknessSokobanSolver.record_level")

    # -- recording ---------------------------------------------------------
    @staticmethod
    def _step(index: int, *, phase: str, changed: bool, optimal) -> dict:
        return {"type": "simple", "index": int(index), "phase": phase,
                "changed": bool(changed), "n_obs": 1,
                "optimal": None if optimal is None else
                [{"type": "simple", "index": int(a)} for a in optimal]}

    def _screen(self, game, direction: str) -> GameAction:
        """The press that makes the engine move ``direction``. The expert plans
        in engine coordinates, the adapter forward-remaps directional input by
        the level's rotation/flips, and what gets RECORDED is this screen press
        -- get it backwards and nothing raises, the recording is just wrong on
        7 of every 8 presentations. See `solvers/common/ps_astar.screen_action`."""
        return screen_action(direction, game._rotation_k, game._hflip, game._vflip)

    def record_level(self, game, level: int, *, schedule=None, exploration=None):
        """Drive one level to a WIN with the mapping expert, recording as we go.

        Returns ``(observations, actions)``, or ``(None, None)`` when the level
        was not won (the step cap, the adapter's action budget, or an
        exploration press that bricked the board more often than
        ``max_resets`` allows)."""
        game.set_level(level)
        expert = DarknessExpert(game)
        expert.start_level()

        observations = [_frame_to_list(game._current_frame)]
        actions = [self._step(RESET_ACTION, phase="reset", changed=False,
                              optimal=None)]
        prev = np.asarray(game._current_frame)
        resets = 0

        def optimal_now() -> list:
            """The expert's target at the CURRENT state, as screen action ids."""
            _, dirs = expert.decide()
            if not dirs:
                return [RESET_ACTION]
            return [int(self._screen(game, d).value) for d in dirs]

        def drive(action: GameAction, phase: str, optimal) -> None:
            nonlocal prev
            fd = game.perform_action(ActionInput(id=action))
            frame = np.asarray(fd.frame[-1] if fd.frame else game._current_frame)
            observations.append(_frame_to_list(frame))
            actions.append(self._step(int(action.value), phase=phase,
                                      changed=not np.array_equal(frame, prev),
                                      optimal=optimal))
            prev = frame
            expert.observe()

        def do_reset() -> None:
            nonlocal prev
            game.perform_action(ActionInput(id=GameAction.RESET))
            frame = np.asarray(game._current_frame)
            observations.append(_frame_to_list(frame))
            actions.append(self._step(RESET_ACTION, phase="reset",
                                      changed=not np.array_equal(frame, prev),
                                      optimal=[RESET_ACTION]))
            prev = frame
            expert.start_level()

        def burst(phase: str) -> None:
            """A run of random presses, with the rollback contract `BaseSolver`
            uses.

            During the SOLVE phase anything goes: the distance field afterwards
            says exactly whether the level is still winnable, and a burst that
            says no is rewound -- engine snapshot restored, the tentatively
            recorded frames truncated, and the expert's map rewound with them so
            it cannot go on acting on knowledge the recording no longer contains.
            During the EXPLORE phase there is no such oracle yet, so the burst is
            drawn only from the directions that do not push a crate: always
            reversible, always informative, never in need of a rollback."""
            nonlocal prev
            pool = (expert.pushless_directions() if phase == "explore"
                    else list(DIRECTIONS))
            if not pool:
                return
            snapshot = game._snapshot()
            know = expert.checkpoint()
            n_obs, n_act, prev_frame = len(observations), len(actions), prev

            length = max(1, int(self.rng.expovariate(
                1.0 / max(1, self._burst_mean))))
            for _ in range(length):
                drive(self._screen(game, self.rng.choice(pool)), "burst",
                      optimal_now())
                if game._state != GameState.NOT_FINISHED:
                    break

            if game._state == GameState.WIN or expert.recoverable():
                return
            game._restore(snapshot)
            expert.restore(know)
            del observations[n_obs:]
            del actions[n_act:]
            prev = prev_frame

        # ---- exploration prefix (episode-wide budget, full action set) ----
        while (schedule is not None and exploration is not None
               and schedule.explore()):
            schedule.advance()
            act = exploration.action(prev)
            if act.is_click:                       # this game exposes no click
                continue
            drive(_ID_TO_GAMEACTION[act.action_id], "explore", optimal_now())
            if game._state in (GameState.WIN, GameState.GAME_OVER):
                break

        # ---- expert loop ----
        for _ in range(self.max_steps):
            if game._state == GameState.WIN:
                break
            if game._state == GameState.GAME_OVER:
                return None, None

            phase, dirs = expert.decide()
            if not dirs:
                # Unwinnable from here (an exploration press wedged a crate) --
                # the honest best move is the one a human makes: hit reset.
                if resets >= self.max_resets:
                    return None, None
                resets += 1
                do_reset()
                continue

            targets = [int(self._screen(game, d).value) for d in dirs]

            if self._burst_prob > 0.0 and self.rng.random() < self._burst_prob:
                burst(phase)
                continue

            drive(_ID_TO_GAMEACTION[self.rng.choice(targets)], "expert", targets)

        if game._state != GameState.WIN:
            return None, None
        return observations, actions

    # -- episode -----------------------------------------------------------
    def solve_episode(self, seed: int, explore: bool = True):
        game = self.make_game(seed)
        do_explore = explore and self.supports_recovery
        schedule = (EpsilonSchedule(self.rng, center=self._explore_center,
                                    jitter=self._explore_jitter)
                    if do_explore else None)
        exploration = (ExplorationPolicy(self.available_actions(game), self.rng)
                       if do_explore else None)

        levels = [lvl for lvl in range(game.n_levels)
                  if lvl not in self.skip_levels]
        out = []
        for level in levels:
            obs, acts = self.record_level(game, level, schedule=schedule,
                                          exploration=exploration)
            if obs is None:
                return False, out
            out.append({"level_id": level, "observations": obs,
                        "actions": acts})
        return len(out) == len(levels), out


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def _model_step(walls, crates, pos, direction):
    """The plain-Sokoban transition, for `--fuzz` to check the interpreter against."""
    dr, dc = DELTA[direction]
    ahead = (pos[0] + dr, pos[1] + dc)
    if ahead in walls:
        return crates, pos
    if ahead in crates:
        beyond = (ahead[0] + dr, ahead[1] + dc)
        if beyond in walls or beyond in crates:
            return crates, pos
        return frozenset((crates - {ahead}) | {beyond}), ahead
    return crates, ahead


def _engine_truth(game):
    """The interpreter's own grid -- for verification only, never for the expert."""
    ps = game._game
    names = {v: k for k, v in ps.obj_name_to_idx.items()}
    walls, crates, lit = set(), set(), set()
    player = None
    for r, row in enumerate(game._engine.grid):
        for c, cell in enumerate(row):
            s = {names[i] for i in cell}
            if "wall" in s:
                walls.add((r, c))
            if "crate" in s:
                crates.add((r, c))
            if "player" in s:
                player = (r, c)
            if s & {"backgroundm", "wallm", "cratem", "targetm"}:
                lit.add((r, c))
    return walls, frozenset(crates), player, lit


def _fuzz(steps: int = 400) -> bool:
    """Check the two claims the expert is built on, against the interpreter:
    the dynamics are plain Sokoban, and the lit set is exactly the plus."""
    rng = random.Random(0)
    bad = 0
    for seed in range(4):
        game = PuzzleScriptAdapter(GAME_NAME, seed=seed)
        height, width = game._engine.height, game._engine.width
        for level in range(game.n_levels):
            game.set_level(level)
            walls, crates, pos, lit = _engine_truth(game)
            if lit:
                print(f"seed {seed} level {level}: board is lit at load")
                bad += 1
            for _ in range(steps // game.n_levels):
                if game._state != GameState.NOT_FINISHED:
                    game.set_level(level)
                    walls, crates, pos, lit = _engine_truth(game)
                screen = rng.choice([GameAction.ACTION1, GameAction.ACTION2,
                                     GameAction.ACTION3, GameAction.ACTION4])
                direction = ACTION_TO_DIR[remap_action_full(
                    screen, game._rotation_k, game._hflip, game._vflip)]
                want_crates, want_pos = _model_step(walls, crates, pos, direction)
                game.perform_action(ActionInput(id=screen))
                walls, crates, pos, lit = _engine_truth(game)
                if (crates, pos) != (want_crates, want_pos):
                    print(f"seed {seed} level {level}: dynamics mismatch "
                          f"{direction} -> {pos} {sorted(crates)} "
                          f"(model {want_pos} {sorted(want_crates)})")
                    bad += 1
                plus = {pos} | {(pos[0] + dr, pos[1] + dc)
                                for dr, dc in DELTA.values()}
                plus = {q for q in plus if 0 <= q[0] < height and 0 <= q[1] < width}
                if lit != plus:
                    print(f"seed {seed} level {level}: lit {sorted(lit)} "
                          f"!= plus {sorted(plus)}")
                    bad += 1
                if bad > 8:
                    return False
    print("fuzz: dynamics and lighting match the model" if not bad
          else f"fuzz: {bad} mismatches")
    return bad == 0


def _audit() -> bool:
    """Whole-frame legibility check: for every cell state the plus can show, swap
    it into one cell of a real board and diff the RENDERED 64x64 frames.

    `CellReader` classifies by the cell block, which is the renderer's own unit,
    but the frame is upscaled and letterboxed after that -- so this checks the
    thing the agent actually sees, per the lesson in [[ps-palette-collisions]]:
    verify the frame, not the sprite."""
    ok = True
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    idx = game._game.obj_name_to_idx
    states = [("floor", "none"), ("target", "none"), ("floor", "wall"),
              ("floor", "crate"), ("target", "crate"), ("floor", "player")]
    for level in range(game.n_levels):
        game.set_level(level)
        reader = CellReader(game)
        grid = game._engine.grid
        # A cell away from the border, so no state we paint into it is clipped.
        probe = (game._engine.height // 2, game._engine.width // 2)
        saved = grid[probe[0]][probe[1]]
        frames = {}
        for state in states + [("floor", "dark")]:
            if state[1] == "dark":
                grid[probe[0]][probe[1]] = frozenset({idx["background"]})
            else:
                grid[probe[0]][probe[1]] = frozenset(
                    idx[n] for n in combo_objects(state[0], state[1], True))
            frames[state] = _render_frame(game._engine, game._game).tobytes()
        grid[probe[0]][probe[1]] = saved
        for i, a in enumerate(frames):
            for b in list(frames)[i + 1:]:
                if frames[a] == frames[b]:
                    print(f"level {level}: {a} and {b} render identically "
                          f"at cell_px={reader.cell_px}")
                    ok = False
    print("audit: every lit cell state is distinguishable in the frame" if ok
          else "audit: FAILED")
    return ok


def _report() -> int:
    """Per-level report: how many steps the expert spends mapping vs solving, and
    how much of the adapter's action budget that leaves."""
    solver = DarknessSokobanSolver()
    game = solver.make_game(0)
    for level in range(game.n_levels):
        game.set_level(level)
        expert = DarknessExpert(game)
        expert.start_level()
        counts = {"explore": 0, "solve": 0, "reset": 0}
        ties = 0
        for _ in range(solver.max_steps):
            if game._state != GameState.NOT_FINISHED:
                break
            phase, dirs = expert.decide()
            if not dirs:
                counts["reset"] += 1
                break
            counts[phase] += 1
            ties += len(dirs) > 1
            game.perform_action(ActionInput(
                id=solver._screen(game, dirs[0])))
            expert.observe()
        total = counts["explore"] + counts["solve"]
        room = "ok" if total < game._max_steps else "OVER BUDGET"
        print(f"level {level}: {counts['explore']:3d} mapping + "
              f"{counts['solve']:3d} solving = {total:3d} presses "
              f"(budget {game._max_steps}, {room}), "
              f"{ties:3d} with a tie set ({ties / max(1, total):.0%}), "
              f"{game._state.name}")
    return 0


if __name__ == "__main__":
    if "--fuzz" in sys.argv:
        raise SystemExit(0 if _fuzz() else 1)
    if "--audit" in sys.argv:
        raise SystemExit(0 if _audit() else 1)
    if "--plans" in sys.argv:
        raise SystemExit(_report())
    raise SystemExit(DarknessSokobanSolver.main())
