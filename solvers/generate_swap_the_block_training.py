"""Generate Phase-1 training data for the PuzzleScript game ps:swap_the_block
("Swap the block!", Amber).

The harness -- the rotation contract, the trajectory recorder and the
`BaseSolver` plumbing -- lives in `solvers/common/ps_astar.py`, shared with the
other ps: generators. This file is the game-specific part: the measured
mechanic, a native model of it, the two-sided exact sweep that model is solved
with, the proof that every plan is shortest, and the render audit.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_swap_the_block",
      "levels": [
        {
          "level_id": 0,
          "observations": [[[c, ...], ...], ...],   # [T, 64, 64] palette idx
          "actions":      [a_0, a_1, ..., a_{T-1}], # length T
        },
        ...
      ]
    }

actions[0] is the RESET that produced obs[0]; actions[i] for i>=1 is the action
that took the agent from obs[i-1] to obs[i]. The recorded index is the *screen*
action (post rotation remap), i.e. the button an agent presses in the augmented
view, so replaying the recorded actions reproduces the recorded frames exactly.
Every expert step carries the full set of equally-optimal presses.

The game
--------
Five levels, and the entire rule list is the same single line ps:swap_sokoban
ships (`solvers/generate_swap_sokoban_training.py` -- same sokoban template,
same Background sprite, different author):

    [ > Player | Crate ] -> [ Crate | Player ]

A player that walks into a crate SWAPS with it: the crate takes the square the
player is leaving (free by construction, so a swap can never be blocked) and the
player takes the crate's. Neither side of the right hand side carries a force,
so the swap is done by the rule and the movement phase then moves nothing.

WHAT MAKES THIS A DIFFERENT GAME FROM SWAP SOKOBAN: **there are two to five
players, and every one of them moves on every press.** That is not a cosmetic
difference, and it is why none of swap_sokoban's model could be reused:

* A crate moves TOWARDS whichever player walked into it, so which crates move
  on a press -- and how far -- depends on where all the other players are.
* The rule CASCADES. PuzzleScript re-runs a rule until it stops matching, so a
  RUN of players standing shoulder to shoulder behind a crate resolves as one
  event: with players at ``i .. j`` and a crate at ``j+1``, the crate ends at
  ``i`` and every player has moved one step forward. **A run of k players
  teleports the crate k squares backwards in a single press.** Building a train
  is how the long hauls in levels 2 and 4 are paid for, and it is also why the
  bound "a crate moves at most one square per press" -- true in every ordinary
  sokoban, and the basis of every distance heuristic one would reach for -- is
  false here.
* **The game is NOT reversible.** In swap_sokoban every press is undone by the
  opposite press, so its `_Field` sweeps backward using the forward step
  function. Here a press that some players cannot make (blocked by a wall, by a
  crate or by a player that is itself blocked) while others can is not undone by
  the opposite press, so predecessors have to be computed properly; see
  `_Board.preds`.

The win is ``All Target on Crate`` -- every target square carries a crate. Level
3 has four crates for three targets, so its spare crate may sit anywhere; the
other four levels have as many crates as targets.

`--selfcheck` measures all of the above against the interpreter rather than
reading it off the rules section: ~15k presses from RANDOMLY PLANTED boards (not
only from the level starts, which never exercise a five-deep player train),
comparing the whole board and the win flag after every one.

The turn, as one line per axis
------------------------------
The model is fast because the turn DECOMPOSES. Movement and swaps happen only
along the pressed axis, and nothing crosses a wall, so a press splits the board
into independent maximal wall-free SEGMENTS (rows for left/right, columns for
up/down) and rewrites each one on its own. Within a segment the rule is:

    for each maximal run of players at [i..j]:
        cell j+1 is a crate -> the crate lands at i, the players fill i+1..j+1
        cell j+1 is empty   -> the players fill i+1..j+1, i empties
        the segment ends    -> nothing moves (the run is jammed against a wall)

Runs are independent (they are separated by at least one non-player cell, and a
run's rewrite touches only ``i..j+1``), so a whole segment is a pure function of
its own contents. Segments are at most 9 cells long, so every ``3^L`` content is
tabulated once per level and one press becomes a handful of table lookups --
which is what makes the millions-of-states sweeps below affordable in numpy.

Expert solver
-------------
A native `_Board` plus `_Field`, an EXACT presses-to-win derivation by a
two-sided breadth-first sweep. Both halves are vectorised over whole BFS layers
(states are ``(player bitmask, crate bitmask)`` pairs over the level's free
squares, at most 46 of them, so a state is two uint64s and a layer is two numpy
columns).

  * FORWARD from the level start over `_Board.step`;
  * BACKWARD from the win over `_Board.preds`. The goal states are enumerated
    directly -- crates on every target, spare crates anywhere, players anywhere
    -- rather than searched for: 37 to 749,398 states per level.

The two sides are grown alternately, smaller frontier first, and the sweep stops
when the best crossing found equals the bound ``f + b + 1`` that the two
completed halves prove (a path of length L touches forward depth f, so if that
state is not within b of the win then L > f + b). That is what makes level 4
tractable: it is 46 free squares with five players and five crates -- 1.0e12
states, and its forward space alone passes 12 million by depth 15 -- yet its
d* = 20 falls out of a forward sweep of 16 layers meeting a backward sweep of 4,
59.6 million states and four minutes all told.

Reading the answer back is exact on both sides. Everything at or past the
meeting layer has a true distance from the backward field, and the forward half
is labelled by ONE induction down the recorded layer-to-layer edges: a state at
depth g is on a shortest route iff one of its successors at depth g+1 is. So:

  * the PLAN is the descent of that structure;
  * the OPTIMAL SETS are every press that stays on it, exactly, nothing
    inferred;
  * a state OFF it -- an exploration detour -- is answered exactly when the
    backward field reaches it, and otherwise not at all, which `record_level`
    reads as "no plan" and turns into a RESET.

Shortest, and how that is known
-------------------------------
d* = 16, 18, 24, 17, 20 for the five levels, all provably shortest, 95 presses
in total. The proofs are independent derivations agreeing on every level:

  * this file's two-sided sweep;
  * a plain forward BFS to the first win over the same model, which is shortest
    by construction and shares nothing with the two-sided bookkeeping
    (``--selfcheck`` pass 3; affordable on levels 0-3, and skipped on level 4,
    where the unbounded forward space is the reason the sweep is two-sided);
  * ``--engine``, which walks the whole reachable space on the REAL INTERPRETER
    -- the successor of a board is whatever `PSEngine.step` makes of it, the
    goal test is `check_win`, and no `_Board` call of any kind participates --
    and sweeps backwards over the edges it recorded. It agrees on levels 0 and 1
    (844 and 55,293 boards, and all 11 of their tie sets); the other three are
    the reason the native model exists, at 3.7 million boards and up.

The 17 tie sets are checked twice more in ``--verify``, by two passes that fail
differently: a BACKWARD-ONLY field (no forward half, no cone induction) which
re-derives distances and sets exactly as far as it reaches, and EXECUTABLE
LABELS, which presses every one of them on the interpreter and requires the
field's own continuation to win in exactly one press fewer.

Recovery
--------
`supports_recovery` with the family's RESET-mode prefix: the episode opens with
exploratory presses and ONE RESET back to the level start, from which the plan
replays to a guaranteed WIN.

``epsilon = 0`` here, unlike ps:swap_sokoban, and the reason is the one that
makes this game interesting: it is not reversible, the state space is far too
large to field completely (level 4 is 1.0e12 states), and so an arbitrary
perturbed board cannot be answered. Detours would therefore be probed, found
unplannable and silently dropped -- the recording would be identical and the
sweeps would be paid for nothing. The exploration prefix plus the RESET is the
recovery arc this game supports, which is the same choice the other
search-bound members of this family make.

The derivations are cached to ``data/swap_the_block_plans.json`` (plan AND
optimal sets, per level, keyed by the start layout). With that file present a
generation run never sweeps at all; delete it to re-derive. It is genuinely
load-bearing here -- the cold sweep is minutes and gigabytes on level 4 -- so a
`parallelize_generator` fan-out must share it rather than repeat it per core.

Augmentation
------------
The engine state after reset is identical for every seed (levels are fixed ASCII
maps), so the only per-(seed, level) variables are the frame rotation
(``rotation_k`` in 0..3) and the two flips, with their matching directional
action remap: 5 levels x 16 presentations = 80.

``_Swap_the_block!`` is in `PuzzleScriptAdapter._FLIP_GAMES`, on the same
argument its sibling is already in that set for and which is written out beside
it there: one rule, stated with the relative ``>`` so the dihedral group maps a
match to a match, no gravity, screen-relative input, and a win condition that
names no direction. The extra players do not weaken it -- they are one object
type with no orientation, and a swap still puts the crate on exactly the square
its player vacates, so no two bodies ever contest a cell and the rule-order
chirality other games have to argue around cannot arise. Player and Wall are not
themselves symmetric under the group, which is the same thing that is true of
Swap Sokoban's and is what the mandatory rotation does to them anyway.

``--symmetry`` measures it rather than arguing it: every level's plan AND a
seeded 200-press random walk, replayed at all seven non-identity presentations
with the presses transformed, requiring the players and crates to land on the
transform of where they landed unaugmented after every single press.

Rendering
---------
Nothing had to change, which is worth stating because the sibling game needed a
fix: Swap Sokoban's Target was a ring on rows/cols 1-3 -- exactly the pixels its
Player paints over -- so a player standing on a target rendered as bare floor.
This game's Target is a full 5x5 frame, and both bodies that can stand on one
leave the corners transparent (the Player) or the whole middle transparent (the
Crate, a hollow box), so the goal shows through either. ``--audit`` is the
measurement: it renders every cell COMPOSITION a board of this game can hold --
floor, wall, target, player, crate, and the player and the crate on a target --
as a whole 64x64 frame of a uniform board, at both cell sizes the five levels
render at (7 and 8 px), and requires them pairwise distinct. Whole frames rather
than one cell sliced out of a mixed board: `_render_frame` upscales and
centre-pads, so slicing by ``cell_px`` arithmetic reads the wrong pixels (the
ps:explod lesson).

Usage (run from the repo root):
    python solvers/generate_swap_the_block_training.py --episodes 200 \
        --out data/training_multi_level/swap_the_block

    python solvers/generate_swap_the_block_training.py --plans      # level report
    python solvers/generate_swap_the_block_training.py --selfcheck  # model + optimality
    python solvers/generate_swap_the_block_training.py --verify     # every tie set
    python solvers/generate_swap_the_block_training.py --engine     # interpreter proof
    python solvers/generate_swap_the_block_training.py --audit      # rendering
    python solvers/generate_swap_the_block_training.py --symmetry   # augmentation

``--plans``, ``--audit`` and ``--symmetry`` are seconds. ``--selfcheck`` and
``--verify`` re-derive the sweeps the plan cache exists to avoid, so they cost
minutes and several GB of RAM on level 4; a generation run does neither.
"""

from __future__ import annotations

import itertools
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.puzzlescript_adapter import (                          # noqa: E402
    PuzzleScriptAdapter, _render_frame)
from solvers.common.ps_astar import (PSAStarSolver, PSExpert, Plan,  # noqa: E402
                                     restore, snapshot)

_REPO_ROOT = Path(__file__).resolve().parent.parent

GAME_NAME = "_Swap_the_block!"

#: Engine directions, in the order ties are broken. No rule in this game has
#: ``action`` on a left-hand side, so the X button cannot change the board and
#: branching on it would double every sweep for nothing. ``--selfcheck`` presses
#: it against the interpreter rather than reading that off the rules section, and
#: the exploration prefix presses it too (a live agent has that button).
DIRS: tuple[str, ...] = ("up", "down", "left", "right")

_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

#: A state as a numpy record: the player bitmask and the crate bitmask over the
#: level's free squares. Structured rather than two columns so whole LAYERS can
#: be sorted, uniqued and set-differenced with one call each.
KEY_DT = np.dtype([("p", "<u8"), ("c", "<u8")])

#: Disk cache of every level's start plan AND its optimal-action sets. Genuinely
#: load-bearing (see the module docstring): the cold derivation is minutes on
#: level 4. Delete the file to re-derive it.
PLAN_CACHE: Path = _REPO_ROOT / "data" / "swap_the_block_plans.json"


# ---------------------------------------------------------------------------
# Small array helpers
# ---------------------------------------------------------------------------

def _keys(pmask, cmask) -> np.ndarray:
    """``(player, crate)`` mask columns as one structured array."""
    out = np.empty(len(pmask), dtype=KEY_DT)
    out["p"] = pmask
    out["c"] = cmask
    return out


def _merge(a: np.ndarray, b: np.ndarray):
    """Merge sorted-unique ``b`` into sorted-unique ``a`` (disjoint).

    Returns ``(merged, positions_of_b, mask_of_a)`` so a parallel payload array
    can be merged the same way without a second sort. Linear after one
    searchsorted, which matters: these sweeps re-merge multi-million-element
    arrays once per BFS layer, and a full re-sort per layer is the difference
    between seconds and minutes."""
    pos = np.searchsorted(a, b)
    out = np.empty(len(a) + len(b), dtype=a.dtype)
    at = pos + np.arange(len(b))
    out[at] = b
    keep = np.ones(len(out), dtype=bool)
    keep[at] = False
    out[keep] = a
    return out, at, keep


def _found(sorted_keys: np.ndarray, probe: np.ndarray) -> np.ndarray:
    """Boolean "is in ``sorted_keys``" for every element of ``probe``."""
    if len(sorted_keys) == 0:
        return np.zeros(len(probe), dtype=bool)
    pos = np.clip(np.searchsorted(sorted_keys, probe), 0, len(sorted_keys) - 1)
    return sorted_keys[pos] == probe


def _locate(sorted_keys: np.ndarray, probe: np.ndarray) -> np.ndarray:
    """Index into ``sorted_keys`` for every element of ``probe``, -1 if absent."""
    if len(sorted_keys) == 0:
        return np.full(len(probe), -1, dtype=np.int64)
    pos = np.clip(np.searchsorted(sorted_keys, probe), 0, len(sorted_keys) - 1)
    return np.where(sorted_keys[pos] == probe, pos, -1)


def _segment_transform(cells: tuple) -> tuple:
    """One press applied to ONE segment, as a pure function of its contents.

    ``cells`` is the segment read along the direction of travel, 0 empty,
    1 player, 2 crate. See the module docstring for the derivation; the two
    things worth restating here are that a run of players jammed against the end
    of the segment does not move at all (the segment ends at a wall or the board
    edge, and nothing pushes), and that the runs cannot interfere -- run
    ``[i..j]`` writes only ``i..j+1`` and the next run starts at ``j+2`` at the
    earliest.
    """
    n = len(cells)
    out = list(cells)
    i = 0
    while i < n:
        if cells[i] != 1:
            i += 1
            continue
        j = i
        while j + 1 < n and cells[j + 1] == 1:
            j += 1
        ahead = j + 1
        if ahead < n:                       # empty square, or a crate to swap with
            out[i] = cells[ahead]           # the crate lands at the run's BACK
            for k in range(i + 1, ahead + 1):
                out[k] = 1                  # every player steps forward one
        i = j + 1
    return tuple(out)


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

class _Board:
    """One level's static geometry, plus the mechanic, vectorised.

    Squares are numbered 0..F-1 over the level's NON-WALL cells only (at most 46
    on the shipped levels), so a state is two 64-bit masks -- the players and the
    crates -- and a whole BFS layer is two numpy columns. Walls and targets never
    change, so they live here rather than in the state.

    `step` and `preds` are both table-driven over SEGMENTS (see the module
    docstring): each maximal wall-free run of squares along the pressed axis is
    rewritten independently, and every one of its ``3^L`` possible contents is
    tabulated at construction. One press is then, per segment, a base-3 encode of
    the state's bits, one lookup and one OR -- all of it over the whole layer at
    once.
    """

    __slots__ = ("h", "w", "cells", "index", "n", "walls", "targets",
                 "tmask", "n_players", "n_crates", "segs", "inv", "sig")

    def __init__(self, h: int, w: int, walls, targets,
                 n_players: int, n_crates: int):
        self.h, self.w = h, w
        self.walls = frozenset(walls)
        self.targets = tuple(sorted(targets))
        #: Free squares in scan order; `index` maps a flat ``r * w + c`` cell to
        #: its bit position.
        self.cells = tuple(i for i in range(h * w) if i not in self.walls)
        self.index = {cell: bit for bit, cell in enumerate(self.cells)}
        self.n = len(self.cells)
        if self.n > 63:                                          # pragma: no cover
            raise ValueError(f"{self.n} free squares does not fit a uint64 mask")
        self.tmask = self._mask(targets)
        #: Invariant: no rule in this game creates or destroys anything, so the
        #: counts belong to the level rather than to a state. `goal_states` is
        #: the only thing that needs them.
        self.n_players = n_players
        self.n_crates = n_crates
        self.segs = {d: self._segments(d) for d in DIRS}
        self.inv = {d: self._inverse(d) for d in DIRS}
        #: The board's identity: geometry AND piece counts, because the counts
        #: are what `goal_states` enumerates against. It is also the key the
        #: expert caches boards, roots and fields under, so the three cannot
        #: drift apart.
        self.sig = (h, w, tuple(sorted(self.walls)), self.targets,
                    n_players, n_crates)

    # -- construction ---------------------------------------------------------
    def _mask(self, cells) -> np.uint64:
        m = 0
        for cell in cells:
            m |= 1 << self.index[cell]
        return np.uint64(m)

    def _segments(self, d: str) -> list:
        """Every maximal wall-free run along ``d``, with its forward table.

        A segment is stored as its bit masks in travel order plus, indexed by the
        base-3 encoding of its contents, the player and crate masks the press
        leaves behind."""
        dr, dc = _DELTA[d]
        out, seen = [], set()
        for cell in self.cells:
            if cell in seen:
                continue
            r, c = divmod(cell, self.w)
            # Walk back to the run's first square. The bounds are tested BEFORE
            # the membership: a flat index is not a safe way to ask "is there a
            # square left of column 0", it is the last square of the row above.
            while (0 <= r - dr < self.h and 0 <= c - dc < self.w
                   and (r - dr) * self.w + (c - dc) in self.index):
                r, c = r - dr, c - dc
            run = []
            while 0 <= r < self.h and 0 <= c < self.w and \
                    r * self.w + c in self.index:
                run.append(r * self.w + c)
                seen.add(r * self.w + c)
                r, c = r + dr, c + dc
            bits = np.array([np.uint64(1) << np.uint64(self.index[x])
                             for x in run], dtype=np.uint64)
            pow3 = np.array([3 ** j for j in range(len(run))], dtype=np.int64)
            size = 3 ** len(run)
            raw_p = np.zeros(size, dtype=np.uint64)
            raw_c = np.zeros(size, dtype=np.uint64)
            new_p = np.zeros(size, dtype=np.uint64)
            new_c = np.zeros(size, dtype=np.uint64)
            for code in range(size):
                content = []
                x = code
                for _ in range(len(run)):
                    content.append(x % 3)
                    x //= 3
                for j, v in enumerate(content):
                    if v == 1:
                        raw_p[code] |= bits[j]
                    elif v == 2:
                        raw_c[code] |= bits[j]
                for j, v in enumerate(_segment_transform(tuple(content))):
                    if v == 1:
                        new_p[code] |= bits[j]
                    elif v == 2:
                        new_c[code] |= bits[j]
            mask = np.uint64(0)
            for b in bits:
                mask |= b
            out.append(dict(bits=bits, pow3=pow3, mask=mask,
                            raw_p=raw_p, raw_c=raw_c,
                            new_p=new_p, new_c=new_c))
        return out

    def _inverse(self, d: str) -> list:
        """Per segment, the PREIMAGE table: every content that the press maps to
        a given content, as a CSR (start, count, list).

        The forward map is a function on ``3^L`` contents, so inverting it is a
        sort. This is what makes the backward sweep exact: the game is not
        reversible, so a predecessor cannot be found by pressing the opposite
        direction (see the module docstring)."""
        out = []
        for seg in self.segs[d]:
            size = len(seg["new_p"])
            fwd = np.zeros(size, dtype=np.int64)
            for code in range(size):
                p, c = int(seg["new_p"][code]), int(seg["new_c"][code])
                enc = 0
                for j, bit in enumerate(seg["bits"]):
                    b = int(bit)
                    enc += (1 if p & b else 2 if c & b else 0) * (3 ** j)
                fwd[code] = enc
            order = np.argsort(fwd, kind="stable")
            ordered = fwd[order]
            grid = np.arange(size)
            start = np.searchsorted(ordered, grid)
            count = np.searchsorted(ordered, grid, side="right") - start
            out.append(dict(start=start, count=count, codes=order))
        return out

    # -- the mechanic ---------------------------------------------------------
    def encode(self, pmask, cmask, seg) -> np.ndarray:
        """The base-3 content code of ``seg`` for a whole column of states."""
        code = np.zeros(np.shape(pmask), dtype=np.int64)
        for j, bit in enumerate(seg["bits"]):
            code += (((pmask & bit) != 0).astype(np.int64)
                     + 2 * ((cmask & bit) != 0).astype(np.int64)) * seg["pow3"][j]
        return code

    def step(self, pmask, cmask, d: str):
        """The state(s) after pressing ``d``. Vectorised: ``pmask``/``cmask`` are
        uint64 arrays and the return is the same shape."""
        out_p = np.zeros_like(pmask)
        out_c = np.zeros_like(cmask)
        for seg in self.segs[d]:
            code = self.encode(pmask, cmask, seg)
            out_p |= seg["new_p"][code]
            out_c |= seg["new_c"][code]
        return out_p, out_c

    def step_one(self, state, d: str) -> tuple:
        """`step` for a single ``(player, crate)`` mask pair, as Python ints."""
        p, c = self.step(np.array([state[0]], dtype=np.uint64),
                         np.array([state[1]], dtype=np.uint64), d)
        return (int(p[0]), int(c[0]))

    def preds(self, pmask, cmask, d: str):
        """Every state ``X`` with ``step(X, d) == (pmask, cmask)``.

        Exact and complete, and it has to be: the mechanic is not reversible, so
        "press the other way" is not an inverse. Segments are independent, so the
        predecessors are the CARTESIAN PRODUCT of each segment's preimages, built
        one segment at a time with `numpy.repeat`."""
        cur_p, cur_c = pmask, cmask
        for seg, inv in zip(self.segs[d], self.inv[d]):
            code = self.encode(cur_p, cur_c, seg)
            count = inv["count"][code]
            total = int(count.sum())
            if total == 0:                                       # pragma: no cover
                empty = np.zeros(0, dtype=np.uint64)
                return empty, empty
            rep = np.repeat(np.arange(len(cur_p)), count)
            offset = (np.arange(total)
                      - np.repeat(np.cumsum(count) - count, count))
            pick = inv["codes"][np.repeat(inv["start"][code], count) + offset]
            keep = ~seg["mask"]
            cur_p = (cur_p[rep] & keep) | seg["raw_p"][pick]
            cur_c = (cur_c[rep] & keep) | seg["raw_c"][pick]
        return cur_p, cur_c

    # -- the win --------------------------------------------------------------
    def won(self, cmask):
        """``All Target on Crate``: every target square carries a crate."""
        return (cmask & self.tmask) == self.tmask

    def goal_states(self):
        """Every state this board calls won, as ``(player masks, crate masks)``.

        Enumerated rather than searched for, which is what lets the backward
        sweep start at the win. The win constrains only the target squares, so a
        board with spare crates has a goal state for every placement of them
        (level 3 has one spare) and the players may stand anywhere free. That is
        37 states on level 0 and 749,398 on level 4.

        Empty when the level cannot be won at all -- fewer crates than targets --
        which would otherwise be mis-read as "the sweep gave up"."""
        spare = self.n_crates - len(self.targets)
        if spare < 0:                                            # pragma: no cover
            return np.zeros(0, np.uint64), np.zeros(0, np.uint64)
        target_bits = [self.index[t] for t in self.targets]
        rest = [b for b in range(self.n) if b not in target_bits]
        base = sum(1 << b for b in target_bits)
        pmasks, cmasks = [], []
        for extra in itertools.combinations(rest, spare):
            cmask = base | sum(1 << b for b in extra)
            free = [b for b in rest if b not in extra]
            for players in itertools.combinations(free, self.n_players):
                pmasks.append(sum(1 << b for b in players))
                cmasks.append(cmask)
        return (np.array(pmasks, dtype=np.uint64),
                np.array(cmasks, dtype=np.uint64))

    # -- presentation ---------------------------------------------------------
    def ascii(self, state) -> str:
        """A model state as ASCII, so a failing check can print the board it
        failed on rather than a pair of integers."""
        p, c = int(state[0]), int(state[1])
        rows = []
        for r in range(self.h):
            line = ""
            for col in range(self.w):
                cell = r * self.w + col
                if cell in self.walls:
                    line += "#"
                    continue
                bit = 1 << self.index[cell]
                on_target = cell in self.targets
                if p & bit:
                    line += "p" if on_target else "P"
                elif c & bit:
                    line += "@" if on_target else "*"
                else:
                    line += "O" if on_target else "."
            rows.append(line)
        return "\n".join(rows)


# ---------------------------------------------------------------------------
# The exact distance derivation
# ---------------------------------------------------------------------------

class _Field:
    """Exact presses-to-win for one level, by a TWO-SIDED breadth-first sweep.

    Neither half alone is enough on this game and the reason is measured, not
    guessed (the layer sizes below are level 4's, the 9x9 with five players and
    five crates):

      * a FORWARD sweep from the start reaches 5.2M states by depth 14 and grows
        by 2.5x a layer, so the 20 it needs is out of reach;
      * a BACKWARD sweep from the win starts at 749,398 goal states -- the win
        constrains only the crates, so every placement of the five players is a
        goal -- and reaches 72M by depth 5.

    Grown TOGETHER, smaller frontier first, they meet at f = 16 and b = 4 for a
    total of ~60M states, which is minutes rather than never.

    THE STOPPING RULE, stated exactly, because an off-by-one here would silently
    ship a plan that is not shortest. After completing forward layers 0..f and
    backward layers 0..b, let ``best`` be the smallest ``dstart(s) + dgoal(s)``
    over the states both halves have labelled. Any path of length L either has
    its depth-f state within b of the win -- in which case ``best <= L`` -- or it
    does not, in which case ``L > f + b``. So ``d* = best`` as soon as
    ``best <= f + b + 1``, and until then the sweep keeps growing.

    READING THE ANSWER BACK. From the meeting layer onwards a state's true
    distance is in the backward field. Before it, the forward half is labelled by
    one induction down the recorded layer-to-layer edges:

        on_cone[f] = {s : f + dgoal(s) == d*}          (dgoal from the backward
                                                        field, which reaches
                                                        exactly that far)
        on_cone[g] = {s : some successor in layer g+1 is on the cone}
                     u {s : g + dgoal(s) == d*}

    A state at forward depth ``g`` is on a shortest route iff one of its depth
    ``g+1`` successors is, because a shortest route from it must step to a state
    whose own distances still sum to ``d*`` and that state's forward depth is
    therefore exactly ``g+1``. The union's second clause catches the states whose
    whole remaining route lies inside the backward field. `optimal` reads a tie
    set straight off this -- nothing is inferred, and `--verify` re-derives every
    shipped set by brute force anyway.

    ``cap`` is a runaway guard, not a budget: past it the sweep gives up and
    `plan` answers None, which `record_level` turns into a RESET. It is not
    reached on any shipped level.
    """

    __slots__ = ("board", "root", "cap", "fseen", "fdist", "flocal", "fedges",
                 "bkeys", "bdist", "bfrontier", "dstar", "cone", "solved",
                 "n_forward", "n_backward")

    def __init__(self, board: _Board, root: tuple, cap: int = 90_000_000):
        self.board = board
        #: The state the FORWARD half is rooted at -- the level start in every
        #: use here (the expert plans from a level's start before anything else
        #: touches it), which is what makes one field per level the right cache.
        self.root = (np.uint64(root[0]), np.uint64(root[1]))
        self.cap = cap
        self.dstar: int | None = None
        self.solved = False
        self.cone: list = []
        self.fedges: list = []
        self.n_forward = self.n_backward = 0

    # -- the two halves -------------------------------------------------------
    def _layer(self, g: int) -> np.ndarray:
        """Forward layer ``g``, in the sorted order its edge indices refer to.

        Derived from ``fseen``/``fdist`` rather than stored: ``fseen`` stays
        sorted through every merge and ``fdist`` is permuted with it, so the
        masked subarray is the same array, in the same order, that the expansion
        which built the layer handed out indices into."""
        return self.fseen[self.fdist == g]

    def _grow_forward(self, g: int) -> int:
        """Expand forward layer ``g`` into layer ``g+1``; return its size."""
        board = self.board
        cur = self._layer(g)
        alive = ~board.won(cur["c"])         # a win ENDS the level: no successors
        outs = []
        for d in DIRS:
            pmask, cmask = board.step(cur["p"], cur["c"], d)
            outs.append(_keys(pmask, cmask))
        allk = np.concatenate(outs)
        alive4 = np.tile(alive, 4)
        cand = np.unique(allk[alive4])
        new = cand[~_found(self.fseen, cand)]
        if len(new):
            self.fseen, at, keep = _merge(self.fseen, new)
            dist = np.empty(len(self.fseen), dtype=np.int32)
            dist[at] = g + 1
            dist[keep] = self.fdist
            self.fdist = dist
        edge = _locate(new, allk).astype(np.int64)
        edge[~alive4] = -1
        self.fedges.append(edge.reshape(4, len(cur)).T.astype(np.int32).copy())
        self.n_forward = len(self.fseen)
        return len(new)

    def _grow_backward(self, b: int) -> int:
        """Expand backward layer ``b`` into layer ``b+1``; return its size."""
        board = self.board
        cur = self.bfrontier
        outs = []
        for d in DIRS:
            pmask, cmask = board.preds(cur["p"].copy(), cur["c"].copy(), d)
            if len(pmask):
                outs.append(_keys(pmask, cmask))
        if not outs:
            self.bfrontier = np.zeros(0, dtype=KEY_DT)
            return 0
        cand = np.unique(np.concatenate(outs))
        new = cand[~_found(self.bkeys, cand)]
        if len(new):
            self.bkeys, at, keep = _merge(self.bkeys, new)
            dist = np.empty(len(self.bkeys), dtype=np.int32)
            dist[at] = b + 1
            dist[keep] = self.bdist
            self.bdist = dist
        self.bfrontier = new
        self.n_backward = len(self.bkeys)
        return len(new)

    # -- the sweep ------------------------------------------------------------
    def solve(self, verbose: bool = False) -> "int | None":
        """Derive ``d*`` and everything `plan` reads. Idempotent."""
        if self.solved:
            return self.dstar
        board = self.board
        self.fseen = _keys(np.array([self.root[0]], dtype=np.uint64),
                           np.array([self.root[1]], dtype=np.uint64))
        self.fdist = np.zeros(1, dtype=np.int32)
        gp, gc = board.goal_states()
        if len(gp) == 0:                                         # pragma: no cover
            self.solved = True
            return None
        self.bkeys = np.unique(_keys(gp, gc))
        self.bdist = np.zeros(len(self.bkeys), dtype=np.int32)
        self.bfrontier = self.bkeys
        self.n_forward, self.n_backward = 1, len(self.bkeys)

        f = b = 0
        best = self._crossing(self._layer(0), 0, forward=True)
        f_live = b_live = True
        while best is None or best > f + b + 1:
            if not (f_live or b_live):
                break                        # both halves exhausted: no path
            grow_forward = f_live and (
                not b_live or len(self._layer(f)) <= len(self.bfrontier))
            if grow_forward:
                if self._grow_forward(f) == 0:
                    f_live = False
                    continue
                f += 1
                hit = self._crossing(self._layer(f), f, forward=True)
            else:
                if self._grow_backward(b) == 0:
                    b_live = False
                    continue
                b += 1
                hit = self._crossing(self.bfrontier, b, forward=False)
            if hit is not None and (best is None or hit < best):
                best = hit
            if verbose:
                print(f"    f={f} ({self.n_forward}) b={b} ({self.n_backward}) "
                      f"best={best}", flush=True)
            if self.n_forward + self.n_backward > self.cap:       # pragma: no cover
                self.solved = True
                return None
        self.dstar = best
        if best is not None:
            self._build_cone(f)
        self.solved = True
        return self.dstar

    def _crossing(self, layer: np.ndarray, depth: int, forward: bool):
        """The best ``dstart + dgoal`` a freshly built layer reveals, or None."""
        if len(layer) == 0:
            return None
        if forward:
            loc = _locate(self.bkeys, layer)
            hit = loc >= 0
            if not hit.any():
                return None
            return int(depth + self.bdist[loc[hit]].min())
        loc = _locate(self.fseen, layer)
        hit = loc >= 0
        if not hit.any():
            return None
        return int(depth + self.fdist[loc[hit]].min())

    def _build_cone(self, f: int) -> None:
        """Mark, per forward layer, the states on a shortest route (see the class
        docstring). Also fixes each state's index WITHIN its layer, which is what
        `optimal` needs to read an edge row."""
        self.flocal = np.zeros(len(self.fseen), dtype=np.int32)
        for g in range(f + 1):
            where = np.flatnonzero(self.fdist == g)
            self.flocal[where] = np.arange(len(where), dtype=np.int32)
        self.cone = [None] * (f + 1)
        for g in range(f, -1, -1):
            layer = self._layer(g)
            loc = _locate(self.bkeys, layer)
            known = loc >= 0
            mark = np.zeros(len(layer), dtype=bool)
            mark[known] = (g + self.bdist[loc[known]]) == self.dstar
            if g < f:
                edge = self.fedges[g]
                nxt = self.cone[g + 1]
                for j in range(4):
                    col = edge[:, j]
                    ok = col >= 0
                    mark[ok] |= nxt[col[ok]]
            self.cone[g] = mark

    # -- reading the answer ---------------------------------------------------
    def _backward_optimal(self, state, rest: int) -> list:
        """Tie set for a state the BACKWARD field has labelled."""
        board = self.board
        pmask = np.array([state[0]], dtype=np.uint64)
        cmask = np.array([state[1]], dtype=np.uint64)
        out = []
        for d in DIRS:
            np_, nc = board.step(pmask, cmask, d)
            if (int(np_[0]), int(nc[0])) == (int(state[0]), int(state[1])):
                continue                 # a non-move is never on a shortest path
            loc = _locate(self.bkeys, _keys(np_, nc))[0]
            if loc >= 0 and int(self.bdist[loc]) == rest - 1:
                out.append(d)
        return out

    def _forward_slot(self, state):
        """``(depth, index in that layer)`` for a state the forward half holds,
        or None."""
        loc = _locate(self.fseen, _keys(np.array([state[0]], dtype=np.uint64),
                                        np.array([state[1]], dtype=np.uint64)))[0]
        if loc < 0:
            return None
        return int(self.fdist[loc]), int(self.flocal[loc])

    def distance(self, state) -> "int | None":
        """Presses to a win from ``state``, or None when this sweep cannot say.

        None means "not derived", NOT "dead": the two halves are bounded, so a
        board far from both the level start and the win is simply outside them.
        `record_level` reads it as "no plan" and RESETs, which is the right
        behaviour either way."""
        self.solve()
        if self.dstar is None:
            return None
        if self.board.won(np.uint64(state[1])):
            return 0
        loc = _locate(self.bkeys, _keys(np.array([state[0]], dtype=np.uint64),
                                        np.array([state[1]], dtype=np.uint64)))[0]
        if loc >= 0:
            return int(self.bdist[loc])
        slot = self._forward_slot(state)
        if slot is None:
            return None
        g, i = slot
        if g < len(self.cone) and self.cone[g][i]:
            return self.dstar - g
        return None

    def optimal(self, state) -> list:
        """Every press that keeps the game on a SHORTEST route to the win.

        Exact on both halves, with nothing inferred: from the backward field a
        press is optimal iff its successor is one nearer the win; from the
        forward half iff its successor is on the cone one layer deeper."""
        self.solve()
        if self.dstar is None:
            return []
        rest = self.distance(state)
        if not rest:                         # None (not derived) or 0 (won)
            return []
        loc = _locate(self.bkeys, _keys(np.array([state[0]], dtype=np.uint64),
                                        np.array([state[1]], dtype=np.uint64)))[0]
        if loc >= 0:
            return self._backward_optimal(state, int(self.bdist[loc]))
        g, i = self._forward_slot(state)
        if g >= len(self.fedges):            # pragma: no cover
            # The deepest forward layer is never reached through this branch: a
            # cone state there was marked by the backward field and is handled
            # above. The guard is here so a future change to the sweep cannot
            # turn that into an IndexError.
            return []
        edge = self.fedges[g][i]
        nxt = self.cone[g + 1]
        return [d for j, d in enumerate(DIRS)
                if edge[j] >= 0 and nxt[edge[j]]]

    def plan(self, state) -> "Plan | None":
        """The shortest press sequence from ``state``, with the EXACT optimal set
        at every step, or None when this sweep cannot answer for it."""
        self.solve()
        rest = self.distance(state)
        if rest is None:
            return None
        presses, optsets = [], []
        cur = state
        for _ in range(rest):
            best = self.optimal(cur)
            if not best:                                         # pragma: no cover
                return None
            presses.append(best[0])
            optsets.append(list(best))
            cur = self.board.step_one(cur, best[0])
        if not self.board.won(np.uint64(cur[1])):                # pragma: no cover
            return None
        return Plan(presses, optsets)


# ---------------------------------------------------------------------------
# The expert
# ---------------------------------------------------------------------------

class SwapTheBlockExpert(PSExpert):
    """`PSExpert`'s plan memo, level scoping and disk cache around a `_Field`.

    Only the strategy underneath changes, the way `PSEnumExpert` replaces it:
    here it is "read the live board and descend the exact two-sided sweep", so
    `heuristic` is never called and asserts rather than returning a number
    nothing would use.

    `_search` reads the ENGINE's current grid every time. It answers exactly for
    the level start and for anything the backward half of that level's sweep
    reaches; for a board further out than both halves it answers None, which
    `record_level` turns into a RESET. That is the honest answer for this game --
    unlike the reversible ps:swap_sokoban, a board here can be genuinely
    unwinnable and is in any case far too expensive to field completely.
    """

    directions = list(DIRS)
    #: `_key` is the dynamic objects only, which is canonical WITHIN a level but
    #: not across them -- walls and targets are static per level and differ
    #: between levels.
    scope_by_level = True
    plan_cache_path = PLAN_CACHE

    #: Runaway guard on one level's sweep, in states summed over both halves.
    #: Level 4 -- the largest by far -- settles at 59.6M.
    field_cap: int = 90_000_000

    def setup(self) -> None:
        g = self.g
        self.player_ids = set(self.game._engine._player_indices)
        self.wall_ids = set(g.resolve_object_name("wall"))
        self.crate_ids = set(g.resolve_object_name("crate"))
        self.target_ids = set(g.resolve_object_name("target"))
        #: `_Board`s and `_Field`s by STATIC signature, so a board is shared by
        #: every state of its level and the sweep is paid once. ``_roots`` is the
        #: FIRST state each board was read at, which is the level start in every
        #: path here (`prepare_expert`, `discover_solvable` and `record_level`
        #: all seat the level before reading it) -- and the forward half of a
        #: sweep has to be rooted there rather than at whatever perturbed board
        #: happens to ask first.
        self._boards: dict = {}
        self._fields: dict = {}
        self._roots: dict = {}

    def heuristic(self, eng) -> int:                              # pragma: no cover
        raise AssertionError(
            "SwapTheBlockExpert reads an exact distance field; heuristic is unused")

    # -- engine <-> model ----------------------------------------------------
    def read(self, eng) -> tuple:
        """``(board, state)`` for the engine's current grid.

        Boards are cached by their STATIC signature (dimensions, walls, targets,
        piece counts), so the segment tables are built once per level however
        many states are read from it."""
        h, w = eng.height, eng.width
        walls, targets, players, crates = [], [], [], []
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if not cell:
                    continue
                i = r * w + c
                if cell & self.wall_ids:
                    walls.append(i)
                if cell & self.target_ids:
                    targets.append(i)
                if cell & self.player_ids:
                    players.append(i)
                if cell & self.crate_ids:
                    crates.append(i)
        sig = (h, w, tuple(walls), tuple(targets), len(players), len(crates))
        board = self._boards.get(sig)
        if board is None:
            board = _Board(h, w, walls, targets, len(players), len(crates))
            # Stored under the BOARD's own signature, which is the key `_roots`
            # and `_fields` use. It equals the probe built above (both list the
            # walls and targets in scan order, which is sorted order); keying the
            # cache off the board itself means a future change that broke that
            # would rebuild boards rather than mis-file their fields.
            self._boards[board.sig] = board
        state = (int(board._mask(players)), int(board._mask(crates)))
        self._roots.setdefault(board.sig, state)
        return board, state

    def field(self, board: _Board) -> _Field:
        """The level's sweep, rooted at the level start (see `setup`), built once
        and kept. A query from a state the sweep does not contain is answered
        None rather than by re-rooting a second sweep at it -- that would be a
        different and equally expensive derivation, and the answer it would give
        for the state actually being asked about is the one this field already
        has or does not have."""
        got = self._fields.get(board.sig)
        if got is None:
            got = self._fields[board.sig] = _Field(
                board, self._roots[board.sig], self.field_cap)
        return got

    def _key(self, eng) -> frozenset:
        """``(row, col, tag)`` triples for the players (0) and the crates (1).

        Built from the MODEL's reading rather than from raw object ids, so the
        key is exactly the two things a state consists of -- and it is what the
        ``plan_cache_path`` signature is stored as, so a cached plan is matched
        against the board it was solved from and an edited level is a miss rather
        than a wrong plan."""
        board, (pmask, cmask) = self.read(eng)
        out = set()
        for bit, cell in enumerate(board.cells):
            if (pmask >> bit) & 1:
                out.add((cell // board.w, cell % board.w, 0))
            elif (cmask >> bit) & 1:
                out.add((cell // board.w, cell % board.w, 1))
        return frozenset(out)

    def _search(self, eng) -> "Plan | None":
        board, state = self.read(eng)
        if board.won(np.uint64(state[1])):
            return Plan([], [])
        return self.field(board).plan(state)

    def optimal_dirs(self, eng, only_if_swept: bool = False) -> list:
        """Every press on a shortest route, from the engine's CURRENT state.

        ``only_if_swept`` returns [] rather than BUILDING the level's field --
        which is what a recording wants: a sweep is minutes on level 4 and a
        recording that triggered one mid-episode would be a performance bug,
        while the answer it wants is already stored in the plan it is following
        (see `SwapTheBlockSolver.optimal_for`)."""
        board, state = self.read(eng)
        if only_if_swept and board.sig not in self._fields:
            return []
        return self.field(board).optimal(state)


# ---------------------------------------------------------------------------
# The solver
# ---------------------------------------------------------------------------

class SwapTheBlockSolver(PSAStarSolver):
    game_id = "puzzlescript_swap_the_block"
    game_name = GAME_NAME
    expert_cls = SwapTheBlockExpert

    #: `games/ps:swap_the_block/ps:swap_the_block.py` is a plain passthrough --
    #: it builds the adapter and nothing else, and no rendering fix was needed
    #: (see the module docstring). Set this if that wrapper ever grows a patch.
    game_module_id = ""

    #: Unused: `SwapTheBlockExpert._search` never calls `_astar`. Left at the base
    #: values so nothing reads a lie off them; `SwapTheBlockExpert.field_cap` is
    #: the knob that actually bounds the sweep.
    weight = 1
    node_cap = 400_000

    #: Room for the longest plan (24 presses on level 2) many times over. The
    #: adapter's own 200-press per-level budget is separate and is reset by the
    #: `set_level` that ends the exploration prefix, so the plan starts from zero.
    max_steps = 150

    #: Every level is solved; nothing is skipped.
    skip_levels = frozenset()

    #: Zero, unlike ps:swap_sokoban's 0.12, and the reason is the mechanic: this
    #: game is NOT reversible and its state space is far too large to field
    #: completely (level 4 alone is 1.0e12 states), so an arbitrary detoured
    #: board usually cannot be answered at all. `record_level` would probe each
    #: detour, get None from the expert and drop it -- an identical recording for
    #: a lot of wasted sweeping. The RESET-mode exploration prefix is the
    #: recovery arc this game supports.
    epsilon = 0.0

    def prepare_expert(self, game, expert) -> None:
        """Derive every level's plan before `discover_solvable` asks for it.

        With ``data/swap_the_block_plans.json`` present this is five disk reads;
        without it, it is the sweeps, and this is where they belong -- once, at
        startup, visible as startup, and written back for every later process."""
        for level in range(game.n_levels):
            game.set_level(level)
            expert.plan(game._engine, level)

    def optimal_for(self, expert, level: int, plan: list, pi: int):
        """Every press that keeps the game on a SHORTEST route, read off the LIVE
        engine state when the field can answer for it, and off the plan's own
        stored sets otherwise.

        The stored sets are the same derivation (`_Field.plan` labels every step
        as it descends) and are what a disk-cached plan carries into a later
        process, where no field exists at all. `record_level` asks for this
        BEFORE executing the press, so the engine is still at the state being
        labelled.

        Falls back to the press about to be taken -- no expert step may ship
        unlabelled."""
        sets = getattr(plan, "optsets", None)
        if sets is not None and pi < len(sets) and sets[pi]:
            return sets[pi]
        best = expert.optimal_dirs(expert.game._engine, only_if_swept=True)
        if best:
            return best
        return [plan[pi]] if pi < len(plan) else None


# ---------------------------------------------------------------------------
# Independent derivations, used only by the checks
# ---------------------------------------------------------------------------

def _forward_bfs(board: _Board, start: tuple, cap: int = 8_000_000):
    """Presses to the first win by a plain FORWARD BFS, or None past ``cap``.

    Shares nothing with `_Field` but `_Board.step` itself, which is the point: it
    is the independent answer the two-sided sweep's ``d*`` is checked against,
    and it uses neither the predecessor tables nor the cone induction -- the two
    pieces of machinery a bug could hide in."""
    seen = _keys(np.array([start[0]], dtype=np.uint64),
                 np.array([start[1]], dtype=np.uint64))
    frontier = seen
    if board.won(frontier["c"])[0]:
        return 0
    depth = 0
    while len(frontier):
        outs = []
        for d in DIRS:
            pmask, cmask = board.step(frontier["p"], frontier["c"], d)
            outs.append(_keys(pmask, cmask))
        allk = np.unique(np.concatenate(outs))
        new = allk[~_found(seen, allk)]
        depth += 1
        if len(new) == 0:
            return None
        if board.won(new["c"]).any():
            return depth
        seen, _at, _keep = _merge(seen, new)
        frontier = new
        if len(seen) > cap:
            return None
    return None


def _pure_backward(board: _Board, want: tuple, cap: int = 20_000_000):
    """A backward-only distance field, grown until ``want`` is labelled.

    The second, independent reading of the same question `_Field` answers: no
    forward half, no cone induction, just breadth-first predecessors out from the
    win. Returns ``(keys, dist, depth_reached)``; ``keys`` is sorted, so a
    distance is one searchsorted."""
    gp, gc = board.goal_states()
    keys = np.unique(_keys(gp, gc))
    dist = np.zeros(len(keys), dtype=np.int32)
    frontier = keys
    probe = _keys(np.array([want[0]], dtype=np.uint64),
                  np.array([want[1]], dtype=np.uint64))
    depth = 0
    while len(frontier) and len(keys) <= cap:
        if _found(keys, probe)[0]:
            break
        outs = []
        for d in DIRS:
            pmask, cmask = board.preds(frontier["p"].copy(),
                                       frontier["c"].copy(), d)
            if len(pmask):
                outs.append(_keys(pmask, cmask))
        if not outs:
            break
        cand = np.unique(np.concatenate(outs))
        new = cand[~_found(keys, cand)]
        if len(new) == 0:
            break
        depth += 1
        keys, at, keep = _merge(keys, new)
        grown = np.empty(len(keys), dtype=np.int32)
        grown[at] = depth
        grown[keep] = dist
        dist = grown
        frontier = new
    return keys, dist, depth


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _new(seed: int = 0):
    """The solver, its adapter and an expert -- without planning anything.

    The reports below each want a different slice (``--audit`` needs no plan at
    all, ``--selfcheck`` needs only the model), so the sweeps are left to
    whoever asks for them rather than paid on every entry point."""
    solver = SwapTheBlockSolver()
    game = solver.make_game(seed)
    return solver, game, SwapTheBlockExpert(game, node_cap=solver.node_cap)


def _seat(game, expert, level: int):
    """``(board, state)`` at a level's start, engine left seated there."""
    game.set_level(level)
    return expert.read(game._engine)


def _plans() -> int:
    """Every level's plan, replayed through the REAL interpreter -- the
    end-to-end test of the native model, the sweep and the tie labelling at
    once."""
    _solver, game, expert = _new()
    eng = game._engine
    total = ties = bad = 0
    for level in range(game.n_levels):
        board, state = _seat(game, expert, level)
        plan = expert.plan(eng, level)
        if plan is None:
            print(f"  L{level}: UNSOLVED")
            bad += 1
            continue
        for direction in plan:
            eng.step(direction)
        won = eng.check_win()
        bad += not won
        sets = getattr(plan, "optsets", None) or [[p] for p in plan]
        tie_steps = sum(1 for s in sets if len(s) > 1)
        total += len(plan)
        ties += tie_steps
        field = expert._fields.get(board.sig)
        sweep = (f"{field.n_forward}+{field.n_backward} states swept"
                 if field is not None else "from the plan cache")
        print(f"  L{level}: {board.h}x{board.w}  {board.n:2d} floor  "
              f"{board.n_players} players  {board.n_crates} crates  "
              f"{len(board.targets)} targets  "
              f"{len(plan):3d} presses  win={won}  "
              f"{tie_steps:2d} steps with a tie set  ({sweep})")
    print(f"  {total} presses, {ties} of them with a second equally-right answer")
    print("  every plan reaches a WIN" if not bad else f"  {bad} PROBLEM(S)")
    return 1 if bad else 0


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def _plant(game, board: _Board, state) -> None:
    """Load an arbitrary model state into the interpreter.

    Random STARTS are the point: a level's own opening never puts five players
    shoulder to shoulder behind a crate, and the cascade that does is the one
    piece of this mechanic a hand-read of the rule gets wrong."""
    eng = game._engine
    g = game._game
    bg = g.obj_name_to_idx["background"]
    wall = g.obj_name_to_idx["wall"]
    target = g.obj_name_to_idx["target"]
    player = g.obj_name_to_idx["player"]
    crate = g.obj_name_to_idx["crate"]
    grid = []
    for r in range(board.h):
        row = []
        for c in range(board.w):
            cell = {bg}
            i = r * board.w + c
            if i in board.walls:
                cell.add(wall)
            if i in board.targets:
                cell.add(target)
            bit = board.index.get(i)
            if bit is not None:
                if (state[0] >> bit) & 1:
                    cell.add(player)
                if (state[1] >> bit) & 1:
                    cell.add(crate)
            row.append(cell)
        grid.append(row)
    eng.grid = grid
    eng._position_index_dirty = True
    eng._rule_noop_cache.clear()


def _random_state(board: _Board, rng: random.Random) -> tuple:
    cells = rng.sample(range(board.n), board.n_players + board.n_crates)
    pmask = sum(1 << b for b in cells[:board.n_players])
    cmask = sum(1 << b for b in cells[board.n_players:])
    return (pmask, cmask)


def selfcheck(trials: int = 60, steps: int = 50, verbose: bool = True) -> int:
    """Four passes, each able to fail differently. Returns the mismatch count.

    1. THE MODEL. Drive presses through BOTH the interpreter and `_Board.step`
       from randomly planted boards, comparing the whole board and the win flag
       after every one. ``action`` is in the alphabet even though the sweeps
       exclude it: "no rule reads the X button" is exactly the kind of claim that
       should be measured rather than read off a rules section.
    2. THE PREDECESSORS. For random states, `preds` of a state's successor must
       be SOUND (every one really does step to it) and COMPLETE (it contains the
       state itself). The backward half of every sweep rests on both, and this
       game gives no cheap alternative: it is not reversible, so pressing the
       opposite direction is not an inverse.
    3. OPTIMALITY, independently. A plain forward BFS to the first win must agree
       with the two-sided sweep's ``d*``. Levels 0-3 only: level 4's forward
       space passes 30 million by depth 16, which is the reason the sweep is
       two-sided in the first place.
    4. RECOVERY. Perturb a level start with random presses and ask the expert to
       plan from where that lands. Any plan it DOES return is replayed on the
       interpreter and must WIN. (It is expected to decline most of them -- see
       `_Field.distance` -- and declining is what `record_level` turns into a
       RESET.)
    """
    solver, game, expert = _new()
    eng = game._engine
    alphabet = DIRS + ("action",)
    total = 0

    for level in range(game.n_levels):
        board, start = _seat(game, expert, level)
        rng = random.Random(f"swap_the_block:model:{level}")
        bad = presses = 0
        for _ in range(trials):
            state = _random_state(board, rng)
            _plant(game, board, state)
            for _ in range(steps):
                d = rng.choice(alphabet)
                want = state if d == "action" else board.step_one(state, d)
                eng.step(d)
                presses += 1
                got = expert.read(eng)[1]
                if want != got:
                    bad += 1
                    print(f"  L{level}: {d} from\n{board.ascii(state)}\n"
                          f"  model gave\n{board.ascii(want)}\n"
                          f"  engine gave\n{board.ascii(got)}")
                    break
                if bool(board.won(np.uint64(want[1]))) != eng.check_win():
                    bad += 1
                    break
                state = want
                if eng.check_win():
                    break
        total += bad
        if verbose:
            print(f"  L{level} model: {'OK' if not bad else f'{bad} MISMATCHES'} "
                  f"({presses} presses vs the interpreter)")

    for level in range(game.n_levels):
        board, _start = _seat(game, expert, level)
        rng = random.Random(f"swap_the_block:preds:{level}")
        bad = checked = 0
        for _ in range(400):
            state = _random_state(board, rng)
            d = rng.choice(DIRS)
            nxt = board.step_one(state, d)
            pmask, cmask = board.preds(
                np.array([nxt[0]], dtype=np.uint64),
                np.array([nxt[1]], dtype=np.uint64), d)
            checked += len(pmask)
            back_p, back_c = board.step(pmask, cmask, d)
            if not (np.all(back_p == np.uint64(nxt[0]))
                    and np.all(back_c == np.uint64(nxt[1]))):
                bad += 1                                         # not sound
            if not np.any((pmask == np.uint64(state[0]))
                          & (cmask == np.uint64(state[1]))):
                bad += 1                                         # not complete
        total += bad
        if verbose:
            print(f"  L{level} preds: {'OK' if not bad else f'{bad} BROKEN'} "
                  f"({checked} predecessors, sound and complete)")

    for level in range(game.n_levels):
        board, start = _seat(game, expert, level)
        plan = expert.plan(eng, level)
        mine = None if plan is None else len(plan)
        theirs = _forward_bfs(board, start)
        if theirs is None:
            if verbose:
                print(f"  L{level} optimality: forward BFS too large -- "
                      f"two-sided sweep says d* = {mine}")
            continue
        ok = mine == theirs
        total += not ok
        if verbose:
            print(f"  L{level} optimality: sweep d* = {mine}, independent "
                  f"forward BFS d* = {theirs} -- {'AGREE' if ok else 'DISAGREE'}")

    for level in range(game.n_levels):
        board, start = _seat(game, expert, level)
        rng = random.Random(f"swap_the_block:recovery:{level}")
        answered = replayed = bad = 0
        for _ in range(40):
            game.set_level(level)
            for _ in range(rng.randint(1, 8)):
                eng.step(rng.choice(DIRS))
            if eng.check_win():
                continue
            found = expert.plan(eng, level)
            if found is None:
                continue
            answered += 1
            for direction in found:
                eng.step(direction)
            replayed += len(found)
            if not eng.check_win():
                bad += 1
        total += bad
        if verbose:
            print(f"  L{level} recovery: {answered}/40 perturbed boards answered, "
                  f"{replayed} presses replayed on the interpreter, "
                  f"{'all WIN' if not bad else f'{bad} DID NOT WIN'}")
    return total


def _verify(cap: int = 20_000_000) -> int:
    """Double-entry check of every shipped plan and every tie set in it.

    `_Field` answers from a two-sided sweep whose delicate part is the cone
    induction over the forward half -- a bug there would produce a
    self-consistent structure, a plan that still wins, and tie sets that are
    quietly wrong, and the training labels ARE those tie sets. So they are
    checked twice, by two passes that fail differently:

      * A SECOND, INDEPENDENT FIELD. Each level's distances are rebuilt by a
        BACKWARD-ONLY sweep -- no forward half, no cone -- and must agree at
        every state of the plan on both the distance to the win and the tie set,
        the latter re-derived by pressing each of the four buttons and looking
        the successor up. This is exact but it is bounded: level 3's win has 712k
        goal states and level 4's has 749,398, and those sweeps pass 25 million
        states by depth 5, so on the big levels it checks the TAIL and reports
        how much of the plan it reached.
      * EXECUTABLE LABELS, which covers every step of every level. At each state
        of the plan, each press the label calls optimal is actually PRESSED on
        the interpreter, and the field's own plan from where it lands is pressed
        out after it: the interpreter must reach a WIN, in exactly one press
        fewer than remained. That takes the labels out of the model and puts them
        through the real thing, and it is what stands behind the tie sets on
        level 4, where nothing can be fielded backwards more than four layers.

    Building the two-sided fields is the bulk of the runtime here: several
    minutes and several GB on level 4 (the generation run itself reads the disk
    cache and sweeps nothing).
    """
    _solver, game, expert = _new()
    eng = game._engine
    bad = 0
    for level in range(game.n_levels):
        board, start = _seat(game, expert, level)
        plan = expert.plan(eng, level)
        if plan is None:                                         # pragma: no cover
            print(f"  L{level}: UNSOLVED")
            bad += 1
            continue
        keys, dist, depth = _pure_backward(board, start, cap)
        state = start
        checked = skipped = 0
        for i, press in enumerate(plan):
            probe = _keys(np.array([state[0]], dtype=np.uint64),
                          np.array([state[1]], dtype=np.uint64))
            loc = _locate(keys, probe)[0]
            if loc < 0:
                skipped += 1
                state = board.step_one(state, press)
                continue
            checked += 1
            rest = int(dist[loc])
            if rest != len(plan) - i:
                bad += 1
                print(f"  L{level}: step {i} is {len(plan) - i} from the win by "
                      f"the sweep and {rest} by the backward-only field")
            best = []
            for d in DIRS:
                nxt = board.step_one(state, d)
                if nxt == state:
                    continue
                at = _locate(keys, _keys(np.array([nxt[0]], dtype=np.uint64),
                                         np.array([nxt[1]], dtype=np.uint64)))[0]
                if at >= 0 and int(dist[at]) == rest - 1:
                    best.append(d)
            shipped = plan.optsets[i]
            if sorted(best) != sorted(shipped):
                bad += 1
                print(f"  L{level}: step {i} ships {shipped} and the "
                      f"backward-only field says {best}")
            state = board.step_one(state, press)
        del keys, dist                       # before the two-sided sweep below

        # -- pass 2: every shipped label, pressed on the interpreter ----------
        field = expert.field(board)
        game.set_level(level)
        state = start
        labels = 0
        for i, press in enumerate(plan):
            rest = len(plan) - i
            before = snapshot(eng)
            for alt in plan.optsets[i]:
                labels += 1
                eng.step(alt)
                landed = expert.read(eng)[1]
                if landed != board.step_one(state, alt):         # pragma: no cover
                    bad += 1
                    print(f"  L{level}: step {i}: the interpreter and the model "
                          f"disagree on where {alt} lands")
                tail = field.plan(landed)
                if tail is None or len(tail) != rest - 1:
                    bad += 1
                    print(f"  L{level}: step {i} calls {alt} optimal, but the "
                          f"field finishes from there in "
                          f"{None if tail is None else len(tail)} rather than "
                          f"{rest - 1}")
                else:
                    for direction in tail:
                        eng.step(direction)
                    if not eng.check_win():                      # pragma: no cover
                        bad += 1
                        print(f"  L{level}: step {i}: pressing {alt} and then "
                              f"the field's plan does not win")
                restore(eng, before)
            eng.step(press)
            state = board.step_one(state, press)
        print(f"  L{level}: backward-only field reached depth {depth} "
              f"({checked} of {len(plan)} plan steps re-derived"
              + (f", {skipped} beyond it" if skipped else "")
              + f"); {labels} labels pressed on the interpreter"
              + (" -- OK" if not bad else " -- see above"))
    print(f"verify: {bad} problems")
    return 1 if bad else 0


def _engine_key(eng, bg: int) -> frozenset:
    """A canonical engine state: every non-background object cell."""
    return frozenset((r, c, o)
                     for r, row in enumerate(eng.grid)
                     for c, cell in enumerate(row)
                     for o in cell if o != bg)


def _engine_proof(cap: int = 120_000) -> int:
    """Re-derive ``d*`` and every tie set with NO native model at all.

    The whole reachable space is walked by pressing real buttons on the real
    interpreter -- the successor of a board is whatever `PSEngine.step` makes of
    it, the goal test is `check_win`, and neither `_Board` nor `_Field` is
    called. One backward sweep over the recorded edges then gives the exact
    distance field, which must agree with the shipped plan's length and with
    every one of its optimal sets.

    Bounded by ``cap`` boards, so only the levels small enough are proved this
    way: levels 0 and 1 (632 and 44,176 boards). The bigger three are the reason
    the native model exists -- level 2 alone is 3.7 million boards to depth 24,
    which is minutes per press through the interpreter rather than seconds for
    the whole sweep.
    """
    _solver, game, expert = _new()
    eng = game._engine
    bg = game._game.obj_name_to_idx["background"]
    bad = 0
    for level in range(game.n_levels):
        _board, _state = _seat(game, expert, level)
        plan = expert.plan(eng, level)
        game.set_level(level)
        states = [snapshot(eng)]
        index = {_engine_key(eng, bg): 0}
        succ: list = []
        won: list = []
        i = 0
        blown = False
        while i < len(states):
            restore(eng, states[i])
            if eng.check_win():
                succ.append(None)                # a win ENDS the level
                won.append(True)
                i += 1
                continue
            won.append(False)
            row = []
            for d in DIRS:
                restore(eng, states[i])
                eng.step(d)
                key = _engine_key(eng, bg)
                j = index.get(key)
                if j is None:
                    j = len(states)
                    index[key] = j
                    states.append(snapshot(eng))
                row.append(j)
            succ.append(tuple(row))
            i += 1
            if len(states) > cap:
                blown = True
                break
        if blown:
            print(f"  L{level}: skipped -- over {cap} boards "
                  f"(this is what the native model is for)")
            continue

        pred: list = [[] for _ in states]
        for src, row in enumerate(succ):
            if row is not None:
                for dst in row:
                    pred[dst].append(src)
        dist = [-1] * len(states)
        queue = deque()
        for j, is_won in enumerate(won):
            if is_won:
                dist[j] = 0
                queue.append(j)
        while queue:
            j = queue.popleft()
            for src in pred[j]:
                if dist[src] < 0:
                    dist[src] = dist[j] + 1
                    queue.append(src)

        if dist[0] != len(plan):
            bad += 1
            print(f"  L{level}: the interpreter says d* = {dist[0]}, "
                  f"the shipped plan is {len(plan)} presses")
        node = 0
        ties = 0
        here = 0
        for step, press in enumerate(plan):
            rest = dist[node]
            best = [DIRS[k] for k, nxt in enumerate(succ[node])
                    if dist[nxt] == rest - 1]
            ties += len(best) > 1
            if sorted(best) != sorted(plan.optsets[step]):
                here += 1
                print(f"  L{level}: step {step} ships {plan.optsets[step]} and "
                      f"the interpreter says {best}")
            node = succ[node][DIRS.index(press)]
        bad += here
        print(f"  L{level}: {len(states)} boards walked on the interpreter, "
              f"d* = {dist[0]}, {len(plan)} steps and {ties} tie sets "
              f"{'agree' if not here else 'DISAGREE'}")
    print(f"engine: {bad} problems")
    return 1 if bad else 0


def _audit() -> int:
    """Assert every cell COMPOSITION a board of this game can hold is distinct in
    the frame, at every cell size the five levels render at.

    The pair this exists for is ``crate+target`` against ``crate``, and
    ``player+target`` against ``player``: the win condition is ``All Target on
    Crate``, so a body that hides the target it stands on makes a winning board
    indistinguishable from a losing one -- and in a game whose bodies SWAP, both
    of them are on and off the target squares constantly. That is the bug the
    sibling ps:swap_sokoban shipped with (its Target was a ring on exactly the
    nine pixels its Player paints over); this game's Target is a full 5x5 frame
    and survives under both bodies, which is what this measures.

    Whole frames are compared, not cell crops: `_render_frame` upscales the board
    and centre-pads it, so the output's cell grid is not ``cell_px``-aligned and
    an arithmetic crop reads the wrong window (the ps:explod lesson).

    ``wall + target`` is absent on purpose: the legend has no character for it,
    no level places one, and a target under a wall would make a level unwinnable
    rather than misread. The loop asserts that no level ships one.
    """
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    sizes: dict = {}
    for level in range(game.n_levels):
        game.set_level(level)
        for row in eng.grid:
            for cell in row:
                assert not (idx["wall"] in cell and idx["target"] in cell), \
                    "a level ships a Target under a Wall"
        sizes.setdefault((eng.height, eng.width), []).append(level)

    comps = [(), ("wall",), ("target",), ("player",), ("crate",),
             ("player", "target"), ("crate", "target")]
    clashes_total = 0
    for (h, w), levels in sorted(sizes.items()):
        shots = {}
        for objs in comps:
            eng.height, eng.width = h, w
            eng.grid = [[{idx["background"]} | {idx[o] for o in objs}
                         for _ in range(w)] for _ in range(h)]
            eng._position_index_dirty = True
            shots[objs] = np.asarray(_render_frame(eng, g)).copy()
        clashes = [(a, b) for a, b in itertools.combinations(shots, 2)
                   if np.array_equal(shots[a], shots[b])]
        clashes_total += len(clashes)
        name = lambda t: "+".join(t) if t else "floor"           # noqa: E731
        print(f"{h}x{w} board (level{'s' if len(levels) > 1 else ''} "
              f"{', '.join(str(x) for x in levels)}), "
              f"cell {max(1, min(64 // h, 64 // w))}px")
        for objs in comps:
            painted = int((shots[objs] != shots[()]).sum())
            over = ""
            if len(objs) == 2:
                base = (objs[0],)
                over = (f", {int((shots[objs] != shots[base]).sum()):4d} px of "
                        f"Target survive under the {objs[0].capitalize()}")
            print(f"  {name(objs):16s} {painted:4d} px differ from bare floor"
                  f"{over}")
        for a, b in clashes:
            print(f"  IDENTICAL: {name(a)} == {name(b)}")
    print("audit clean" if not clashes_total
          else f"AUDIT FAILED: {clashes_total} indistinguishable pairs")
    return 0 if not clashes_total else 1


def _transform(k: int, mirror: bool):
    """``(cell_fn, dims_fn, direction_map)`` for one element of the 8-group:
    mirror left-right first, then turn clockwise ``k`` times.

    The direction map is DERIVED from the same linear part that moves the cells,
    so the two cannot drift apart."""
    def cell(rc, hw):
        r, c = rc
        h, w = hw
        if mirror:
            c = w - 1 - c
        for _ in range(k):
            r, c, h, w = c, h - 1 - r, w, h
        return (r, c)

    def dims(hw):
        h, w = hw
        return (w, h) if k % 2 else (h, w)

    dmap = {}
    for name, (dr, dc) in _DELTA.items():
        if mirror:
            dc = -dc
        for _ in range(k):
            dr, dc = dc, -dr
        dmap[name] = next(n for n, d in _DELTA.items() if d == (dr, dc))
    return cell, dims, dmap


def _pieces(game, expert) -> tuple:
    """The player and crate cells of the engine's board as ``(row, col)`` sets --
    the presentation-independent reading a transform can be checked against."""
    eng = game._engine
    players, crates = set(), set()
    for r, row in enumerate(eng.grid):
        for c, cell in enumerate(row):
            if cell & expert.player_ids:
                players.add((r, c))
            if cell & expert.crate_ids:
                crates.add((r, c))
    return frozenset(players), frozenset(crates)


def _symmetry(walk: int = 200) -> int:
    """Prove ON THE INTERPRETER that all eight presentations are exact
    symmetries of the mechanic -- which is what entitles this game to its
    augmentation.

    The adapter applies a ROTATION (0..3) to every ps: game, so the four turns
    have to be exact or the corpus is wrong, and ``_Swap_the_block!`` is in
    `PuzzleScriptAdapter._FLIP_GAMES` as well, so the four mirrored ones have to
    be too. The argument is recorded beside the game's entry in that set; this
    is the measurement, and it belongs in a number rather than in prose.

    Each level is replayed twice per presentation: its own PLAN, which is the
    sequence the corpus records, and a seeded RANDOM WALK, which does what a
    plan never does (shove crates into walls, drag them back off their targets,
    train several players up behind one crate, press the unbound ACTION key).
    Every board along the way must be the exact transform of the unaugmented
    one.
    """
    _solver, game, expert = _new()
    eng, g = game._engine, game._game
    bad = 0
    for level in range(game.n_levels):
        _board, _state = _seat(game, expert, level)
        plan = list(expert.plan(eng, level))
        rng = random.Random(f"swap_the_block:symmetry:{level}")
        runs = {"plan": plan,
                "walk": [rng.choice(DIRS + ("action",)) for _ in range(walk)]}
        layout = g.levels[level]
        hw = (len(layout), len(layout[0]))
        notes = []
        for kind, presses in runs.items():
            eng.load_level(layout)
            ref = [_pieces(game, expert)]
            for d in presses:
                eng.step(d)
                ref.append(_pieces(game, expert))
            for k, mirror in itertools.product(range(4), (False, True)):
                if (k, mirror) == (0, False):
                    continue
                cell, dims, dmap = _transform(k, mirror)
                th, tw = dims(hw)
                turned = [[set() for _ in range(tw)] for _ in range(th)]
                for r, row in enumerate(layout):
                    for c, objs in enumerate(row):
                        tr, tc = cell((r, c), hw)
                        turned[tr][tc] = set(objs)
                eng.load_level(turned)
                for i, d in enumerate(presses):
                    eng.step(d if d == "action" else dmap[d])
                    want = tuple(frozenset(cell(rc, hw) for rc in side)
                                 for side in ref[i + 1])
                    if _pieces(game, expert) != want:
                        notes.append(f"{kind}:rot{k}{'m' if mirror else ''}"
                                     f"@press{i}")
                        break
        bad += len(notes)
        print(f"  L{level}: ({len(plan)} plan + {walk} random) presses "
              f"x 7 presentations: "
              f"{'SYMMETRIC' if not notes else 'CHIRAL ' + ', '.join(notes[:3])}")
    print("symmetry clean -- the rotation and flip augmentation is exact"
          if not bad
          else f"SYMMETRY FAILED: {bad} presentations diverge")
    return 0 if not bad else 1


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        problems = selfcheck()
        print(f"selfcheck: {problems} mismatches")
        sys.exit(1 if problems else _plans())
    if "--plans" in sys.argv:
        sys.exit(_plans())
    if "--verify" in sys.argv:
        sys.exit(_verify())
    if "--engine" in sys.argv:
        sys.exit(_engine_proof())
    if "--audit" in sys.argv:
        sys.exit(_audit())
    if "--symmetry" in sys.argv:
        sys.exit(_symmetry())
    sys.exit(SwapTheBlockSolver.main())
