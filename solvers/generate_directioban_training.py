"""Generate Phase-1 training data for the PuzzleScript game ps:directioban
("Directioban", Franklin P. Dyer's Sokoban where every crate moves by its own
rule).

The harness -- the rotation contract, the trajectory recorder, the plan cache
and the BaseSolver plumbing -- lives in `solvers/common/ps_astar.py`, shared with
the other ps: generators. This file is the game-specific part: a native model of
the four crate rules, an exact distance field over it, and the levels.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_directioban",
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
action (post rotation/flip remap), i.e. the button an agent presses in the
augmented view, so replaying the recorded actions reproduces the recorded frames
exactly. Every step also carries the full set of equally-optimal presses.

The game
--------
Sokoban with four crates that each obey a different movement rule, and the
sprite IS the rule -- which is why no crate may be recoloured apart (see "The
palette fix"). The win is ``All Target on Crate``: every target has to end up
covered, spare crates are allowed, and nothing else matters.

  Crate1 (vertical bars)   moves only VERTICALLY. The player pushes it (standing
                           behind it) AND pulls it (standing on the far side and
                           walking away) -- ``VERTICAL [ < Player | Crate1 ]``.
                           Every rule that can move it is VERTICAL-prefixed, so
                           it is locked to its column for the whole level.
  Crate2 (horizontal bars) the same, mirrored: horizontal only, push and pull,
                           locked to its row forever.
  Crate3 (hollow box)      the ordinary Sokoban crate: pushed in any direction,
                           never pulled.
  Crate4 (grid)            TELEKINETIC and never pushed by contact at all. When
                           the player moves vertically, every Crate4 sharing the
                           player's ROW moves the same way, at any distance and
                           straight through walls; a horizontal move does the
                           same to every Crate4 in the player's COLUMN. Walking
                           into one just stops the player.

Two consequences drive the whole design of the levels:

  * A target can only ever be covered by a crate that can REACH it, and Crate1 /
    Crate2 can never change column / row. A target off their line needs some
    other crate.
  * Once the player shares an axis with a Crate4 they move together along it
    forever -- a vertical move shifts both -- so the ONLY way to let go of a
    Crate4 is to park it against a wall (or another crate) and keep walking.
    Crate4 levels are therefore ice-slide puzzles: shove until it jams, release,
    re-approach on the other axis. An empty room with one Crate4 and an
    off-axis target is unsolvable, which is the first thing the field reports.

Rule ORDER matters and is not symmetric: the crate1 / crate2 / crate3 chain
rules are applied before the Crate4 ones, so a moving Crate1 can shove a Crate3
but a Crate3 can never shove a Crate1, and a telekinetically-moved Crate4 pushes
only other Crate4s. The interpreter applies the rule list once per turn (nothing
here uses ``again``), which is what makes that ordering observable.

The expert
----------
A NATIVE model of the ruleset (`_Board`) plus an EXACT distance field: forward
BFS over the level's whole reachable state space, then a reverse BFS from the
winning states back over the collected edges. That gives, for free and exactly:

  * a shortest plan (the field is a BFS field in presses, the unit the agent
    pays), so no heuristic has to be trusted;
  * the optimal-action SET at every state on it -- every press whose successor
    is one step closer -- which is what the policy is trained against;
  * a total solvability verdict per level, used while authoring the boards.

A native model is what makes that affordable: an interpreter step costs ~320us,
so the 280k-state field of the biggest level would be ~6 minutes of A*-free
search on the engine, versus 6 seconds here. The model's fidelity is not
assumed -- ``--fuzz`` plays random boards (random walls, all four crate types,
up to 10 crates) move-for-move against the real interpreter and compares the
full grid after every step; 37.5k transitions with zero mismatches is what it
was accepted on. Re-run it after ANY change to the .txt rules or to the
adapter's rule/force code.

`PSExpert` supplies everything around that: the in-memory memo, the on-disk
start-plan cache and the snapshot discipline. Only `_search` is overridden.

The levels
----------
The shipped file has six boards, and two of them are EMPTY rooms -- no player,
no crates, no targets -- which the ``All Target on Crate`` win reads as
vacuously won at step 0. Those two are gone. The four real ones are kept (they
are the best boards in the game) and sixteen more are authored around them, so
each rule gets introduced on its own before the boards start combining them:

    level  size  crates          plan  states   what it teaches
    0      7x7   1x3                6     578   the ordinary push
    1      7x7   2x1                7     556   vertical-only, pushed
    2      7x7   1x1                5      50   vertical-only, PULLED (its top
                                                is walled, so pushing is out)
    3      7x7   1x2                5      50   the same lesson, mirrored
    4      7x7   2x2                6     554   two horizontal-only crates
    5      7x7   1x1 1x2            7     531   both axis-locked crates at once
    6      7x7   2x4                4      18   SHIPPED: telekinesis from the
                                                room next door
    7      7x7   1x4                8      68   telekinesis + parking it on a
                                                wall to get away from it
    8      7x7   2x3               14    6877   two ordinary crates
    9      7x7   2x4               13     109   two telekinetic crates
    10     7x7   2x3 2x4           15     244   SHIPPED: two rooms
    11     7x7   1x3 1x4           16     371   ordinary + telekinetic
    12     7x7   3x3               14   28850   chain pushes
    13     7x7   3x3               16   28638   chain pushes, tighter
    14     7x7   1x1 1x2 1x3       18   10040   everything but telekinesis
    15     7x7   2x4               19     173   telekinesis around a wall
    16     9x9   1x1 1x3 1x4       21  123880   a bigger board, all three
    17     7x7   1x3 1x4           22     594   two targets in one corner
    18     7x7   1x1 1x2 1x3       22    7679   long hauls on locked axes
    19     9x9   1x1 1x3 1x4       30   21807   a bigger board again
    20     7x7   5x3               50  281763   SHIPPED: the author's opener
    21     7x7   2x1 1x2 3x3       54  277777   SHIPPED: the author's finale

``plan`` is the exact optimum (the field's distance at the start state), and
``states`` is the size of the reachable space it was read off. Every board is
solvable by construction -- the field says so before it ships -- none is won at
step 0, and the longest plan is 54 moves, comfortably inside the adapter's
200-step per-level budget even after an exploration prefix.

The 22 fields cost ~15s to build in total, once, and are written to
``data/directioban_plans.json`` (start plan + optimal sets per level) so no
shard of `parallelize_generator` re-derives them. Delete that file to re-derive.

Optimal-action sets
-------------------
Read straight off the field: at each step, every direction whose successor state
is one closer to a win. That is the exact tie set, not an approximation of one --
there is no "walk to the push cell" shortcut here to annotate separately,
because in this game a bare move is not free (it drags Crate1/Crate2 neighbours
along and telekineses every aligned Crate4), so the field is the only honest
source. No step ever ships unlabelled (the always-emit-optimal-targets rule).

The palette fix
---------------
Three collisions, all in the original art, all fixed in
``data/puzzlescript_games/Directioban.txt`` (no rule, sprite SHAPE or level was
touched):

  * Every Crate was ``Orange`` and Wall is ``BROWN DARKBROWN`` -- ARC 12 in both
    cases -- so all four crates rendered in exactly the wall's colour. They are
    ``Purple`` (15) now. All four keep ONE colour on purpose: the shape is the
    rule, and a colour per type would offer a cue that does not survive the
    rotation augmentation (a vertical-only crate moves horizontally on screen at
    k=1) while the sprite, which rotates with the board, always does.
  * Player was ``Black Orange White Blue``: its face shared the wall's 12 and
    its legs shared Target's 9. It is ``Black Red White Black``.
  * Target was a hollow RING, and the player's sprite is opaque across exactly
    the ring's cells -- so a player standing on a target hid it completely, and
    the board stopped showing where the remaining work was. Target is a solid
    square now, which shows through at the player's transparent corners and
    through every crate's open middle. (Same shape as the pads in
    ps:dang_im_huge; the general rule is in the ps-palette-collisions note.)

``--audit`` is the regression test: it renders every cell COMPOSITION the game
can show (floor, wall, target, each crate on floor and on a target, player on
floor and on a target) at every cell size the levels actually use and asserts
they are pairwise distinct. Cell size matters -- the 5x5 sprites are sampled
down per cell, and at 4px the middle row is dropped entirely, which makes
Crate2, Crate3 and Crate4 the same picture. That is why no level is bigger than
12 cells on a side.

Augmentation
------------
Engine state after reset is identical for every seed (levels are fixed ASCII
maps), so the only per-(seed, level) variables are the presentation
augmentations: frame rotation (k in {0,1,2,3}) plus an independent horizontal
and vertical flip with the matching directional action remap
(`PuzzleScriptAdapter._FLIP_GAMES`), which re-samples the board's 8-element
symmetry group. The flips are exact symmetries here: the game is gravity-free,
input is screen-relative, and the two axis-locked crates keep their axis under a
mirror (a flip maps vertical to vertical) while their sprites are symmetric
under both flips, so no mirror lands one object's art on another's. Rotation is
consistent for the same reason in the other direction: it turns Crate1's bars
horizontal on screen exactly when it turns its movement horizontal. 22 levels x
16 presentations = 352.

Usage (run from the repo root):
    python solvers/generate_directioban_training.py --episodes 200 \
        --out data/training_multi_level/directioban

    python solvers/generate_directioban_training.py --plans   # level report
    python solvers/generate_directioban_training.py --audit   # rendering audit
    python solvers/generate_directioban_training.py --fuzz    # model vs engine
    python solvers/generate_directioban_training.py --verify  # replay on engine
"""

from __future__ import annotations

import itertools
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.puzzlescript_adapter import _render_frame               # noqa: E402
from solvers.common.ps_astar import PSAStarSolver, PSExpert, Plan     # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent

GAME_NAME = "Directioban"

#: Disk cache of the per-level start plan AND its optimal-action sets.
PLAN_CACHE: Path = _REPO_ROOT / "data" / "directioban_plans.json"

#: Engine direction names, in the order `_Board` indexes them.
DIRS = ["up", "down", "left", "right"]
_DELTA = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_VERTICAL = (0, 1)

#: Give up on a level whose reachable space is bigger than this. Every shipped
#: board settles under 300k; the cap is a runaway guard for a board authored
#: later, and hitting it drops the level rather than the run.
STATE_CAP = 2_000_000


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

class _Board:
    """The static half of a level (size, walls, targets) plus the ruleset.

    A state is ``(player, crates)`` with ``player`` an ``(r, c)`` cell and
    ``crates`` a sorted tuple of ``((r, c), type)`` -- canonical, because crates
    of the same type are interchangeable.

    `step` reproduces one interpreter turn: assign forces by applying the rule
    list IN ORDER (each rule to fixpoint, which is what the engine's
    direction-ordered scan achieves for these propagation rules), then resolve
    them the way `PSEngine._resolve_forces` does -- trace each push chain, drop
    the chains whose endpoint is blocked, drop the chains that claim the same
    cell, move the rest, repeat until stable. It is verified against the real
    interpreter by ``--fuzz``.
    """

    def __init__(self, h, w, walls, targets):
        self.h, self.w = h, w
        self.walls = frozenset(walls)
        self.targets = frozenset(targets)

    # -- dynamics ------------------------------------------------------------
    def step(self, state, di):
        player, crates = state
        occ = dict(crates)
        dr, dc = _DELTA[di]

        if (player[0] + dr, player[1] + dc) in self.walls:
            # [ > Player | Wall ] -> [ Player | Wall ] strips the player's force,
            # and the player is the only source of one: the turn is a no-op.
            # (Off the grid is NOT a Wall object, so the force survives there and
            # only the movement resolution stops it -- every level is walled.)
            return state

        forces = {player: di}

        # Crate1: VERTICAL push (crate ahead) and pull (crate behind), then the
        # vertical chain rule. Crate2 is the same statement, rotated.
        for kind, axis in ((1, True), (2, False)):
            if (di in _VERTICAL) == axis:
                ahead = (player[0] + dr, player[1] + dc)
                behind = (player[0] - dr, player[1] - dc)
                if occ.get(ahead) == kind:
                    forces[ahead] = di
                if occ.get(behind) == kind:
                    forces[behind] = di
            self._chain(occ, forces, kind,
                        _VERTICAL if kind == 1 else (2, 3))

        # Crate3: the ordinary push, any direction, then its chain.
        ahead = (player[0] + dr, player[1] + dc)
        if occ.get(ahead) == 3:
            forces[ahead] = di
        self._chain(occ, forces, 3, (0, 1, 2, 3))

        # Crate4: telekinesis. A vertical move takes every Crate4 in the
        # player's ROW with it (the `[ ^ Player |...| Crate4 ]` pair expanded
        # over all four rule directions), a horizontal move every Crate4 in its
        # COLUMN. Distance and walls in between are irrelevant.
        axis = 0 if di in _VERTICAL else 1
        for q, t in occ.items():
            if t == 4 and q[axis] == player[axis]:
                forces[q] = di
        self._chain(occ, forces, 4, (0, 1, 2, 3))

        return self._resolve(player, occ, forces)

    def _chain(self, occ, forces, kind, dirs):
        """``[ > Crate | CrateN ] -> [ > Crate | > CrateN ]``, to fixpoint.

        Only crates propagate (the player's own force is handled by the push /
        pull rules above), and only along ``dirs`` -- the crate1 and crate2
        chain rules carry the VERTICAL / HORIZONTAL prefix."""
        changed = True
        while changed:
            changed = False
            for pos, d in list(forces.items()):
                if d not in dirs or pos not in occ:
                    continue
                nb = (pos[0] + _DELTA[d][0], pos[1] + _DELTA[d][1])
                if occ.get(nb) == kind and forces.get(nb) != d:
                    forces[nb] = d
                    changed = True

    def _resolve(self, player, occ, forces):
        for _ in range(20):
            movable, blocked, deferred = [], set(), False
            for pos, d in list(forces.items()):
                if pos in blocked:
                    continue
                dr, dc = _DELTA[d]
                chain, cur, free, blocker_moves = [pos], pos, False, False
                while True:
                    nb = (cur[0] + dr, cur[1] + dc)
                    if not (0 <= nb[0] < self.h and 0 <= nb[1] < self.w):
                        break                       # off the board: blocked
                    if nb in self.walls:
                        break
                    if nb not in occ and nb != player:
                        free = True
                        break
                    if forces.get(nb) == d:
                        chain.append(nb)            # same-direction chain member
                        cur = nb
                    else:
                        # A blocker heading elsewhere may still vacate this pass.
                        blocker_moves = forces.get(nb) is not None
                        break
                if free:
                    movable.append((chain, d))
                elif blocker_moves:
                    deferred = True
                else:
                    for ent in chain:
                        forces.pop(ent, None)
                        blocked.add(ent)

            non_head = set()
            for chain, _d in movable:
                non_head.update(chain[1:])          # subsume shorter chains
            live = [(ch, d) for ch, d in movable if ch[0] not in non_head]

            claims, bad = {}, set()
            for i, (chain, d) in enumerate(live):
                for ent in chain:
                    tgt = (ent[0] + _DELTA[d][0], ent[1] + _DELTA[d][1])
                    other = claims.get(tgt)
                    if other is not None and other != i:
                        bad.add(i)
                        bad.add(other)              # two chains, one cell: both stop
                    else:
                        claims[tgt] = i

            moved = False
            for i, (chain, d) in enumerate(live):
                if i in bad:
                    for ent in chain:
                        forces.pop(ent, None)
                    continue
                dr, dc = _DELTA[d]
                for ent in reversed(chain):
                    tgt = (ent[0] + dr, ent[1] + dc)
                    if ent == player:
                        player = tgt
                    else:
                        occ[tgt] = occ.pop(ent)
                    forces.pop(ent, None)
                moved = True
            if not forces or (not moved and not deferred):
                break
        return (player, tuple(sorted(occ.items())))

    def won(self, state):
        return self.targets <= {p for p, _t in state[1]}

    def signature(self):
        return (self.h, self.w, self.walls, self.targets)

    # -- the exact distance-to-win field --------------------------------------
    def field(self, start, cap=STATE_CAP):
        """``{state: presses to a win}`` over everything reachable from
        ``start``, or None if the space is bigger than ``cap``.

        Forward BFS collects the edges (win states are terminal and are not
        expanded), then one reverse BFS from the wins labels every state that
        can still reach one. States that cannot are simply absent."""
        ids = {start: 0}
        states = [start]
        succ = []
        wins = []
        queue = deque([0])
        while queue:
            i = queue.popleft()
            s = states[i]
            if self.won(s):
                succ.append(())
                wins.append(i)
                continue
            row = []
            for di in range(4):
                nxt = self.step(s, di)
                j = ids.get(nxt)
                if j is None:
                    if len(states) >= cap:
                        return None
                    j = len(states)
                    ids[nxt] = j
                    states.append(nxt)
                    queue.append(j)
                row.append(j)
            succ.append(tuple(row))

        pred = [[] for _ in range(len(states))]
        for i, row in enumerate(succ):
            for j in row:
                if j != i:
                    pred[j].append(i)
        dist = [-1] * len(states)
        queue = deque(wins)
        for w in wins:
            dist[w] = 0
        while queue:
            j = queue.popleft()
            for i in pred[j]:
                if dist[i] < 0:
                    dist[i] = dist[j] + 1
                    queue.append(i)
        return {states[i]: d for i, d in enumerate(dist) if d >= 0}


# ---------------------------------------------------------------------------
# Reading a board out of the interpreter
# ---------------------------------------------------------------------------

def read_board(eng, g):
    """``(_Board, state)`` for the interpreter's current grid."""
    inv = {v: k for k, v in g.obj_name_to_idx.items()}
    walls, targets, crates, player = set(), set(), {}, None
    for r, row in enumerate(eng.grid):
        for c, cell in enumerate(row):
            for o in cell:
                name = inv[o]
                if name == "wall":
                    walls.add((r, c))
                elif name == "target":
                    targets.add((r, c))
                elif name == "player":
                    player = (r, c)
                elif name.startswith("crate"):
                    crates[(r, c)] = int(name[-1])
    board = _Board(eng.height, eng.width, walls, targets)
    return board, (player, tuple(sorted(crates.items())))


# ---------------------------------------------------------------------------
# The expert
# ---------------------------------------------------------------------------

class DirectiobanExpert(PSExpert):
    """Exact shortest plans (and exact tie sets) read off a BFS field over the
    native model. See the module docstring for why the model exists and how it
    is verified.

    Only ONE level's field is held at a time: they run to a few hundred thousand
    states and every caller here plans one level and moves on, so keeping them
    all would cost a gigabyte to answer questions nobody asks twice. The disk
    cache (`PSExpert.plan_cache_path`) is what makes that free across runs."""

    #: ACTION5 is bound to nothing in this game.
    directions = DIRS
    plan_cache_path = PLAN_CACHE

    def setup(self) -> None:
        self._field = None
        self._field_sig = None

    def _field_for(self, board, start):
        sig = board.signature()
        if sig != self._field_sig:
            self._field = board.field(start)
            self._field_sig = sig
        return self._field

    def heuristic(self, eng) -> int:                 # pragma: no cover
        raise NotImplementedError("the field is exact; no heuristic is used")

    def _search(self, eng) -> list | None:
        """Walk the field downhill from the engine's current state, recording
        the press taken and every press that would have been equally short."""
        board, state = read_board(eng, self.g)
        dist = self._field_for(board, state)
        if dist is not None and state not in dist:
            # A state the held field never enumerated -- it was built from a
            # different start on the same board. Rebuild from here before
            # concluding the level is lost (if it really is, the rebuilt field
            # leaves this state out too, and the second answer is the true one).
            self._field_sig = None
            dist = self._field_for(board, state)
        if dist is None or state not in dist:
            return None                              # too big, or already lost
        presses, optsets = [], []
        while dist[state] > 0:
            here = dist[state]
            best = []
            nxts = {}
            for di, name in enumerate(DIRS):
                nxt = board.step(state, di)
                nxts[name] = nxt
                if dist.get(nxt, 1 << 30) == here - 1:
                    best.append(name)
            presses.append(best[0])
            optsets.append(best)
            state = nxts[best[0]]
        return Plan(presses, optsets)


class DirectiobanSolver(PSAStarSolver):
    game_id = "puzzlescript_directioban"
    game_name = GAME_NAME
    expert_cls = DirectiobanExpert

    #: Record against the game folder's adapter -- the one `game_envs` hands a
    #: live agent. It is a plain passthrough today; going through it means a
    #: later step limit or sprite fix there cannot silently diverge from what is
    #: taped here.
    game_module_id = "ps:directioban"

    #: The longest plan is 54 moves; the rest is room for the exploration prefix
    #: and its RESET. Stays inside the adapter's own 200-step per-level budget.
    max_steps = 150


# ---------------------------------------------------------------------------
# Reports and checks
# ---------------------------------------------------------------------------

def _report() -> int:
    """Per-level board size, crate mix, exact optimum and tie coverage."""
    solver = DirectiobanSolver()
    game = solver.make_game(0)
    expert = DirectiobanExpert(game)
    total = 0
    for level in range(game.n_levels):
        game.set_level(level)
        eng = game._engine
        _board, state = read_board(eng, game._game)
        mix = {}
        for _cell, t in state[1]:
            mix[t] = mix.get(t, 0) + 1
        found = expert.plan(eng, level)
        if found is None:
            print(f"level {level:2d}: NO PLAN")
            continue
        sets = getattr(found, "optsets", []) or []
        ties = sum(1 for s in sets if len(s) > 1)
        total += len(found)
        room = "ok" if len(found) < game._max_steps else "OVER BUDGET"
        print(f"level {level:2d}: {eng.height}x{eng.width} "
              f"crates {' '.join(f'{n}x{t}' for t, n in sorted(mix.items()))}"
              f"{'':4s} {len(found):3d} moves ({room}), "
              f"{ties:3d} steps with a tie set "
              f"({ties / max(1, len(found)):.0%})")
    print(f"total {total} moves over {game.n_levels} levels")
    return 0


def _verify() -> int:
    """Replay every level's plan on the REAL interpreter and require a WIN.

    The plans come off a model; this is the line that says the model and the
    interpreter agree about the boards that actually ship."""
    solver = DirectiobanSolver()
    game = solver.make_game(0)
    expert = DirectiobanExpert(game)
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        eng = game._engine
        plan = expert.plan(eng, level)
        if plan is None:
            print(f"level {level:2d}: NO PLAN")
            bad += 1
            continue
        game.set_level(level)
        for d in plan:
            eng.step(d)
        ok = eng.check_win()
        bad += not ok
        print(f"level {level:2d}: {len(plan):3d} presses -> "
              f"{'WIN' if ok else 'NOT WON'}")
    print("verify clean" if not bad else f"VERIFY FAILED on {bad} levels")
    return 0 if not bad else 1


def _audit() -> int:
    """Assert every cell COMPOSITION is distinct at every cell size in use.

    Composition, not object: a crate ON a target has to read as both, and what
    separates the four crate types is a few pixels of sprite that the per-cell
    downsample can drop. See the palette note in the module docstring."""
    game = DirectiobanSolver().make_game(0)
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    comps = {"floor": (), "wall": ("wall",), "target": ("target",),
             "player": ("player",), "player_on_target": ("target", "player")}
    for n in "1234":
        comps[f"crate{n}"] = (f"crate{n}",)
        comps[f"crate{n}_on_target"] = ("target", f"crate{n}")

    sizes = {}
    for level in range(game.n_levels):
        game.set_level(level)
        sizes.setdefault((eng.height, eng.width), []).append(level)

    bad = 0
    for (h, w), levels in sorted(sizes.items()):
        shots = {}
        for name, objs in comps.items():
            eng.height, eng.width = h, w
            eng.grid = [[{idx["background"]} | {idx[o] for o in objs}
                         for _ in range(w)] for _ in range(h)]
            eng._position_index_dirty = True
            frame = np.asarray(_render_frame(eng, g))
            cell = 64 // max(h, w)
            r, c = h // 2, w // 2
            shots[name] = frame[cell * r:cell * (r + 1),
                                cell * c:cell * (c + 1)].copy()
        clashes = [(a, b) for a, b in itertools.combinations(shots, 2)
                   if np.array_equal(shots[a], shots[b])]
        bad += len(clashes)
        print(f"{h:2d}x{w:2d} (cell {64 // max(h, w)}px, levels "
              f"{','.join(str(x) for x in levels)}): "
              f"{'OK' if not clashes else 'IDENTICAL ' + str(clashes)}")
    print("audit clean" if not bad
          else f"AUDIT FAILED: {bad} indistinguishable pairs")
    return 0 if not bad else 1


def _fuzz(n_boards: int = 1500, n_steps: int = 25, seed: int = 0) -> int:
    """Differential fuzz: random boards played move-for-move against the real
    interpreter, full grid compared after every step.

    This is the only thing standing between `_Board` and a silently wrong
    corpus, so it covers what the shipped levels do not: dense crate mixes, all
    four types on one board, interior walls, and every board size the sprites
    are legible at."""
    game = DirectiobanSolver().make_game(0)
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    rng = random.Random(seed)
    mismatches = 0
    steps = 0
    for _ in range(n_boards):
        h, w = rng.choice([5, 6, 7, 8]), rng.choice([5, 6, 7, 8])
        mix = rng.choice([[1, 2, 3, 4], [4, 4, 4, 3], [1, 1, 2, 2],
                          [3, 3, 4, 4], [1, 2, 3, 4, 4, 4]])
        cells = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
        rng.shuffle(cells)
        walls = {(r, c) for r in range(h) for c in range(w)
                 if r in (0, h - 1) or c in (0, w - 1)}
        nwall = rng.randint(0, 4)
        walls |= set(cells[:nwall])
        rest = cells[nwall:]
        ncrate = rng.randint(1, max(1, min(10, len(rest) - 2)))
        crates = {rest[i]: rng.choice(mix) for i in range(ncrate)}
        player = rest[ncrate]
        targets = set(rest[ncrate + 1: ncrate + 1 + rng.randint(0, 4)])

        board = _Board(h, w, walls, targets)
        state = (player, tuple(sorted(crates.items())))
        grid = []
        for r in range(h):
            row = []
            for c in range(w):
                cell = {idx["background"]}
                if (r, c) in walls:
                    cell = {idx["wall"]}
                else:
                    if (r, c) in targets:
                        cell.add(idx["target"])
                    if (r, c) == player:
                        cell.add(idx["player"])
                row.append(cell)
            grid.append(row)
        for (r, c), t in crates.items():
            grid[r][c].add(idx[f"crate{t}"])
        eng.grid, eng.height, eng.width = grid, h, w
        eng._position_index_dirty = True
        eng._rule_noop_cache.clear()

        for _s in range(n_steps):
            di = rng.randrange(4)
            eng.step(DIRS[di])
            _b, truth = read_board(eng, g)
            state = board.step(state, di)
            steps += 1
            if state != truth:
                mismatches += 1
                print(f"MISMATCH after {DIRS[di]}\n  model {state}\n"
                      f"  engine {truth}")
                state = truth
                if mismatches > 5:
                    return 1
    print(f"fuzz: {steps} transitions, {mismatches} mismatches")
    return 0 if not mismatches else 1


if __name__ == "__main__":
    if "--plans" in sys.argv:
        sys.exit(_report())
    if "--verify" in sys.argv:
        sys.exit(_verify())
    if "--audit" in sys.argv:
        sys.exit(_audit())
    if "--fuzz" in sys.argv:
        sys.exit(_fuzz())
    sys.exit(DirectiobanSolver.main())
