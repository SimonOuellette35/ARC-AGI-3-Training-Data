"""Generate Phase-1 training data for the PuzzleScript game ps:epicjamgame
("EpicJamGame", Toombler -- the game where you never touch the thing you are
trying to move).

The harness -- the rotation contract, the trajectory recorder, the plan cache and
the BaseSolver plumbing -- lives in `solvers/common/ps_astar.py`, shared with the
other ps: generators. This file is the game-specific part: a native model of the
line-of-sight chase, an exact distance field over it, and the levels.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_epicjamgame",
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
You walk; a Spirit runs. The win is ``[stationary Spirit Target] -> win``: the
spirit has to be standing on the target on a turn when it does not move.

You cannot push the spirit by touching it. The only thing that moves it is your
LINE OF SIGHT:

  * Whenever the player and the spirit share a row or a column, the spirit takes
    one step directly AWAY from the player -- and a `late ... again` rule
    re-runs the turn for as long as they stay in line, so the spirit does not
    step, it BOLTS: one keypress that lines you up sends it running until
    something stops it.
  * A Wall or a Crate anywhere strictly between the two blocks the sightline and
    the spirit does not budge (that is what the four Blocking markers compute).
    The same Wall or Crate immediately BEHIND the spirit is what stops the run.
  * Water blocks nothing. The spirit runs over it happily; the player cannot
    enter it at all.

So a press is a shove whose LENGTH you do not control, and the three things that
can catch a runaway spirit are a wall, a crate, and the edge of the board.

Two consequences shape every level and the whole search:

  * MOST MOVES ARE FATAL. A row or column with nothing to stop the spirit ends
    at the board edge, and the border ring is a prison: to push a spirit off the
    edge cell (r, 0) to the right, the player would have to stand at (r, -1).
    Once there, the spirit can only ever slide ALONG that edge, so unless the
    target is on it the level is lost -- silently, with the level still looking
    playable. On the shipped boards 68% of the reachable states are already
    lost; the corpus can only be recorded because recovery here is a RESET.
  * The player is boxed in by its own sightlines. Crossing the spirit's row or
    column always shoves it, so getting to the far side of the board without
    ruining it means crossing where a wall breaks the sightline, or spending the
    shove on something useful. That is the puzzle.

Crates are ordinary sokoban crates (``[ > Player | Crate ] -> [ > Player | >
Crate ]``, no chain rule, so two in a row jam), and they are the only catcher the
player can BUILD. A crate shoved onto water can never be moved again -- the
player's move into a water cell is cancelled before the push rule sees it.

The expert
----------
A NATIVE model of the ruleset (`_Board`) plus an EXACT distance field: forward
BFS over the level's reachable state space, then a reverse BFS from the winning
transitions back over the collected edges. That gives, for free and exactly:

  * a shortest plan (the field is a BFS field in presses, the unit the agent
    pays), so no heuristic has to be trusted;
  * the optimal-action SET at every state on it -- every press whose successor is
    one step closer -- which is what the policy is trained against;
  * a total solvability verdict per level, used while authoring the boards.

The model's fidelity is not assumed: ``--fuzz`` plays random boards (random
walls, water, crates) move-for-move against the real interpreter and compares the
piece positions AND the win flag after every step. 21k transitions with zero
mismatches is what it was accepted on. Re-run it after ANY change to the .txt
rules or to the adapter's rule/force code.

DEAD-SPIRIT PRUNING is what makes the field affordable, and it is worth its own
paragraph because without it the biggest board costs 2.2M states / 1.2 GB / 45 s
and with it 62k / 130 MB / 1.1 s. `_live_cells` answers "could the spirit ever
reach a target from this cell?" in a relaxation where crates do not exist and a
run may stop wherever it likes -- a strict over-approximation of the real
dynamics, so a state it calls dead really is dead. Every state whose spirit sits
outside that set is dropped un-expanded, which deletes the whole "spirit escaped
into the water and is sliding around the border forever" subtree. It is exact:
all 23 fields report the same start distances and the same live-state counts
computed with and without it.

`PSExpert` supplies everything around that: the in-memory memo, the on-disk
start-plan cache and the snapshot discipline. Only `_search` is overridden.

The levels
----------
Levels 15-22 are Toombler's own boards in shipped order (level 17 is the board he
left commented out in the file -- it plays, so it is back in). Levels 0-14 are
authored for this corpus: three tutorials that isolate the three catchers, then a
ramp, because the shipped boards open at 23 presses with nothing that introduces
the mechanic:

    level  size   crates  plan  states  what it teaches
    0      8x8    -          4      15  the board EDGE catches the spirit
    1      8x9    -          4      91  a WALL catches it one cell earlier
    2      7x10   1         14     873  BUILD the catcher: shove a crate behind
                                        the target first, and cross the spirit's
                                        column where a wall breaks the sightline
    3      9x9    -          6      94  no water: only the four edges catch
    4      10x11  1         11      45  a crate that is already in place
    5      9x9    -         12     277
    6      10x10  -         13      53
    7      10x11  1         14     131
    8      10x11  1         16    1100
    9      10x10  -         17      91
    10     10x11  1         20    5107
    11     11x11  2         21  150827  the widest board of the set
    12     10x10  -         25     114
    13     11x11  2         33   93968
    14     11x11  2         35   11870
    15     10x10  -         23     185  SHIPPED 1
    16     11x11  -         35     118  SHIPPED 2
    17     10x11  -         15      75  SHIPPED, was commented out
    18     10x12  -         26     184  SHIPPED 3
    19     11x12  -         41     117  SHIPPED 4
    20     10x15  1         37    5322  SHIPPED 5, the first crate board
    21     11x13  1         73    9847  SHIPPED 6, the longest plan in the game
    22     11x14  2         51   61933  SHIPPED 7

``plan`` is the exact optimum (the field's distance at the start state) and
``states`` the size of the pruned space it was read off. Every board is solvable
by construction -- the field says so before it ships -- none is won at step 0,
and the longest plan is 73 presses, comfortably inside the adapter's 200-step
per-level budget even after the exploration prefix.

The 23 fields cost ~12 s to build in total, once, and are written to
``data/epicjamgame_plans.json`` (start plan + optimal sets per level) so no shard
of `parallelize_generator` re-derives them. Run ``--plans`` once before a
parallel run; delete that file to re-derive.

Optimal-action sets
-------------------
Read straight off the field: at each step, every direction whose successor is one
closer to a win. That is the exact tie set, not an approximation of one. There is
no "walk to the push cell" shortcut to annotate separately here, because in this
game a bare move is NOT free -- stepping into the spirit's line is itself the
shove -- so the field is the only honest source. No step ever ships unlabelled
(the always-emit-optimal-targets rule).

The palette fix
---------------
Four collisions, all in the original art, all fixed in
``data/puzzlescript_games/EpicJamGame.txt`` (no rule and no shipped level was
touched):

  * Target was a hollow RING in ``DarkBlue``, and the Spirit is a solid 3x3 block
    that covers exactly that ring -- so the WINNING frame, spirit on target, was
    pixel-identical to the spirit standing anywhere else. Target is a solid
    square now: its border shows around the spirit, through the crate's open
    middle and at the player's transparent corners.
  * ``DarkBlue`` is ARC 9 and so is Water's ``#27A599``: the target was painted
    in the colour of the terrain that fills half of every board.
  * Wall was ``BROWN DARKBROWN`` = ARC 12 + 13, and 12 is exactly the
    Background's colour, so most of the brick pattern was floor-coloured. Now
    ``Gray DarkGray``.
  * Crate was ``darkbrown`` = ARC 13 = the Wall's other colour, i.e. the only
    pushable object in the game rendered as scenery. Now ``Pink``.
  * (Also: the Player's body was ARC 9, Water again. Now black.)

``--audit`` is the regression test: it renders every cell COMPOSITION the game
can show -- including spirit-on-water and crate-on-water, which are reachable and
irreversible -- at every cell size the levels actually use, and asserts they are
pairwise distinct.

Augmentation
------------
Engine state after reset is identical for every seed (levels are fixed ASCII
maps), so the only per-(seed, level) variables are the presentation
augmentations: frame rotation (k in {0,1,2,3}) plus an independent horizontal and
vertical flip with the matching directional action remap
(`PuzzleScriptAdapter._FLIP_GAMES`), which re-samples the board's 8-element
symmetry group. The flips are exact symmetries here: the game is gravity-free,
input is screen-relative, and every rule is stated over all four directions at
once, so a mirrored board is a legal board of the same game. No sprite is
directional, so no mirror lands one object's art on another's. 23 levels x 8
presentations = 184.

Usage (run from the repo root):
    python solvers/generate_epicjamgame_training.py --episodes 200 \
        --out data/training_multi_level/epicjamgame

    python solvers/generate_epicjamgame_training.py --plans   # level report
    python solvers/generate_epicjamgame_training.py --audit   # rendering audit
    python solvers/generate_epicjamgame_training.py --fuzz    # model vs engine
    python solvers/generate_epicjamgame_training.py --verify  # replay on engine
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

GAME_NAME = "EpicJamGame"

#: Disk cache of the per-level start plan AND its optimal-action sets.
PLAN_CACHE: Path = _REPO_ROOT / "data" / "epicjamgame_plans.json"

#: Engine direction names, in the order `_Board` indexes them.
DIRS = ["up", "down", "left", "right"]
_DELTA = [(-1, 0), (1, 0), (0, -1), (0, 1)]

#: Give up on a level whose (pruned) reachable space is bigger than this. The
#: biggest shipped board settles at 62k and the biggest authored one at 151k; the
#: cap is a runaway guard for a board authored later, and hitting it drops the
#: level rather than the run.
STATE_CAP = 2_000_000

#: `[Blocking] -> []` and the four marking rules re-derive the sightline markers
#: every tick, so they never change the outcome of a turn -- but they DO sit in
#: the grid, which is why the model reads only the four real object classes.
_PIECES = ("player", "spirit", "crate")


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

class _Board:
    """The static half of a level (size, walls, targets, water) plus the ruleset.

    A state is ``(player, spirit, crates)`` with cells as ``(r, c)`` and
    ``crates`` a sorted tuple -- crates are interchangeable, so that is canonical.

    `step` reproduces one whole KEYPRESS, i.e. the first turn plus the ``again``
    continuation the late rule drives. One turn is: assign forces by applying the
    rule list in order, note whether the win rule fires (it reads the spirit's
    force, so it must be evaluated BEFORE anything moves), then resolve the
    forces the way `PSEngine._resolve_forces` does. It is verified against the
    real interpreter by ``--fuzz``.
    """

    def __init__(self, h, w, walls, targets, water):
        self.h, self.w = h, w
        self.walls = frozenset(walls)
        self.targets = frozenset(targets)
        self.water = frozenset(water)

    def inb(self, p):
        return 0 <= p[0] < self.h and 0 <= p[1] < self.w

    # -- dynamics ------------------------------------------------------------
    def _flee(self, player, spirit, crates):
        """The direction the spirit is given this turn, or None.

        ``[ Player | ... | Spirit | no wall no crate Background ]`` fires when the
        two are in line and the cell BEHIND the spirit is a free cell of the
        board; the four ``Blocking`` cancel rules then take the force away again
        when an Opaque (Wall or Crate) sits strictly between them. Water is
        neither -- it stops nothing, in either role."""
        pr, pc = player
        sr, sc = spirit
        if pr == sr and pc != sc:
            d = 3 if pc < sc else 2
            lo, hi = (pc, sc) if pc < sc else (sc, pc)
            between = [(pr, c) for c in range(lo + 1, hi)]
        elif pc == sc and pr != sr:
            d = 1 if pr < sr else 0
            lo, hi = (pr, sr) if pr < sr else (sr, pr)
            between = [(r, pc) for r in range(lo + 1, hi)]
        else:
            return None
        for b in between:
            if b in self.walls or b in crates:
                return None                      # sightline blocked: it stays put
        nb = (sr + _DELTA[d][0], sc + _DELTA[d][1])
        if not self.inb(nb) or nb in self.walls or nb in crates:
            return None                          # nowhere to run: it stays put
        return d

    def _turn(self, player, spirit, crates, di):
        """One rule pass + force resolution. ``di`` is None on an `again` tick,
        where PuzzleScript re-runs the rules with no input."""
        forces = {}
        if di is not None:
            dr, dc = _DELTA[di]
            ahead = (player[0] + dr, player[1] + dc)
            # [ > Player | Water ] -> [ Player | Water ] strips the player's
            # force BEFORE the push rule can read it, so a crate sitting on water
            # is a crate nobody will ever move again.
            if ahead not in self.water:
                forces[player] = di
                if ahead in crates:
                    forces[ahead] = di           # [ > Player | Crate ] -> push
        sd = self._flee(player, spirit, crates)
        if sd is not None:
            forces[spirit] = sd
        # [ stationary Spirit Target ] -> win, evaluated with the pre-movement
        # grid and the forces assigned so far: a spirit that is ON the target but
        # running does NOT win, which is why an unattended win has to be claimed
        # by a later press.
        won = spirit in self.targets and spirit not in forces
        player, spirit, crates = self._resolve(player, spirit, crates, forces)
        return player, spirit, crates, won

    def _resolve(self, player, spirit, crates, forces):
        """`PSEngine._resolve_forces`: trace each push chain, drop the chains
        whose endpoint is blocked, drop the chains that claim the same cell, move
        the rest, repeat until stable."""
        occ = {player: "P", spirit: "S"}
        for c in crates:
            occ[c] = "C"
        for _ in range(10):
            movable, blocked, deferred = [], set(), False
            for pos, d in list(forces.items()):
                if pos in blocked:
                    continue
                dr, dc = _DELTA[d]
                chain, cur, free, blocker_moves = [pos], pos, False, False
                while True:
                    nb = (cur[0] + dr, cur[1] + dc)
                    if not self.inb(nb) or nb in self.walls:
                        break                    # off the board / wall: blocked
                    if nb not in occ:
                        free = True
                        break
                    if forces.get(nb) == d:
                        chain.append(nb)         # same-direction chain member
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
                non_head.update(chain[1:])       # subsume shorter chains
            live = [(ch, d) for ch, d in movable if ch[0] not in non_head]

            claims, bad = {}, set()
            for i, (chain, d) in enumerate(live):
                for ent in chain:
                    tgt = (ent[0] + _DELTA[d][0], ent[1] + _DELTA[d][1])
                    other = claims.get(tgt)
                    if other is not None and other != i:
                        bad.add(i)
                        bad.add(other)           # two chains, one cell: both stop
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
                    occ[(ent[0] + dr, ent[1] + dc)] = occ.pop(ent)
                    forces.pop(ent, None)
                moved = True
            if not forces or (not moved and not deferred):
                break

        player = spirit = None
        crates = []
        for pos, kind in occ.items():
            if kind == "P":
                player = pos
            elif kind == "S":
                spirit = pos
            else:
                crates.append(pos)
        return player, spirit, tuple(sorted(crates))

    def step(self, state, di):
        """One keypress: ``((player, spirit, crates), won)``.

        The `late [Player | ... | Spirit] -> again` rule re-runs the turn while
        the two are in line, and the engine re-triggers `again` only when the
        grid actually CHANGED over the iteration -- so a spirit that is in line
        but jammed ends the press instead of spinning to the 50-iteration cap.
        The engine also breaks the loop the moment the win fires."""
        player, spirit, crates = state
        won = False
        prev = state
        for it in range(50):
            player, spirit, crates, w = self._turn(
                player, spirit, crates, di if it == 0 else None)
            won = won or w
            cur = (player, spirit, crates)
            if w or cur == prev:
                break
            if player[0] != spirit[0] and player[1] != spirit[1]:
                break                            # out of line: no `again`
            prev = cur
        return (player, spirit, crates), won

    def signature(self):
        return (self.h, self.w, self.walls, self.targets, self.water)

    # -- the exact distance-to-win field --------------------------------------
    def live_cells(self):
        """Spirit cells from which a target is still reachable, in a relaxation
        where crates do not exist and a run may stop wherever it likes.

        Both changes only ADD spirit moves, so a cell outside this set is
        genuinely unreachable-from and the state can be dropped un-expanded. That
        is what keeps the field affordable: it deletes the entire "spirit escaped
        into the water / onto the border ring" subtree, which is most of the
        reachable space on every board (see the module docstring)."""
        cells = [(r, c) for r in range(self.h) for c in range(self.w)
                 if (r, c) not in self.walls]
        pred = {c: set() for c in cells}
        for x in cells:
            for dr, dc in _DELTA:
                # The player has to stand somewhere BEHIND the spirit, on land,
                # with no wall in the way. Walls never move, so keeping that test
                # exact costs nothing; crates might, so they are ignored.
                behind = (x[0] - dr, x[1] - dc)
                ok = False
                while self.inb(behind) and behind not in self.walls:
                    if behind not in self.water:
                        ok = True
                        break
                    behind = (behind[0] - dr, behind[1] - dc)
                if not ok:
                    continue
                y = (x[0] + dr, x[1] + dc)
                while self.inb(y) and y not in self.walls:
                    pred[y].add(x)
                    y = (y[0] + dr, y[1] + dc)
        live = {t for t in self.targets if t not in self.walls}
        queue = deque(live)
        while queue:
            y = queue.popleft()
            for x in pred[y]:
                if x not in live:
                    live.add(x)
                    queue.append(x)
        return live

    def field(self, start, cap=STATE_CAP):
        """``{state: presses to a win}`` over everything reachable from
        ``start``, or None if the space is bigger than ``cap``.

        Forward BFS collects the edges (a winning press is an edge to a virtual
        sink, and dead-spirit successors are dropped), then one reverse BFS from
        the wins labels every state that can still reach one. States that cannot
        are simply absent."""
        live = self.live_cells()
        ids = {start: 0}
        states = [start]
        succ = []
        queue = deque([0])
        while queue:
            i = queue.popleft()
            s = states[i]
            row = []
            for di in range(4):
                nxt, won = self.step(s, di)
                if won:
                    row.append(-1)               # the sink
                elif nxt[1] not in live:
                    row.append(-2)               # dead spirit: never expanded
                else:
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
        dist = [-1] * len(states)
        queue = deque()
        for i, row in enumerate(succ):
            for j in row:
                if j == -1:
                    if dist[i] < 0:
                        dist[i] = 1
                        queue.append(i)
                elif j >= 0 and j != i:
                    pred[j].append(i)
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
    walls, targets, water, crates = set(), set(), set(), []
    player = spirit = None
    for r, row in enumerate(eng.grid):
        for c, cell in enumerate(row):
            for o in cell:
                name = inv[o]
                if name == "wall":
                    walls.add((r, c))
                elif name == "target":
                    targets.add((r, c))
                elif name == "water":
                    water.add((r, c))
                elif name == "crate":
                    crates.append((r, c))
                elif name == "player":
                    player = (r, c)
                elif name == "spirit":
                    spirit = (r, c)
    board = _Board(eng.height, eng.width, walls, targets, water)
    return board, (player, spirit, tuple(sorted(crates)))


# ---------------------------------------------------------------------------
# The expert
# ---------------------------------------------------------------------------

class EpicJamGameExpert(PSExpert):
    """Exact shortest plans (and exact tie sets) read off a BFS field over the
    native model. See the module docstring for why the model exists and how it is
    verified.

    Only ONE level's field is held at a time: the biggest runs to 151k states and
    every caller here plans one level and moves on, so keeping them all would cost
    a gigabyte to answer questions nobody asks twice. The disk cache
    (`PSExpert.plan_cache_path`) is what makes that free across runs."""

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
        """Walk the field downhill from the engine's current state, recording the
        press taken and every press that would have been equally short."""
        board, state = read_board(eng, self.g)
        if state[0] is None or state[1] is None:
            return None                              # no player / no spirit
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
        while True:
            here = dist[state]
            best, nxts = [], {}
            for di, name in enumerate(DIRS):
                nxt, won = board.step(state, di)
                nxts[name] = (nxt, won)
                if (0 if won else dist.get(nxt, 1 << 30)) == here - 1:
                    best.append(name)
            presses.append(best[0])
            optsets.append(best)
            state, won = nxts[best[0]]
            if won:
                break
        return Plan(presses, optsets)


class EpicJamGameSolver(PSAStarSolver):
    game_id = "puzzlescript_epicjamgame"
    game_name = GAME_NAME
    expert_cls = EpicJamGameExpert

    #: Record against the game folder's adapter -- the one `game_envs` hands a
    #: live agent. It is a plain passthrough today; going through it means a later
    #: step limit or sprite fix there cannot silently diverge from what is taped
    #: here.
    game_module_id = "ps:epicjamgame"

    #: The longest plan is 73 presses; the rest is room for the exploration prefix
    #: and its RESET. Stays inside the adapter's own 200-step per-level budget.
    max_steps = 150


# ---------------------------------------------------------------------------
# Reports and checks
# ---------------------------------------------------------------------------

def _report() -> int:
    """Per-level board size, crate count, exact optimum and tie coverage."""
    solver = EpicJamGameSolver()
    game = solver.make_game(0)
    expert = EpicJamGameExpert(game)
    total = 0
    for level in range(game.n_levels):
        game.set_level(level)
        eng = game._engine
        _board, state = read_board(eng, game._game)
        found = expert.plan(eng, level)
        if found is None:
            print(f"level {level:2d}: NO PLAN")
            continue
        sets = getattr(found, "optsets", []) or []
        ties = sum(1 for s in sets if len(s) > 1)
        total += len(found)
        room = "ok" if len(found) < game._max_steps else "OVER BUDGET"
        print(f"level {level:2d}: {eng.height:2d}x{eng.width:2d} "
              f"{len(state[2])} crates   {len(found):3d} presses ({room}), "
              f"{ties:3d} steps with a tie set "
              f"({ties / max(1, len(found)):.0%})")
    print(f"total {total} presses over {game.n_levels} levels")
    return 0


def _verify() -> int:
    """Replay every level's plan on the REAL interpreter and require a WIN.

    The plans come off a model; this is the line that says the model and the
    interpreter agree about the boards that actually ship."""
    solver = EpicJamGameSolver()
    game = solver.make_game(0)
    expert = EpicJamGameExpert(game)
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
        ok = False
        for d in plan:
            eng.step(d)
            ok = eng.check_win()                 # the win is a TRANSITION here:
        bad += not ok                            # _rule_win resets every step
        print(f"level {level:2d}: {len(plan):3d} presses -> "
              f"{'WIN' if ok else 'NOT WON'}")
    print("verify clean" if not bad else f"VERIFY FAILED on {bad} levels")
    return 0 if not bad else 1


def _audit() -> int:
    """Assert every cell COMPOSITION is distinct at every cell size in use.

    Composition, not object: the spirit ON the target is the winning frame and
    has to differ from the spirit anywhere else, and a piece shoved onto water is
    a permanent state change that has to be visible. See the palette note in the
    module docstring."""
    game = EpicJamGameSolver().make_game(0)
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    comps = {
        "floor": (), "wall": ("wall",), "water": ("water",),
        "target": ("target",),
        "player": ("player",), "player_on_target": ("target", "player"),
        "spirit": ("spirit",), "spirit_on_target": ("target", "spirit"),
        "spirit_on_water": ("water", "spirit"),
        "crate": ("crate",), "crate_on_target": ("target", "crate"),
        "crate_on_water": ("water", "crate"),
    }

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


def _fuzz(n_boards: int = 900, n_steps: int = 25, seed: int = 0) -> int:
    """Differential fuzz: random boards played move-for-move against the real
    interpreter, piece positions AND the win flag compared after every step.

    This is the only thing standing between `_Board` and a silently wrong corpus,
    so it covers what the shipped levels do not: water in the middle of the
    board, several crates, crates already on water, and spirits that start on the
    target (the "win it by standing still" case the shipped boards only reach at
    the very end)."""
    game = EpicJamGameSolver().make_game(0)
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    rng = random.Random(seed)
    mismatches = 0
    steps = 0
    for _ in range(n_boards):
        h, w = rng.choice([6, 7, 8, 9]), rng.choice([6, 7, 8, 9])
        cells = [(r, c) for r in range(h) for c in range(w)]
        rng.shuffle(cells)
        nwall = rng.randint(0, (h * w) // 5)
        walls = set(cells[:nwall])
        rest = cells[nwall:]
        nwater = rng.randint(0, (h * w) // 3)
        water = set(rest[:nwater])
        rest = rest[nwater:]
        if len(rest) < 4:
            continue
        ncrate = rng.randint(0, min(4, len(rest) - 3))
        crates = set(rest[:ncrate])
        rest = rest[ncrate:]
        player, spirit = rest[0], rest[1]
        targets = set(rest[2:2 + rng.randint(0, 2)])
        if rng.random() < 0.25:
            targets.add(spirit)                  # start ON the target

        board = _Board(h, w, walls, targets, water)
        state = (player, spirit, tuple(sorted(crates)))
        grid = []
        for r in range(h):
            row = []
            for c in range(w):
                cell = {idx["background"]}
                if (r, c) in water:
                    cell.add(idx["water"])
                if (r, c) in targets:
                    cell.add(idx["target"])
                if (r, c) in walls:
                    cell = {idx["background"], idx["wall"]}
                elif (r, c) in crates:
                    cell.add(idx["crate"])
                elif (r, c) == player:
                    cell.add(idx["player"])
                elif (r, c) == spirit:
                    cell.add(idx["spirit"])
                row.append(cell)
            grid.append(row)
        eng.grid, eng.height, eng.width = grid, h, w
        eng._position_index_dirty = True
        eng._rule_noop_cache.clear()
        eng._rule_win = False

        for _s in range(n_steps):
            di = rng.randrange(4)
            eng.step(DIRS[di])
            truth_won = eng.check_win()
            _b, truth = read_board(eng, g)
            state, won = board.step(state, di)
            steps += 1
            if state != truth or won != truth_won:
                mismatches += 1
                print(f"MISMATCH after {DIRS[di]}\n"
                      f"  board {h}x{w} walls {sorted(walls)} "
                      f"water {sorted(water)} targets {sorted(targets)}\n"
                      f"  model  {state} won={won}\n"
                      f"  engine {truth} won={truth_won}")
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
    sys.exit(EpicJamGameSolver.main())
