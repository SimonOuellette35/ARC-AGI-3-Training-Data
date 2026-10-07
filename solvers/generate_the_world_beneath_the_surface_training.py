"""ps:the_world_beneath_the_surface -- a match-3 you play with a two-square
CURSOR and a hard budget of swaps, where the swaps are the only thing that
costs anything and walking is free.

Sixteen 6x12 levels. You steer a rigid pair of cursors, X swaps the two blocks
they cover, gravity packs everything down, three or more of a colour in a line
explode, and the level is won when there is nothing left. The counter under the
board says how many SWAPS the level allows; when it reaches zero the level
restarts. That is the whole game, and almost none of what follows is visible in
it.

THE RULES THAT MATTER

    [Turn0] -> restart                                                   (1)
    [Explosion Block] -> []                                              (2)
    [> Player1|Wall][Player2] -> [Stationary Player1|Wall][Stationary P2] (3)
    [Action Player1 Block][Corner1] -> [Player1][Corner1 Block]           (4)
    [Wall Block][Turn1] -> [Wall Block][Turn0]   (and Turn2..Turn9)       (5)
    [Corner1 Block][Player2 No Block] -> [Corner1][Player2 Block]         (6)
    Down [Block|No Block No Wall] -> [|Block] again                       (7)
    [X|X|X] -> [X Explosion|X Explosion|X Explosion]  (and |4, |5, x7)    (8)
    late [BombBlock Explosion|Block] -> [... |Block Explosion] again      (9)

plus the mirrored (4) and (6) for Player2/Corner2, and `All Block on Explosion`.

SEVEN CONSEQUENCES, none of them visible in those rules as written

* **Walking is FREE; only a swap costs a turn.** Rule 5 needs a Block standing
  on a Wall, and the only way that ever happens is rules 4 and 6 parking the
  swapped blocks in the frame's top corners for the length of one tick. No rule
  reads a direction key at all. So a level is "win in at most B swaps", the
  press count is walk + swaps, and the search is over WHERE to swap, not how
  often.

* **A swap where only ONE cursor holds a block MOVES that block sideways**, and
  charges a full turn for it. Rule 4 parks whatever Player1 covers in Corner1,
  rule 6 hands it to Player2 -- and rule 6's guard is `Player2 No Block`, which
  an empty square satisfies. Swapping two blocks of the SAME colour also costs
  a turn and changes nothing at all: the countdown does not care what came back.

* **A block falls all the way in ONE tick, so nothing ever matches in mid-air.**
  Rule 7 is driven to its fixpoint inside a single rule pass, not one square per
  `again` continuation. It is the difference between a block dropping past a
  pair of its own colour harmlessly and detonating as it goes by: the one-square
  model is wrong on 320 of 4000 random boards (`--selfcheck`).

* **The WHOLE maximal run explodes**, not three of it -- the 5-, 4- and 3-wide
  rule families between them cover every window of a longer line, and rule 8
  only ADDS an Explosion, so no axis or scan order can change the answer. The
  three-only model is wrong on 110 of 4000.

* **An explosion is not a deletion. It is a mark that survives until the next
  press**, whichever press that is. Rule 2 runs at the top of the NEXT tick, so
  a swap that matches leaves the dead blocks sitting on the board -- still
  holding up everything above them -- and the following press, even a walk into
  a wall, is what deletes them and lets the cascade run. That also makes the
  mark a pure FUNCTION of the settled board (`doomed`), which is why a state
  here is just `(cells, cursor, turns)`.

* **Rule 1 is FIRST, so it fires inside the cascade too.** Spend the last turn
  on a swap and the `again` continuation that would finish the cascade starts by
  restarting the level. The last swap of a level must therefore win on the very
  tick it is made -- one gravity settle, one match, one bomb pass, done. A swap
  that needs a second tick to clear the board is a restart, not a win. The
  no-restart model is wrong on 37 of 4000.

* **The bomb chain is four independent sweeps, not a blast radius.** Rule 9 has
  no direction, so the interpreter expands it to four and drives each ONE to its
  own fixpoint, in the order up, down, left, right, never going back -- and then
  the next tick's rule 2 deletes every exploding block before the chain could
  continue. A chain that has to turn a corner the wrong way stops. Modelling it
  as the 4-connected closure it looks like over-kills the board on 23 of 4000.
  The same rule is also what keeps a cascade alive: `again` is armed by the rule
  MATCHING, not by it marking anything new, and three bombs in a row match it
  against each other.

THE SEARCH: the exact distance FIELD, on every level

A state is `(cells, cursor, turns)` -- one byte per square, the index of the
left-hand cursor, and the countdown -- which is exact and complete, because the
walls are static, the scoreboard is decoration, and the Explosions are a
function of `cells`. `SurfaceBoard.field` enumerates the whole reachable
component layer by layer and induces backwards over the layers, which buys a
plan that is provably SHORTEST with no heuristic or node cap to get wrong, and
the EXACT optimal press set at every step of it. Fifteen levels cost between 41
and 102k states and answer in under 20 seconds. Level 14 is the outlier -- its
space holds 4,766,927 states within 21 presses and its answer is 22, so the
field enumerates 5,046,613 of them at ~15 minutes and ~1.4 GB, and proves the
same bound from the inside that a separate flat sweep to depth 21 proves from
the outside.
`PSExpert.plan_cache_path` keeps every answer in
`data/the_world_beneath_the_surface_plans.json`, so that is paid once for the
life of the cache. A beam plus `certify_tail` is kept as the fallback a level
would take if a future edit put it out of the field's reach; at the shipped
`field_cap` nothing uses it (the beam does find level 14's 22, in 7 seconds,
which is how the field's cap was chosen).

LEVEL 10 IS PROVABLY UNWINNABLE HERE, and the proof says exactly why. Its
entire reachable component is 164,460 states and not one of them wins; it is not
a budget problem either -- hand it a seventh turn and the 521,101 states of that
component do not win either. It is the one level with a colour that cannot be
matched at all (two green blocks, and a match needs three), so every plan for it
has to end with a bomb chain that reaches both of them, and that is precisely
what the four one-way sweeps will not do. Swap the chain for the 4-connected
closure it looks like and the level solves in 33 presses -- so the level as
DESIGNED is fine, and what it runs into is the interpreter's late-rule
semantics, which are shared by every ps: game in this tree and are not something
to change for one level. `discover_solvable` drops it; the other fifteen are
recorded, and `--selfcheck` measures the sweep-versus-closure question on random
boards rather than taking either on trust.

WHAT THE FRAME DOES NOT SHOW: the TURNS counter. The scoreboard under the wall
frame is four rows of `Board` and `Turn` objects and nothing else, so
`_render_frame`'s generic HUD crop takes all four off every frame -- which is
also why a cell is 4 pixels rather than 3, and why the art below could be fixed
at all. The budget is left cropped on purpose: it is a CONSTANT of each level
(the sixteen boards are fixed and each ships one digit), the swaps already spent
are visible in the board itself, and nothing in a recorded trajectory ever
depends on reading the digit -- against a permanent 30% loss of resolution in
every cell of every frame for the fix.

ART. Sixteen colour lines and eight sprites in
`data/puzzlescript_games/The_World_Beneath_the_Surface.txt`; that file's own
header has the detail and `--audit` is the regression test. Three things were
unplayable as shipped. A yellow block, a green block, the wall frame and a
SingleWall pillar were all ARC 14 -- one flat colour, four different things.
Red, blue and magenta were all 13. And the Explosion drew a single pixel of the
BACKGROUND colour, i.e. it was invisible -- so the win condition could not be
seen, and neither could the game's whole short-term memory (the previous
bullet: an exploded block is not deleted until the next press). The seven block
classes now take seven distinct palette entries, the walls take an eighth, the
block sprites had their black outline moved off every sampled pixel, and the
Explosion fills the cell bar one notch. The geometry that decides all of this
is that `_render_frame` composites the PLAYER last, above every collision
layer, and at four pixels the cursor sprite is a solid RING -- so on a square
the cursor stands on, only the inner 2x2 is visible at all, and which block,
that a cursor is there and whether it is exploding all have to be legible in
those four pixels. That is what the redraw arranges and what `--audit`
asserts, at the post-HUD-crop geometry the game really renders at (an earlier
version of that test measured the uncropped 18x14 and passed a sprite set that
was wrong for the size in use).

RECOVERY is the family's RESET prefix (`recovery_mode = "reset"`). A cleared
block does not come back and a spent turn is spent, so a perturbed board cannot
be re-planned from in general; one RESET restores exactly the state the cached
plan was solved from.

VERIFIED. Fifteen levels, 311 presses, every one provably shortest and exactly
labelled; 88 of those steps have a second right answer. `--plans` replays all
fifteen through the interpreter and every one reports `win=True`. `--selfcheck`
fuzzes 19,248 rollout presses over the sixteen level starts in three play styles
-- uniform, board-changing, and a cursor that hunts for blocks and swaps (684
swaps, 176 blocks cleared, 148 run-out-of-turns restarts) -- plus 4,000 random
boards, with 0 divergences from the interpreter in either mode, 0 disagreements
about the win predicate, and the four wrong-model counts quoted above.
`--audit`: 32 cell compositions, all pairwise distinct at the 4px cell the game
renders at, the two cursor halves excepted on purpose. `--symmetry`: 16 levels x
12 seeds x 90 presses, all four rotations drawn, every frame exactly the
rotation of the unaugmented one. Recorded in-process: 6 seeds x 15 levels =
90/90 WIN, all 2027 frames replay exactly through the adapter from the recorded
SCREEN actions, 1872/1872 expert steps labelled (1.28 optimal presses per step,
the taken press always in its own set), every index in 0..6, and the longest
level record is 35 actions against the adapter's 200-press cap. Three processes
at different ``PYTHONHASHSEED``s write byte-identical episodes. `--ties`
re-derives 303 of the 311 labels with a depth-limited search that shares nothing
with the field but the model: 0 disagreements in either direction, and the eight
it does not reach are level 14's first eight, where the probe budget runs out
and the label falls back to the plan's own press. `--bfs` re-derives eleven
levels' plan lengths by iterative deepening, 0 mismatches -- level 10 included,
where it independently reports no win at any depth.

CLI
---
    --selfcheck   the native model fuzzed against the interpreter, and the
                  four decisions above measured against variants of themselves
    --audit       every cell composition renders distinctly, at the cell size
                  the game actually uses
    --symmetry    every presentation is an exact rotation of the unaugmented
                  one, driven through the adapter with the remapped press
    --plans       every level's plan, replayed through the interpreter
    --bfs         every plan length re-derived by iterative deepening, which
                  shares nothing with the field but the model
    --ties        every optimal-action label re-derived exhaustively, and
                  disagreements counted in both directions
"""

from __future__ import annotations

import itertools
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.puzzlescript_adapter import (        # noqa: E402
    PuzzleScriptAdapter, _render_frame)
from arcengine import ActionInput                   # noqa: E402
from solvers.common.ps_astar import (              # noqa: E402
    PSAStarSolver, PSExpert, Plan, screen_action)

GAME_NAME = "The_World_Beneath_the_Surface"
GAME_ID = "ps:the_world_beneath_the_surface"

#: Engine presses, in the order the model indexes them. ACTION is the swap key
#: -- and, when neither cursor holds a block, a pure WAIT press that still
#: drains a pending explosion, exactly like a move into a wall.
DIRNAMES = ("up", "down", "left", "right", "action")

#: The seven block classes, indexed 1..7 in the model's cells (0 is empty).
BLOCKS = ("redblock", "blueblock", "yellowblock", "greenblock",
          "cyanblock", "magentablock", "bombblock")
BOMB = 7                                   # index of bombblock in `BLOCKS`

#: A run this long or longer marks EVERY cell of the maximal run.
MATCH = 3

#: Wall classes: the level frame plus the free-standing SingleWall pillars.
WALLS = ("horizontalwall", "verticalwall", "corner1", "corner2", "corner3",
         "corner4", "singlewall")

_INF = 1 << 30


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

class SurfaceBoard:
    """The whole game as ``(cells, cursor, turns)``.

    ``cells`` is one byte per interior square, 0 for empty and 1..7 for a block
    class; ``cursor`` is the index of Player1 (Player2 is always the square to
    its right); ``turns`` is the countdown the level shipped with. Nothing else
    in the game moves -- the walls are static, the board digits are decoration,
    and an Explosion is a pure FUNCTION of ``cells`` (see `doomed`), so it
    never has to be carried.
    """

    def __init__(self, eng, ids):
        self.h, self.w, self.top, self.left = _interior(eng, ids)
        self.n = self.h * self.w
        self.wall = [False] * self.n
        cells = [0] * self.n
        wall_ids = {ids[nm] for nm in WALLS if nm in ids}
        cursor = 0
        for r in range(self.h):
            for c in range(self.w):
                cell = eng.grid[self.top + r][self.left + c]
                i = r * self.w + c
                if cell & wall_ids:
                    self.wall[i] = True
                for k, nm in enumerate(BLOCKS):
                    if ids[nm] in cell:
                        cells[i] = k + 1
                if ids["player1"] in cell:
                    cursor = i
        self.turns = _turns_left(eng, ids)
        self.state = (bytes(cells), cursor, self.turns)

        # -- static tables ---------------------------------------------------
        # Cursor moves: the PAIR is rigid, and either half meeting a Wall stops
        # both ("[> Player1|Wall][Player2] -> stationary"), so a move is legal
        # only when BOTH destination squares are floor.
        self.nb = [[-1] * 4 for _ in range(self.n)]
        for r in range(self.h):
            for c in range(self.w - 1):
                i = r * self.w + c
                if self.wall[i] or self.wall[i + 1]:
                    continue
                for k, (dr, dc) in enumerate(((-1, 0), (1, 0), (0, -1), (0, 1))):
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < self.h and 0 <= nc < self.w - 1):
                        continue
                    j = nr * self.w + nc
                    if self.wall[j] or self.wall[j + 1]:
                        continue
                    self.nb[i][k] = j

        # Gravity acts inside each maximal wall-free vertical SEGMENT of a
        # column: a block rests on a wall exactly as it rests on a block.
        self.columns: list[list[int]] = []
        for c in range(self.w):
            seg: list[int] = []
            for r in range(self.h):
                i = r * self.w + c
                if self.wall[i]:
                    if len(seg) > 1:
                        self.columns.append(seg)
                    seg = []
                else:
                    seg.append(i)
            if len(seg) > 1:
                self.columns.append(seg)

        # Every maximal straight LINE of floor squares, in both axes. A run of
        # three inside one of these is a match; a wall breaks a line in two,
        # and so does the board edge.
        self.lines: list[list[int]] = []
        for c in range(self.w):
            self.lines.extend(self._runs([r * self.w + c for r in range(self.h)]))
        for r in range(self.h):
            self.lines.extend(self._runs([r * self.w + c for c in range(self.w)]))

        # The bomb chain is FOUR SEPARATE directional sweeps, in the order the
        # interpreter expands an undirected rule -- up, down, left, right -- so
        # they are kept apart here. `dirnb[k][i]` is the neighbour of `i` in
        # direction `k`, or -1 off the board.
        self.dirnb = [[-1] * self.n for _ in range(4)]
        for r in range(self.h):
            for c in range(self.w):
                i = r * self.w + c
                for k, (dr, dc) in enumerate(((-1, 0), (1, 0), (0, -1), (0, 1))):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.h and 0 <= nc < self.w:
                        self.dirnb[k][i] = nr * self.w + nc

        self.dist = self._walk_field()

    def _runs(self, line: list[int]) -> list[list[int]]:
        out, run = [], []
        for i in line + [None]:
            if i is not None and not self.wall[i]:
                run.append(i)
                continue
            if len(run) >= MATCH:
                out.append(run)
            run = []
        return out

    def _walk_field(self) -> list[list[int]]:
        """All-pairs shortest cursor walks. The cursor never interacts with a
        block (Player and Wall share a collision layer; Block does not), so
        these distances are static for the whole level and every walk between
        two swaps costs exactly ``dist[a][b]`` presses."""
        out = []
        for src in range(self.n):
            d = [_INF] * self.n
            if (src % self.w >= self.w - 1 or self.wall[src]
                    or self.wall[src + 1]):
                out.append(d)          # not a square the cursor pair can hold
                continue
            d[src] = 0
            frontier = [src]
            while frontier:
                nxt = []
                for i in frontier:
                    for j in self.nb[i]:
                        if j >= 0 and d[j] == _INF:
                            d[j] = d[i] + 1
                            nxt.append(j)
                frontier = nxt
            out.append(d)
        return out

    # -- the turn -----------------------------------------------------------
    def matched(self, cells) -> int:
        """The bitmask the seven ``[X|X|X]`` rule families mark.

        The rules only ADD an Explosion, so which axis the interpreter drives
        to a fixpoint first cannot matter, and the 5-, 4- and 3-wide variants
        together cover every window of a longer run -- so this is exactly "every
        cell of every maximal same-colour run of three or more"."""
        mask = 0
        for line in self.lines:
            run, prev = 0, 0
            for a, i in enumerate(line):
                v = cells[i]
                if v and v == prev:
                    run += 1
                    continue
                if run >= MATCH:
                    for j in line[a - run:a]:
                        mask |= 1 << j
                run = 1 if v else 0
                prev = v
            if run >= MATCH:
                for j in line[len(line) - run:]:
                    mask |= 1 << j
        return mask

    def spread(self, cells, mask: int) -> tuple[int, bool]:
        """``late [BombBlock Explosion|Block] -> [... |Block Explosion] again``.

        Returns ``(mask, matched)``. NOT a 4-connected closure: the interpreter
        drives each DIRECTION of an undirected rule to its own fixpoint, in the
        order up, down, left, right, and never goes back -- so a chain that has
        to turn left and then go up gets only the left half, and the next tick
        deletes every exploding block before it could finish. Modelling it as a
        closure over-kills the board on 23 of 4000 random bomb fields.

        ``matched`` is whether the LHS matched at ALL, which is what arms the
        rule's ``again`` -- and it is true even when nothing new is marked (an
        exploding bomb wedged against a block that is already exploding still
        matches). Three bombs in a row are exactly that case, and getting it
        wrong leaves the whole cascade parked for a press.
        """
        matched = False
        for k in range(4):
            nb = self.dirnb[k]
            changed = True
            while changed:
                changed = False
                m = mask
                while m:
                    low = m & -m
                    i = low.bit_length() - 1
                    m ^= low
                    if cells[i] != BOMB:
                        continue
                    j = nb[i]
                    if j >= 0 and cells[j]:
                        matched = True
                        if not (mask >> j) & 1:
                            mask |= 1 << j
                            changed = True
        return mask, matched

    def doomed(self, cells) -> int:
        """The Explosions standing on the board in state ``cells``.

        A settled board carries its marks until the NEXT press deletes them
        (`step`'s first act), and they are never stored: a returned state is
        always gravity-settled, and the marks on a settled board are exactly
        what the match + bomb rules would write on it."""
        return self.spread(cells, self.matched(cells))[0]

    def gravity(self, cells: list) -> bool:
        """``Down [Block | No Block No Wall] -> [ | Block] again``, driven to
        its FIXPOINT inside one rule pass -- which is what the interpreter
        does, and is not a detail: a one-cell-per-tick fall would let a block
        match in mid-air on the way past (measured, see ``--selfcheck``)."""
        moved = False
        for seg in self.columns:
            k = len(seg) - 1
            for r in range(len(seg) - 1, -1, -1):
                v = cells[seg[r]]
                if not v:
                    continue
                if r != k:
                    cells[seg[k]] = v
                    cells[seg[r]] = 0
                    moved = True
                k -= 1
        return moved

    def won(self, st) -> bool:
        """``All Block on Explosion`` -- vacuously true with no blocks left."""
        cells = st[0]
        mask = self.doomed(cells)
        return all(not v or (mask >> i) & 1 for i, v in enumerate(cells))

    def step(self, st, k: int):
        """One press. Returns the next state, or None for a state the level
        cannot come back from.

        None is the ``[Turn0] -> restart`` rule, which is the FIRST rule in the
        file and therefore fires at the top of every tick -- including the
        ``again`` continuations inside the very press that spent the last turn.
        So a level is lost two ways: pressing anything at all with the counter
        already on zero, and spending the last turn on a swap whose cascade
        needs a second tick to finish. Only a swap that wins on the tick it is
        made can be the last one.
        """
        cells_b, cur, turns = st
        if turns == 0:
            return None                         # [Turn0] -> restart
        cells = list(cells_b)
        for i in _bits(self.doomed(cells)):     # [Explosion Block] -> []
            cells[i] = 0
        if k == 4:
            a, b = cur, cur + 1
            if cells[a] or cells[b]:
                cells[a], cells[b] = cells[b], cells[a]
                turns -= 1                      # [Wall Block][TurnN] -> [TurnN-1]
        else:
            j = self.nb[cur][k]
            if j >= 0:
                cur = j
        while True:
            fell = self.gravity(cells)
            mask, matched = self.spread(cells, self.matched(cells))
            if all(not v or (mask >> i) & 1 for i, v in enumerate(cells)):
                return (bytes(cells), cur, turns)          # WIN, mid-tick
            # `again` fires when a rule carrying it MATCHED and the grid moved
            # at all. Gravity firing is both at once; the bomb rule matching
            # implies an Explosion was written this tick, so it is both too.
            if not (fell or matched):
                return (bytes(cells), cur, turns)          # settled, marks kept
            if turns == 0:
                return None                                # restart next tick
            for i in _bits(mask):
                cells[i] = 0

    # -- search: the exact distance field ------------------------------------
    def field(self, start, cap: int = 4_000_000, verbose=False):
        """Layered breadth-first enumeration from ``start``, then backward
        induction over the layers.

        Returns ``(plan, optsets, stats)`` with a PROVABLY SHORTEST plan and the
        EXACT optimal press set at every step, ``(None, None, stats)`` when the
        whole reachable component was enumerated and none of it wins, or
        ``(False, None, stats)`` when ``cap`` states were reached first.

        ``good[d]`` is the set of depth-``d`` states that still win in
        ``d* - d`` presses, so a press is optimal at depth ``d`` iff it lands in
        ``good[d + 1]``; nothing optimal can hide in an earlier layer because a
        successor's depths from both ends must sum to at least ``d*``.
        """
        t0 = time.time()
        layers = [{start}]
        seen = {start}
        win_depth = None
        while layers[-1] and win_depth is None:
            nxt = set()
            for s in layers[-1]:
                for k in range(5):
                    ns = self.step(s, k)
                    if ns is None or ns in seen:
                        continue
                    if self.won(ns):
                        win_depth = len(layers)
                        break
                    seen.add(ns)
                    nxt.add(ns)
                if win_depth is not None:
                    break
            if win_depth is not None:
                break
            if len(seen) > cap:
                return False, None, {"states": len(seen), "secs": time.time() - t0}
            layers.append(nxt)
            if verbose:
                print(f"    depth {len(layers) - 1}: {len(nxt)} new, "
                      f"{len(seen)} total, {time.time() - t0:.1f}s", flush=True)
        stats = {"states": len(seen), "secs": time.time() - t0,
                 "depth": win_depth}
        if win_depth is None:
            return None, None, stats

        def wins(s, k):
            ns = self.step(s, k)
            return ns is not None and self.won(ns)

        good = {s for s in layers[win_depth - 1]
                if any(wins(s, k) for k in range(5))}
        goods = [good]
        for d in range(win_depth - 2, -1, -1):
            good = {s for s in layers[d]
                    if any(self.step(s, k) in good for k in range(5))}
            goods.append(good)
        goods.reverse()

        presses, optsets, s = [], [], start
        for d in range(win_depth):
            if d == win_depth - 1:
                best = [k for k in range(5) if wins(s, k)]
            else:
                best = [k for k in range(5) if self.step(s, k) in goods[d + 1]]
            presses.append(best[0])
            optsets.append(best)
            s = self.step(s, best[0])
        return presses, optsets, stats

    # -- search: bounded, for the levels the field cannot hold ---------------
    def bounded(self, start, limit: int, cap: int = 400_000):
        """Shortest win from ``start`` in at most ``limit`` presses, or None.

        A None refusal is only meaningful when ``stats["capped"]`` is False;
        every caller checks it."""
        stats = {"states": 1, "capped": False}
        if self.won(start):
            return [], stats
        if limit <= 0:
            return None, stats
        seen = {start}
        frontier = [(start, ())]
        for _ in range(limit):
            nxt = []
            for s, path in frontier:
                for k in range(5):
                    ns = self.step(s, k)
                    if ns is None or ns in seen:
                        continue
                    if self.won(ns):
                        stats["states"] = len(seen)
                        return list(path) + [k], stats
                    seen.add(ns)
                    nxt.append((ns, path + (k,)))
            if len(seen) > cap:
                stats.update(states=len(seen), capped=True)
                return None, stats
            frontier = nxt
            if not frontier:
                stats["exhausted"] = True
                break
        stats["states"] = len(seen)
        return None, stats

    # -- search: beam, for the levels whose space is out of reach ------------
    def score(self, st) -> int:
        """Beam ordering. Blocks left dominates (that is the win condition),
        then how far every surviving block is from a same-coloured partner --
        a match is "two partners lined up next to me" -- then the walk to the
        nearest block, so the cursor does not drift away from the work."""
        cells, cur, turns = st
        live = [i for i, v in enumerate(cells) if v]
        if not live:
            return -1
        rc = [(i // self.w, i % self.w) for i in live]
        total = 0
        for a, ia in zip(rc, live):
            ds = sorted(abs(a[0] - b[0]) + abs(a[1] - b[1])
                        for b, ib in zip(rc, live)
                        if ib != ia and cells[ib] == cells[ia])
            total += sum(ds[:2]) + (self.h + self.w) * (2 - len(ds[:2]))
        total += min(self.dist[cur][i] if self.dist[cur][i] < _INF else self.h
                     for i in live)
        return 8 * self.h * self.w * len(live) + total

    def beam(self, start, width: int, max_depth: int = 220):
        """Best-first beam over the same dynamics, deduplicated globally.
        Deterministic: candidates sort by ``(score, state)`` and a state is a
        tuple of plain values, so the tie-break does not depend on
        ``PYTHONHASHSEED``."""
        if self.won(start):
            return []
        seen = {start}
        frontier = [(start, ())]
        for _ in range(max_depth):
            cand = []
            for s, path in frontier:
                for k in range(5):
                    ns = self.step(s, k)
                    if ns is None or ns in seen:
                        continue
                    if self.won(ns):
                        return list(path) + [k]
                    seen.add(ns)
                    cand.append((self.score(ns), ns, path + (k,)))
            if not cand:
                return None
            cand.sort(key=lambda x: (x[0], x[1]))
            frontier = [(s, path) for _s, s, path in cand[:width]]
        return None

    def certify_tail(self, start, plan, cap: int = 400_000, verbose=False):
        """Prove (and where possible shorten) a beamed plan from its END
        backwards, and label every step the probe budget reaches.

        At each state the plan passes through, an exhaustive bounded BFS asks
        two exact questions: is there a win strictly shorter than the suffix
        spent here (splice it in and restart), and which presses still finish
        in the presses that are left (the step's optimal SET, exact both ways).
        Backwards, because the bound -- and the cost -- shrinks towards the end.
        The first probe to hit ``cap`` ends the certification; that step and
        every earlier one carry the plan's own press and nothing else, which
        can only omit a tie, never invent one.

        Returns ``(plan, optsets, certified)``; the tail of length ``certified``
        is proved shortest and exactly labelled.
        """
        while True:
            states = [start]
            for k in plan:
                states.append(self.step(states[-1], k))
            sets: list = [None] * len(plan)
            certified = 0
            spliced = False
            for i in range(len(plan) - 1, -1, -1):
                remaining = len(plan) - i - 1
                shorter, stats = self.bounded(states[i], remaining, cap)
                if stats["capped"]:
                    break
                if shorter is not None:
                    if verbose:
                        print(f"    splice at {i}: {len(plan) - i} -> "
                              f"{len(shorter)} presses", flush=True)
                    plan = plan[:i] + shorter
                    spliced = True
                    break
                best, capped = [], False
                for k in range(5):
                    ns = self.step(states[i], k)
                    if ns is None or ns == states[i]:
                        continue
                    if self.won(ns):
                        if remaining == 0:
                            best.append(k)
                        continue
                    if remaining == 0:
                        continue
                    found, probe = self.bounded(ns, remaining, cap)
                    if probe["capped"]:
                        capped = True
                        break
                    if found is not None:
                        best.append(k)
                if capped or plan[i] not in best:
                    break
                sets[i] = best
                certified += 1
            if spliced:
                continue
            for i in range(len(plan)):
                if sets[i] is None:
                    sets[i] = [plan[i]]
            return plan, sets, certified

    # -- the whole strategy, in one call -------------------------------------
    def solve(self, cap: int = 8_000_000, beam_widths=(4_000, 16_000),
              probe_cap: int = 400_000, verbose=False):
        """A shortest plan with exact optimal sets when the reachable space
        fits in ``cap``; the best beam-and-certify plan otherwise. Returns
        ``(presses, optsets, info)``, or ``(None, None, info)`` when the level
        is provably unwinnable."""
        presses, optsets, stats = self.field(self.state, cap, verbose=verbose)
        if presses is not False:
            info = dict(stats, method="field", certified=len(presses or []))
            return presses, optsets, info
        plan = best_width = None
        for width in beam_widths:
            found = self.beam(self.state, width)
            if found is not None and (plan is None or len(found) < len(plan)):
                plan, best_width = found, width
        if plan is None:
            return None, None, dict(stats, method="beam")
        raw = len(plan)
        plan, optsets, certified = self.certify_tail(self.state, plan,
                                                     probe_cap, verbose=verbose)
        return plan, optsets, dict(stats, method="beam", beam_width=best_width,
                                   beam_presses=raw, certified=certified)


def _bits(mask: int):
    while mask:
        low = mask & -mask
        yield low.bit_length() - 1
        mask ^= low


def _interior(eng, ids) -> tuple[int, int, int, int]:
    """``(h, w, top, left)`` of the playfield: the open rectangle inside the
    level's wall frame. Everything outside it -- the decorated rock columns and
    the LEVEL / TURNS scoreboard under the frame -- is decoration on a collision
    layer nothing mechanical shares."""
    corner1, corner2 = ids["corner1"], ids["corner2"]
    top = left = right = None
    for r, row in enumerate(eng.grid):
        for c, cell in enumerate(row):
            if corner1 in cell:
                top, left = r + 1, c + 1
            if corner2 in cell and top is not None:
                right = c
    bottom = None
    for r, row in enumerate(eng.grid):
        if any(ids["corner3"] in cell for cell in row):
            bottom = r
    return bottom - top, right - left, top, left


def _turns_left(eng, ids) -> int:
    """The countdown, read off the TURNS digit under the frame. Zero when the
    digit is Turn0, which is the state in which the next press of anything
    restarts the level."""
    for n in range(1, 10):
        idx = ids[f"turn{n}"]
        if any(idx in cell for row in eng.grid for cell in row):
            return n
    return 0


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class SurfaceExpert(PSExpert):
    """`PSExpert`'s plan memo, restore discipline and disk cache around the
    native search. `heuristic` is never called -- nothing here is A*."""

    directions = list(DIRNAMES)
    plan_cache_path = (Path(__file__).resolve().parent.parent
                       / "data" / "the_world_beneath_the_surface_plans.json")

    #: `SurfaceBoard.field`'s state budget. Fifteen of the sixteen levels need
    #: at most 102k and are answered in under 20 s; level 14 is the outlier at
    #: 4,766,927 states inside 21 presses (its answer is 22, and the field
    #: enumerates 5,046,613), and this is set to hold it, so every level is
    #: solved by the exact field and nothing falls through to the beam. That
    #: level costs ~15 minutes and ~1.4 GB the first time and is then served
    #: from `plan_cache_path` forever.
    field_cap: int = 8_000_000
    #: Bound on the exhaustive probes `certify_tail` runs for a BEAMED level.
    #: Nothing uses it at the shipped `field_cap`; it is what a level would
    #: fall back on if a future edit made one too big for the field.
    probe_cap: int = 400_000
    beam_widths: tuple = (4_000, 16_000)

    def setup(self):
        names = list(BLOCKS) + list(WALLS) + ["player1", "player2"] + \
            [f"turn{n}" for n in range(0, 10)]
        self.ids = {n: self.g.obj_name_to_idx[n] for n in names
                    if n in self.g.obj_name_to_idx}
        self.info: dict = {}
        self._last_info = None

    def plan(self, eng, level=None):
        """`PSExpert.plan`, plus a note of HOW the plan was found. Only a
        search that actually ran leaves one -- a plan served from
        ``plan_cache_path`` reports itself as cached, which is the truth."""
        found = super().plan(eng, level)
        if level is not None and self._last_info is not None:
            self.info[level] = self._last_info
        self._last_info = None
        return found

    def heuristic(self, eng) -> int:
        raise AssertionError(
            "SurfaceExpert enumerates or beams; heuristic is unused")

    def board(self, eng) -> SurfaceBoard:
        return SurfaceBoard(eng, self.ids)

    def _search(self, eng):
        board = self.board(eng)
        presses, optsets, info = board.solve(cap=self.field_cap,
                                             beam_widths=self.beam_widths,
                                             probe_cap=self.probe_cap)
        self._last_info = info
        if presses is None:
            return None
        return Plan([DIRNAMES[k] for k in presses],
                    [[DIRNAMES[k] for k in s] for s in optsets])


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class SurfaceSolver(PSAStarSolver):
    """The recording harness. Recovery is the family's RESET prefix
    (`recovery_mode = "reset"`, inherited): a cleared block does not come back
    and a spent turn is spent, so a perturbed board cannot be re-planned from
    in general, and ONE reset restores exactly the state the cached plan was
    solved from. Level 10 is provably unwinnable and `discover_solvable` drops
    it; see the module docstring.

    ``record_spans`` is deliberately left off even though a whole cascade
    happens inside one press. `PuzzleScriptAdapter.perform_action` only returns
    the mid-press snapshots for a game with "playing" objects driving an
    animation, and this one has none -- so the settled frame IS everything a
    live agent is shown, and taping anything else would train on frames that do
    not exist at play time."""

    game_id = GAME_ID
    game_name = GAME_NAME
    game_module_id = GAME_ID
    expert_cls = SurfaceExpert
    #: The longest plan is level 11's 35 presses; this leaves room for it plus
    #: the exploration prefix and stays well inside the adapter's own 200-press
    #: per-level budget.
    max_steps = 140


# ---------------------------------------------------------------------------
# --selfcheck: the model against the interpreter
# ---------------------------------------------------------------------------

def _read_engine(board, eng, ids):
    """The interpreter's grid, in the model's own terms."""
    cells = [0] * board.n
    cur = 0
    for r in range(board.h):
        for c in range(board.w):
            cell = eng.grid[board.top + r][board.left + c]
            i = r * board.w + c
            for k, nm in enumerate(BLOCKS):
                if ids[nm] in cell:
                    cells[i] = k + 1
            if ids["player1"] in cell:
                cur = i
    return (bytes(cells), cur, _turns_left(eng, ids))


def _write_engine(board, eng, ids, st):
    """Load a model state INTO the interpreter, so a random board can be
    stepped by both and compared.

    The Explosions are written too, on exactly ``doomed(cells)``. That is not a
    convenience: a settled board in the interpreter ALWAYS carries an Explosion
    on every cell of every finished match (the marks the last ``again`` tick
    wrote, waiting for the next press to delete them), so a random board loaded
    without them is a state the game can never be in -- and the first fuzz run
    'diverged' on 1720 of its first 2500 boards for exactly that reason."""
    cells, cur, turns = st
    bg = ids["background"] if "background" in ids else ids["tile"]
    wall_ids = {ids[nm] for nm in WALLS if nm in ids}
    mask = board.doomed(cells)
    for r in range(board.h):
        for c in range(board.w):
            i = r * board.w + c
            cell = eng.grid[board.top + r][board.left + c]
            keep = {x for x in cell if x in wall_ids}
            new = {bg} | keep
            if cells[i]:
                new.add(ids[BLOCKS[cells[i] - 1]])
                if (mask >> i) & 1:
                    new.add(ids["explosion"])
            eng.grid[board.top + r][board.left + c] = new
    eng.grid[board.top + cur // board.w][board.left + cur % board.w].add(
        ids["player1"])
    eng.grid[board.top + cur // board.w][board.left + cur % board.w + 1].add(
        ids["player2"])
    for n in range(0, 10):
        for row in eng.grid:
            for cell in row:
                cell.discard(ids[f"turn{n}"])
    tr, tc = _TURN_CELL["rc"]
    eng.grid[tr][tc].add(ids[f"turn{turns}"])
    eng._position_index_dirty = True
    eng._rule_noop_cache = {}


#: Where the TURNS digit sits, found once at `selfcheck` startup. Every level
#: puts it in the same square of the scoreboard, and `_write_engine` needs it to
#: set a random board's countdown.
_TURN_CELL: dict = {}


def _variant_step(board, st, k, *, one_cell_gravity=False, three_only=False,
                  no_restart=False, closure_chain=False):
    """`SurfaceBoard.step` with one load-bearing decision deliberately made the
    other way, so ``--selfcheck`` can MEASURE that each one is a fact about the
    interpreter and not a preference.

    ``one_cell_gravity`` lets a block fall a single square per ``again`` tick,
    so a block can match in MID-AIR on the way past; ``three_only`` marks just
    the first three cells of a longer run instead of all of it; ``no_restart``
    lets the last turn's cascade run on past the ``[Turn0] -> restart`` rule;
    ``closure_chain`` makes the bomb chain a 4-connected CLOSURE (and, with it,
    arms ``again`` off the marks it added rather than off the rule matching at
    all) -- which is what the rule looks like it says and is not what four
    independent directional sweeps do.
    """
    cells_b, cur, turns = st
    if turns == 0 and not no_restart:
        return None
    cells = list(cells_b)
    for i in _bits(board.doomed(cells)):
        cells[i] = 0
    if k == 4:
        a, b = cur, cur + 1
        if cells[a] or cells[b]:
            cells[a], cells[b] = cells[b], cells[a]
            turns -= 1
    else:
        j = board.nb[cur][k]
        if j >= 0:
            cur = j
    for _ in range(400):
        fell = (_gravity_one(board, cells) if one_cell_gravity
                else board.gravity(cells))
        raw = (_matched_three(board, cells) if three_only
               else board.matched(cells))
        if closure_chain:
            mask = _closure_chain(board, cells, raw)
            matched = mask != raw
        else:
            mask, matched = board.spread(cells, raw)
        if all(not v or (mask >> i) & 1 for i, v in enumerate(cells)):
            return (bytes(cells), cur, turns)
        if not (fell or matched):
            return (bytes(cells), cur, turns)
        if turns == 0 and not no_restart:
            return None
        for i in _bits(mask):
            cells[i] = 0
    return (bytes(cells), cur, turns)


def _hunt(board, st, rng) -> int:
    """A press that walks the cursor toward the nearest square pair holding a
    block, and swaps once it is on one -- the fuzz's way of actually spending
    turns. Only used to CHOOSE a press; the model and the interpreter then both
    execute it and are compared, so a bad choice here costs coverage, never
    correctness."""
    cells, cur, _turns = st
    if cells[cur] or cells[cur + 1]:
        return 4
    d = board.dist[cur]
    targets = [q for q in range(board.n)
               if d[q] < _INF and (cells[q] or cells[q + 1])]
    if not targets:
        return rng.randrange(5)
    goal = min(targets, key=lambda q: (d[q], q))
    steps = [k for k in range(4)
             if board.nb[cur][k] >= 0
             and board.dist[board.nb[cur][k]][goal] < d[goal]]
    return rng.choice(steps) if steps else rng.randrange(5)


def _closure_chain(board, cells, mask: int) -> int:
    """The bomb chain as the 4-connected closure the rule text suggests."""
    stack = [i for i in _bits(mask) if cells[i] == BOMB]
    while stack:
        i = stack.pop()
        for k in range(4):
            j = board.dirnb[k][i]
            if j >= 0 and cells[j] and not (mask >> j) & 1:
                mask |= 1 << j
                if cells[j] == BOMB:
                    stack.append(j)
    return mask


def _gravity_one(board, cells) -> bool:
    """Gravity as ONE square per `again` tick, the way a rule that is not
    driven to a fixpoint inside its own pass would behave."""
    moved = False
    for seg in board.columns:
        for r in range(len(seg) - 2, -1, -1):
            if cells[seg[r]] and not cells[seg[r + 1]]:
                cells[seg[r + 1]] = cells[seg[r]]
                cells[seg[r]] = 0
                moved = True
    return moved


def _matched_three(board, cells) -> int:
    """The match rule as "the first three of a run", the way it reads if the
    5-wide and 4-wide families are ignored."""
    mask = 0
    for line in board.lines:
        for a in range(len(line) - MATCH + 1):
            v = cells[line[a]]
            if v and all(cells[line[a + d]] == v for d in range(MATCH)):
                for d in range(MATCH):
                    mask |= 1 << line[a + d]
                break
    return mask


def selfcheck(trials=24, steps=70, boards=4000, verbose=True):
    """Fuzz the model against the real interpreter, two ways.

    ROLLOUTS from every level start are the honest fuzz: random play walks into
    walls (an exact no-op for the cursor that still drains a pending
    explosion), swaps blocks with empty squares, spends turns and eventually
    restarts, and occasionally lines three up. But sixteen hand-made levels
    barely reach the bomb chain or a deep cascade, so the second mode loads
    RANDOM boards -- blocks and bombs scattered over the level's own floor,
    with a random cursor and a random countdown -- and steps one random press
    on each. That is where the four decisions the model turns on are measured
    against variants that make each the other way.
    """
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    eng, parsed = game._engine, game._game
    names = (list(BLOCKS) + list(WALLS)
             + ["player1", "player2", "background", "tile", "explosion"]
             + [f"turn{n}" for n in range(10)])
    ids = {n: parsed.obj_name_to_idx[n] for n in names
           if n in parsed.obj_name_to_idx}
    game.set_level(0)
    _TURN_CELL["rc"] = next((r, c) for r in range(eng.height)
                            for c in range(eng.width)
                            for n in range(1, 10)
                            if ids[f"turn{n}"] in eng.grid[r][c])
    rng = random.Random(20260821)
    bad = 0
    stats = {"presses": 0, "noop": 0, "swaps": 0, "cleared": 0, "restart": 0}

    for level in range(game.n_levels):
        game.set_level(level)
        board = SurfaceBoard(eng, ids)
        for t in range(trials):
            game.set_level(level)
            st = board.state
            mode = t % 3
            assert _read_engine(board, eng, ids) == st, f"L{level}: parse"
            for _ in range(steps):
                # Three play styles, because each covers what the others miss.
                # UNIFORM is what tests the no-op classification, the wall
                # bumps and the run-out-of-turns restart. GUIDED draws from the
                # presses the model believes change the BOARD, which is how a
                # rollout reaches a cascade at all. HUNT walks the cursor onto
                # the nearest block and swaps, which is the only way to get a
                # useful number of SWAPS out of random play -- the two other
                # modes together manage about one per hundred presses.
                # Uniform and guided both stay in: on its own, guided would
                # only ever test what the model already believes.
                k = rng.randrange(5)
                if mode == 1 and rng.random() < 0.75:
                    useful = [j for j in range(5)
                              if (board.step(st, j) or st)[0] != st[0]]
                    if useful:
                        k = rng.choice(useful)
                elif mode == 2 and rng.random() < 0.9:
                    k = _hunt(board, st, rng)
                model = board.step(st, k)
                before = _read_engine(board, eng, ids)
                eng._rule_restart = False
                eng.step(DIRNAMES[k])
                restarted = eng._rule_restart
                stats["presses"] += 1
                if model is None:
                    stats["restart"] += 1
                    if not restarted:
                        bad += 1
                        if verbose and bad < 6:
                            print(f"  L{level} t{t}: model says RESTART, "
                                  f"engine does not ({DIRNAMES[k]})")
                    break
                if restarted:
                    bad += 1
                    if verbose and bad < 6:
                        print(f"  L{level} t{t}: engine RESTART, model does "
                              f"not ({DIRNAMES[k]})")
                    break
                real = _read_engine(board, eng, ids)
                if real != model:
                    bad += 1
                    if verbose and bad < 6:
                        print(f"  L{level} t{t}: {DIRNAMES[k]} diverges\n"
                              f"    before {_show(board, before)}\n"
                              f"    model  {_show(board, model)}\n"
                              f"    engine {_show(board, real)}")
                    break
                if board.won(model) != eng.check_win():
                    bad += 1
                    if verbose and bad < 6:
                        print(f"  L{level} t{t}: win predicate disagrees after "
                              f"{DIRNAMES[k]}: model {board.won(model)}, "
                              f"engine {eng.check_win()}")
                    break
                if model == before:
                    stats["noop"] += 1
                if k == 4 and model[2] != before[2]:
                    stats["swaps"] += 1
                cleared = (sum(1 for v in before[0] if v)
                           - sum(1 for v in real[0] if v))
                stats["cleared"] += max(0, cleared)
                if eng.check_win():
                    break
                st = model

    # -- random boards -------------------------------------------------------
    # Its own RNG stream, so the four numbers this phase reports do not move
    # when the rollout phase above is edited -- they are quoted in the module
    # docstring as measurements, and a measurement that drifts with unrelated
    # changes is not one.
    rng = random.Random(915_2026)
    # Every wall layout the game ships, so the pillars -- which stop the cursor,
    # break a match line and hold a block up in mid-air -- are covered too.
    layouts = []
    for level in range(game.n_levels):
        game.set_level(level)
        b = SurfaceBoard(eng, ids)
        if not any(x.wall == b.wall for _lvl, x in layouts):
            layouts.append((level, b))
    wrong = {"one_cell_gravity": 0, "three_only": 0, "no_restart": 0,
             "closure_chain": 0}
    diverged = 0
    for _ in range(boards):
        level, board = rng.choice(layouts)
        game.set_level(level)
        density = rng.choice((0.25, 0.45, 0.7))
        # Bomb-heavy palettes are over-represented on purpose: the chain is
        # the one rule the sixteen shipped levels barely exercise, and it is
        # the one the model's least obvious decision (four one-way sweeps, not
        # a closure) turns on.
        palette = rng.choice(((1, 2, 3), (1, 2, 3, 7), (1, 2), (7, 7, 1, 2),
                              (7, 7, 7, 1), (7, 7, 7, 1, 2),
                              (1, 2, 3, 4, 5, 6, 7)))
        cells = [0] * board.n
        for i in range(board.n):
            if not board.wall[i] and rng.random() < density:
                cells[i] = rng.choice(palette)
        # settle it the way the game always presents a board
        board.gravity(cells)
        cur = rng.choice([i for i in range(board.n)
                          if i % board.w < board.w - 1
                          and not board.wall[i] and not board.wall[i + 1]])
        st = (bytes(cells), cur, rng.randint(1, 9))
        k = rng.randrange(5)
        _write_engine(board, eng, ids, st)
        eng._rule_restart = False
        eng.step(DIRNAMES[k])
        real = None if eng._rule_restart else _read_engine(board, eng, ids)
        model = board.step(st, k)
        if real != model:
            diverged += 1
            bad += 1
            if verbose and diverged < 4:
                print(f"  random board: {DIRNAMES[k]} diverges\n"
                      f"    before {_show(board, st)}\n"
                      f"    model  {_show(board, model)}\n"
                      f"    engine {_show(board, real)}")
        for name in wrong:
            if _variant_step(board, st, k, **{name: True}) != real:
                wrong[name] += 1

    if verbose:
        print(f"  rollouts: {stats['presses']} presses, {stats['noop']} exact "
              f"no-ops, {stats['swaps']} swaps, {stats['cleared']} blocks "
              f"cleared, {stats['restart']} restarts")
        print(f"  random boards: {boards} stepped, {diverged} divergences")
        for name, n in wrong.items():
            print(f"    variant {name}: wrong on {n}/{boards}")
    return bad


def _show(board, st) -> str:
    if st is None:
        return "RESTART"
    cells, cur, turns = st
    sym = ".rbygcm@"
    rows = []
    for r in range(board.h):
        rows.append("".join("#" if board.wall[r * board.w + c]
                            else sym[cells[r * board.w + c]]
                            for c in range(board.w)))
    return f"[{'/'.join(rows)}] cur={cur} turns={turns}"


# ---------------------------------------------------------------------------
# --audit: every cell composition, pixel-distinct at every cell size
# ---------------------------------------------------------------------------

#: Every stack a square of the PLAYFIELD can hold. Block, Wall and Explosion
#: are three collision layers, and the two cursors are on the Wall layer, so a
#: square holds at most one block, at most one cursor half, and the Explosion.
_AUDIT_CASES: dict = {"floor": (), "wall": ("singlewall",),
                      "cursor L": ("player1",), "cursor R": ("player2",)}
for _b in BLOCKS:
    _AUDIT_CASES[_b] = (_b,)
    _AUDIT_CASES[_b + " + boom"] = (_b, "explosion")
    _AUDIT_CASES[_b + " + cursor"] = (_b, "player1")
    _AUDIT_CASES[_b + " + cursor + boom"] = (_b, "player1", "explosion")

# NB there is deliberately no TURNS-digit group here. The scoreboard under the
# frame is four rows of `Board` and `Turn` objects and nothing else, so
# `_render_frame`'s HUD crop takes all four off every frame this game ever
# shows: the countdown is not drawn, and the module docstring says what that
# costs and why it is left alone.

#: The one pair that is ALLOWED to render identically. Player1 and Player2 are
#: the two halves of one rigid cursor and the game itself draws them with the
#: same sprite in the same colour; which half is which is fixed by geometry
#: (Player1 is always the square to Player2's left, in engine space), so no
#: decision anywhere depends on telling them apart by pixels.
_AUDIT_ALLOWED = {("cursor L", "cursor R")}


def audit(verbose=True):
    """Assert every cell COMPOSITION the playfield can hold renders
    differently, at every cell size the levels actually use.

    A level is 18x14 on the grid but 14x14 in the FRAME -- `_render_frame`
    crops the four scoreboard rows as HUD -- so a cell is FOUR pixels and a
    5x5 sprite is sampled at rows and columns 0, 1, 3 and 4. Getting that
    wrong is not academic: the first version of this test built its boards at
    18x14, measured three-pixel cells, and passed a sprite set that was wrong
    for the size the game renders at. So the test overwrites only the
    PLAYFIELD rows of a real level and leaves the scoreboard exactly as
    shipped, which makes the HUD crop take the same rows off as it does in
    play.

    Whole frames are compared rather than one cell sliced out, because
    `_render_frame` upscales and then letterboxes, so the cell grid in the
    output is not ``cell_px``-aligned and a crop would land in the wrong
    window. Rendering is per-cell independent, so filling the board with one
    composition is the same test done where the geometry cannot drift."""
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    eng, parsed = game._engine, game._game
    idx = parsed.obj_name_to_idx
    hud = {idx[n] for g in ("board", "turn") for n in parsed.or_groups.get(g, ())
           if n in idx} | {idx["turn0"]}

    shapes = {}
    for level in range(game.n_levels):
        game.set_level(level)
        play = [r for r in range(eng.height)
                if not all(eng.grid[r][c] & hud for c in range(eng.width))]
        base = [[set(cell) for cell in row] for row in eng.grid]
        shapes.setdefault((len(play), eng.width, tuple(play)),
                          []).append((level, base))

    bad = 0
    for (ph, w, play), members in sorted(shapes.items()):
        level, base = members[0]
        game.set_level(level)
        shots = {}
        for name, objs in _AUDIT_CASES.items():
            # Only the PLAYFIELD rows are overwritten. The scoreboard rows stay
            # exactly as the level shipped them, so `_render_frame`'s HUD crop
            # takes the same rows off as it does in a real frame and the cell
            # size is the real one (4px, not the 3px an uncropped 18x14 would
            # give -- the first version of this audit made that mistake and
            # passed a sprite set that was wrong for the size actually used).
            eng.grid = [[set(cell) for cell in row] for row in base]
            for r in play:
                for c in range(w):
                    eng.grid[r][c] = {idx["tile"]} | {idx[o] for o in objs}
            eng._position_index_dirty = True
            shots[name] = np.asarray(_render_frame(eng, parsed))
        clashes = [(a, b) for a, b in itertools.combinations(shots, 2)
                   if np.array_equal(shots[a], shots[b])
                   and (a, b) not in _AUDIT_ALLOWED]
        bad += len(clashes)
        if verbose:
            lv = ",".join(str(x) for x, _ in members)
            print(f"  {ph}x{w} after the HUD crop (cell {64 // max(ph, w)}px, "
                  f"levels {lv}): {len(_AUDIT_CASES)} compositions, "
                  f"{'all distinct' if not clashes else 'IDENTICAL ' + str(clashes)}")
    return bad


# ---------------------------------------------------------------------------
# --symmetry: the presentation augmentation is a true symmetry of the mechanic
# ---------------------------------------------------------------------------

def symmetry(walk_presses=90, seeds=12, verbose=True):
    """Drive the ADAPTER at every presentation it can draw, pressing the SCREEN
    action for each planned engine direction, and require every frame to be
    exactly the rotation of the unaugmented one.

    This is the end-to-end test of the rotation contract these generators run
    on (`solvers/common/ps_astar.screen_action`): the expert plans in ENGINE
    space, the adapter forward-remaps the press it is given, and the two have
    to cancel. Get it backwards and nothing raises -- the adapter just executes
    a different direction than the plan meant.

    Rotation only. A quarter turn is a rigid motion of the plane, so gravity,
    the rigid cursor pair and the match lines all turn with it. A MIRROR does
    not: it swaps the two halves of the cursor, and the game's swap is written
    as two rules that are NOT mirror images (`[Action Player1 Block][Corner1]`
    parks Player1's block in the TOP-LEFT corner and hands it back to Player2;
    Player2's goes to the top-right and comes back to Player1). The net effect
    happens to be the same swap, but the game also has gravity, which a
    vertical flip inverts outright -- so the game stays out of ``_FLIP_GAMES``
    and this checks the four rotations it does get."""
    ref = PuzzleScriptAdapter(GAME_NAME, seed=0)
    rng = random.Random(7)
    presses = [rng.randrange(5) for _ in range(walk_presses)]
    bad = 0
    rotations = set()
    for level in range(ref.n_levels):
        ref.set_level(level)
        base = []
        for k in presses:
            # The reference walks the raw engine, so it has to do the two
            # things `perform_action` does around it: reload the level when the
            # countdown ran out and a rule fired `restart` (the engine only
            # sets the flag), and stop at a WIN, after which the adapter
            # short-circuits and freezes the frame. Without the first of those
            # this test "found" 12 violations that were only the reference
            # refusing to restart -- at rotation 0, where a rotation bug cannot
            # exist.
            ref._engine.step(DIRNAMES[k])
            if ref._engine._rule_restart:
                ref._engine.load_level(
                    ref._game.levels[level % len(ref._game.levels)])
            base.append(np.asarray(_render_frame(ref._engine, ref._game)))
            if ref._engine.check_win():
                break
        for seed in range(seeds):
            g = PuzzleScriptAdapter(GAME_NAME, seed=seed)
            g.set_level(level)
            rot, hf, vf = g._rotation_k, g._hflip, g._vflip
            rotations.add((rot, hf, vf))
            for i in range(len(base)):
                act = screen_action(DIRNAMES[presses[i]], rot, hf, vf)
                g.perform_action(ActionInput(id=act))
                got = np.asarray(g._current_frame)
                want = np.ascontiguousarray(np.rot90(base[i], k=rot))
                if not np.array_equal(got, want):
                    bad += 1
                    if verbose and bad < 4:
                        print(f"  L{level} seed{seed} rot{rot} press {i} "
                              f"({DIRNAMES[presses[i]]} -> {act.name}): frame "
                              f"is not the rotation of the unaugmented one")
                    break
    if verbose:
        print(f"  {ref.n_levels} levels x {seeds} seeds x {walk_presses} "
              f"presses; presentations seen (rot, hflip, vflip): "
              f"{sorted(rotations)}")
    return bad


# ---------------------------------------------------------------------------
# --bfs / --ties: the plans and their labels, re-derived independently
# ---------------------------------------------------------------------------

def _boards(levels=None):
    """``(level, SurfaceBoard)`` for the levels asked for."""
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    parsed = game._game
    names = (list(BLOCKS) + list(WALLS) + ["player1", "player2"]
             + [f"turn{n}" for n in range(10)])
    ids = {n: parsed.obj_name_to_idx[n] for n in names
           if n in parsed.obj_name_to_idx}
    for level in (range(game.n_levels) if levels is None else levels):
        game.set_level(level)
        yield level, SurfaceBoard(game._engine, ids)


def bfs_report(levels=None, cap=2_000_000, verbose=True):
    """Re-derive every level's shortest win with a search that shares nothing
    with `SurfaceBoard.field` but the model.

    `field` layers the space and then induces backwards over the layers, which
    is where a subtle off-by-one would live. This instead asks
    `SurfaceBoard.bounded` -- a plain depth-limited frontier that re-expands
    from the start every time -- for a win at 0, 1, 2, ... presses and takes
    the first depth that answers. Same number, or the plans are wrong.

    A level whose space does not fit in ``cap`` is reported as out of budget
    rather than as a mismatch: level 14 needs 4.77M states just to reach its
    answer, and re-expanding that twenty-two times over is not the cheap
    cross-check this is meant to be. Give it a level list and a bigger cap if
    you want it anyway."""
    solver = SurfaceSolver()
    game, expert, _ = solver._ensure(0)
    bad = 0
    for level, board in _boards(levels):
        game.set_level(level)
        plan = expert.plan(game._engine, level)
        want = None if plan is None else len(plan)
        got, capped, limit = None, False, 0
        while True:
            found, stats = board.bounded(board.state, limit, cap)
            if found is not None:
                got = len(found)
                break
            if stats["capped"]:
                capped = True
                break
            if stats.get("exhausted"):
                break                      # whole component, no win anywhere
            if limit > (want if want is not None else 60):
                break
            limit += 1
        verdict = ("out of budget" if capped else
                   "ok" if got == want else "MISMATCH")
        bad += 1 if verdict == "MISMATCH" else 0
        if verbose:
            print(f"  L{level:2d}: plan {want}, iterative deepening {got}"
                  f"  {verdict}")
    return bad


def ties_report(levels=None, cap=400_000, verbose=True):
    """Re-derive every optimal-action LABEL, exhaustively and from scratch.

    A press is optimal at step ``i`` iff a win still exists in the presses the
    plan has left after it -- so every step is decided by five bounded searches
    that know nothing about the layers `field` labelled it from. Disagreement
    is counted in BOTH directions: a press the plan calls optimal that cannot
    finish in time, and a press it omits that can.

    Walked BACKWARDS from the end of the plan, because the bound each probe
    runs under -- and so its cost -- shrinks towards the end, and once a probe
    hits ``cap`` every earlier step would too. The report says where the line
    fell on each level."""
    solver = SurfaceSolver()
    game, expert, _ = solver._ensure(0)
    bad = total = ties = 0
    for level, board in _boards(levels):
        game.set_level(level)
        plan = expert.plan(game._engine, level)
        if plan is None:
            if verbose:
                print(f"  L{level:2d}: no plan (provably unwinnable)")
            continue
        sets = getattr(plan, "optsets", None) or [[d] for d in plan]
        states = [board.state]
        for direction in plan:
            states.append(board.step(states[-1], DIRNAMES.index(direction)))
        wrong = done = 0
        for i in range(len(plan) - 1, -1, -1):
            remaining = len(plan) - i - 1
            want, capped = set(), False
            for k in range(5):
                ns = board.step(states[i], k)
                if ns is None:
                    continue
                if board.won(ns):
                    if remaining == 0:
                        want.add(k)
                    continue
                if remaining == 0:
                    continue
                found, stats = board.bounded(ns, remaining, cap)
                if stats["capped"]:
                    capped = True
                    break
                if found is not None:
                    want.add(k)
            if capped:
                break
            got = {DIRNAMES.index(d) for d in sets[i]}
            total += 1
            ties += len(got) - 1
            done += 1
            if got != want:
                wrong += 1
                if verbose and bad + wrong < 5:
                    print(f"  L{level:2d} step {i}: labelled "
                          f"{sorted(DIRNAMES[k] for k in got)}, exhaustive "
                          f"{sorted(DIRNAMES[k] for k in want)}")
        bad += wrong
        if verbose:
            print(f"  L{level:2d}: {done}/{len(plan)} steps re-derived, "
                  f"{'all confirmed' if not wrong else f'{wrong} DISAGREE'}")
    if verbose:
        print(f"  {total} steps re-derived, {ties} of them with a tie")
    return bad


# ---------------------------------------------------------------------------
# --plans
# ---------------------------------------------------------------------------

def _plan_report():
    """Every level's plan, replayed through the real interpreter -- the
    end-to-end test of the model, the search and the tie labelling at once.
    ``[cached]`` means the plan came out of `plan_cache_path` without a search
    running, which is the truth about that run and not a claim about the plan."""
    solver = SurfaceSolver()
    game, expert, solvable = solver._ensure(0)
    total = ties = 0
    for level in range(game.n_levels):
        game.set_level(level)
        eng = game._engine
        board = expert.board(eng)
        plan = expert.plan(eng, level)
        if plan is None:
            print(f"  L{level:2d}: {sum(1 for v in board.state[0] if v):2d} "
                  f"blocks, {board.turns} turns -> UNSOLVED")
            continue
        blocks = sum(1 for v in board.state[0] if v)
        for direction in plan:
            eng.step(direction)
        sets = getattr(plan, "optsets", None) or [[p] for p in plan]
        step_ties = sum(len(s) - 1 for s in sets)
        swaps = sum(1 for d in plan if d == "action")
        total += len(plan)
        ties += step_ties
        info = expert.info.get(level)
        how = ("cached" if info is None else
               f"{info['method']}, {info['states']} states, "
               f"{info['certified']}/{len(plan)} certified")
        print(f"  L{level:2d}: {blocks:2d} blocks, {board.turns} turns "
              f"-> {len(plan):3d} presses ({swaps} action)"
              f"  win={eng.check_win()}  {step_ties:3d} ties  [{how}]")
    print(f"  solvable levels: {solvable}")
    print(f"  {total} presses, {ties} of them with a second right answer")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        violations = selfcheck()
        print(f"selfcheck: {violations} violations")
        sys.exit(1 if violations else 0)
    if "--audit" in sys.argv:
        collisions = audit()
        print(f"audit: {collisions} colliding compositions")
        sys.exit(1 if collisions else 0)
    if "--symmetry" in sys.argv:
        violations = symmetry()
        print(f"symmetry: {violations} violations")
        sys.exit(1 if violations else 0)
    if "--bfs" in sys.argv:
        rest = sys.argv[sys.argv.index("--bfs") + 1:]
        args = [int(a) for a in rest if a.isdigit()]
        opts = dict(a.split("=") for a in rest if "=" in a)
        violations = bfs_report(args or None,
                                cap=int(opts.get("--cap", 2_000_000)))
        print(f"bfs: {violations} mismatches")
        sys.exit(1 if violations else 0)
    if "--ties" in sys.argv:
        rest = sys.argv[sys.argv.index("--ties") + 1:]
        args = [int(a) for a in rest if a.isdigit()]
        opts = dict(a.split("=") for a in rest if "=" in a)
        violations = ties_report(args or None,
                                 cap=int(opts.get("--cap", 400_000)))
        print(f"ties: {violations} disagreements")
        sys.exit(1 if violations else 0)
    if "--plans" in sys.argv:
        _plan_report()
        sys.exit(0)
    sys.exit(SurfaceSolver.main())
