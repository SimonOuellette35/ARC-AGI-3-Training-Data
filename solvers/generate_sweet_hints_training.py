"""Generate Phase-1 training data for the PuzzleScript game ps:sweet_hints
("Sweet Hints", Marcos Donnantuoni).

The harness -- the rotation contract, the trajectory recorder and the
`BaseSolver` plumbing -- lives in `solvers/common/ps_astar.py`, shared with the
other ps: generators. This file is the game-specific part: the measured
mechanic, a native model of it, the exact distance FIELD that model is solved
with, the proofs that every plan is shortest, and the render audit.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_sweet_hints",
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
exactly. Every expert step carries the full set of equally-optimal presses.

The game
--------
A WITNESS PANEL drawn with the arrow keys. Five levels, each a set of grey
`space` squares on which the player DRAWS A LINE from a fixed start to the one
`target` square, and the win is not reaching the target -- it is reaching it
with a line that satisfies the level's constraint.

The state is therefore the LINE, and it is a simple path. Every press is one of
exactly three things, and the rules say so directly:

* **extend** -- ``right [ right player | floor ] -> [ lineR no floor | player
  floor ]``. The player steps onto an adjacent square that still has floor
  (`space` or `target`); the square it leaves keeps a **lineR** and LOSES its
  floor. So a square the line already covers can never be entered again, which
  is what makes the line simple.
* **retreat** -- ``right [ right player floor | lineL ] -> [ no player floor |
  player space ]``. Pressing toward a neighbour whose line points BACK at the
  head pops the head: the line segment is deleted and the square becomes plain
  `space` again. Exactly one square can ever point at the head (the path is
  simple, so the head has one predecessor), so this is the pop and nothing else.
* **nothing** -- ``[ > player | no space ] -> cancel``. Into a wall, into the
  board edge, or into a line segment that is not the predecessor: the whole turn
  is cancelled, so it is a true no-op.

Extend and retreat are exact inverses, so **the reachable space is an UNDIRECTED
graph** -- which is what the solver below is built on.

The constraint is checked ONLY when the player is standing on the target. A
`late`-less cascade then floods an invisible `endmark` over the whole board and
three families of rules paint a red `error` wherever the drawn line has failed:

* **dots** (``[ dot endmark no line ] -> [ dot error ]``) -- every marked square
  must be COVERED by the line. Level 4.
* **squares** -- `sqmark` floods out of every sq1 through everything that is not
  `sep` (= wall, border, line, player), and a square of a different colour
  reached by that flood is an error. Repeated per colour, that is the Witness
  separation rule: **the regions the line cuts the board into must each hold at
  most one colour of square.** Level 5.
* **stars** -- six `random`-assigned marks flood the same way, and a star with
  anything but exactly two of them is an error: **a star's region must hold
  exactly two objects of its colour** (stars and squares together). No shipped
  level contains a star, so this branch of `_Panel.won` is modelled and never
  exercised; `--plans` reports it.

Win is ``all player on target`` and ``no error``, so a wrong arrival is not a
loss -- the board just sits there with red boxes on it and the line has to be
walked back out.

Three facts about the shipped levels that the solver leans on, all asserted by
`--plans` rather than assumed:

* **the target is a DEAD END.** In all five levels the target square has exactly
  one walkable neighbour, so the only press available from it is the retreat
  back down the line. That matters because walking OFF a target is the one
  irreversible act in this game: the extend rule strips the floor from the
  square it leaves, and the retreat rule puts back `space` -- not `target`. A
  level where the target had two walkable neighbours could be bricked. `_Panel`
  models that transition anyway (`step` reports it) and `_Field` refuses to
  expand it; on these levels it is unreachable.
* **there is exactly one target**, so ``all player on target`` is "the head is
  the target square".
* **ACTION does nothing.** The game declares `noaction` and no rule has `action`
  on a left-hand side, but that is the argument ps:ouroboros falsified, so
  `--selfcheck` MEASURES it: over 2500 presses from live boards, an ACTION press
  changes no object, no rendered pixel and no win flag. (On an already-WON board
  it does clear the leftover `endmark` layer, which is invisible and reached
  only after the episode is over.)

Expert solver
-------------
A native model of the mechanic above (`_Panel`) plus an EXHAUSTIVE distance
field (`_Field`) -- the whole reachable space, every level, and a backward BFS
from the wins. The spaces are 5 / 9 / 32 / 421 / 6449 states, so there is
nothing here worth searching: enumeration is 0.05 s for the entire game and it
is the only tier that gives all three of

  * plans that are provably SHORTEST -- no heuristic, no weight, no node cap;
  * EXACT optimal-action sets (``dist(succ) == dist - 1``), measured, not
    inferred;
  * an answer from ANY board, which is the whole `supports_recovery` contract.

The enumeration starts from whatever state it is asked about rather than from
the level start, and that is not a shortcut: extend and retreat are inverses, so
the space is one undirected component and the two agree. `--selfcheck` measures
that (the component enumerated from 200 random states is the same set as the one
enumerated from the start, on every level).

Winning states are not expanded, which loses nothing: a win's only successors
are the retreat -- which is its own predecessor, already enumerated -- and the
extend off the target, which the target's dead-end shape rules out.

Shortest, and how that is known
-------------------------------
Three independent derivations agree on all five levels:

  * this file's backward field over `_Panel`;
  * a plain forward BFS to the first win over the same model (`--selfcheck`
    pass 4), with the per-step tie sets re-derived a second time by brute force
    (pass 5);
  * the same enumeration driven by the REAL INTERPRETER (`--engine`): a DFS that
    walks the whole space by pressing actual buttons and UN-pressing them to
    come back, so it needs neither snapshots nor a decoder, keying each board by
    what is on the grid and testing the goal with `check_win`. No `_Panel` call
    participates. It agrees on the state SET (6916 states, exactly the model's),
    on all five lengths and on every tie set -- and, because the walk back is an
    assertion at every state-changing edge, it is also the measurement of the
    reversibility claim the field rests on: 13822 undos out of 41486 presses, 0
    exceptions.

Recovery
--------
`supports_recovery` with the family's RESET-mode prefix, plus ``epsilon = 0.12``
detours inside the expert replay. The expert re-plans from the LIVE board (a
dict lookup at whatever state the engine is in), so both kinds of perturbation
are answered exactly: the taken action is the mistake and ``optimal`` is the
recovery, which is the signal a policy needs after its own error.

This game is unusually good at that. Reversibility means a detour can never
brick a board, so `record_level`'s "is this still winnable" probe always passes
and the detours actually happen instead of being silently skipped -- and every
level's shipped plan is FORCED, so without the detours the corpus would never
once show what a wrong press looks like or how to undo it -- and the retreat is
half of this game's mechanic.

Ties are all but absent here, and that is a property of the game rather than of
the labelling: over the 6916 states of all five levels exactly ONE has two
equally shortest presses (``--plans`` counts them). A panel has one solution
line, so off it there is one way back onto it.

Augmentation
------------
The engine state after reset is identical for every seed (levels are fixed ASCII
maps), so the only per-(seed, level) variables are the frame rotation
(``rotation_k`` in 0..3) and the two flips, with their matching directional
action remap. 5 levels x 16 presentations = 80.

``Sweet_Hints`` is in `PuzzleScriptAdapter._FLIP_GAMES`; the argument is
recorded there and ``--symmetry`` measures it in the strong form -- each level's
plan and a seeded random walk replayed on all eight turned and mirrored copies
of its own board (built by transforming the LEVEL LAYOUT, so the interpreter
re-derives everything itself), requiring the rendered frame to be the exact
transform at every press. That is the form that matters here, because the four
movement rules are written as four per-direction blocks -- the shape that hid
ps:gobble_rush's chirality.

Rendering
---------
Four sprite defects had to be fixed, and the full statement is the header
comment in ``data/puzzlescript_games/Sweet_Hints.txt``. In short: `border` was
drawn in the Background's own blue (it is the ring that makes level 5's
separation puzzle well-posed, and it was not on screen); the four `lineX` were
one identical white block, so nothing said which neighbour the head can retreat
onto; the four directional `playerX` were chosen by a rule ORDER, which no
rotation can carry, so they showed a policy contradictory evidence for a picture
that decides nothing; and `dot` was a chiral blob, so one marker was four
pictures across the sixteen presentations. No object, layer, legend entry, rule,
level or win condition was touched.

``--audit`` renders every cell COMPOSITION a board of this game can hold as a
whole 64x64 frame of a uniform board, at both cell sizes the levels use (5 and 7
px), and requires them distinct -- whole frames rather than one cell sliced out
of a mixed board, because `_render_frame` upscales and centre-pads (the
ps:explod lesson). It then checks the two DIRECTIONAL families as orbits: every
rotation and mirror of a `lineX` must be the art of the lineX that transform
relabels it to, and likewise for `targetX`.

Usage (run from the repo root):
    python solvers/generate_sweet_hints_training.py --episodes 200 \
        --out data/training_multi_level/sweet_hints

    python solvers/generate_sweet_hints_training.py --plans      # level report
    python solvers/generate_sweet_hints_training.py --selfcheck  # model + optimality
    python solvers/generate_sweet_hints_training.py --engine     # interpreter proof
    python solvers/generate_sweet_hints_training.py --audit      # rendering
    python solvers/generate_sweet_hints_training.py --symmetry   # augmentation
"""

from __future__ import annotations

import itertools
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arcengine import ActionInput, GameState                          # noqa: E402
from solvers.base_solver import _ID_TO_GAMEACTION                     # noqa: E402
from solvers.common.ps_astar import (PSAStarSolver, PSExpert, Plan,   # noqa: E402
                                     screen_action, snapshot, restore)
from utils.rotation import inverse_remap_action_full                  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent

GAME_NAME = "Sweet_Hints"

#: Engine directions, in the order ties are broken. ACTION is left out: it is
#: measured to be a total no-op on every live board (`--selfcheck` pass 0), so
#: branching on it would double every sweep for nothing. The exploration prefix
#: still presses it -- a live agent has that button -- which is why
#: ``--symmetry``'s random walk presses it too.
DIRS: tuple[str, ...] = ("up", "down", "left", "right")

_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
#: ``DIRS`` index of the press that undoes ``DIRS[i]``. Retreat is the exact
#: inverse of extend, which is what `_Field`, `_engine`'s snapshot-free walk and
#: the recovery contract all rest on.
_OPPOSITE = {0: 1, 1: 0, 2: 3, 3: 2}

#: The line object left behind by each press, i.e. ``lineU`` for an up-press.
#: Read only by the reports (seating a board, drawing one), never by the model.
_LINE_OF = {"up": "lineu", "down": "lined", "left": "linel", "right": "liner"}

#: Disk cache of every level's start plan AND its optimal-action sets. The whole
#: game's fields take well under a second, so this is not the load-bearing cache
#: it is on the big ps: games -- it is here so a `parallelize_generator` fan-out
#: shares one derivation rather than repeating it on every core. Delete the file
#: to re-derive it.
PLAN_CACHE: Path = _REPO_ROOT / "data" / "sweet_hints_plans.json"


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

class _Panel:
    """One level's static geometry, plus the mechanic and the constraint.

    Cells are flat ``r * w + c`` indices and a STATE is the LINE: a tuple of
    cells, ``path[0]`` the level's start square and ``path[-1]`` the head the
    player is standing on. Everything else about the board is a function of it --
    ``path[:-1]`` are the squares carrying a line segment and having lost their
    floor, and the direction of each segment is the step to the next cell -- so
    nothing else needs to be in the key.

    THE GEOMETRY, all of it derived from the level start (see
    `SweetHintsExpert.read`, which can rebuild every field of this class from any
    mid-game board):

      * ``sep`` -- wall and border. The two are one object mechanically (same
        collision layer, named by no rule except through ``sep``, in no win
        condition) and are now drawn alike; they block the player AND block the
        region flood.
      * ``floor0`` -- every square that had floor at the level start, i.e. every
        square the line can ever run over.
      * ``targets`` / ``dots`` / ``squares`` / ``stars`` -- the constraint.

    ``nb`` is 4-neighbourhood ignoring ``sep``, which is what the region flood
    needs: `Background`, `space`, `target`, `dot` and the coloured squares are
    all things the flood passes through. ``edge`` is the same neighbourhood WITH
    ``sep`` removed, which is what a press needs.
    """

    __slots__ = ("h", "w", "n", "sep", "floor0", "targets", "dots", "squares",
                 "stars", "edge", "nb", "sig")

    def __init__(self, h, w, sep, floor0, targets, dots, squares, stars):
        self.h, self.w, self.n = h, w, h * w
        self.sep = frozenset(sep)
        self.floor0 = frozenset(floor0)
        self.targets = frozenset(targets)
        self.dots = frozenset(dots)
        self.squares = dict(squares)
        self.stars = dict(stars)
        self.edge, self.nb = [], []
        for i in range(self.n):
            r, c = divmod(i, w)
            press, flood = [], []
            for d in DIRS:
                dr, dc = _DELTA[d]
                nr, nc = r + dr, c + dc
                j = nr * w + nc if 0 <= nr < h and 0 <= nc < w else -1
                press.append(-1 if j < 0 or j in self.sep else j)
                if j >= 0:
                    flood.append(j)
            self.edge.append(tuple(press))
            self.nb.append(tuple(flood))
        self.sig = (h, w, tuple(sorted(self.sep)), tuple(sorted(self.floor0)),
                    tuple(sorted(self.targets)), tuple(sorted(self.dots)),
                    tuple(sorted(self.squares.items())),
                    tuple(sorted(self.stars.items())))

    # -- the mechanic --------------------------------------------------------
    def step(self, path, di: int):
        """``(new path, breaks a target)`` after pressing ``DIRS[di]``.

        The path is None when the press does nothing at all -- into ``sep``, off
        the board, or into a line segment that is not the head's predecessor, all
        of which the interpreter answers with ``cancel``.

        ``breaks`` flags the one irreversible press in the game: extending OFF a
        target square strips its floor, and the retreat that would undo it puts
        back plain `space`, so the target is gone for good and the level can
        never be won. `_Field` refuses to expand those; on every shipped level
        the target is a dead end and no such press exists (`--plans` asserts it).
        """
        head = path[-1]
        q = self.edge[head][di]
        if q < 0:
            return None, False
        if len(path) >= 2 and q == path[-2]:
            return path[:-1], False                  # retreat
        if q in self.floor0 and q not in path:
            return path + (q,), head in self.targets  # extend
        return None, False

    # -- the constraint ------------------------------------------------------
    def regions(self, path):
        """The connected components of everything the line does NOT cut off.

        ``sep`` is wall, border, LINE and PLAYER, so the blocked set is the
        static walls plus the whole path, head included, and the components are
        taken over every other cell of the grid -- background outside the panel
        included, which is exactly what the interpreter's flood does."""
        blocked = self.sep | set(path)
        seen = set(blocked)
        out = []
        for i in range(self.n):
            if i in seen:
                continue
            comp = [i]
            seen.add(i)
            queue = deque([i])
            while queue:
                cur = queue.popleft()
                for j in self.nb[cur]:
                    if j not in seen:
                        seen.add(j)
                        comp.append(j)
                        queue.append(j)
            out.append(comp)
        return out

    def won(self, path) -> bool:
        """``all player on target`` and ``no error``, verbatim.

        The head must be a target square (that is the whole of the first
        condition -- there is one player), and then the three error families:
        every dot covered by a line SEGMENT (the head carries `player`, not
        `line`, so a dot under the player is not yet satisfied), every region
        holding at most one colour of square, and every star's region holding
        exactly two objects of the star's colour."""
        if path[-1] not in self.targets:
            return False
        if not self.dots <= set(path[:-1]):
            return False
        if not self.squares and not self.stars:
            return True
        for comp in self.regions(path):
            colours = {self.squares[c] for c in comp if c in self.squares}
            if len(colours) > 1:
                return False
            if not self.stars:
                continue
            for c in comp:
                k = self.stars.get(c)
                if k is None:
                    continue
                pair = sum(1 for x in comp
                           if self.stars.get(x) == k or self.squares.get(x) == k)
                if pair != 2:
                    return False
        return True


class _Field:
    """Exact presses-to-win for EVERY state, by one exhaustive enumeration.

    The whole reachable space of the biggest level is 6449 states, so there is
    nothing to search: `_enumerate` walks it all and a backward BFS from the
    winning states labels every one of them with its true distance. That gives
    provably shortest plans, exact optimal-action SETS and an answer from an
    arbitrary board, none of which a heuristic search can offer.

    THE SEED STATE DOES NOT MATTER. Extend and retreat are exact inverses, so
    the space is an UNDIRECTED graph and the component reached from any state is
    the component reached from the level start -- including from a WON state,
    which is why winning states are expanded like any other rather than treated
    as terminal. (They end the EPISODE, but a field that stopped at them would be
    a single state if it were ever seeded at one, and their only successor is
    their own predecessor anyway: the extend off a target is what `_Panel.step`
    flags as target-breaking, and the shipped levels make it impossible.) That
    is what lets a re-plan after an epsilon detour reuse the field built for the
    level's opening press -- and it is measured, not assumed (`--selfcheck`
    pass 3, and `--engine` asserts the reversibility itself at every edge it
    walks on the interpreter).

    ``cap`` is a runaway guard for an edited level, not a budget: past it the
    field answers None for everything, which `_search` turns into "no plan" and
    `record_level` turns into a skipped level. The shipped levels use 0.1% of it.
    """

    __slots__ = ("panel", "succ", "dist", "wins", "capped")

    def __init__(self, panel: _Panel, seed_state, cap: int):
        self.panel = panel
        self.succ: dict = {}
        self.wins: set = set()
        self.capped = False
        self._enumerate(seed_state, cap)
        self.dist = {} if self.capped else self._distances()

    def _enumerate(self, seed_state, cap: int) -> None:
        panel, succ, wins = self.panel, self.succ, self.wins
        seen = {seed_state}
        queue = deque([seed_state])
        while queue:
            cur = queue.popleft()
            if panel.won(cur):
                wins.add(cur)
            edges = {}
            for di in range(4):
                nxt, breaks = panel.step(cur, di)
                if nxt is None or breaks:
                    continue
                edges[di] = nxt
                if nxt not in seen:
                    if len(seen) >= cap:
                        self.capped = True
                        return
                    seen.add(nxt)
                    queue.append(nxt)
            succ[cur] = edges

    def _distances(self) -> dict:
        rev: dict = {}
        for k, edges in self.succ.items():
            for nk in edges.values():
                rev.setdefault(nk, []).append(k)
        dist = {s: 0 for s in self.wins}
        queue = deque(self.wins)
        while queue:
            k = queue.popleft()
            for p in rev.get(k, ()):
                if p not in dist:
                    dist[p] = dist[k] + 1
                    queue.append(p)
        return dist

    def get(self, state) -> "int | None":
        return self.dist.get(state)

    def optimal(self, state) -> list:
        """``[(direction index, successor), ...]`` for every press on a shortest
        path from ``state``, in ``DIRS`` order (which is what makes a re-derived
        plan byte-identical across processes).

        Exact and measured: a press is optimal iff its successor is one step
        nearer a win, and both distances come out of the same exhaustive field."""
        rest = self.dist.get(state)
        if not rest:                      # None (unreachable/dead) or 0 (won)
            return []
        return [(di, nxt) for di, nxt in sorted(self.succ.get(state, {}).items())
                if self.dist.get(nxt) == rest - 1]

    def plan(self, state) -> "Plan | None":
        rest = self.dist.get(state)
        if rest is None:
            return None
        presses, optsets = [], []
        cur = state
        for _ in range(rest):
            best = self.optimal(cur)
            if not best:                  # a labelled state always has a step
                return None
            presses.append(DIRS[best[0][0]])
            optsets.append([DIRS[di] for di, _ in best])
            cur = best[0][1]
        return Plan(presses, optsets)


# ---------------------------------------------------------------------------
# The expert
# ---------------------------------------------------------------------------

class SweetHintsExpert(PSExpert):
    """`PSExpert`'s plan memo and snapshot discipline around a `_Field`.

    The base class keeps the memo, the level scoping and the disk cache; only the
    strategy underneath changes, the way `PSEnumExpert` replaces it. Here the
    strategy is "read the live board and look the state up in an exhaustive
    distance field", so `heuristic` is never called and asserts rather than
    returning a number nothing would use.

    `read` rebuilds the panel from the ENGINE's current grid every time, and it
    is exact at any point in an episode rather than only at a level start. The
    one field that is not simply sitting there is ``floor0`` -- a square the line
    covers has had its floor stripped -- and it is recovered exactly, because the
    squares that ever had floor are precisely ``space + target + line``: an
    unvisited square still has its floor, a covered one carries a line segment,
    a retreated one has been given `space` back, and the head has both the player
    and its floor.

    ``targets`` is read the same way, which makes a broken board self-reporting:
    if a level were ever edited so the line could run OFF the target, the target
    object would be gone from the grid, the panel would have no targets, the
    field no wins, and `plan` would return None rather than a wrong answer.
    """

    directions = list(DIRS)
    #: `_key` is the line and the targets only, which is canonical WITHIN a level
    #: but not across them -- the walls, the floor and the constraint are static
    #: per level yet differ between levels.
    scope_by_level = True
    plan_cache_path = PLAN_CACHE

    #: Runaway guard on one level's enumeration; see `_Field`. The shipped levels
    #: reach 6449.
    field_cap: int = 4_000_000

    def setup(self) -> None:
        g = self.g
        idx = g.obj_name_to_idx
        self.player_ids = set(self.game._engine._player_indices)
        self.sep_ids = set(g.resolve_object_name("wall")) | {idx["border"]}
        self.space_ids = {idx["space"]}
        self.target_ids = set(g.resolve_object_name("target"))
        self.dot_ids = {idx["dot"]}
        #: line object id -> the (dr, dc) it points at, i.e. towards the cell's
        #: SUCCESSOR on the path. `read` walks these backwards to rebuild the line.
        self.line_step = {idx["linel"]: (0, -1), idx["liner"]: (0, 1),
                          idx["lineu"]: (-1, 0), idx["lined"]: (1, 0)}
        self.square_ids = {idx["sq1"]: 1, idx["sq2"]: 2, idx["sq3"]: 3}
        self.star_ids = {idx["st1"]: 1, idx["st2"]: 2, idx["st3"]: 3}
        #: `_Panel`s and `_Field`s by the panel's STATIC signature, not by level
        #: index, so every state of a level shares one enumeration.
        self._panels: dict = {}
        self._fields: dict = {}

    def heuristic(self, eng) -> int:
        raise AssertionError(
            "SweetHintsExpert reads an exhaustive distance field; "
            "heuristic is unused")

    # -- engine <-> model ----------------------------------------------------
    def read(self, eng):
        """``(panel, path)`` for the engine's current grid, or ``(panel, None)``
        when there is no player on it.

        Panels are cached by their static signature, so the neighbour tables are
        built once per level however many states are read from it."""
        h, w = eng.height, eng.width
        sep, targets, dots = [], [], []
        floor0: set = set()
        squares, stars, succ = {}, {}, {}
        head = None
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                i = r * w + c
                if cell & self.sep_ids:
                    sep.append(i)
                if cell & self.space_ids or cell & self.target_ids:
                    floor0.add(i)
                if cell & self.target_ids:
                    targets.append(i)
                if cell & self.dot_ids:
                    dots.append(i)
                if cell & self.player_ids:
                    head = i
                for o in cell:
                    if o in self.square_ids:
                        squares[i] = self.square_ids[o]
                    if o in self.star_ids:
                        stars[i] = self.star_ids[o]
                    if o in self.line_step:
                        floor0.add(i)                 # a covered square HAD floor
                        dr, dc = self.line_step[o]
                        succ[i] = (r + dr) * w + (c + dc)
        sig = (h, w, tuple(sorted(sep)), tuple(sorted(floor0)),
               tuple(sorted(targets)), tuple(sorted(dots)),
               tuple(sorted(squares.items())), tuple(sorted(stars.items())))
        panel = self._panels.get(sig)
        if panel is None:
            panel = self._panels[sig] = _Panel(h, w, sep, floor0, targets, dots,
                                               squares, stars)
        if head is None:
            return panel, None
        pred = {v: k for k, v in succ.items()}
        path = [head]
        while path[-1] in pred:
            path.append(pred[path[-1]])
        path.reverse()
        return panel, tuple(path)

    def field(self, panel: _Panel, state) -> _Field:
        got = self._fields.get(panel.sig)
        if got is None:
            got = self._fields[panel.sig] = _Field(panel, state, self.field_cap)
        return got

    def _key(self, eng) -> frozenset:
        """``(row, col, tag)`` triples for the line (its index along the path)
        and the live targets.

        Built from the MODEL's reading rather than from raw object ids, so the
        key is exactly the state -- which is also what the ``plan_cache_path``
        signature is stored as, so a cached plan is matched against the board it
        was solved from and an edited level is a miss rather than a wrong plan.
        The targets are in it because losing one is a real (and irreversible)
        change of state that the path alone does not show."""
        panel, path = self.read(eng)
        w = panel.w
        if path is None:
            return frozenset()
        return frozenset(
            [(i // w, i % w, step) for step, i in enumerate(path)]
            + [(t // w, t % w, -1) for t in panel.targets])

    def _search(self, eng) -> "Plan | None":
        panel, path = self.read(eng)
        if path is None or not panel.targets:
            return None                   # no player, or the target is gone
        if panel.won(path):
            return Plan([], [])
        return self.field(panel, path).plan(path)


# ---------------------------------------------------------------------------
# The solver
# ---------------------------------------------------------------------------

class SweetHintsSolver(PSAStarSolver):
    game_id = "puzzlescript_sweet_hints"
    game_name = GAME_NAME
    expert_cls = SweetHintsExpert

    #: `games/ps:sweet_hints/ps:sweet_hints.py` is a plain passthrough -- it
    #: builds the adapter and nothing else, and the rendering fixes are in the
    #: .txt, which both paths read. Set this if that wrapper ever grows a patch.
    game_module_id = ""

    #: Unused: `SweetHintsExpert._search` never calls `_astar`. Left at the base
    #: values so nothing reads a lie off them; `SweetHintsExpert.field_cap` is the
    #: knob that actually bounds the enumeration.
    weight = 1
    node_cap = 400_000

    #: Room for the longest plan (25 presses on level 4) plus the re-plans an
    #: epsilon detour costs. The adapter's own 200-press per-level budget is
    #: separate and is reset by the `set_level` that ends the exploration prefix,
    #: so the plan starts it from zero.
    max_steps = 120

    #: Every level is solved; nothing is skipped.
    skip_levels = frozenset()

    #: On top of the RESET prefix: roughly one press in eight is a random legal
    #: alternative, after which the expert re-plans from wherever it landed --
    #: the taken action is the mistake and ``optimal`` is the recovery.
    #:
    #: It carries more weight in this game than in most: every shipped level's
    #: shortest plan is FORCED (exactly one state in the whole game has two
    #: equally shortest presses), so a corpus without detours would never show a
    #: wrong press or the retreat that undoes one -- and the retreat is half the
    #: mechanic. Safe at this rate because the game
    #: is reversible, so a detour can never brick a board; `record_level`'s own
    #: winnability probe is consequently never the thing that rejects one.
    epsilon = 0.12

    def prepare_expert(self, game, expert) -> None:
        """Enumerate every level's field before `discover_solvable` asks for it --
        the same work either way, but it fills the disk cache in one pass and
        makes the startup cost visible as startup."""
        for level in range(game.n_levels):
            game.set_level(level)
            expert.plan(game._engine, level)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _new(seed: int = 0):
    """The solver, its adapter and an expert -- without planning anything.

    The reports below each want a different slice (``--audit`` needs no plan at
    all, ``--selfcheck`` needs only the model), so planning is left to whoever
    asks for it rather than paid on every entry point."""
    solver = SweetHintsSolver()
    game = solver.make_game(seed)
    return solver, game, SweetHintsExpert(game, node_cap=solver.node_cap)


def _start(game, expert, level: int):
    """``(panel, path)`` at a level's start, engine left seated there."""
    game.set_level(level)
    return expert.read(game._engine)


def _ascii(panel: _Panel, path) -> str:
    """A model state as ASCII, so a failing check can print the board it failed
    on rather than a tuple of integers."""
    body = set(path[:-1]) if path else set()
    out = []
    for r in range(panel.h):
        line = ""
        for c in range(panel.w):
            i = r * panel.w + c
            if path and i == path[-1]:
                line += "@"
            elif i in body:
                line += "*"
            elif i in panel.sep:
                line += "#"
            elif i in panel.targets:
                line += "T"
            elif i in panel.dots:
                line += "o"
            elif i in panel.squares:
                line += "123"[panel.squares[i] - 1]
            elif i in panel.stars:
                line += "abc"[panel.stars[i] - 1]
            elif i in panel.floor0:
                line += "."
            else:
                line += " "
        out.append(line)
    return "\n".join(out)


def _direction(panel: _Panel, a: int, b: int) -> str:
    """The press that steps from cell ``a`` to adjacent cell ``b``."""
    dr, dc = (b // panel.w) - (a // panel.w), (b % panel.w) - (a % panel.w)
    return next(d for d, delta in _DELTA.items() if delta == (dr, dc))


def _plans(verbose: bool = True) -> int:
    """Every level's plan, replayed through the REAL interpreter -- the
    end-to-end test of the native model, the field and the tie labelling at once
    -- plus the three structural facts the solver leans on."""
    _solver, game, expert = _new()
    eng = game._engine
    total = ties = bad = 0
    for level in range(game.n_levels):
        panel, start = _start(game, expert, level)
        t0 = time.time()
        plan = expert.plan(eng, level)
        field = expert.field(panel, start)
        elapsed = time.time() - t0

        # The three assertions the module docstring makes about these levels.
        one_target = len(panel.targets) == 1
        dead_end = all(sum(1 for di in range(4)
                           if panel.edge[t][di] >= 0
                           and panel.edge[t][di] in panel.floor0) <= 1
                       for t in panel.targets)
        no_stars = not panel.stars
        bad += (not one_target) + (not dead_end)

        if plan is None:
            print(f"  L{level:2d}: UNSOLVED")
            bad += 1
            continue
        for direction in plan:
            eng.step(direction)
        won = eng.check_win()
        bad += not won
        tie_steps = sum(1 for s in plan.optsets if len(s) > 1)
        off = sum(1 for s in field.dist if len(field.optimal(s)) > 1)
        total += len(plan)
        ties += tie_steps
        room = "ok" if len(plan) < game._max_steps else "OVER BUDGET"
        if verbose:
            print(f"  L{level:2d}: {panel.h:2d}x{panel.w:2d}  "
                  f"{len(panel.floor0):2d} floor  {len(panel.dots)} dot(s)  "
                  f"{len(panel.squares)} square(s)  {len(panel.stars)} star(s)  "
                  f"| {len(field.succ):5d} states ({len(field.wins)} won, "
                  f"{len(field.succ) - len(field.dist)} that cannot win) in "
                  f"{elapsed:.2f}s  | {len(plan):3d} presses  win={won}  "
                  f"(budget {game._max_steps}, {room})  {tie_steps} plan step(s) "
                  f"with a tie set, {off} state(s) with one")
            print(f"        one target: {one_target}   target is a dead end "
                  f"(so it can never be walked off and destroyed): {dead_end}   "
                  f"stars: {'none' if no_stars else 'PRESENT'}")
    print(f"  {total} presses, {ties} of them with a second equally-right answer")
    print("  every plan reaches a WIN" if not bad else f"  {bad} PROBLEM(S)")
    return 1 if bad else 0


def _forward(panel: _Panel, state, limit: int) -> "int | None":
    """Presses to a win from ``state``, by a plain FORWARD BFS, or None past
    ``limit``.

    Shares nothing with `_Field` but `_Panel` itself, which is the point: it is
    the independent answer the optimality claims below are checked against, and
    it uses neither reversibility nor the backward sweep."""
    if panel.won(state):
        return 0
    seen = {state}
    queue = deque([(state, 0)])
    while queue:
        cur, d = queue.popleft()
        if d >= limit:
            continue
        for di in range(4):
            nxt, breaks = panel.step(cur, di)
            if nxt is None or breaks or nxt in seen:
                continue
            if panel.won(nxt):
                return d + 1
            seen.add(nxt)
            queue.append((nxt, d + 1))
    return None


def _component(panel: _Panel, state) -> set:
    """Every state reachable from ``state``, by the same rule `_Field` uses."""
    seen = {state}
    queue = deque([state])
    while queue:
        cur = queue.popleft()
        for di in range(4):
            nxt, breaks = panel.step(cur, di)
            if nxt is not None and not breaks and nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return seen


def _selfcheck(walk_presses: int = 600, samples: int = 120,
               verbose: bool = True) -> int:
    """Six things the expert would otherwise be trusted on.

    0. **ACTION is a no-op.** `noaction` in the prelude and no `action` on any
       left-hand side is exactly the argument ps:ouroboros falsified, so the
       press is measured instead: from every state of a random walk, an ACTION
       press must leave the objects, the win flag AND the rendered frame
       untouched. (It is excluded from `DIRS`, so a real effect would mean the
       expert had been planning in the wrong action space.)
    1. **The native model is the interpreter's.** A seeded random walk on every
       level, comparing `_Panel`'s path and win verdict against the engine's grid
       after EVERY press -- including the presses that do nothing, which is where
       a `cancel` mistake hides.
    2. **The constraint is the interpreter's, on every board that can test it.**
       `_Panel.won` is a re-implementation of a dozen error rules, and a random
       walk almost never arrives at a target, let alone at a failing one. So
       every state of the whole reachable space whose head is a target square is
       replayed on the interpreter and `check_win` compared -- 146 of them on
       level 5, most of which FAIL the separation rule, and 14 on level 4.
    3. **The field's seed state does not matter.** The component enumerated from
       the level start must be exactly the component enumerated from each of
       ``samples`` states reached by random presses -- WON states included, which
       is why `_Field` expands them. That is the undirectedness the whole design
       rests on, stated as a set equality.
    4. **The plans are shortest.** A plain forward BFS to the first win, which is
       shortest by construction and shares no code with the backward sweep, must
       return the field's plan length on every level.
    5. **The tie sets are exact, and recovery answers from ANY board.** Every
       step of every level's plan has its optimal set re-derived by brute force
       (a fresh depth-bounded forward BFS from each of the four successors); then
       from ``samples`` states reached by random presses, the field must return a
       plan whose length that same brute force agrees with and which WINS when
       replayed through the interpreter.
    """
    from adapters.puzzlescript_adapter import _render_frame            # noqa: PLC0415

    _solver, game, expert = _new()
    eng = game._engine
    bad = 0

    for level in range(game.n_levels):
        panel, state = _start(game, expert, level)
        rng = random.Random(f"sweet_hints:action:{level}")
        probes = changed = 0
        for _ in range(500):
            before = snapshot(eng)
            frame = np.asarray(_render_frame(eng, game._game)).copy()
            won = eng.check_win()
            eng.step("action")
            probes += 1
            changed += (eng.grid != before or eng.check_win() != won
                        or not np.array_equal(
                            np.asarray(_render_frame(eng, game._game)), frame))
            restore(eng, before)
            eng.step(DIRS[rng.randrange(4)])
            if eng.check_win():
                game.set_level(level)
        bad += changed
        if verbose:
            print(f"  L{level:2d}: ACTION changed the board on {changed} of "
                  f"{probes} live states")

    for level in range(game.n_levels):
        panel, state = _start(game, expert, level)
        rng = random.Random(f"sweet_hints:model:{level}")
        drift = 0
        for _ in range(walk_presses):
            di = rng.randrange(4)
            nxt, breaks = panel.step(state, di)
            eng.step(DIRS[di])
            expect = state if (nxt is None or breaks) else nxt
            _p, live = expert.read(eng)
            if live != expect or eng.check_win() != panel.won(expect):
                drift += 1
                print(f"    L{level} DIVERGES on {DIRS[di]}:\n"
                      f"{_ascii(panel, live)}")
                break
            state = expect
            if eng.check_win():
                game.set_level(level)
                _p, state = expert.read(eng)
        bad += drift
        if verbose:
            print(f"  L{level:2d}: model {'MATCHES' if not drift else 'DIVERGES from'}"
                  f" the interpreter over {walk_presses} random presses")

    for level in range(game.n_levels):
        panel, start = _start(game, expert, level)
        arrivals = [s for s in _component(panel, start) if s[-1] in panel.targets]
        wrong = 0
        for path in arrivals:
            game.set_level(level)
            for a, b in zip(path, path[1:]):
                eng.step(_direction(panel, a, b))
            _p, live = expert.read(eng)
            if live != path or eng.check_win() != panel.won(path):
                wrong += 1
        bad += wrong
        if verbose:
            print(f"  L{level:2d}: {len(arrivals):3d} boards with the head on the "
                  f"target ({sum(1 for p in arrivals if panel.won(p))} winning) "
                  f"-- {wrong} verdict(s) the interpreter disagrees with")

    for level in range(game.n_levels):
        panel, start = _start(game, expert, level)
        truth = _component(panel, start)
        rng = random.Random(f"sweet_hints:component:{level}")
        differs = 0
        for _ in range(samples):
            game.set_level(level)
            for _ in range(rng.randrange(1, 26)):
                eng.step(DIRS[rng.randrange(4)])
                if eng.check_win():
                    break
            _p, here = expert.read(eng)
            if _component(panel, here) != truth:
                differs += 1
        bad += differs
        if verbose:
            print(f"  L{level:2d}: component from the start is {len(truth)} "
                  f"states; {differs} of {samples} random states disagree")

    for level in range(game.n_levels):
        panel, start = _start(game, expert, level)
        plan = expert.plan(eng, level)
        shortest = _forward(panel, start, len(plan) + 1 if plan else 200)
        ok = plan is not None and shortest == len(plan)
        bad += not ok
        if verbose:
            print(f"  L{level:2d}: field {len(plan) if plan else None} presses vs "
                  f"forward BFS {shortest} -- "
                  f"{'SHORTEST' if ok else 'NOT SHORTEST'}")

    for level in range(game.n_levels):
        panel, start = _start(game, expert, level)
        plan = expert.plan(eng, level)
        mismatched = 0
        cur = start
        for i, direction in enumerate(plan):
            rest = len(plan) - i
            truth = []
            for di in range(4):
                nxt, breaks = panel.step(cur, di)
                if nxt is None or breaks:
                    continue
                if _forward(panel, nxt, rest - 1) == rest - 1:
                    truth.append(DIRS[di])
            if sorted(truth) != sorted(plan.optsets[i]):
                mismatched += 1
            cur = panel.step(cur, DIRS.index(direction))[0]
        bad += mismatched

        rng = random.Random(f"sweet_hints:recovery:{level}")
        wrong = dead = won_after = 0
        for _ in range(samples):
            game.set_level(level)
            for _ in range(rng.randrange(1, 26)):
                eng.step(DIRS[rng.randrange(4)])
                if eng.check_win():
                    break
            if eng.check_win():
                continue                 # the walk already won; nothing to plan
            _p, here = expert.read(eng)
            found = expert._search(eng)
            truth = _forward(panel, here, panel.n * 4)
            if found is None:
                dead += 1
                wrong += 1               # reversibility says this cannot happen
                print(f"    L{level} DEAD board (forward BFS says {truth}):\n"
                      f"{_ascii(panel, here)}")
                continue
            if truth != len(found):
                wrong += 1
                continue
            for direction in found:
                eng.step(direction)
            if eng.check_win():
                won_after += 1
            else:
                wrong += 1
        bad += wrong
        if verbose:
            print(f"  L{level:2d}: {len(plan)} plan tie set(s) "
                  f"{'match' if not mismatched else f'{mismatched} MISMATCH'} "
                  f"brute force; {samples} random boards -- {won_after} "
                  f"re-planned to a WIN in the interpreter, {dead} with no plan, "
                  f"{wrong} WRONG")
    return bad


# ---------------------------------------------------------------------------
# The interpreter proof
# ---------------------------------------------------------------------------

def _engine_key(eng, expert):
    """``(path, live targets)`` straight off the engine grid, via `read`.

    The targets are in the key because losing one is a state change the path
    cannot show -- and it is the only irreversible thing this game can do, so an
    enumeration that keyed on the path alone would silently merge a live board
    with a bricked one."""
    panel, path = expert.read(eng)
    return (path, panel.targets)


def _engine(levels=None, verbose: bool = True) -> int:
    """The proof that owes nothing to the native model: the WHOLE reachable
    space, enumerated by pressing actual buttons.

    A depth-first walk that needs no snapshots and no decoder, because the game
    itself provides the undo: every press that changes the board is walked BACK
    with the opposite press, and the board it lands on is required to be the one
    it left. So the enumeration is also the measurement of the reversibility
    claim `_Field` rests on -- one assertion per edge, ~26k of them across the
    five levels -- and the states are keyed by what is on the grid, with the goal
    tested by `check_win`. No `_Panel` call of any kind participates in the
    walk.

    Then a backward BFS from the boards `check_win` accepted gives the exact
    distance to a win for every state the interpreter admits, and the plan and
    the tie sets read off it are compared against the field's. The state SET is
    compared too, which is the check that the model's refusal to expand a
    target-breaking press (and its treatment of winning boards as terminal) is
    not quietly hiding part of the game.

    It is slow and it is meant to be run rarely: the interpreter floods `sqmark`
    and six star marks over the whole board on EVERY press, which puts level 5 at
    ~90 presses/s, so its 6449 states take 410 s (the other four take 12 s
    between them). Pass level numbers to restrict it."""
    _solver, game, expert = _new()
    eng = game._engine
    bad = 0
    for level in (range(game.n_levels) if levels is None else levels):
        panel, start_state = _start(game, expert, level)
        plan = expert.plan(eng, level)
        game.set_level(level)
        t0 = time.time()

        start = _engine_key(eng, expert)
        seen = {start}
        succ: dict = {}
        wins: set = set()
        presses = irreversible = 0
        # (state, next direction to try, the direction that got us here)
        stack = [[start, 0, -1]]
        while stack:
            cur, di, back = stack[-1]
            if di == 4:
                stack.pop()
                if back >= 0:
                    eng.step(DIRS[_OPPOSITE[back]])
                    presses += 1
                    irreversible += _engine_key(eng, expert) != stack[-1][0]
                continue
            stack[-1][1] = di + 1
            eng.step(DIRS[di])
            presses += 1
            nxt = _engine_key(eng, expert)
            if nxt == cur:
                continue                        # the interpreter cancelled it
            succ.setdefault(cur, {})[di] = nxt
            if eng.check_win():
                wins.add(nxt)
            if nxt in seen:
                eng.step(DIRS[_OPPOSITE[di]])
                presses += 1
                irreversible += _engine_key(eng, expert) != cur
                continue
            seen.add(nxt)
            stack.append([nxt, 0, di])

        rev: dict = {}
        for k, edges in succ.items():
            for nk in edges.values():
                rev.setdefault(nk, []).append(k)
        dist = {s: 0 for s in wins}
        queue = deque(wins)
        while queue:
            k = queue.popleft()
            for p in rev.get(k, ()):
                if p not in dist:
                    dist[p] = dist[k] + 1
                    queue.append(p)

        epresses, esets = [], []
        cur = start
        while dist.get(cur):
            rest = dist[cur]
            best = [(di, nk) for di, nk in sorted(succ[cur].items())
                    if dist.get(nk) == rest - 1]
            epresses.append(DIRS[best[0][0]])
            esets.append(sorted(DIRS[di] for di, _ in best))
            cur = best[0][1]

        model_states = _component(panel, start_state)
        same_set = {s for s, _t in seen} == model_states
        same_len = len(epresses) == len(plan)
        same_sets = esets == [sorted(s) for s in plan.optsets]
        bad += irreversible + (not same_set) + (not same_len) + (not same_sets)
        if verbose:
            print(f"  L{level:2d}: {len(seen):5d} engine states / "
                  f"{len(model_states):5d} model states -- "
                  f"{'SAME SET' if same_set else 'DIFFERENT'}; "
                  f"{len(epresses):3d} presses vs the field's {len(plan):3d} -- "
                  f"{'AGREE' if same_len else 'DISAGREE'}; tie sets "
                  f"{'AGREE' if same_sets else 'DISAGREE'}; "
                  f"{irreversible} irreversible press(es) of {presses} "
                  f"({time.time() - t0:.1f}s)")
        game.set_level(level)
    return bad


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

#: Every cell stack a board of this game can hold. Background is on its own layer
#: and is under everything; `wall`/`border` share the second; `space`/`target` the
#: third; the coloured squares, the stars and `dot` have one each; the player and
#: the four line segments share one, so at most one of those is ever in a cell.
#: `error` sits above them all and is painted only on a dot, a square or a star.
#: The invisible bookkeeping layers (`endmark`, `sqmark`, `amark`..`fmark`) are
#: `transparent` and are left out.
_COMPOSITIONS = [
    (), ("wall",), ("border",), ("space",),
    ("targetu",), ("targetr",), ("targetl",), ("targetd",),
    ("dot",), ("dot", "space"),
    ("sq1",), ("sq2",), ("sq3",), ("st1",), ("st2",), ("st3",),
    ("liner",), ("linel",), ("lineu",), ("lined",),
    ("liner", "dot"), ("linel", "dot"), ("lineu", "dot"), ("lined", "dot"),
    ("players", "space"), ("playerr", "space"), ("playerl", "space"),
    ("playeru", "space"), ("playerd", "space"),
    ("players", "targetu"), ("players", "targetr"),
    ("players", "targetl"), ("players", "targetd"),
    ("playerr", "targetr"), ("playerl", "targetl"),
    ("playeru", "targetu"), ("playerd", "targetd"),
    ("error", "dot", "space"), ("error", "sq1"), ("error", "sq2"),
    ("error", "sq3"), ("error", "st1"), ("error", "st2"), ("error", "st3"),
]

#: Compositions that are DELIBERATELY the same picture, with the reason.
#:
#: * `wall` / `border` -- one object mechanically. They share a collision layer,
#:   no rule names either except through `sep`, and no win condition mentions
#:   them, so nothing in this game can tell them apart. Drawing them apart would
#:   be inventing a distinction that carries no information; see the .txt header.
#: * the five `playerX` -- one object with five pictures, chosen by a rule ORDER
#:   that no rotation can carry. The facing decides nothing (it is a function of
#:   the neighbouring lines, which are on screen and now carry the direction
#:   themselves), so it is drawn invariant. Again, see the .txt header.
_MERGED = [
    {("wall",), ("border",)},
    {("players", "space"), ("playerr", "space"), ("playerl", "space"),
     ("playeru", "space"), ("playerd", "space")},
    {("players", "targetu"), ("playeru", "targetu")},
    {("players", "targetr"), ("playerr", "targetr")},
    {("players", "targetl"), ("playerl", "targetl")},
    {("players", "targetd"), ("playerd", "targetd")},
]

#: The direction relabel the presentation group applies to a directional object.
#: ``np.rot90(frame)`` turns the picture COUNTER-clockwise, so one application
#: sends what pointed up to pointing left.
_CCW = {"u": "l", "l": "d", "d": "r", "r": "u"}
_HFLIP = {"u": "u", "d": "d", "l": "r", "r": "l"}
_VFLIP = {"u": "d", "d": "u", "l": "l", "r": "r"}
_FAMILIES = ("line", "target", "player")


def _comp_name(comp) -> str:
    return "+".join(comp) or "bare background"


def _relabel(comp, k: int, hflip: bool, vflip: bool) -> tuple:
    """``comp`` as the presentation group's transform relabels it: a `lineR` seen
    through a quarter turn IS a `lineD`, and a `targetU` IS a `targetL`."""
    out = []
    for name in comp:
        fam = next((f for f in _FAMILIES
                    if name.startswith(f) and name[len(f):] in "udlr"
                    and len(name) == len(f) + 1), None)
        if fam is None:
            out.append(name)
            continue
        ch = name[-1]
        for _ in range(k):
            ch = _CCW[ch]
        if hflip:
            ch = _HFLIP[ch]
        if vflip:
            ch = _VFLIP[ch]
        out.append(fam + ch)
    return tuple(out)


def _audit(verbose: bool = True) -> int:
    """Two passes over every cell COMPOSITION a board of this game can hold.

    **Pass 1 -- distinctness**, at every cell size the five boards use (5 and 7
    px; the size list is derived from the levels rather than assumed, so an
    edited level is covered). Whole 64x64 frames of uniform boards are compared
    rather than one cell out of a mixed board: `_render_frame` upscales and
    centre-pads, so slicing a cell by ``cell_px`` arithmetic reads the wrong
    pixels (the ps:explod lesson). Two uniform boards render identically iff
    their cells do. The deliberate merges in `_MERGED` are asserted to hold
    rather than merely tolerated.

    This is the pass that failed on the shipped .txt in three places -- `border`
    was `Background`, the four `lineX` were one picture, and the four directional
    `playerX` were four. See the header of
    ``data/puzzlescript_games/Sweet_Hints.txt``.

    **Pass 2 -- the group.** The game is augmented with a frame rotation AND both
    flips, so a board is drawn at any of 16 presentations, and three of its
    object families ARE directional. That is only sound if the art is an exact
    ORBIT: every transform of a composition must be the art of the composition
    that transform relabels it to (a turned `lineR` must BE a `lineD`), and must
    not be the art of any other. Both halves are checked, on SQUARE boards
    because a transform of a non-square board also moves the letterbox and every
    comparison would then pass for the wrong reason.
    """
    from adapters.puzzlescript_adapter import _render_frame            # noqa: PLC0415

    _solver, game, _expert = _new()
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx

    sizes: dict = {}
    for level in range(game.n_levels):
        game.set_level(level)
        sizes.setdefault((eng.height, eng.width), []).append(level)

    def shoot(h, w, comp):
        eng.height, eng.width = h, w
        cell = {idx["background"]} | {idx[o] for o in comp}
        eng.grid = [[set(cell) for _ in range(w)] for _ in range(h)]
        eng._position_index_dirty = True
        return np.asarray(_render_frame(eng, g)).copy()

    bad = 0
    for (h, w), levels in sorted(sizes.items()):
        shots = {c: shoot(h, w, c) for c in _COMPOSITIONS}
        clashes = [(a, b) for a, b in itertools.combinations(_COMPOSITIONS, 2)
                   if np.array_equal(shots[a], shots[b])
                   and not any({a, b} <= m for m in _MERGED)]
        broken = [(a, b) for m in _MERGED for a, b in itertools.combinations(m, 2)
                  if not np.array_equal(shots[a], shots[b])]
        bad += len(clashes) + len(broken)
        if verbose:
            print(f"  {h:2d}x{w:2d} (cell {min(64 // h, 64 // w)}px, levels "
                  f"{','.join(str(x) for x in levels)}): {len(shots)} "
                  f"compositions, {len(_MERGED)} deliberate merge(s) -- "
                  f"{'OK' if not clashes and not broken else 'PROBLEM'}")
        for a, b in clashes:
            print(f"      IDENTICAL: {_comp_name(a)}  ==  {_comp_name(b)}")
        for a, b in broken:
            print(f"      MERGE BROKEN: {_comp_name(a)}  !=  {_comp_name(b)}")

    def transform(frame, k, hflip, vflip):
        if k:
            frame = np.rot90(frame, k=k)
        if hflip:
            frame = np.fliplr(frame)
        if vflip:
            frame = np.flipud(frame)
        return np.ascontiguousarray(frame)

    for cell in sorted({min(64 // h, 64 // w) for h, w in sizes}):
        n = 64 // cell
        shots = {c: shoot(n, n, c) for c in _COMPOSITIONS}
        wrong, stray = [], []
        for comp in _COMPOSITIONS:
            for k, hflip, vflip in itertools.product(range(4), (False, True),
                                                     (False, True)):
                turned = transform(shots[comp], k, hflip, vflip)
                want = _relabel(comp, k, hflip, vflip)
                if not np.array_equal(turned, shots[want]):
                    wrong.append((comp, (k, hflip, vflip), want))
                stray += [(comp, (k, hflip, vflip), other)
                          for other in _COMPOSITIONS
                          if np.array_equal(turned, shots[other])
                          and not any({want, other} <= m for m in _MERGED)
                          and other != want]
        bad += len(wrong) + len(stray)
        if verbose:
            print(f"  cell {cell}px ({n}x{n}): {len(wrong)} composition(s) whose "
                  f"transform is not the art of what the transform relabels it "
                  f"to, {len(stray)} that land on something else")
        for comp, t, want in wrong[:6]:
            print(f"      {_comp_name(comp)} under {t} != {_comp_name(want)}")
        for comp, t, other in stray[:6]:
            print(f"      {_comp_name(comp)} under {t} == {_comp_name(other)}")
    return bad


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

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


def _dynamic(eng, expert, error_id: int):
    """The mechanically meaningful content of a board: the line, the live targets
    and where the errors are.

    Deliberately NOT the raw grid. The five player pictures and the four line
    pictures are relabelled by a transform, and asserting on object identities
    would be asserting that the interpreter picks the same OBJECT rather than
    that it plays the same GAME. The errors are in it because they are what the
    region flood and the dot/square/star rules produce, and they are the half of
    the mechanic that a movement-only comparison would never touch."""
    panel, path = expert.read(eng)
    errors = frozenset((r, c) for r, row in enumerate(eng.grid)
                       for c, cell in enumerate(row) if error_id in cell)
    return path, panel.targets, errors


def _symmetry(walk_presses: int = 150, verbose: bool = True) -> int:
    """Two measurements of the 16-presentation augmentation.

    **Stage 1 -- the mechanic is equivariant.** Every level's plan AND a seeded
    random walk are replayed on all eight turned and mirrored copies of the
    level's own board, built by transforming the LEVEL LAYOUT so the interpreter
    re-runs the level-start rules and derives everything itself. After every
    press the line, the targets and the ERROR cells must be exactly where the
    transform of the reference run put them.

    This is the stage that matters, because the four movement rules are written
    as four per-direction blocks (``right [ right player | floor ] -> ...``) --
    the shape that hid ps:gobble_rush's chirality -- and because the constraint
    is decided by a flood whose propagation rule the interpreter expands per
    direction as well. Reading the blocks and declaring them copies of each other
    is exactly the argument that was wrong there. The random walk is what reaches
    the boards a shortest plan never visits: dead ends, retreats back down the
    line, and arrivals at the target that FAIL the constraint and paint errors.

    **Stage 2 -- the adapter presents what the expert planned.** Every level's
    plan driven through `PuzzleScriptAdapter.perform_action` at each of the 16
    presentations, with the presses converted by `screen_action`, requiring the
    frames to be the exact transform of the unaugmented run and the level to WIN.
    The frame equality is close to automatic (the adapter transforms the rendered
    frame rather than the board), so what this really tests is the ROTATION
    CONTRACT -- that the generator emits the screen button whose remap is the
    engine press it planned. That contract is what a copy-paste broke in
    ps:enqueue, silently, and it is what makes the recorded actions replayable.
    """
    solver, game, expert = _new()
    eng, g = game._engine, game._game
    error_id = g.obj_name_to_idx["error"]
    plans = {}
    for level in range(game.n_levels):
        game.set_level(level)
        plans[level] = expert.plan(eng, level)

    bad = 0
    for level in range(game.n_levels):
        rng = random.Random(f"sweet_hints:symmetry:{level}")
        walk = [DIRS[rng.randrange(4)] for _ in range(walk_presses)]
        layout = g.levels[level]
        hw = (len(layout), len(layout[0]))

        runs = {}
        for tag, presses in (("plan", list(plans[level])), ("walk", walk)):
            eng.load_level(layout)
            runs[tag] = ([_dynamic(eng, expert, error_id)], presses)
            for direction in presses:
                eng.step(direction)
                runs[tag][0].append(_dynamic(eng, expert, error_id))

        notes = []
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

            def tflat(i, _cell=cell, _hw=hw, _tw=tw):
                r, c = _cell(divmod(i, _hw[1]), _hw)
                return r * _tw + c

            for tag, (ref, presses) in runs.items():
                eng.load_level(turned)
                for i, direction in enumerate(presses):
                    eng.step(dmap[direction])
                    path, targets, errors = ref[i + 1]
                    want = (None if path is None else tuple(tflat(x) for x in path),
                            frozenset(tflat(x) for x in targets),
                            frozenset(cell(x, hw) for x in errors))
                    if _dynamic(eng, expert, error_id) != want:
                        notes.append(f"rot{k}{'m' if mirror else ''}:{tag}@{i}")
                        break
        bad += len(notes)
        if verbose:
            print(f"  L{level:2d}: {len(plans[level])} plan presses + "
                  f"{walk_presses} random presses x 7 presentations of the "
                  f"LAYOUT: {'SYMMETRIC' if not notes else 'CHIRAL ' + ', '.join(notes[:3])}")

    def drive(gm, level, presses):
        gm.set_level(level)
        out = [np.asarray(gm._current_frame)]
        for act in presses:
            fd = gm.perform_action(ActionInput(id=act))
            out.append(np.asarray(fd.frame[-1] if fd.frame else gm._current_frame))
        return out

    def frame_transform(frame, k, hflip, vflip):
        if k:
            frame = np.rot90(frame, k=k)
        if hflip:
            frame = np.fliplr(frame)
        if vflip:
            frame = np.flipud(frame)
        return np.ascontiguousarray(frame)

    ref, seen, done = {}, set(), set()
    for seed in range(400):
        if len(done) == 16 * game.n_levels:
            break
        gm = solver.make_game(seed)
        for level in range(gm.n_levels):
            gm.set_level(level)
            key3 = (gm._rotation_k, gm._hflip, gm._vflip)
            seen.add(key3)
            if (level, key3) in done:
                continue      # this (level, presentation) is already measured
            done.add((level, key3))
            rng = random.Random(f"sweet_hints:screen:{level}")
            walk = [_ID_TO_GAMEACTION[rng.randrange(1, 6)]
                    for _ in range(walk_presses)]
            for tag, presses in (
                    ("plan", [screen_action(d, *key3) for d in plans[level]]),
                    ("walk", [inverse_remap_action_full(a, *key3) for a in walk])):
                frames = drive(gm, level, presses)
                if tag == "plan" and gm._state != GameState.WIN:
                    print(f"    seed {seed} L{level} {key3}: plan did not win")
                    bad += 1
                slot = (level, tag)
                if slot not in ref:
                    if key3 != (0, False, False):
                        continue          # nothing to compare against yet
                    ref[slot] = frames
                    continue
                if any(not np.array_equal(frame_transform(a, *key3), b)
                       for a, b in zip(ref[slot], frames)):
                    print(f"    seed {seed} L{level} {key3}: {tag} frames are not "
                          f"the transform of the unaugmented ones")
                    bad += 1
    if verbose:
        print(f"  {len(seen)} presentations drawn, {len(done)} "
              f"(level, presentation) pairs replayed through the adapter: "
              f"{sorted(seen)}")
    return bad


if __name__ == "__main__":
    if "--plans" in sys.argv:
        sys.exit(_plans())
    if "--selfcheck" in sys.argv:
        violations = _selfcheck()
        print(f"selfcheck: {violations} violations")
        sys.exit(1 if violations else 0)
    if "--engine" in sys.argv:
        args = [int(a) for a in sys.argv[sys.argv.index("--engine") + 1:]
                if a.isdigit()]
        violations = _engine(args or None)
        print(f"engine enumeration: {violations} disagreements")
        sys.exit(1 if violations else 0)
    if "--audit" in sys.argv:
        collisions = _audit()
        print(f"audit: {collisions} problem(s)")
        sys.exit(1 if collisions else 0)
    if "--symmetry" in sys.argv:
        violations = _symmetry()
        print(f"symmetry: {violations} violations")
        sys.exit(1 if violations else 0)
    sys.exit(SweetHintsSolver.main())
