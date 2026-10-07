"""ps:sokoban_match3 -- increpare's "Match 3 Block Push": a sokoban whose crates
ANNIHILATE when three of them line up.

THE WHOLE GAME IS TWO RULES

    [ > Player | Crate ] -> [ > Player | > Crate ]        push one crate
    LATE [ Crate | Crate | Crate ] -> [ | | ]             three in a line die

and the win is `all Crate on Target`. Everything hard about it comes from the
second rule interacting with a win condition that counts crates it no longer
has.

FOUR CONSEQUENCES, none of them visible in the rules as written

* **Annihilating crates is not a trick, it is the only way to win, and an
  EMPTY board wins too.** `all X on Y` is universally quantified, and the
  interpreter's one guard against vacuity covers the PLAYER only
  (`_check_single_win_condition`), so a level with every crate deleted is won.
  Level 0 ships five crates and TWO targets, so it obviously cannot be solved
  without deleting three. Level 1 ships four crates and four targets and looks
  like an ordinary sokoban -- and is not: the exhaustive BFS in `--bfs` shows
  that NO reachable state of it has all four crates on targets, so it too has
  to annihilate three and park the survivor. Targets do not have to be covered.
  Emptying the board is reachable on both (19 presses on level 0, 27 on level
  1) and wins on both; it is simply slower than the 13 and 17 the plans take.
  Level 0's 13 is reached two different ways -- a line of three leaving two
  crates for its two targets, or a line of FOUR leaving one for either of
  them -- and the branch is its very FIRST press: the optimal set there is
  {down, right} for the three-line ending and {left} for the four-line one, all
  three offered, which is exactly the kind of choice a labelling that recorded
  only the expert's own route would teach away.

* **A whole maximal RUN dies, not three of it.** The rule deletes three, but
  the interpreter re-scans until no match is left and every overlapping window
  of a longer line matches, so a run of four, five or six vanishes entirely.
  Four crates in a row on level 1 is therefore an instant (vacuous) win, and a
  five-run on level 0 is too. `--selfcheck` is the measurement; the model that
  deletes only three mismatches the interpreter on 542 of 2000 random boards.

* **Vertical runs are resolved BEFORE horizontal ones.** The rule is written
  without a direction, so it expands to all four, and the interpreter applies
  them in the order `up, down, left, right`. On a plus/cross the column dies
  and the two horizontal arms SURVIVE, two cells apart. Swap the two passes in
  the model and it mismatches the interpreter on 607 of the same 2000 boards.
  (This is a fact about the ENGINE grid, not about the screen, so it is not a
  chirality problem for the presentation augmentation -- see `symmetry`.)

* **Nothing else happens.** No chain push (a crate shoved into a crate blocks,
  and the player with it), no pull, no gravity, and ACTION is a no-op: no rule
  reads the action force, so the button is four presses' worth of nothing. The
  expert searches four directions.

THE SEARCH: uniform-cost MACRO Dijkstra, and the plans are PROVABLY SHORTEST

A primitive-move search is what the plans are measured in, but not what finds
them: level 0's reachable space is 9.07M (player, crates) states, 25 s and
1.1 GB of dict to enumerate. A macro is `walk the shortest route to
the square behind a crate, then push it`, its cost is `walk + 1` presses, and
because nothing in this game happens except when a crate moves, the shortest
macro path IS the shortest press sequence -- there is no reason ever to take a
press that moves neither the player toward a push nor a crate. Dijkstra over
macros (no heuristic, so no admissibility to get wrong) visits 52k states in
1 s on level 0 and 1.7k on level 1, and both answers agree exactly with the
exhaustive primitive BFS: 13 and 17 presses (`--bfs` re-runs that proof).

OPTIMAL SETS ARE MEASURED, exactly. For every step of the plan and every one of
the four presses, the press is offered iff an exact bounded re-solve from the
state it lands on still wins in the presses that remain. That is a proof both
ways -- nothing optimal is omitted and nothing offered is second-best -- and it
costs ~50 bounded searches per level, which at this speed is seconds. It
matters here for more than walk interleavings: a plan spends most of its length
walking around the board between pushes, and on these open boards a walk
usually has several equally short routes.

RECOVERY is the family's RESET prefix (`recovery_mode = "reset"`). Deletion is
irreversible and a crate shoved against a wall usually is too, so a flailed
board cannot be re-planned from in general; one RESET restores exactly the
state the cached plan was solved from. The plans live in
`data/sokoban_match3_plans.json`.

ART. One sprite in `data/puzzlescript_games/sokoban_match3.txt`: Target was a
hollow darkblue ring drawn exactly under the Player's opaque middle, so a
player standing on a target was pixel-identical to a player standing on grass,
and `darkblue` is the same ARC index as the player's own `blue` anyway. It is
now a solid purple square, which shows through the player's nine transparent
pixels AND through the crate's open middle. The file's own header has the
detail; `--audit` is the regression test.

VERIFIED. Both levels, 30 presses, every one of them provably shortest, and the
proof is independent of the search that found them: `--bfs` enumerates each
level's WHOLE reachable space (9.07M and 940k states, 27 s and 1.1 GB together)
with a primitive-press BFS that knows nothing about macros, and its shortest
wins are 13 and 17, exactly the macro plans. Cold build is 2.7 s and 123 MB.
`--selfcheck` fuzzes 9600 presses over both levels (1929 exact no-ops, 731
crate moves, 57 crates annihilated) plus 4000 random crate fields (2452 of them
with a deletion) with 0 divergences from the interpreter. `--ties` re-derives
all 98 labelled presses with a primitive BFS: 0 disagreements, in both
directions. `--audit`: 7 cell compositions pairwise distinct at cell_px 7, the
only size the two boards use. `--symmetry`: 2 levels x 16 presentations, every
plan a WIN and every frame of both the plan and a 200-press random walk exactly
the transform of the unaugmented one. 40 seeds x 2 levels recorded in-process:
80/80 WIN, all 2400 frames replay exactly through the adapter from the recorded
SCREEN actions, 1200/1200 expert steps labelled (1.13 optimal presses per step,
the taken press always in its own set), indices in 0..5, no zero-expert level,
and three processes at different PYTHONHASHSEEDs write byte-identical episodes.
Longest level record is 30 actions, against the adapter's 200-press cap.

CLI
---
    --selfcheck   the native model fuzzed against the interpreter
    --audit       every cell composition renders distinctly, at every cell size
    --symmetry    every presentation is an exact transform of the unaugmented
    --plans       every level's plan, replayed through the interpreter
    --ties        every optimal-action label re-derived by a primitive BFS
    --bfs         exhaustive primitive BFS over both whole state spaces --
                  the independent optimality proof (27 s, 1.1 GB)
"""

from __future__ import annotations

import heapq
import itertools
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.puzzlescript_adapter import (        # noqa: E402
    PuzzleScriptAdapter, _render_frame)
from arcengine import ActionInput, GameState        # noqa: E402
from solvers.base_solver import _ID_TO_GAMEACTION   # noqa: E402
from solvers.common.ps_astar import (              # noqa: E402
    PSAStarSolver, PSExpert, Plan, screen_action)
from utils.rotation import inverse_remap_action_full  # noqa: E402

GAME_NAME = "sokoban_match3"
GAME_ID = "ps:sokoban_match3"

#: Engine directions, in the order the model indexes them. The pairs (0,1) and
#: (2,3) are the two AXES, and `Match3Board.settle` resolves them in that
#: order -- vertical first -- because that is the order the interpreter expands
#: an undirected rule's four directions. See the module docstring.
DIRNAMES = ("up", "down", "left", "right")
_DELTA = ((-1, 0), (1, 0), (0, -1), (0, 1))

#: A line this long or longer annihilates. Three, as advertised; named because
#: the whole of the run dies, not this many of it.
MATCH = 3

_INF = 1 << 30


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

class Match3Board:
    """One level, as a model the search can step in microseconds.

    A state is ``(player, crates)`` -- a flat cell index and a sorted tuple of
    them. Walls and targets are static and nothing else in the game moves, so
    that is exact; `selfcheck` is the proof.
    """

    def __init__(self, eng, ids):
        grid = eng.grid
        self.h, self.w = len(grid), len(grid[0])
        n = self.h * self.w
        self.wall = [False] * n
        targets, crates, player = [], [], None
        for r in range(self.h):
            for c in range(self.w):
                cell, i = grid[r][c], r * self.w + c
                if ids["wall"] in cell:
                    self.wall[i] = True
                if ids["target"] in cell:
                    targets.append(i)
                if ids["crate"] in cell:
                    crates.append(i)
                if ids["player"] in cell:
                    player = i
        self.targets = frozenset(targets)
        self.state = (player, tuple(sorted(crates)))

        # Neighbour table. ``None`` off the board -- the levels are not all
        # ringed by wall (level 1 leaves bare Background outside its walls), so
        # "is it a wall?" is not the same question as "is it on the board?".
        self.nb = [[None] * 4 for _ in range(n)]
        for r in range(self.h):
            for c in range(self.w):
                i = r * self.w + c
                for k, (dr, dc) in enumerate(_DELTA):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < self.h and 0 <= cc < self.w:
                        self.nb[i][k] = rr * self.w + cc

    # -- dynamics -----------------------------------------------------------
    def settle(self, crates) -> tuple:
        """The LATE rule, run to fixpoint: delete every maximal line of three
        or more crates, the vertical axis before the horizontal.

        Two passes are enough and a third would change nothing: the first
        leaves no vertical line of three, the second only removes crates, and
        removing crates can never lengthen a line. Both details -- whole runs,
        and this axis order -- are measured against the interpreter by
        `selfcheck`, and each is what separates this from a model that is wrong
        on a quarter of random boards.
        """
        cr = set(crates)
        for axis in ((0, 1), (2, 3)):
            doomed, seen = set(), set()
            for x in cr:
                if x in seen:
                    continue
                run = [x]                   # the maximal line through x
                seen.add(x)
                for k in axis:              # both ways along this axis
                    y = self.nb[x][k]
                    while y is not None and y in cr:
                        run.append(y)
                        seen.add(y)
                        y = self.nb[y][k]
                if len(run) >= MATCH:
                    doomed.update(run)
            cr -= doomed
        return tuple(sorted(cr))

    def step(self, st, k):
        """One press. Returns the SAME tuple value when the press does nothing
        -- into a wall, off the board, or into a crate the board refuses to
        move. ACTION is not modelled at all because no rule reads it."""
        p, crates = st
        nxt = self.nb[p][k]
        if nxt is None or self.wall[nxt]:
            return st
        if nxt in crates:
            beyond = self.nb[nxt][k]
            if beyond is None or self.wall[beyond] or beyond in crates:
                return st
            cr = [x for x in crates if x != nxt]
            cr.append(beyond)
            return (nxt, self.settle(cr))
        return (nxt, crates)

    def won(self, st) -> bool:
        """`all Crate on Target` -- including VACUOUSLY, with no crates left,
        which is how three of this game's four win conditions are reached. See
        the module docstring."""
        return self.targets.issuperset(st[1])

    # -- macros -------------------------------------------------------------
    def walk_tree(self, st):
        """BFS parent pointers from the player over the squares it may walk on:
        no wall, no crate (walking into a crate is a press that moves it, which
        is a macro, not a walk), and on the board."""
        p, crates = st
        cr = frozenset(crates)
        parent = {p: None}
        q = deque([p])
        while q:
            x = q.popleft()
            for k in range(4):
                y = self.nb[x][k]
                if y is None or self.wall[y] or y in parent or y in cr:
                    continue
                parent[y] = (x, k)
                q.append(y)
        return parent

    @staticmethod
    def route(parent, cell) -> list:
        out = []
        while parent[cell] is not None:
            cell, k = parent[cell]
            out.append(k)
        out.reverse()
        return out

    def macros(self, st):
        """Every ``walk to the square behind a crate, then push it``, as
        ``(presses, resulting state)``.

        This is the whole reason the search is affordable, and it costs nothing
        in optimality: no press in this game has any effect except moving the
        player, so a shortest press sequence is a shortest sequence of pushes
        with shortest walks between them, which is exactly what a macro path
        is. The player ends on the square the crate left, which is why a state
        needs no separate "where is the player" beyond its own first field."""
        parent = self.walk_tree(st)
        out = []
        for crate in st[1]:
            for k in range(4):
                stance = self.nb[crate][k ^ 1]
                dest = self.nb[crate][k]
                if stance is None or stance not in parent:
                    continue
                if dest is None or self.wall[dest] or dest in st[1]:
                    continue
                cr = [x for x in st[1] if x != crate]
                cr.append(dest)
                out.append((self.route(parent, stance) + [k],
                            (crate, self.settle(cr))))
        return out

    # -- search -------------------------------------------------------------
    def search(self, st, bound=None, node_cap=_INF):
        """Uniform-cost macro Dijkstra. Returns ``(presses, cost)`` for a
        shortest winning press sequence, or ``(None, None)``.

        ``bound`` stops the search once every remaining node costs more than it
        -- which is what makes the optimal-set measurement cheap, since an
        alternative press only has to be shown NOT to finish in the presses
        that are left. A win is returned when its node is POPPED, never when it
        is generated: macros have different costs, so a 3-press macro found
        from a g=45 node can beat an 11-press one found from g=40.
        """
        if self.won(st):
            return [], 0
        dist = {st: 0}
        parent = {st: None}
        pq = [(0, 0, st)]
        counter = 0
        while pq:
            g, _c, s = heapq.heappop(pq)
            if g > dist.get(s, _INF):
                continue
            if bound is not None and g > bound:
                return None, None
            if self.won(s):
                presses = []
                while parent[s] is not None:
                    s, taken = parent[s]
                    presses = taken + presses
                return presses, g
            if len(dist) > node_cap:
                return None, None
            for presses, ns in self.macros(s):
                ng = g + len(presses)
                if bound is not None and ng > bound:
                    continue
                if ng < dist.get(ns, _INF):
                    dist[ns] = ng
                    parent[ns] = (s, presses)
                    counter += 1
                    heapq.heappush(pq, (ng, counter, ns))
        return None, None

    def optimal_sets(self, plan) -> list:
        """Per-step optimal press SETS, MEASURED and exact in both directions.

        A press is offered at step ``i`` iff the state it lands on can still be
        won in ``len(plan) - i - 1`` presses, which a bounded re-solve either
        exhibits or refutes by exhausting the frontier. Nothing optimal is
        omitted and nothing offered is second best. The bound is also a proof
        that ``>= len(plan) - i - 1`` needs no checking: the plan is shortest,
        so no successor can do better than one press cheaper.
        """
        st = self.state
        out = []
        for i, taken in enumerate(plan):
            left = len(plan) - i - 1
            best = []
            for k in range(4):
                ns = self.step(st, k)
                if ns == st:                       # a press that does nothing
                    continue
                if self.won(ns):
                    if left == 0:
                        best.append(k)
                    continue
                if left and self.search(ns, bound=left)[0] is not None:
                    best.append(k)
            assert taken in best, (i, taken, best)
            out.append(best)
            st = self.step(st, taken)
        return out


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class Match3Expert(PSExpert):
    """`PSExpert`'s plan memo, restore discipline and disk cache around the
    native search. `heuristic` is never called -- the search is uniform-cost."""

    directions = list(DIRNAMES)
    plan_cache_path = (Path(__file__).resolve().parent.parent
                       / "data" / "sokoban_match3_plans.json")

    OBJECTS = ("wall", "player", "crate", "target")

    def setup(self):
        self.ids = {n: self.g.obj_name_to_idx[n] for n in self.OBJECTS}
        self.stats = {}                      # level -> (macro states, presses)

    def heuristic(self, eng):
        raise AssertionError(
            "Match3Expert plans on its native model; heuristic is unused")

    def board(self, eng) -> Match3Board:
        return Match3Board(eng, self.ids)

    def _search(self, eng):
        board = self.board(eng)
        plan, _cost = board.search(board.state, node_cap=self.node_cap)
        if plan is None:
            return None
        return Plan([DIRNAMES[k] for k in plan],
                    [[DIRNAMES[k] for k in s]
                     for s in board.optimal_sets(plan)])


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class Match3Solver(PSAStarSolver):
    game_id = GAME_ID
    game_name = GAME_NAME
    game_module_id = GAME_ID
    expert_cls = Match3Expert
    #: Both plans are under 20 presses; this only bounds the exploration prefix
    #: plus the replay, well inside the adapter's own 200-press level budget.
    max_steps = 120


# ---------------------------------------------------------------------------
# --selfcheck: the model against the interpreter
# ---------------------------------------------------------------------------

def _read_engine(eng, ids):
    """The interpreter's board, in the model's own terms."""
    player, crates = None, []
    w = len(eng.grid[0])
    for r, row in enumerate(eng.grid):
        for c, cell in enumerate(row):
            i = r * w + c
            if ids["player"] in cell:
                player = i
            if ids["crate"] in cell:
                crates.append(i)
    return (player, tuple(sorted(crates)))


def selfcheck(trials=30, steps=80, boards=4000, verbose=True):
    """Fuzz the model against the real interpreter, two ways.

    ROLLOUTS from every level start are the honest fuzz: random play walks into
    walls and into crates it cannot move (both must be exact no-ops), presses
    ACTION (which must do nothing at all), wedges crates into corners and
    occasionally lines three of them up. Guided play draws mostly from the
    presses the model believes move a crate, so the rollouts actually reach the
    deletion rule instead of spending themselves on no-ops; uniform play stays
    in because it is what covers the no-op classification, and on its own the
    guided mode would be circular.

    But rollouts from two hand-made levels barely exercise `settle`, which is
    the half of this game that is not ordinary sokoban. So the second mode
    loads RANDOM crate fields -- up to twenty crates on a 5x7 field, with the
    player sealed in a one-square pocket so the press cannot move anything --
    and compares the board the LATE rule leaves. That is where the two facts
    the model turns on are measured: a variant deleting only three of a longer
    run misses on 542 of these 4000 boards, and one resolving the horizontal
    axis first misses on 607.
    """
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    eng = game._engine
    ids = {n: game._game.obj_name_to_idx[n] for n in Match3Expert.OBJECTS}
    bad = presses = noops = moved = deleted = wins = 0

    for level in range(game.n_levels):
        for guided in (False, True):
            for t in range(trials):
                game.set_level(level)
                board = Match3Board(eng, ids)
                st = board.state
                rng = random.Random(f"{GAME_NAME}:selfcheck:{level}:{guided}:{t}")
                for _ in range(steps):
                    nexts = [board.step(st, k) for k in range(4)]
                    crateful = [k for k in range(4) if nexts[k][1] != st[1]]
                    live = [k for k in range(4) if nexts[k] != st]
                    roll = rng.random()
                    if guided and crateful and roll < 0.6:
                        k = rng.choice(crateful)
                    elif guided and live and roll < 0.95:
                        k = rng.choice(live)
                    else:
                        k = rng.randrange(5)          # 5 = ACTION, a no-op
                    before = st
                    eng.step("action" if k == 4 else DIRNAMES[k])
                    st = st if k == 4 else board.step(st, k)
                    presses += 1
                    if st == before:
                        noops += 1
                    else:
                        if st[1] != before[1]:
                            moved += 1
                        deleted += len(before[1]) - len(st[1])
                    if (_read_engine(eng, ids) != st
                            or eng.check_win() != board.won(st)):
                        bad += 1
                        if bad < 4:
                            print(f"  L{level} t{t} guided={guided}: model {st}"
                                  f" != engine {_read_engine(eng, ids)}")
                        break
                    if board.won(st):
                        wins += 1
                        break

    # -- the LATE rule on its own, over random crate fields -----------------
    h, w = 9, 9
    ids_b, ids_w = ids["crate"], ids["wall"]
    bg = game._game.obj_name_to_idx["background"]
    field = [(r, c) for r in range(3, 8) for c in range(1, 8)]
    board = None
    rng = random.Random(f"{GAME_NAME}:selfcheck:fields")
    fields = fdel = 0
    for _ in range(boards):
        crates = rng.sample(field, rng.randint(3, 20))
        grid = [[{bg} for _ in range(w)] for _ in range(h)]
        for r in range(h):
            for c in range(w):
                if r in (0, h - 1) or c in (0, w - 1) or r in (1, 2):
                    grid[r][c].add(ids_w)
        grid[1][1] = {bg, ids["player"]}          # sealed in a one-square pocket
        for (r, c) in crates:
            grid[r][c].add(ids_b)
        eng.load_level(grid)
        if board is None:
            board = Match3Board(eng, ids)
        eng.step(DIRNAMES[rng.randrange(4)])
        flat = tuple(sorted(r * w + c for (r, c) in crates))
        mine = board.settle(flat)
        live = _read_engine(eng, ids)[1]
        fields += 1
        if live != flat:
            fdel += 1
        if live != mine:
            bad += 1
            if bad < 8:
                print(f"  field fuzz: model {mine} != engine {live}")
    if verbose:
        print(f"  {presses} presses fuzzed over {game.n_levels} levels "
              f"({noops} no-ops, {moved} crate moves, {deleted} crates "
              f"annihilated, {wins} accidental wins)")
        print(f"  {fields} random crate fields settled "
              f"({fdel} of them with a deletion)")
    return bad


# ---------------------------------------------------------------------------
# --audit: every cell composition, pixel-distinct at every cell size
# ---------------------------------------------------------------------------

#: Every stack of objects a square in this game can hold. Player and Crate and
#: Wall share one collision layer, so no two of them ever meet; Target is on
#: its own layer, under all three. `wall on target` is left out because no
#: level's legend can produce it.
_AUDIT_CASES = {
    "floor": (),
    "wall": ("wall",),
    "target": ("target",),
    "crate": ("crate",),
    "crate on target": ("crate", "target"),
    "player": ("player",),
    "player on target": ("player", "target"),
}


def audit(verbose=True):
    """Assert every cell COMPOSITION renders differently at every cell size the
    levels use.

    Composition, not object: the bug this is the regression test for was in the
    STACK -- Target's ring sat exactly under the Player's opaque middle, so the
    two compositions that tell you where the targets are were one picture.

    The board is FILLED with the composition and whole frames are compared,
    rather than one cell being sliced out by ``cell_px`` arithmetic:
    `_render_frame` upscales a sub-64 render to fill the frame and then
    letterboxes it, so the cell grid in the output is not ``cell_px``-aligned
    and the crop lands in the wrong window. Rendering is per-cell independent,
    so a whole-frame comparison is the same test, done where the geometry
    cannot drift."""
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    eng, parsed = game._engine, game._game
    idx = parsed.obj_name_to_idx
    sizes = {}
    for level in range(game.n_levels):
        game.set_level(level)
        sizes.setdefault((eng.height, eng.width), []).append(level)

    bad = 0
    for (h, w), levels in sorted(sizes.items()):
        shots = {}
        for name, objs in _AUDIT_CASES.items():
            eng.height, eng.width = h, w
            eng.grid = [[{idx["background"]} | {idx[o] for o in objs}
                         for _ in range(w)] for _ in range(h)]
            eng._position_index_dirty = True
            shots[name] = np.asarray(_render_frame(eng, parsed))
        clashes = [(a, b) for a, b in itertools.combinations(shots, 2)
                   if np.array_equal(shots[a], shots[b])]
        bad += len(clashes)
        if verbose:
            print(f"  {h}x{w} (cell {64 // max(h, w)}px, levels "
                  f"{','.join(str(x) for x in levels)}): "
                  f"{len(_AUDIT_CASES)} compositions, "
                  f"{'all distinct' if not clashes else 'IDENTICAL ' + str(clashes)}")
    return bad


# ---------------------------------------------------------------------------
# --symmetry: the presentation augmentation is a true symmetry of the mechanic
# ---------------------------------------------------------------------------

def symmetry(walk_presses=200, verbose=True):
    """Replay every level through the ADAPTER at every presentation it can draw
    and require the frames to be exactly the transform of the unaugmented ones.

    This is the evidence for putting the game in `_FLIP_GAMES`. The structural
    argument is the ESCAPE! one -- gravity-free, screen-relative input, a win
    condition that names no direction, one moving body per press so nothing can
    contest a cell, and no sprite that encodes a facing. The one asymmetry the
    game does have, `settle` resolving columns before rows, is a fact about the
    ENGINE grid and the engine grid is never transformed: for a ps: game the
    adapter rotates and mirrors the rendered PICTURE and remaps the input, and
    nothing else. This measures exactly that: the same engine directions driven
    at all 16 presentations, frames compared against the unaugmented run.

    Both the PLANS and a seeded random walk are replayed -- the walk is what
    reaches the boards a plan never visits (crates wedged in corners, lines of
    four annihilated at once), and it presses ACTION too."""
    solver = Match3Solver()
    game, expert, _ = solver._ensure(0)
    plans = {}
    for level in range(game.n_levels):
        game.set_level(level)
        plans[level] = expert.plan(game._engine, level)

    def drive(g, level, presses):
        g.set_level(level)
        out = [np.asarray(g._current_frame)]
        for act in presses:
            fd = g.perform_action(ActionInput(id=act))
            out.append(np.asarray(fd.frame[-1] if fd.frame
                                  else g._current_frame))
        return out

    def transform(frame, k, hflip, vflip):
        if k:
            frame = np.rot90(frame, k=k)
        if hflip:
            frame = np.fliplr(frame)
        if vflip:
            frame = np.flipud(frame)
        return np.ascontiguousarray(frame)

    ref, seen, bad = {}, set(), 0
    for seed in range(400):
        if len(seen) == 16 and seed > 40:
            break
        g = solver.make_game(seed)
        for level in range(g.n_levels):
            g.set_level(level)
            k = (g._rotation_k, g._hflip, g._vflip)
            seen.add(k)
            rng = random.Random(f"{GAME_NAME}:symmetry:{level}")
            walk = [_ID_TO_GAMEACTION[rng.randrange(1, 6)]
                    for _ in range(walk_presses)]
            for tag, presses in (("plan", [screen_action(d, *k)
                                           for d in plans[level]]),
                                 ("walk", [inverse_remap_action_full(a, *k)
                                           for a in walk])):
                frames = drive(g, level, presses)
                if tag == "plan" and g._state != GameState.WIN:
                    print(f"  seed {seed} L{level} {k}: plan did not win")
                    bad += 1
                key = (level, tag)
                if key not in ref:
                    if k != (0, False, False):
                        continue          # nothing to compare against yet
                    ref[key] = frames
                    continue
                if any(not np.array_equal(transform(a, *k), b)
                       for a, b in zip(ref[key], frames)):
                    print(f"  seed {seed} L{level} {k}: {tag} frames are not "
                          f"the transform of the unaugmented ones")
                    bad += 1
    if verbose:
        print(f"  {len(seen)} presentations drawn: {sorted(seen)}")
    return bad


# ---------------------------------------------------------------------------
# --bfs: the independent optimality proof
# ---------------------------------------------------------------------------

def bfs_report(levels=(0, 1), verbose=True):
    """Exhaustive primitive-press BFS over the whole reachable state space, and
    the shortest win it finds compared against the macro plan.

    This is the check that the macro reformulation is not quietly losing a
    shorter answer -- it searches single presses with no notion of a macro at
    all. Level 1 is 940k reachable states and 2 s; level 0 is 9.07M and 25 s in
    1.1 GB of dict. Both agree: 13 and 17 presses, the macro plans exactly."""
    solver = Match3Solver()
    game, expert, _ = solver._ensure(0)
    eng = game._engine
    bad = 0
    for level in levels:
        game.set_level(level)
        board = expert.board(eng)
        plan = expert.plan(eng, level)
        dist = {board.state: 0}
        q = deque([board.state])
        best = None
        while q:
            s = q.popleft()
            if board.won(s):
                if best is None:
                    best = dist[s]
                continue
            for k in range(4):
                ns = board.step(s, k)
                if ns not in dist:
                    dist[ns] = dist[s] + 1
                    q.append(ns)
        ok = best == len(plan)
        bad += 0 if ok else 1
        if verbose:
            print(f"  L{level}: {len(dist)} reachable states, shortest win "
                  f"{best} presses, plan {len(plan)} -- "
                  f"{'PROVED SHORTEST' if ok else 'MISMATCH'}")
    return bad


# ---------------------------------------------------------------------------
# --ties: every optimal-action label re-derived by an independent primitive BFS
# ---------------------------------------------------------------------------

def ties_report(levels=None, verbose=True):
    """Re-derive every step's optimal SET with a bounded primitive-press BFS and
    require it to equal the label the plan carries.

    `Match3Board.optimal_sets` measures the same thing through the macro search,
    so this is the check that the macro reformulation has not quietly changed
    what "optimal" means -- the BFS here has no notion of a macro, a stance or a
    walk, it just presses all four keys breadth-first and asks whether a win is
    still `left` presses away. Agreement in BOTH directions is the claim: a
    press the labels omit and the BFS finds would be a target the corpus teaches
    is wrong, and one the labels offer and the BFS refutes would be a
    second-best press taught as optimal.

    Cheap despite the exhaustive-sounding description (a fraction of a second
    for both levels): the bound shrinks by one at every step, and an optimal
    press is found near the front of the frontier."""
    solver = Match3Solver()
    game, expert, _ = solver._ensure(0)
    eng = game._engine
    bad = checked = 0

    def wins_within(board, st, limit):
        if board.won(st):
            return True
        seen, frontier = {st}, [st]
        for _ in range(limit):
            nxt = []
            for s in frontier:
                for k in range(4):
                    ns = board.step(s, k)
                    if ns in seen:
                        continue
                    if board.won(ns):
                        return True
                    seen.add(ns)
                    nxt.append(ns)
            frontier = nxt
            if not frontier:
                break
        return False

    for level in (levels or range(game.n_levels)):
        game.set_level(level)
        board = expert.board(eng)
        plan = expert.plan(eng, level)
        sets = getattr(plan, "optsets", None) or [[p] for p in plan]
        st = board.state
        for i, taken in enumerate(plan):
            left = len(plan) - i - 1
            truth = []
            for k in range(4):
                ns = board.step(st, k)
                if ns == st:                       # a press that does nothing
                    continue
                checked += 1
                if wins_within(board, ns, left):
                    truth.append(DIRNAMES[k])
            if sorted(truth) != sorted(sets[i]):
                bad += 1
                print(f"  L{level} step {i}: labels {sorted(sets[i])} != "
                      f"BFS truth {sorted(truth)}")
            st = board.step(st, DIRNAMES.index(taken))
        if verbose:
            print(f"  L{level}: {len(plan)} steps re-derived")
    if verbose:
        print(f"  {checked} presses re-derived by primitive BFS, "
              f"{bad} disagreements")
    return bad


# ---------------------------------------------------------------------------
# --plans
# ---------------------------------------------------------------------------

def _plan_report():
    """Every level's plan, replayed through the real interpreter -- the
    end-to-end test of the model, the search and the tie labelling at once."""
    solver = Match3Solver()
    game, expert, solvable = solver._ensure(0)
    total = ties = 0
    for level in range(game.n_levels):
        game.set_level(level)
        eng = game._engine
        board = expert.board(eng)
        plan = expert.plan(eng, level)
        if plan is None:
            print(f"  L{level:2d}: UNSOLVED")
            continue
        crates = len(board.state[1])
        for direction in plan:
            eng.step(direction)
        left = len(_read_engine(eng, expert.ids)[1])
        sets = getattr(plan, "optsets", None) or [[p] for p in plan]
        step_ties = sum(len(s) - 1 for s in sets)
        total += len(plan)
        ties += step_ties
        print(f"  L{level:2d}: {eng.height}x{eng.width}, {crates} crates -> "
              f"{left} left, {len(plan):3d} presses  win={eng.check_win()}  "
              f"{step_ties:3d} tie-presses")
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
        args = [int(a) for a in sys.argv[sys.argv.index("--bfs") + 1:]
                if a.isdigit()]
        violations = bfs_report(tuple(args) if args else (0, 1))
        print(f"bfs: {violations} mismatches")
        sys.exit(1 if violations else 0)
    if "--ties" in sys.argv:
        args = [int(a) for a in sys.argv[sys.argv.index("--ties") + 1:]
                if a.isdigit()]
        violations = ties_report(args or None)
        print(f"ties: {violations} disagreements")
        sys.exit(1 if violations else 0)
    if "--plans" in sys.argv:
        _plan_report()
        sys.exit(0)
    sys.exit(Match3Solver.main())
