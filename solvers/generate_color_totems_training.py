"""Generate Phase-1 training data for the PuzzleScript game ps:color_totems
("Color Totems" by Jeff Schubbe -- a colour-algebra collection puzzle).

The harness -- the rotation contract, the trajectory recorder, the RESET-recovery
prefix and the BaseSolver plumbing -- lives in `solvers/common/ps_astar.py`,
shared with the other ps: generators. This file is the game-specific part: the
mechanic notes, a native model of the whole game, the macro beam that plans on
it, and the disk plan cache.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_color_totems",
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
presented view, so replaying the recorded actions reproduces the frames exactly.

The game
--------
A golem walks a board picking up coins and standing on the exit. Four arrow
keys and nothing else (`noaction`), and `require_player_movement` cancels any
turn the golem did not actually move in -- so there is no wait, no undo and no
way to "pass". Win is ``No Coin`` AND ``Some Player On Target``: every coin must
be gone AND the golem must be standing on the exit at that moment.

EVERYTHING is a statement about one 3-bit colour code, ``R=1 G=2 B=4``, giving
the eight colours black(0) red(1) green(2) yellow(3) blue(4) magenta(5) cyan(6)
white(7). The four mechanics are four operations on it:

  * **A coin is picked up only by a golem of EXACTLY its colour.** ``late
    [PlayerC CoinC] -> [PlayerC No CoinC]``, one rule per colour, so a red golem
    walks straight over a black coin without touching it.
  * **A pad XORs its light into the golem.** All 48 pad rules are exactly
    ``colour ^= pad_mask`` -- e.g. the red pad sends black->red, red->black,
    cyan->white, yellow->green. A plain pad is REUSABLE: it flips to a ``_2``
    twin on entry and the ``[< Player PadR_2] -> [< Player PadR]`` rule flips it
    back the moment the golem moves off, so one XOR is applied per ENTRY. A
    LIMITED pad (``LtdPad*``) applies its XOR once and becomes an inert
    ``EmptyPad`` forever. Stepping off a reusable pad and straight back on is
    therefore a legal 2-press way to apply ONE XOR without going anywhere, and
    levels 6 and 15 are unwinnable without it (see `routes`).
  * **A magic wall passes only a golem that CONTAINS its light**, i.e. iff
    ``colour & mask == mask``: the red wall passes red/yellow/magenta/white, the
    cyan wall passes only cyan and white. The black wall is the one exception --
    ``[> Player No PlayerBl | WallBl]`` blocks everything except the black
    golem, so its test is ``colour == 0``, not the (vacuous) subset test.
  * **A totem adds or removes its light from every coin BEHIND it.** ``late
    [PlayerR | TotemR | ... | CoinB] -> [PlayerR | TotemR | ... | CoinM]``:
    with the golem standing next to a totem, every coin further along that same
    line gets the totem's bit SET if the golem's colour contains it and CLEARED
    if it does not. The ellipsis crosses walls, all four directions fire at
    once, and every coin in the line is hit, not just the first. Only the red,
    green and blue totems are live (the file's cyan/yellow/magenta totem rules
    are commented out, and no level places one). Totems block movement.

The late rules run pads, then pickups, then limited pads, then totems -- so a
coin that a totem has just recoloured under the golem's feet is picked up on the
NEXT press, not this one. All of the above is asserted against the interpreter
by ``--selfcheck``.

The model (why this game is not solved by searching the interpreter)
--------------------------------------------------------------------
`PSExpert`'s engine-blackbox A* is the wrong tool here: the puzzle is a
travelling-salesman route (level 7 has 45 coins, level 17 has 272), so plans are
hundreds of presses long and a primitive search over the interpreter never gets
past the first few. `Board` / `St` / `step` re-implement the four mechanics
above in ~30 lines of Python, which is ~3 orders of magnitude faster than the
interpreter step and -- crucially -- lets the planner ask "where can I get to and
what colour would I be" as a graph question instead of by simulation.

The model is not trusted on faith. ``--selfcheck`` fuzzes it against the real
interpreter from every level start (full state compared after EVERY press, plus
the win flag), and on top of that every plan this file returns is REPLAYED on
the interpreter and discarded unless the engine itself reports the win, so a
model that drifts costs coverage rather than correctness.

The planner
-----------
Two layers, because the two hard parts are independent:

  * `routes` answers "what can the golem reach, and as what colour" with one
    Dijkstra over JOINT ``(cell, colour)`` states. Colour is part of the node, so
    a magic wall is just an edge that exists for some nodes and not others, and a
    reusable pad is just an edge that changes the colour half of the node --
    including the step-off-and-back-on cycle that no cell-only search can
    express. Limited pads are ABSORBING (they are consumable, so a route that
    crosses one ends there and the next macro continues from the new state).
    Steps cost 2, except a step onto a coin the arriving colour collects, which
    costs 1 -- so a route sweeps up coins that lie on its way instead of walking
    politely around them, which is most of what makes the big boards tractable.
  * `beam_plan` searches the ORDER of objectives, in a width-capped
    breadth-first beam over MACROS: "go collect that coin", "go step on that
    pad", "go stand next to that totem", "go to the exit". Depth is therefore
    the number of objectives rather than the number of presses, and only the
    cheapest route to each distinct OUTCOME is kept, so the branching factor is
    the number of interesting things on the board rather than 4.

    Ranking is ``1000 * coins_left + presses_so_far``: fewer coins always wins,
    ties go to the shorter route.

    Widths are tried in a ladder and the SHORTEST plan found wins; a wider beam
    is not monotonically better, it just explores a different frontier.

`alive` is what makes that greedy ranking survivable, and it is the single most
valuable fact about this game. Ranking on coins collected is exactly the wrong
instinct here, because the limited pads are ONE-SHOT: the beam spends them on
whatever it can reach soonest and only discovers twenty presses later that a
coin it walked past can no longer be met -- with nothing on screen to say so.
`alive` decides that question up front, and soundly: a coin's light can only be
changed by a totem whose LINE covers its cell, so every other bit of its colour
is frozen for the rest of the level, and the colours the golem can still become
are the current one XOR any subset of the surviving pads. Both halves
over-estimate (geometry and ordering are ignored), so a state it rejects
provably holds no win, and dropping those instead of queueing them is the
difference between solving level 10 in four seconds and not solving it at any
width -- its sixteen coins fall into four groups, each collectible in only two
of the three colour phases its two pads allow, so every ordering that collects
before transmuting strands one. Level 3 is the same trap in miniature.

Plans are engine-verified, memoized by state key, and seed-independent (levels
are fixed ASCII maps; only the PRESENTATION is augmented per seed), so seed 0
pays for every search and later seeds replay them under their own rotation/flip.
They are also written to ``data/color_totems_plans.json``, or every shard of
`parallelize_generator` would repeat the searches on every core. Delete the file
to re-derive it.

Optimal-action sets
-------------------
Most presses in this game are the golem walking across open floor, and a walk's
order is free -- so labelling one arbitrary interleaving as the only right answer
would train the policy against the truth on the majority of steps. `annotate`
labels each step of a macro with EVERY direction that stays on a shortest route
to that macro's destination, but only after proving the alternatives are
genuinely equivalent: it requires the route to be shortest at all (the
coin-preferring Dijkstra sometimes takes a longer one), and every joint node that
lies on ANY shortest route -- other than the two endpoints, which every route
shares -- to be INERT: no coin this colour would collect, no unused limited pad,
and not adjacent to a totem. Those are exactly the three things that make one
route differ from another; where any of them sits in the middle of the corridor
the step is labelled with itself alone. No step ever ships unlabelled (see the
always-emit-optimal-targets rule).

The palette adaptation
----------------------
``data/puzzlescript_games/Color_Totems.txt`` carries an ARC recolor of the
original art (colours and sprite shapes only -- no rule was touched). It has to:
the entire game is the eight colour codes, and in the original #0000AA (blue)
and #00AAAA (cyan) both quantize onto ARC index 9 for the player, the coins, the
walls AND the pads, so a cyan golem and a blue golem were the same picture. Each
code now uses an exact ARC palette hex. See the note at the top of the .txt.

The step limit
--------------
``games/ps:color_totems/ps:color_totems.py`` subclasses the adapter to raise the
per-level step budget, because `PuzzleScriptAdapter` gives every level 200
actions and flips to GAME_OVER on the 201st -- and the sweep levels need more
than that. The failure is silent (the plan replays, the frames are right, the
level just never reaches WIN), so ``--plans`` prints the budget column.

Augmentation
------------
Color Totems is gravity-free, every rule is stated over relative directions (the
totem rules fire in all four, the pad and pickup rules are same-cell, the magic
wall rules are ``>``), and its input is screen-relative, so the board's full
8-element symmetry group is a valid presentation augmentation: the game is in
`PuzzleScriptAdapter._FLIP_GAMES`, giving rotation_k in {0,1,2,3} x horizontal x
vertical flip with the matching directional action remap. No colour augmentation
(the game is not in ``_RECOLOR_GAMES`` and must not be): which colour is which IS
the puzzle, and relabelling the palette would break the XOR arithmetic the frames
are supposed to teach.

Usage (run from the repo root):
    python solvers/generate_color_totems_training.py --episodes 200 \
        --out data/training_multi_level/color_totems

    python solvers/generate_color_totems_training.py --plans      # level report
    python solvers/generate_color_totems_training.py --selfcheck  # mechanic audit
"""

from __future__ import annotations

import heapq
import json
import os
import random
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.puzzlescript_adapter import PuzzleScriptAdapter        # noqa: E402
from solvers.common.ps_astar import PSAStarSolver, PSExpert          # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent

GAME_NAME = "Color_Totems"

#: Disk cache of the per-level start plan AND its optimal-action sets. The
#: searches are seed-independent but cost minutes of beam, which every shard of
#: `parallelize_generator` would otherwise repeat on every core. Delete to
#: re-derive.
PLAN_CACHE: Path = _REPO_ROOT / "data" / "color_totems_plans.json"

#: The four keys the game binds. It declares `noaction`, so ACTION5 is not a
#: move and never appears in a plan.
DELTA: dict[str, tuple[int, int]] = {
    "up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1),
}
DIRECTIONS = list(DELTA)

#: Object-name suffix -> its colour code (R=1, G=2, B=4). ``bl`` is black (no
#: light) and ``w`` is white (all three); the other six are the singletons and
#: the pairs. Pads XOR it, magic walls test it as a subset, totems set/clear it.
_CODE: dict[str, int] = {"bl": 0, "r": 1, "g": 2, "y": 3,
                         "b": 4, "m": 5, "c": 6, "w": 7}


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

class Board:
    """The static half of a level: geometry that no rule ever changes.

    ``wall[cell]`` is ``'X'`` for an impassable cell (the brick wall, and the
    totems -- they share the player's collision layer), ``'K'`` for the black
    magic wall (which passes ONLY the black golem, the one case that is not a
    subset test) and otherwise the magic wall's colour mask.

    ``adj_totem[cell]`` is every ``(step, mask)`` such that standing on ``cell``
    puts a totem one step away, i.e. exactly the totem effects a golem standing
    there fires. Precomputing it inverts the rule once instead of scanning the
    four neighbours of every cell the golem visits."""

    __slots__ = ("h", "w", "wall", "totem", "target", "adj_totem", "totem_bits")

    def __init__(self, h, w, wall, totem, target):
        self.h, self.w, self.wall, self.totem, self.target = h, w, wall, totem, target
        self.adj_totem: dict[tuple[int, int], list] = {}
        #: cell -> the lights that could EVER be flipped on a coin sitting there,
        #: i.e. the union of the masks of the totems whose line covers it. The
        #: complement is what makes `alive` a real theorem: a coin's other bits
        #: are frozen for the rest of the level. Geometry beyond "is the stand
        #: cell a wall" is deliberately ignored -- this must OVER-estimate.
        self.totem_bits: dict[tuple[int, int], int] = {}
        for (tr, tc), mask in totem.items():
            for (dr, dc) in DELTA.values():
                stand = (tr - dr, tc - dc)
                if not (0 <= stand[0] < h and 0 <= stand[1] < w):
                    continue
                self.adj_totem.setdefault(stand, []).append(((dr, dc), mask))
                if wall.get(stand) == 'X':
                    continue                    # nothing can stand there to fire
                r, c = tr + dr, tc + dc
                while 0 <= r < h and 0 <= c < w:
                    self.totem_bits[(r, c)] = self.totem_bits.get((r, c), 0) | mask
                    r += dr
                    c += dc

    def passable(self, cell, colour) -> bool:
        if not (0 <= cell[0] < self.h and 0 <= cell[1] < self.w):
            return False
        m = self.wall.get(cell)
        if m is None:
            return True
        if m == 'X':
            return False
        if m == 'K':
            return colour == 0
        return (colour & m) == m


class St:
    """The dynamic half: where the golem is, what colour it is, which coins are
    left (and what colour each has become) and which pads are still live."""

    __slots__ = ("pos", "colour", "coins", "pads")

    def __init__(self, pos, colour, coins, pads):
        self.pos, self.colour, self.coins, self.pads = pos, colour, coins, pads

    def copy(self) -> "St":
        return St(self.pos, self.colour, dict(self.coins), dict(self.pads))

    def key(self):
        return (self.pos, self.colour, frozenset(self.coins.items()),
                frozenset(self.pads))


def read_state(eng, name_of) -> tuple[Board, St]:
    """Build ``(Board, St)`` from the interpreter's current grid."""
    h, w = len(eng.grid), len(eng.grid[0])
    wall, totem, coins, pads = {}, {}, {}, {}
    target = pos = None
    colour = 0
    for r in range(h):
        for c in range(w):
            for o in eng.grid[r][c]:
                n = name_of[o]
                if n == "wall":
                    wall[(r, c)] = 'X'
                elif n == "wallbl":
                    wall[(r, c)] = 'K'
                elif n.startswith("wall"):
                    wall[(r, c)] = _CODE[n[4:]]
                elif n.startswith("totem"):
                    totem[(r, c)] = _CODE[n[5:]]
                    wall[(r, c)] = 'X'          # totems share the player's layer
                elif n == "target":
                    target = (r, c)
                elif n.startswith("coin"):
                    coins[(r, c)] = _CODE[n[4:]]
                elif n == "emptypad":
                    pass                        # a spent limited pad is inert
                elif n.startswith("ltdpad"):
                    pads[(r, c)] = ('ltd', _CODE[n[6:]])
                elif n.startswith("pad"):
                    suffix = n[3:]
                    # PadR / PadR_2 are the same pad; the _2 twin only marks
                    # "the golem is standing on it right now".
                    pads[(r, c)] = ('reg', _CODE[suffix[:-2] if
                                                 suffix.endswith("_2") else suffix])
                elif n.startswith("player"):
                    pos, colour = (r, c), _CODE[n[6:]]
    return Board(h, w, wall, totem, target), St(pos, colour, coins, pads)


def step(board: Board, st: St, direction: str) -> bool:
    """One press, applied to ``st`` in place. Returns False when the turn was
    cancelled (``require_player_movement``: the golem did not move, so nothing
    at all happened).

    The order is the interpreter's late-rule order: pad, then pickup, then
    limited pad, then totems."""
    dr, dc = DELTA[direction]
    dest = (st.pos[0] + dr, st.pos[1] + dc)
    if not board.passable(dest, st.colour):
        return False
    st.pos = dest
    pad = st.pads.get(dest)
    if pad is not None:
        st.colour ^= pad[1]
        if pad[0] == 'ltd':
            del st.pads[dest]
    if st.coins.get(dest) == st.colour:
        del st.coins[dest]
    for (er, ec), mask in board.adj_totem.get(dest, ()):
        add = bool(st.colour & mask)
        r, c = dest[0] + 2 * er, dest[1] + 2 * ec
        while 0 <= r < board.h and 0 <= c < board.w:
            coin = st.coins.get((r, c))
            if coin is not None:
                st.coins[(r, c)] = (coin | mask) if add else (coin & ~mask)
            r += er
            c += ec
    return True


def run(board: Board, st: St, path) -> tuple[St, int]:
    """Simulate ``path`` on a COPY of ``st``. Returns ``(state, win_index)``,
    where ``win_index`` is the index of the press that won or -1.

    The win is checked after EVERY press, not just at the end: a route that
    crosses the exit after the last coin has already won, and walking on would
    silently un-win it."""
    s = st.copy()
    for i, direction in enumerate(path):
        step(board, s, direction)
        if not s.coins and s.pos == board.target:
            return s, i
    return s, -1


# ---------------------------------------------------------------------------
# The death test: a coin whose colour can never be met again
# ---------------------------------------------------------------------------

def future_colours(st: St) -> set[int]:
    """Every colour the golem could still become -- over-estimated.

    A reusable pad can be entered any number of times and a limited one once, so
    in both cases the achievable colours are ``colour`` XOR any SUBSET of the
    surviving pads' masks. Whether the golem can actually walk to those pads is
    deliberately not considered: this feeds a death test, which may only ever
    claim too FEW deaths."""
    out = {st.colour}
    for _kind, mask in st.pads.values():
        out |= {c ^ mask for c in out}
    return out


def alive(board: Board, st: St) -> bool:
    """False when some surviving coin can provably never be collected.

    This is the single most valuable fact about the game, and the beam is lost
    without it. The pads are ONE-SHOT colour changes, so a greedy collector
    spends them on the first coins it can reach and only finds out twenty
    presses later that a coin it walked past is now uncollectible -- with
    nothing on screen to say so. Level 10 is the pure case: its two limited pads
    take the golem black -> green -> cyan, and its sixteen coins split into four
    groups that are each only collectible in two of those phases, so every
    ordering that grabs coins before transmuting them strands one.

    The theorem: a coin's light can only be changed by a totem whose LINE covers
    its cell (`Board.totem_bits`) -- every other bit of its colour is frozen for
    the rest of the level. And a totem only SETS its light when the golem has
    it, only CLEARS it when the golem does not. So with ``F`` the colours the
    golem can still become, the coin is collectible only if some ``f`` in ``F``
    agrees with it on every frozen bit and differs only on bits some member of
    ``F`` can supply (to set) or lack (to clear). Both halves over-estimate --
    geometry and ordering are ignored -- so a state this rejects really does
    hold no win."""
    if not st.coins:
        return True
    colours = future_colours(st)
    settable = 0
    clearable = 0
    for f in colours:
        settable |= f
        clearable |= ~f & 7
    for cell, coin in st.coins.items():
        free = board.totem_bits.get(cell, 0)
        for f in colours:
            diff = f ^ coin
            if diff & ~free & 7:
                continue                      # disagrees on a frozen light
            if (diff & f & ~settable) or (diff & ~f & 7 & ~clearable):
                continue                      # no golem can supply/lack that bit
            break
        else:
            return False
    return True


# ---------------------------------------------------------------------------
# Routing: one Dijkstra over joint (cell, colour) states
# ---------------------------------------------------------------------------

def routes(board: Board, st: St):
    """Cheapest route from the golem to every reachable ``(cell, colour)``.

    Colour is HALF THE NODE, which is what makes this the whole navigation
    problem in one pass: a magic wall is an edge that exists for some nodes and
    not others, and a reusable pad is an edge that flips a bit of the node. In
    particular the two-press cycle "step off a reusable pad, step back on" is an
    ordinary edge pair here, and it is the only way to change colour on levels 6
    and 15 -- both are unwinnable to a planner that treats a pad as somewhere you
    pass through once.

    LIMITED pads are absorbing: they are consumed on entry, so a route that
    steps on one ends there and the next macro continues from the new state.

    A step costs 2, or 1 onto a coin the arriving colour picks up, so among
    equal-length routes the one that sweeps coins wins. `annotate` checks
    afterwards whether the chosen route was also SHORTEST before it labels any
    ties."""
    coins, pads = st.coins, st.pads
    start = (st.pos, st.colour)
    dist = {start: 0}
    parent = {start: None}
    queue = [(0, start)]
    while queue:
        d0, node = heapq.heappop(queue)
        if d0 != dist.get(node):
            continue
        cell, colour = node
        here = pads.get(cell)
        if node != start and here is not None and here[0] == 'ltd':
            continue
        for name, (dr, dc) in DELTA.items():
            nxt = (cell[0] + dr, cell[1] + dc)
            if not board.passable(nxt, colour):
                continue
            pad = pads.get(nxt)
            ncolour = colour ^ pad[1] if pad is not None else colour
            nnode = (nxt, ncolour)
            cost = d0 + (1 if coins.get(nxt) == ncolour else 2)
            if cost < dist.get(nnode, 1 << 30):
                dist[nnode] = cost
                parent[nnode] = (node, name)
                heapq.heappush(queue, (cost, nnode))

    def path_to(node):
        out = []
        while parent[node] is not None:
            node, name = parent[node]
            out.append(name)
        out.reverse()
        return out

    return dist, path_to


class Macro:
    """One objective and the route that reaches it: ``kind`` in
    ``{'coin', 'pad', 'totem', 'exit'}``, ``node`` the ``(cell, colour)`` it
    ends on, ``path`` the presses."""

    __slots__ = ("kind", "node", "path")

    def __init__(self, kind, node, path):
        self.kind, self.node, self.path = kind, node, path


def macros(board: Board, st: St) -> list[Macro]:
    """Every objective the golem can reach right now, one per distinct OUTCOME.

    Four things are worth walking to: a coin this colour collects, a limited pad
    (the only irreversible colour change), a cell beside a totem (which
    transmutes the coins behind it), and the exit once the board is clear.
    Reusable pads are NOT objectives -- they are edges in `routes`, so any colour
    they can produce is already reachable as some other objective's arrival
    colour.

    Two arrivals at the same totem cell with different colours are the same
    outcome whenever they set/clear the same bits, so the signature -- not the
    colour -- is what deduplicates them.

    Nothing is capped. Offering only the nearest two dozen objectives was tried
    (it is the obvious way to hold the branching factor down on the 45- and
    49-coin sweeps) and it is a false economy: it did not measurably speed any
    level up, and on level 8 it hid the one distant coin that had to be picked
    up before a pad was spent, taking the level from solved to unsolvable."""
    dist, path_to = routes(board, st)
    start = (st.pos, st.colour)
    best: dict = {}
    for node, cost in dist.items():
        if node == start:
            continue
        cell, colour = node
        pad = st.pads.get(cell)
        if st.coins.get(cell) == colour:
            key = ('coin', cell)
        elif pad is not None and pad[0] == 'ltd':
            key = ('pad', cell, colour)
        elif cell in board.adj_totem:
            key = ('totem', cell,
                   tuple(sorted((d, bool(colour & m))
                                for d, m in board.adj_totem[cell])))
        elif cell == board.target and not st.coins:
            key = ('exit', cell)
        else:
            continue
        if cost < best.get(key, (1 << 30, None))[0]:
            best[key] = (cost, node)
    return [Macro(key[0], node, path_to(node)) for key, (_c, node) in best.items()]


# ---------------------------------------------------------------------------
# The macro beam
# ---------------------------------------------------------------------------

class _Node:
    """One beam state. The plan is kept as a parent chain, not a list per state:
    the big boards run hundreds of presses deep and hundreds wide, and copying
    the prefix into every child is what made an early version eat gigabytes."""

    __slots__ = ("st", "g", "parent", "seg", "macs")

    def __init__(self, st, g, parent, seg, macs):
        self.st, self.g, self.parent, self.seg, self.macs = st, g, parent, seg, macs

    def segments(self) -> list:
        out, node = [], self
        while node.seg is not None:
            out.append(node.seg)
            node = node.parent
        out.reverse()
        return out


def beam_plan(board: Board, st0: St, width: int, depth: int,
              node_cap: int) -> list | None:
    """Width-capped breadth-first beam over macros. Returns the winning plan as
    a list of ``(path, dest_node)`` segments, or None.

    Ranking is ``1000 * coins_left + presses_so_far``: the coin count dominates
    (this is a collection puzzle -- a state that has cleared more of the board is
    nearer the win in a way no distance term captures), and the press count
    breaks its ties toward the shorter route.

    Dedup is a HASH of the state key rather than the key itself: on level 17 the
    key holds 272 coins and keeping a quarter of a million of them costs
    gigabytes. A hash collision can only drop a state the beam might have wanted,
    never produce a wrong plan -- every plan is engine-verified downstream."""
    if not st0.coins and st0.pos == board.target:
        return []
    frontier = [_Node(st0, 0, None, None, macros(board, st0))]
    seen = {hash(st0.key())}
    nodes = 0
    for _depth in range(depth):
        kids = []
        for node in frontier:
            for macro in node.macs:
                nodes += 1
                if nodes > node_cap:
                    return None
                nst, win = run(board, node.st, macro.path)
                if win >= 0:
                    won = _Node(nst, 0, node,
                                (macro.path[:win + 1], macro.node), None)
                    return won.segments()
                digest = hash(nst.key())
                if digest in seen:
                    continue
                seen.add(digest)
                if not alive(board, nst):
                    continue          # a coin can never be met again: lost
                nmacs = macros(board, nst)
                if nst.coins and not any(m.kind != 'exit' for m in nmacs):
                    # Coins left and nothing that could ever help: no collectible
                    # coin, no pad to change colour with, no totem to recolour
                    # one. Provably lost, so it must not eat a beam slot.
                    continue
                g = node.g + len(macro.path)
                kids.append((1000 * len(nst.coins) + g,
                             _Node(nst, g, node, (macro.path, macro.node), nmacs)))
        if not kids:
            return None
        kids.sort(key=lambda kid: kid[0])
        frontier = [kid for _rank, kid in kids[:width]]
    return None


#: Beam widths tried in order; the SHORTEST plan any of them finds wins. A wider
#: beam is not monotonically better -- it explores a different frontier, and the
#: narrow ones are also the cheap ones, so this is both a quality and a
#: wall-clock ladder.
_WIDTHS = (24, 96, 320)


def solve(board: Board, st0: St, widths=_WIDTHS, depth: int = 500,
          node_cap: int = 120_000) -> list | None:
    best = None
    for width in widths:
        segs = beam_plan(board, st0, width, depth, node_cap)
        if segs is None:
            continue
        length = sum(len(path) for path, _n in segs)
        if best is None or length < best[0]:
            best = (length, segs)
    return None if best is None else best[1]


# ---------------------------------------------------------------------------
# Optimal-action sets
# ---------------------------------------------------------------------------

def _successor(board: Board, st: St, node, direction):
    """The joint ``(cell, colour)`` node one press away, or None if the press is
    refused. Mirrors `routes`' edge rule exactly."""
    cell, colour = node
    dr, dc = DELTA[direction]
    nxt = (cell[0] + dr, cell[1] + dc)
    if not board.passable(nxt, colour):
        return None
    pad = st.pads.get(nxt)
    return (nxt, colour ^ pad[1] if pad is not None else colour)


def _unit_field(board: Board, st: St, start):
    """Breadth-first PRESS distances from ``start`` over the same joint graph
    `routes` walks (limited pads absorbing), plus the successor map, which the
    backward pass reverses."""
    dist = {start: 0}
    succ: dict = {}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        cell = node[0]
        here = st.pads.get(cell)
        if node != start and here is not None and here[0] == 'ltd':
            succ[node] = []
            continue
        outs = []
        for direction in DIRECTIONS:
            nxt = _successor(board, st, node, direction)
            if nxt is None:
                continue
            outs.append((direction, nxt))
            if nxt not in dist:
                dist[nxt] = dist[node] + 1
                queue.append(nxt)
        succ[node] = outs
    return dist, succ


def _inert(board: Board, st: St, node) -> bool:
    """True when passing through ``node`` changes nothing about the world.

    Three things make one route differ from another: a coin this colour would
    pick up, an unused limited pad (consumed on entry), and a cell beside a
    totem (whose firing depends on the colour standing there). A REUSABLE pad is
    inert -- it changes only the colour, and the colour is part of the node, so
    two routes that arrive at the same node arrive identical."""
    cell, colour = node
    pad = st.pads.get(cell)
    return (st.coins.get(cell) != colour
            and not (pad is not None and pad[0] == 'ltd')
            and cell not in board.adj_totem)


def route_dag(board: Board, st: St, length: int, dest):
    """``(on_route, back)`` -- every joint node lying on SOME shortest route of
    ``length`` presses from the golem to ``dest``, and the distance-to-``dest``
    field they are read off. None when the ties are not claimable.

    Two conditions have to hold before one route may be called as good as
    another. The recorded route has to be shortest at all -- the Dijkstra in
    `routes` prefers coins over distance and sometimes buys one with a detour --
    and every node on any shortest route, other than the two endpoints that
    every route shares, has to be `_inert`. Those are exactly the ways two
    routes of the same length can leave different boards behind."""
    start = (st.pos, st.colour)
    forward, succ = _unit_field(board, st, start)
    if forward.get(dest) != length:
        return None
    preds: dict = {}
    for node, outs in succ.items():
        for _d, nxt in outs:
            preds.setdefault(nxt, []).append(node)
    back = {dest: 0}
    queue = deque([dest])
    while queue:
        node = queue.popleft()
        for prev in preds.get(node, ()):
            if prev not in back:
                back[prev] = back[node] + 1
                queue.append(prev)
    on_route = {n for n, df in forward.items()
                if back.get(n, 1 << 30) + df == length}
    if any(not _inert(board, st, n) for n in on_route
           if n != start and n != dest):
        return None
    return on_route, back


def annotate(board: Board, st: St, path, dest) -> list[list[str]]:
    """Per-press optimal-direction SETS for one macro, with ``st`` at the state
    the macro starts from. Every step is labelled, with the recorded press alone
    where `route_dag` refuses to certify the alternatives."""
    solo = [[d] for d in path]
    if not path:
        return solo
    dag = route_dag(board, st, len(path), dest)
    if dag is None:
        return solo
    on_route, back = dag
    out, node = [], (st.pos, st.colour)
    for direction in path:
        alts = []
        for cand in DIRECTIONS:
            nxt = _successor(board, st, node, cand)
            if nxt is not None and nxt in on_route and back[nxt] == back[node] - 1:
                alts.append(cand)
        out.append(alts if direction in alts else [direction])
        node = _successor(board, st, node, direction)
    return out


# ---------------------------------------------------------------------------
# The expert
# ---------------------------------------------------------------------------

class Plan(list):
    """The press sequence, carrying the optimal SET for each of its steps, so a
    disk-cached plan replayed in a later process still labels every step."""

    optsets: list

    def __init__(self, presses, optsets):
        super().__init__(presses)
        self.optsets = optsets


class ColorTotemsExpert(PSExpert):
    """Plans on the native model (`solve`) and verifies on the interpreter.

    `PSExpert` is subclassed for its memo, its snapshot/restore discipline and
    its `plan` contract only -- the A* it ships is replaced wholesale, because
    this game's plans are hundreds of presses long (see the module docstring)."""

    #: `noaction`: the game binds nothing to X, so ACTION5 is not a move.
    directions = DIRECTIONS

    def setup(self) -> None:
        self._name_of = {i: n for n, i in self.g.obj_name_to_idx.items()}
        self._disk = _load_disk()

    def heuristic(self, eng) -> int:                    # pragma: no cover
        raise AssertionError("ColorTotemsExpert plans on the model, not by A*")

    def model(self, eng) -> tuple[Board, St]:
        return read_state(eng, self._name_of)

    def _search(self, eng) -> list | None:
        """Plan on the model, annotate the ties, then REPLAY on the interpreter
        and keep the plan only if the engine itself reports the win.

        `PSExpert.plan` snapshots around this call, so stepping the engine here
        is free; and it is the only check that matters -- a model that has
        drifted from the interpreter costs coverage, never a bad episode."""
        board, st0 = self.model(eng)
        segs = solve(board, st0)
        if segs is None:
            return None
        presses, optsets = [], []
        state = st0.copy()
        for path, dest in segs:
            optsets.extend(annotate(board, state, path, dest))
            presses.extend(path)
            state, _win = run(board, state, path)
        for direction in presses:
            eng.step(direction)
            if eng.check_win():
                break
        if not eng.check_win():
            return None
        return Plan(presses, optsets)

    # -- disk plan cache ------------------------------------------------------
    def plan(self, eng, level: int | None = None) -> list | None:
        """`PSExpert.plan`, with the level's START plan also cached on disk.

        Only that one state is worth keeping: every seed and every process plans
        from it and nothing else, because recovery is a RESET back to it. The
        first entry stored for a level is therefore its start; a re-plan from a
        mid-level state never overwrites it."""
        if level is None:
            return super().plan(eng, level)
        sig = sorted([r, c, o] for (r, c, o) in self._key(eng))
        entry = self._disk.get(level)
        if entry is not None:
            if entry["start"] != sig:
                return super().plan(eng, level)
            return (Plan(entry["plan"], entry["optsets"])
                    if entry["plan"] is not None else None)
        found = super().plan(eng, level)
        self._disk[level] = {
            "start": sig,
            "plan": None if found is None else list(found),
            "optsets": None if found is None else found.optsets,
        }
        _save_disk(self._disk)
        return found


def _load_disk() -> dict:
    """``{level: {"start": sig, "plan": [...] | None, "optsets": [...]}}``, or
    empty if unreadable -- a cache that cannot be parsed is a miss, never a
    crash."""
    try:
        raw = json.loads(PLAN_CACHE.read_text())
        return {int(k): v for k, v in raw.items()}
    except Exception:                                            # noqa: BLE001
        return {}


def _save_disk(disk: dict) -> None:
    """Write atomically -- shards started together would otherwise interleave
    into a truncated file (see `parallelize_generator`)."""
    try:
        PLAN_CACHE.parent.mkdir(parents=True, exist_ok=True)
        tmp = PLAN_CACHE.with_suffix(f".json.tmp{os.getpid()}")
        tmp.write_text(json.dumps({str(k): disk[k] for k in sorted(disk)},
                                  indent=1))
        os.replace(tmp, PLAN_CACHE)
    except OSError:
        pass                                                     # cache optional


class ColorTotemsSolver(PSAStarSolver):
    game_id = "puzzlescript_color_totems"
    game_name = GAME_NAME
    expert_cls = ColorTotemsExpert

    #: The adapter subclass in `games/ps:color_totems` raises the per-level step
    #: budget above the stock 200; recording against the plain adapter would tape
    #: frames a live agent never gets (and silently GAME_OVER the sweep levels).
    game_module_id = "ps:color_totems"

    #: Level 17 is the victory lap -- a 25x27 board carrying 272 coins in four
    #: 68-coin blobs, with five one-shot pads to spend on them. No width in
    #: `_WIDTHS` gets near it, and it would not be worth much if it did: at
    #: 64/27 the board renders at TWO pixels per cell, which is the floor of
    #: what the sprites survive. Skipped up front so discovery does not burn the
    #: whole ladder on it at every startup.
    skip_levels: frozenset[int] = frozenset({17})

    #: Longest plan plus room for the exploration prefix and a re-plan after it.
    max_steps: int = 900
    #: Recovery data comes from the RESET prefix, not from detours inside the
    #: replay -- a detour here would need a re-plan, and a re-plan is a beam.
    epsilon: float = 0.0

    def prepare_expert(self, game, expert) -> None:
        """Plan every level before `discover_solvable` asks for it -- the same
        work either way, but it makes the one-time cost visible as startup
        rather than as a mysteriously slow first seed, and it fills the disk
        cache in one pass."""
        for level in range(game.n_levels):
            if level in self.skip_levels:
                continue
            game.set_level(level)
            expert.plan(game._engine, level)

    def optimal_for(self, expert, level: int, plan: list, pi: int):
        """The full set of equally-shortest presses at this step -- see
        `annotate`. Falls back to the press about to be taken, so no expert step
        ever ships unlabelled (the always-emit-optimal-targets rule)."""
        sets = getattr(plan, "optsets", None)
        if sets is not None and pi < len(sets):
            return sets[pi]
        return [plan[pi]] if pi < len(plan) else None


# ---------------------------------------------------------------------------
# Mechanic self-check
# ---------------------------------------------------------------------------

def selfcheck(trials: int = 25, steps: int = 150, verbose: bool = True) -> int:
    """Audit, against the interpreter, every claim the solver rests on.

    The four mechanics in the module docstring are checked as statements on
    hand-built boards; the tie sets are checked by walking alternative shortest
    routes and comparing the boards they leave; and then the WHOLE model is
    fuzzed against the interpreter from every level start, comparing the full
    state (golem cell and colour, every coin and its colour, every live pad)
    after EVERY press together with the win flag. That last one is the check
    that matters -- the planner trusts the model for hundreds of presses at a
    time, so any rule interaction nobody thought of has to show up here.
    Returns the number of violations."""
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    eng = game._engine
    name_of = {i: n for n, i in game._game.obj_name_to_idx.items()}
    idx = game._game.obj_name_to_idx
    bad = 0

    def fail(msg: str) -> None:
        nonlocal bad
        bad += 1
        print(f"  VIOLATION: {msg}")

    def build(row: str, colour_name: str, at: int = 1):
        """A one-row test board inside level 0's extents (``check_win`` indexes
        the grid it was loaded with), with the golem at column ``at``."""
        game.set_level(0)
        h, w = len(eng.grid), len(eng.grid[0])
        grid = [[{idx["background"]} for _ in range(w)] for _ in range(h)]
        for c in range(w):
            grid[0][c].add(idx["wall"])
            grid[h - 1][c].add(idx["wall"])
        for r in range(h):
            grid[r][0].add(idx["wall"])
            grid[r][w - 1].add(idx["wall"])
        for c, ch in enumerate(row):
            if ch != ".":
                grid[2][c + 1].add(idx[_SHORT[ch]])
        grid[2][at].add(idx[colour_name])
        eng.grid = grid
        eng._position_index_dirty = True
        eng._rule_noop_cache.clear()

    def state():
        return read_state(eng, name_of)[1]

    # 1. a pad XORs its light in, and a reusable pad does it on every ENTRY.
    for pad, mask in (("padr", 1), ("padg", 2), ("padb", 4),
                      ("padc", 6), ("pady", 3), ("padm", 5)):
        for start, colour in _CODE.items():
            build("." + _PAD_CH[pad], "player" + start)
            eng.step("right")
            if state().colour != colour ^ mask:
                fail(f"{pad} on {start}: {state().colour} != {colour ^ mask}")
    build(".n.", "playerbl")                       # off and back on: one XOR each
    eng.step("right"); eng.step("right"); eng.step("left")
    if state().colour != 0:                        # black -> red -> red -> black
        fail(f"re-entering a reusable pad did not re-apply its XOR "
             f"({state().colour} != 0)")

    # 2. a limited pad fires once and is inert forever after.
    build(".8.", "playerbl")
    eng.step("right"); eng.step("right"); eng.step("left")
    if state().colour != 1:
        fail(f"a limited pad fired twice ({state().colour} != 1)")

    # 3. a coin is picked up only by a golem of EXACTLY its colour.
    for coin, colour in _CODE.items():
        for start, scolour in _CODE.items():
            build("." + _COIN_CH[coin], "player" + start)
            eng.step("right")
            got = not state().coins
            if got != (colour == scolour):
                fail(f"{start} golem on a {coin} coin: picked_up={got}")

    # 4. a magic wall passes exactly the golems that CONTAIN its light -- and
    #    the black wall exactly the black golem.
    for wall, mask in (("wallr", 1), ("wallg", 2), ("wallb", 4),
                       ("wallc", 6), ("wally", 3), ("wallm", 5), ("wallw", 7)):
        for start, colour in _CODE.items():
            build("." + _WALL_CH[wall], "player" + start)
            eng.step("right")
            passed = state().pos[1] == 2
            if passed != ((colour & mask) == mask):
                fail(f"{start} golem into {wall}: passed={passed}")
    for start, colour in _CODE.items():
        build(".e", "player" + start)
        eng.step("right")
        if (state().pos[1] == 2) != (colour == 0):
            fail(f"{start} golem into the black wall: passed="
                 f"{state().pos[1] == 2}")

    # 5. a totem sets its light on every coin behind it when the golem has that
    #    light and clears it otherwise -- through walls, and for EVERY coin in
    #    the line, not just the first.
    for start, colour in (("playerbl", 0), ("playerr", 1)):
        build("..%3#3", start)        # step right and the totem is one cell on
        eng.step("right")
        coins = read_state(eng, name_of)[1].coins
        want = 1 if colour & 1 else 0
        if sorted(coins.values()) != [want, want]:
            fail(f"red totem with a {start}: coins {sorted(coins.values())} "
                 f"!= [{want}, {want}] (through-wall / multi-coin)")

    # 6. the tie sets: every alternative shortest route really is equivalent.
    #    `annotate` labels a step with more than one press only when it has
    #    proved that all the routes reaching the same destination in the same
    #    number of presses leave the SAME board, so walking random ones and
    #    comparing states is a direct test of the claim -- and mislabelled
    #    targets are the one bug in this file that would be invisible
    #    downstream (the frames stay right; only the answer is wrong).
    rng = random.Random("colortotems:ties")
    for level in range(game.n_levels):
        if level in ColorTotemsSolver.skip_levels:
            continue
        game.set_level(level)
        board, state = read_state(eng, name_of)
        segs = solve(board, state)
        if segs is None:
            continue
        for path, dest in segs:
            sets = annotate(board, state, path, dest)
            if any(len(s) > 1 for s in sets):
                want = run(board, state, path)[0].key()
                for _ in range(8):
                    alt = _random_shortest_route(board, state, len(path), dest, rng)
                    if alt is None or run(board, state, alt)[0].key() != want:
                        fail(f"L{level}: an alternative shortest route to {dest} "
                             f"left a different board -- the tie set is wrong")
                        break
            state = run(board, state, path)[0]
        if verbose:
            print(f"  L{level}: tie sets audited", flush=True)

    # 7. the model, fuzzed against the interpreter from every level start.
    for level in range(game.n_levels):
        fails = 0
        for trial in range(trials):
            game.set_level(level)
            board, st = read_state(eng, name_of)
            rng = random.Random(f"colortotems:{level}:{trial}")
            for _ in range(steps):
                direction = rng.choice(DIRECTIONS)
                eng.step(direction)
                step(board, st, direction)
                if st.key() != read_state(eng, name_of)[1].key():
                    fails += 1
                    fail(f"L{level} trial {trial}: the model diverged on "
                         f"'{direction}'")
                    break
                won = not st.coins and st.pos == board.target
                if won != bool(eng.check_win()):
                    fails += 1
                    fail(f"L{level} trial {trial}: win flag {won} != "
                         f"{bool(eng.check_win())}")
                    break
                if eng.check_win():
                    break
        if verbose:
            print(f"  L{level}: {trials} rollouts, {fails} diverged", flush=True)
    return bad


def _random_shortest_route(board: Board, st: St, length: int, dest, rng):
    """A uniformly-random walk down `route_dag`, i.e. an arbitrary shortest
    route to ``dest``. None when the DAG refused to certify the ties."""
    dag = route_dag(board, st, length, dest)
    if dag is None:
        return None
    on_route, back = dag
    out, node = [], (st.pos, st.colour)
    while back[node]:
        choices = [(d, nxt) for d in DIRECTIONS
                   for nxt in (_successor(board, st, node, d),)
                   if nxt is not None and nxt in on_route
                   and back.get(nxt, 1 << 30) == back[node] - 1]
        direction, node = rng.choice(choices)
        out.append(direction)
    return out


#: Level-file legend letters `selfcheck`'s hand-built boards use.
_SHORT = {"n": "padr", "q": "padg", "w": "padb", "z": "padc", "x": "pady",
          "p": "padm", "8": "ltdpadr", "9": "ltdpadg", "0": "ltdpadb",
          "5": "ltdpadc", "6": "ltdpady", "7": "ltdpadm", "[": "ltdpadw",
          "?": "emptypad", "#": "wall", "+": "wallr", "t": "wallg",
          "{": "wallb", "u": "wallc", "i": "wally", "}": "wallm",
          "e": "wallbl", "~": "wallw", "3": "coinbl", "f": "coinr",
          "j": "coing", "k": "coinb", "a": "coinc", "s": "coiny",
          "d": "coinm", "4": "coinw", "%": "totemr", "*": "totemg",
          "&": "totemb", "O": "target"}
_PAD_CH = {"padr": "n", "padg": "q", "padb": "w", "padc": "z", "pady": "x",
           "padm": "p"}
_COIN_CH = {"bl": "3", "r": "f", "g": "j", "b": "k", "c": "a", "y": "s",
            "m": "d", "w": "4"}
_WALL_CH = {"wallr": "+", "wallg": "t", "wallb": "{", "wallc": "u",
            "wally": "i", "wallm": "}", "wallw": "~"}


def _plan_report() -> None:
    """Plan length, engine-verified win, tie coverage and the level's step
    budget for every level -- the quick "is this game still solved" check.

    The budget column exists because the adapter's default is 200 actions and a
    level whose plan is longer than its budget is UNWINNABLE with no error
    anywhere; see `games/ps:color_totems`."""
    solver = ColorTotemsSolver()
    game, expert, solvable = solver._ensure(0)
    total = 0
    for level in range(game.n_levels):
        game.set_level(level)
        eng = game._engine
        budget = game._max_steps
        plan = expert.plan(eng, level)
        if plan is None:
            print(f"  L{level:2d}: UNSOLVED")
            continue
        for direction in plan:
            eng.step(direction)
            if eng.check_win():
                break
        total += len(plan)
        sets = getattr(plan, "optsets", []) or []
        ties = sum(1 for s in sets if len(s) > 1)
        flag = "" if len(plan) <= budget else "   *** OVER BUDGET ***"
        print(f"  L{level:2d}: {len(plan):4d} presses  win={eng.check_win()}  "
              f"budget={budget:4d}  {ties:4d}/{len(plan)} steps with a tie "
              f"set{flag}", flush=True)
    print(f"  total {total} presses over {len(solvable)}/{game.n_levels} levels")
    print(f"  solvable levels: {solvable}")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        violations = selfcheck()
        print(f"selfcheck: {violations} violations")
        sys.exit(1 if violations else 0)
    if "--plans" in sys.argv:
        _plan_report()
        sys.exit(0)
    sys.exit(ColorTotemsSolver.main())
