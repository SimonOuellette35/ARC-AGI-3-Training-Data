"""Generate Phase-1 training data for the PuzzleScript game ps:blind_maze_a1
("Blind Maze a1" by bregehr) -- a PARTIALLY OBSERVABLE localization puzzle.

Each solved seed yields one multi-level episode JSON in the shared schema::

    {
      "game_id": "puzzlescript_blind_maze_a1",
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
A board of cells, each randomly RED or WHITE, a handful of FLAGS scattered over
it, and a player who is **invisible**. The whole point of the game is that you
cannot see where you are:

  * ``Player`` is a transparent object. Nothing on the board marks your cell.
  * The only feedback is the **view panel** -- the block of ``q`` cells walled off
    to the right of the board. Every turn it is painted with the colour of the
    cell the player is standing on, and nothing else.
  * A few levels also ship a ``playerview`` marker (a blue rounded square that
    the late rule ``[playerview no player][player] -> [][playerview player]``
    teleports onto the player each turn). Those levels are the tutorial: the
    player IS visible there. Levels 0, 1, 2, 7, 9, 12 and 13 have it; the other
    18 are blind.

Movement is a **torus shift, not a walk into a wall**::

    left [ left player | no color] -> [movingright player | ]
    late left [player movingleft | color] -> [ | player movingleft color]   (loop)

i.e. stepping off the edge of the coloured region does not bump: it sets the
opposite "moving" flag, and the late loop then slides the player all the way to
the far end of the contiguous coloured run. So each direction is a *permutation*
of the board cells, left/right are mutual inverses, and so are up/down -- which
is what the game's own "HINT: You can loop from one end to the other" means.
A consequence that matters for the expert: the movement graph is undirected and
strongly connected, and movement alone reveals NOTHING (the whole belief shifts
rigidly). Every bit of information comes from the panel.

Pressing ACTION (X) is the only other move, and it is **lethal**::

    (find target)  [action player target]   -> [player success]
    (flag fail)    [action player no target] -> [fail] sfx0

On a flag the flag is collected. Anywhere else the *player object is deleted*
and a black X is painted on the cell: the level is over, permanently unwinnable,
with no lose condition to end it. So the expert must be certain before it presses.

Three modifiers hide information on top of that:

  * ``unknown`` (dark-green noise tiles, level-start ``u``/``d`` cells and the
    three tiles an ``unknownflag`` scatters when collected): the sprite is fully
    opaque, so the cell's colour is not visible -- but it is CONSTANT, so
    standing on the cell reveals it, and it stays revealed for the rest of the
    level.
  * ``swirl`` (magenta, level-start ``s`` cells and the two an ``swirlflag``
    scatters): ``[swirl color] -> [swirl]`` / ``[swirl no color] -> [swirl random
    color]`` re-rolls the cell's colour EVERY turn, so a reading taken on a swirl
    is a coin flip and carries no information at all.
  * ``fog`` (grey checkerboard): ``late [player fog no screenfog] -> [player fog
    random screenfog]`` picks one of four screen states, and on the one named
    ``screenfail`` the panel is painted a random colour instead of the true one.
    That is P(wrong reading) = 1/4 x 1/2 = **12.5%** (measured: 24/230 and
    23/182 in two probes). Readings on fog are evidence, not proof.
  * ``flipper`` (the dark X marker): exactly one flipper is switched on at level
    start (``random [start][flipper] -> [start][flipper flipperon]``) and the 3x3
    area around it reports the INVERTED colour. Levels 12/13/14 reveal which one
    with the magenta ring (``flipperactivatedimage``); levels 15-19 do not, so
    *which flipper is live* is a second hidden variable the expert has to pin
    down alongside its own position. A flipped fog cell is NOT noisy -- the
    flip rule overwrites the fog's tempcolor, so flip beats fog.

Two engine behaviours worth writing down because they differ from the source:

  * **No post-flag teleport.** The author's rule
    ``random [Player][success][board][flag] -> [][][player board][flag]`` is meant
    to re-randomize your position after every pickup. Under this interpreter the
    multi-bracket overlap heuristic (see [[ps-engine-render-gotchas]] #1)
    restricts the ``[board]`` group to the match overlapping the ``[Player]``
    group -- the player's own cell -- so the "teleport" is a no-op and the player
    keeps its position. Verified: collecting all three flags on level 3 never
    moved the player. Localize once, then collect everything.
  * **Messages are dropped** by the level parser, so the tutorial text (including
    the now-false "Your position is randomized...") never reaches a frame.


THE EXPERT: A BELIEF FILTER, NOT A SEARCH
=========================================
This is a partially observable game and it is solved as one -- see
[[no-privileged-solvers]]. The expert NEVER reads the player's position, the
``flipped`` set, or the colour hiding under an ``unknown`` tile. It maintains a
Bayes filter over the hidden state and acts on it.

**The honesty invariant is structural, not a promise.** `_visible_stack` walks
each grid cell top-down through the collision layers and keeps an object only if
it actually renders: objects with no colour (``player``, ``flipped``,
``flipperon``, ``success``, ``board``, ``screenfog``, ``tempcolor``, ``start``,
``test``, ``movingg``) are transparent and are skipped, and the walk STOPS after
the first fully opaque sprite, so whatever that sprite covers is not readable
either. That single rule is what hides the player (transparent), the active
flipper (transparent), and the colour under an ``unknown``/``swirl`` (opaque
sprite above the colour layer) -- exactly the three things the game hides from a
human looking at the 64x64 frame. Reading the derived quantities off the engine
grid rather than off the pixels is a convenience, but the quantities themselves
are precisely the renderable ones.

Even the view panel is found the way a viewer finds it: the panel is not a
distinguishable object once painted (``view`` sits BELOW the colour layer, so a
painted panel cell renders as a plain colour block), so `read` splits the
board-looking cells into connected components and calls the component that holds
the flags the board and the other one the panel.

**Hidden state.** ``h = (position, which flipper is live, colours deduced for
unknown cells)``, with a weight. Dynamics are deterministic, so the hypothesis
set only ever shrinks or gets re-weighted:

  1. move every hypothesis through `transition` (the torus shift above);
  2. multiply by ``P(panel reading | h)`` from `_reading_model`:
     swirl -> 1/2 either way (no information); un-deduced unknown under fog ->
     1/2 either way; un-deduced unknown, no fog -> 1/2, and the reading DEDUCES
     the cell's colour for that hypothesis alone; known colour -> flip it if the
     hypothesis says the cell is in the live flipper's 3x3, then 7/8 : 1/8 on fog
     and 1 : 0 otherwise;
  3. renormalize, prune, and -- if a ``playerview`` marker is on screen -- just
     collapse onto it.

The initial hypothesis set is the game's own start rule read off the frame:
``random [start][color no flag no flipper] -> [start][color player]`` means every
board cell except the flag cells and the flipper cells, crossed with every
flipper that could be the live one. If the belief ever empties (a mis-modelled
event), it widens back to every board cell rather than lying.

**Acting.** Two modes, chosen by the position marginal:

  * *Localize* while no single cell holds >= ``certainty``: pick the direction
    that maximises the expected post-observation collision score
    ``sum_p m_p^2`` under a depth-2 lookahead (exact expectation over both
    possible readings at each level). This is the classic
    minimise-expected-residual-uncertainty policy, and on a board whose colours
    are drawn i.i.d. it splits the belief roughly in half per step.
  * *Collect* once localized: distances come from a BFS over the (undirected)
    movement graph, and the visiting ORDER from an exact Held-Karp DP over the
    <= 6 remaining flags -- collecting is free once you are standing on a flag,
    so the cost to minimise is purely the walk. The optimal action SET is every
    direction that keeps the agent on some optimal tour, which is what gets
    recorded as the target.

If the belief will not collapse within the step budget (a board with a genuine
translational colour symmetry is genuinely unlocalizable) the level is abandoned
rather than gambled on -- a wrong ACTION5 is unrecoverable, and the WIN-only
corpus contract means a lost gamble would cost the whole seed anyway.


RECOVERY  (``recovery_mode = "replan"``)
========================================
A belief filter is the ideal replan-mode expert: it is driven by (action,
observation) pairs and does not care WHO chose the action. So an exploration
step is not damage to be undone, it is a free observation -- the filter absorbs
it and the expert simply carries on from wherever the detour left it. Concretely:

  * the episode-wide exploration PREFIX runs with the full action set (1-5). If
    one of its ACTION5 presses kills the player (detected honestly -- the black
    ``fail`` X is a rendered sprite), the level takes ONE recorded RESET and the
    expert re-localizes from scratch: the human "flail, die, hit reset, then
    solve" arc. If it does not kill, no RESET is recorded at all.
  * perturbation BURSTS are directions only. ACTION5 is excluded from them
    deliberately: it is irreversibly lethal off a flag, and a burst that bricks
    the level would have to be rolled back, which erases exactly the recovery
    data the burst was for. A random-direction burst, by contrast, is always
    survivable and always informative, so it is kept and labelled with what the
    expert would have done.

Both prefix and burst steps carry the expert's own optimal set as ``optimal``,
so every recorded step has a target to train on.
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arcengine import ActionInput, GameAction, GameState            # noqa: E402
from adapters.puzzlescript_adapter import PuzzleScriptAdapter       # noqa: E402
from solvers.base_solver import BaseSolver, _ID_TO_GAMEACTION       # noqa: E402
from solvers.common.ps_astar import screen_action                   # noqa: E402
from utils.explore import (EpsilonSchedule, ExplorationPolicy,       # noqa: E402
                           RESET_ACTION)
from utils.rotation import remap_action_full                        # noqa: E402

GAME_NAME = "Blind_Maze_a1"

DIRECTIONS = ("up", "down", "left", "right")
DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
ACTION_TO_DIR = {GameAction.ACTION1: "up", GameAction.ACTION2: "down",
                 GameAction.ACTION3: "left", GameAction.ACTION4: "right",
                 GameAction.ACTION5: "action"}

#: Objects that make a cell read as a board tile. ``red``/``white`` is the plain
#: case; ``unknown`` and the two swirl frames are opaque sprites that sit ON a
#: coloured cell and hide the colour without hiding the fact that it is a tile.
BOARD_LIKE = frozenset({"red", "white", "unknown", "swirl1", "swirl2"})
FLAG_KINDS = ("target", "unknownflag", "swirlflag")

#: P(the panel lies) while standing on fog: one of four screen states is
#: ``screenfail``, which repaints the panel with a colour drawn uniformly, so the
#: reading is wrong an eighth of the time.
FOG_WRONG = 0.125

OTHER = {"R": "W", "W": "R"}

#: The bookkeeping objects that carry the hidden state -- where the player is,
#: which cells the live flipper inverts, which screen state the fog rolled. Every
#: one of them is declared with no colour in the game file, so it draws nothing
#: and `FrameReader._visible_stack` drops it. Asserted at construction: if a
#: future sprite edit ever gives one of these a colour, this generator stops
#: rather than quietly becoming a privileged solver.
MUST_STAY_HIDDEN = frozenset({
    "player", "flipped", "flipperon", "success", "board", "start", "test",
    "screenfail", "screenwin", "screenwin2", "screenwin3", "tempred",
    "tempwhite", "movingleft", "movingright", "movingup", "movingdown",
    "animation"})

#: The opaque tiles the game hides a cell's colour behind. Their sprites must
#: have no transparent pixel, or the colour would leak out from under them.
MUST_STAY_OPAQUE = frozenset({"unknown", "swirl1", "swirl2"})


# ---------------------------------------------------------------------------
# Observation: everything a viewer of the 64x64 frame can see, and nothing else
# ---------------------------------------------------------------------------

@dataclass
class Obs:
    """One frame's worth of *renderable* facts. See the module docstring's
    honesty invariant -- every field here is something a human watching the
    frame reads off it, and the hidden variables (player cell, live flipper,
    colour under an opaque tile) are deliberately absent."""
    board: frozenset            #: cells that render as a board tile
    colour: dict                #: board cell -> "R"/"W", only where it is visible
    unknown: frozenset          #: cells whose colour is hidden by an unknown tile
    swirl: frozenset            #: cells whose colour re-rolls every turn
    fog: frozenset              #: cells whose panel reading is 12.5% wrong
    flippers: tuple             #: every flipper marker on the board
    active_flipper: object      #: the flipper the magenta ring reveals, or None
    flags: dict                 #: cell -> "target" / "unknownflag" / "swirlflag"
    panel: object               #: "R"/"W" -- the view panel's colour, or None
    playerview: object          #: the blue marker's cell, or None
    dead: bool                  #: a black `fail` X is on the board


class FrameReader:
    """Turns the interpreter's grid into an `Obs` using only what renders.

    The whole honesty contract is `_visible_stack`: walk a cell's objects from
    the top collision layer down, skip anything with no colour (it draws
    nothing), and stop after the first fully opaque sprite (it covers everything
    below). Applied to this game that hides precisely the three things the game
    hides from a player: the transparent ``player`` / ``flipped`` / ``flipperon``
    bookkeeping, and the colour beneath an ``unknown`` or ``swirl`` tile."""

    def __init__(self, game: PuzzleScriptAdapter) -> None:
        self.game = game
        ps = game._game
        self._names = {v: k for k, v in ps.obj_name_to_idx.items()}
        self._layer = {n: ps.get_collision_layer(i)
                       for n, i in ps.obj_name_to_idx.items()}
        self._transparent = set()
        self._opaque = set()
        for name, obj in ps.objects.items():
            if obj.colors and all(c < 0 for c in obj.colors):
                self._transparent.add(name)          # draws nothing at all
                continue
            if obj.sprite is None:                   # solid fill of the cell
                self._opaque.add(name)
            elif all(px >= 0 for row in obj.sprite for px in row):
                self._opaque.add(name)               # sprite with no holes

        leaked = sorted(MUST_STAY_HIDDEN.intersection(ps.objects) -
                        self._transparent)
        if leaked:
            raise RuntimeError(
                f"{GAME_NAME}: {leaked} render now, so reading the grid would "
                "expose hidden state -- re-audit FrameReader before generating")
        see_through = sorted(MUST_STAY_OPAQUE.intersection(ps.objects) -
                             self._opaque)
        if see_through:
            raise RuntimeError(
                f"{GAME_NAME}: {see_through} are no longer opaque, so the "
                "colour they are supposed to hide would leak into the belief")

    def _visible_stack(self, cell) -> set:
        names = sorted((self._names[i] for i in cell),
                       key=lambda n: -self._layer[n])
        out = set()
        for name in names:
            if name in self._transparent:
                continue
            out.add(name)
            if name in self._opaque:
                break
        return out

    def read(self) -> Obs:
        grid = self.game._engine.grid
        board, colour, unknown, swirl, fog = set(), {}, set(), set(), set()
        flippers, flags = set(), {}
        active = playerview = None
        dead = False
        for r, row in enumerate(grid):
            for c, cell in enumerate(row):
                s = self._visible_stack(cell)
                if not s:
                    continue
                if s & BOARD_LIKE:
                    board.add((r, c))
                if "red" in s:
                    colour[(r, c)] = "R"
                elif "white" in s:
                    colour[(r, c)] = "W"
                if "unknown" in s:
                    unknown.add((r, c))
                if "swirl1" in s or "swirl2" in s:
                    swirl.add((r, c))
                if "fog1" in s or "fog2" in s:
                    fog.add((r, c))
                if "flipper" in s:
                    flippers.add((r, c))
                if "flipperactivatedimage" in s:
                    active = (r, c)
                for kind in FLAG_KINDS:
                    if kind in s:
                        flags[(r, c)] = kind
                if "playerview" in s:
                    playerview = (r, c)
                if "fail" in s:
                    dead = True

        # The view panel renders as a plain block of colour (`view` sits below
        # the colour layer, so the painted panel is not distinguishable as an
        # object) -- it is identified the way a viewer identifies it: it is the
        # coloured region that is walled off from the one holding the flags.
        play, panel_cells = self._split_regions(board, flags)
        panel = None
        panel_colours = {colour.get(p) for p in panel_cells}
        if len(panel_colours) == 1:
            panel = panel_colours.pop()

        return Obs(board=frozenset(play),
                   colour={k: v for k, v in colour.items() if k in play},
                   unknown=frozenset(unknown & play),
                   swirl=frozenset(swirl & play),
                   fog=frozenset(fog & play),
                   flippers=tuple(sorted(flippers & play)),
                   active_flipper=active,
                   flags={k: v for k, v in flags.items() if k in play},
                   panel=panel, playerview=playerview, dead=dead)

    @staticmethod
    def _split_regions(board: set, flags: dict):
        """Split the board-looking cells into (playfield, panel) by connectivity.
        The playfield is the component the flags sit in (falling back to the
        largest one, which the playfield always is: 25-42 cells against the
        panel's 6-21)."""
        seen, comps = set(), []
        for start in board:
            if start in seen:
                continue
            stack, comp = [start], {start}
            seen.add(start)
            while stack:
                r, c = stack.pop()
                for dr, dc in DELTA.values():
                    n = (r + dr, c + dc)
                    if n in board and n not in seen:
                        seen.add(n)
                        comp.add(n)
                        stack.append(n)
            comps.append(comp)
        if not comps:
            return set(), set()
        play = next((cp for cp in comps if cp & set(flags)),
                    max(comps, key=len))
        panel = set().union(*[cp for cp in comps if cp is not play]) \
            if len(comps) > 1 else set()
        return play, panel


# ---------------------------------------------------------------------------
# Board geometry (all of it derived from the renderable board set)
# ---------------------------------------------------------------------------

def transition(pos, direction: str, board: frozenset):
    """Where a player at ``pos`` ends up after pressing ``direction``.

    The one movement rule of the game: step onto the neighbour if it is a board
    cell, otherwise slide all the way to the far end of the contiguous run in
    the OPPOSITE direction (the wrap). Verified exact against the interpreter
    over 2000 random steps spanning all 25 levels."""
    dr, dc = DELTA[direction]
    ahead = (pos[0] + dr, pos[1] + dc)
    if ahead in board:
        return ahead
    cur = pos
    while (cur[0] - dr, cur[1] - dc) in board:
        cur = (cur[0] - dr, cur[1] - dc)
    return cur


def flipped_cells(flipper, board: frozenset) -> frozenset:
    """The cells whose panel reading is inverted when ``flipper`` is the live one.

    A transcription of the three level-start rules::

        horizontal [flipperon | color no flipped] -> [flipperon flipped | flipped]
        vertical   [flipped   | color no test]    -> [flipped | color test]
        [test] -> [flipped]

    i.e. the flipper cell plus its horizontal board neighbours, then the vertical
    board neighbours of those -- the 3x3 block clipped to the board. Note the
    flipper cell itself is only flipped if it HAS a horizontal neighbour, which
    is why this mirrors the rules instead of just taking a 3x3."""
    if flipper is None:
        return frozenset()
    horiz = set()
    for dc in (-1, 1):
        q = (flipper[0], flipper[1] + dc)
        if q in board:
            horiz.add(q)
            horiz.add(flipper)
    out = set(horiz)
    for cell in horiz:
        for dr in (-1, 1):
            q = (cell[0] + dr, cell[1])
            if q in board:
                out.add(q)
    return frozenset(out)


def movement_graph(board: frozenset) -> dict:
    """Undirected adjacency over board cells. Each direction is a permutation and
    left/right (up/down) are mutual inverses, so the graph really is undirected
    and BFS distances are symmetric."""
    adj = {p: set() for p in board}
    for p in board:
        for d in DIRECTIONS:
            q = transition(p, d, board)
            if q != p:
                adj[p].add(q)
                adj[q].add(p)
    return adj


def bfs_distances(source, adj: dict) -> dict:
    dist = {source: 0}
    frontier = [source]
    while frontier:
        nxt = []
        for p in frontier:
            for q in adj[p]:
                if q not in dist:
                    dist[q] = dist[p] + 1
                    nxt.append(q)
        frontier = nxt
    return dist


# ---------------------------------------------------------------------------
# The belief-state expert
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Hyp:
    """One hypothesis about the hidden state. ``slots`` because the depth-2
    lookahead allocates a few thousand of these per decision."""
    pos: tuple
    flip: object                #: which flipper is live (None when there are none)
    learned: dict               #: unknown cell -> colour deduced under THIS hypothesis
    w: float


class BlindMazeExpert:
    """Bayes filter over (position, live flipper, deduced unknown colours) plus
    an information-greedy / shortest-tour policy on top of it."""

    #: Position marginal a single cell must reach before ACTION5 is allowed. A
    #: wrong press is unrecoverable, and on a fog level the belief never reaches
    #: exactly 1 -- each consistent fog reading only multiplies the odds by 7.
    certainty = 1.0 - 1e-7
    #: Depth of the exact expected-uncertainty lookahead used while localizing.
    lookahead = 2
    #: Hypotheses are pruned against the best weight and then hard-capped.
    prune_ratio = 1e-12
    max_hyps = 3000

    def __init__(self, game: PuzzleScriptAdapter, rng: random.Random) -> None:
        self.game = game
        self.rng = rng
        self.reader = FrameReader(game)
        self.obs: Obs = None
        self.known: dict = {}
        self.hyps: list = []
        self._flip_cache: dict = {}
        self._adj: dict = None
        self._adj_board = None

    # -- lifecycle ---------------------------------------------------------
    def start_level(self) -> None:
        """(Re)seed the belief at a level start / after a RESET."""
        self.obs = self.reader.read()
        self.known = dict(self.obs.colour)
        self._flip_cache = {}
        self._seed_hypotheses(fresh_start=True)
        self._absorb()

    def _flip_options(self) -> list:
        if self.obs.active_flipper is not None:
            return [self.obs.active_flipper]
        if self.obs.flippers:
            return list(self.obs.flippers)
        return [None]

    def flipped_for(self, flipper) -> frozenset:
        key = flipper
        if key not in self._flip_cache:
            self._flip_cache[key] = flipped_cells(flipper, self.obs.board)
        return self._flip_cache[key]

    def _seed_hypotheses(self, *, fresh_start: bool) -> None:
        """All hidden states still consistent with nothing but the level layout.

        At a level start the game's own placement rule
        ``random [start][color no flag no flipper] -> [start][color player]``
        rules out the flag cells and the flipper cells -- a fact any player picks
        up after a couple of episodes. Anywhere else (or if that leaves nothing)
        the belief widens to the whole board rather than inventing a constraint.
        """
        cells = set(self.obs.board)
        if fresh_start:
            narrowed = cells - set(self.obs.flags) - set(self.obs.flippers)
            if narrowed:
                cells = narrowed
        self.hyps = [Hyp(pos=p, flip=f, learned={}, w=1.0)
                     for p in sorted(cells) for f in self._flip_options()]

    # -- observation model -------------------------------------------------
    def _reading_model(self, hyp: Hyp, cell) -> dict:
        """``{panel reading: (likelihood, deduced-colour or None)}`` for a
        hypothesis sitting on ``cell``. This is the whole sensor model; see the
        module docstring for where each branch comes from."""
        if cell in self.obs.swirl:
            return {"R": (0.5, None), "W": (0.5, None)}       # re-rolled: no info
        base = self.known.get(cell) or hyp.learned.get(cell)
        flipped = cell in self.flipped_for(hyp.flip)
        if base is None:
            # An unknown tile we have never stood on under this hypothesis.
            if cell in self.obs.fog:
                return {"R": (0.5, None), "W": (0.5, None)}   # noisy AND latent
            # Deterministic: the reading pins the hidden colour down for good.
            return {o: (0.5, OTHER[o] if flipped else o) for o in ("R", "W")}
        expected = OTHER[base] if flipped else base
        if cell in self.obs.fog and not flipped:
            return {expected: (1.0 - FOG_WRONG, None),
                    OTHER[expected]: (FOG_WRONG, None)}
        return {expected: (1.0, None), OTHER[expected]: (0.0, None)}

    def _absorb(self, _widened: bool = False) -> None:
        """Fold the current frame's panel reading (and playerview, when the level
        ships one) into the belief."""
        if self.obs.playerview is not None:
            kept = [h for h in self.hyps if h.pos == self.obs.playerview]
            if not kept:
                kept = [Hyp(pos=self.obs.playerview, flip=f, learned={}, w=1.0)
                        for f in self._flip_options()]
            self.hyps = kept
        if self.obs.panel is not None:
            reading = self.obs.panel
            kept = []
            for h in self.hyps:
                prob, deduced = self._reading_model(h, h.pos)[reading]
                if prob <= 0.0:
                    continue
                h.w *= prob
                if deduced is not None:
                    h.learned = dict(h.learned)
                    h.learned[h.pos] = deduced
                kept.append(h)
            self.hyps = kept
        if not self.hyps:
            # The filter contradicted itself (an event we do not model). Widen
            # back to the whole board and re-absorb rather than acting on a lie.
            # ``_widened`` stops the retry recursing: if even the full board
            # cannot explain this reading, keep the wide belief (the expert then
            # simply never reaches `certainty` and the level is abandoned)
            # rather than looping.
            self._seed_hypotheses(fresh_start=False)
            if not _widened:
                self._absorb(_widened=True)
            return
        self._normalize()

    def _normalize(self) -> None:
        best = max(h.w for h in self.hyps)
        if best <= 0.0:
            return
        floor = best * self.prune_ratio
        self.hyps = [h for h in self.hyps if h.w >= floor]
        if len(self.hyps) > self.max_hyps:
            self.hyps.sort(key=lambda h: -h.w)
            del self.hyps[self.max_hyps:]
        total = sum(h.w for h in self.hyps)
        for h in self.hyps:
            h.w /= total

    def observe(self, direction: str) -> None:
        """Advance the filter after ``direction`` ("up"/.../"action") has been
        executed on the real engine."""
        new = self.reader.read()
        # A cell that just became a swirl re-rolls from now on: forget its colour.
        for cell in new.swirl - self.obs.swirl:
            self.known.pop(cell, None)
        for h in self.hyps:
            if h.learned:
                h.learned = {c: v for c, v in h.learned.items()
                             if c not in new.swirl}
        board_changed = new.board != self.obs.board
        self.obs = new
        self.known.update(new.colour)
        if board_changed:
            self._flip_cache = {}
            self._adj = None
        if direction in DELTA:
            for h in self.hyps:
                h.pos = transition(h.pos, direction, new.board)
        self._absorb()

    # -- policy ------------------------------------------------------------
    def position_marginals(self) -> dict:
        marg = {}
        for h in self.hyps:
            marg[h.pos] = marg.get(h.pos, 0.0) + h.w
        return marg

    def localized(self):
        """The cell we are sure of, or None."""
        marg = self.position_marginals()
        if not marg:
            return None
        cell, prob = max(marg.items(), key=lambda kv: kv[1])
        return cell if prob >= self.certainty else None

    def decide(self):
        """``(kind, [engine directions])`` -- ``kind`` is "press" or "move"."""
        cell = self.localized()
        if cell is not None and self.obs.flags:
            if cell in self.obs.flags:
                return "press", ["action"]
            return "move", self._tour_actions(cell)
        return "move", self._informative_actions()

    # ---- localization ----
    def _informative_actions(self) -> list:
        scores = {d: self._lookahead(self.hyps, d, self.lookahead)
                  for d in DIRECTIONS}
        best = max(scores.values())
        return [d for d in DIRECTIONS if scores[d] >= best - 1e-12]

    def _lookahead(self, hyps: list, direction: str, depth: int) -> float:
        """Expected collision score ``sum_p m_p^2`` after taking ``direction``
        and then playing ``depth-1`` more optimal steps. 1.0 means localized."""
        board = self.obs.board
        buckets = {"R": [], "W": []}
        for h in hyps:
            pos = transition(h.pos, direction, board)
            model = self._reading_model(h, pos)
            for reading, (prob, deduced) in model.items():
                if prob <= 0.0:
                    continue
                learned = h.learned
                if deduced is not None:
                    learned = dict(learned)
                    learned[pos] = deduced
                buckets[reading].append(
                    Hyp(pos=pos, flip=h.flip, learned=learned, w=h.w * prob))
        total = sum(h.w for bucket in buckets.values() for h in bucket)
        if total <= 0.0:
            return 0.0
        score = 0.0
        for bucket in buckets.values():
            mass = sum(h.w for h in bucket)
            if mass <= 0.0:
                continue
            if depth > 1 and len(bucket) > 1:
                inner = max(self._lookahead(bucket, d, depth - 1)
                            for d in DIRECTIONS)
            else:
                marg = {}
                for h in bucket:
                    marg[h.pos] = marg.get(h.pos, 0.0) + h.w
                inner = sum((m / mass) ** 2 for m in marg.values())
            score += (mass / total) * inner
        return score

    # ---- collection ----
    def _adjacency(self) -> dict:
        if self._adj is None or self._adj_board != self.obs.board:
            self._adj = movement_graph(self.obs.board)
            self._adj_board = self.obs.board
        return self._adj

    def _tour_actions(self, cell) -> list:
        """Every direction that keeps us on a shortest tour of the flags.

        Collecting is free once you are standing on a flag (the press is
        mandatory anyway and the player does not move), so the cost to minimise
        is the walk -- a shortest Hamiltonian path over <= 6 flag cells, solved
        exactly by Held-Karp on the BFS distance matrix."""
        adj = self._adjacency()
        flags = sorted(self.obs.flags)
        dist = {f: bfs_distances(f, adj) for f in flags}
        n = len(flags)
        full = (1 << n) - 1
        memo: dict = {}

        def tour(i: int, mask: int) -> float:
            """Min walk from flags[i] visiting every flag in ``mask``."""
            if mask == 0:
                return 0.0
            key = (i, mask)
            if key in memo:
                return memo[key]
            best = float("inf")
            for j in range(n):
                if not mask & (1 << j):
                    continue
                step = dist[flags[j]].get(flags[i], float("inf"))
                best = min(best, step + tour(j, mask & ~(1 << j)))
            memo[key] = best
            return best

        def cost_from(src) -> float:
            best = float("inf")
            for j in range(n):
                step = dist[flags[j]].get(src, float("inf"))
                best = min(best, step + tour(j, full & ~(1 << j)))
            return best

        here = cost_from(cell)
        board = self.obs.board
        opts = [d for d in DIRECTIONS
                if 1 + cost_from(transition(cell, d, board)) <= here + 1e-9]
        if opts:
            return opts
        # Unreachable flag (should not happen: the playfield is one component).
        near = min(flags, key=lambda f: dist[f].get(cell, float("inf")))
        base = dist[near].get(cell, float("inf"))
        return [d for d in DIRECTIONS
                if dist[near].get(transition(cell, d, board),
                                  float("inf")) < base] or list(DIRECTIONS)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def _frame_to_list(frame) -> list:
    return np.asarray(frame).tolist()


class BlindMazeA1Solver(BaseSolver):
    """`BaseSolver` for ps:blind_maze_a1.

    Like the other ps: generators this drives an external interpreter rather than
    a native `ARCBaseGame`, and an episode keeps only the levels it actually
    wins -- so `solve_episode` is overridden and only the CLI (`main`/`run`) and
    the ``{"game_id", "levels": [...]}`` save schema come from the base."""

    game_id = "puzzlescript_blind_maze_a1"
    game_name = GAME_NAME

    #: Levels never attempted. Empty: the filter wins all 25 (250/250 level
    #: solves across seeds 0-9, mean 17 actions, worst level 11 at 33). Even
    #: level 21 -- whose 14 ``d`` cells are fog AND unknown at once, so a reading
    #: on any of them is both latent and noisy -- localizes off the plain cells.
    skip_levels: frozenset = frozenset()

    #: Hard cap on actions per level. The adapter ends the episode in GAME_OVER
    #: at 200 *effective* actions, so the expert has to localize and collect
    #: inside that; anything still running at the cap is abandoned.
    max_steps: int = 180
    #: Localization steps before a level is written off as unlocalizable (a board
    #: with a translational colour symmetry genuinely is).
    localize_cap: int = 90

    supports_recovery = True
    recovery_mode = "replan"
    max_resets = 2

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._game = None

    # -- plumbing ----------------------------------------------------------
    def make_game(self, seed: int):
        return PuzzleScriptAdapter(self.game_name, seed=seed)

    def solve_from(self, game, level_idx: int, seed: int) -> list:
        """Not used: this generator drives its own belief-filter loop (the expert
        is a policy over a belief state, not a plan over a visible one), so
        `solve_episode` is overridden and the base record loop never runs."""
        raise NotImplementedError(
            "blind_maze_a1 records through its own belief-filter loop; "
            "see BlindMazeA1Solver.solve_episode")

    def _ensure(self, seed: int) -> PuzzleScriptAdapter:
        if self._game is None:
            self._game = self.make_game(seed)
        self._game._seed = seed        # augmentation + board draw re-derive
        return self._game

    # -- recording ---------------------------------------------------------
    @staticmethod
    def _step(action: GameAction, *, phase: str, changed: bool,
              optimal: list | None) -> dict:
        return {"type": "simple", "index": int(action.value), "phase": phase,
                "changed": bool(changed), "n_obs": 1,
                "optimal": None if optimal is None else
                [{"type": "simple", "index": int(a.value)} for a in optimal]}

    def _screen(self, game, direction: str) -> GameAction:
        return screen_action(direction, game._rotation_k, game._hflip,
                             game._vflip)

    def _engine_dir(self, game, action: GameAction) -> str:
        return ACTION_TO_DIR[remap_action_full(action, game._rotation_k,
                                               game._hflip, game._vflip)]

    def record_level(self, game, level: int, *, schedule=None, exploration=None):
        """Drive one level to a WIN with the belief expert, recording as we go.

        Returns ``(observations, actions)`` or ``(None, None)`` if the level was
        not won (unlocalizable board, a lethal exploration press we could not
        recover from, or the step cap)."""
        game.set_level(level)
        expert = BlindMazeExpert(game, self.rng)
        expert.start_level()

        observations = [_frame_to_list(game._current_frame)]
        actions = [{"type": "simple", "index": RESET_ACTION, "phase": "reset",
                    "changed": False, "n_obs": 1, "optimal": None}]
        prev = np.asarray(game._current_frame)
        resets = 0
        localizing = 0

        def optimal_screen():
            """The expert's target at the CURRENT state, as screen actions."""
            _, dirs = expert.decide()
            return [self._screen(game, d) for d in dirs]

        def drive(action: GameAction, phase: str, optimal):
            nonlocal prev
            fd = game.perform_action(ActionInput(id=action))
            frame = np.asarray(fd.frame[-1] if fd.frame else game._current_frame)
            observations.append(_frame_to_list(frame))
            actions.append(self._step(action, phase=phase,
                                      changed=not np.array_equal(frame, prev),
                                      optimal=optimal))
            prev = frame
            expert.observe(self._engine_dir(game, action))

        def do_reset():
            nonlocal prev
            game.perform_action(ActionInput(id=GameAction.RESET))
            frame = np.asarray(game._current_frame)
            observations.append(_frame_to_list(frame))
            actions.append({"type": "simple", "index": RESET_ACTION,
                            "phase": "reset",
                            "changed": not np.array_equal(frame, prev),
                            "n_obs": 1,
                            "optimal": [{"type": "simple",
                                         "index": RESET_ACTION}]})
            prev = frame
            expert.start_level()

        # ---- exploration prefix (episode-wide budget; full action set) ----
        while (schedule is not None and exploration is not None
               and schedule.explore()):
            schedule.advance()
            act = exploration.action(prev)
            if act.is_click:                     # this game exposes no click
                continue
            drive(_ID_TO_GAMEACTION[act.action_id], "explore", optimal_screen())
            if game._state in (GameState.WIN, GameState.GAME_OVER):
                break

        # ---- expert loop ----
        for _ in range(self.max_steps):
            if game._state == GameState.WIN:
                break
            if game._state == GameState.GAME_OVER:
                return None, None
            if expert.obs.dead:
                # A press off a flag deleted the player: the level can only be
                # recovered by a RESET (the human "flail, die, reset" arc).
                if resets >= self.max_resets:
                    return None, None
                resets += 1
                do_reset()
                localizing = 0
                continue

            kind, dirs = expert.decide()
            if kind == "move":
                localizing = localizing + 1 if expert.localized() is None else 0
                if localizing > self.localize_cap:
                    return None, None      # this board cannot be localized

            targets = [self._screen(game, d) for d in dirs]

            # Perturbation burst: random DIRECTIONS only (ACTION5 is lethal off a
            # flag and would brick the level -- see the module docstring).
            if (self._burst_prob > 0.0 and kind == "move"
                    and self.rng.random() < self._burst_prob):
                length = max(1, int(self.rng.expovariate(1.0 / self._burst_mean)))
                for _ in range(length):
                    drive(self._screen(game, self.rng.choice(DIRECTIONS)),
                          "burst", optimal_screen())
                    if game._state != GameState.NOT_FINISHED:
                        break
                continue

            drive(self.rng.choice(targets), "expert", targets)

        if game._state != GameState.WIN:
            return None, None
        return observations, actions

    # -- episode -----------------------------------------------------------
    def solve_episode(self, seed: int, explore: bool = True):
        game = self._ensure(seed)
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
                continue
            out.append({"level_id": level, "observations": obs,
                        "actions": acts})
        return len(out) == len(levels), out


if __name__ == "__main__":
    raise SystemExit(BlindMazeA1Solver.main())
