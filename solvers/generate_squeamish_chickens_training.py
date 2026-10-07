"""Generate Phase-1 training data for the PuzzleScript game
ps:the_dungeon_of_squeamish_chickens ("The Dungeon of Squeamish Chickens",
Ampersand_S).

The harness -- the rotation contract, the trajectory recorder and the
`BaseSolver` plumbing -- lives in `solvers/common/ps_astar.py`, shared with the
other ps: generators. This file is the game-specific part: the measured
mechanics, a native model of them, the exact distance FIELD that model is
searched with, the proof that every plan is shortest, and the render audit.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_squeamish_chickens",
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
A sokoban with two win conditions (every Target under a Crate, and no Coin left
on the board) and three ways to die. Seven levels, 7x9 to 12x13, all winnable,
none already won at reset. The whole rule list is eleven lines and four of them
are dead code (`BossChick1`, `BossChick2`, `proj` and `SpecialCoin` appear in no
shipped level), which leaves six rules -- and their ORDER is the game:

    1  [ >  Player | Crate ]   -> [ > Player | > Crate ]
    2  [ >  Player | Spoike ]  -> [ PlayerDead | Spoike ]
    3  [ Player | Spoike2 ]    -> [ PlayerDead | Spoike2 ]
    4  [ >  Player | Coin ]    -> [ > Player | NoCoin ]
    5  [ Chicken | ... | Player ] -> [ > Chicken | ... | Player ]
    6  [ Player | Chicken ]    -> [ PlayerDead | Chicken ]

PuzzleScript runs all six against the SETTLED grid, collecting forces, and only
then moves anything. Every clause below was measured against this interpreter
rather than read off the .txt:

* **The two spike types are two different mechanics, and the sprites do not say
  so.** `Spoike` (rule 2, black) kills only the player that PRESSES INTO it: the
  `>` binds the rule to a queued force, so standing beside one forever is safe
  and one press towards it is fatal. `Spoike2` (rule 3, red) has no `>` at all
  and no direction, so it kills any player ORTHOGONALLY ADJACENT to it, whatever
  that player pressed -- including pressing away, and including the ACTION key.
  Both sprites are the same five-pixel X. (Measured on level 3, whose border is
  a solid Spoike2 ring: stepping onto the ring's neighbour cell is survivable,
  and then every press kills, `down` back out of it included.)

* **A chicken is one step SLOWER than the rule reads.** Rule 5 fires whenever a
  chicken shares a row or column with a player, at any distance -- the ellipsis
  constrains none of the cells it spans, so the link reaches THROUGH walls,
  crates and other chickens (measured: a chicken three cells the far side of a
  wall still walks up to that wall and grinds against it). Rule 6 then kills any
  player adjacent to a chicken. But rule 6 reads the grid, and rule 5 has only
  queued a force, so the turn a chicken STEPS into adjacency is survivable and
  the turn after is not, whatever the player presses -- the chicken moves into
  the corpse's cell on that press. The practical rule is "never end a turn next
  to a chicken", which is not what either rule says on its own.

* **The ACTION key is a WAIT, and waiting is a real move.** No rule has `action`
  on a left-hand side, so ACTION5 moves the player nowhere -- but it is still a
  turn, so rules 3, 5 and 6 all fire on it: the chickens advance one step and a
  Spoike2 neighbour still kills. It is therefore in the search's press set, not
  pruned as a no-op the way it is for the rest of this family.

* **Coins are collected at ARM'S LENGTH, and a crate collects them too.** Rule 4
  fires on the cell AHEAD of a moving player, so pressing towards an adjacent
  coin banks it (the player then also walks onto the now-empty cell, because
  NoCoin does not block). A crate CAN be shoved onto an uncollected coin -- rule
  1 does not look at layer 2 -- and then shoving that crate OFF again collects
  the coin, because rules 1 and 4 both match the same cell on the same press.

* **Spikes are bodies.** Spoike and Spoike2 sit on the player's collision layer,
  so they block a push exactly as a wall does. That is the one clause this
  model got wrong on the first pass, and it is a thin margin: the fuzz caught
  it in 3 walks out of 420, all three on level 2, whose two crates each start
  directly under a spike -- the model let them be pushed into it.

* **A crate a dying player shoved still moves.** Rule 1 runs before rules 2, 3
  and 6, so the force is already on the crate when the Player object is
  replaced; force resolution then drops the (vanished) player's force and keeps
  the crate's.

* **Death is not always the end, and a corpse EDITS the board.** PlayerDead is
  on layer 2 with Target, Coin and NoCoin, so it EVICTS whatever mark the dying
  player was standing on -- a corpse on a Target deletes that target and makes
  `All Target on Crate` easier, which turns out to be how two of the seven
  levels are won fastest. Level 2 ships TWO players on one key, so one can die
  while the other plays on. Both facts are modelled rather than assumed away;
  see "Shortest" below, where the first one stops being a curiosity.

Everything above is what ``--selfcheck``'s first pass measures: a seeded random
walk on every level with the model's board compared against the engine's grid
after EVERY press -- including the presses that do nothing, the presses after a
death, and the win predicate itself.

Expert solver
-------------
A NATIVE model of the mechanic above (`_Board`), searched to an exact distance
field rather than heuristically. `_Board.step` is the interpreter's turn, in
order: the six rules against the frozen grid, then a transcription of
`PuzzleScriptAdapter._resolve_forces` restricted to the one collision layer
every body of this game lives on (walls and both spike types immovable). Nothing
in it is a planner-side approximation, which is what lets the fuzz compare whole
boards rather than a projection of them.

`_Board.solve` is two breadth-first sweeps over
``(players, crates, coins, chickens, corpses)`` states:

  1. FORWARD from the state being planned, stopping at the depth ``d*`` of the
     first win. That fixes the answer's length and collects ``F``, every state
     within ``d*`` presses of the start.
  2. BACKWARD from the winning configurations, pruned to ``F``, over the edges
     the forward sweep RECORDED rather than over an inverted press.

The edges are recorded rather than re-derived because inverting a press here
means inverting a simultaneous multi-body move -- the player, a crate it may
have shoved, and every chicken that happened to be aligned, some of them blocked
-- and a predecessor enumerator that got one of those cases wrong would be a
silent wrong answer rather than a crash. The forward sweep already visits every
edge it needs. The whole game is 140,873 states across the seven levels and the
largest single level is 126,716, so keeping them costs nothing that matters.

Pruning the backward sweep to ``F`` is exact where it is read: if ``s`` lies on
a shortest start-to-win path then its whole continuation has forward distance at
most ``d*``, so that continuation never leaves ``F``. States on no shortest path
may come out too high and none of them is ever asked about.

Optimal-action sets
-------------------
The field gives them EXACTLY, with no inference: at a state ``d`` presses from a
win, the optimal presses are every one of the five whose successor is ``d - 1``
from one. ``--selfcheck`` re-derives every set on every level by brute force (a
fresh depth-bounded BFS from each successor) and requires an exact match;
``--engine`` re-derives them a third time from the interpreter. No step ever
ships unlabelled (the always-emit-optimal-targets rule).

16 of the 146 expert presses have a second equally-right answer (12 of the 162
in ``--survive``), and they concentrate on the chicken levels, where the tie is
usually "step aside now" against "step aside next turn". The largest single set
is all five presses at once, which is honest rather than degenerate: it is the
last press of level 4, where the chicken is already adjacent, so every button
including the wait ends the same way.

Shortest, and how that is known
-------------------------------
Every plan is provably shortest, and the proof is three independent derivations
agreeing on all seven levels:

  * this file's field (forward BFS + backward BFS over the recorded edges);
  * a plain forward BFS to the first win, which is shortest by construction and
    shares no code with the field's backward half (``--selfcheck``);
  * a full depth-``d*`` enumeration of the real INTERPRETER -- the same two
    sweeps driven by `eng.step` with no native model involved at all
    (``--engine``). That is the derivation that closes the loop back to the
    engine, and it agrees on every length AND on every tie set.

**Two of the seven shortest plans WIN BY GETTING THE PLAYER EATEN, and one of
them never touches the sokoban at all.** This is the single most important thing
in this file and it is not a bug in the model; three independent derivations and
the adapter itself agree on it. The chain is:

  * PlayerDead is on COLLISION LAYER 2, with Target, Coin, NoCoin and the dark
    floor. One object per layer per cell, so the corpse EVICTS whatever was
    under the player.
  * A player can stand on a Target -- targets are floor marks, they block
    nothing.
  * So dying while standing on a Target DELETES that target, and
    `All Target on Crate` is quantified over the targets that are still on the
    grid.

Level 5 is therefore won by walking onto its one Target, collecting the one
coin, and waiting for the chicken: the crate is still sitting on its starting
square in the winning frame. Level 4 is the same trick for one of its two
targets (the other crate really is pushed home). ``--plans`` reports the
`wins by DYING` tag for exactly these two, and the adapter reports
`GameState.WIN` for both at all four rotations.

This is faithful PuzzleScript, not an adapter artefact -- the layer line and the
six rules are the shipped ones -- but it is plainly an exploit rather than the
puzzle, so BOTH answers ship:

  * default: the interpreter's own shortest, 6/16/14/10/**18**/**20**/62 = 146
    presses;
  * ``--survive``: the same exact field over the move set with every
    corpse-creating press removed, 6/16/14/10/**25**/**29**/62 = 162 presses,
    every level solved as designed. It is still provably shortest, of the plans
    that keep every player alive, and every verification pass below runs in it
    too (they all take the same switch, and all of them come back clean: 162
    tie sets brute-forced, six of the seven levels re-enumerated on the
    interpreter -- level 4 is over `_engine`'s ceiling there, see below). Its
    plans live in their own cache file so the two modes cannot serve each
    other's answers. It is the slower mode by a lot: refusing to die makes
    level 4's sweep ten times bigger (126,716 states -> 1,397,307, 29 s and
    0.7 GB), which is also what sets the `state_cap`.

The two modes agree on the five levels with no chicken to be eaten by, so the
whole difference is levels 4 and 5, and it is 7 and 9 presses.

The default is also why the search may NOT use the "prune any killing successor
unless it wins" shortcut that ps:stand_iii uses: there a death forced a restart
and was strictly worse, here a corpse EDITS the win condition, so a mid-path
death is not provably useless and is left in the space for the field to judge.
It never turns out to be worth one -- ``--plans`` reports that no plan in either
mode carries a corpse into any state but its last -- but that is a MEASUREMENT,
not an assumption.

Recovery
--------
`supports_recovery` with RESET-mode recovery. The expert re-plans from the LIVE
board, so an exploration prefix that leaves the board anywhere is answered from
where it actually is. Two guards keep that affordable, because a prefix really
can strand this game (a crate in a corner, the only player dead):

  * **no live player -> dead**, answered without a search. Six of the seven
    levels have exactly one player (only level 2 has two), so the
    overwhelmingly common way a prefix ruins a board also has an O(1) answer.
  * a `state_cap` on the forward sweep, because proving a LIVE board dead means
    exhausting its reachable set and level 4's is ~5.6M states. The cap is 4M,
    2.9x the largest sweep a shipped level start needs -- and the level that
    sets that number is ``--survive``'s level 4 at 1,397,307 states, not the
    default mode's at 126,716, so a cap sized off the default alone would have
    silently dropped a level in the other mode. It cannot affect a plan; it only
    turns "provably dead" into "not solvable within the cap" for some perturbed
    boards, and both answers make `record_level` fall back to a RESET, which is
    the same recovery either way.

``--selfcheck``'s fourth pass walks random boards on every level and requires
both halves to be right against the interpreter: every plan returned from a
random board is replayed and must WIN, and every None must be a board an
independent forward BFS also proves dead.

One thing ``--survive`` does not control: the exploration prefix presses at
random and `record_level` stops the moment the adapter says WIN, so a prefix
that stumbles into the death-win on level 4 or 5 would be taped even in that
mode. It is the prefix's win, not the expert's, and it is the same shape as
every other prefix that happens to win outright; nothing downstream
distinguishes them.

Augmentation
------------
The engine state after reset is identical for every seed (levels are fixed ASCII
maps), so the only per-(seed, level) variable is the frame rotation
(``rotation_k`` in 0..3) with its matching directional action remap. 7 levels x
4 rotations = 28 presentations.

No flips. The structural argument for the rotation is that no rule in the game
names an axis -- the push is written with the relative force ``>``, and rules 3,
5 and 6 are undirected four-way expansions -- and ``--symmetry`` measures it:
every level's plan AND a seeded random walk (which does what a plan never does:
jams crates into spikes, walks into chickens and presses the unbound ACTION key)
replayed at all four rotations, requiring every frame to be the exact transform
of the unrotated one. A flip would need the same evidence for the ART, and the
art is where it fails: the player is a little figure with two legs and the
chicken has a beak on one side, so a mirror draws a chicken that this game has
no sprite for. A quarter turn invents sprites too -- a chicken on its side is
not a sprite this game has either -- and the reason that is nonetheless safe is
that the WHOLE frame turns together, so a turned board is a consistent board and
the agent's own action remap turns with it. What would break that is a turn of
one cell composition landing on the art of a DIFFERENT one, since then two
distinct boards would share a frame; ``--audit``'s third pass checks all four
turns of all 32 compositions against each other and finds none.

Rendering
---------
One thing had to change, and it was the win condition itself. See the header
comment in ``data/puzzlescript_games/The_Dungeon_of_Squeamish_Chickens.txt``.

``--audit`` renders every cell COMPOSITION the game can show -- floor, the dark
floor decoration, a wall, a dark wall, a target, a coin, a collected coin, a
corpse, and each of those under each of the five bodies (player, crate, chicken,
Spoike, Spoike2) -- at every board shape the seven levels use, and requires them
all to be pixel-distinct. The one deliberate family of collisions is NoCoin:
NoCoin IS "the coin is gone", it is declared `transparent`, and the audit checks
that as a positive (``nocoin+X`` must be pixel-identical to ``X``) instead of
excusing it as a clash.
"""

from __future__ import annotations

import itertools
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arcengine import ActionInput, GameState                          # noqa: E402
from solvers.base_solver import _ID_TO_GAMEACTION                     # noqa: E402
from solvers.common.ps_astar import (PSAStarSolver, PSExpert, Plan,   # noqa: E402
                                     restore, screen_action, snapshot)
from utils.rotation import inverse_remap_action_full                  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent

GAME_NAME = "The_Dungeon_of_Squeamish_Chickens"

#: Every press, in the order ties are broken. ACTION5 is last because it is the
#: WAIT: it is bound to no rule, so it never moves the player -- but it is a
#: turn, so the chickens advance and a Spoike2 neighbour still kills, which is
#: why it is in the set at all. See the module docstring.
DIRS: tuple[str, ...] = ("up", "down", "left", "right", "action")

#: Index of the wait inside `DIRS`; `_Board.step` branches on it.
WAIT = 4

_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

#: Disk cache of every level's start plan AND its optimal-action sets. Six of
#: the seven fields are milliseconds and level 4's is ~2 s, so this is not the
#: load-bearing cache it is on the big ps: sokobans -- it is here so a
#: `parallelize_generator` fan-out shares one derivation rather than repeating
#: it on every core. Delete the file to re-derive it.
PLAN_CACHE: Path = _REPO_ROOT / "data" / "squeamish_chickens_plans.json"

#: The same, for ``--survive``. A separate file because the two modes answer a
#: different question and a shared cache would serve one mode the other's plan.
SURVIVE_PLAN_CACHE: Path = (_REPO_ROOT / "data"
                            / "squeamish_chickens_survive_plans.json")


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

class _Board:
    """One level's static geometry, plus the mechanic, plus the search.

    Cells are flat ``r * w + c`` indices and a STATE is five sorted tuples:

        (players, crates, coins, chickens, corpses)

    Walls, both spike types and the Target set never change, so they live here
    rather than in the state. ``corpses`` is in the state because PlayerDead is
    on layer 2 with the Targets and EVICTS the one it lands on, so where the
    dead players fell changes the win condition -- see `won`.

    THE MECHANIC, as one press. The interpreter runs every rule against the
    settled grid, collecting forces, and only then moves anything; `step` is
    that, in the .txt's rule order:

      1. Push. A crate directly ahead of a moving player is forced the same way.
         There is no rule matching two crates, so a crate never pushes a crate
         and the chain stops at one.
      2. Spoike. A player pressing INTO one dies. (`>` on the left-hand side, so
         a queued force is required and standing beside one is safe.)
      3. Spoike2. A player orthogonally ADJACENT to one dies, whatever it
         pressed, the wait included.
      4. Coin. The cell ahead of a still-living moving player stops being a coin.
      5. Chickens. A chicken sharing a row or column with a living player is
         forced one step towards it. The rule is written with an ellipsis, which
         constrains nothing in between, so the link ignores walls and bodies.
      6. Chicken kill. A player orthogonally adjacent to a chicken dies -- to the
         chickens' PRE-move positions, because rule 5 only queued forces.
      7. Resolution: `_resolve`, a transcription of the interpreter's chain
         resolver for the single collision layer every body here shares.

    Deaths are applied between the rules exactly where they fall: a player
    killed by rule 2 is already gone when rule 4 looks for coins, and one killed
    by rule 6 is not. A dying player's own force is dropped and a crate it
    already shoved keeps moving.
    """

    __slots__ = ("h", "w", "blocked", "spoike", "spoike2", "targets",
                 "edge", "adj", "s2adj", "row", "col")

    def __init__(self, h: int, w: int, walls, spoike, spoike2, targets):
        self.h, self.w = h, w
        self.spoike = frozenset(spoike)
        self.spoike2 = frozenset(spoike2)
        self.targets = frozenset(targets)
        #: Everything IMMOVABLE on the bodies' collision layer. Both spike types
        #: share that layer with the player, the crates and the chickens, so
        #: they stop a push exactly the way a wall does -- the clause the fuzz
        #: caught missing (see the module docstring).
        self.blocked = bytearray(h * w)
        for i in list(walls) + list(spoike) + list(spoike2):
            self.blocked[i] = 1
        #: ``edge[cell][di]`` is the neighbour cell, or -1 off the board; built
        #: once so the inner loops never do bounds arithmetic.
        self.edge = []
        for i in range(h * w):
            r, c = divmod(i, w)
            row = []
            for d in DIRS[:4]:
                dr, dc = _DELTA[d]
                nr, nc = r + dr, c + dc
                row.append(nr * w + nc if 0 <= nr < h and 0 <= nc < w else -1)
            self.edge.append(tuple(row))
        self.adj = [tuple(n for n in e if n >= 0) for e in self.edge]
        #: Rule 3 as a lookup: 1 where a player would be killed by a Spoike2
        #: this turn. Static, because Spoike2 never moves.
        self.s2adj = bytearray(h * w)
        for i in range(h * w):
            if any(n in self.spoike2 for n in self.adj[i]):
                self.s2adj[i] = 1
        #: Row / column of every cell. Rule 5's alignment test runs once per
        #: chicken on every successor a sweep generates, so it does not divide.
        self.row = [i // w for i in range(h * w)]
        self.col = [i % w for i in range(h * w)]

    # -- the mechanic --------------------------------------------------------
    def step(self, state, di: int):
        """The state after pressing ``DIRS[di]``, or ``state`` itself if the
        press changes nothing at all (a non-move: no shortest path contains
        one)."""
        players, crates, coins, chickens, corpses = state
        edge = self.edge
        d = None if di == WAIT else di

        #: One dict for every body on the shared collision layer: cell -> tag,
        #: 0 player / 1 crate / 2 chicken. Resolution does not care which is
        #: which -- they block each other identically -- so the tag is only
        #: there to take the state apart again afterwards.
        bodies = {}
        for p in players:
            bodies[p] = 0
        for c in crates:
            bodies[c] = 1
        for k in chickens:
            bodies[k] = 2

        forces = {}
        if d is not None:
            for p in players:
                forces[p] = d
            for p in players:                        # rule 1: push
                a = edge[p][d]
                if a >= 0 and bodies.get(a) == 1:
                    forces[a] = d

        alive = list(players)
        died = []
        if d is not None:                            # rule 2: press into Spoike
            for p in list(alive):
                a = edge[p][d]
                if a >= 0 and a in self.spoike:
                    alive.remove(p)
                    died.append(p)
        for p in list(alive):                        # rule 3: Spoike2 adjacent
            if self.s2adj[p]:
                alive.remove(p)
                died.append(p)
        new_coins = set(coins)
        if d is not None:                            # rule 4: coin ahead
            for p in alive:
                a = edge[p][d]
                if a >= 0:
                    new_coins.discard(a)
        for k in chickens:                           # rule 5: chickens advance
            dd = self._chicken_dir(k, alive)
            if dd is not None:
                forces[k] = dd
        live_chickens = set(chickens)
        for p in list(alive):                        # rule 6: chicken adjacent
            if any(n in live_chickens for n in self.adj[p]):
                alive.remove(p)
                died.append(p)

        for p in died:            # the corpse drops off the body layer at once
            del bodies[p]
            forces.pop(p, None)

        self._resolve(bodies, forces)

        new_players, new_crates, new_chickens = [], [], []
        for pos, tag in bodies.items():
            (new_players if tag == 0
             else new_crates if tag == 1 else new_chickens).append(pos)
        new_corpses = set(corpses)
        for p in died:
            new_corpses.add(p)
            #: PlayerDead evicts the layer-2 mark it lands on. A player can
            #: never be standing on an uncollected Coin (rule 4 banks one the
            #: moment you press towards it, and a coin under a crate is a cell
            #: no player can occupy), so this discard is unreachable on the
            #: shipped boards -- it is here so an edited level cannot make the
            #: model quietly disagree with the interpreter about `no Coin`.
            new_coins.discard(p)
        return (tuple(sorted(new_players)), tuple(sorted(new_crates)),
                tuple(sorted(new_coins)), tuple(sorted(new_chickens)),
                tuple(sorted(new_corpses)))

    def _chicken_dir(self, k: int, players) -> "int | None":
        """Rule 5 for one chicken: the direction index it is forced in, or None.

        A chicken and a player share at most one of a row and a column (sharing
        both would put them in the same cell), so a single-player board answers
        this unambiguously. `_Board` still scans a LIST because level 2 has two
        players -- and `SqueamishChickensExpert.read` refuses to build a board
        that has both several players and a chicken, since which of two matches
        the interpreter's scan would leave standing is not something the shipped
        levels can be used to measure.
        """
        row, col = self.row, self.col
        kr, kc = row[k], col[k]
        for p in players:
            if row[p] == kr:
                return 3 if col[p] > kc else 2
            if col[p] == kc:
                return 1 if row[p] > kr else 0
        return None

    def _resolve(self, bodies: dict, forces: dict) -> None:
        """Move every forced body, in place.

        A transcription of `PuzzleScriptAdapter._resolve_forces` for the single
        collision layer this game's bodies share, which collapses its layer
        bookkeeping and drops its rigid-group cascade (no rule here is `rigid`).
        What is kept is the part that decides the answers: chains of bodies
        moving the SAME way move together, a chain blocked by anything else is
        cancelled whole, a chain blocked by a body heading somewhere ELSE is
        deferred to the next iteration in case that body vacates, and two chains
        claiming the same cell block each other.
        """
        edge, blocked = self.edge, self.blocked
        for _ in range(20):
            moved = False
            deferred = False
            at_iter_start = dict(forces)
            resolved: set[int] = set()
            movable: list[tuple[list[int], int]] = []
            for pos, dd in list(forces.items()):
                if pos in resolved:
                    continue
                if pos not in bodies:            # its object died this turn
                    forces.pop(pos, None)
                    resolved.add(pos)
                    continue
                chain = [pos]
                cur = pos
                free = False
                blocker_is_mover = False
                while True:
                    n = edge[cur][dd]
                    if n < 0 or blocked[n]:
                        break
                    if n not in bodies:
                        free = True
                        break
                    if forces.get(n) == dd:
                        chain.append(n)
                        cur = n
                    else:
                        blocker_is_mover = n in forces
                        break
                if free:
                    movable.append((chain, dd))
                elif blocker_is_mover:
                    deferred = True
                else:
                    for e in chain:
                        forces.pop(e, None)
                        resolved.add(e)
            non_head = {e: i for i, (chain, _) in enumerate(movable)
                        for e in chain[1:]}
            subsumed = {i for i, (chain, _) in enumerate(movable)
                        if chain[0] in non_head}
            claims: dict[int, int] = {}
            conflicting: set[int] = set()
            for i, (chain, dd) in enumerate(movable):
                if i in subsumed:
                    continue
                for e in chain:
                    t = edge[e][dd]
                    other = claims.get(t)
                    if other is not None and other != i:
                        conflicting.add(i)
                        conflicting.add(other)
                    else:
                        claims[t] = i
            for i, (chain, dd) in enumerate(movable):
                if i in subsumed:
                    continue
                if i in conflicting:
                    for e in chain:
                        forces.pop(e, None)
                        resolved.add(e)
                    continue
                for e in reversed(chain):
                    bodies[edge[e][dd]] = bodies.pop(e)
                    forces.pop(e, None)
                    resolved.add(e)
                    moved = True
            if not moved:
                if deferred and forces == at_iter_start:
                    forces.clear()
                break

    # -- the goal ------------------------------------------------------------
    def won(self, state) -> bool:
        """``All Target on Crate`` and ``no Coin``.

        The first condition is read off the .txt literally -- every Target cell
        must carry a Crate, which is not the same as every crate being home, and
        the shipped levels have equal counts so it makes no difference there.
        Targets a corpse has EATEN are gone from the grid, hence gone from the
        condition; that is the interpreter's behaviour, not a relaxation.
        """
        players, crates, coins, chickens, corpses = state
        return (not coins
                and (self.targets - set(corpses)) <= set(crates))

    # -- the search ----------------------------------------------------------
    def field(self, state, state_cap: int = 4_000_000, survive: bool = False):
        """``(dist, d_star)``: presses-to-win for every state on a shortest path
        from ``state``, and the length of that path. ``(None, None)`` when this
        board can never be won from here, or when proving that would cost more
        than ``state_cap`` states (see the module docstring on recovery -- the
        cap is 12x the largest shipped sweep, so it cannot touch a plan).

        ``survive`` drops every edge that creates a corpse, which is the
        ``--survive`` mode: the same exact field over a restricted move set, so
        its answers are still shortest -- shortest among the plans that keep
        every player alive. See the module docstring for why that mode exists.

        Sweep 1 is a forward BFS stopped at the depth of the first win, which
        both fixes ``d_star`` and collects ``F``. Sweep 2 is a backward BFS from
        the won boards over the edges sweep 1 recorded -- see the module
        docstring for why the edges are kept rather than a press inverted.
        """
        if self.won(state):
            return {state: 0}, 0
        depth = {state: 0}
        queue = deque([state])
        rev: dict = {}
        d_star = None
        while queue:
            cur = queue.popleft()
            d = depth[cur]
            if d_star is not None and d >= d_star:
                break
            for di in range(5):
                nxt = self.step(cur, di)
                if nxt == cur:
                    continue
                if survive and len(nxt[4]) > len(cur[4]):
                    continue
                rev.setdefault(nxt, []).append(cur)
                if nxt in depth:
                    continue
                depth[nxt] = d + 1
                if self.won(nxt):
                    if d_star is None:
                        d_star = d + 1
                    continue                     # a won board is terminal
                queue.append(nxt)
            if len(depth) > state_cap:
                return None, None
        if d_star is None:
            return None, None

        dist = {}
        back = deque()
        for s in depth:
            if self.won(s):
                dist[s] = 0
                back.append(s)
        while back:
            cur = back.popleft()
            for prev in rev.get(cur, ()):
                if prev not in dist:
                    dist[prev] = dist[cur] + 1
                    back.append(prev)
        return dist, d_star

    def optimal(self, dist, state, survive: bool = False):
        """``[(direction index, successor), ...]`` for every press on a shortest
        path from ``state``. Ties come out in ``DIRS`` order, which is what makes
        a re-derived plan byte-identical across processes."""
        rest = dist.get(state)
        if not rest:
            return []
        out = []
        for di in range(5):
            nxt = self.step(state, di)
            if nxt == state or (survive and len(nxt[4]) > len(state[4])):
                continue
            if dist.get(nxt, -1) == rest - 1:
                out.append((di, nxt))
        return out

    def solve(self, state, state_cap: int = 4_000_000,
              survive: bool = False) -> "Plan | None":
        """The shortest press sequence from ``state``, with the EXACT optimal set
        at every step, or None if this board cannot be won from here."""
        dist, d_star = self.field(state, state_cap, survive)
        if dist is None:
            return None
        presses, optsets = [], []
        cur = state
        for _ in range(d_star):
            best = self.optimal(dist, cur, survive)
            if not best:                # unreachable: dist[cur] > 0 has a step
                return None
            presses.append(DIRS[best[0][0]])
            optsets.append([DIRS[di] for di, _ in best])
            cur = best[0][1]
        return Plan(presses, optsets)


# ---------------------------------------------------------------------------
# The expert
# ---------------------------------------------------------------------------

class SqueamishChickensExpert(PSExpert):
    """`PSExpert`'s plan memo and snapshot discipline around a `_Board` field.

    The base class keeps the memo, the level scoping and the disk cache; only
    the strategy underneath changes, the way `PSEnumExpert` replaces it. Here
    the strategy is "read the live board, sweep it forwards then backwards, hand
    back the shortest plan and its exact tie sets", so `heuristic` is never
    called and asserts rather than returning a number nothing would use.

    `_search` reads the ENGINE's current grid every time, so a re-plan from an
    arbitrary state (a recovery prefix, an epsilon detour) is answered from
    where the board actually is -- including "this board is now dead", which an
    exploration prefix produces here constantly, since three of the six rules
    are ways to lose the only player.
    """

    directions = list(DIRS)
    #: `_key` is the dynamic objects only, which is canonical WITHIN a level but
    #: not across them -- the walls, the two spike sets and the Target set are
    #: static per level and differ between levels.
    scope_by_level = True
    plan_cache_path = PLAN_CACHE

    #: Ceiling on the forward sweep, in states. See `_Board.field` and the
    #: module docstring. 4M is 2.9x the largest sweep any shipped level start
    #: needs -- which is NOT the default mode's level 4 (126,716) but
    #: ``--survive``'s (1,397,307, 29 s and 0.7 GB), so a cap sized off the
    #: default alone would have made ``--survive`` drop level 4 and it very
    #: nearly did. It is unreachable while planning and only bounds the cost of
    #: proving a PERTURBED board dead.
    state_cap = 4_000_000

    #: ``--survive``: refuse every press that creates a corpse, so the plans are
    #: shortest among the ones that keep every player alive. Flipped (together
    #: with `plan_cache_path`) by the ``__main__`` block before anything is
    #: constructed, because `PSExpert.__init__` reads the cache path. Default
    #: OFF: the shipped answer is the interpreter's shortest, which on levels 4
    #: and 5 wins by dying on a Target. See the module docstring.
    survive = False

    def setup(self) -> None:
        g = self.g
        self.player_ids = set(self.game._engine._player_indices)
        #: WallDark is a second wall sprite, not decoration; BackgroundDark IS
        #: decoration and sits on layer 2, so it is walkable and absent here.
        self.wall_ids = (set(g.resolve_object_name("wall"))
                         | set(g.resolve_object_name("walldark")))
        self.crate_ids = set(g.resolve_object_name("crate"))
        self.coin_ids = set(g.resolve_object_name("coin"))
        self.chicken_ids = set(g.resolve_object_name("chicken"))
        self.spoike_ids = set(g.resolve_object_name("spoike"))
        self.spoike2_ids = set(g.resolve_object_name("spoike2"))
        self.target_ids = set(g.resolve_object_name("target"))
        self.corpse_ids = set(g.resolve_object_name("playerdead"))
        #: `_Board`s by STATIC signature (see `read`), not by level index.
        self._boards: dict = {}

    def heuristic(self, eng) -> int:
        raise AssertionError(
            "SqueamishChickensExpert reads an exact distance field; "
            "heuristic is unused")

    def _key(self, eng) -> frozenset:
        """``(row, col, tag)`` triples for the five things a rule can move or
        remove: players (0), crates (1), coins (2), chickens (3) and corpses (4).

        Built from the MODEL's reading rather than from raw object ids so the
        key is exactly the model's state -- two boards with the same key are the
        same planning problem by construction, which is what the plan memo and
        the disk cache's staleness check both assume. NoCoin is deliberately not
        in it: it is the ABSENCE of a coin, already carried by (2)'s absence,
        and keying it would split one board into two.
        """
        _board, state = self.read(eng)
        if state is None:
            return frozenset()
        w = eng.width
        return frozenset(
            (pos // w, pos % w, tag)
            for tag, group in enumerate(state)
            for pos in group)

    # -- engine <-> model ----------------------------------------------------
    def read(self, eng) -> tuple:
        """``(board, state)`` for the engine's current grid, or ``(board, None)``
        when the level pairs several players with a chicken (see
        `_Board._chicken_dir`) -- which no shipped level does.

        Boards are cached by their STATIC signature (dimensions, walls, the two
        spike sets, the targets), not by level index, so the sweeps never
        rebuild the edge table and one board is shared by every state of its
        level.
        """
        h, w = eng.height, eng.width
        walls, spoike, spoike2, targets = [], [], [], []
        players, crates, coins, chickens, corpses = [], [], [], [], []
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if not cell:
                    continue
                i = r * w + c
                if cell & self.wall_ids:
                    walls.append(i)
                if cell & self.spoike_ids:
                    spoike.append(i)
                if cell & self.spoike2_ids:
                    spoike2.append(i)
                if cell & self.target_ids:
                    targets.append(i)
                if cell & self.player_ids:
                    players.append(i)
                if cell & self.crate_ids:
                    crates.append(i)
                if cell & self.coin_ids:
                    coins.append(i)
                if cell & self.chicken_ids:
                    chickens.append(i)
                if cell & self.corpse_ids:
                    corpses.append(i)
        #: A corpse has EATEN the Target it fell on, so a board read after a
        #: death no longer has that target in its grid -- and `_Board.won`
        #: subtracts the corpses from the static set, which would subtract it
        #: twice. The static set is therefore the grid's targets PLUS EVERY
        #: corpse cell. That is a superset of the level's real Target set (it
        #: also picks up corpses that fell on bare floor), and it is the right
        #: one anyway: `won` computes ``targets - corpses``, so every cell this
        #: adds is a cell it immediately takes back out, and what is left is
        #: exactly the targets still on the grid. Boards read at reset and
        #: mid-episode therefore key differently and are built twice; they agree
        #: on every state either is ever asked about, which is what matters.
        sig = (h, w, tuple(walls), tuple(spoike), tuple(spoike2),
               tuple(sorted(set(targets) | set(corpses))))
        board = self._boards.get(sig)
        if board is None:
            board = self._boards[sig] = _Board(h, w, walls, spoike, spoike2,
                                               sig[5])
        if len(players) > 1 and chickens:
            return board, None
        return board, (tuple(sorted(players)), tuple(sorted(crates)),
                       tuple(sorted(coins)), tuple(sorted(chickens)),
                       tuple(sorted(corpses)))

    def _search(self, eng) -> "Plan | None":
        board, state = self.read(eng)
        if state is None:
            return None
        if board.won(state):
            return Plan([], [])
        if not state[0]:
            #: No living player: rules 1-4 and 6 need one and rule 5 has nothing
            #: to walk towards, so every press is a no-op and the board is
            #: frozen. Answered here rather than by the sweep because it is the
            #: overwhelmingly common way an exploration prefix ends on the five
            #: single-player levels, and the sweep would pay a full step per
            #: press to discover the same thing.
            return None
        return board.solve(state, self.state_cap, self.survive)


# ---------------------------------------------------------------------------
# The solver
# ---------------------------------------------------------------------------

class SqueamishChickensSolver(PSAStarSolver):
    game_id = "puzzlescript_squeamish_chickens"
    game_name = GAME_NAME
    expert_cls = SqueamishChickensExpert

    #: `games/ps:the_dungeon_of_squeamish_chickens/...py` is a plain passthrough
    #: (it constructs the adapter and nothing else), so there is nothing to gain
    #: by routing through it -- but it IS what a live agent is handed, so if that
    #: wrapper ever grows a patch this must be set to the folder's id.
    game_module_id = ""

    #: Unused: the expert enumerates an exact field rather than searching under a
    #: heuristic, so there is no weight to trade and no node budget that could
    #: quietly bite. Left at the base values so nothing reads a lie off them.
    weight = 1
    node_cap = 400_000

    #: The longest plan is 62 presses (level 6, the maze); the rest is room for
    #: a re-plan after the exploration prefix. Stays well under the adapter's own
    #: 200-step per-level budget, which `set_level` resets before the plan starts
    #: anyway.
    max_steps = 150

    def prepare_expert(self, game, expert) -> None:
        """Plan every level before `discover_solvable` asks for it -- the same
        work either way, but it fills the disk cache in one pass and makes the
        startup cost visible as startup."""
        for level in range(game.n_levels):
            if level in self.skip_levels:
                continue
            game.set_level(level)
            expert.plan(game._engine, level)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _new():
    solver = SqueamishChickensSolver()
    game, expert, solvable = solver._ensure(0)
    return solver, game, expert, solvable


def _plans() -> int:
    """Every level's plan, replayed through the REAL interpreter -- the
    end-to-end test of the native model, the field and the tie labelling at
    once. Also reports where the corpses fall, because two of the seven
    shortest plans win by dying (see the module docstring)."""
    _solver, game, expert, solvable = _new()
    eng = game._engine
    mode = ("--survive (no press may create a corpse)" if expert.survive
            else "shortest (the interpreter's own answer)")
    print(f"  mode: {mode}")
    total = ties = bad = mid_corpse = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board, start = expert.read(eng)
        plan = expert.plan(eng, level)
        if plan is None:
            print(f"  L{level:2d}: UNSOLVED")
            bad += 1
            continue
        #: Walk the plan in the MODEL alongside the engine so the corpse count
        #: can be read per step, then check the engine agrees on the win.
        state = start
        corpse_before_end = False
        for i, direction in enumerate(plan):
            state = board.step(state, DIRS.index(direction))
            if state[4] and i < len(plan) - 1:
                corpse_before_end = True
            eng.step(direction)
        won = eng.check_win()
        bad += not won
        mid_corpse += corpse_before_end
        sets = getattr(plan, "optsets", None) or [[p] for p in plan]
        tie_steps = sum(1 for s in sets if len(s) > 1)
        total += len(plan)
        ties += tie_steps
        room = "ok" if len(plan) < game._max_steps else "OVER BUDGET"
        end = "wins by DYING" if state[4] else "survives"
        print(f"  L{level:2d}: {eng.height:2d}x{eng.width:2d}  "
              f"{len(start[0])}p {len(start[1])} crates {len(start[2])} coins "
              f"{len(start[3])} chickens  {len(plan):3d} presses  win={won}  "
              f"(budget {game._max_steps}, {room})  "
              f"{tie_steps:2d} tie steps  {end}")
    print(f"  solvable levels: {solvable}")
    print(f"  {total} presses, {ties} of them with a second equally-right answer")
    print(f"  {mid_corpse} plan(s) carry a corpse into a state that is not the "
          f"last one")
    print("  every plan reaches a WIN" if not bad else f"  {bad} PROBLEM(S)")
    return 1 if bad else 0


def _brute(board, state, limit, survive=False):
    """Presses to a win from ``state``, searched fresh, or None past ``limit``.

    Shares nothing with `_Board.field` but `_Board.step` itself, which is the
    point: it is the independent answer every optimality claim below is checked
    against."""
    if board.won(state):
        return 0
    seen = {state}
    queue = deque([(state, 0)])
    while queue:
        cur, d = queue.popleft()
        if d >= limit:
            continue
        for di in range(5):
            nxt = board.step(cur, di)
            if nxt == cur or nxt in seen:
                continue
            if survive and len(nxt[4]) > len(cur[4]):
                continue
            if board.won(nxt):
                return d + 1
            seen.add(nxt)
            queue.append((nxt, d + 1))
    return None


def _exhaust(board, state, cap, survive=False):
    """``(presses_to_win | None, exhausted)`` -- an unbounded forward BFS with a
    STATE ceiling. ``exhausted`` is True when the whole reachable set was
    visited, which is what makes a None mean "provably dead" rather than "gave
    up"."""
    if board.won(state):
        return 0, True
    seen = {state}
    queue = deque([(state, 0)])
    while queue:
        cur, d = queue.popleft()
        for di in range(5):
            nxt = board.step(cur, di)
            if nxt == cur or nxt in seen:
                continue
            if survive and len(nxt[4]) > len(cur[4]):
                continue
            if board.won(nxt):
                return d + 1, True
            seen.add(nxt)
            queue.append((nxt, d + 1))
        if len(seen) > cap:
            return None, False
    return None, True


def _selfcheck(walks: int = 400, walk_presses: int = 200,
               probes: int = 60, prefix: int = 6, verbose: bool = True) -> int:
    """Four passes. Every one of them is against something that is not this
    file's distance field.

    1. **The mechanic.** A seeded random walk on every level, the model's board
       compared against the interpreter's grid after EVERY press -- the presses
       that do nothing and the presses after a death included -- plus the win
       predicate. This is the pass that measures the module docstring's list of
       clauses; it is also the one that found spikes blocking pushes.
    2. **The lengths.** A plain forward BFS to the first win (`_brute`) on every
       level, which is shortest by construction and shares no code with the
       field's backward half.
    3. **The tie sets.** Every optimal set on every plan re-derived by brute
       force: a fresh depth-bounded BFS from each of the five successors, with
       the set being every press whose successor finishes in ``rest - 1``.
       Requires an exact match, not a superset.
    4. **Recovery.** Random boards reached by a short random prefix. A plan the
       expert returns is replayed through the INTERPRETER and must WIN; a None
       must be a board an independent exhaustive BFS also proves dead.
    """
    _solver, game, expert, _solvable = _new()
    eng = game._engine
    violations = 0

    # -- 1: the mechanic ----------------------------------------------------
    presses_checked = deaths = 0
    for level in range(game.n_levels):
        for trial in range(walks):
            game.set_level(level)
            board, state = expert.read(eng)
            rng = random.Random(f"squeamish:{level}:{trial}")
            for _ in range(walk_presses):
                di = rng.randrange(5)
                state = board.step(state, di)
                eng.step(DIRS[di])
                _b, real = expert.read(eng)
                presses_checked += 1
                if state != real:
                    print(f"    L{level} walk {trial}: model board != engine "
                          f"grid after {DIRS[di]}")
                    print(f"      model  {state}")
                    print(f"      engine {real}")
                    violations += 1
                    break
                if board.won(state) != eng.check_win():
                    print(f"    L{level} walk {trial}: win predicate disagrees "
                          f"(model {board.won(state)}, engine {eng.check_win()})")
                    violations += 1
                    break
                if eng.check_win():
                    break
                if not state[0]:
                    deaths += 1
                    break
    if verbose:
        print(f"  1. mechanic: {presses_checked} presses compared board-for-"
              f"board against the interpreter, {deaths} of the walks ended with "
              f"every player dead")

    # -- 2 and 3: lengths and tie sets --------------------------------------
    lengths, setcount = [], 0
    for level in range(game.n_levels):
        game.set_level(level)
        board, start = expert.read(eng)
        plan = expert.plan(eng, level)
        if plan is None:
            print(f"    L{level}: no plan")
            violations += 1
            continue
        d_star = len(plan)
        lengths.append(d_star)
        independent = _brute(board, start, d_star, expert.survive)
        if independent != d_star:
            print(f"    L{level}: plan is {d_star} presses, an independent BFS "
                  f"finds {independent}")
            violations += 1
        state = start
        for i, direction in enumerate(plan):
            rest = d_star - i
            want = set()
            for di in range(5):
                nxt = board.step(state, di)
                if nxt == state:
                    continue
                if expert.survive and len(nxt[4]) > len(state[4]):
                    continue
                got = _brute(board, nxt, rest - 1, expert.survive)
                if got is not None and got == rest - 1:
                    want.add(DIRS[di])
            have = set(plan.optsets[i])
            setcount += 1
            if want != have:
                print(f"    L{level} step {i}: optimal set {sorted(have)} but "
                      f"brute force says {sorted(want)}")
                violations += 1
            state = board.step(state, DIRS.index(direction))
    if verbose:
        print(f"  2. lengths: {lengths} -- every one re-derived by an "
              f"independent forward BFS")
        print(f"  3. tie sets: all {setcount} re-derived by brute force")

    # -- 4: recovery --------------------------------------------------------
    planned = dead = unconfirmed = 0
    for level in range(game.n_levels):
        for trial in range(probes):
            game.set_level(level)
            rng = random.Random(f"squeamish:probe:{level}:{trial}")
            for _ in range(rng.randrange(1, prefix + 1)):
                eng.step(DIRS[rng.randrange(5)])
            if eng.check_win():
                continue
            board, state = expert.read(eng)
            probe = snapshot(eng)
            found = expert._search(eng)
            restore(eng, probe)
            if found is None:
                _d, exhausted = _exhaust(board, state, 300_000, expert.survive)
                if not exhausted:
                    unconfirmed += 1
                    continue
                if _d is not None:
                    print(f"    L{level} probe {trial}: expert gave up on a "
                          f"board winnable in {_d}")
                    violations += 1
                    continue
                dead += 1
                continue
            for direction in found:
                eng.step(direction)
            if not eng.check_win():
                print(f"    L{level} probe {trial}: plan of {len(found)} "
                      f"presses did not win")
                violations += 1
                continue
            planned += 1
    if verbose:
        print(f"  4. recovery: {planned} random boards replanned and replayed "
              f"to a WIN, {dead} proved dead by an independent exhaustive BFS, "
              f"{unconfirmed} left unconfirmed at the 300k-state ceiling")
    return violations


# ---------------------------------------------------------------------------
# The interpreter-only derivation
# ---------------------------------------------------------------------------

def _pack(eng) -> bytes:
    """The engine grid as one immutable blob, ~4 bytes per cell.

    `snapshot` is the natural key but it is a list of sets: level 4's sweep
    visits 126,716 states and holding that many nested-set snapshots is
    gigabytes, where the blob is 40 MB. Every object index in this game is under
    32, so a cell is one 32-bit membership mask."""
    w = eng.width
    out = bytearray(len(eng.grid) * w * 4)
    i = 0
    for row in eng.grid:
        for cell in row:
            mask = 0
            for o in cell:
                mask |= 1 << o
            out[i:i + 4] = mask.to_bytes(4, "little")
            i += 4
    return bytes(out)


def _unpack(eng, blob: bytes, h: int, w: int, cache: dict) -> None:
    """Restore a `_pack`ed grid. ``cache`` memoises mask -> cell set; a board has
    only a few dozen distinct cell contents, so this turns the inner loop into a
    dict lookup."""
    grid = []
    i = 0
    for _r in range(h):
        row = []
        for _c in range(w):
            mask = int.from_bytes(blob[i:i + 4], "little")
            i += 4
            cell = cache.get(mask)
            if cell is None:
                cell = cache[mask] = frozenset(
                    o for o in range(32) if mask >> o & 1)
            row.append(set(cell))
        grid.append(row)
    eng.grid = grid
    eng._position_index_dirty = True
    eng._rule_noop_cache.clear()


def _engine(verbose: bool = True, engine_cap: int = 250_000) -> int:
    """Re-derive every level's length AND every tie set from the REAL
    interpreter, with no native model involved at all.

    The same two sweeps `_Board.field` runs, with `eng.step` as the transition
    function and `_pack`ed grids as the states. That makes it a genuinely
    independent derivation of the same two claims -- how long the shortest win
    is, and which presses are on one -- and it is the one that closes the loop
    back to the engine the frames are rendered from.

    It is exhaustive to depth ``d*`` rather than a spot check because these
    boards are small enough to afford it: 140,873 states across the seven
    levels. Level 4 is 126,716 of them and takes a couple of minutes; the other
    six are seconds.

    ``engine_cap`` is the one thing this pass will not do. An interpreter state
    costs ~350 us and ~600 bytes here (against ~20 us for the native one), so
    ``--survive``'s level 4 -- 1,397,307 states, because refusing to die makes
    the board much bigger -- would be about an hour and several gigabytes. That
    level is skipped with a printed reason rather than silently, and the native
    field still has two independent checks on it there (`--selfcheck`'s passes 2
    and 3). Nothing is skipped in the default mode, whose largest sweep is half
    the ceiling.
    """
    _solver, game, expert, _solvable = _new()
    eng = game._engine
    #: With ``--survive`` the interpreter sweep has to be restricted the same
    #: way the field is, or it would answer a different question (level 4's
    #: unrestricted shortest is the 18-press death-win) and report a
    #: disagreement that is really a difference of mode. Counted off the grid,
    #: not off the model.
    corpse_ids = set(game._game.resolve_object_name("playerdead"))

    def corpses() -> int:
        return sum(1 for row in eng.grid for cell in row if cell & corpse_ids)

    bad = 0
    total_states = 0
    for level in range(game.n_levels):
        game.set_level(level)
        h, w = eng.height, eng.width
        plan = expert.plan(eng, level)
        game.set_level(level)
        if eng.check_win():
            #: No shipped level is won at reset (`--plans` shows all seven
            #: starting unsolved), so this is a guard against an edited level
            #: rather than a case: the sweep below never keys a won board.
            print(f"    L{level}: already won at reset, nothing to enumerate")
            continue
        start = _pack(eng)
        cache: dict = {}

        depth = {start: 0}
        queue = deque([start])
        rev: dict = {}
        d_star = None
        over_cap = False
        while queue:
            cur = queue.popleft()
            d = depth[cur]
            if d_star is not None and d >= d_star:
                break
            if len(depth) > engine_cap:
                over_cap = True
                break
            for direction in DIRS:
                _unpack(eng, cur, h, w, cache)
                before = corpses() if expert.survive else 0
                eng.step(direction)
                won = eng.check_win()
                nxt = _pack(eng)
                if nxt == cur:
                    continue
                if expert.survive and corpses() > before:
                    continue
                rev.setdefault(nxt, []).append(cur)
                if nxt in depth:
                    continue
                depth[nxt] = d + 1
                if won:
                    if d_star is None:
                        d_star = d + 1
                    continue
                queue.append(nxt)
        if not over_cap:
            total_states += len(depth)

        if over_cap:
            print(f"  L{level:2d}: SKIPPED -- over the {engine_cap:,}-state "
                  f"interpreter ceiling (the native sweep is "
                  f"{'much ' if expert.survive else ''}larger here; see "
                  f"`_engine`)")
            continue

        if d_star is None or plan is None or d_star != len(plan):
            print(f"    L{level}: interpreter says {d_star}, the field says "
                  f"{None if plan is None else len(plan)}")
            bad += 1
            continue

        #: The backward half, over the recorded edges. A won board is terminal
        #: and never expanded, so it is keyed but has no outgoing edges -- the
        #: same shape as the native sweep.
        dist = {}
        back = deque()
        for blob in depth:
            _unpack(eng, blob, h, w, cache)
            if eng.check_win():
                dist[blob] = 0
                back.append(blob)
        while back:
            cur = back.popleft()
            for prev in rev.get(cur, ()):
                if prev not in dist:
                    dist[prev] = dist[cur] + 1
                    back.append(prev)

        game.set_level(level)
        cur = _pack(eng)
        mismatch = 0
        for i, direction in enumerate(plan):
            rest = dist[cur]
            want = set()
            follow = None
            for di, d_name in enumerate(DIRS):
                _unpack(eng, cur, h, w, cache)
                before = corpses() if expert.survive else 0
                eng.step(d_name)
                nxt = _pack(eng)
                if nxt == cur:
                    continue
                if expert.survive and corpses() > before:
                    continue
                if dist.get(nxt, -1) == rest - 1:
                    want.add(d_name)
                    if d_name == direction:
                        follow = nxt
            if want != set(plan.optsets[i]):
                print(f"    L{level} step {i}: interpreter says "
                      f"{sorted(want)}, the field says "
                      f"{sorted(plan.optsets[i])}")
                mismatch += 1
            cur = follow if follow is not None else cur
        bad += mismatch
        if verbose:
            print(f"  L{level:2d}: {len(depth):7d} interpreter states to depth "
                  f"{d_star:2d}; length and all {len(plan)} tie sets "
                  f"{'AGREE' if not mismatch else 'DISAGREE'}")
    if verbose:
        print(f"  {total_states} engine states enumerated in total")
    return bad


# ---------------------------------------------------------------------------
# The render audit
# ---------------------------------------------------------------------------

#: Everything that can sit on layer 2 under a body. BackgroundDark is the dark
#: floor decoration the levels edge themselves with; PlayerDead is where a
#: player died and is the reason a Target can vanish. They are mutually
#: exclusive -- one layer, one object.
_MARKS = [(), ("backgrounddark",), ("target",), ("coin",), ("playerdead",)]

#: Everything that can sit on the body layer. The two spike types never move but
#: they are drawn like bodies and they must not be confusable with each other:
#: pressing towards a Spoike is fatal and merely STANDING beside a Spoike2 is,
#: so telling them apart is the difference between two entirely different games.
_BODIES = [(), ("player",), ("crate",), ("chicken",), ("spoike",), ("spoike2",)]


def _compositions():
    comps = [("background",) + m + b for m in _MARKS for b in _BODIES]
    #: Walls are opaque and are only ever placed on bare Background, so they get
    #: one composition each rather than a row of them.
    comps += [("background", "wall"), ("background", "walldark")]
    return comps


def _comp_name(comp) -> str:
    return "+".join(o for o in comp if o != "background") or "floor"


def _audit(verbose: bool = True) -> int:
    """Three passes over every cell COMPOSITION the game can show.

    **Pass 1 -- distinctness**, at every board shape the seven levels use.
    Whole 64x64 frames of uniform boards are compared rather than one cell out
    of a mixed board: `_render_frame` upscales and centre-pads, so slicing a
    cell by ``cell_px`` arithmetic reads the wrong pixels (the ps:explod
    lesson). Two uniform boards render identically iff their cells do.

    **Pass 2 -- NoCoin draws nothing**, as a positive rather than as an excused
    clash. NoCoin is what a collected Coin becomes; it is declared
    ``transparent`` and the win condition is ``no Coin``, so a collected coin
    MUST be pixel-identical to a cell that never had one. Every composition is
    re-shot with NoCoin added and required to be byte-identical.

    **Pass 3 -- the rotation group.** The game is augmented with a frame
    rotation, so a board is drawn at any of the four turns. Sprites need not be
    turn-invariant, but a turn of one composition must not land on the art of a
    DIFFERENT one, or a turned frame would be a legal frame of another board. It
    runs on SQUARE boards, because a rotation of a non-square board also moves
    the letterbox and every comparison would pass for the wrong reason.
    """
    from adapters.puzzlescript_adapter import _render_frame            # noqa: PLC0415

    _solver, game, _expert, _solvable = _new()
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    comps = _compositions()

    sizes: dict = {}
    for level in range(game.n_levels):
        game.set_level(level)
        sizes.setdefault((eng.height, eng.width), []).append(level)

    def shoot(h, w, comp):
        eng.height, eng.width = h, w
        eng.grid = [[{idx[o] for o in comp} for _ in range(w)] for _ in range(h)]
        eng._position_index_dirty = True
        return np.asarray(_render_frame(eng, g)).copy()

    bad = 0
    for (h, w), levels in sorted(sizes.items()):
        shots = {c: shoot(h, w, c) for c in comps}
        clashes = [(a, b) for a, b in itertools.combinations(comps, 2)
                   if np.array_equal(shots[a], shots[b])]
        bad += len(clashes)
        if verbose:
            print(f"  {h:2d}x{w:2d} (cell {min(64 // h, 64 // w)}px, level "
                  f"{','.join(str(x) for x in levels)}): {len(shots)} "
                  f"compositions, {'OK' if not clashes else 'CLASH'}")
        for a, b in clashes:
            print(f"      IDENTICAL: {_comp_name(a)}  ==  {_comp_name(b)}")

    for (h, w), levels in sorted(sizes.items()):
        wrong = [c for c in comps
                 if not np.array_equal(shoot(h, w, c),
                                       shoot(h, w, c + ("nocoin",)))]
        bad += len(wrong)
        if verbose:
            print(f"  {h:2d}x{w:2d}: NoCoin is invisible on "
                  f"{len(comps) - len(wrong)}/{len(comps)} compositions")
        for c in wrong:
            print(f"      NoCoin CHANGED {_comp_name(c)}")

    for cell in sorted({min(64 // h, 64 // w) for h, w in sizes}):
        n = 64 // cell
        shots = {c: shoot(n, n, c) for c in comps}
        clashes = [(a, b, k) for a, b in itertools.permutations(comps, 2)
                   for k in range(1, 4)
                   if np.array_equal(np.ascontiguousarray(
                       np.rot90(shots[a], k=k)), shots[b])]
        invariant = [_comp_name(c) for c in comps
                     if all(np.array_equal(np.ascontiguousarray(
                         np.rot90(shots[c], k=k)), shots[c]) for k in range(4))]
        bad += len(clashes)
        if verbose:
            print(f"  cell {cell}px ({n}x{n}): {len(clashes)} composition(s) "
                  f"whose turn is another's art; turn-invariant: "
                  f"{', '.join(invariant)}")
        for a, b, k in clashes:
            print(f"      {_comp_name(a)} turned {k} == {_comp_name(b)}")
    return bad


# ---------------------------------------------------------------------------
# The augmentation evidence
# ---------------------------------------------------------------------------

def _symmetry(walk_presses: int = 200, verbose: bool = True) -> int:
    """Replay every level through the ADAPTER at every presentation it can draw
    and require the frames to be exactly the transform of the unaugmented ones.

    This is the evidence for the rotation augmentation. The structural argument
    is in the module docstring; this is the measurement. Two drives per (level,
    rotation): the PLAN, which must still win, and a seeded random WALK, which
    does what a plan never does -- shoves crates into spikes, walks into
    chickens and presses the unbound ACTION key -- so the frames of the states a
    plan avoids are checked too.
    """
    from adapters.puzzlescript_adapter import PuzzleScriptAdapter      # noqa: PLC0415

    _solver, game, expert, _solvable = _new()
    plans = {}
    for level in range(game.n_levels):
        game.set_level(level)
        plans[level] = list(expert.plan(game._engine, level))

    def drive(g, level, presses):
        g.set_level(level)
        frames = [np.asarray(g._current_frame)]
        for act in presses:
            fd = g.perform_action(ActionInput(id=act))
            frames.append(np.asarray(fd.frame[-1] if fd.frame
                                     else g._current_frame))
        return frames

    def transform(frame, k, hflip, vflip):
        out = np.rot90(frame, k=k)
        if hflip:
            out = np.fliplr(out)
        if vflip:
            out = np.flipud(out)
        return np.ascontiguousarray(out)

    bad = 0
    ref: dict = {}
    seen = set()
    #: Enough seeds to draw all four rotations on all seven levels; the adapter
    #: derives the rotation from (seed + level index), so a dozen seeds covers
    #: the 28 presentations several times over.
    for seed in range(12):
        g = PuzzleScriptAdapter(GAME_NAME, seed=seed)
        for level in range(g.n_levels):
            g.set_level(level)
            k = (g._rotation_k, g._hflip, g._vflip)
            seen.add(k)
            rng = random.Random(f"squeamish:symmetry:{level}")
            walk = [_ID_TO_GAMEACTION[rng.randrange(1, 6)]
                    for _ in range(walk_presses)]
            for tag, presses in (
                    ("plan", [screen_action(d, *k) for d in plans[level]]),
                    ("walk", [inverse_remap_action_full(a, *k) for a in walk])):
                frames = drive(g, level, presses)
                if tag == "plan" and g._state != GameState.WIN:
                    print(f"    seed {seed} L{level} {k}: plan did not win")
                    bad += 1
                key = (level, tag)
                if key not in ref:
                    if k != (0, False, False):
                        continue          # nothing to compare against yet
                    ref[key] = frames
                    continue
                if any(not np.array_equal(transform(a, *k), b)
                       for a, b in zip(ref[key], frames)):
                    print(f"    seed {seed} L{level} {k}: {tag} frames are not "
                          f"the transform of the unaugmented ones")
                    bad += 1
    if verbose:
        print(f"  {len(seen)} presentations drawn: {sorted(seen)}")
    return bad


if __name__ == "__main__":
    if "--survive" in sys.argv:
        #: Handled here rather than through `BaseSolver.build_argparser` because
        #: `PSExpert.__init__` reads `plan_cache_path`, so the switch has to be
        #: thrown before anything is constructed -- and because it applies to
        #: the report flags below too, which never reach the argparser.
        sys.argv.remove("--survive")
        SqueamishChickensExpert.survive = True
        SqueamishChickensExpert.plan_cache_path = SURVIVE_PLAN_CACHE
    if "--plans" in sys.argv:
        sys.exit(_plans())
    if "--selfcheck" in sys.argv:
        violations = _selfcheck()
        print(f"selfcheck: {violations} violations")
        sys.exit(1 if violations else 0)
    if "--engine" in sys.argv:
        violations = _engine()
        print(f"engine enumeration: {violations} disagreements")
        sys.exit(1 if violations else 0)
    if "--audit" in sys.argv:
        collisions = _audit()
        print(f"audit: {collisions} colliding compositions")
        sys.exit(1 if collisions else 0)
    if "--symmetry" in sys.argv:
        violations = _symmetry()
        print(f"symmetry: {violations} violations")
        sys.exit(1 if violations else 0)
    sys.exit(SqueamishChickensSolver.main())
