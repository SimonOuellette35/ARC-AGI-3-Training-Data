"""Generate Phase-1 training data for the PuzzleScript game ps:goblin_hooblob
("Goblin Hooblob", Evan Kuhn and Michael Franklin).

The harness -- the rotation contract, the trajectory recorder, the plan memo and
its disk cache, and the BaseSolver plumbing -- lives in
`solvers/common/ps_astar.py`, shared with the other ps: generators. This file is
the game-specific part: a native model of the sixteen rules, the exact search
that plans over it, the optimal-action oracle, and the two rendering fixes the
game needed.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_goblin_hooblob",
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
augmented view, so replaying the recorded actions reproduces the recorded
frames exactly. Every expert step also carries the full set of equally-optimal
presses.

The game
--------
A goblin has to reach his gold. ``All Player on Target`` is the win -- the
crates are furniture, not cargo -- with a second condition, ``No Splatter on
Background``, that is really a *lose* condition: Splatter is only ever created
by the goblin dying and no rule removes it, so a level in which the player has
died can never be won again.

Three mechanics, and they interact:

  * **CratePush** is the sokoban crate: ``[> Player | CratePush] -> [> Player |
    > CratePush]``. Pushes never chain (nothing gives a second crate a force),
    so a crate against a crate or a wall cancels the turn.
  * **CratePull** is the sticky crate: ``[< Player | CratePull] -> [< Player |
    < CratePull]``. It is the same rule read backwards -- the crate BEHIND the
    goblin follows him when he walks away from it -- and it appears in no push
    rule at all, so a pull crate can never be shoved, only dragged, and walking
    into one is simply refused.
  * **Hoomans** patrol. A Hooman has a facing (Up/Down/Left/Right, four
    distinct objects) and takes exactly one step in it **every time the goblin
    moves** -- the eight movement rules are all conditioned on ``> Player``,
    with a second variant for the turn the goblin spends pushing. A Hooman that
    would walk into a Wall or a Crate turns around instead and stays put
    (``[> HoomanUp | Object] -> [HoomanDown | Object]``); a Hooman that would
    walk into another Hooman just fails to move and tries again next turn.
    Touching one in either direction is fatal: ``[> Hooman | Player]`` and
    ``[> Player | Hooman]`` both replace the goblin with a Splatter.

Two consequences shape every plan here, and neither is visible in the rule
listing:

  * **There is no wait move.** A press the rules refuse -- into a wall, into a
    pull crate, into a crate with no room behind it -- matches none of the
    Hooman rules either (they all require the goblin's destination to be free
    of Objects), so the whole turn is a no-op and the patrols do not advance.
    The only way to spend time is to walk, so timing a patrol means finding a
    detour of the right LENGTH, and the parity of that detour is what several
    of these levels are really about.
  * **The patrol is a function of the crates.** A Hooman turns around at
    whatever is in front of it *at rule time*, so shoving a crate into (or out
    of) a corridor re-times every patrol that uses it -- and a Hooman wedged
    between two Objects (level 5's right wall) oscillates in place forever,
    permanently blocking its cell, until the crate beside it is moved away.

The levels
----------
Six shipped boards, all six solved and every plan PROVED shortest:

    level  size   crates  hoomans  plan  ties  search  what it is
      0    7x10     3        0       13   23%    0.0s  three crates in a slot,
                                                       and only the middle one
                                                       is in the way
      1    9x11     5        0       22    9%    0.0s  the sticky crate: drag
                                                       it out of the doorway
      2    6x15     0        2       18   33%    0.0s  two patrols, open room
      3    9x15     6        1       31   32%    0.7s  a patrolled hall, and a
                                                       crate pen off to one side
      4    9x 7     7        0       24    0%    0.1s  a stack of crates in a
                                                       shaft, one route down
      5    9x15    19        5       36    6%   72.4s  THE VAULT: a lattice of
                                                       pull crates, three
                                                       patrols wedged in place
                                                       and two on rounds

144 presses over the six, the longest 36 against the adapter's 200-action
per-level budget, so this game needs no `games/` step-limit wrapper the way
ps:count_mover does. ``ties`` is the share of steps with more than one
equally-optimal press, ``search`` the one-time cost, cached to
``data/goblin_hooblob_plans.json`` so `parallelize_generator`'s shards do not
each re-derive it. Delete that file to re-derive.

Levels 0-4 were also solved independently by `PSExpert`'s engine-blackbox A*
over the real interpreter, which agreed on all five lengths (13/22/18/31/24) --
the cross-check that the model below is not solving a different game.

Expert solver
-------------
Not a search over the interpreter. `_Board` is a native model of all sixteen
rules, including PuzzleScript's simultaneous-movement resolution (chains,
deferred blockers and the multi-way conflict that cancels two objects claiming
one cell), because the mechanic needs it: hoomans, crates and the goblin all
share one collision layer and all move on the same tick.

Why the model at all -- `PSExpert`'s engine-blackbox A* solves levels 0-4 in
under 12s and is the obvious first choice. It cannot reach level 5: the
interpreter runs at ~4.6k steps/s here and every queued node holds a full grid
snapshot, so the search was at 5 GB of RSS and still running after 25 minutes.
The model is ~100x faster per step and far smaller per state, which is what
makes an EXACT plan possible on the vault -- 36 presses, proved shortest, in
72s and under 400 MB.

The vault is why the search is the three-pass one in `_Board.optimal_plan`
rather than a plain A*. Its reachable space grows by a factor of ~1.46 per
press (an exhaustive BFS passes 4M states at depth 35 and the goal is at 36),
and the heuristic below cannot help with the part that matters -- dragging one
crate out of a lattice moves the goblin no closer to the gold -- so an
unweighted A* has nothing to steer with. What works is to get *a* win cheaply
and then use its length as a pruning bound.

The model is fuzz-checked against the interpreter, not trusted: ``--fuzz``
plays boards move-for-move against the real engine and compares the goblin,
both crate sets, every hooman's cell AND facing, the death flag and the win
flag after every press. It runs both random dense boards and the shipped levels
from every prefix of their own plans, and it prints COVERAGE COUNTERS rather
than only a verdict -- 30582 presses over 1644 boards, no mismatches, with 500
pushes, 703 pulls, 6771 turn-arounds, 6828 patrol stalls (a patrol that could
have moved and was blocked by another mover), 1158 deaths of both kinds and 177
wins. The counters are the point: "no mismatches" over presses that never
pushed a crate is not evidence about pushing. The one real bug the fuzz caught
-- a patrol at the board EDGE turning around, which it must not -- is reachable
on no shipped level, since all six are walled in.

Optimal-action sets
-------------------
Measured, not inferred. A* (unweighted, admissible heuristic) proves the
shortest length ``d*``; a second layered BFS bounded by ``depth + h <= d*``
then keeps exactly the states on shortest paths, and one backward sweep over
that DAG gives, for every such state, the exact set of presses that still
finish in ``d*``. See `_Board.optimal_plan`.

That matters more here than in a push game: this goblin walks a lot, and a walk
across an open room is a free choice of interleaving -- a third of level 2's
and level 3's presses take either axis and still finish in the same number of
moves. Labelling one arbitrary staircase as the only right answer would train a
coin flip the policy cannot win. It is also why the sets have to be MEASURED
rather than read off a walk BFS the way `PSPushExpert.annotate_walks` does:
walking two cells of an L in the other order re-times every patrol against the
goblin, so some of those interleavings are lethal and some merely late, and
only a re-solve can tell which.

The rendering fixes
-------------------
Both are in the game file (see the header comment in
data/puzzlescript_games/Goblin_Hooblob.txt), and ``--audit`` is the regression
test for them:

  * **The four Hooman facings shipped as pixel-identical sprites.** The one
    piece of state that decides whether a press is fatal was not on screen at
    all, so no observation-conditioned policy could have learned the mechanic
    (see the badge_placement failure mode). Each facing now carries a two-pixel
    black pip on the side it walks toward. The pip is a pair straddling the
    centre because `_render_cell_sprite` takes CENTRED nearest samples: at the
    4px cells three of these levels render at, sprite row/column 2 is dropped
    and a single centred pip would vanish (the trap ps:explod's fuse hit). The
    four pip positions are an orbit of the whole 8-element symmetry group, so
    a hooman's pip points the way it moves ON SCREEN at every augmentation.
  * **Both crates were solid 5x5 blocks**, so a crate parked on the Target hid
    the one square the win condition names -- and the exploration prefix parks
    things on it. Their corners are transparent now, exactly as in ps:escape.

Augmentation
------------
The engine state after reset is identical for every seed (levels are fixed
ASCII maps), so the only per-(seed, level) variables are the presentation
augmentations: the frame rotation (rotation_k in {0,1,2,3}) plus an independent
horizontal and vertical flip, each with the matching directional action remap
(`PuzzleScriptAdapter._FLIP_GAMES`), which together re-sample the board's
8-element symmetry group. 6 levels x 16 presentations = 96. The expert plan is
therefore seed-independent: solved once per level, cached, and replayed per
seed with that seed's remapped screen actions.

The flips are sound for the same reasons ESCAPE!'s are -- gravity-free, every
rule stated with the relative ``>`` force, a win condition that names no
direction, screen-relative input -- with one extra thing to check that no other
game in `_FLIP_GAMES` has: this game's art IS directional now. It survives
because the pips were drawn as an orbit of the full group rather than of the
rotation alone. Each pair straddles the sprite's centre and is symmetric about
its own axis, so a mirror maps facing art to facing art exactly as it maps the
motion: vertically flipped, HoomanUp is drawn with its pip at the bottom and
walks down the screen. There is deliberately no colour augmentation -- a crate
on the gold is read as the target's corners showing through the crate's
transparent ones, which a flattening recolor would erase.

Usage (run from the repo root):
    python solvers/generate_goblin_hooblob_training.py --episodes 200 \
        --out data/training_multi_level/goblin_hooblob

    python solvers/generate_goblin_hooblob_training.py --plans   # level report
    python solvers/generate_goblin_hooblob_training.py --audit   # rendering
    python solvers/generate_goblin_hooblob_training.py --fuzz    # model check
"""

from __future__ import annotations

import heapq
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solvers.common.ps_astar import (PSAStarSolver, PSExpert,      # noqa: E402
                                     Plan)

_REPO_ROOT = Path(__file__).resolve().parent.parent

GAME_NAME = "Goblin_Hooblob"

#: Disk cache of each level's start plan and its optimal-action sets. The
#: searches are seed-independent, so without a file on disk every shard
#: `parallelize_generator` starts would re-derive all of them. Delete to
#: re-derive.
PLAN_CACHE: Path = _REPO_ROOT / "data" / "goblin_hooblob_plans.json"

#: Engine direction names, in the order the model indexes them. ``d ^ 1`` is
#: the reverse of ``d`` (which the pull rule and the turn-around rule both
#: need), so up/down and left/right have to be adjacent pairs.
DIRS = ("up", "down", "left", "right")
_DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))

#: Heuristic weight for the upper-bound probe in `_Board.optimal_plan`. High
#: on purpose: the probe's job is to reach *a* win as fast as possible so the
#: exact pass has something to prune with, and its own plan length is never
#: shipped unless the exact pass runs out of budget.
PROBE_WEIGHT = 8

#: Object kinds the model tracks. Walls are static and kept out of the
#: occupancy map; everything else shares one collision layer, so a cell holds
#: at most one of these.
PUSH, PULL, PLAYER, HOO = range(4)


class _Board:
    """Native model of Goblin Hooblob, and the search over it.

    STATE is ``(player, push, pull, hoomans)``: the goblin's cell index, the
    two crate sets as bitmasks over cell indices, and the patrols as a sorted
    tuple of ``cell * 4 + facing``. Walls and targets are static and live on
    the board, not in the state. Death is not a state -- `step` returns None
    for it, because Splatter is permanent and the win condition forbids it, so
    a dead goblin is a state no search should ever expand.

    THE TICK, in the interpreter's own rule order:

    1. ``[> Player | CratePush] -> [> Player | > CratePush]`` -- the crate the
       goblin walks into gets his force.
    2. ``[< Player | CratePull] -> [< Player | < CratePull]`` -- the sticky
       crate BEHIND him gets it too.
    3. the eight patrol rules -- if the goblin's destination is free of Objects
       (or holds a crate with a free cell behind it), every Hooman gets a force
       in its own facing. Note what this does *not* test: a Hooman or a
       Splatter in the goblin's way is not an ``Object``, so the patrols still
       tick on the turn he dies.
    4. ``[> Hooman | Player]`` then ``[> Player | Hooman]`` -- either contact
       kills. The first consumes the killer's force (it stays put); the second
       cannot fire after it, because there is no Player left to match.
    5. ``[> HoomanUp | Object] -> [HoomanDown | Object]`` and its three
       siblings -- a patrol facing a Wall or a Crate turns around and loses its
       force, so it spends the turn in place. It cannot turn twice: the
       rewritten object has no force for the next rule to match.

    and then PuzzleScript's force resolution, which is not a detail here --
    everything that moves shares one collision layer and moves on the same
    tick, so it decides whether the push into a patrol's cell goes through
    (`_resolve`).
    """

    def __init__(self, walls, targets, height, width):
        self.h, self.w = height, width
        self.walls = frozenset(walls)
        self.targets = frozenset(targets)
        n = height * width
        # nbr[d][cell] -> cell index, or -1 off the board.
        self.nbr = []
        for dr, dc in _DELTAS:
            row = [-1] * n
            for r in range(height):
                for c in range(width):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width:
                        row[r * width + c] = nr * width + nc
            self.nbr.append(row)
        self._frozen_cache: dict = {}

    # -- the tick -------------------------------------------------------------
    def step(self, state, d):
        """One press. Returns the next state, or None if the goblin died."""
        player, push, pull, hoo = state
        nbr = self.nbr
        walls = self.walls

        occ = {}
        for c in _bits(push):
            occ[c] = PUSH
        for c in _bits(pull):
            occ[c] = PULL
        hdir = {}
        for packed in hoo:
            cell = packed >> 2
            occ[cell] = HOO
            hdir[cell] = packed & 3
        occ[player] = PLAYER

        front = nbr[d][player]
        # Force insertion order mirrors the rule order (see the class doc); the
        # resolution below is order-independent, but keeping it makes a
        # divergence in `--fuzz` mean a rule bug rather than a tie broken
        # differently.
        forces = {player: d}
        if front >= 0 and occ.get(front) == PUSH:                    # rule 1
            forces[front] = d
        back = nbr[d ^ 1][player]
        if back >= 0 and occ.get(back) == PULL:                      # rule 2
            forces[back] = d

        tick = False                                                 # rule 3
        if front >= 0 and front not in walls:
            kind = occ.get(front)
            if kind != PUSH and kind != PULL:
                tick = True
            elif kind == PUSH:
                beyond = nbr[d][front]
                if (beyond >= 0 and beyond not in walls
                        and occ.get(beyond) not in (PUSH, PULL)):
                    tick = True
        if tick:
            for cell, hd in hdir.items():
                forces[cell] = hd

            for cell, hd in hdir.items():                            # rule 4a
                if cell in forces and nbr[hd][cell] == player:
                    return None
        if front >= 0 and occ.get(front) == HOO:                     # rule 4b
            return None

        for cell in list(hdir):                                      # rule 5
            if cell not in forces:
                continue
            hd = hdir[cell]
            ahead = nbr[hd][cell]
            # Off the board is NOT a turn-around: the rule is
            # ``[> HoomanUp | Object]`` and the board edge is not a cell, so
            # nothing matches and the patrol keeps its facing -- the
            # resolution below then refuses the move anyway. (Every shipped
            # level is walled in, so this only shows up under `--fuzz`, which
            # is exactly what it is for.)
            if ahead >= 0 and (ahead in walls
                               or occ.get(ahead) in (PUSH, PULL)):
                forces.pop(cell)
                hdir[cell] = hd ^ 1

        self._resolve(occ, hdir, forces)

        new_player = -1
        new_push = new_pull = 0
        new_hoo = []
        for cell, kind in occ.items():
            if kind == PLAYER:
                new_player = cell
            elif kind == PUSH:
                new_push |= 1 << cell
            elif kind == PULL:
                new_pull |= 1 << cell
            else:
                new_hoo.append(cell * 4 + hdir[cell])
        new_hoo.sort()
        return (new_player, new_push, new_pull, tuple(new_hoo))

    def _resolve(self, occ, hdir, forces):
        """PuzzleScript's simultaneous-movement resolution, in place.

        A faithful port of `PSEngine._resolve_forces` restricted to this game's
        single collision layer. Four things it does that a naive "move if the
        destination is empty" does not, and all four are load-bearing here:

          * a chain of objects moving the same way moves together (a patrol
            following a patrol; the goblin behind the crate he is pushing);
          * a mover blocked by a mover heading a DIFFERENT way is deferred, not
            cancelled, and retried on the next pass -- it may be about to
            vacate;
          * two independent chains claiming one cell cancel EACH OTHER, so two
            patrols walking into the same square both stop, and a crate shoved
            at a square a patrol is entering is refused (which cancels the
            goblin's move with it);
          * a deadlock -- two objects trying to swap, the commonest way for two
            patrols to meet head-on -- ends with nothing moving at all.
        """
        nbr, walls = self.nbr, self.walls
        for _ in range(20):
            moved_any = False
            deferred_any = False
            forces_at_start = dict(forces)
            resolved = set()
            movable = []
            for cell, dr in list(forces.items()):
                if cell in resolved:
                    continue
                if cell not in occ:
                    forces.pop(cell, None)
                    resolved.add(cell)
                    continue
                chain = [cell]
                cur = cell
                chain_free = False
                blocker_is_mover = False
                while True:
                    nxt = nbr[dr][cur]
                    if nxt < 0 or nxt in walls:
                        break
                    if nxt not in occ:
                        chain_free = True
                        break
                    if forces.get(nxt) == dr:
                        chain.append(nxt)
                        cur = nxt
                    else:
                        blocker_is_mover = nxt in forces
                        break
                if chain_free:
                    movable.append((chain, dr))
                elif blocker_is_mover:
                    deferred_any = True
                else:
                    for c in chain:
                        forces.pop(c, None)
                        resolved.add(c)

            non_head = {}
            for i, (chain, _dr) in enumerate(movable):
                for ent in chain[1:]:
                    non_head[ent] = i
            subsumed = {i for i, (chain, _dr) in enumerate(movable)
                        if chain[0] in non_head}

            claims = {}
            conflicting = set()
            for i, (chain, dr) in enumerate(movable):
                if i in subsumed:
                    continue
                for c in chain:
                    key = nbr[dr][c]
                    other = claims.get(key)
                    if other is not None and other != i:
                        conflicting.add(i)
                        conflicting.add(other)
                    else:
                        claims[key] = i

            for i, (chain, dr) in enumerate(movable):
                if i in subsumed:
                    continue
                if i in conflicting:
                    for c in chain:
                        forces.pop(c, None)
                        resolved.add(c)
                    continue
                for c in reversed(chain):
                    kind = occ.pop(c)
                    dest = nbr[dr][c]
                    occ[dest] = kind
                    if kind == HOO:
                        hdir[dest] = hdir.pop(c)
                    forces.pop(c, None)
                    resolved.add(c)
                    moved_any = True

            if not moved_any:
                if deferred_any and forces == forces_at_start:
                    forces.clear()
                break

    # -- search ---------------------------------------------------------------
    def won(self, state):
        return state[0] in self.targets

    def heuristic(self, state):
        """Shortest walk from the goblin to the gold over cells no wall and no
        FROZEN crate occupies, in presses. Admissible, and that is what lets
        the optimal-action sets below be a measurement rather than a guess.

        Loose crates are ignored: a push crate's cell is crossed in a single
        press by shoving it, so charging anything for one would break the
        bound, and a pull crate can be dragged out of the way. Patrols are
        ignored: they can only ever cost presses, never save them.

        Frozen crates are the one place the estimate gets to be sharper.
        Consulting only WALLS, a push crate at ``X`` can leave along ``d`` only
        if the goblin can stand at ``X - d`` and the crate can land on
        ``X + d``; a pull crate at ``X`` can only leave along ``d`` if the
        goblin can stand at ``X + d`` and step on to ``X + 2d``, dragging it.
        A crate with no such ``d`` can never move again by any sequence, so its
        cell is a permanent wall and removing it from the graph stays
        admissible. Level 4 is a stack of them and level 5's vault is built out
        of them."""
        player = state[0]
        if player in self.targets:
            return 0
        blocked = self._frozen(state[1], state[2])
        nbr = self.nbr
        targets = self.targets
        seen = {player}
        frontier = [player]
        d = 0
        while frontier:
            d += 1
            nxt = []
            for cell in frontier:
                for dd in range(4):
                    n = nbr[dd][cell]
                    if n < 0 or n in seen or n in blocked:
                        continue
                    if n in targets:
                        return d
                    seen.add(n)
                    nxt.append(n)
            frontier = nxt
        return 1 << 20                # the gold is walled off for good

    def _frozen(self, push, pull):
        """``walls | crates that can never move again``; memoized on the crate
        sets, because `heuristic` runs on every generated state."""
        key = (push, pull)
        got = self._frozen_cache.get(key)
        if got is not None:
            return got
        walls, nbr = self.walls, self.nbr
        blocked = set(walls)
        for cell in _bits(push):
            for d in range(4):
                stand, land = nbr[d ^ 1][cell], nbr[d][cell]
                if (stand >= 0 and stand not in walls
                        and land >= 0 and land not in walls):
                    break
            else:
                blocked.add(cell)
        for cell in _bits(pull):
            for d in range(4):
                stand = nbr[d][cell]
                land = nbr[d][stand] if stand >= 0 else -1
                if (stand >= 0 and stand not in walls
                        and land >= 0 and land not in walls):
                    break
            else:
                blocked.add(cell)
        self._frozen_cache[key] = blocked
        return blocked

    def astar(self, start, node_cap=4_000_000, weight=1, bound=None):
        """``(plan, closed)``: a press sequence to the gold, and whether the
        search closed rather than running out of budget.

        ``weight`` above 1 is a probe -- it finds a winning plan far faster and
        that plan is not shortest, but its LENGTH is an upper bound the exact
        pass can prune with, which is the only thing it is used for here.

        ``bound`` drops every node whose ``g + h`` reaches it. With an
        admissible heuristic that keeps exactly the states from which a plan
        shorter than ``bound`` could still exist, so a bounded search that
        CLOSES (``closed`` true, no plan) is a proof that ``bound`` cannot be
        beaten -- which is how a probe's plan gets certified as shortest
        without ever running an unbounded search.
        """
        if self.won(start):
            return [], True
        counter = 0
        pq = [(weight * self.heuristic(start), 0, counter, start)]
        best = {start: 0}
        came = {}
        nodes = 0
        while pq:
            _f, g, _c, state = heapq.heappop(pq)
            if g > best.get(state, 1 << 30):
                continue
            for d in range(4):
                nxt = self.step(state, d)
                nodes += 1
                if nxt is None or nxt == state:
                    continue
                if self.won(nxt):
                    path = [d]
                    cur = state
                    while cur in came:
                        cur, pd = came[cur]
                        path.append(pd)
                    path.reverse()
                    return path, True
                ng = g + 1
                if best.get(nxt, 1 << 30) <= ng:
                    continue
                h = self.heuristic(nxt)
                if h >= (1 << 20):
                    continue          # the gold is sealed off: never a win
                if bound is not None and ng + h >= bound:
                    continue
                best[nxt] = ng
                came[nxt] = (state, d)
                counter += 1
                heapq.heappush(pq, (ng + weight * h, ng, counter, nxt))
            if nodes >= node_cap:
                return None, False
        return None, True

    def optimal_plan(self, start, node_cap=4_000_000):
        """``(plan, optsets, proven)``: a winning plan, the set of equally good
        presses at each of its steps, and whether the plan is PROVED shortest.
        ``(None, None, False)`` when the board cannot be won inside the budget.

        Three sweeps, and the first two exist because a plain unweighted A* is
        not affordable on the last level -- its space grows ~1.46x per press
        and passes 4M states by depth 35, with the goal at 36.

        1. **Probe.** Weighted A* (``PROBE_WEIGHT``) finds *a* win fast. Its
           length is not shortest and is never used as one; it is used as an
           upper bound.
        2. **Prove.** Unweighted A* bounded by that length, repeated on each
           improvement. A bounded run that CLOSES without finding anything
           shorter is a proof (the heuristic is admissible), so the bound the
           probe supplied is what makes the proof cheap: level 5 goes 38 -> 36
           -> closed in ~55s and 350 MB, where the unbounded search on the same
           model does not finish.
        3. **Label.** A layered BFS from the start pruned by
           ``depth + heuristic > d*``, which keeps exactly the shortest-path
           DAG, plus one backward pass marking the states that can still finish
           in the presses they have left. A press is optimal at a state iff it
           lands on one of those (or wins outright on the last move). Both
           sweeps measure the same distance with no dedup or macro
           approximation between them, so the sets are exact.

        A plan the prove step could not certify is still a genuine win -- the
        interpreter walks it -- and is labelled with the press it took, never
        with nothing (see the always-emit-optimal-targets rule).
        """
        plan, _closed = self.astar(start, node_cap, weight=PROBE_WEIGHT)
        if plan is None:
            return None, None, False
        while True:
            better, closed = self.astar(start, node_cap, weight=1,
                                        bound=len(plan))
            if better is None:
                proven = closed
                break
            plan = better
        if not proven:
            return plan, [[d] for d in plan], False

        limit = len(plan)
        # Forward: the layers of the shortest-path DAG, with each state's
        # successor list. A state is kept at its FIRST depth only -- that is
        # its true distance from the start, and re-reaching it later can never
        # be on a shortest path.
        depth = {start: 0}
        layers = [[start]]
        succ: dict = {}
        wins: set = set()
        for i in range(limit):
            nxt_layer = []
            for state in layers[i]:
                out = []
                for d in range(4):
                    nxt = self.step(state, d)
                    if nxt is None or nxt == state:
                        continue
                    if self.won(nxt):
                        if i + 1 == limit:
                            wins.add((state, d))
                        continue
                    if i + 1 >= limit:
                        continue
                    if nxt in depth:
                        if depth[nxt] == i + 1:
                            out.append((d, nxt))
                        continue
                    if i + 1 + self.heuristic(nxt) > limit:
                        continue
                    depth[nxt] = i + 1
                    out.append((d, nxt))
                    nxt_layer.append(nxt)
                succ[state] = out
            layers.append(nxt_layer)

        # Backward: which states still finish in the presses they have left.
        good: dict = {}
        for i in range(limit - 1, -1, -1):
            for state in layers[i]:
                best = [d for d, nxt in succ[state] if good.get(nxt)]
                if i + 1 == limit:
                    best += [d for d in range(4) if (state, d) in wins]
                if best:
                    good[state] = sorted(set(best))

        optsets, cur = [], start
        for i in range(limit):
            sets = good.get(cur)
            if not sets or plan[i] not in sets:
                # Unreachable: the plan is a proved-shortest path, so every
                # state on it is in the DAG and its own press is in the set.
                # Fall back to labelling what the expert did rather than
                # shipping a set that does not contain it.
                return plan, [[d] for d in plan], True
            optsets.append(sets)
            cur = self.step(cur, plan[i])
        return plan, optsets, True


def _bits(mask):
    """The set bit indices of ``mask``, low to high."""
    while mask:
        low = mask & -mask
        yield low.bit_length() - 1
        mask ^= low


# ---------------------------------------------------------------------------
# Reading a board off the interpreter
# ---------------------------------------------------------------------------

def board_from_engine(eng, game):
    """``(board, state)`` for the engine's current grid.

    The model is built from the live interpreter rather than from the level
    text, so a level edited in the .txt needs no second definition here, and
    the fuzz harness can start both sides from the same random board."""
    idx = game.obj_name_to_idx
    wall, target = idx["wall"], idx["target"]
    push_id, pull_id = idx["cratepush"], idx["cratepull"]
    player_id, splat_id = idx["player"], idx["splatter"]
    facing = {idx["hoomanup"]: 0, idx["hoomandown"]: 1,
              idx["hoomanleft"]: 2, idx["hoomanright"]: 3}
    stationary = idx["hoomanstationary"]

    h, w = eng.height, eng.width
    walls, targets = [], []
    push = pull = 0
    player = -1
    hoo = []
    for r in range(h):
        for c in range(w):
            cell = eng.grid[r][c]
            i = r * w + c
            if wall in cell:
                walls.append(i)
            if target in cell:
                targets.append(i)
            if push_id in cell:
                push |= 1 << i
            if pull_id in cell:
                pull |= 1 << i
            if player_id in cell:
                player = i
            if splat_id in cell:
                raise ValueError("board already holds a Splatter")
            if stationary in cell:
                # No level ships one and no rule creates one; it has no
                # movement rule either, so it would be a permanent obstacle
                # that still kills on contact. Refuse rather than model a
                # piece the game never shows.
                raise ValueError("HoomanStationary is not modelled")
            for oid, d in facing.items():
                if oid in cell:
                    hoo.append(i * 4 + d)
    hoo.sort()
    return _Board(walls, targets, h, w), (player, push, pull, tuple(hoo))


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class GoblinHooblobExpert(PSExpert):
    """Plans on `_Board` and lets the interpreter certify.

    `PSExpert` supplies everything around the search -- the plan memo, the disk
    cache with its staleness check, and the snapshot/restore discipline -- so
    all this overrides is `_search`, which builds the model off the engine's
    current grid and returns a `Plan` carrying the measured optimal sets.
    `heuristic` is never called (the model owns its own), and says so."""

    directions = list(DIRS)
    scope_by_level = True
    plan_cache_path = PLAN_CACHE

    def setup(self) -> None:
        # Dynamic objects only (walls and targets are static in this game and
        # `scope_by_level` keeps the memo honest across levels), plus the
        # Splatter: a dead board is a distinct, unwinnable state.
        self._dyn = {self.g.obj_name_to_idx[n] for n in
                     ("player", "cratepush", "cratepull", "splatter",
                      "hoomanup", "hoomandown", "hoomanleft", "hoomanright")}

    def heuristic(self, eng):
        raise AssertionError("the model plans; PSExpert's A* is unused here")

    def _key(self, eng):
        dyn = self._dyn
        return frozenset(
            (r, c, o)
            for r, row in enumerate(eng.grid)
            for c, cell in enumerate(row)
            for o in (cell & dyn))

    def _search(self, eng):
        board, state = board_from_engine(eng, self.g)
        # The third value is whether the plan was PROVED shortest. It is not
        # kept: a winning plan is recorded either way, its labels are already
        # correct for both cases, and the flag would not survive the disk
        # cache's ``{start, plan, optsets}`` format. `--plans` re-derives and
        # reports it.
        plan, optsets, _proven = board.optimal_plan(state, self.node_cap)
        if plan is None:
            return None
        return Plan([DIRS[d] for d in plan],
                    [[DIRS[d] for d in s] for s in optsets])


class GoblinHooblobSolver(PSAStarSolver):
    game_id = "puzzlescript_goblin_hooblob"
    game_name = GAME_NAME
    expert_cls = GoblinHooblobExpert

    #: Record against the GAME FOLDER's adapter -- the object `game_envs` hands
    #: a live agent -- so a step cap or sprite patch added there later cannot
    #: silently make this generator tape a game nobody plays.
    game_module_id = "ps:goblin_hooblob"

    #: Generated-state budget per search, passed through to the expert. A
    #: budget, not a dial: the shipped levels do not come near it, and a level
    #: that did would be reporting that its plan is not provably shortest
    #: rather than that it needs patience.
    node_cap = 6_000_000

    #: The longest plan is well under the adapter's 200-action per-level budget
    #: (which would otherwise flip a level to GAME_OVER mid-plan); the rest is
    #: room for the RESET exploration prefix and the re-plan after it.
    max_steps = 150

    def prepare_expert(self, game, expert) -> None:
        """Plan every level before `discover_solvable` asks for it -- the same
        work either way, but it makes the one-time cost visible as startup and
        fills the disk cache in one pass."""
        for level in range(game.n_levels):
            if level in self.skip_levels:
                continue
            game.set_level(level)
            expert.plan(game._engine, level)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _report() -> int:
    """Per-level size, piece counts, plan length, optimality and tie coverage.

    Plans are re-derived rather than read from `PLAN_CACHE`: the cache stores
    the plan and its labels but not whether the search PROVED the length
    shortest, and that column is the whole point of the report. Each plan is
    then replayed through the real interpreter, which is the only thing that
    certifies the model."""
    import time
    game = GoblinHooblobSolver().make_game(0)
    eng, g = game._engine, game._game
    total = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board, state = board_from_engine(eng, g)
        crates = bin(state[1]).count("1") + bin(state[2]).count("1")
        t0 = time.time()
        plan, optsets, proven = board.optimal_plan(
            state, GoblinHooblobSolver.node_cap)
        dt = time.time() - t0
        head = (f"level {level:2d}: {eng.height:2d}x{eng.width:2d} "
                f"{crates:2d} crates {len(state[3]):d} hoomans")
        if plan is None:
            print(f"{head} -- NO PLAN ({dt:7.1f}s)")
            continue
        ties = sum(1 for s in optsets if len(s) > 1)
        total += len(plan)
        room = "ok" if len(plan) < game._max_steps else "OVER BUDGET"
        print(f"{head}, {len(plan):3d} moves "
              f"({'proved shortest' if proven else 'NOT PROVED SHORTEST'}, "
              f"budget {game._max_steps}, {room}), "
              f"{ties:3d} tie steps ({ties / max(1, len(plan)):3.0%}), "
              f"{dt:7.1f}s")
        _verify(game, level, [DIRS[d] for d in plan])
    print(f"total {total} moves")
    return 0


def _verify(game, level, plan) -> None:
    """Replay a plan through the REAL interpreter: it must win on the last
    press and on no earlier one."""
    game.set_level(level)
    eng = game._engine
    for i, d in enumerate(plan):
        eng.step(d)
        if eng.check_win():
            assert i == len(plan) - 1, f"level {level} won early at {i}"
            return
    raise AssertionError(f"level {level}: plan does not win")


def _audit() -> int:
    """Assert every cell COMPOSITION is distinct at every cell size in use.

    Composition, not object: a crate ON the gold has to differ from a crate on
    the floor (it is the difference between a winnable board and a dead one),
    and each of the four Hooman facings has to differ from the other three (it
    is the difference between a safe press and a fatal one). The whole board is
    filled with the composition and whole frames are compared -- cropping one
    cell out by arithmetic, as the older audits in this tree do, is wrong for
    any board `_render_frame` upscales, which is all of them."""
    import itertools

    import numpy as np

    from adapters.puzzlescript_adapter import _render_frame

    game = GoblinHooblobSolver().make_game(0)
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    comps = {"floor": (), "wall": ("wall",), "target": ("target",)}
    for name in ("cratepush", "cratepull", "player", "splatter", "hoomanup",
                 "hoomandown", "hoomanleft", "hoomanright"):
        comps[name] = (name,)
        comps[name + "_on_target"] = ("target", name)

    sizes: dict = {}
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
            shots[name] = np.asarray(_render_frame(eng, g))
        clashes = [(a, b) for a, b in itertools.combinations(shots, 2)
                   if np.array_equal(shots[a], shots[b])]
        bad += len(clashes)
        print(f"{h:2d}x{w:2d} (cell {64 // max(h, w)}px, levels "
              f"{','.join(str(x) for x in levels)}): "
              f"{'OK' if not clashes else 'IDENTICAL ' + str(clashes)}")
    print("audit clean" if not bad
          else f"AUDIT FAILED: {bad} indistinguishable pairs")
    return 0 if not bad else 1


def _fuzz(n_boards: int = 1500, n_steps: int = 60, seed: int = 0) -> int:
    """Differential fuzz: boards played move-for-move against the real
    interpreter, comparing the goblin, both crate sets, every patrol's cell AND
    facing, the death flag and the win flag after every press.

    TWO SOURCES OF BOARDS, and the game needs both.

    *Random dense boards*, because the shipped levels exercise almost none of
    the collision cases the resolution exists for: two patrols claiming one
    cell, a patrol following a patrol, a crate shoved at the square a patrol is
    walking into, a patrol against the board edge (which is NOT a turn-around
    -- that one was a real model bug and only random boards ever hit it, since
    every shipped level is walled in).

    *The shipped levels, from every prefix of their own solutions*, because
    random play on a 15-wide vault mostly wanders: the states a plan actually
    passes through are reached by planning, not by luck.

    The run prints COVERAGE COUNTERS rather than only a verdict -- "no
    mismatches" over presses that never pushed a crate is not evidence about
    pushing.
    """
    import random

    game = GoblinHooblobSolver().make_game(0)
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    # Crate-heavy and wall-light on purpose: the counters below are the
    # evidence this run produces, and a wall-heavy board spends its presses on
    # refusals, which prove nothing about pushing.
    names = ["background", "wall", "cratepush", "cratepull",
             "hoomanup", "hoomandown", "hoomanleft", "hoomanright", "target"]
    weights = [26, 5, 9, 9, 3, 3, 3, 3, 2]
    rng = random.Random(seed)
    cover = {"push": 0, "pull": 0, "refused": 0, "turn": 0, "patrol stall": 0,
             "death by patrol": 0, "death by contact": 0, "win": 0}
    mismatches = presses = boards = 0

    def _rollout(board, state, n):
        """Random presses from ``state`` with both sides in lockstep. The
        engine is assumed to be AT ``state`` already."""
        nonlocal mismatches, presses
        for _ in range(n):
            d = rng.randrange(4)
            before = state
            eng.step(DIRS[d])
            presses += 1
            got = board.step(before, d)
            try:
                _b, live = board_from_engine(eng, g)
                live_dead = live[0] < 0
            except ValueError:                # a Splatter: the goblin died
                live_dead, live = True, None
            if got is None:
                if not live_dead:
                    mismatches += 1
                    print(f"MISMATCH after {DIRS[d]}: model says the goblin "
                          f"died, the engine does not\n  from {before}")
                else:
                    front = board.nbr[d][before[0]]
                    hoo = {p >> 2 for p in before[3]}
                    cover["death by contact" if front in hoo
                          else "death by patrol"] += 1
                return
            if live_dead or got != live:
                mismatches += 1
                print(f"MISMATCH after {DIRS[d]}:\n  from   {before}\n"
                      f"  model  {got}\n  engine {live} dead={live_dead}")
                return
            if board.won(got) != bool(eng.check_win()):
                mismatches += 1
                print(f"WIN MISMATCH: model {board.won(got)} "
                      f"engine {eng.check_win()}")
                return
            if got == before:
                cover["refused"] += 1
            else:
                if got[1] != before[1]:
                    cover["push"] += 1
                if got[2] != before[2]:
                    cover["pull"] += 1
                old = {p >> 2: p & 3 for p in before[3]}
                new = {p >> 2: p & 3 for p in got[3]}
                cover["turn"] += sum(1 for c, f in new.items()
                                     if old.get(c) not in (None, f))
                cover["patrol stall"] += sum(
                    1 for c, f in new.items() if old.get(c) == f
                    and board.nbr[f][c] >= 0
                    and board.nbr[f][c] not in board.walls)
            state = got
            if board.won(state):
                cover["win"] += 1
                return

    # 1. random dense boards
    for _ in range(n_boards):
        h, w = rng.choice([4, 5, 6]), rng.choice([4, 5, 6])
        grid = [[{idx["background"], idx[rng.choices(names, weights)[0]]}
                 for _ in range(w)] for _ in range(h)]
        pr, pc = rng.randrange(h), rng.randrange(w)
        grid[pr][pc] = {idx["background"], idx["player"]}
        eng.grid = grid
        eng.height, eng.width = h, w
        eng._position_index_dirty = True
        eng._rule_noop_cache.clear()
        try:
            board, state = board_from_engine(eng, g)
        except ValueError:
            continue
        if board.won(state):
            continue
        boards += 1
        _rollout(board, state, n_steps)

    # 2. the shipped levels, from every prefix of their own plans
    expert = GoblinHooblobExpert(game, node_cap=GoblinHooblobSolver.node_cap)
    for level in range(game.n_levels):
        game.set_level(level)
        plan = expert.plan(eng, level)
        if plan is None:
            continue
        for cut in range(len(plan)):
            game.set_level(level)
            for d in plan[:cut]:
                eng.step(d)
            board, state = board_from_engine(eng, g)
            boards += 1
            _rollout(board, state, n_steps)

    print("coverage: " + ", ".join(f"{k} {v}" for k, v in cover.items()))
    print(f"{presses} presses over {boards} boards: "
          + ("no mismatches" if not mismatches
             else f"{mismatches} MISMATCHES"))
    return 0 if not mismatches else 1


if __name__ == "__main__":
    if "--plans" in sys.argv:
        sys.exit(_report())
    if "--audit" in sys.argv:
        sys.exit(_audit())
    if "--fuzz" in sys.argv:
        sys.exit(_fuzz())
    sys.exit(GoblinHooblobSolver.main())
