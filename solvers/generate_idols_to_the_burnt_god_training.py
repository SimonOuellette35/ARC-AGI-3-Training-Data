"""Generate Phase-1 training data for the PuzzleScript game
ps:idols_to_the_burnt_god ("IDOLS TO THE BURNT GOD", Edalcmagal).

The harness -- the rotation contract, the trajectory recorder, the plan memo and
its disk cache, and the BaseSolver plumbing -- lives in
`solvers/common/ps_astar.py`, shared with the other ps: generators. This file is
the game-specific part: a native model of the burn mechanic, the search that
plans over it, the optimal-action oracle, and the three rendering fixes the game
needed.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_idols_to_the_burnt_god",
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
An acolyte walks through a temple full of statues and burns them by touching
them. Every statue has a burn ladder and the win condition is a pair of exact
stopping points:

    IdolCool -> Idol -> Burn1 -> Burn2 -> Burn3 -> Ash
    HereticIdol      -> HereticBurn1 -> .. -> HereticBurn3 -> HereticAsh

``no Unburnt`` requires every ordinary statue to reach **Burn3 and stop
there**; ``no Ash`` and ``no MiniAsh`` mean that burning one step further
DESTROYS the level -- nothing turns ash back into a statue, so an over-burnt
idol is a permanent loss, not a setback. ``no Heretic`` is the opposite
demand: a heretic must be burnt all the way THROUGH Burn3 into ash, and the
heretic ashes are not named by any win condition, so they are absorbing and
free. Every statue therefore has an exact touch quota -- 3 for an Idol, 4 for
an IdolCool or a Heretic, 1 for a Burn2 -- and the whole puzzle is routing a
walk that meets all of them at once.

Four mechanics, and only the first is stated anywhere in the rule listing:

  * **A statue burns one step for every statue orthogonally adjacent to the
    cell the player MOVES INTO.** All eighteen burn rules are guarded by
    ``[Player Move | ...]`` and are ordered from the top of the ladder down, so
    a statue advances exactly one step per press, and up to four statues
    advance at once. That last part is what the tight levels are built on: the
    3x3 blocks of levels 11-14 and 19-21 are only solvable because standing
    inside them burns three neighbours per press.
  * **`Move` is not a flag, it is an OBJECT the player leaves behind.** ``late
    [Player no Move no Marker1] -> [Player Move]`` grants it at the player's
    new cell and ``[Move no Player] -> []`` sweeps up the copy stranded at the
    old one, so the burn rules fire if and only if the player actually CHANGED
    CELLS. A press the rules refuse -- into a wall, into a full-size statue,
    into a jammed push -- leaves the Move sitting on the player, the first late
    rule swaps it for a Marker1, the burn rules miss, and the last late rule
    swaps it back. So **a refused press is a complete no-op**: this game has no
    wait move and needs none, since nothing on the board moves on its own.
    **Except on the very first press of a level**, where the player is not
    carrying a Move yet, the grant fires, and a refused press burns all four
    neighbours for free. `_Board.step` models it; `--fuzz` is what found it.
  * **Mini statues are sokoban crates and full-size ones are walls.** ``[>
    Player | MiniStatue] -> [> Player | > MiniStatue]`` plus a chain rule pushes
    a whole row of minis; the full-size Idol / Heretic objects appear in no push
    rule and share the player's collision layer, so they are immovable. A push
    is also a burn: the shoved mini lands orthogonally adjacent to the player
    that shoved it.
  * **A mini can be pushed ONTO a Wall.** Only the player has a rule cancelling
    a move into a wall (``[> Player | Wall] -> [Player | Wall]``) and Wall has a
    collision layer to itself, so nothing stops a crate ending up on one -- and
    once there it can never be pushed again, because that same cancel rule
    fires before the push rule and the player can never stand beside it in
    line. Level 26 has walls and minis; this is a real way to lose it.

Two more things the .txt does not say, both checked against the interpreter:

  * The four ash objects (Ash, MiniAsh, HereticAsh, MiniHereticAsh) share ONE
    collision layer, so a new ash EVICTS whatever ash was already lying in that
    cell. The model does not have to represent that, because it drops a heretic
    from the state entirely once it ashes (ash is inert, does not block, cannot
    be pushed and cannot be burnt further) and treats an ordinary ash as death.
  * The Loop / Loop2 / MarkerChange machinery in levels 15 and 26 is DEAD.
    ``[> Player][Loop] -> [> Player][Loop2]`` fires on every keypress, so the
    background- and wall-swapping rules that require a surviving ``Loop`` never
    match; and even if they did, Background and Background2 are pixel-identical
    and no ImmovableWall is in any level. `--fuzz` covers those levels and sees
    nothing move.

The expert
----------
A native `_Board` model, because the interpreter runs at ~320 steps/s here (the
Marker1 sweep at the end of the rule list repaints every cell on the board every
turn), which is two orders of magnitude short of what these searches need. The
interpreter's job is to CERTIFY: `--plans` replays every plan through it and
`--fuzz` plays 40k presses of random boards move-for-move against it.

STATE is ``(player, fixed_stages, mobs)``: the player's cell, the burn stage of
each immovable statue in a fixed order, and the pushable ones as a sorted tuple
of ``cell << 4 | heretic << 3 | stage``. `_Board.step` returns None for a press
that ashes an ordinary statue -- that is not a state, it is a dead board -- and
that prune is most of what makes these searches finish: once a statue reaches
its quota its four neighbours become cells the player may never enter again, so
the reachable space collapses as the level is solved rather than growing.

The search is three passes (see `_Board.optimal_plan`), the middle one lifted
from ps:goblin_hooblob:

  1. **Probe.** A width-capped layered beam ordered by ``8 * remaining touches
     + distance to the nearest unfinished statue``. That greedy score, not the
     admissible heuristic, is what steers it: the admissible bound has to
     assume four statues burn per press and is ~3x short over the whole middle
     of a level, and beam-ordering by it wanders. Level 26 falls out in 5s.
  2. **Prove.** Unweighted A* bounded by the probe's length, repeated on each
     improvement. A bounded run that CLOSES without finding anything shorter is
     a proof that the bound cannot be beaten.
  3. **Label.** A layered BFS from the start pruned by ``depth + heuristic >
     d*``, keeping exactly the shortest-path DAG, plus a backward pass marking
     the states that can still finish in the presses they have left. A press is
     optimal at a state iff it lands on one of those. Both sweeps measure the
     same distance with nothing approximated in between, so the sets are exact.

That matters here because the burn is symmetric in the two axes: crossing an
open floor to the next statue is a free choice of interleaving, and 30-60% of
the presses in these plans have a tie. Labelling one arbitrary staircase as the
only right answer would train a coin flip the policy cannot win.

The levels
----------
All 27 shipped boards are solved and 26 of the plans are PROVED shortest:

    level   0   1   2   3   4   5   6   7   8   9  10  11  12  13
    moves  11  19  24  22  37  30  17  15  35  19  26  17  23  11

    level  14  15  16  17  18  19  20  21  22  23  24  25  26
    moves  25  30  21  15  16  25  16  26  24  17  17  11  38*

587 presses in all, the longest 38 against the adapter's 200-action per-level
budget, and 79 of them (13%) carry a genuine tie.

``* `` level 26, the finale, is a genuine win the interpreter walks but is not
proved shortest: the bounded pass does not close inside the node budget (40M
generated nodes and 6.6 GB do not close it either), so its steps are labelled
with the press taken rather than with a measured tie set. Every other level's
labels are exact. Nothing shorter than 38 has been seen -- the probe returns 38
at beam widths from 2 000 to 400 000, and the bounded exact pass finds no win
below it before it runs out of budget.

Level 16 ("hell") is the one that shows the mechanic cleanly: fifteen of its
twenty-three statues are ALREADY Burn3, so the puzzle is a 21-press route that
touches each of the eight Burn2s exactly once and never once steps beside a
finished one.

The rendering fixes
-------------------
All three are in the game file (see the header comment in
data/puzzlescript_games/IDOLS_TO_THE_BURNT_GOD.txt), and ``--audit`` is the
regression test: it fills a whole board with each cell COMPOSITION at each cell
size the levels render at and asserts every pair of frames differs.

  * **The walls were invisible.** ``Wall`` was a bare ``darkred`` block, the
    floor is ``#773043``, and both quantise to ARC palette index 13. Levels 22
    and 26 are built around walls the frame did not show. Wall is ``grey`` now.
  * **Ash and HereticAsh were the same picture**, as were MiniAsh and
    MiniHereticAsh -- i.e. a board that can never be won and a board that has
    just satisfied a win condition rendered identically. The heretic ashes now
    use the heretic palette. The SHAPES are deliberately left alone: mirroring
    one of them would have been undone by the flip augmentation, which recolour
    is not.
  * **MiniAsh hid entirely under the player.** Its pixels all sat inside the
    player sprite's silhouette, so a board whose only ash was underfoot looked
    clean. It is the Ash mound shifted down one row now -- the same convention
    every other Mini* sprite in the file uses -- and its bottom corners fall
    outside the player.

Augmentation
------------
The engine state after reset is identical for every seed (levels are fixed
ASCII maps), so the only per-(seed, level) variables are the presentation
augmentations: the frame rotation (rotation_k in {0,1,2,3}) plus an independent
horizontal and vertical flip, each with the matching directional action remap
(`PuzzleScriptAdapter._FLIP_GAMES`), which together re-sample the board's
8-element symmetry group. 27 levels x 16 presentations = 432. The expert plan is
therefore seed-independent: solved once per level, cached to disk, and replayed
per seed with that seed's remapped screen actions.

The flips are sound here for the strongest possible reason: the game is
direction-blind. There is no gravity, every movement rule is stated with the
relative ``>`` force, the burn rules are written without a direction prefix (so
each expands to all four), no win condition names a direction, and no sprite
encodes one. Nothing can contest a cell either -- every statue burns exactly
once per press however the interpreter orders the four expansions -- so the
rule-order chirality that ps:gobble_rush and ps:im_sick_today have to argue
around cannot arise. ``--symmetry`` measures it rather than assuming it: all 27
plans replayed on all 8 turned and mirrored copies of their own boards, and
every one lands where the transform says.

Usage (run from the repo root):
    python solvers/generate_idols_to_the_burnt_god_training.py --episodes 200 \
        --out data/training_multi_level/idols_to_the_burnt_god

    python solvers/generate_idols_to_the_burnt_god_training.py --plans
    python solvers/generate_idols_to_the_burnt_god_training.py --audit
    python solvers/generate_idols_to_the_burnt_god_training.py --fuzz
    python solvers/generate_idols_to_the_burnt_god_training.py --symmetry
"""

from __future__ import annotations

import heapq
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solvers.common.ps_astar import (PSAStarSolver, PSExpert,      # noqa: E402
                                     Plan)

_REPO_ROOT = Path(__file__).resolve().parent.parent

GAME_NAME = "IDOLS_TO_THE_BURNT_GOD"

#: Disk cache of each level's start plan and its optimal-action sets. The
#: searches are seed-independent, so without a file on disk every shard
#: `parallelize_generator` starts would re-derive all of them. Delete to
#: re-derive.
PLAN_CACHE: Path = _REPO_ROOT / "data" / "idols_to_the_burnt_god_plans.json"

#: Engine direction names, in the order the model indexes them.
DIRS = ("up", "down", "left", "right")
_DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))

#: Frontier width of the upper-bound probe in `_Board.optimal_plan`. Sized from
#: the hardest level: 2 000 already finds level 26's 38-press plan and 400 000
#: finds nothing shorter, so this is comfort, not a dial. Its plan length is
#: never shipped as "shortest" -- it is only the bound the exact pass prunes
#: with.
PROBE_WIDTH = 20_000

#: Statue families. MOBILE says whether the family is pushed by the player
#: (the Mini* objects, which the `MiniStatue` or-group names); HERETIC says
#: whether the win condition wants it burnt to ash rather than stopped at
#: Burn3.
K_FULL, K_MINI, K_HFULL, K_HMINI = 0, 1, 2, 3
MOBILE = (False, True, False, True)
HERETIC = (False, False, True, True)

#: Burn stage. 0 = the "cool" statue (one extra step, and only the ordinary
#: families have one), 1 = the idol, 2-4 = Burn1..Burn3, and GONE is a heretic
#: that has burnt away: its ash is on no layer the mechanic reads, so the model
#: stops tracking it entirely. Ordinary ash is never a stage -- it is death,
#: and `_Board.step` returns None instead.
GONE = 6

#: PuzzleScript object name -> (family, stage). The ash objects are handled
#: separately (see DEAD_NAMES): they are not states of a tracked piece.
STATUE_NAMES: dict[str, tuple[int, int]] = {
    "idolcool": (K_FULL, 0), "idol": (K_FULL, 1), "burn1": (K_FULL, 2),
    "burn2": (K_FULL, 3), "burn3": (K_FULL, 4),
    "miniidolcool": (K_MINI, 0), "miniidol": (K_MINI, 1),
    "miniburn1": (K_MINI, 2), "miniburn2": (K_MINI, 3),
    "miniburn3": (K_MINI, 4),
    "hereticidol": (K_HFULL, 1), "hereticburn1": (K_HFULL, 2),
    "hereticburn2": (K_HFULL, 3), "hereticburn3": (K_HFULL, 4),
    "miniheretic": (K_HMINI, 1), "minihereticburn1": (K_HMINI, 2),
    "minihereticburn2": (K_HMINI, 3), "minihereticburn3": (K_HMINI, 4),
}

#: The two ashes the win condition forbids. A board holding either can never be
#: won, so it is refused rather than modelled.
DEAD_NAMES = ("ash", "miniash")


class _Board:
    """Native model of Idols to the Burnt God, and the search over it.

    STATE is ``(player, fixed, mobs, fresh)``:

      * ``player`` -- the acolyte's cell index (``row * width + col``).
      * ``fixed``  -- the burn stage of every immovable statue, in the board's
        own order. A heretic that has ashed carries GONE, which is both "done"
        and "no longer blocking".
      * ``mobs``   -- the pushable statues, each packed as
        ``cell << 4 | heretic << 3 | stage`` and kept SORTED, which is what
        makes two boards with the same crates in different discovery orders one
        state. A mob never carries stage 5: an ordinary one ashing is death and
        a heretic one ashing is dropped from the tuple.
      * ``fresh``  -- 1 before the level's FIRST press and 0 ever after. It
        carries the one piece of the interpreter's Move bookkeeping a plan can
        exploit; see `step`. Only the root of a search is ever fresh, so it
        costs the state space one extra node.

    Walls are static and live on the board rather than in the state. So do the
    immovable statues' cells -- ``fixed_cell[i]`` and ``fixed_kind[i]`` -- since
    nothing in this game moves a full-size statue.
    """

    def __init__(self, height, width, walls, fixed):
        self.height, self.width = height, width
        self.n = height * width
        self.walls = frozenset(walls)
        self.fixed_cell = tuple(c for c, _k in fixed)
        self.fixed_kind = tuple(k for _c, k in fixed)
        self.fixed_at = {c: i for i, (c, _k) in enumerate(fixed)}
        nbr = []
        for dr, dc in _DELTAS:
            row = []
            for i in range(self.n):
                r, c = divmod(i, width)
                r2, c2 = r + dr, c + dc
                row.append(r2 * width + c2
                           if 0 <= r2 < height and 0 <= c2 < width else -1)
            nbr.append(tuple(row))
        #: ``nbr[d][cell]`` -- the cell one step in direction ``d``, or -1 off
        #: the board (there is no rule for the edge; the interpreter simply
        #: fails the move, and a push into it fails the whole press).
        self.nbr = tuple(nbr)
        #: ``adj[cell]`` -- the on-board orthogonal neighbours, i.e. exactly the
        #: cells whose statues burn when the player moves into ``cell``.
        self.adj = tuple(tuple(x for x in (nbr[d][i] for d in range(4)) if x >= 0)
                         for i in range(self.n))

    # -- dynamics ------------------------------------------------------------

    def step(self, state, d):
        """The settled board after pressing ``d``, ``state`` unchanged if the
        press is refused, or None if it ashed an ordinary statue (a dead board,
        not a state).

        The order mirrors the interpreter's: the wall cancel first, then the
        push chain, then movement, and only then the burn -- which reads the
        board AFTER everything has moved, so a shoved mini is burnt by the press
        that shoved it.

        THE FIRST PRESS OF A LEVEL BURNS EVEN WHEN IT IS REFUSED. ``late
        [Player no Move no Marker1] -> [Player Move]`` grants the Move object
        whenever the player is not carrying one, and a freshly loaded level
        carries none; every later turn the player ends holding one either way
        (granted at the cell it walked to, or swapped to a Marker1 and back
        where it stood). So on turn one, and only turn one, walking into a
        wall, into an idol or into a jammed push still lights up all four
        neighbours -- a free burn with no step taken.

        No shipped level's shortest plan actually spends it (checked: every one
        of the 27 opens with a press that moves), and modelling it changed no
        plan length. It is here because it is REAL: the fuzz found it, a
        player will find it, and a model that quietly disagrees with the
        interpreter about a legal press is a model that will be wrong about
        something else later.
        """
        player, fixed, mobs, fresh = state
        tgt = self.nbr[d][player]
        stuck = False
        if tgt < 0 or tgt in self.walls:
            stuck = True                       # off the board / [> Player|Wall]
        else:
            at = self.fixed_at.get(tgt)
            if at is not None and fixed[at] != GONE:
                stuck = True                   # full-size statues are furniture
        mob_at = {m >> 4: j for j, m in enumerate(mobs)}
        chain = []
        if not stuck and tgt in mob_at:
            cell = tgt
            while cell in mob_at:              # [> MiniStatue | MiniStatue]
                chain.append(mob_at[cell])
                cell = self.nbr[d][cell]
                if cell < 0:
                    stuck = True
                    break
            else:
                # The landing cell may be a WALL (nothing refuses that) or an
                # ash, but not a full-size statue: same collision layer.
                at = self.fixed_at.get(cell)
                if at is not None and fixed[at] != GONE:
                    stuck = True
            if stuck:
                chain = []
        if stuck:
            if not fresh:
                return state
            tgt = player                       # burn where the acolyte stands
        mobs = list(mobs)
        if chain:
            hop = self.nbr[d]
            for j in chain:
                m = mobs[j]
                mobs[j] = (hop[m >> 4] << 4) | (m & 15)
            mob_at = {m >> 4: j for j, m in enumerate(mobs)}
        fixed = list(fixed)
        for a in self.adj[tgt]:
            at = self.fixed_at.get(a)
            if at is not None and fixed[at] != GONE:
                fixed[at] += 1
                if fixed[at] == 5:
                    if not HERETIC[self.fixed_kind[at]]:
                        return None            # Burn3 -> Ash: unwinnable
                    fixed[at] = GONE
            j = mob_at.get(a)
            if j is not None:
                m = mobs[j]
                stage = (m & 7) + 1
                if stage == 5:
                    if not (m & 8):
                        return None            # MiniBurn3 -> MiniAsh
                    mobs[j] = -1               # a heretic's ash is inert
                else:
                    mobs[j] = (m & ~7) | stage
        mobs = [m for m in mobs if m >= 0]
        mobs.sort()
        return (tgt, tuple(fixed), tuple(mobs), 0)

    def won(self, state):
        """``no Heretic and no Ash and no MiniAsh and no Unburnt``: every
        ordinary statue stopped at Burn3, every heretic burnt away."""
        _p, fixed, mobs, _fresh = state
        for i, stage in enumerate(fixed):
            if stage != (GONE if HERETIC[self.fixed_kind[i]] else 4):
                return False
        # A surviving mob is ordinary (heretics are dropped when they ash), so
        # ``cell << 4 | 0 << 3 | 4`` is the only finished packing.
        return all(m & 15 == 4 for m in mobs)

    def remaining(self, state):
        """``(total touches still owed, [(cell, owed, is_mobile), ...])``."""
        _p, fixed, mobs, _fresh = state
        out, total = [], 0
        for i, stage in enumerate(fixed):
            if stage == GONE:
                continue
            owed = (5 if HERETIC[self.fixed_kind[i]] else 4) - stage
            if owed > 0:
                out.append((self.fixed_cell[i], owed, False))
                total += owed
        for m in mobs:
            owed = (5 if (m & 8) else 4) - (m & 7)
            if owed > 0:
                out.append((m >> 4, owed, True))
                total += owed
        return total, out

    # -- heuristics ----------------------------------------------------------

    def heuristic(self, state):
        """An ADMISSIBLE lower bound on the presses still needed.

        Two independent bounds, whichever is larger:

          * ``ceil(owed / 4)`` -- a press burns at most its four neighbours.
          * Per statue, ``(distance - 1) + (owed - 1) * repeat``. Manhattan
            distance falls by at most one per press even when the statue is
            being SHOVED (the player closes one cell and the crate opens one),
            so reaching it costs ``distance - 1``. ``repeat`` is the gap
            between two touches of the SAME statue: 2 for an immovable one,
            because no two of its neighbour cells are adjacent to each other so
            the player has to step away and back, and 1 for a mini, because
            shoving it again keeps the player beside it.

        Deliberately loose -- the "four at once" relaxation is about 3x short
        over the middle of a level, and closing that gap needs reasoning about
        which statues can share a press. It is enough for the exact pass to
        prune with, and the probe uses `beam_score` instead.
        """
        total, items = self.remaining(state)
        if not total:
            return 0
        width = self.width
        pr, pc = divmod(state[0], width)
        best = 0
        for cell, owed, mobile in items:
            cr, cc = divmod(cell, width)
            dist = abs(cr - pr) + abs(cc - pc)
            bound = max(0, dist - 1) + (owed - 1) * (1 if mobile else 2)
            if bound > best:
                best = bound
        return max(best, -(-total // 4))

    def beam_score(self, state):
        """Greedy ordering for the probe: touches still owed dominate, with the
        walk to the nearest unfinished statue as the tie-break. Not admissible
        and not meant to be -- it is a progress measure, and it is the reason
        the beam finds level 26 at all where ordering by `heuristic` (flat
        across the whole middle of the board) wanders and dies out."""
        total, items = self.remaining(state)
        if not total:
            return 0
        width = self.width
        pr, pc = divmod(state[0], width)
        near = min(abs(divmod(c, width)[0] - pr) + abs(divmod(c, width)[1] - pc)
                   for c, _owed, _mob in items)
        return total * 8 + near

    # -- search --------------------------------------------------------------

    def beam(self, start, width=PROBE_WIDTH, depth=250):
        """A layered BFS keeping the ``width`` best states of each layer by
        `beam_score`. Returns a winning press list or None.

        Used only as the upper-bound probe. The visited set spans layers, so
        this is a real (pruned) BFS: the depth it wins at is the length of the
        plan it returns.
        """
        if self.won(start):
            return []
        frontier = {start: []}
        seen = {start}
        for _ in range(depth):
            nxt = {}
            for state, path in frontier.items():
                for d in range(4):
                    s2 = self.step(state, d)
                    if s2 is None or s2 == state or s2 in seen:
                        continue
                    if self.won(s2):
                        return path + [d]
                    seen.add(s2)
                    nxt[s2] = path + [d]
            if not nxt:
                return None
            if len(nxt) > width:
                keep = sorted(nxt, key=self.beam_score)[:width]
                nxt = {k: nxt[k] for k in keep}
            frontier = nxt
        return None

    def astar(self, start, node_cap, bound=None):
        """``(plan, closed)``: an unweighted A* over the model.

        ``bound`` drops every node whose ``g + h`` reaches it. With an
        admissible heuristic that keeps exactly the states from which a plan
        shorter than ``bound`` could still exist, so a bounded search that
        CLOSES (``closed`` true, no plan) is a proof that ``bound`` cannot be
        beaten -- which is how the probe's plan gets certified as shortest
        without ever running an unbounded search.

        ``node_cap`` counts GENERATED nodes, and it is a MEMORY budget rather
        than a patience one: every queued node holds a full state, so counting
        expansions would let peak usage swing with the branching factor.
        """
        if self.won(start):
            return [], True
        counter = 0
        pq = [(self.heuristic(start), 0, counter, start)]
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
                if bound is not None and ng + h >= bound:
                    continue
                best[nxt] = ng
                came[nxt] = (state, d)
                counter += 1
                heapq.heappush(pq, (ng + h, ng, counter, nxt))
            if nodes >= node_cap:
                return None, False
        return None, True

    def optimal_plan(self, start, node_cap):
        """``(plan, optsets, proven)``: a winning plan, the set of equally good
        presses at each of its steps, and whether the plan is PROVED shortest.
        ``(None, None, False)`` when the board cannot be won at all.

        Three sweeps -- probe, prove, label; see the module docstring. A plan
        the prove step could not certify is still a genuine win (the
        interpreter walks it) and is labelled with the press it took, never
        with nothing.
        """
        plan = self.beam(start)
        if plan is None:
            # Nothing to bound with. One unbounded exact run then decides it:
            # either it wins (and closed) or the board really is unwinnable.
            plan, closed = self.astar(start, node_cap)
            if plan is None:
                return None, None, False
            proven = closed
        else:
            while True:
                better, closed = self.astar(start, node_cap, bound=len(plan))
                if better is None:
                    proven = closed
                    break
                plan = better
        if not proven:
            return plan, [[d] for d in plan], False

        limit = len(plan)
        # Forward: the layers of the shortest-path DAG, with each state's
        # successor list. A state is kept at its FIRST depth only -- that is its
        # true distance from the start, so re-reaching it later can never be on
        # a shortest path.
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

        # Backward: which states still finish in the presses they have left,
        # and with which presses.
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


# ---------------------------------------------------------------------------
# Reading a board off the interpreter
# ---------------------------------------------------------------------------

def board_from_engine(eng, game):
    """``(board, state)`` for the engine's current grid.

    The model is built from the live interpreter rather than from the level
    text, so a level edited in the .txt needs no second definition here, and
    the fuzz harness can start both sides from the same random board.
    """
    names = game.obj_idx_to_name
    height, width = eng.height, eng.width
    walls, fixed = [], []
    for r in range(height):
        for c in range(width):
            i = r * width + c
            for o in eng.grid[r][c]:
                name = names[o]
                if name == "wall":
                    walls.append(i)
                elif name in DEAD_NAMES:
                    raise ValueError("board already holds Ash / MiniAsh")
                else:
                    entry = STATUE_NAMES.get(name)
                    if entry is not None and not MOBILE[entry[0]]:
                        fixed.append((i, entry[0]))
    fixed.sort()
    board = _Board(height, width, walls, fixed)
    return board, read_state(board, eng, game)


def read_state(board, eng, game):
    """The engine grid's state in ``board``'s coordinates.

    Separate from `board_from_engine` because the board's static parts (walls,
    the immovable statues' cells) are fixed at construction: a heretic that has
    ashed since is simply absent from the grid, and reads back as GONE rather
    than shrinking the tuple.

    ``fresh`` is read off the grid too: the player carries a ``Move`` object
    from its first press onward (see `_Board.step`), so "no Move under the
    player" is exactly "this level has not been pressed yet". Marker1 is not
    consulted -- the last late rule sweeps every one of them away, so none ever
    survives a turn.
    """
    names = game.obj_idx_to_name
    idx = game.obj_name_to_idx
    dead = {idx[n] for n in DEAD_NAMES}
    player_id, move_id = idx["player"], idx["move"]
    fixed = [GONE] * len(board.fixed_cell)
    mobs = []
    player = fresh = -1
    for r in range(eng.height):
        for c in range(eng.width):
            i = r * board.width + c
            cell = eng.grid[r][c]
            for o in cell:
                if o in dead:
                    raise ValueError("board holds Ash / MiniAsh")
                if o == player_id:
                    player = i
                    fresh = 0 if move_id in cell else 1
                    continue
                entry = STATUE_NAMES.get(names[o])
                if entry is None:
                    continue
                kind, stage = entry
                if MOBILE[kind]:
                    mobs.append((i << 4) | ((1 if HERETIC[kind] else 0) << 3)
                                | stage)
                else:
                    fixed[board.fixed_at[i]] = stage
    if player < 0:
        raise ValueError("no player on the board")
    mobs.sort()
    return (player, tuple(fixed), tuple(mobs), fresh)


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class IdolsExpert(PSExpert):
    """Plans on `_Board` and lets the interpreter certify.

    `PSExpert` supplies everything around the search -- the plan memo, the disk
    cache with its staleness check, and the snapshot/restore discipline -- so
    all this overrides is `_search`, which builds the model off the engine's
    current grid and returns a `Plan` carrying the measured optimal sets.
    `heuristic` is never called (the model owns its own), and says so.
    """

    directions = list(DIRS)         # `noaction` in the prelude: ACTION5 is dead
    scope_by_level = True
    plan_cache_path = PLAN_CACHE

    def setup(self) -> None:
        # Dynamic objects only -- walls are static and `scope_by_level` keeps
        # the memo honest across levels. Every statue object is included
        # because every one of them is a distinct burn stage, and the two
        # ordinary ashes because a board holding one is a distinct, unwinnable
        # state that must never share a cache entry with a live board.
        #
        # `move` is in the key on purpose: it is the interpreter bookkeeping
        # that makes a level's FIRST press burn even when refused (see
        # `_Board.step`), so a board reached by walking out and back is NOT the
        # start board and must not be served the start board's plan.
        idx = self.g.obj_name_to_idx
        self._dyn = {idx[n] for n in
                     ("player", "move", *STATUE_NAMES, *DEAD_NAMES,
                      "hereticash", "minihereticash")}

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


class IdolsSolver(PSAStarSolver):
    game_id = "puzzlescript_idols_to_the_burnt_god"
    game_name = GAME_NAME
    expert_cls = IdolsExpert

    #: Record against the GAME FOLDER's adapter -- the object `game_envs` hands
    #: a live agent -- so a step cap or sprite patch added there later cannot
    #: silently make this generator tape a game nobody plays.
    game_module_id = "ps:idols_to_the_burnt_god"

    #: Generated-state budget per search. Sized as MEMORY (~1 GB at this state
    #: size), not as patience: 26 of the 27 levels close far inside it, and
    #: level 26 would not close at 40M / 6.6 GB either, so raising it buys a
    #: longer startup and the same plan.
    node_cap = 6_000_000

    #: The longest plan is 38, well under the adapter's 200-action per-level
    #: budget (which would otherwise flip a level to GAME_OVER mid-plan); the
    #: rest is room for the RESET exploration prefix and the re-plan after it.
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
    game = IdolsSolver().make_game(0)
    eng, g = game._engine, game._game
    total = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board, state = board_from_engine(eng, g)
        owed, items = board.remaining(state)
        minis = sum(1 for _c, _r, mob in items if mob)
        t0 = time.time()
        plan, optsets, proven = board.optimal_plan(state, IdolsSolver.node_cap)
        dt = time.time() - t0
        head = (f"level {level:2d}: {eng.height:d}x{eng.width:d} "
                f"{len(items):2d} statues ({minis:d} mini) {owed:2d} touches")
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
              f"{dt:7.1f}s", flush=True)
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

    Composition, not object: a mini shoved onto a Wall has to differ from the
    same mini on the floor (on the wall it is stuck forever), and each ash has
    to differ from the other three (Ash is a lost board, HereticAsh is a
    satisfied win condition). The whole board is filled with the composition
    and whole frames are compared -- cropping one cell out by arithmetic, as
    the older audits in this tree do, is wrong for any board `_render_frame`
    upscales, which is all of them.
    """
    import itertools

    import numpy as np

    from adapters.puzzlescript_adapter import _render_frame

    game = IdolsSolver().make_game(0)
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx

    ashes = ["ash", "miniash", "hereticash", "minihereticash"]
    comps = {"floor": [], "wall": ["wall"], "player": ["player"]}
    for name in (*STATUE_NAMES, *ashes):
        comps[name] = [name]
    for name, (kind, _stage) in STATUE_NAMES.items():
        if not MOBILE[kind]:
            continue
        comps[f"{name}+wall"] = [name, "wall"]       # shoved onto a wall
        for ash in ashes:
            comps[f"{name}+{ash}"] = [name, ash]     # shoved onto an ash
    for ash in ashes:
        comps[f"player+{ash}"] = ["player", ash]     # standing on an ash

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
    print(f"audit clean ({len(comps)} compositions)" if not bad
          else f"AUDIT FAILED: {bad} indistinguishable pairs")
    return 0 if not bad else 1


#: ``_SYM[k]`` maps an engine direction index through the k-th rotation of the
#: board (90 degrees counter-clockwise each, as `np.rot90` turns the frame):
#: up -> left -> down -> right -> up. ``_MIRROR`` is the extra left/right swap a
#: horizontal flip adds. Composed, the eight of them are the square's symmetry
#: group -- exactly the presentations `PuzzleScriptAdapter` samples from.
_ROT_DIR = (2, 3, 1, 0)          # up->left, down->right, left->down, right->up
_MIRROR_DIR = (0, 1, 3, 2)       # left <-> right


def _symmetry() -> int:
    """Replay every level's plan on all 8 turned and mirrored copies of its own
    board and require the interpreter to land the pieces where the transform
    says.

    The flip augmentation (`PuzzleScriptAdapter._FLIP_GAMES`) is only sound if
    the MECHANIC is mirror-symmetric, and the rules alone are not proof of that:
    ps:gobble_rush and ps:im_sick_today both read as direction-free and are not,
    because two rule matches contesting one cell are settled by the order the
    interpreter expands a rule's four directions -- a fact about the screen, not
    the board. This game should have no such case (nothing here contests a cell:
    every statue burns exactly once per press however the four expansions are
    ordered), so this is the measurement that says so rather than an argument
    that assumes it.
    """
    game = IdolsSolver().make_game(0)
    eng, g = game._engine, game._game
    expert = IdolsExpert(game, node_cap=IdolsSolver.node_cap)
    dir_index = {d: i for i, d in enumerate(DIRS)}

    def transform(grid, h, w, k, mirror):
        """The grid turned k quarter-turns counter-clockwise, then mirrored."""
        for _ in range(k):
            grid = [[grid[r][c] for r in range(h)] for c in range(w - 1, -1, -1)]
            h, w = w, h
        if mirror:
            grid = [list(reversed(row)) for row in grid]
        return grid, h, w

    bad = checks = 0
    for level in range(game.n_levels):
        game.set_level(level)
        plan = expert.plan(eng, level)
        if plan is None:
            continue
        h0, w0 = eng.height, eng.width
        start = [[set(cell) for cell in row] for row in eng.grid]
        for d in plan:
            eng.step(d)
        want = [[set(cell) for cell in row] for row in eng.grid]
        for k in range(4):
            for mirror in (False, True):
                grid, h, w = transform(start, h0, w0, k, mirror)
                eng.grid = [[set(cell) for cell in row] for row in grid]
                eng.height, eng.width = h, w
                eng._position_index_dirty = True
                eng._rule_noop_cache.clear()
                for d in plan:
                    i = dir_index[d]
                    for _ in range(k):
                        i = _ROT_DIR[i]
                    if mirror:
                        i = _MIRROR_DIR[i]
                    eng.step(DIRS[i])
                expect, _h, _w = transform(want, h0, w0, k, mirror)
                checks += 1
                if eng.grid != expect or not eng.check_win():
                    bad += 1
                    print(f"level {level}: rot {k}"
                          f"{' + mirror' if mirror else ''} does NOT reproduce "
                          f"the plan's result")
    print(f"{checks} replays over {game.n_levels} levels x 8 presentations: "
          + ("every one lands where the transform says"
             if not bad else f"{bad} ASYMMETRIC"))
    return 0 if not bad else 1


def _fuzz(n_boards: int = 3000, n_steps: int = 40, seed: int = 0) -> int:
    """Differential fuzz: boards played move-for-move against the real
    interpreter, comparing the player, every statue's cell AND burn stage, the
    ash-death flag and the win flag after every press.

    TWO SOURCES OF BOARDS, and the game needs both.

    *Random dense boards*, because the shipped levels exercise almost none of
    the cases the model's branches exist for: a crate shoved onto a wall, a
    crate shoved onto an ash, a chain of three, a heretic ashing on top of
    another ash (the four ashes share a collision layer, so one evicts the
    other -- which is why heretic ash is dropped from the state rather than
    tracked), a push that runs off the board edge and cancels the whole press,
    and the ``fresh`` first press that burns without moving. That last one was
    found HERE, by this pass, after the levels had already been "solved"
    against a model that did not have it.

    *The shipped levels, from every prefix of their own solutions*, because
    random play on a 3x3 block of idols ashes something within a few presses:
    the states a plan actually passes through are reached by planning, not by
    luck, and they are the ones where a statue sits at exactly its quota.

    The run prints COVERAGE COUNTERS rather than only a verdict -- "no
    mismatches" over presses that never pushed a crate is not evidence about
    pushing.
    """
    import random

    game = IdolsSolver().make_game(0)
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    ash_ids = {idx[n] for n in DEAD_NAMES}

    # Statue-heavy and wall-light on purpose: the counters below are the
    # evidence this run produces, and a wall-heavy board spends its presses on
    # refusals, which prove nothing about burning.
    names = (["background"] * 14 + ["wall"] * 3
             + list(STATUE_NAMES) + ["ash", "miniash",
                                     "hereticash", "minihereticash"])
    rng = random.Random(seed)
    cover = {"refused": 0, "push": 0, "chain push": 0, "onto wall": 0,
             "burn": 0, "heretic ash": 0, "ash death": 0, "win": 0}
    mismatches = presses = boards = 0

    def _dead():
        return any(o in ash_ids
                   for row in eng.grid for cell in row for o in cell)

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
            dead = _dead()
            if got is None:
                if not dead:
                    mismatches += 1
                    print(f"MISMATCH after {DIRS[d]}: the model says an "
                          f"ordinary statue ashed, the engine does not\n"
                          f"  from {before}")
                else:
                    cover["ash death"] += 1
                return
            live = read_state(board, eng, g)
            if dead or got != live:
                mismatches += 1
                print(f"MISMATCH after {DIRS[d]}:\n  from   {before}\n"
                      f"  model  {got}\n  engine {live} dead={dead}")
                return
            if board.won(got) != bool(eng.check_win()):
                mismatches += 1
                print(f"WIN MISMATCH at {got}: model {board.won(got)} "
                      f"engine {eng.check_win()}")
                return
            if got == before:
                cover["refused"] += 1
            else:
                old = sorted(m >> 4 for m in before[2])
                new = sorted(m >> 4 for m in got[2])
                if len(old) == len(new) and old != new:
                    cover["push"] += 1
                    if sum(1 for a, b in zip(old, new) if a != b) > 1:
                        cover["chain push"] += 1
                    if any(c in board.walls for c in new):
                        cover["onto wall"] += 1
                if (got[1] != before[1]
                        or [m & 7 for m in got[2]] != [m & 7 for m in before[2]]):
                    cover["burn"] += 1
                if (sum(1 for s in got[1] if s == GONE)
                        > sum(1 for s in before[1] if s == GONE)
                        or len(got[2]) < len(before[2])):
                    cover["heretic ash"] += 1
            state = got
            if board.won(state):
                cover["win"] += 1
                return

    # 1. random dense boards
    for _ in range(n_boards):
        h, w = rng.choice([3, 4, 5]), rng.choice([3, 4, 5])
        grid = [[{idx["background"], idx[rng.choice(names)]} for _ in range(w)]
                for _ in range(h)]
        pr, pc = rng.randrange(h), rng.randrange(w)
        grid[pr][pc] = {idx["background"], idx["player"]}
        eng.grid = grid
        eng.height, eng.width = h, w
        eng._position_index_dirty = True
        eng._rule_noop_cache.clear()
        # No warm-up press: a hand-built board carries no Move object, which is
        # exactly the ``fresh`` case `read_state` detects and `_Board.step`
        # models, and half the point of this pass is to exercise it.
        try:
            board, state = board_from_engine(eng, g)
        except ValueError:
            continue
        if board.won(state):
            continue
        boards += 1
        _rollout(board, state, n_steps)

    # 2. the shipped levels, from every prefix of their own plans
    expert = IdolsExpert(game, node_cap=IdolsSolver.node_cap)
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
    if "--symmetry" in sys.argv:
        sys.exit(_symmetry())
    sys.exit(IdolsSolver.main())
