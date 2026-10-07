"""Generate Phase-1 training data for the PuzzleScript game ps:space_expedition
("Space Expedition", Riley Van Etten and Orry Paynter).

The harness -- the rotation contract, the trajectory recorder, the plan memo and
its disk cache, and the BaseSolver plumbing -- lives in
`solvers/common/ps_astar.py`, shared with the other ps: generators. This file is
the game-specific part: a native model of the alien-chase mechanic, the two-pass
search that plans over it, and the checks that pin the model to the interpreter.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_space_expedition",
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
Each step also carries the full set of equally-optimal presses.

The game
--------
Pick up every astronaut and land on the planet: the win is ``Some Player on
Planet`` AND ``No Astronaut`` AND ``No Corpse``. You walk one cell per press;
ACTION picks an astronaut up when you are standing on it, and is otherwise a
genuine WAIT: there is no ``require_player_movement``, so a press that does
nothing still advances the aliens. White asteroids are pushed one at a time, by
the player and by the aliens; walls are not.

**Space Expedition is a game about lines, and almost everything about it follows
from one rule that is easy to read the wrong way:**

    [ Alien | ... | Player ] -> [ > Alien | ... | Player ]

The ellipsis is content-agnostic in this interpreter, so an alien that shares
your row or column charges one cell down that line **through walls, asteroids,
astronauts and other aliens**. Sight is blocked by nothing. There is no hiding;
there is only not sharing a line. Two consequences run the whole search:

  * **An alien orthogonally adjacent to you kills you, whatever you press.** It
    is adjacent, so it is in line, so it gets a force, and ``[ > Alien | Player ]``
    fires among the rules -- before anything moves. Fleeing, waiting and walking
    into it are the same move. The only escape from an alien one cell away is to
    already not be there.
  * **You die by touching a WALL.** ``[ > Player | Wall ] -> [ PlayerDead | Wall ]``:
    walking into a wall is not a refused press, it is fatal. Walking into an
    asteroid that cannot move IS a refused press, and is safe -- the crate rules
    come after the death rules, so a blocked push never reads as a wall.

Everything the model must get right beyond that, all of it found by pinning the
model against the interpreter:

  * **A convoy annihilates itself, from the far end.** ``[ > Alien | Alien ]``
    deletes both aliens, one match at a time, and every alien carrying a given
    force is by construction on the one ray running from the player -- so the
    pair that dies is the one FURTHEST from you and the survivor is the one
    nearest. Three in a line leave one alive, not none. The interpreter scans
    each of the four directional copies against the charge direction (bottom to
    top for ``up``, right to left for ``left``, row-major otherwise), which is
    exactly what makes that true in all four; getting the scan backwards leaves
    the far alien alive instead of the near one, which is the difference between
    a press being safe and being fatal. Four random boards convicted the first
    version of `_Board._annihilate` of precisely that -- and NO shipped level
    ever reaches the rule on a shortest path (measured: 0 annihilations across
    all 8 plans, and ~4 in the whole level fuzz), which is why the check that
    found it had to be synthetic boards rather than play on these eight.
  * **Because the scan runs against the charge, the mechanic is
    rotation-equivariant** and this game has none of the chirality ps:gobble_rush
    had to steer around. ``--symmetry`` measures it rather than trusting it:
    every plan replayed on all 8 turned and mirrored copies of its own board
    lands where the transform says.
  * **The pickup beats the corpse.** ``[ ACTION Player Astronaut ]`` is listed
    before ``[ > Alien | Astronaut ] -> [ Alien | Corpse ]``, so pressing ACTION
    while standing on an astronaut an alien is about to reach saves it. A corpse
    is terminal -- ``No Corpse`` is a win condition and nothing removes one --
    so `_Board.step` reports it as a dead successor exactly like the player's own
    death, and neither ever enters the search.
  * A blocked chain is a complete no-op: there are no multi-crate pushes (only
    ``[ > Player | AsteroidM ]``, which gives the crate a force but never the
    crate behind it), and an asteroid shoved into the player's cell simply stops.

**And the game is DETERMINISTIC, despite opening with a random rule.**
``[ STATIONARY Alien ] -> [ RANDOMDIR Alien ]`` looks like the thing that would
force belief-space planning, but ``randomdir`` is not a token this parser knows
(the same hole Buoy Deploy's octopi fall through), so the RHS resolves to no
objects and the rule is an identity. Aliens move only when they have a line to
you. Nothing else in the file is random, and no rule declares ``again``, so one
press is exactly one rule pass plus one force resolution. Check for this before
assuming a ps: game with a random rule needs to plan under uncertainty.

Expert solver
-------------
Not a search over the interpreter. `_Board` is a native model of the whole rule
set over integer cell indices -- state ``(player, aliens, rocks, crew)``, with
walls and planets static and both terminal markers folded into a dead successor
-- and it plans in two passes:

  1. **A-star** with an admissible, consistent heuristic: a shortest tour through
     every astronaut still out there, ending on a planet, plus one ACTION press
     per astronaut. Distances are BFS over everything that is not a wall, so
     asteroids count as floor -- a relaxation, and a tight one, since walking
     into an asteroid with a clear far side costs the same one press a plain
     step would. This proves ``d*`` while touching a tiny slice of the space (23
     states on level 0; 171k on level 7, whose board carries 8 aliens and 86
     asteroids).
  2. **The exact field inside that bound**: a forward sweep pruned by
     ``g + h <= d*`` records predecessors, and a reverse sweep from the winning
     states gives the true remaining press count of every state a shortest path
     can pass through.

The second pass is what makes the **optimal-action SETS measured rather than
inferred** -- at distance ``d``, a press is optimal iff it lands on a state at
distance ``d - 1``. Without them every free stretch of a walk would train one
arbitrary interleaving of the two axes as the single right answer.

The levels
----------
All 8 shipped levels are solved and every plan is PROVED shortest; nothing was
authored or skipped. 345 presses in all, 35 of them steps with more than one
equally-optimal press (381 labelled presses). Twelve of the 345 are the ACTION
pickups and none of the rest is a WAIT -- measured, not assumed: waiting always
costs a press and the levels never make it pay, so ACTION survives in the search
only because dropping it would be an unproven prune. The longest plan is 106 presses
(level 6) against the adapter's 200-press per-level budget, so this game needs
no ``games/`` step-limit wrapper. Level 7 -- a 25x23 board with no walls at all,
ringed by 86 asteroids, 8 aliens and 3 astronauts -- is the only expensive
search at ~40 s; it is cached to ``data/space_expedition_plans.json``, which is
what keeps every `parallelize_generator` shard from re-deriving it.

The five checks, and what each one can and cannot see:

  * ``--plans`` builds every field and replays its plan through the real
    interpreter, requiring the win on the LAST press and no earlier. That is the
    only check that can catch a plan which is valid but not shortest.
  * ``--fuzz`` compares `_Board` against the interpreter over ~52k transitions
    from two populations, because neither alone is enough. Random play from
    every PREFIX of every level's own plan puts the model in the configurations
    the plans visit -- random play from a level START dies within a few presses,
    since an alien adjacent in line is unsurvivable. And ~3000 dense random
    5x5..8x8 boards with 2-5 aliens are the ONLY way to reach
    ``[ > Alien | Alien ]`` at all: no shipped level ever gets two aliens
    adjacent on one of the player's lines, so the levels annihilate ~4 aliens
    across the whole run and the random boards annihilate ~440. The counters are
    printed for exactly that reason.
  * ``--verify`` re-derives every plan length, asserts the heuristic admissible
    at every state of every field (``h <= dist``), and re-derives every tie set
    with a bounded forward search sharing no code with the predecessor map they
    came from.
  * ``--symmetry`` replays each plan on all 8 turned and mirrored copies of its
    own board THROUGH THE INTERPRETER. It is the only check that can see a
    rule-order chirality: the fuzz plays one board, where the model and the
    interpreter agree on an order neither can perceive.
  * ``--audit`` renders every cell composition at every cell size in use and
    asserts pairwise distinctness.

Rendering (the sprites in the .txt had to be redrawn)
-----------------------------------------------------
``--audit`` reported 94 indistinguishable pairs before this generator existed,
and the worst of them was the WIN itself: at the 3 px cells level 6 renders at
and the 2 px cells level 7 renders at, a player standing on the planet was
pixel-identical to a player standing on floor. So was a player standing on an
astronaut, which is the one thing you have to see to know to press ACTION.

The fix is the two-hole convention. `_render_cell_sprite` samples a 5x5 sprite
at ``(2i+1)*5 // (2*cell_px)``, so the rows a cell size keeps are 2px -> {1,3},
3px -> {0,2,4}, 4px -> {0,1,3,4}, 5px and 6px -> all of them. **No single sprite
pixel survives at both 2 px and 3 px** -- the two sets are disjoint -- so a
marker has to be a PAIR. Every layer-3 sprite (wall, asteroid, alien, the four
player facings, PlayerDead) is now transparent at (1,1) AND (2,2), and each of
the three layer-2 objects claims both with one signature colour: Pink for the
Planet, White for the Astronaut, Red for the Corpse, against the Background's
Black. Every cell size samples at least one of the pair, so "what am I standing
on" is readable on all eight boards.

The four player facings were redrawn as an exact 90-degree ORBIT of one base
sprite, so a turned frame shows facing art turned the same way the motion is.
The orbit is C4 and not the full dihedral group, and it cannot be made D4: the
(1,1) hole maps to (1,3) under a mirror, and at 2 px the only four pixels that
exist ARE that orbit -- reserving all four would leave every layer-3 object
invisible on level 7, and reserving the mirror-invariant (2,2) alone would put
the win frame back out of sight. That is the reason this game is NOT in
`PuzzleScriptAdapter._FLIP_GAMES`: the rotation it already gets is coherent, a
mirror would not be, and the trade is not worth the 4x in presentations.

Augmentation
------------
The engine state after reset is identical for every seed (the levels are fixed
ASCII maps), so the only per-(seed, level) variable is presentation: the frame
rotation plus the matching directional action remap. The expert plan is
therefore seed-independent -- solved once per level, cached to disk, and
replayed per seed with that seed's remapped screen actions.

Recovery is the shared ``recovery_mode = "reset"`` arc, and this game is the
sharpest case for it in the corpus: an exploration prefix here almost always
ends with the player dead (random play dies within a handful of presses), the
board cannot be recovered by playing on, and ONE RESET restores the level start
the cached plan was solved from.

Usage (run from the repo root):
    python solvers/generate_space_expedition_training.py \
        --episodes 200 --out data/training_multi_level/space_expedition

    python solvers/generate_space_expedition_training.py --plans
    python solvers/generate_space_expedition_training.py --verify   # + 7 for level 7
    python solvers/generate_space_expedition_training.py --fuzz
    python solvers/generate_space_expedition_training.py --symmetry
    python solvers/generate_space_expedition_training.py --audit
"""

from __future__ import annotations

import itertools
import random
import sys
import time
from collections import deque
from heapq import heappush, heappop
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.puzzlescript_adapter import _render_frame                # noqa: E402
from solvers.common.ps_astar import PSAStarSolver, PSExpert, Plan      # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent

GAME_NAME = "Space_Expedition"

#: Disk cache of each level's start plan AND its optimal-action sets. The
#: searches are seed-independent and level 7 alone costs ~40 s, which every
#: shard of `parallelize_generator` would otherwise repeat on every core. Delete
#: the file to re-derive it.
PLAN_CACHE: Path = _REPO_ROOT / "data" / "space_expedition_plans.json"

#: Refuse to search past this many states. It is a MEMORY budget -- every state
#: is a tuple of tuples -- and it exists so an edited level fails loudly instead
#: of swapping. The largest level here expands 171k.
CAP = 4_000_000

#: Engine direction names in the order the interpreter expands a rule's four
#: directional copies. `_Board` mirrors that order exactly, so this tuple is part
#: of the model rather than a display convention.
_ORDER: tuple[str, ...] = ("up", "down", "left", "right")
_DELTA: tuple[tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
_CODE: dict[str, int] = {name: i for i, name in enumerate(_ORDER)}
#: The five presses, ACTION last. ACTION is a genuine WAIT here (there is no
#: ``require_player_movement``) as well as the astronaut pickup.
_PRESSES: tuple[str, ...] = _ORDER + ("action",)


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

class _Board:
    """Space Expedition's whole rule set, over integer cell indices.

    A state is ``(player, aliens, rocks, crew)``:

      * ``player`` -- cell index of the live player;
      * ``aliens`` -- sorted tuple of alien cells;
      * ``rocks``  -- sorted tuple of movable-asteroid cells;
      * ``crew``   -- sorted tuple of the astronauts still to be picked up.

    Walls and planets are static (no rule creates or destroys either) so they
    live on the board rather than in the state. Corpses and PlayerDead are not
    in the state at all: both are terminal, because the win conditions are
    ``Some Player on Planet`` AND ``No Astronaut`` AND ``No Corpse`` and nothing
    in the game removes a corpse or revives the player. `step` reports either as
    a dead successor.
    """

    def __init__(self, h, w, walls, planets, start):
        self.h, self.w = h, w
        n = h * w
        self.walls, self.planets = walls, planets
        self.nbr = [[-1] * 4 for _ in range(n)]
        for r in range(h):
            for c in range(w):
                cell = r * w + c
                for d, (dr, dc) in enumerate(_DELTA):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w:
                        self.nbr[cell][d] = rr * w + cc
        self.start = start

    @classmethod
    def read(cls, eng, ids) -> "_Board":
        """Build the model, and its start state, from the interpreter's grid."""
        wall, planet, player, alien, rock, crew = ids
        w = eng.width
        walls, planets = set(), set()
        p, al, ro, cr = -1, [], [], []
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                i = r * w + c
                if cell & wall:
                    walls.add(i)
                if cell & planet:
                    planets.add(i)
                if cell & player:
                    p = i
                if cell & alien:
                    al.append(i)
                if cell & rock:
                    ro.append(i)
                if cell & crew:
                    cr.append(i)
        start = (p, tuple(sorted(al)), tuple(sorted(ro)), tuple(sorted(cr)))
        return cls(eng.height, w, walls, planets, start)

    # -- dynamics -------------------------------------------------------------
    def win(self, state) -> bool:
        """``Some Player on Planet`` and ``No Astronaut`` (and ``No Corpse``,
        which holds by construction -- a corpse is a dead successor)."""
        return state[0] in self.planets and not state[3]

    def _alien_forces(self, p, aliens) -> dict:
        """Rule ``[ Alien | ... | Player ] -> [ > Alien | ... | Player ]``.

        The ellipsis is content-agnostic in this interpreter, so an alien
        sharing the player's row or column is sent down that line **even
        through walls, asteroids, astronauts and other aliens**. That is the
        single most important fact about the game: sight is not blocked by
        anything, so the only defence is to not share a line."""
        w = self.w
        pr, pc = divmod(p, w)
        out = {}
        for a in aliens:
            ar, ac = divmod(a, w)
            if ar == pr:
                if ac < pc:
                    out[a] = 3
                elif ac > pc:
                    out[a] = 2
            elif ac == pc:
                if ar < pr:
                    out[a] = 1
                else:
                    out[a] = 0
        return out

    def _annihilate(self, aliens, af) -> None:
        """Rule ``[ > Alien | Alien ] -> [ No Alien | No Alien ]``.

        Two aliens that end up adjacent along the line they are both charging
        down destroy each other. Matches are applied ONE AT A TIME, so a line of
        three loses the pair the scan reaches first and the third survives -- and
        WHICH pair that is depends on the scan order, which the interpreter runs
        AGAINST the rule's direction (bottom-to-top for ``up``, right-to-left for
        ``left``, row-major otherwise; see `_apply_single_rule_forces`).

        That detail is the whole rule. Get it backwards and the survivor is the
        alien at the far end of the line instead of the near one, which on a
        line pointed at the player is the difference between the press being
        safe and being fatal. Four of the fuzz's random boards convicted it.

        Note what those four scan orders amount to. Every alien carrying force
        ``d`` is, by construction, on the one ray running from the player away
        in direction ``-d`` -- that is how it got the force -- and each of the
        four orders runs that ray FROM ITS FAR END. So the pair that dies is
        always the one furthest from the player and the survivor is always the
        nearest, whichever way the board is turned: the rule is
        rotation-equivariant, and this game has none of the chirality
        ps:gobble_rush had to steer around. ``--symmetry`` measures that rather
        than trusting it."""
        nbr, w = self.nbr, self.w
        for d in range(4):
            if d == 0:                                  # up: scan bottom-to-top
                key = lambda x: (-(x // w), x % w)      # noqa: E731
            elif d == 2:                                # left: right-to-left
                key = lambda x: (x // w, -(x % w))      # noqa: E731
            else:                                       # down / right: row-major
                key = lambda x: x                       # noqa: E731
            while True:
                fired = False
                for a in sorted((x for x, dd in af.items() if dd == d), key=key):
                    if a not in aliens:
                        continue
                    n = nbr[a][d]
                    if n >= 0 and n in aliens:
                        aliens.discard(a)
                        aliens.discard(n)
                        af.pop(a, None)
                        af.pop(n, None)
                        fired = True
                if not fired:
                    break

    def step(self, state, press):
        """One press. Returns ``(next state, contended)``.

        The next state is None when the press leads to a state that can never
        win -- the player walked into a wall or an alien, an alien reached the
        player, or an alien turned an astronaut into a corpse -- and ``state``
        itself when the press changed nothing.

        ``contended`` is True when two independently-moving objects wanted the
        same cell in the same turn, which is the only way this game's outcome
        could depend on the order the interpreter expands a rule's four
        directions (see the module docstring)."""
        code = _CODE.get(press, 4) if isinstance(press, str) else press
        p0, al0, ro0, cr0 = state
        aliens, rocks, crew = set(al0), set(ro0), set(cr0)
        nbr, walls = self.nbr, self.walls
        pd = code if code < 4 else None

        af = self._alien_forces(p0, aliens)
        self._annihilate(aliens, af)

        if pd is not None:
            t = nbr[p0][pd]
            if t >= 0 and (t in walls or t in aliens):
                return None, False            # rules 3 and 4: the player dies
        for a, d in af.items():
            n = nbr[a][d]
            if n == p0:
                return None, False            # rule 5: an alien reaches you
        # Rules 6 and 7: a moving player or alien shoves the asteroid ahead of it.
        rf = {}
        if pd is not None:
            t = nbr[p0][pd]
            if t >= 0 and t in rocks:
                rf[t] = pd
        for a, d in af.items():
            n = nbr[a][d]
            if n >= 0 and n in rocks:
                rf[n] = d
        # Rule 12 (pickup) runs BEFORE rule 13 (the corpse), so pressing ACTION
        # on an astronaut an alien is about to reach saves it.
        if code == 4 and p0 in crew:
            crew.discard(p0)
        for a, d in af.items():
            n = nbr[a][d]
            if n >= 0 and n in crew:
                return None, False            # rule 13: an astronaut is killed

        p, aliens, rocks, contended = self._resolve(p0, pd, aliens, af, rocks, rf)
        nxt = (p, tuple(sorted(aliens)), tuple(sorted(rocks)),
               tuple(sorted(crew)))
        return (state if nxt == state else nxt), contended

    def _resolve(self, p, pd, aliens, af, rocks, rf):
        """Move everything that has a force, PuzzleScript-style.

        Every movable object here (player, alien, asteroid) is on the same
        collision layer as the walls, so a mover is blocked by anything solid
        that does not vacate the cell first. Chains of same-direction movers
        (the player and the asteroid it is pushing) move together when the far
        end is free; two chains claiming one cell both stop."""
        nbr, walls = self.nbr, self.walls
        # cell -> ('p' | 'a' | 'r'); one solid object per cell by construction.
        occ = {}
        for a in aliens:
            occ[a] = "a"
        for r in rocks:
            occ[r] = "r"
        occ[p] = "p"
        forces = {}
        if pd is not None:
            forces[p] = pd
        forces.update(af)
        forces.update(rf)
        contended = False

        for _ in range(20):
            movable, moved = [], False
            for cell, d in list(forces.items()):
                chain, cur, free, blocker_moves = [cell], cell, False, False
                while True:
                    n = nbr[cur][d]
                    if n < 0:
                        break
                    if n not in walls and n not in occ:
                        free = True
                        break
                    if n not in walls and forces.get(n) == d:
                        chain.append(n)
                        cur = n
                        continue
                    if n not in walls and n in forces:
                        blocker_moves = True
                    break
                if free:
                    movable.append((chain, d))
                elif blocker_moves:
                    # The blocker is heading somewhere else and may vacate this
                    # cell later in the pass; keep the force and retry.
                    pass
                else:
                    for x in chain:
                        forces.pop(x, None)
            # A chain whose head is the tail of a longer one is the same push
            # group seen from further back; only the longer one moves.
            tails = {x for chain, _ in movable for x in chain[1:]}
            live = [m for m in movable if m[0][0] not in tails]
            claims, clash = {}, set()
            for i, (chain, d) in enumerate(live):
                for x in chain:
                    t = nbr[x][d]
                    if t in claims and claims[t] != i:
                        clash.add(i)
                        clash.add(claims[t])
                        contended = True
                    else:
                        claims[t] = i
            for i, (chain, d) in enumerate(live):
                if i in clash:
                    for x in chain:
                        forces.pop(x, None)
                    continue
                for x in reversed(chain):
                    kind = occ.pop(x)
                    occ[nbr[x][d]] = kind
                    forces.pop(x, None)
                    moved = True
            if not moved or not forces:
                break
        p = next(c for c, k in occ.items() if k == "p")
        aliens = {c for c, k in occ.items() if k == "a"}
        rocks = {c for c, k in occ.items() if k == "r"}
        return p, aliens, rocks, contended

    # -- planning -------------------------------------------------------------
    def setup_goal(self, crew, planets) -> None:
        """Precompute the walk distances the heuristic reads.

        Distances are BFS over cells that are not WALL, i.e. asteroids count as
        free. That is a relaxation and so keeps the bound admissible, and it is
        also nearly tight: walking into an asteroid whose far side is clear is a
        push, which costs exactly the one press a plain step would have."""
        self.goal_crew = tuple(sorted(crew))
        self.dist_from = {t: self._bfs(t) for t in
                          set(self.goal_crew) | set(planets)}
        self.planet_list = tuple(sorted(planets))

    def _bfs(self, src) -> list:
        INF = 1 << 20
        d = [INF] * (self.h * self.w)
        d[src] = 0
        q = deque([src])
        while q:
            cur = q.popleft()
            for n in self.nbr[cur]:
                if n >= 0 and n not in self.walls and d[n] == INF:
                    d[n] = d[cur] + 1
                    q.append(n)
        return d

    def heuristic(self, state) -> int:
        """Presses that must still be spent: a shortest tour through every
        astronaut left, ending on a planet, plus one ACTION per astronaut.

        Admissible -- the tour ignores aliens entirely and treats asteroids as
        floor -- and consistent, since one press moves the player at most one
        cell and a pickup removes exactly the ACTION it costs."""
        p, _al, _ro, crew = state
        best = 1 << 20
        for order in itertools.permutations(crew):
            cur, total = p, 0
            for c in order:
                total += self.dist_from[c][cur]
                cur = c
            total += min(self.dist_from[t][cur] for t in self.planet_list)
            best = min(best, total)
        return best + len(crew)

    def astar(self, cap: int):
        """A* over presses; returns ``(d*, expanded)`` or ``(None, expanded)``.

        The goal is tested on GENERATION rather than on the pop, which is sound
        here for a reason worth stating: ``heuristic`` is zero exactly at a win
        (an empty crew and the player on a planet), so every non-winning node
        has ``f > g`` strictly. A win reachable one press sooner would have a
        predecessor whose ``f`` is at most that shorter length -- strictly less
        than the ``f`` of anything we could be expanding -- so it would have
        been popped, and its win generated, first."""
        start = self.start
        if self.win(start):
            return 0, 0
        h0 = self.heuristic(start)
        pq = [(h0, 0, start)]
        best = {start: 0}
        expanded = 0
        while pq:
            _f, g, state = heappop(pq)
            if best.get(state, -1) != g:
                continue
            expanded += 1
            if expanded > cap:
                return None, expanded
            for press in range(5):
                nxt, _cont = self.step(state, press)
                if nxt is None or nxt == state:
                    continue
                ng = g + 1
                if self.win(nxt):
                    return ng, expanded
                if best.get(nxt, 1 << 30) <= ng:
                    continue
                best[nxt] = ng
                heappush(pq, (ng + self.heuristic(nxt), ng, nxt))
        return None, expanded

    def field(self, dstar: int, cap: int):
        """Exact distance-to-win over every state on a SHORTEST path.

        A forward sweep bounded by ``g + h <= dstar`` (admissible ``h``, so it
        cannot drop an optimal path) records each state's predecessors; a
        reverse sweep from the winning states then gives the true remaining
        press count. Walking that field downhill is what makes both the plan
        provably shortest and its per-step optimal SETS exact rather than
        inferred."""
        pred: dict = {}
        depth = {self.start: 0}
        frontier = [self.start]
        wins = []
        for g in range(dstar):
            nxt_frontier = []
            for state in frontier:
                for press in range(5):
                    nxt, _cont = self.step(state, press)
                    if nxt is None or nxt == state:
                        continue
                    if g + 1 + self.heuristic(nxt) > dstar:
                        continue
                    pred.setdefault(nxt, []).append(state)
                    if nxt in depth:
                        continue
                    depth[nxt] = g + 1
                    if self.win(nxt):
                        wins.append(nxt)
                    else:
                        nxt_frontier.append(nxt)
                if len(depth) > cap:
                    raise MemoryError(f"field exceeded {cap} states")
            frontier = nxt_frontier
        dist = {state: 0 for state in wins}
        queue = deque(wins)
        while queue:
            state = queue.popleft()
            d = dist[state] + 1
            for prev in pred.get(state, ()):
                if prev not in dist:
                    dist[prev] = d
                    queue.append(prev)
        self.reachable = len(depth)
        return dist

    def plan(self, dist: dict):
        """``(presses, optsets)`` from the start, or ``(None, None)``.

        At every state the optimal SET is exactly the presses landing on
        distance ``d - 1`` -- measured off the field, not inferred from what
        kind of move it was."""
        state = self.start
        if state not in dist:
            return None, None
        presses, optsets = [], []
        while dist[state]:
            want = dist[state] - 1
            best = []
            for i, name in enumerate(_PRESSES):
                nxt, _cont = self.step(state, i)
                if nxt is not None and dist.get(nxt, -1) == want:
                    best.append(name)
            if not best:
                return None, None
            presses.append(best[0])
            optsets.append(best)
            state = self.step(state, _PRESSES.index(best[0]))[0]
        return presses, optsets


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class SpaceExpeditionExpert(PSExpert):
    """Plans by building `_Board` off the engine grid and searching THAT; never
    steps the interpreter.

    `PSExpert` still owns everything around the search -- the in-memory memo,
    the on-disk plan cache with its staleness check, the level scoping and the
    snapshot/restore discipline -- so the only override is `_search`.
    `heuristic` is unreachable by construction: no A* runs over engine states
    here, only over `_Board`'s.
    """

    #: `_key` below is dynamic-objects-only, canonical only WITHIN a level.
    scope_by_level = True
    plan_cache_path = PLAN_CACHE

    #: ACTION is in the press set because it is both the astronaut pickup and a
    #: real WAIT. No shipped plan spends one on waiting, but dropping it would
    #: be an unproven prune rather than a measured one.
    directions = list(_PRESSES)

    def setup(self) -> None:
        g = self.g
        name = g.resolve_object_name
        self.wall_ids = set(name("wall"))
        self.planet_ids = set(name("planet"))
        self.player_ids = set(self.game._engine._player_indices)
        self.alien_ids = set(name("alien"))
        self.rock_ids = set(name("asteroidm"))
        self.crew_ids = set(name("astronaut"))
        self.corpse_ids = set(name("corpse"))
        self.bg_id = self.g.obj_name_to_idx["background"]
        self.dyn_ids = (self.player_ids | self.alien_ids | self.rock_ids
                        | self.crew_ids | self.corpse_ids)

    def _key(self, eng) -> frozenset:
        """Every object a settled frame can differ by. Walls and planets are
        left out deliberately -- no rule in this game creates or destroys
        either, so they are static per level (hence ``scope_by_level``) and
        including them would only make the key bigger. The corpse IS in it:
        the key is also the disk cache's staleness signature, and a board with
        a corpse on it is a different (and unwinnable) board."""
        dyn = self.dyn_ids
        return frozenset(
            (r, c, o)
            for r, row in enumerate(eng.grid)
            for c, cell in enumerate(row)
            for o in (cell & dyn)
        )

    def heuristic(self, eng) -> int:
        raise AssertionError(
            "SpaceExpeditionExpert searches the native model; the engine-state "
            "heuristic is unused")

    def board(self, eng) -> _Board:
        board = _Board.read(eng, (self.wall_ids, self.planet_ids,
                                  self.player_ids, self.alien_ids,
                                  self.rock_ids, self.crew_ids))
        board.setup_goal(board.start[3], board.planets)
        return board

    def _search(self, eng) -> "Plan | None":
        """A* for the shortest length, then the exact field inside that bound.

        Two passes rather than one because they answer different questions. A*
        with the admissible tour heuristic proves ``d*`` while touching a tiny
        slice of the space (23 states on level 0, 171k on level 7 against a
        board with 8 aliens and 86 asteroids). The ``g + h <= d*`` field then
        enumerates exactly the states a shortest path can pass through, which
        is what makes the per-step optimal SETS measured rather than inferred
        -- without them every free stretch of the walk would train one
        arbitrary interleaving of the two axes as the only right answer."""
        board = self.board(eng)
        dstar, _expanded = board.astar(self.node_cap)
        if dstar is None:
            return None
        presses, optsets = board.plan(board.field(dstar, self.node_cap))
        return None if presses is None else Plan(presses, optsets)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def _levels():
    solver = SpaceExpeditionSolver()
    game = solver.make_game(0)
    return solver, game, SpaceExpeditionExpert(game)


def _engine_state(eng, expert):
    """The interpreter's grid in `_Board`'s state encoding, plus the two
    terminal markers the model folds into a dead successor."""
    w = eng.width
    p, al, ro, cr = -1, [], [], []
    corpse = False
    for r, row in enumerate(eng.grid):
        for c, cell in enumerate(row):
            i = r * w + c
            if cell & expert.player_ids:
                p = i
            if cell & expert.alien_ids:
                al.append(i)
            if cell & expert.rock_ids:
                ro.append(i)
            if cell & expert.crew_ids:
                cr.append(i)
            if cell & expert.corpse_ids:
                corpse = True
    return ((p, tuple(sorted(al)), tuple(sorted(ro)), tuple(sorted(cr))),
            p < 0 or corpse)


def _report(cap: int = CAP) -> int:
    """Per-level piece counts, search size, plan length and tie coverage -- and
    CERTIFY each plan by replaying it through the interpreter."""
    _solver, game, expert = _levels()
    bad = 0
    total = labels = 0
    for level in range(game.n_levels):
        game.set_level(level)
        eng = game._engine
        board = expert.board(eng)
        _p, al, ro, cr = board.start
        t = time.time()
        dstar, expanded = board.astar(cap)
        if dstar is None:
            print(f"level {level:2d}: {len(al)}A {len(ro):2d}C {len(cr)}N, "
                  f"NO PLAN ({expanded} states in {time.time() - t:.1f}s)")
            bad += 1
            continue
        dist = board.field(dstar, cap)
        presses, optsets = board.plan(dist)
        took = time.time() - t
        won_at = None
        for i, press in enumerate(presses):
            eng.step(press)
            if eng.check_win():
                won_at = i
                break
        ok = won_at == len(presses) - 1
        bad += not ok
        total += len(presses)
        ties = sum(1 for x in optsets if len(x) > 1)
        here = sum(len(x) for x in optsets)
        labels += here
        print(f"level {level:2d}: {len(al)}A {len(ro):2d}C {len(cr)}N, "
              f"{len(presses):3d} presses, {ties:2d} tie steps, "
              f"{here:3d} labels, "
              f"A* {expanded:8d} states, field {board.reachable:7d}, "
              f"{took:6.2f}s, "
              f"{'CERTIFIED' if ok else 'PLAN REJECTED BY THE INTERPRETER'}")
    print(f"total {total} presses over {game.n_levels} levels, "
          f"{labels} labelled presses")
    return 0 if not bad else 1


def _plans_for(game, expert) -> dict:
    """``{level: (presses, optsets)}`` -- every level's field plan, so the fuzz
    and the checks can start from the states a plan actually visits."""
    out = {}
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(game._engine)
        dstar, _n = board.astar(CAP)
        if dstar is None:
            continue
        out[level] = board.plan(board.field(dstar, CAP))
    return out


def _walk(board, eng, expert, state, rng, length, totals) -> int:
    """Random-play ``length`` presses from ``state``, asserting the model and
    the interpreter agree after every one. Returns the number of mismatches
    (0 or 1 -- the walk stops at the first)."""
    for _ in range(length):
        press = rng.choice(_PRESSES)
        before = state
        eng.step(press)
        state, contended = board.step(state, press)
        totals["presses"] += 1
        totals["contended"] += contended
        theirs, dead = _engine_state(eng, expert)
        if state is None:
            totals["deaths"] += 1
            if not dead:
                print(f"  FALSE DEATH after {press} from {before}\n"
                      f"    engine {theirs}")
                return 1
            return 0
        if dead or state != theirs:
            print(f"  MISMATCH after {press}\n    before {before}\n"
                  f"    model  {state}\n    engine {theirs} dead={dead}")
            return 1
        if board.win(state) != eng.check_win():
            print(f"  WIN MISMATCH at {state}")
            return 1
        totals["refused"] += state == before
        totals["pushes"] += before[2] != state[2]
        totals["picks"] += len(before[3]) - len(state[3])
        totals["kills"] += len(before[1]) - len(state[1])
        if board.win(state):
            return 0
    return 0


def _random_board(rng, expert) -> list:
    """A dense synthetic level, and the only thing here that can reach
    ``[ > Alien | Alien ]`` at all.

    Convoy annihilation is the one rule whose SCAN ORDER changes the answer --
    which of a line of aliens survives, and therefore whether the press is
    fatal -- and no shipped board ever gets two aliens adjacent on one of the
    player's lines (measured: zero on all 8 shortest plans, ~4 across the whole
    level fuzz). So the configuration has to be manufactured. These boards do
    nothing else."""
    h, w = rng.randint(5, 8), rng.randint(5, 8)
    cells = [(r, c) for r in range(h) for c in range(w)]
    rng.shuffle(cells)
    grid = [[{expert.bg_id} for _ in range(w)] for _ in range(h)]
    def place(ids, n):
        oid = min(ids)
        for _ in range(n):
            if not cells:
                return
            r, c = cells.pop()
            grid[r][c].add(oid)
    place(expert.player_ids, 1)
    place(expert.alien_ids, rng.randint(2, 5))
    place(expert.rock_ids, rng.randint(0, 4))
    place(expert.wall_ids, rng.randint(0, 6))
    place(expert.crew_ids, rng.randint(0, 2))
    place(expert.planet_ids, 1)
    return grid


def _fuzz(walks: int = 60, length: int = 40, boards: int = 3000) -> int:
    """Assert `_Board` reproduces the interpreter exactly.

    Two populations, because neither alone is enough:

      * random play from every PREFIX of every level's own plan, which puts the
        model in the configurations the shipped plans actually visit (random
        play from a level start dies within a few presses -- an alien adjacent
        in line is unsurvivable -- and would never reach the later boards);
      * dense random 5x5..8x8 boards with 2-5 aliens, which is the only way to
        reach ``[ > Alien | Alien ]`` at all: no shipped level ever gets two
        aliens adjacent on one of the player's lines.

    The counters are printed for exactly that reason -- a run that reports
    agreement while having annihilated no alien has not checked the rule that
    decides half of levels 2-7."""
    _solver, game, expert = _levels()
    eng, g = game._engine, game._game
    rng = random.Random(20260818)
    plans = _plans_for(game, expert)
    bad = 0
    totals = dict(presses=0, refused=0, deaths=0, pushes=0, picks=0, kills=0,
                  contended=0)
    for level in range(game.n_levels):
        game.set_level(level)
        layout = g.levels[level]
        board = expert.board(eng)
        presses = plans.get(level, ([], []))[0]
        seen = dict(totals)
        for prefix in range(0, len(presses) + 1, max(1, len(presses) // 12)):
            for _ in range(walks):
                eng.load_level(layout)
                state = board.start
                for press in presses[:prefix]:
                    eng.step(press)
                    state = board.step(state, press)[0]
                bad += _walk(board, eng, expert, state, rng, length, totals)
        print(f"level {level:2d}: {totals['presses'] - seen['presses']:6d} presses, "
              f"{totals['deaths'] - seen['deaths']:5d} deaths, "
              f"{totals['refused'] - seen['refused']:5d} refused, "
              f"{totals['pushes'] - seen['pushes']:5d} pushes, "
              f"{totals['picks'] - seen['picks']:3d} pickups, "
              f"{totals['kills'] - seen['kills']:4d} aliens destroyed")
    seen = dict(totals)
    for _ in range(boards):
        grid = _random_board(rng, expert)
        eng.load_level(grid)
        board = _Board.read(eng, (expert.wall_ids, expert.planet_ids,
                                  expert.player_ids, expert.alien_ids,
                                  expert.rock_ids, expert.crew_ids))
        bad += _walk(board, eng, expert, board.start, rng, 12, totals)
    print(f"random boards: {totals['presses'] - seen['presses']:6d} presses, "
          f"{totals['deaths'] - seen['deaths']:5d} deaths, "
          f"{totals['refused'] - seen['refused']:5d} refused, "
          f"{totals['pushes'] - seen['pushes']:5d} pushes, "
          f"{totals['picks'] - seen['picks']:3d} pickups, "
          f"{totals['kills'] - seen['kills']:4d} aliens destroyed")
    print(f"{totals['presses']} transitions ({totals['deaths']} fatal, "
          f"{totals['refused']} refused, {totals['kills']} aliens annihilated, "
          f"{totals['contended']} contended): "
          f"{'model matches the interpreter' if not bad else f'{bad} MISMATCHES'}")
    return 0 if not bad else 1


def _independent(board: _Board, state, limit: int) -> "int | None":
    """Presses from ``state`` to a win, or None if that is more than ``limit``.

    A plain forward breadth-first search to the FIRST winning state, pruned by
    ``depth + heuristic > limit`` (sound, because the heuristic is admissible --
    which `_verify` asserts against the field before trusting it here). It is
    the second book in `_verify`'s double entry: `_Board.field` derives its
    answer BACKWARDS, from the win states over a predecessor map built during
    the forward sweep, and that bookkeeping is the one piece of logic here the
    interpreter fuzz cannot reach -- the fuzz only ever exercises `step`. This
    shares nothing with it but `step` and `heuristic`.

    The bound is also what makes the recomputation affordable: a press is
    convicted by showing no win at the expected depth, which never needs a
    deeper search than that."""
    if board.win(state):
        return 0
    seen = {state}
    frontier = [state]
    for depth in range(1, limit + 1):
        nxt_frontier = []
        for cur in frontier:
            for press in range(5):
                nxt, _c = board.step(cur, press)
                if nxt is None or nxt == cur or nxt in seen:
                    continue
                if board.win(nxt):
                    return depth
                if depth + board.heuristic(nxt) > limit:
                    continue
                seen.add(nxt)
                nxt_frontier.append(nxt)
        frontier = nxt_frontier
    return None


def _verify(levels: "tuple[int, ...] | None" = None) -> int:
    """Double-entry check of every plan length AND every tie set.

    Three separate claims, checked three separate ways:

      * the LENGTH is already double-entered by `_report`: A* proves ``d*``
        without ever touching the field, and the interpreter certifies that the
        field's plan wins on its LAST press and no earlier. Both are re-asserted
        here so one command covers everything.
      * the HEURISTIC is admissible, asserted over every state of the field
        rather than argued -- ``h(s) <= dist(s)`` everywhere. Everything below
        leans on it, and `field`'s ``g + h <= d*`` prune leans on it too, so a
        heuristic that over-estimated would silently drop optimal paths from
        both.
      * the TIE SETS are re-derived by `_independent`, which shares no code with
        the predecessor map they come from.

    Level 7 is excluded by default: its field is 173k states and re-searching it
    once per candidate press costs minutes. Pass ``--verify 7`` to pay for it."""
    _solver, game, expert = _levels()
    bad = 0
    for level in (levels if levels is not None else range(game.n_levels - 1)):
        game.set_level(level)
        board = expert.board(game._engine)
        dstar, _n = board.astar(CAP)
        dist = board.field(dstar, CAP)
        presses, optsets = board.plan(dist)
        star = dist[board.start]
        t = time.time()
        notes = []
        if star != len(presses) or star != dstar:
            notes.append(f"LENGTH {star} != {len(presses)} / A* {dstar}")
        loose = [x for x in dist if board.heuristic(x) > dist[x]]
        if loose:
            notes.append(f"HEURISTIC INADMISSIBLE at {len(loose)} states")
        state = board.start
        for i, press in enumerate(presses):
            want = []
            for j, name in enumerate(_PRESSES):
                nxt, _c = board.step(state, j)
                if nxt is None or nxt == state:
                    continue
                if _independent(board, nxt, star - i - 1) == star - i - 1:
                    want.append(name)
            if want != optsets[i]:
                notes.append(f"step {i}: {optsets[i]} != {want}")
            state = board.step(state, _PRESSES.index(press))[0]
        bad += len(notes)
        print(f"level {level:2d}: d*={star:3d}, {len(dist):6d} field states "
              f"admissible, {sum(len(x) for x in optsets):3d} labelled presses "
              f"re-derived in {time.time() - t:6.2f}s: "
              f"{'AGREES' if not notes else '; '.join(notes[:3])}")
    print("verify clean" if not bad else f"VERIFY FAILED: {bad} disagreements")
    return 0 if not bad else 1


#: Grid transform -> the direction relabelling that goes with it. A rotation
#: does not only move the cells: the player facing PlayerUp on the reference
#: board is PlayerRight on the board turned 90 degrees clockwise, and a
#: comparison that forgets this reports a divergence on every single press.
_ROT_D: dict[str, str] = {"up": "right", "right": "down",
                          "down": "left", "left": "up"}
_FLIP_D: dict[str, str] = {"up": "up", "down": "down",
                           "left": "right", "right": "left"}


def _tdir(direction: str, k: int, mirror: bool) -> str:
    if direction == "action":
        return direction
    for _ in range(k):
        direction = _ROT_D[direction]
    return _FLIP_D[direction] if mirror else direction


def _tgrid(grid, k: int, mirror: bool, face_ids):
    """``grid`` turned ``k`` quarter-turns clockwise, then optionally mirrored
    left-right, with every player FACING relabelled to match."""
    for _ in range(k):
        h = len(grid)
        grid = [[grid[h - 1 - r][c] for r in range(h)]
                for c in range(len(grid[0]))]
    if mirror:
        grid = [list(reversed(row)) for row in grid]
    out = []
    for row in grid:
        new_row = []
        for cell in row:
            c = set(cell)
            for d, oid in face_ids.items():
                if oid in cell:
                    c.discard(oid)
                    c.add(face_ids[_tdir(d, k, mirror)])
            new_row.append(c)
        out.append(new_row)
    return out


def _symmetry() -> int:
    """Confirm ON THE INTERPRETER that no plan step depends on which way the
    board is facing.

    `_execute_single` drives each of a rule's four directional copies to a
    fixpoint IN TURN (up, down, left, right), so a rule whose matches can
    contend is settled by an order that is a fact about the SCREEN, not about
    the board -- and the mandatory rotation augmentation then presents the same
    physical situation resolving both ways (this is what ps:gobble_rush had to
    measure and steer around). The one rule here that could do it is
    ``[ > Alien | Alien ]``: a line of three charging aliens loses the pair the
    scan reaches first, and which pair that is flips under a 180-degree turn.

    So each level's plan is replayed on all 8 turned and mirrored copies of its
    own board and the result compared against the transform of the reference
    run, press by press. This is the only check that can see it: the fuzz plays
    one board, where the model and the interpreter agree on an order neither
    can perceive."""
    _solver, game, expert = _levels()
    eng, g = game._engine, game._game
    faces = {"up": g.obj_name_to_idx["playerup"],
             "down": g.obj_name_to_idx["playerdown"],
             "left": g.obj_name_to_idx["playerleft"],
             "right": g.obj_name_to_idx["playerright"]}
    plans = _plans_for(game, expert)
    bad = 0
    for level in range(game.n_levels):
        presses = plans.get(level, ([], []))[0]
        eng.load_level(g.levels[level])
        ref = [[[frozenset(c) for c in row] for row in eng.grid]]
        for press in presses:
            eng.step(press)
            ref.append([[frozenset(c) for c in row] for row in eng.grid])
        measured = None
        for k in range(4):
            for mirror in (False, True):
                if (k, mirror) == (0, False):
                    continue
                eng.load_level(_tgrid(g.levels[level], k, mirror, faces))
                want = [_tgrid(x, k, mirror, faces) for x in ref]
                for i, press in enumerate(presses):
                    eng.step(_tdir(press, k, mirror))
                    if eng.grid != want[i + 1]:
                        measured = i if measured is None else min(measured, i)
                        break
        bad += measured is not None
        print(f"level {level:2d}: {len(presses):3d} presses on 8 presentations: "
              f"{'ORIENTATION-FREE' if measured is None else f'DIVERGES AT STEP {measured}'}")
    print("symmetry check clean" if not bad
          else f"SYMMETRY FAILED: {bad} levels depend on the screen's facing")
    return 0 if not bad else 1


#: Every cell stack the eight levels can present: one of the four layer-2
#: objects (or none) under one of the layer-3 ones (or none). The four player
#: FACINGS are separate sprites and are listed separately -- they carry no
#: state, but a facing that renders identically to another composition would
#: make a frame ambiguous all the same. ``player_on_planet`` is the WIN frame
#: and is the one a colour-only audit is most likely to be blind to.
_LAYER2: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("", ()), ("planet", ("planet",)), ("crew", ("astronaut",)),
    ("corpse", ("corpse",)),
)
_LAYER3: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("", ()), ("wall", ("wall",)),
    ("playerup", ("playerup",)), ("playerdown", ("playerdown",)),
    ("playerleft", ("playerleft",)), ("playerright", ("playerright",)),
    ("dead", ("playerdead",)), ("alien", ("alien",)), ("rock", ("asteroidm",)),
)


def _compositions() -> dict:
    out = {}
    for n2, o2 in _LAYER2:
        for n3, o3 in _LAYER3:
            name = "_on_".join(x for x in (n3, n2) if x) or "floor"
            out[name] = o2 + o3
    return out


def _audit() -> int:
    """Assert every cell COMPOSITION is distinct at every cell size in use.

    Whole frames are compared, not cell crops: `_render_frame` upscales the
    board to fill 64x64 and letterboxes it, so the output's cell grid is not
    ``cell_px``-aligned and an arithmetic crop reads the wrong window (see
    ps:esl_puzzle_game and ps:explod, where exactly that made an audit report
    every composition identical to floor). This game spans cell sizes 2 through
    6 -- the widest range of any ps: game here -- and the 25x23 level renders at
    2 px, where a 5x5 sprite is sampled at two of its rows and columns."""
    _solver, game, _expert = _levels()
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    comps = _compositions()

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
            shots[name] = np.asarray(_render_frame(eng, g)).copy()
        clashes = [(a, b) for a, b in itertools.combinations(shots, 2)
                   if np.array_equal(shots[a], shots[b])]
        bad += len(clashes)
        print(f"{h:2d}x{w:2d} (cell {64 // max(h, w)}px, levels "
              f"{','.join(str(x) for x in levels)}): "
              f"{'OK' if not clashes else 'IDENTICAL ' + str(clashes)}")
    print("audit clean" if not bad
          else f"AUDIT FAILED: {bad} indistinguishable pairs")
    return 0 if not bad else 1


class SpaceExpeditionSolver(PSAStarSolver):
    game_id = "puzzlescript_space_expedition"
    game_name = GAME_NAME
    expert_cls = SpaceExpeditionExpert

    #: The longest plan is 106 presses (level 6); the rest is room for the
    #: episode's exploration prefix (at most 15 presses plus the RESET) inside
    #: the adapter's 200-press per-level budget.
    max_steps = 160

    #: Level 7 expands 171k states. The cap is a MEMORY budget, not a time one.
    node_cap = CAP

    def prepare_expert(self, game, expert) -> None:
        """Build every level's plan before `discover_solvable` asks for it --
        the same work either way, but it makes the one-time cost visible as
        startup rather than as a mysteriously slow first seed, and it fills the
        disk cache in one pass."""
        for level in range(game.n_levels):
            if level in self.skip_levels:
                continue
            game.set_level(level)
            expert.plan(game._engine, level)


if __name__ == "__main__":
    if "--verify" in sys.argv:
        i = sys.argv.index("--verify")
        rest = tuple(int(x) for x in sys.argv[i + 1:] if x.isdigit())
        sys.exit(_verify(rest or None))
    if "--symmetry" in sys.argv:
        sys.exit(_symmetry())
    if "--audit" in sys.argv:
        sys.exit(_audit())
    if "--plans" in sys.argv:
        sys.exit(_report())
    if "--fuzz" in sys.argv:
        sys.exit(_fuzz())
    sys.exit(SpaceExpeditionSolver.main())
