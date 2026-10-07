"""Shared BaseSolver base for the MiniGrid gridworld generators.

ONE expert for the whole MiniGrid family. Every env in `MinigridAdapter`'s
supported list -- empty rooms, four-rooms, multi-room, lava/wall crossings,
dist-shift, dynamic obstacles, door+key, unlock, (blocked-)unlock-pickup,
key-corridor, locked-room and obstructed-maze -- is the SAME planning problem
once you write the mechanics down properly, so the per-game generators are pure
configuration (`env_id`, `objective`, `target`, `action_set`) and carry no search
code at all.

The egocentric action set (see ``MinigridAdapter._ACTION_MAP``)::

    ACTION1 forward    ACTION2 drop    ACTION3 turn-left
    ACTION4 turn-right ACTION5 interact

``ACTION5`` is CONTEXT-SENSITIVE in the adapter: facing a door it toggles it,
facing a box with contents it opens the box, facing anything pickable it picks it
up. That collapses MiniGrid's separate toggle/pickup into one key, which is what
lets the whole family fit in ACTION1..ACTION5 -- ACTION6 is the mouse click in
ARC-AGI-3 (``utils.explore.CLICK_ACTION``) and ACTION7 is the adapter's UNDO, so
neither is available as a game action.

The planner
-----------
A single exact shortest-path field over the compact state

    (x, y, dir, carried_object, cleared_blockers_mask)

with a virtual terminal node ``_T`` that every objective hooks into:

* ``goal``      -- stepping FORWARD onto a goal cell ends the episode;
* ``carry``     -- picking up the object matching ``target`` ends it;
* ``open_door`` -- toggling a locked/closed door open ends it (MiniGrid-Unlock);
* ``open_box``  -- internal fallback objective: reach and open ANY box, used by
                   the obstructed-maze family where keys are hidden inside boxes.

Two modelling tricks keep the state vector that small while staying exact enough
to drive the engine:

* **doors are folded into the move.** Passing a shut door costs 2 (``ACTION5``
  then ``ACTION1``) and a LOCKED one additionally requires the matching key in
  hand -- so door state never enters the state vector.
* **the world is re-read every step.** ``solve_from`` / ``optimal_set_from`` both
  read the LIVE grid, and the field is cached against a signature of that grid
  (``Grid.encode()``), so any world change -- a door opened, an object picked up,
  dropped or revealed, an obstacle that moved -- invalidates it and re-plans. The
  only approximations the search makes (a dropped object vanishes; a picked-up
  object's cell stays blocked) are therefore *lookahead* approximations that are
  corrected the instant they happen; the action actually taken is always chosen
  against the true current world.

Only genuinely-blocking objects (a ball parked in a doorway) get a bit in the
``cleared`` mask -- without it the search would call e.g. BlockedUnlockPickup
unsolvable at step 0, because the ball sits on the one cell from which the locked
door can be reached.

NOT privileged (see [[no-privileged-solvers]]): the planner reads the grid, the
agent pose and what the agent itself is carrying -- all of which the fully-
observed frame shows or the agent's own action history implies. It never reads
``Box.contains``; hidden keys are found by opening boxes and observing.

Subclass: set ``env_id`` (``game_id`` is derived from it, matching
``MinigridAdapter.game_id``), ``objective``/``target`` and the ``action_set``.
"""
from __future__ import annotations

import heapq
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np                                          # noqa: E402
from arcengine import ActionInput, GameAction, GameState    # noqa: E402
from adapters import MinigridAdapter                        # noqa: E402
from utils.explore import Action                            # noqa: E402
from solvers.base_solver import BaseSolver, DriveResult, _as_action  # noqa: E402

# (dx, dy) for each MiniGrid agent_dir: 0=east, 1=south, 2=west, 3=north.
DIR_VEC: list[tuple[int, int]] = [(1, 0), (0, 1), (-1, 0), (0, -1)]

# Egocentric action ids, as mapped by MinigridAdapter._ACTION_MAP.
A_FORWARD, A_DROP, A_LEFT, A_RIGHT, A_INTERACT = 1, 2, 3, 4, 5
_ID_TO_GA = {
    A_FORWARD: GameAction.ACTION1, A_DROP: GameAction.ACTION2,
    A_LEFT: GameAction.ACTION3, A_RIGHT: GameAction.ACTION4,
    A_INTERACT: GameAction.ACTION5,
}

_PICKABLE = ("key", "ball", "box")
_T = "WIN"                 # virtual terminal node of the search graph


# ---------------------------------------------------------------------------
# grid helpers
# ---------------------------------------------------------------------------
def _nbrs(p, cells):
    x, y = p
    for q in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
        if q in cells:
            yield q


def _articulation(cells: set) -> set:
    """Articulation points of the 4-connected graph induced on ``cells``.

    Used for two things: deciding which parked objects genuinely BLOCK a passage
    (and so deserve a bit in the cleared-mask), and refusing to drop a carried
    object onto a cell whose loss would cut the map in two. Iterative Tarjan, so
    a 25x25 MultiRoom can't blow the recursion limit."""
    disc: dict = {}
    low: dict = {}
    ap: set = set()
    timer = 0
    for root in cells:
        if root in disc:
            continue
        disc[root] = low[root] = timer
        timer += 1
        root_children = 0
        stack = [(root, None, _nbrs(root, cells))]
        while stack:
            node, parent, it = stack[-1]
            descended = False
            for nb in it:
                if nb == parent:
                    continue
                if nb in disc:
                    if disc[nb] < low[node]:
                        low[node] = disc[nb]
                else:
                    disc[nb] = low[nb] = timer
                    timer += 1
                    if node == root:
                        root_children += 1
                    stack.append((nb, node, _nbrs(nb, cells)))
                    descended = True
                    break
            if descended:
                continue
            stack.pop()
            if stack:
                par = stack[-1][0]
                if low[node] < low[par]:
                    low[par] = low[node]
                if par != root and low[node] >= disc[par]:
                    ap.add(par)
        if root_children > 1:
            ap.add(root)
    return ap


# ---------------------------------------------------------------------------
# the model + exact field
# ---------------------------------------------------------------------------
class World:
    """What the planner is allowed to know about the board.

    The full-observation solvers fill this straight from the env's grid
    (`from_env`); the partial-observation ones fill it from their REMEMBERED map
    of what the 7x7 view has actually shown, with everything not yet seen listed
    in ``unknown`` (see ``common/minigrid_partial.py``). Keeping it a plain value
    object is what lets one search serve both -- the planner never touches the
    env, so it cannot accidentally read a cell the agent has not seen.

    Cells are listed POSITIVELY -- `open`, `goals`, `doors`, `objs`, `unknown` --
    and anything named in none of them is impassable. That is what lets the same
    search run on a partial map with no idea how big the board is: a wall, a lava
    tile and a cell beyond everything ever seen are all simply "not listed"."""

    __slots__ = ("open", "goals", "doors", "objs", "carry", "pos", "dir",
                 "unknown")

    def __init__(self, *, open, goals, doors, objs, carry, pos, dir,
                 unknown=frozenset()) -> None:
        self.open = open        # known-empty, walkable cells
        self.goals = goals
        self.doors = doors      # pos -> (color, "open"|"closed"|"locked")
        self.objs = objs        # pos -> (kind, color)
        self.carry = carry      # (kind, color) or None
        self.pos, self.dir = pos, dir
        self.unknown = unknown  # cells never yet seen (partial observation only)

    @property
    def reachable_ever(self) -> set:
        """Every cell that could be stood on once the movable things move --
        used for the articulation tests behind blocker bits and drop targets."""
        return self.open | self.goals | set(self.doors) | set(self.objs)

    @classmethod
    def from_env(cls, u) -> "World":
        grid = u.grid
        open_, goals, doors, objs = set(), set(), {}, {}
        for x in range(grid.width):
            for y in range(grid.height):
                c = grid.get(x, y)
                if c is None or c.type == "floor":
                    open_.add((x, y))
                    continue
                t = c.type
                if t == "goal":
                    goals.add((x, y))
                elif t == "door":
                    st = ("open" if c.is_open
                          else ("locked" if c.is_locked else "closed"))
                    doors[(x, y)] = (c.color, st)
                elif t in _PICKABLE:
                    objs[(x, y)] = (t, c.color)
                # wall / lava: listed nowhere, so impassable
        car = u.carrying
        return cls(open=open_, goals=goals, doors=doors, objs=objs,
                   carry=(car.type, car.color) if car is not None else None,
                   pos=(int(u.agent_pos[0]), int(u.agent_pos[1])),
                   dir=int(u.agent_dir))


class _Model:
    """A `World` plus the exact action-cost field over it.

    Built once per (world signature, objective) and then answers every step by a
    table lookup: `plan_from` descends the field for an optimal action sequence
    and `ties_at` reads off EVERY equally-optimal next action, so the plan's head
    is always a member of the tie set by construction."""

    def __init__(self, world: World, *, objective: str, target, action_set,
                 boxes_contain_keys: bool, max_blocker_bits: int,
                 max_lock_bits: int = 6, allow_pickup: bool = True) -> None:
        self.objective = objective
        self.target = target
        self.boxes_contain_keys = boxes_contain_keys
        self.can_interact = A_INTERACT in action_set
        self.can_drop = A_DROP in action_set and allow_pickup
        #: When False the search may still open doors and grab the OBJECTIVE, but
        #: not pick things up along the way. That collapses the carried-object and
        #: cleared-blocker dimensions to nothing, which is what makes the
        #: "just walk there" model the partial-observation solver leans on for
        #: exploration cost a millisecond instead of a tenth of a second.
        self.allow_pickup = allow_pickup

        self.open = world.open          # known-empty, walkable cells
        self.goals = world.goals
        self.doors = world.doors        # pos -> (color, "open"|"closed"|"locked")
        self.objs = world.objs          # pos -> (kind, color)
        self.unknown = world.unknown    # frontier targets (partial observation)

        carry = world.carry
        self.pos = world.pos
        self.dir = world.dir

        # Objects the agent may hold. Container boxes can't be picked up (the
        # adapter turns ACTION5-on-a-full-box into "open it"), so they are only
        # a carry option when a box IS the target.
        pick = ({kc for p, kc in self.objs.items()
                 if not (kc[0] == "box" and boxes_contain_keys)}
                if (self.can_interact and allow_pickup) else set())
        if carry is not None:
            pick.add(carry)
        self.carry_opts = [None] + sorted(pick)
        self.ci_of = {kc: i for i, kc in enumerate(self.carry_opts)}
        self.start_ci = self.ci_of.get(carry, 0)

        # "wide" = everything that is passable once the movable stuff is moved.
        wide = world.reachable_ever
        cuts = (_articulation(wide)
                if (self.can_drop or (self.can_interact and self.objs))
                else set())

        # Cells we are willing to DROP a carried object onto: never a door, never
        # the goal, and never a cut vertex (dropping there would wall us off).
        drop_ok = (wide - cuts - self.goals - set(self.doors)
                   if self.can_drop else set())

        # Cleared-blocker bits: which parked objects the search may model itself
        # picking up in order to free their cell.
        #
        # Giving EVERY object a bit explodes the state space and, worse, invites
        # plans that pick something up purely to walk through it and then drop it
        # somewhere else -- which the "a dropped object vanishes" approximation
        # scores as free, so the expert dithers. Giving bits only to objects that
        # are individually a cut vertex misses the case that actually dead-ends
        # obstructed-maze levels: several objects that are each harmless alone but
        # TOGETHER wall off half a room. Hence the middle rule -- an object earns a
        # bit when it plausibly obstructs: parked in a doorway, sitting on a cut
        # vertex, or touching two different components of the object-free space.
        self.blk_bit: dict = {}
        if self.can_interact and allow_pickup:
            comp = self._components(wide - set(self.objs))
            cands = [p for p in sorted(self.objs)
                     if not (self.objs[p][0] == "box" and boxes_contain_keys)]
            near_door, splits = [], []
            for p in cands:
                orth = ((p[0] + 1, p[1]), (p[0] - 1, p[1]),
                        (p[0], p[1] + 1), (p[0], p[1] - 1))
                if any(q in self.doors for q in orth):
                    near_door.append(p)
                elif len({comp[q] for q in orth if q in comp}) > 1 or p in cuts:
                    splits.append(p)
            for p in (near_door + splits)[:max_blocker_bits]:
                self.blk_bit[p] = len(self.blk_bit)

        # LOCKED doors get a bit of their own: unlike a merely-shut door (which
        # anyone can re-open, so its toggle is folded into the +1 move cost), a
        # locked one is only openable WHILE holding its key -- and the agent
        # routinely has to drop that key later to free its hands. Without
        # remembering that the door is now unlocked the search would believe the
        # way back through it is barred, and call the level unsolvable.
        self.lock_bit: dict = {}
        if self.can_interact:
            base = len(self.blk_bit)
            for p, (_c, st) in sorted(self.doors.items()):
                if st == "locked" and len(self.lock_bit) < max_lock_bits:
                    self.lock_bit[p] = base + len(self.lock_bit)

        # A dropped object is modelled as GONE -- the search may not use it again,
        # which is true of every key and blocking ball these envs make you put
        # down, but it also means the drop cell is modelled as staying free. That
        # is the one approximation in this planner that is not self-correcting, so
        # it is fenced in twice: drop targets are never cut vertices (parking
        # something can't seal a passage) and dropping back onto a cell we had
        # just cleared re-blocks it (see `_succ`) -- without that, "pick this up
        # and put it straight back down" reads as a strict improvement, hands free
        # AND the cell free, and the expert dithers on one square forever. The
        # residual case, alternating between two whole world states, is caught by
        # the recorder-side loop guard in `MinigridSolver._lookup`.
        self.drop_ok = drop_ok

        self.start = (self.pos[0], self.pos[1], self.dir, self.start_ci, 0)
        self.edges: dict = {}
        self.togo: dict = {}
        #: Set by `_build` when the world holds no terminal for this objective at
        #: all, so an empty `togo` means "impossible here", not "the root pose was
        #: enumerated from somewhere else" -- see the re-root branch in `_lookup`.
        self.impossible = False
        self._build()

    # ── search graph ─────────────────────────────────────────────────────────
    @staticmethod
    def _components(cells: set) -> dict:
        """cell -> connected-component id, over the 4-connected graph."""
        comp: dict = {}
        for root in cells:
            if root in comp:
                continue
            cid = len(comp) + 1
            stack = [root]
            comp[root] = cid
            while stack:
                for q in _nbrs(stack.pop(), cells):
                    if q not in comp:
                        comp[q] = cid
                        stack.append(q)
        return comp

    def _is_target(self, kind, color) -> bool:
        return (self.target is not None and kind == self.target[0]
                and (self.target[1] is None or color == self.target[1]))

    def _succ(self, s):
        """Outgoing edges ``(next_state, cost, action_seq)`` of one state."""
        x, y, d, ci, mask = s
        out = [((x, y, (d - 1) % 4, ci, mask), 1, (A_LEFT,)),
               ((x, y, (d + 1) % 4, ci, mask), 1, (A_RIGHT,))]
        dx, dy = DIR_VEC[d]
        px, py = x + dx, y + dy
        p = (px, py)
        if p in self.unknown:
            # FRONTIER objective (partial observation): standing here facing an
            # unseen cell is the win, because arriving reveals the whole 7x7 cone
            # in that direction. A free edge -- the information is the payoff, and
            # the actions that got us here are already paid for.
            if self.objective == "frontier":
                out.append((_T, 0, ()))
            return out

        bit = self.blk_bit.get(p)
        obj_there = p in self.objs and not (bit is not None and (mask >> bit) & 1)

        if p in self.goals:
            if self.objective == "goal":
                out.append((_T, 1, (A_FORWARD,)))
            else:
                out.append(((px, py, d, ci, mask), 1, (A_FORWARD,)))
        elif p in self.doors:
            color, st = self.doors[p]
            lb = self.lock_bit.get(p)
            if st == "open" or (lb is not None and (mask >> lb) & 1):
                out.append(((px, py, d, ci, mask), 1, (A_FORWARD,)))
            elif self.can_interact:
                if st == "closed":
                    openable, nmask = True, mask
                else:
                    openable = self.carry_opts[ci] == ("key", color)
                    nmask = mask if lb is None else (mask | (1 << lb))
                if openable:
                    if self.objective == "open_door":
                        out.append((_T, 1, (A_INTERACT,)))
                    else:
                        out.append(((px, py, d, ci, nmask), 2,
                                    (A_INTERACT, A_FORWARD)))
        elif obj_there:
            kind, color = self.objs[p]
            if not self.can_interact:
                return out
            if kind == "box" and self.boxes_contain_keys:
                if self.objective == "open_box":
                    out.append((_T, 1, (A_INTERACT,)))
            elif ci == 0:                              # hands free -> pick it up
                if self.objective == "carry" and self._is_target(kind, color):
                    out.append((_T, 1, (A_INTERACT,)))
                    return out                         # picking it up IS the win
                nci = self.ci_of.get((kind, color)) if self.allow_pickup else None
                if nci is not None:
                    nmask = mask if bit is None else (mask | (1 << bit))
                    out.append(((x, y, d, nci, nmask), 1, (A_INTERACT,)))
        elif p in self.open or bit is not None:        # empty, or a cleared cell
            out.append(((px, py, d, ci, mask), 1, (A_FORWARD,)))
            if self.can_drop and ci != 0 and p in self.drop_ok:
                # Dropping back onto a cell we had cleared puts an object on it
                # again, so its cleared bit has to go out (see the drop_ok note).
                nmask = mask if bit is None else (mask & ~(1 << bit))
                out.append(((x, y, d, 0, nmask), 1, (A_DROP,)))
        return out

    def _terminal_exists(self) -> bool:
        """Could ANY state in this world reach `_T`? Read straight off the world,
        without enumerating anything.

        Every objective hooks its terminal edge onto a specific feature of the
        board (`_succ`): "goal" onto a goal tile, "carry" onto a target object to
        pick up, "open_box" onto a box, "open_door" onto a door, "frontier" onto
        an unseen cell. When the world holds none of them the whole enumeration
        below can only conclude "no plan" -- so this returns False and `togo`
        stays empty, which is EXACTLY the answer the full search would give.

        This is not an approximation, and it is where the partial-observation
        family spends most of its time: blind, the real objective is unreachable
        for most of the level (the ball is behind a door nobody has opened yet),
        so `_lookup` was rebuilding the full model every single step just to be
        told again that the thing it wants has never been seen. Only the negative
        is decided here -- anything not provably impossible falls through to the
        real search."""
        o = self.objective
        if o == "goal":
            return bool(self.goals)
        if o == "frontier":
            return bool(self.unknown)
        if not self.can_interact:
            # Every remaining objective ends in an ACTION5 the agent cannot take.
            return False
        if o == "carry":
            return any(self._is_target(k, c) for k, c in self.objs.values())
        if o == "open_box":
            return self.boxes_contain_keys and any(
                k == "box" for k, _c in self.objs.values())
        if o == "open_door":
            return bool(self.doors)
        return True

    def _build(self) -> None:
        """Enumerate the states reachable from the live pose, then reverse-
        Dijkstra from the terminal so every state carries its exact
        actions-to-win. One build serves every step until the world changes."""
        start = self.start
        if self.objective == "carry" and self.carry_opts[self.start_ci] is not None:
            k, c = self.carry_opts[self.start_ci]
            if self._is_target(k, c):
                self.togo = {start: 0}
                return
        if not self._terminal_exists():
            self.impossible = True       # no reachable win; `togo` stays empty
            return
        edges = self.edges
        stack = [start]
        seen = {start}
        while stack:
            s = stack.pop()
            es = self._succ(s)
            edges[s] = es
            for t, _c, _q in es:
                if t is not _T and t not in seen:
                    seen.add(t)
                    stack.append(t)

        pred: dict = {}
        for s, es in edges.items():
            for t, c, _q in es:
                pred.setdefault(t, []).append((s, c))
        if _T not in pred:
            return
        togo = {_T: 0}
        pq = [(0, 0, _T)]
        tick = 1
        while pq:
            d, _n, s = heapq.heappop(pq)
            if d > togo.get(s, 1 << 60):
                continue
            for u, c in pred.get(s, ()):
                nd = d + c
                if nd < togo.get(u, 1 << 60):
                    togo[u] = nd
                    heapq.heappush(pq, (nd, tick, u))
                    tick += 1
        self.togo = togo

    # ── read-out ─────────────────────────────────────────────────────────────
    def state_at(self, pose):
        """The live search state for ``pose = (pos, dir, carry)``. ``mask`` is
        always 0: a blocker that has been picked up is simply gone from the
        re-read world, so it is no longer a blocker at all."""
        (x, y), d, carry = pose
        return (x, y, d, self.ci_of.get(carry, 0), 0)

    def solved(self, s) -> bool:
        return self.togo.get(s) == 0

    def plan_from(self, s) -> list:
        """An action-optimal plan from ``s``, by descending the field."""
        d = self.togo.get(s)
        if d is None:
            return []
        plan: list = []
        guard = 0
        while d > 0 and guard < 4000:
            guard += 1
            for t, c, q in self.edges.get(s, ()):
                if self.togo.get(t, 1 << 60) == d - c:
                    plan.extend(q)
                    s, d = t, d - c
                    break
            else:
                return []
            if s is _T:
                break
        return plan

    def ties_at(self, s) -> list:
        """Every equally-optimal next action at ``s`` -- the first action of each
        edge that reduces the exact actions-to-win by its own cost."""
        d = self.togo.get(s)
        if not d:
            return []
        seen: list = []
        for t, c, q in self.edges.get(s, ()):
            if self.togo.get(t, 1 << 60) == d - c and q[0] not in seen:
                seen.append(q[0])
        return seen


# ---------------------------------------------------------------------------
# BaseSolver
# ---------------------------------------------------------------------------
class MinigridSolver(BaseSolver):
    """Standard multilevel BaseSolver for a MiniGrid env. Recovery-enabled.

    A "level" is one maze; an episode is ``levels_per_episode`` distinct mazes.
    Episode ``e`` owns the maze block ``[e*L, e*L+L)`` so episodes never overlap:
    ``make_game`` seeds the adapter at ``e*L`` and ``set_level(i)`` selects maze
    ``e*L+i``. Frame rotation is a per-level adapter augmentation that does NOT
    touch the egocentric actions, so the planner works in env space and needs no
    inverse-rotation."""

    env_id: str = ""

    #: What ends the episode. "goal" = step onto the green goal; "carry" = hold
    #: the object described by `target`; "open_door" = toggle a door open.
    objective: str = "goal"
    #: (kind, color) of the object to carry, ``color=None`` = any. Only for
    #: ``objective == "carry"``.
    target: tuple | None = None
    #: Valid action ids. Drop (2) is only needed where the agent must free its
    #: hands; interact (5) only where there are doors / objects.
    action_set: tuple = (1, 3, 4)
    #: Obstructed-maze family: boxes are CONTAINERS holding the door keys, so
    #: ACTION5 on one opens it rather than picking it up. Their contents are
    #: hidden, so the expert opens boxes to discover keys instead of reading them.
    boxes_contain_keys: bool = False
    #: Dynamic-obstacle family: when moving obstacles temporarily seal the route,
    #: turn on the spot (a free wait) instead of failing the level.
    stall_if_stuck: bool = False
    #: Re-derive the plan at EVERY step of the post-burst solvability check
    #: (`_burst_recovery_wins`) rather than following the plan in hand. Measured
    #: on keycorridors6r3, following the plan is 1.5x faster and keeps a THIRD as
    #: many bursts (28% vs 77%) -- the recorded trajectory loses a quarter of its
    #: frames, and recovery data is the entire point of bursting. So this stays on
    #: everywhere; it is a knob for trading recovery yield for speed, not a
    #: default worth changing.
    recovery_replan_every_step: bool = True

    levels_per_episode: int = 10
    supports_recovery = True
    #: More RESET recoveries than the base default: MiniGrid has two ways for an
    #: exploratory action to end the level outright -- the step budget running out
    #: and, in the dynamic-obstacle envs, walking into an obstacle OR a wall --
    #: and the episode-wide exploration prefix spends most of its budget inside
    #: the FIRST level, so three is not enough headroom there.
    max_resets = 6
    #: Cap on cleared-blocker bits; keeps the obstructed-maze state space finite.
    max_blocker_bits: int = 6
    #: Cap on remembered-unlocked-door bits. Each one doubles the state space, and
    #: past a handful of locked doors that is the whole cost of the search. Safe
    #: to cap: the bit only matters for LOOKAHEAD, since a door that has actually
    #: been opened simply reads as open when the world is re-read.
    max_lock_bits: int = 6
    #: How often the expert may plan from the SAME live state within one level
    #: before the state is declared a dither (see `_lookup`). Generous, because
    #: exploration and bursts legitimately walk back over old ground.
    loop_guard: int = 10
    #: Step cap when engine-verifying that a perturbation burst is recoverable.
    recovery_steps: int = 800

    #: Prefix `game_id` is derived with; the partial-observation base overrides it
    #: to ``minigrid_partial_``, matching ``MinigridAdapter``'s own naming.
    game_id_prefix: str = "minigrid_"

    def __init_subclass__(cls, **kw) -> None:
        """Derive ``game_id`` from ``env_id`` exactly as ``MinigridAdapter`` does,
        so the generator's output directory matches the adapter's game_id."""
        super().__init_subclass__(**kw)
        if not cls.__dict__.get("game_id") and cls.__dict__.get("env_id"):
            clean = (cls.env_id.replace("MiniGrid-", "")
                     .replace("-v0", "").replace("-v1", ""))
            cls.game_id = cls.game_id_prefix + clean.lower().replace("-", "_")

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self._cache: dict = {}          # objective -> _Model, all at `_cache_sig`
        # The world those models were built for. Every `_cache.clear()` in this
        # file and in the partial subclass means "force a rebuild", and an empty
        # dict does that on its own whatever this still says -- so those sites do
        # not have to reset it.
        self._cache_sig = None
        self._seen: dict = {}           # live state -> times planned from
        self._last = None               # (state key, plan, ties) of the last lookup

    # ── engine plumbing ──────────────────────────────────────────────────────
    def make_game(self, seed: int):
        # Episode `seed` owns maze block [seed*L, seed*L+L): non-overlapping.
        self._cache.clear()
        self._seen.clear()
        self._last = None
        return MinigridAdapter(self.env_id, seed=seed * self.levels_per_episode)

    def set_level(self, game, level_idx: int) -> None:
        super().set_level(game, level_idx)
        self._seen.clear()              # the loop guard is per level
        self._last = None

    def num_levels(self, game) -> int:
        return self.levels_per_episode

    def render(self, game) -> np.ndarray:
        return np.asarray(game._current_frame)

    def available_actions(self, game) -> list[int]:
        return list(self.action_set)

    def drive(self, game, action) -> DriveResult:
        game.perform_action(ActionInput(id=_ID_TO_GA[action.action_id]))
        frame = np.asarray(game._current_frame)
        solved = game._state == GameState.WIN
        dead = game._state == GameState.GAME_OVER      # step-budget truncation
        return DriveResult(frame, solved, dead)

    # ── the field, cached against the live grid ──────────────────────────────
    def _model(self, game, objective: str):
        """The exact field for the LIVE world under ``objective``.

        `world_sig` is a complete description of the board, so it is the cache
        key: any world change -- an opened door, a moved obstacle, a picked-up /
        dropped / revealed object -- misses the cache and rebuilds. Between
        changes every step is a dict lookup.

        The cache holds ONE world: the live one. Everything under an older
        signature is unreachable -- `world_sig` only ever moves forward -- so a
        change drops the lot and the survivors are exactly the fields for the
        board in front of us, at most one per objective. The previous form keyed
        on ``(sig, objective)`` in a flat dict and cleared it wholesale past six
        entries, which with four objectives is every second step: it threw away
        models built for the CURRENT world and rebuilt them immediately. That is
        most of the cost of the partial-observation family, whose objectives are
        four deep and whose fields are the most expensive in the file."""
        sig = self.world_sig(game)
        if sig != self._cache_sig:
            self._cache.clear()
            self._cache_sig = sig
        m = self._cache.get(objective)
        if m is None:
            # "<objective>!walk": the same objective, but the search may not pick
            # anything up on the way. That drops the carried-object and
            # cleared-blocker dimensions entirely, which is the difference
            # between a millisecond and a tenth of a second -- and it is enough
            # for the overwhelmingly common case of "just walk over there". The
            # full model is tried straight after whenever it isn't.
            walk = objective.endswith("!walk")
            m = _Model(self.world(game),
                       objective=objective[:-5] if walk else objective,
                       target=self.target, action_set=self.action_set,
                       boxes_contain_keys=self.boxes_contain_keys,
                       max_blocker_bits=0 if walk else self.max_blocker_bits,
                       max_lock_bits=self.max_lock_bits, allow_pickup=not walk)
            self._cache[objective] = m
        return m

    # ── what the planner is allowed to know (overridden for partial obs) ─────
    def world(self, game) -> World:
        """The board as the planner may see it. Full observation here; the
        partial-observation subclass returns its remembered map instead."""
        return World.from_env(game._env.unwrapped)

    def world_sig(self, game):
        """A hashable identity for `world` MINUS the agent pose. ``Grid.encode()``
        gives object type, colour and door state per cell, which is exactly the
        part of the board the field depends on."""
        u = game._env.unwrapped
        car = u.carrying
        return (u.grid.encode().tobytes(),
                None if car is None else (car.type, car.color))

    def pose(self, game):
        """``((x, y), dir, carry)`` -- where the agent is and what it holds."""
        u = game._env.unwrapped
        car = u.carrying
        return ((int(u.agent_pos[0]), int(u.agent_pos[1])), int(u.agent_dir),
                (car.type, car.color) if car is not None else None)

    def objectives(self) -> tuple:
        """The objectives to try, in order. The obstructed-maze family falls back
        to "go open a box" when the real objective needs a key it cannot see."""
        return ((self.objective, "open_box") if self.boxes_contain_keys
                else (self.objective,))

    def _lookup(self, game):
        """``(plan, ties)`` at the live state -- both read off the SAME field, so
        the plan's head is always a member of the tie set.

        LOOP GUARD: an exact expert never revisits a state, so a state seen many
        times over means the search is oscillating -- the residual failure mode of
        the vanishing-drop approximation (see `_Model.drop_ok`), where the model
        wants to pick an object up in one world and put it down again in the
        next. Report "no plan" once that happens: the recorder then RESETs and,
        if the level really is beyond the model, fails the seed immediately rather
        than burning the whole engine step budget dithering on one square."""
        pose = self.pose(game)
        key = (self.world_sig(game), pose)
        # `solve_from` and `optimal_set_from` are both asked at the same state
        # within one step; answer the second from here so the state counts as ONE
        # visit for the guard below (and the descent is done once, not twice).
        if self._last is not None and self._last[0] == key:
            return self._last[1], self._last[2]
        self._seen[key] = n = self._seen.get(key, 0) + 1

        def answer(plan, ties):
            self._last = (key, plan, ties)
            return plan, ties

        if n > self.loop_guard:
            return answer([], [])
        for objective in self.objectives():
            m = self._model(game, objective)
            s = m.state_at(pose)
            if s not in m.togo and not m.impossible:
                # The live pose fell outside the enumerated set (e.g. the
                # adapter's lava bounce teleported us). Rebuild rooted here.
                # Skipped when the model reports the objective IMPOSSIBLE in this
                # world: that verdict is a property of the board, not of the root,
                # so re-rooting would burn a full rebuild -- and a cache clear that
                # costs every other objective its field too -- to be told the same
                # thing. Blind, that is the common case on every step before the
                # goal has ever been seen.
                self._cache.clear()
                m = self._model(game, objective)
                s = m.state_at(pose)
            plan = m.plan_from(s)
            if plan:
                return answer(plan, m.ties_at(s))
            if m.solved(s):
                return answer([], [])
        if self.stall_if_stuck:
            return answer([A_LEFT], [A_LEFT, A_RIGHT])
        return answer([], [])

    # ── the solver API ───────────────────────────────────────────────────────
    def solve_from(self, game, level_idx: int, seed: int):
        """An ACTION-optimal plan from the agent's LIVE pose, re-derived from the
        live grid every call -- which is exactly what makes exploration prefixes
        and perturbation bursts recoverable."""
        return [_ID_TO_GA[a] for a in self._lookup(game)[0]]

    def optimal_set_from(self, game, level_idx: int, seed: int):
        """Every equally-optimal next action at the LIVE pose. MiniGrid mazes have
        genuine ties (turn/forward orders that reach the goal in the same number
        of actions), so this feeds the base's STOCHASTIC OPTIMAL sampling with
        real trajectory diversity."""
        ties = self._lookup(game)[1]
        return [_ID_TO_GA[a] for a in ties] if ties else None

    def _optimal_set(self, game, level_idx: int, seed: int, plan: list):
        """Force a re-plan every step.

        The base caches a plan and pops it action by action; here the world can
        change UNDER that plan (dynamic obstacles move, a dropped object lands on
        a cell the search modelled as free), so a stale plan must never be
        executed. Truncating ``plan`` in place to the live optimum makes the base
        take the freshly-computed action and re-`solve_from` on the next step --
        cheap, because the field itself is cached against the grid signature."""
        opts = super()._optimal_set(game, level_idx, seed, plan)
        if opts:
            plan[:] = opts[:1]
        return opts

    def _burst_recovery_wins(self, game, plan: list) -> bool:
        """Engine-verify that the post-burst state is really winnable, by playing
        it out. Restores the live state, so this leaves no trace.

        The verdict is SOUND however the rollout is driven: it says True only when
        the engine itself reported a WIN. A cheaper driver can therefore only make
        the check less complete -- some genuinely recoverable burst gets rolled
        back and its recovery data is lost -- never wrong. That is what buys the
        plan-following below.

        Re-deriving the plan at every step is needed only where the board moves
        under the plan (the dynamic-obstacle family, `recovery_replan_every_step`).
        Everywhere else, following the plan in hand and re-planning when it runs
        out reaches the same verdict for a small fraction of the fields -- and for
        the partial-observation family that is also the natural rhythm, because a
        plan that walks to the frontier ENDS at the frontier, which is exactly
        where the next observation makes re-planning worth doing."""
        # Bound the check by the plan it is checking, not by a flat constant: the
        # question is only "does this state still lead to a win", and the answer
        # arrives within a small multiple of the plan already in hand. Left
        # unbounded it dominates the whole run on the expensive games -- every
        # burst paying for hundreds of re-plans over a board that has to be
        # re-searched each step.
        budget = min(self.recovery_steps, 4 * len(plan) + 50)
        snap = self._game_snapshot(game)
        try:
            p: list = []
            for _ in range(budget):
                if not p or self.recovery_replan_every_step:
                    p = [_as_action(a) for a in self.solve_from(game, 0, 0)]
                    if not p:
                        return False
                res = self.drive(game, p.pop(0))
                if res.solved:
                    return True
                if res.dead:
                    return False
            return False
        except Exception:                  # noqa: BLE001 -- a crash is not a win
            return False
        finally:
            self._cache.clear()
            self._last = None
            self._game_restore(game, snap)

    # ── CLI: expose --levels-per-episode alongside the base flags ────────────
    @classmethod
    def build_argparser(cls):
        p = super().build_argparser()
        p.add_argument("--levels-per-episode", type=int,
                       default=cls.levels_per_episode,
                       help="Distinct mazes bundled per episode.")
        return p

    @classmethod
    def main(cls, argv=None) -> int:
        args = cls.build_argparser().parse_args(argv)
        solver = cls(rng=random.Random(args.seed),
                     burst_prob=args.noise, burst_mean=args.burst)
        solver.levels_per_episode = args.levels_per_episode
        out = args.out or (Path("data/training_multi_level") / cls.game_id)
        return solver.run(args.episodes, out, start_seed=args.start_seed,
                          max_attempts=args.max_attempts,
                          progress_every=args.progress_every,
                          explore=not args.no_exploration)
