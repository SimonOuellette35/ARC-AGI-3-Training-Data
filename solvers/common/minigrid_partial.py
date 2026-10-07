"""Shared BaseSolver base for the PARTIALLY-OBSERVED MiniGrid generators.

Same 44 games as ``common/minigrid.py``, but driven through
``MinigridAdapter(..., partial_obs=True)``: the frame is the agent's 7x7
egocentric cone, not the board. So the expert cannot be handed a plan -- it has
to explore, remember, and re-plan as the map fills in.

WHAT THIS SOLVER IS ALLOWED TO READ (see [[no-privileged-solvers]])
------------------------------------------------------------------
Exactly one thing: ``env.gen_obs()["image"]`` -- the 7x7x3 observation the
adapter itself renders into the frame, with everything out of view or behind a
wall encoded as ``unseen``. It never touches ``env.grid``, ``env.agent_pos``,
``env.agent_dir`` or ``env.carrying``. Concretely that means:

* **the map is memory.** Every observation is transformed from view coordinates
  into the agent's own frame and merged into `_known`; the planner then runs on
  that remembered map, with everything never seen listed as ``unknown`` and
  therefore impassable.
* **position is dead reckoning.** The agent's frame is anchored at wherever the
  level started -- (0, 0) facing 0 -- and the pose is advanced from the action
  taken plus what the map says is in front. Absolute board coordinates never
  exist anywhere in this file.
* **what it carries is observed too.** MiniGrid writes the carried object into
  the agent's own cell of the view, so `_ingest` reads it from there rather than
  from ``env.carrying``.

Dead reckoning is exact for every env here except the dynamic-obstacle family,
where an obstacle can move into the cell being entered AFTER it was observed to
be clear, and the step silently fails. So a forward move keeps BOTH hypotheses --
moved and didn't -- and `_agree` scores the next observation against the
remembered map under each, keeping whichever explains what is now on screen.
Walls, lava and goal tiles never move, so they are the evidence; where the view
shows none of them the two hypotheses tie and the prediction stands, which is
exactly the case (open floor) where nothing can block a step anyway.

EXPLORING
---------
`objectives` adds one fallback to the full-observation chain: when the real
objective is not reachable through what has been SEEN, the expert switches to
``frontier`` -- get to a pose facing a cell never yet observed. That edge is free
(`_Model._succ`), because arriving reveals the whole 7x7 cone in that direction,
so the field naturally picks the cheapest look-somewhere-new including the turns
it takes to face that way. As soon as the goal, the key or the door it needs
comes into view, the real objective becomes reachable and the expert commits to
it. The recorded target is therefore always the best action GIVEN WHAT IS KNOWN,
which is the only thing an optimal target can mean under partial observation.

A RESET inside a level (the recorder's recovery from a death or a bricked state)
restores the board but not the agent's memory of it, so `set_level` keeps the
immovable terrain -- walls, lava, goal -- and forgets doors and objects, whose
state the reset has just rolled back.

Subclass: set ``env_id`` and the same ``objective`` / ``target`` /
``action_set`` configuration as the full-observation twin.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np                                          # noqa: E402
from minigrid.core.constants import (                       # noqa: E402
    IDX_TO_COLOR, IDX_TO_OBJECT, OBJECT_TO_IDX)
from adapters import MinigridAdapter                        # noqa: E402
from solvers.common.minigrid import (                       # noqa: E402
    A_FORWARD, A_INTERACT, A_LEFT, A_RIGHT, DIR_VEC, MinigridSolver, World,
    _PICKABLE)

_UNSEEN = OBJECT_TO_IDX["unseen"]
_EMPTY = ("empty", None, None)
#: Tiles that cannot change, so a disagreement between two of them is decisive
#: evidence in `_agree`. ("empty" belongs here: floor never becomes wall.)
_TERRAIN = frozenset({"wall", "lava", "goal", "empty"})
#: The immovable subset -- what survives a RESET inside a level.
_STABLE = frozenset({"wall", "lava", "goal"})
#: Door state channel: 0 open, 1 closed, 2 locked.
_DOOR_STATE = ("open", "closed", "locked")
_ORTH = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _decode_cell(o: int, c: int, s: int):
    """One observation cell -> the map's ``(kind, colour, state)``."""
    name = IDX_TO_OBJECT[o]
    if name in ("empty", "floor", "agent"):
        return _EMPTY
    if name == "door":
        return ("door", IDX_TO_COLOR[c], _DOOR_STATE[s] if s < 3 else "closed")
    if name in _PICKABLE:
        return (name, IDX_TO_COLOR[c], None)
    return (name, None, None)                    # wall / lava / goal


class MinigridPartialSolver(MinigridSolver):
    """Partially-observed twin of `MinigridSolver`: same search, but over a map
    the agent has to build for itself out of 7x7 glimpses."""

    game_id_prefix = "minigrid_partial_"

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self._known: dict = {}      # cell -> (kind, colour, state), agent frame
        self._pos = (0, 0)          # dead-reckoned, anchored at the level start
        self._dir = 0
        self._carry = None
        self._version = 0           # bumped whenever `_known` changes
        self._level = None
        self._prev_view = None      # last observation, for the shift test

    # ── engine plumbing ──────────────────────────────────────────────────────
    def make_game(self, seed: int):
        self._cache.clear()
        self._seen.clear()
        self._last = None
        self._level = None
        return MinigridAdapter(self.env_id, partial_obs=True,
                               seed=seed * self.levels_per_episode)

    def set_level(self, game, level_idx: int) -> None:
        super().set_level(game, level_idx)
        if self._level == level_idx:
            # A RESET inside the level: the board rolled back, so what we knew
            # about doors and objects is stale -- but walls, lava and the goal
            # are where they were, and re-exploring them would burn the step
            # budget we just recovered.
            self._known = {p: c for p, c in self._known.items()
                           if c[0] in _STABLE or c[0] == "empty"}
        else:
            self._known = {}
        self._level = level_idx
        self._pos, self._dir, self._carry = (0, 0), 0, None
        self._version += 1
        self._cache.clear()
        self._last = None
        self._prev_view = None
        view = self._view(game)
        self._ingest(view)
        self._prev_view = view

    def drive(self, game, action):
        res = super().drive(game, action)
        self._advance(game, action.action_id)
        return res

    # ── observation -> memory ────────────────────────────────────────────────
    @staticmethod
    def _view(game) -> np.ndarray:
        """THE OBSERVATION: the 7x7x3 egocentric image, exactly what the adapter
        renders into the frame the agent is shown. This is the ONLY thing this
        solver ever reads off the environment."""
        return game._env.unwrapped.gen_obs()["image"]

    @staticmethod
    def _cells(view: np.ndarray, pos, d):
        """Every VISIBLE observation cell, mapped from view coordinates into the
        agent's own frame.

        In the observation the agent stands at ``(size // 2, size - 1)`` facing
        "up", so a cell at view ``(i, j)`` is ``(size - 1) - j`` ahead of the
        agent and ``i - size // 2`` to its right -- which is all the transform
        needs. The agent's own cell is skipped: MiniGrid overwrites it with
        whatever is being CARRIED, so it says nothing about the floor."""
        size = view.shape[0]
        half = size // 2
        fx, fy = DIR_VEC[d]
        rx, ry = DIR_VEC[(d + 1) % 4]
        for i in range(size):
            for j in range(size):
                o = int(view[i, j, 0])
                if o == _UNSEEN or (i == half and j == size - 1):
                    continue
                fwd, rgt = (size - 1) - j, i - half
                yield ((pos[0] + fx * fwd + rx * rgt,
                        pos[1] + fy * fwd + ry * rgt),
                       _decode_cell(o, int(view[i, j, 1]), int(view[i, j, 2])))

    @staticmethod
    def _view_front(view: np.ndarray):
        """The raw observation cell one step ahead of the agent -- always visible,
        whatever the pose turns out to have been."""
        size = view.shape[0]
        return view[size // 2, size - 2]

    def _ingest(self, view: np.ndarray) -> None:
        """Merge one observation into the map at the current believed pose."""
        size = view.shape[0]
        own = view[size // 2, size - 1]
        name = IDX_TO_OBJECT[int(own[0])]
        self._carry = ((name, IDX_TO_COLOR[int(own[1])])
                       if name in _PICKABLE else None)
        changed = False
        for p, cell in self._cells(view, self._pos, self._dir):
            if self._known.get(p) != cell:
                self._known[p] = cell
                changed = True
        if self._known.get(self._pos) != _EMPTY:
            self._known[self._pos] = _EMPTY      # we are standing on it
            changed = True
        if changed:
            self._version += 1

    def _agree(self, pose, view: np.ndarray) -> int:
        """Evidence for the agent being at ``pose``, as ``-(contradictions)``.

        Only CONTRADICTIONS count, never agreements: a hypothesis that leaves the
        agent where it already was overlaps more of the explored map than one that
        moves it, so counting matches systematically votes "you didn't move".
        Terrain, on the other hand, never changes -- so a remembered wall that the
        view now says is floor (or vice versa) rules that hypothesis out, and the
        true one scores a clean zero. Objects are ignored: they move, and a ball
        arriving on a cell last seen empty is not evidence of anything."""
        conflicts = 0
        for p, cell in self._cells(view, *pose):
            old = self._known.get(p)
            if old is None or old[0] == cell[0]:
                continue
            if old[0] in _TERRAIN and cell[0] in _TERRAIN:
                conflicts += 1
        return -conflicts

    def _shift_conflicts(self, new: np.ndarray, moved: bool) -> int:
        """Conflicts between the PREVIOUS observation and this one, under the
        hypothesis that the forward step did / did not happen.

        This is the decisive test, and it needs no map at all. A forward step
        does not turn the agent, so the two views differ by a pure translation:
        stand still and the new view matches the old cell for cell; step forward
        and everything slides exactly one row nearer, ``new[i][j] ==
        prev[i][j-1]``. A wall three squares off in one reading is four squares
        off in the other, so the alignment picks itself. Terrain disagreements
        are decisive (walls do not move); object ones are only a hint, because
        the dynamic-obstacle balls really do move every step."""
        prev = self._prev_view
        if prev is None:
            return 0
        size = new.shape[0]
        off = 1 if moved else 0
        hard = soft = 0  # noqa: F841 -- see the note at the return
        for i in range(size):
            for j in range(off, size):
                if i == size // 2 and (j == size - 1 or j - off == size - 1):
                    continue                     # the agent cell holds its cargo
                a, b = prev[i, j - off], new[i, j]
                if int(a[0]) == _UNSEEN or int(b[0]) == _UNSEEN:
                    continue
                ka = _decode_cell(int(a[0]), int(a[1]), int(a[2]))[0]
                kb = _decode_cell(int(b[0]), int(b[1]), int(b[2]))[0]
                if ka == kb:
                    continue
                if ka in _TERRAIN and kb in _TERRAIN:
                    hard += 1
        # Object disagreements are deliberately NOT counted: measured against
        # ground truth they only ever added noise (every dynamic obstacle moves
        # every step), while terrain alone tracks the pose exactly.
        del soft
        return hard

    def _advance(self, game, action_id: int) -> None:
        """Update the believed pose after ``action_id`` was performed, then fold
        the new observation in. Candidates are ordered most-likely first, so a
        tie in the evidence keeps the prediction."""
        view = self._view(game)
        d = self._dir
        ambiguous = False
        if action_id == A_LEFT:
            cands = [(self._pos, (d - 1) % 4)]
        elif action_id == A_RIGHT:
            cands = [(self._pos, (d + 1) % 4)]
        elif action_id == A_FORWARD:
            fx, fy = DIR_VEC[d]
            ahead = (self._pos[0] + fx, self._pos[1] + fy)
            cell = self._known.get(ahead)
            kind = cell[0] if cell else None
            if kind == "lava":
                # The adapter does not kill on lava; it teleports us back to the
                # level start (and docks 5 steps). That is where we anchored the
                # frame, so we know exactly where we land.
                cands = [((0, 0), 0)]
            elif kind in ("empty", "goal") or (kind == "door"
                                               and cell[2] == "open"):
                # We aimed at a cell we had just seen was clear, so normally we
                # moved -- but in the dynamic-obstacle envs a ball can slide into
                # it after we looked and silently eat the step. Decide from the
                # evidence; the prior is that we moved.
                cands = [(ahead, d), (self._pos, d)]
                ambiguous = True
                # Tie-break prior for a big open room, where the view can show no
                # terrain at all and the shift test has nothing to go on. The
                # fresh front cell is one step ahead of wherever we really are:
                # an object there that we did NOT already know was two ahead is
                # almost certainly the thing that just slid in and ate the step.
                front = _decode_cell(*(int(v) for v in self._view_front(view)))
                beyond = self._known.get((ahead[0] + fx, ahead[1] + fy))
                if front[0] in _PICKABLE and (beyond is None
                                              or beyond[0] != front[0]):
                    cands.reverse()
            else:
                cands = [(self._pos, d)]         # wall / shut door / object
        else:
            cands = [(self._pos, d)]             # drop and interact never move us

        if ambiguous:
            best = min(
                range(len(cands)),
                key=lambda i: (self._shift_conflicts(view, cands[i][0] != self._pos)
                               - self._agree(cands[i], view), i))
            cands = [cands[best]]
        self._pos, self._dir = cands[0]
        self._ingest(view)
        self._prev_view = view

    # ── what the planner is allowed to know ──────────────────────────────────
    def world(self, game) -> World:
        """The remembered map as a `World`. Everything never seen goes into
        ``unknown`` -- impassable for the real objectives, and the target of the
        ``frontier`` one. Only cells adjacent to somewhere we could stand are
        listed, because those are the only ones a frontier edge can point at."""
        open_, goals, doors, objs = set(), set(), {}, {}
        for p, (kind, color, state) in self._known.items():
            if kind == "empty":
                open_.add(p)
            elif kind == "goal":
                goals.add(p)
            elif kind == "door":
                doors[p] = (color, state)
            elif kind in _PICKABLE:
                objs[p] = (kind, color)
            # wall / lava: listed nowhere, so impassable
        walkable = open_ | goals | set(doors) | set(objs)
        known = self._known
        unknown = {(p[0] + dx, p[1] + dy) for p in walkable for dx, dy in _ORTH
                   if (p[0] + dx, p[1] + dy) not in known}
        return World(open=open_, goals=goals, doors=doors, objs=objs,
                     carry=self._carry, pos=self._pos, dir=self._dir,
                     unknown=frozenset(unknown))

    def world_sig(self, game):
        return (self._version, self._carry)

    def pose(self, game):
        return (self._pos, self._dir, self._carry)

    def objectives(self) -> tuple:
        """The real objective first, then -- when it is not reachable through what
        has been seen -- go and see more.

        The exploration objectives are cheap walk-only models (see `_model`).
        Rebuilding the field every step is the price of a map that grows every
        step, and paying for the carried-object and blocker dimensions on all of
        those rebuilds is most of the run time for nothing. Measured on
        obstructed-maze, the walk-only form of "go open a box" is 8x faster than
        the full one AND wins slightly more levels -- a box you cannot simply walk
        to is better left until more of the maze is known. Exploration does still
        need the full model as a last resort, because getting further sometimes
        means unlocking a door, which means fetching its key first. The REAL
        objective is never approximated this way: its plan is the recorded
        target."""
        base = super().objectives()
        return (base[:1] + tuple(o + "!walk" for o in base[1:])
                + ("frontier!walk", "frontier"))

    # ── burst rollback has to rewind the BELIEF too ──────────────────────────
    def _game_snapshot(self, game):
        return ("partial", super()._game_snapshot(game),
                (dict(self._known), self._pos, self._dir, self._carry,
                 self._version, self._prev_view))

    def _game_restore(self, game, snap) -> None:
        _kind, inner, belief = snap
        super()._game_restore(game, inner)
        (known, self._pos, self._dir, self._carry, self._version,
         self._prev_view) = belief
        self._known = dict(known)
        self._cache.clear()
        self._last = None
