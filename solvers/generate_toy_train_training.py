"""Generate Phase-1 training data for toy_train (Toy Train Routes).

toy_train (games/toy_train/toy_train.py) is a pickup-and-delivery puzzle on a rail
network: an oval loop with inward dead-end branches hanging off *switch* cells.
The train may only step between two cells that connect to each other, a switch
cell connects differently depending on its lever, and ACTION5 is overloaded --
it flips the lever on a switch, picks a waiting passenger up at their station,
or drops a carried passenger at their matching destination. The train carries at
most two passengers; the level is won the instant the last one is delivered.

toy_train action set (simple actions only):
    ACTION1 (north)  ACTION2 (south)  ACTION3 (west)  ACTION4 (east)
    ACTION5 (interact: flip lever / pick up / drop off)

ACTION7 (the game's built-in undo) is deliberately NOT used: undo is not a real
competition action, so it must never appear in the corpus -- neither in a plan nor
in an exploratory detour. `available_actions` therefore reports 1-5 only.

THE EXPERT IS A DISTANCE FIELD, NOT A SEARCH PER QUERY. The abstract state is

    (cell, lever bitmask, per-passenger status)

with status in {waiting, on train, delivered} per passenger -- at most
``ncell * 2^switches * 3^passengers`` ~ 1e6 states on the largest level -- and the
winning states are exactly those with every passenger delivered. So instead of
searching forward from wherever the train happens to stand, this solver runs ONE
reverse BFS from the whole win set and keeps the exact distance-to-win field,
cached per level geometry. Then:

  * ``solve_from`` is a greedy descent down the field -- optimal from ANY state,
    at ~zero cost, which is what makes ``supports_recovery`` cheap: an exploration
    prefix or a perturbation burst just lands on a different cell of the same
    precomputed field;
  * ``optimal_set_from`` is a 5-way lookup, so the recorded target is the FULL set
    of equally-optimal actions (and the base samples the taken action from it) with
    no extra solve. The tie set here is genuinely wide -- the oval can be circled
    either way, an idle lever flip is sometimes free, and independent
    pickup/delivery orders interleave -- so a single seed yields many distinct,
    still-perfectly-optimal trajectories.

The reverse BFS inverts the game's transition relation. Moves and lever flips are
their own inverses (the connection test is symmetric, and a flip is an involution),
so those edges are undirected; pickup and drop-off are the only one-way edges and
are walked backwards explicitly. It is vectorised over whole BFS frontiers with
numpy -- state codes are plain ints, so a frontier is an int array and each of the
~14 predecessor rules is a handful of array ops -- which keeps a full field at
well under a second per level.

Recovery: nothing in this game is irreversible except progress itself. A train
that wanders off, flips every lever and detours down a branch is always still able
to finish, so ``solve_from`` re-plans from the live state and a RESET is never
needed in practice -- exactly the "recover from anywhere" behaviour the corpus
wants. The one dead configuration the field represents exactly (a train stranded
on a branch whose lever is shut) is unreachable, because a lever can only be
flipped while standing on its switch.

Usage (run from the repo root):
    python solvers/generate_toy_train_training.py --episodes 1000 \
        --out data/training_multi_level/toy_train
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                              # noqa: E402

from games.toy_train.toy_train import ToyTrain, _DIR_TO_BIT      # noqa: E402
from solvers.base_solver import BaseSolver                       # noqa: E402

# Movement action ids, index-aligned with the game's own ACTION1..4 -> (dx, dy)
# table in ``ToyTrain._MOVE_DELTAS``.
_DELTAS = ((0, -1), (0, 1), (-1, 0), (1, 0))
_MOVE_IDS = (1, 2, 3, 4)
_INTERACT = 5
_ACTION_IDS = (1, 2, 3, 4, 5)

_TRAIN_CAPACITY = 2          # mirrors toy_train._TRAIN_CAPACITY


class _Field:
    """Exact distance-to-win field for one level geometry.

    A state is the int ``cell * M + levers * ST + status``, where ``ST = 3 **
    passengers`` packs each passenger's status as a base-3 digit (0 waiting, 1 on
    the train, 2 delivered) and ``M = 2 ** switches * ST``. The whole field is
    therefore one flat ``ncell * M`` array and a BFS frontier is just an int
    array, which is what lets the reverse BFS run vectorised.

    Distances are stored as ``d + 1`` so that 0 means "cannot win from here".
    """

    def __init__(self, data: dict) -> None:
        grid = data["grid"]
        w, h = data["width"], data["height"]

        # Only TRACK cells are states: the train can never stand on background.
        cells = [(x, y) for y in range(h) for x in range(w) if grid[y][x]]
        self.cells = cells
        self.cell_index = {c: i for i, c in enumerate(cells)}
        n = len(cells)

        sw_pos = data["switch_positions"]
        S = len(sw_pos)
        P = data["num_passengers"]
        self.n, self.S, self.P = n, S, P
        self.LV = 1 << S
        self.ST = 3 ** P
        self.M = self.LV * self.ST
        self.pow3 = [3 ** i for i in range(P)]

        # ``bits[ci, lv]`` = the cell's effective connection bitmask under lever
        # state ``lv``. Only switch cells vary with ``lv``; this mirrors
        # ``ToyTrain._active_bits``.
        bits = np.zeros((n, self.LV), dtype=np.int64)
        for ci, (x, y) in enumerate(cells):
            val = grid[y][x]
            if val < 100:
                bits[ci, :] = val
            else:
                si = val - 100
                b0, b1 = data["switch_bits"][si]
                lv = np.arange(self.LV)
                bits[ci, :] = np.where((lv >> si) & 1, b1, b0)

        # ``nbr[d, ci * LV + lv]`` = the cell one step in direction d, or -1 when
        # the two cells do not connect to each other under ``lv``. Both halves of
        # ``ToyTrain._can_move`` are folded in here, so the table is symmetric:
        # if ``nbr[d]`` takes a -> b then ``nbr[opposite(d)]`` takes b -> a. That
        # symmetry is what makes the movement edges undirected for the reverse BFS.
        nbr = np.full((4, n, self.LV), -1, dtype=np.int64)
        for d, (dx, dy) in enumerate(_DELTAS):
            bit = _DIR_TO_BIT[(dx, dy)]
            obit = _DIR_TO_BIT[(-dx, -dy)]
            for ci, (x, y) in enumerate(cells):
                t = self.cell_index.get((x + dx, y + dy))
                if t is None:
                    continue
                ok = ((bits[ci] & bit) != 0) & ((bits[t] & obit) != 0)
                nbr[d, ci, ok] = t
        self.nbr = nbr.reshape(4, n * self.LV)

        # Cell -> role lookups (a cell is at most one of these: the level
        # generator keeps switches, stations and destinations disjoint).
        self.sw_of_cell = {self.cell_index[p]: si for si, p in enumerate(sw_pos)}
        self.station_of_cell = {self.cell_index[p]: pi
                                for pi, p in enumerate(data["stations"])}
        self.dest_of_cell = {self.cell_index[p]: pi
                             for pi, p in enumerate(data["dests"])}
        self.sw_ci = [self.cell_index[p] for p in sw_pos]
        self.station_ci = [self.cell_index[p] for p in data["stations"]]
        self.dest_ci = [self.cell_index[p] for p in data["dests"]]

        # ``ntrain[st]`` = how many passengers the status code has on the train,
        # for the capacity test (both forward and in the reverse BFS).
        self.ntrain = np.array(
            [sum(1 for i in range(P) if (st // self.pow3[i]) % 3 == 1)
             for st in range(self.ST)], dtype=np.int8)

        self.dist = np.zeros(n * self.M, dtype=np.uint16)
        self._reverse_bfs()

    # ── encoding ────────────────────────────────────────────────────────────
    def code(self, ci: int, lv: int, st: int) -> int:
        return ci * self.M + lv * self.ST + st

    # ── the one reverse BFS ─────────────────────────────────────────────────
    def _reverse_bfs(self) -> None:
        """Fill ``dist`` with (distance-to-win + 1) by walking toy_train's
        transition relation BACKWARDS from its win set.

        The win set is every state whose status code is all-delivered (the level
        ends on that final drop-off, so no cell or lever state distinguishes
        them). A state ``s`` is a predecessor of ``s'`` iff one action takes s to
        s', which for each of the four rules is:

          * MOVE: ``s'``'s cell has a connecting neighbour ``t`` -- the train came
            from ``t``. Undirected, because the connection test is symmetric.
          * LEVER: ``s'`` stands on switch ``si`` and ``s`` is the same state with
            ``si`` flipped. An involution, so also undirected.
          * PICKUP (one-way): passenger ``pi`` is on the train in ``s'``, ``s'``
            stands on ``pi``'s station, and ``pi`` was waiting in ``s``.
          * DROP-OFF (one-way): ``pi`` is delivered in ``s'``, ``s'`` stands on
            ``pi``'s destination, and ``pi`` was on the train in ``s`` -- which
            must not overflow the train, or the engine would have refused the
            pickup that put it there.

        Every rule is applied to a whole frontier at once, so a BFS layer is a
        few numpy ops rather than a Python loop over states.
        """
        n, M, ST, LV = self.n, self.M, self.ST, self.LV
        dist, nbr, ntrain = self.dist, self.nbr, self.ntrain

        win = (np.arange(n, dtype=np.int64)[:, None] * M
               + np.arange(LV, dtype=np.int64)[None, :] * ST + (ST - 1)).ravel()
        dist[win] = 1
        frontier = win

        depth = 1
        while frontier.size:
            depth += 1
            if depth > np.iinfo(np.uint16).max:      # unreachable in practice
                return
            ci = frontier // M
            rem = frontier % M
            lv = rem // ST
            st = rem % ST
            key = ci * LV + lv
            preds: list[np.ndarray] = []

            for d in range(4):                       # MOVE (undirected)
                t = nbr[d][key]
                ok = t >= 0
                if ok.any():
                    preds.append(t[ok] * M + rem[ok])

            for si in range(self.S):                 # LEVER (involution)
                m = ci == self.sw_ci[si]
                if m.any():
                    preds.append(ci[m] * M + (lv[m] ^ (1 << si)) * ST + st[m])

            for pi in range(self.P):
                p3 = self.pow3[pi]
                dig = (st // p3) % 3
                # PICKUP backwards: on-train -> waiting (code drops one p3).
                m = (dig == 1) & (ci == self.station_ci[pi])
                if m.any():
                    preds.append(frontier[m] - p3)
                # DROP-OFF backwards: delivered -> on-train. Skip predecessors
                # that would exceed capacity -- they are states the engine can
                # never produce, and admitting them would leak bogus shortcuts
                # into the field.
                m = ((dig == 2) & (ci == self.dest_ci[pi])
                     & (ntrain[st] < _TRAIN_CAPACITY))
                if m.any():
                    preds.append(frontier[m] - p3)

            if not preds:
                return
            cand = np.unique(np.concatenate(preds))
            cand = cand[dist[cand] == 0]
            if cand.size == 0:
                return
            dist[cand] = depth
            frontier = cand

    # ── queries ─────────────────────────────────────────────────────────────
    def steps_to_win(self, ci: int, lv: int, st: int) -> int | None:
        """Optimal number of actions to win from this state, or None if the state
        cannot win at all."""
        d = int(self.dist[self.code(ci, lv, st)])
        return None if d == 0 else d - 1

    def successor(self, ci: int, lv: int, st: int, aid: int):
        """toy_train's forward transition for action ``aid``, or None when the
        action changes nothing (a disconnected direction, an interact with
        nothing to interact with). Mirrors ``ToyTrain.step`` exactly, INCLUDING
        the order in which ACTION5's three cases are tested."""
        if aid != _INTERACT:
            t = int(self.nbr[aid - 1][ci * self.LV + lv])
            return None if t < 0 else (t, lv, st)

        si = self.sw_of_cell.get(ci)
        if si is not None:                           # on a switch: flip the lever
            return (ci, lv ^ (1 << si), st)

        pi = self.station_of_cell.get(ci)
        if pi is not None:                           # at a station: pick up
            p3 = self.pow3[pi]
            if (st // p3) % 3 == 0 and self.ntrain[st] < _TRAIN_CAPACITY:
                return (ci, lv, st + p3)

        pi = self.dest_of_cell.get(ci)
        if pi is not None:                           # at a destination: drop off
            p3 = self.pow3[pi]
            if (st // p3) % 3 == 1:
                return (ci, lv, st + p3)
        return None

    def optimal_actions(self, ci: int, lv: int, st: int) -> list[int]:
        """Every action that strictly descends the field -- the equally-optimal
        moves -- in ACTION1..5 order."""
        here = self.steps_to_win(ci, lv, st)
        if not here:                                 # dead, or already won
            return []
        out = []
        for aid in _ACTION_IDS:
            nxt = self.successor(ci, lv, st, aid)
            if nxt is None:
                continue
            there = self.steps_to_win(*nxt)
            if there is not None and there == here - 1:
                out.append(aid)
        return out


class ToyTrainSolver(BaseSolver):
    game_id = "toy_train"
    # ``solve_from`` is a lookup in an exact distance-to-win field over the LIVE
    # (cell, levers, passenger statuses) state, so it re-plans optimally from any
    # board an exploratory detour or a perturbation burst leaves behind. The only
    # one-way transitions are pickup and delivery, and neither can strand the
    # train (destinations sit on the always-connected oval), so recovery here is
    # pure replanning -- no RESET needed.
    supports_recovery = True
    recovery_mode = "replan"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # One-entry cache: the field depends only on the level GEOMETRY (track,
        # switches, stations, destinations), never on where the train currently
        # is or which levers are thrown, so a single entry serves a whole level --
        # every replan, burst and RESET inside it.
        self._cache_key = None
        self._cache_field: _Field | None = None

    def make_game(self, seed: int):
        # The seed fixes every level's track layout, switch placement, branch
        # lengths, passenger colours and station/destination cells, so a recorded
        # episode replays byte-for-byte.
        return ToyTrain(seed=seed)

    def available_actions(self, game) -> list[int]:
        # The game also offers ACTION7 (undo), but undo is not a real competition
        # action, so it must never reach the corpus -- not even as an exploratory
        # detour. Movement + interact only.
        return list(_ACTION_IDS)

    # ── burst snapshot: the game's own tuple, not a whole-game deepcopy ──────
    # The base captures a burst-rollback point with ``copy.deepcopy(game)``, which
    # here clones all SEVEN levels' sprite trees and level dictionaries -- profiling
    # one episode at the default noise put 23 s of 34 s inside deepcopy, for state
    # that is a handful of ints. ToyTrain already exposes a matched
    # ``_snapshot``/``_restore`` pair (the ACTION7 undo stack) holding exactly the
    # level's mutable state: train cell, levers, and each passenger's
    # waiting/aboard/delivered status -- and ``_restore`` re-derives every sprite
    # from it. So snapshot that plus the engine bookkeeping a burst (or the base's
    # verification replay) can disturb:
    #
    #   * ``steps_remaining`` -- rendered as the HUD bar on row 63, so failing to
    #     restore it would leave the post-rollback frame differing from the anchor
    #     frame the buffer was truncated back to;
    #   * ``_next_level`` / ``_state`` / ``_score`` -- ``_burst_recovery_wins``
    #     drives a WINNING plan on the live state before restoring, and
    #     ``next_level`` sets those. ``_current_level_index`` is deliberately NOT
    #     restored: the engine only advances it in ``_really_set_next_level``, which
    #     ``drive`` never reaches (it breaks on the ``_next_level`` flag), so it
    #     cannot move during a burst;
    #   * ``_history`` -- inert (ACTION7 is never emitted) but kept exact anyway.
    #     A shallow list copy suffices: ``_restore`` rebuilds the containers inside
    #     each entry rather than mutating them.
    def _game_snapshot(self, game):
        return (game._snapshot(), list(game._history),
                game._step_counter.steps_remaining, game._state,
                game._next_level, game._score)

    def _game_restore(self, game, snap) -> None:
        core, history, steps, state, next_level, score = snap
        game._restore(core)
        game._history = list(history)
        game._step_counter.steps_remaining = steps
        game._state = state
        game._next_level = next_level
        game._score = score

    # ── live state / field ──────────────────────────────────────────────────
    def _field(self, game, level_idx: int) -> _Field:
        data = game._level_data[level_idx]
        key = (level_idx, data["width"], data["height"],
               tuple(data["switch_positions"]), tuple(data["stations"]),
               tuple(data["dests"]))
        if key != self._cache_key:
            self._cache_field = _Field(data)
            self._cache_key = key
        return self._cache_field

    @staticmethod
    def _live_state(game, field: _Field) -> tuple[int, int, int]:
        """The LIVE (cell, levers, status) triple read off the running game."""
        ci = field.cell_index[game._grid_pos()]
        lv = 0
        for si, bit in enumerate(game._lever_states):
            if bit:
                lv |= 1 << si
        st = 0
        for pi in range(field.P):
            if pi in game._pass_delivered:
                st += 2 * field.pow3[pi]
            elif pi in game._pass_on_train:
                st += field.pow3[pi]
        return ci, lv, st

    # ── the solver API ──────────────────────────────────────────────────────
    def solve_from(self, game, level_idx: int, seed: int):
        """An optimal plan from the game's LIVE state: greedy descent down the
        distance field, breaking ties at random so repeated episodes of the same
        seed take different (equally optimal) routes. ``[]`` when the live state
        cannot win, which the field represents exactly."""
        field = self._field(game, level_idx)
        ci, lv, st = self._live_state(game, field)

        remaining = field.steps_to_win(ci, lv, st)
        if not remaining:
            return []
        plan: list[int] = []
        while remaining:
            acts = field.optimal_actions(ci, lv, st)
            if not acts:                             # unreachable: field is exact
                return []
            aid = self.rng.choice(acts)
            plan.append(aid)
            ci, lv, st = field.successor(ci, lv, st, aid)
            remaining -= 1
        return plan

    def optimal_set_from(self, game, level_idx: int, seed: int):
        """Every equally-optimal next action at the LIVE state -- the actions that
        strictly descend the distance field. A 5-way lookup, so the full tie set
        is recorded as the training target at no extra cost."""
        field = self._field(game, level_idx)
        acts = field.optimal_actions(*self._live_state(game, field))
        return acts or None


if __name__ == "__main__":
    raise SystemExit(ToyTrainSolver.main())
