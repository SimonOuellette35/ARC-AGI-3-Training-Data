"""Exact transition model + distance-to-win field for the ul01 "key & door" walk.

ONE statement of the mechanic, used by both sides of the game (the same split as
``utils/tug_puzzle.py`` does for tk01):

  * ``games/ul01/ul01.py`` builds a field while SAMPLING a level, so a per-seed
    layout is accepted only if its optimal solution has the length the difficulty
    ramp asks for -- difficulty by search, exactly, rather than by a Manhattan
    estimate that the door can invalidate;
  * ``solvers/generate_ul01_training.py`` builds the same field as its EXPERT:
    ``solve_from`` is a greedy descent down it and ``optimal_set_from`` a 4-way
    lookup.

Keeping the two in one place is what stops them drifting, and the module is
verified against the real engine by an exhaustive differential test (every
``(player, has_key)`` state x every action, per level).

THE MECHANIC (game space, before the display rotation)::

    ACTION1/2/3/4   walk up/down/left/right one cell.

    * walking off the board does nothing (the action is still consumed);
    * walking onto the KEY collects it (the key sprite disappears and the
      has-key indicator lights up) and the player ends up on that cell;
    * walking onto the DOOR is BLOCKED without the key, and WINS the level with
      it (door and key are both gone, so the engine advances the level).

So the only irreversible event is picking the key up, and it strictly helps: from
EVERY reachable state the level is still winnable (a single blocked cell cannot
disconnect a rectangular board). That is what makes ul01 fully recoverable -- an
exploratory detour, a perturbation burst, even one that collects the key early, is
just a different start state for the same field.

STATE + ENCODING. A state is ``(player_cell, has_key)`` -- at most 2 * 64 = 128
states on an 8x8 board -- encoded as ``has_key * n + player_cell``. At that size a
plain-Python reverse BFS costs ~50 us, well under the numpy setup cost the bigger
fields in this directory need, so the transition relation is written once, in
scalar form, and the field is built from it. There is exactly one implementation
of the mechanic here, not a fast one and a readable one that can disagree.
"""

from __future__ import annotations

# Game-space deltas for ACTION1..4, index-aligned with the action indices used
# throughout this module (and with ul01's own ACTION1..4 -> (dx, dy) mapping).
DELTAS = ((0, -1), (0, 1), (-1, 0), (1, 0))
NUM_ACTIONS = 4


class KeyDoorField:
    """Every ``(player, has_key)`` state's exact distance to the win.

    ``key`` and ``door`` are ``(x, y)`` grid cells; the field depends ONLY on that
    geometry (never on where the player currently is), so one field answers every
    query for a level -- each replan, burst and RESET inside it.

    ``key=None`` means the key has already been collected: the ``has_key=False``
    half of the field is then unreachable (its states get distance ``-1``) and the
    ``has_key=True`` half -- the only half that can be live -- is unaffected, since
    the key cell is a plain empty cell once the sprite is gone. So a solver that
    first sees the board AFTER an exploratory detour picked the key up does not
    need to know where the key used to be.
    """

    def __init__(self, gw: int, gh: int, key, door) -> None:
        self.gw, self.gh = gw, gh
        self.n = n = gw * gh
        self.key = None if key is None else key[1] * gw + key[0]
        self.door = door[1] * gw + door[0]

        # ``nbr[d][c]`` = the cell one step in direction d from c, or -1 off-board.
        nbr = [[-1] * n for _ in range(NUM_ACTIONS)]
        for d, (dx, dy) in enumerate(DELTAS):
            for y in range(gh):
                for x in range(gw):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < gw and 0 <= ny < gh:
                        nbr[d][y * gw + x] = ny * gw + nx
        self.nbr = nbr
        self.dist = self._reverse_bfs()

    # ── the transition relation (the single statement of the mechanic) ────────
    def step(self, p: int, has_key: bool, a: int) -> tuple[int, bool]:
        """The successor of walking ``a`` from ``(p, has_key)``.

        A blocked walk (off the board, or into the door without the key) returns
        the state unchanged -- the engine consumes the action and moves nothing.
        Stepping onto the door WITH the key is the winning move; it is reported
        here as "the player stands on the door cell", which `wins` tests for.
        """
        t = self.nbr[a][p]
        if t < 0:                                   # off the board: nothing moves
            return p, has_key
        if t == self.door and not has_key:          # locked door blocks the player
            return p, has_key
        if t == self.key and not has_key:           # collect it, and step on
            return t, True
        return t, has_key

    def wins(self, p: int, has_key: bool, a: int) -> bool:
        """Does walking ``a`` end the level? Only stepping onto the door with the
        key does: that removes the door, and the key is already gone, so the
        engine's "no door and no key left" test fires."""
        return has_key and self.nbr[a][p] == self.door

    # ── the one reverse BFS ─────────────────────────────────────────────────
    def _code(self, p: int, has_key: bool) -> int:
        return (self.n if has_key else 0) + p

    def _reverse_bfs(self) -> list[int]:
        """``dist[s]`` = walks to the win from ``s`` (-1 = cannot win from here).

        Backward from the win: layer 1 is every state a single walk wins from, and
        layer k+1 is every not-yet-labelled state with SOME action into layer k.
        The predecessor relation is not needed explicitly -- at 128 states the
        whole state space is rescanned per layer, which is cheaper than building
        one.
        """
        n = self.n
        dist = [-1] * (2 * n)
        # The door cell is never a live player position: reaching it IS the win, so
        # it is left out of the state space rather than treated as a state the game
        # continues from.
        states = [(p, hk) for hk in (False, True) for p in range(n)
                  if p != self.door]
        frontier = []
        for p, hk in states:
            if any(self.wins(p, hk, a) for a in range(NUM_ACTIONS)):
                dist[self._code(p, hk)] = 1
                frontier.append((p, hk))
        depth = 1
        while frontier:
            depth += 1
            fresh = []
            for p, hk in states:
                if dist[self._code(p, hk)] >= 0:
                    continue
                for a in range(NUM_ACTIONS):
                    nxt = self.step(p, hk, a)
                    if nxt != (p, hk) and dist[self._code(*nxt)] == depth - 1:
                        dist[self._code(p, hk)] = depth
                        fresh.append((p, hk))
                        break
            frontier = fresh
        return dist

    # ── scalar queries (all read the arrays above) ──────────────────────────
    def cell(self, x: int, y: int) -> int:
        return y * self.gw + x

    def xy(self, cell: int) -> tuple[int, int]:
        return cell % self.gw, cell // self.gw

    def steps_to_win(self, p: int, has_key: bool) -> int | None:
        d = self.dist[self._code(p, has_key)]
        return None if d < 0 else d

    def optimal_actions(self, p: int, has_key: bool) -> list[int]:
        """Every equally-optimal action at ``(p, has_key)``, in ACTION1..4 order.

        At distance 1 that is exactly the winning walks; deeper, it is every walk
        that strictly descends the field. A walk that changes nothing (into an
        edge, or into the locked door) never descends, so it is never offered."""
        here = self.steps_to_win(p, has_key)
        if not here:                                # unreachable, or already won
            return []
        if here == 1:
            return [a for a in range(NUM_ACTIONS) if self.wins(p, has_key, a)]
        out = []
        for a in range(NUM_ACTIONS):
            nxt = self.step(p, has_key, a)
            if nxt == (p, has_key):
                continue
            there = self.steps_to_win(*nxt)
            if there is not None and there == here - 1:
                out.append(a)
        return out

    # ── level sampling (used by the game's per-seed layout generator) ────────
    def start_cells(self, lo: int, hi: int) -> list[int]:
        """Every legal player START cell whose optimal solution is ``lo..hi`` walks.

        "Legal" means the player starts without the key on a cell that is neither
        the key's nor the door's, so nothing is hidden under anything else at t=0
        and the level does not start already solved.
        """
        out = []
        for p in range(self.n):
            if p == self.door or p == self.key:
                continue
            d = self.dist[self._code(p, False)]
            if lo <= d <= hi:
                out.append(p)
        return out
