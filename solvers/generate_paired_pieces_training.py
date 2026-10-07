"""Generate Phase-1 training data for the ``paired_pieces`` game.

paired_pieces (games/paired_pieces/paired_pieces.py) is a multi-piece
rearrangement puzzle: ACTION1..4 move the ACTIVE piece one cell, ACTION5 cycles
the active piece forward (``active = (active + 1) % n``), and some pieces are
PAIRED -- moving one moves its partner by the SAME offset. A move is refused
outright (a no-op) if the active piece OR its partner would enter a wall or leave
the board. The level is won the moment every piece sits on ITS OWN goal.

PARTIALLY OBSERVABLE -- and solved as such
------------------------------------------
Two facts the win condition depends on are NOT visible in the frame:

* **which goal belongs to which piece** -- every goal renders in one colour and
  shape, and every piece in another, so the identity assignment the engine checks
  (``piece i on goals[i]``) is hidden;
* **which pieces are paired** -- pairing has no visual marker at all.

This solver therefore NEVER reads either of them. It reads only what a viewer of
the 64x64 frame can see -- wall/floor cells, the SET of goal cells, each piece's
cell and its current colour, and which piece is highlighted -- and recovers the
hidden part by trial and error, exactly as an agent must. Three observable
channels do the work:

1. **The on-goal colour is a per-piece oracle bit.** A piece standing on its own
   goal renders in the ``on_goal`` colour; on somebody else's goal it stays
   ``inactive``-coloured. The ACTIVE piece is always drawn highlighted, so the
   bit is readable exactly while the piece is NOT selected -- i.e. the ACTION5
   press you were going to make anyway reveals the answer for the piece you just
   parked. Which colour means "inactive" is itself learned, not assumed: any
   non-active piece standing OFF a goal cell must be inactive-coloured.
2. **Pair offsets are observable and constrain the assignment hard.** Press a
   direction and whatever moves in lockstep with the active piece is its partner
   -- so the first successful move of each body identifies its pairing for free.
   A pair separated by delta can then only target two goal cells separated by
   exactly delta, which in practice pins most pairs with zero further probing.
3. **Reachability prunes.** A body must be able to reach a goal configuration
   through cells where every one of its halves stands on floor.

Knowledge is a candidate set per piece, narrowed by those channels plus
bijectivity (a confirmed goal belongs to nobody else) and whole-assignment
refutation (all pieces parked on their hypothesised goals and the level did NOT
end => that assignment is wrong). The planner acts on the CHEAPEST assignment
consistent with everything known so far, keeps it sticky until the evidence
refutes it, and re-plans on refutation -- the hypothesis / pursuit / refutation /
revision arc DESIGN.md wants in the corpus. Eliminations are sound (the true
assignment is never eliminated) and every refutation strictly shrinks the
candidate space, so the search terminates.

Planning under a fixed hypothesis is exact and needs no search
--------------------------------------------------------------
Given an assignment the puzzle decomposes, because pieces never collide (only
walls block) and a pair is a rigid body (its separation is invariant). So each
piece/pair is one **body** with a 2-D configuration space, and its
distance-to-goal is a BFS field over that space (cached per level: it depends
only on the walls, the hypothesised goal and the invariant offsets). The only
coupling left is the ACTION5 cursor -- reaching a body costs
``(index - active) % n``. Cycling is forward-only, so driving the bodies in
increasing cyclic order costs exactly ``max cyclic distance`` in total, and a
body is best driven from whichever of its (one or two) indices is cyclically
nearer. Hence, under the current hypothesis,

    cost = sum(distance of every unplaced body)
         + max over unplaced bodies of the cyclic distance to its nearer index

`_cost` evaluates that in O(pieces), which makes `solve_from` an
exactly-optimal-under-the-hypothesis plan builder and `optimal_set_from` its
exact co-optimal SET (five successor evaluations, no engine copies) -- so
stochastic-optimal sampling still yields many distinct routes per board.

A piece whose pairing is still unknown is provisionally planned as a single. If
that is wrong the very first move reveals the partner (replan), and if the move
is REFUSED although the visible grid says the piece's own target is floor, that
refusal proves a hidden partner is against a wall: the (cell, direction) pair is
remembered as a dead move and the body detours, which is itself the probe.

The game also stashes a reverse-scramble ``solution_steps`` list. It is never
read: it is the answer, and reading it would make this a privileged solver.

Recovery: every move is reversible and pieces cannot block each other, so no
action sequence can brick a level -- `_burst_recovery_wins` is overridden to say
so. Exploration prefixes and perturbation bursts are relabelled with the optimal
continuation under the then-current hypothesis and never need a RESET; they also
help, since random moves reveal pairings for free.

Per-seed augmentation is inherent: ``PairedPieces(seed)`` draws the walls, the
piece/goal cells, the pairing, the scramble AND the palette + piece/goal shapes
for all 7 levels from that one seed, so one episode == one seed, reproducible.

Usage (run from the repo root):
    python solvers/generate_paired_pieces_training.py --episodes 1000 \
        --out data/training_multi_level/paired_pieces
"""

from __future__ import annotations

import sys
from collections import Counter, deque
from pathlib import Path

# Repo root (parent of solvers/) -- that's where the games/ package lives.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from games.paired_pieces.paired_pieces import CELL, PairedPieces  # noqa: E402
from solvers.base_solver import BaseSolver  # noqa: E402

# (dx, dy) -> action id, matching PairedPieces._DELTAS.
_MOVES = {(0, -1): 1, (0, 1): 2, (-1, 0): 3, (1, 0): 4}
_DIRS = tuple(_MOVES)
_CYCLE = 5                      # ACTION5 -- advance the active piece
_INF = float("inf")

_UNKNOWN = None                 # pairing not yet observed
_SINGLE = -1                    # observed to move alone
_SEARCH_NODES = 200_000         # cap on the assignment search


class _Body:
    """One independently-movable unit under the current hypothesis: a single
    piece, or a piece plus the partner observed to move with it.

    ``controls`` are the piece indices that can drive it (either half of a pair
    drives it identically), ``pos`` is the ANCHOR half's cell, and ``field`` maps
    every legal anchor cell to its distance to the hypothesised goal
    configuration -- so ``field.get(p)`` doubles as the legality test for a move
    to ``p``."""

    __slots__ = ("controls", "anchor", "pos", "field", "d")

    def __init__(self, anchor, controls, pos, field):
        self.anchor = anchor
        self.controls = controls
        self.pos = pos
        self.field = field
        self.d = field.get(pos, _INF)


class _Knowledge:
    """Everything the agent has LEARNED about one (seed, level) from the frames.

    Nothing here is read from the engine's hidden bookkeeping: ``partner`` is
    filled in by watching what moves in lockstep, ``cands`` by the on-goal colour
    bit plus the pair-offset and bijectivity constraints, ``dead`` by moves the
    engine refused, and ``failed`` by whole assignments that were completed
    without the level ending."""

    __slots__ = ("n", "partner", "cands", "dead", "failed", "hyp",
                 "inactive_colour", "last", "moved")

    def __init__(self, n: int, goal_cells) -> None:
        self.n = n
        self.partner = [_UNKNOWN] * n
        self.cands = [set(goal_cells) for _ in range(n)]
        self.dead = [set() for _ in range(n)]     # (own cell, dir) refused by the engine
        self.failed = set()                       # assignments completed with no win
        self.hyp = None                           # sticky assignment: list[cell]
        self.inactive_colour = None               # learned, not assumed
        self.last = None                          # (positions, active) previously seen
        self.moved = False                        # any piece moved since level (re)start?


class PairedPiecesSolver(BaseSolver):
    game_id = "paired_pieces"
    # solve_from re-derives everything from the LIVE frame-visible state plus the
    # knowledge it accumulated by observing; every move is reversible and pieces
    # never block each other, so an exploratory detour is always re-plannable.
    supports_recovery = True
    recovery_mode = "replan"

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self._fields: dict = {}        # (level, offsets, goal) -> BFS distance field
        self._know: dict = {}          # level -> _Knowledge
        self._pressed = None           # the action WE last pressed (an agent knows this)

    # ── engine plumbing ──────────────────────────────────────────────────────
    def make_game(self, seed: int):
        self._seed = seed
        # One instance carries all 7 levels; every level's layout, scramble and
        # palette is drawn from `seed` in the constructor. Both caches are
        # per-seed: the boards change, and so does what has been learned.
        self._fields.clear()
        self._know.clear()
        return PairedPieces(seed=seed)

    def available_actions(self, game) -> list[int]:
        # ACTION7 (the game's undo) is deliberately excluded: it is not a real
        # competition action, so it must never enter the corpus -- not even as an
        # exploratory step.
        return [1, 2, 3, 4, 5]

    def set_level(self, game, level_idx: int) -> None:
        super().set_level(game, level_idx)
        # A level (re)start teleports every piece, so the next observation must
        # not read the jump as "a body translated" / "my move was refused". Also
        # used for RESET, which the base routes through here.
        self._pressed = None
        k = self._know.get(level_idx)
        if k is not None:
            k.last = None
            k.moved = False

    def drive(self, game, action):
        # Remember what we pressed: a refused move is only interpretable if you
        # know which direction you asked for (an agent always does). `_observe`
        # CONSUMES this, so one press is interpreted exactly once even though
        # both solve_from and optimal_set_from observe on the same step.
        self._pressed = action
        return super().drive(game, action)

    def _burst_recovery_wins(self, game, plan: list) -> bool:
        """Always True -- and it must be overridden to say so.

        The base drives the recovery plan on a throwaway copy and demands a WIN,
        as a guard against a solver confidently mis-claiming a stranded state is
        winnable. Here that test is wrong in both directions: this game cannot be
        stranded (every move is reversible and pieces never block each other, so
        the goal configuration stays reachable from anywhere a burst can go), and
        our plan is a HYPOTHESIS, not a proof -- while the assignment is still
        being narrowed down a single replay legitimately ends without a win. Left
        as the base's version, every burst taken before the assignment is pinned
        down would be rolled back and lost from the corpus."""
        return True

    # ── observation: only what the rendered frame shows ───────────────────────
    def _read(self, game):
        """Read the frame-visible state: ``(floor, goals, pos, colour, active)``.

        Taken from the level's sprites rather than by pixel-crawling the 64x64
        frame (the camera centres and letterboxes the board, so cell -> pixel
        arithmetic would add nothing but offsets), but restricted to what the
        frame actually shows: which cells are floor vs wall (visibly different
        colours), the SET of goal cells (all goals render identically -- their
        order, which is what the win condition keys on, is NOT read), each piece's
        cell, and each piece's current colour. Piece indices are observable: they
        are the order ACTION5 cycles through, and the highlighted piece is
        ``game._active``."""
        floor, goals, pieces = set(), [], {}
        for s in game.current_level.get_sprites():
            cell = (s.x // CELL, s.y // CELL)
            tags = s.tags
            if "piece" in tags:
                pieces[int(s.name.rsplit("_", 1)[1])] = (cell, int(s.pixels.max()))
            elif "goal" in tags:
                goals.append(cell)
            elif s.name.startswith("floor_"):
                floor.add(cell)
        n = len(pieces)
        pos = [pieces[i][0] for i in range(n)]
        colour = [pieces[i][1] for i in range(n)]
        return floor, frozenset(goals), pos, colour, int(game._active)

    def _observe(self, game, level_idx: int):
        """Read the frame and fold every new fact into this level's knowledge.

        Called at the top of both `solve_from` and `optimal_set_from`, so it runs
        once per recorded step -- i.e. it sees the state each action produced,
        which is exactly when the engine's answer to the previous press is on
        screen."""
        obs = self._read(game)
        floor, goals, pos, colour, active = obs
        n = len(pos)
        k = self._know.get(level_idx)
        if k is None or k.n != n:
            k = self._know[level_idx] = _Knowledge(n, goals)

        # ── what moved? -> pairing, or a refused move ────────────────────────
        # Interpreting the diff needs the press that caused it, and each press is
        # interpreted ONCE: `_observe` runs from both solve_from and
        # optimal_set_from on the same step, and a second reading would see "no
        # piece moved" and mistake the already-applied move for a refused one.
        pressed = None if self._pressed is None else self._pressed.action_id
        self._pressed = None
        if k.last is not None and len(k.last[0]) == n and pressed in (1, 2, 3, 4):
            prev_pos, prev_active = k.last
            movers = [i for i in range(n) if pos[i] != prev_pos[i]]
            step = None
            if movers:
                step = (pos[prev_active][0] - prev_pos[prev_active][0],
                        pos[prev_active][1] - prev_pos[prev_active][1])
            if (movers and prev_active in movers and step in _DIRS
                    and all((pos[i][0] - prev_pos[i][0],
                             pos[i][1] - prev_pos[i][1]) == step for i in movers)):
                k.moved = True
                others = [i for i in movers if i != prev_active]
                if not others:                       # nothing came along -> single
                    k.partner[prev_active] = _SINGLE
                elif len(others) == 1:               # its partner, revealed
                    k.partner[prev_active] = others[0]
                    k.partner[others[0]] = prev_active
            elif movers:
                # Not a single-body translation: a RESET or a burst rollback moved
                # us wholesale. Re-sync without inferring anything.
                k.moved = False
            else:
                # Refused move. Remember it so we stop asking, and note that the
                # visible grid may not explain it -- when the active piece's own
                # target cell is floor, the refusal PROVES a hidden partner sits
                # against a wall.
                for (dx, dy), aid in _MOVES.items():
                    if aid == pressed:
                        k.dead[prev_active].add((prev_pos[prev_active], (dx, dy)))
        k.last = (list(pos), active)

        # ── the on-goal colour bit ───────────────────────────────────────────
        # Any non-active piece standing OFF a goal cell must be inactive-coloured,
        # which is how the colour code is learned rather than assumed.
        off_goal = {colour[i] for i in range(n) if i != active and pos[i] not in goals}
        if len(off_goal) == 1:
            k.inactive_colour = off_goal.pop()
        if k.inactive_colour is not None:
            crowd = Counter(pos)
            for i in range(n):
                # The active piece is drawn highlighted, and a piece sharing a
                # cell with another may be hidden underneath it -- in both cases
                # the frame does not show its colour, so nothing is inferred.
                if i == active or crowd[pos[i]] > 1 or pos[i] not in goals:
                    continue
                if colour[i] == k.inactive_colour:
                    k.cands[i].discard(pos[i])       # on a goal, but not mine
                else:
                    k.cands[i] = {pos[i]}            # on MY goal: confirmed
        self._propagate(k, pos)
        return k, obs

    @staticmethod
    def _propagate(k: _Knowledge, pos) -> None:
        """Close the candidate sets under the two structural constraints: goals
        are used by exactly one piece each, and a pair's two goals are separated
        by exactly the pair's (observed, invariant) offset."""
        for _ in range(k.n + 2):
            changed = False
            for i in range(k.n):
                if len(k.cands[i]) == 1:
                    (c,) = tuple(k.cands[i])
                    for j in range(k.n):
                        if j != i and c in k.cands[j]:
                            k.cands[j].discard(c)
                            changed = True
                j = k.partner[i]
                if j is None or j == _SINGLE:
                    continue
                off = (pos[j][0] - pos[i][0], pos[j][1] - pos[i][1])
                keep = {c for c in k.cands[i]
                        if (c[0] + off[0], c[1] + off[1]) in k.cands[j]}
                if keep != k.cands[i]:
                    k.cands[i] = keep
                    changed = True
            if not changed:
                return

    # ── configuration-space distance fields ──────────────────────────────────
    def _field(self, level_idx: int, offsets, floor, goal):
        """BFS distance-to-goal over a body's configuration space: the anchor
        cells where EVERY half stands on floor. Cached -- it depends only on the
        level's walls, the body's offsets and the goal it is aimed at."""
        key = (level_idx, offsets, goal)
        cached = self._fields.get(key)
        if cached is not None:
            return cached

        def free(p):
            return all((p[0] + ox, p[1] + oy) in floor for ox, oy in offsets)

        field: dict = {}
        if free(goal):
            field[goal] = 0
            q = deque([goal])
            while q:
                x, y = q.popleft()
                d = field[(x, y)] + 1
                for dx, dy in _DIRS:
                    nb = (x + dx, y + dy)
                    if nb not in field and free(nb):
                        field[nb] = d
                        q.append(nb)
        self._fields[key] = field
        return field

    # ── hypothesis: the cheapest assignment consistent with what is known ────
    @staticmethod
    def _specs(k: _Knowledge, pos):
        """Group the pieces into bodies from the pairing observed so far. A piece
        whose pairing is still unknown is provisionally its own body -- its first
        successful move settles the question."""
        specs, seen = [], set()
        for i in range(k.n):
            if i in seen:
                continue
            j = k.partner[i]
            members = (i,) if j is None or j == _SINGLE else (i, j)
            seen.update(members)
            offsets = tuple((pos[m][0] - pos[i][0], pos[m][1] - pos[i][1])
                            for m in members)
            specs.append((i, members, offsets))
        return specs

    def _options(self, k, obs, level_idx):
        """Per body, every goal configuration still allowed by the candidate sets,
        the pair-offset constraint and reachability: ``(goal, d, cells, field)``.
        ``None`` if some body has no option left (should not happen -- the true
        assignment is never eliminated)."""
        floor, goals, pos, _colour, _active = obs
        options = []
        for anchor, members, offsets in self._specs(k, pos):
            opts = []
            for g in sorted(goals):
                cells = tuple((g[0] + ox, g[1] + oy) for ox, oy in offsets)
                if any(c not in goals for c in cells):
                    continue                       # offsets must land on goals
                if any(c not in k.cands[m] for c, m in zip(cells, members)):
                    continue
                field = self._field(level_idx, offsets, floor, g)
                d = field.get(pos[anchor])
                if d is None:
                    continue                       # unreachable configuration
                opts.append((g, d, cells, field))
            if not opts:
                return None
            options.append((anchor, members, opts))
        return options

    def _hypothesise(self, k, obs, level_idx):
        """The cheapest assignment consistent with everything known: an exact
        cover of the goal cells by the bodies, minimising the plan cost. Cheapest
        first is also the right order to TEST hypotheses in -- a refutation costs
        what the walk cost."""
        options = self._options(k, obs, level_idx)
        if options is None:
            return None
        active = obs[4]
        order = sorted(range(len(options)), key=lambda t: len(options[t][2]))
        choice: list = [None] * len(options)
        used: set = set()
        best: list = [None, None]                  # [cost, assignment]
        nodes = [0]

        def leaf():
            hyp = [None] * k.n
            total = span = 0
            for oi, ch in enumerate(choice):
                anchor, members, _opts = options[oi]
                g, d, cells, _field = ch
                for m, c in zip(members, cells):
                    hyp[m] = c
                if d:
                    total += d
                    span = max(span, min((i - active) % k.n for i in members))
            if tuple(hyp) in k.failed:
                return
            cost = total + span
            if best[0] is None or cost < best[0]:
                best[0], best[1] = cost, hyp

        def rec(t: int, dsum: int) -> None:
            if nodes[0] > _SEARCH_NODES:
                return
            nodes[0] += 1
            if best[0] is not None and dsum >= best[0]:
                return                             # span >= 0, so dsum bounds below
            if t == len(order):
                leaf()
                return
            oi = order[t]
            _anchor, _members, opts = options[oi]
            for ch in opts:
                if any(c in used for c in ch[2]):
                    continue
                used.update(ch[2])
                choice[oi] = ch
                rec(t + 1, dsum + ch[1])
                used.difference_update(ch[2])
                choice[oi] = None

        rec(0, 0)
        return best[1]

    def _build(self, k, obs, level_idx):
        """Turn the sticky hypothesis into bodies with live distances, or ``None``
        if the hypothesis no longer fits what is known (new pairing, refuted cell,
        unreachable configuration)."""
        hyp = k.hyp
        floor, goals, pos, _colour, _active = obs
        if hyp is None or len(hyp) != k.n or len(set(hyp)) != k.n:
            return None
        bodies = []
        for anchor, members, offsets in self._specs(k, pos):
            g = hyp[anchor]
            if any(hyp[m] not in k.cands[m] for m in members):
                return None
            if any((g[0] + ox, g[1] + oy) != hyp[m]
                   for m, (ox, oy) in zip(members, offsets)):
                return None                        # offsets no longer consistent
            field = self._field(level_idx, offsets, floor, g)
            if pos[anchor] not in field:
                return None
            bodies.append(_Body(anchor, members, pos[anchor], field))
        return bodies

    def _plan_state(self, game, level_idx: int):
        """``(k, obs, bodies, complete)`` under a hypothesis that is consistent
        with every observation so far, or ``None`` if the level is unsolvable
        given what we know (a bug -- the truth is never eliminated).

        ``complete`` means every piece already sits on its hypothesised goal. If
        that happens after a move and the level did NOT end, the assignment as a
        whole is refuted -- the engine tested it for us -- so it is recorded as
        failed and a fresh hypothesis is drawn. (Reached with NO move made since
        the level began it means something else entirely: the board started
        solved, and the engine only tests the win condition inside a move, so it
        needs a nudge -- see `_cost`.)"""
        k, obs = self._observe(game, level_idx)
        for _ in range(k.n + 2):
            bodies = self._build(k, obs, level_idx) if k.hyp is not None else None
            if bodies is None:
                k.hyp = self._hypothesise(k, obs, level_idx)
                if k.hyp is None:
                    return None
                bodies = self._build(k, obs, level_idx)
                if bodies is None:
                    return None
            complete = not any(b.d for b in bodies)
            if complete and k.moved:
                k.failed.add(tuple(k.hyp))
                k.hyp = None
                continue
            return k, obs, bodies, complete
        return None

    # ── cost under the current hypothesis ────────────────────────────────────
    @staticmethod
    def _cost(n: int, active: int, bodies, *, after_move: bool = False) -> float:
        """Actions still needed to place every body, assuming the hypothesis holds.

        Unplaced bodies contribute their distance-to-goal, plus the ACTION5
        cycles: visiting the chosen indices in increasing cyclic order costs
        exactly the LARGEST cyclic distance among them (forward-only cycling
        wraps at most once), and each body is driven from its cyclically nearer
        control index -- so minimising each term minimises their max.

        ``after_move`` disambiguates the all-placed state, whose cost genuinely
        depends on HOW it was reached: the engine tests the win condition only
        inside a MOVE, so all-placed *after a move* is a win (cost 0), while a
        board that merely STARTS all-placed still needs a nudge -- cycle to a
        movable body, step off a goal and step back (2 moves)."""
        todo = [b for b in bodies if b.d]
        if not todo:
            if after_move:
                return 0
            best = _INF
            for b in bodies:
                if any((b.pos[0] + dx, b.pos[1] + dy) in b.field for dx, dy in _DIRS):
                    best = min(best, min((i - active) % n for i in b.controls) + 2)
            return best
        total = span = 0
        for b in todo:
            if b.d == _INF:
                return _INF
            total += b.d
            span = max(span, min((i - active) % n for i in b.controls))
        return total + span

    # ── the plan ─────────────────────────────────────────────────────────────
    def solve_from(self, game, level_idx: int, seed: int) -> list:
        """The best plan given everything observed so far: exactly optimal under
        the current hypothesis, and information-seeking where the hypothesis is
        not yet pinned down."""
        state = self._plan_state(game, level_idx)
        if state is None:
            return []
        k, obs, bodies, complete = state
        _floor, _goals, pos, _colour, active = obs
        n = k.n

        if complete:
            # Every piece is on its goal and no move has been made since the level
            # opened, so the engine has never tested the win condition. Nudge:
            # cycle to the nearest movable body, step off and step back.
            best = None
            for b in bodies:
                idx = min(b.controls, key=lambda i: (i - active) % n)
                for dx, dy in _DIRS:
                    if ((b.pos[0] + dx, b.pos[1] + dy) in b.field
                            and (pos[idx], (dx, dy)) not in k.dead[idx]):
                        cyc = (idx - active) % n
                        if best is None or cyc < best[0]:
                            best = (cyc, (dx, dy))
                        break
            if best is None:
                return []
            cyc, (dx, dy) = best
            return [_CYCLE] * cyc + [_MOVES[(dx, dy)], _MOVES[(-dx, -dy)]]

        todo = [b for b in bodies if b.d]
        order = sorted(((min((i - active) % n for i in b.controls), b) for b in todo),
                       key=lambda t: t[0])
        plan: list[int] = []
        cur = active
        for _cyc, body in order:
            idx = min(body.controls, key=lambda i: (i - cur) % n)
            plan.extend([_CYCLE] * ((idx - cur) % n))
            cur = idx
            p, d, field = body.pos, body.d, body.field
            cell = pos[idx]                        # the driving half's own cell
            while d:
                step = next((dd for dd in _DIRS
                             if field.get((p[0] + dd[0], p[1] + dd[1])) == d - 1
                             and (cell, dd) not in k.dead[idx]), None)
                if step is None:
                    # Every shortest step from here has been refused, so a hidden
                    # partner is in the way: detour sideways (that IS the probe --
                    # the move reveals the partner) and re-plan afterwards.
                    step = next((dd for dd in _DIRS
                                 if (p[0] + dd[0], p[1] + dd[1]) in field
                                 and (cell, dd) not in k.dead[idx]), None)
                    if step is None:
                        # The body cannot move at all. Mobility is symmetric, so
                        # it never moved and never will: its goal must be where it
                        # stands. Record that, cycle (always legal), re-plan.
                        for m in body.controls:
                            k.cands[m] &= {pos[m]}
                        k.hyp = None
                        return [_CYCLE]
                    plan.append(_MOVES[step])
                    return plan
                plan.append(_MOVES[step])
                p = (p[0] + step[0], p[1] + step[1])
                cell = (cell[0] + step[0], cell[1] + step[1])
                d -= 1
        return plan

    def optimal_set_from(self, game, level_idx: int, seed: int) -> list | None:
        """Every action that strictly reduces `_cost` -- the exact co-optimal set
        under the current hypothesis (stochastic optimal play, DESIGN.md). Cheap:
        five successor evaluations of an O(pieces) formula, no engine copies.
        ``None`` where the plan is information-seeking rather than cost-reducing
        (a probe, or a cycle to expose a colour), so the base falls back to the
        plan head."""
        state = self._plan_state(game, level_idx)
        if state is None:
            return None
        k, obs, bodies, _complete = state
        _floor, _goals, pos, _colour, active = obs
        n = k.n
        cost = self._cost(n, active, bodies)
        if cost == _INF:
            return None
        target = cost - 1
        out: list[int] = []

        if n > 1 and self._cost(n, (active + 1) % n, bodies) == target:
            out.append(_CYCLE)

        body = next(b for b in bodies if active in b.controls)
        saved = (body.pos, body.d)
        for step, aid in _MOVES.items():
            if (pos[active], step) in k.dead[active]:
                continue                           # the engine already refused it
            nb = (body.pos[0] + step[0], body.pos[1] + step[1])
            nd = body.field.get(nb)
            if nd is None:
                continue
            body.pos, body.d = nb, nd
            if self._cost(n, active, bodies, after_move=True) == target:
                out.append(aid)
            body.pos, body.d = saved
        return out or None


if __name__ == "__main__":
    raise SystemExit(PairedPiecesSolver.main())
