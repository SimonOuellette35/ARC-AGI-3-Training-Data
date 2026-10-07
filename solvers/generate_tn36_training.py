"""Generate Phase-1 training data for the TN36 game (mouse-only / ACTION6).

TN36 is **"teach-then-program"**: two side-by-side panels each own a *tablet* of columns,
and every column is a stack of marks read as a binary code (LSB = topmost mark, ``code =
sum(1<<i for lit marks)``).  A *run* executes the player tablet's columns left-to-right as a
little program on that panel's *block*, and the level is won when the block matches the target
sprite on ``(x, y, rotation, scale, colour)``.  The left panel is a read-only DEMO whose action
boxes replay one taught code at a time; the right panel is the player's.  Everything is driven
with a single action -- ACTION6, a mouse click -- so a solution is a sequence of clicks:
toggle marks to dial in each column's code, then click the run button.

Mechanics (obfuscated ``games/tn36/tn36.py``; all verified against the engine):

  * **A code's effect** (move ±1/±2 cells, rotate ±90/180/270, scale ±1, recolour, noop=0) is
    held in the panel's opcode table ``dfguzecnsr``, which the solver never reads, and which is
    re-keyed per seed (``Tn36._randomize_codes``) so it cannot be memorised either.  The mapping
    is **not derivable from any single frame**: a static board shows the block, the target, the
    walls and the tablet layout, but nothing that says what a bar pattern does.  The demo panel
    has no target sprite, and clicking an action box reseats the demo block and runs its program
    inside ONE action -- so at one-frame-per-action even the demo is uninterpretable (several
    demos start and end on the same cell).  The mapping exists only in the **animation**: the
    engine renders after every internal ``step()`` and returns all of those frames for a single
    action, one per executed column, with the tablet's bars drawn and a highlight walking the
    executing column.  ``_discover`` therefore plays the teaching material ON CAMERA and reads
    the effect table out of exactly those recorded frames (``_learn_from_run``).  The learned
    effect is replayed through the game's own actuators (moves/scales via the wall+hazard-aware
    ``otrzjnmayi``/``adjust_scale``, rotate/recolour on the block), so the transition still
    respects walls and traps without ever inspecting the rule.
  * **Runs reset.**  At the end of a run that does not reach the target the block SNAPS BACK to
    its last checkpoint (``aasnichwxq``; the player tablet's reset flag defaults True).  The
    checkpoint only advances when a run *ends on a hole* at matching scale (``nklxbrmrww``).
    So progress is made hole-by-hole: L6 (index 5) needs 2 runs, L7 (index 6) needs 3.
  * **Hazards** (index 5/6 only) are toggled every third executed column (``pdawcsxhlx`` at
    ``col % 3 == 2``) and kill the block if they flip on top of it; a killed block just wastes
    the run (it revives at the checkpoint).  Moving onto a live hazard kills too.
  * **Budget.**  The timer bar advances one tick per click (every *second* click for index>=5).
    Every mark toggled is a click and every run is a click, so a code costs its popcount; the
    game's ``MAX_MARKS_PER_CODE=3`` keeps that affordable.  Running the timer out loses.

Migration notes (this file is now a thin ``BaseSolver`` subclass)
----------------------------------------------------------------
The record/replay loop, the episode schema and the WIN-filter CLI live in ``BaseSolver``. This
file supplies only the TN36 expert.  Four TN36-specific decisions carry over:

1. **The engine IS the model, and it is driven with FRESH instances, never copies.**  Every
    opcode in ``dfguzecnsr`` is a ``lambda: self.otrzjnmayi(...)`` closed over the panel, and
    ``deepcopy`` does not copy function objects -- a copied game's opcode table still mutates the
    ORIGINAL block.  So the whole solver avoids ``copy`` entirely: a candidate run is evaluated by
    building a fresh ``Tn36(seed).set_level(idx)`` and replaying every run so far plus the
    candidate.  Geometry+augmentation are a pure function of ``(seed, level)``, so a fresh
    instance reproduces any point in a play exactly.  Because the solver never plans from an
    arbitrary perturbed state (only a level's initial one), recovery is ``"reset"`` mode:
    ``supports_recovery = True`` with ONE RESET back to the initial state before the plan runs.

2. **Two-level search: outer BFS over checkpoints, inner BFS over block states.**  Every candidate
    program is verified by a real run before it is trusted -- the engine is authoritative for
    holes, hazards and the timer.

3. **Discovery happens IN the demonstration, not beside it.**  The recorded trajectory opens with
    the agent spending real clicks to find out what the codes do -- every demo action box on
    levels 2-7, and one-column experiment runs on level 1, which has no demo panel -- and the
    search then runs on the table read out of those frames.  The alphabet is the union of the
    codes the demo tablet DISPLAYS (level data ``Programs`` is never read: it would also reveal
    the codes of boxes the agent never clicked); the demo-less level tries every enterable code,
    a bound read off the tablet's mark count.  Nothing is learned off-camera, so an agent
    watching the recording sees every step the solver used.  Cost is well inside the timer:
    worst observed totals are 19/61 (L1) and 66/122 (L7).

4. **Rotation stays LIVE; clicks are inverse-rotated.**  Each seed draws a frame rotation the
    engine undoes on every click via ``remap_click``.  ``solve_from`` plans in grid space and
    emits each click through ``_inv_rot_click`` (the r11l convention), so a seed varies the frames
    AND the recorded click coordinates.  The click sequence is precomputed against a fresh sim
    (marks persist within a level, so run k's toggles depend on the post-run-(k-1) tablet), then
    replayed against the live recording instance -- both evolve identically.

Action schema (same corpus convention as r11l / lp85):

    RESET :  {"type": "simple", "index": 0}                       (leading action for obs[0])
    click :  {"type": "mouse",  "index": 6, "data": {"x": x, "y": y}}

Usage:
    python solvers/generate_tn36_training.py --episodes 1600 --out data/training_multi_level/tn36
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arcengine import ActionInput, GameAction, GameState  # noqa: E402
from games.tn36.tn36 import Tn36  # noqa: E402
from solvers.base_solver import BaseSolver  # noqa: E402
from utils.explore import (  # noqa: E402
    Action, CLICK_ACTION, EpsilonSchedule, ExplorationPolicy)

_STEP_GUARD = 5000
_GRID = 64

# Search bounds (generous; every real level solves far inside them).
_MAX_RUNS = 8            # checkpoints to chain before giving up
_OUTER_NODES = 4000      # outer-BFS expansion cap


# ── engine driving ──────────────────────────────────────────────────────────────
def _make_level(seed: int, level_idx: int) -> Tn36:
    """A fresh game pinned on ``level_idx``.  ``Tn36`` derives every augmentation from
    ``(seed, level)``, so this reproduces any level state exactly."""
    game = Tn36(seed=seed)
    game.set_level(level_idx)
    return game


def _inv_rot_click(gx: int, gy: int, k: int) -> tuple[int, int]:
    """The grid click, unconverted.

    Solvers plan and emit in the core game's UPRIGHT space; `BaseSolver` does the
    one conversion to screen space when it records. Converting here as well
    rotated twice and put the click on the wrong cell at k != 0."""
    return gx, gy



def _drive_click(game, gx: int, gy: int, watch=None):
    """One ACTION6 click at grid ``(gx, gy)`` (inverse-rotated to display space), run to
    completion.  Returns ``(solved, dead, (dx, dy), states)``.  Used to advance the throwaway
    sim while precomputing the click sequence.

    ``watch`` is an optional panel whose block is sampled after EVERY rendered frame, giving
    ``states`` -- one entry per frame of the animation this click produces.  That is exactly
    the frame sequence `BaseSolver.drive` now records (the engine renders after each internal
    ``step()`` and hands an agent all of them for one action), so anything the solver infers
    from ``states`` is inferable from the demonstration itself.  ``states`` is empty when
    ``watch`` is None."""
    k = game._rotation_k
    dx, dy = _inv_rot_click(gx, gy, k)
    game._full_reset = False
    game._set_action(ActionInput(id=GameAction.ACTION6, data={"x": int(dx), "y": int(dy)}))
    states = []
    guard = 0
    while not game.is_action_complete():
        if game._next_level or guard > _STEP_GUARD:
            break
        game.step()
        guard += 1
        if watch is not None:
            states.append(_blk(watch.ravxreuqho))
    solved = bool(game._next_level) or game._state == GameState.WIN
    dead = game._state == GameState.GAME_OVER
    return solved, dead, (int(dx), int(dy)), states


# ── block / level readers ─────────────────────────────────────────────────────────
def _player(game):
    return game.tsflfunycx.xsseeglmfh


def _blk(block) -> tuple:
    """The five matched fields (x, y, rotation, scale, colour)."""
    return (block.x, block.y, block.rotation, block.scale, block.dtxpbtpcbh)


def _target(P) -> tuple:
    d = P.ddzsdagbti
    return (d.x, d.y, d.rotation, d.scale, d.dtxpbtpcbh)


def _set_block(block, st) -> None:
    block.set_position(st[0], st[1])
    block.set_rotation(st[2])
    block.set_scale(st[3])
    block.vompvzytco(st[4])
    block.yudazjrktv()


def _enterable_limit(P) -> int:
    """Largest code this tablet can express: a column of ``nbars`` marks is a binary number,
    so codes run 0..2^nbars-1.  Read off the tablet (the marks are drawn), not from level data."""
    return (1 << len(P.tlwkpfljid.sfdxmndvyn[0].puakvdstpr)) - 1


# ── learning what each code DOES, from the frames the demonstration records ───────
def _classify(start, after) -> tuple | None:
    """Turn a single (start -> after) observation into an effect descriptor, or None if nothing
    changed.  Exactly one field moves per real code, so the first difference names the effect."""
    if (after[0], after[1]) != (start[0], start[1]):
        return ("move", after[0] - start[0], after[1] - start[1])
    if after[2] != start[2]:
        return ("rot", (after[2] - start[2]) % 360)
    if after[3] != start[3]:
        return ("scale", after[3] - start[3])
    if after[4] != start[4]:
        return ("color", after[4])
    return None


def _learn_from_run(codes, states, effects) -> None:
    """Fold ONE watched run into ``effects``, in place.

    A run executes its columns left-to-right, one per rendered frame, so ``states[k]`` is the
    block after ``k`` columns and column ``i`` (holding ``codes[i]``) is the transition
    ``states[i] -> states[i+1]``.  Verified against the engine for both panels: frame 0 is the
    pre-execution state and each later frame advances exactly one column, while the tablet keeps
    the program's bars drawn and moves a highlight onto the executing column.  So this is the
    inference a viewer makes from the recorded animation, nothing more.

    A code observed as "nothing changed" stays ``None`` (unknown) rather than being fixed as a
    noop: the same code usually recurs later in the same demo, and a move that happened to be
    wall-blocked here may resolve on a subsequent occurrence.  ``_effects_final`` settles the
    still-unknown ones as noops."""
    for i, code in enumerate(codes):
        if i + 1 >= len(states):
            break
        eff = _classify(states[i], states[i + 1])
        if effects.get(code) is None:
            effects[code] = eff


def _effects_final(effects, alpha) -> dict:
    """Settle the learned table: anything still unobserved (or observed as inert) is a noop."""
    out = {c: effects.get(c) or ("noop",) for c in alpha}
    out[0] = ("noop",)                                # a blank column does nothing
    return out


def _apply_effect(P, eff) -> None:
    """Apply a learned effect to ``P``'s block using the game's own actuators."""
    if eff is None:
        return
    kind = eff[0]
    if kind == "move":
        P.otrzjnmayi(eff[1], eff[2])
    elif kind == "rot":
        P.ravxreuqho.rotate(eff[1])
    elif kind == "scale":
        P.adjust_scale(eff[1])
    elif kind == "color":
        P.ravxreuqho.vompvzytco(eff[1])
    # ("noop",): nothing changes


# ── inner search: one run's worth of columns ──────────────────────────────────────
def _inner_candidates(P, ncols, alpha, effects, start_blk, start_haz):
    """Hazard-aware BFS over block states for a single run.  Returns a list of
    ``(program, end_block, is_win)`` for every reachable end that is the target or sits on a hole;
    programs are padded with noops to ``ncols``."""
    block = P.ravxreuqho
    hazards = P.cbbgkmkxku
    holes = P.wpilazztrp
    target = _target(P)

    def set_haz(vis):
        for h, v in zip(hazards, vis):
            h.set_visible(v)

    out = []
    seen = {(start_blk, start_haz)}
    frontier = [(start_blk, start_haz, [])]
    for depth in range(ncols):
        nxt = []
        for st, hv, path in frontier:
            for code in alpha:
                set_haz(hv)
                _set_block(block, st)
                _apply_effect(P, effects.get(code))
                if not block.knpmhsjjij:      # moved onto a live hazard -> killed
                    continue
                if depth < ncols - 1 and depth % 3 == 2:
                    P.txgixiygnw = depth
                    P.pdawcsxhlx()             # trap flip may kill a stationary block
                    if not block.knpmhsjjij:
                        continue
                ns = _blk(block)
                nhv = tuple(h.is_visible for h in hazards)
                prog = path + [code] + [0] * (ncols - len(path) - 1)
                is_win = ns == target
                for h in holes:               # engine shows holes then tests them at run end
                    h.set_visible(True)
                on_hole = any(h.scale == ns[3] and h.collides_with(block) for h in holes)
                if is_win or on_hole:
                    out.append((prog, ns, is_win))
                key = (ns, nhv)
                if key in seen:
                    continue
                seen.add(key)
                nxt.append((ns, nhv, path + [code]))
        frontier = nxt
    return out


# ── run replay (authoritative) ────────────────────────────────────────────────────
def _replay_runs(seed, level_idx, runs):
    """Fresh game; execute each run's program directly (set marks, run, drain).  Returns
    (game, won).  Ground-truth evaluator -- holes, hazards and resets included."""
    game = _make_level(seed, level_idx)
    ctx = game.tsflfunycx
    P = ctx.xsseeglmfh
    for prog in runs:
        P.fcrcpemsnv.mdzirgzgmn(list(prog))
        ctx.jucgkipjhs()
        guard = 0
        while ctx.nwjrtjcxpo and guard < _STEP_GUARD:
            ctx.step()
            guard += 1
        if P.yxabhsirzl:
            return game, True
    return game, P.yxabhsirzl


def solve_level(seed, level_idx, alpha, effects, *, max_runs=_MAX_RUNS):
    """Return a list of run-programs (each a game-space list of codes) that wins, or None.

    ``alpha``/``effects`` come from `_discover` -- what the level TAUGHT, learned by watching
    the teaching material play out.  Planning starts from the level's initial state, which the
    discovery phase leaves intact: demo-box clicks drive the DEMO panel only, and the
    player-side probes used on the demo-less level always snap the block back (that level has
    no holes, so no probe can advance the checkpoint).  Only the timer moves, and the click
    budget is checked by the real replay.

    Outer BFS over checkpoints; the inner BFS proposes candidate runs and each is confirmed by
    a real replay (which also yields the true resulting checkpoint)."""
    game = _make_level(seed, level_idx)
    P = _player(game)
    ncols = len(P.tlwkpfljid.sfdxmndvyn)
    start = _blk(P.ravxreuqho)
    haz0 = tuple(h.is_visible for h in P.cbbgkmkxku)

    seen = {start}
    q = deque([(start, haz0, [])])
    expanded = 0
    while q:
        cp, hv, runs = q.popleft()
        expanded += 1
        if expanded > _OUTER_NODES or len(runs) >= max_runs:
            continue
        cands = _inner_candidates(P, ncols, alpha, effects, cp, hv)
        # Wins first -- a confirmed win ends the search immediately.
        for prog, _ns, is_win in cands:
            if is_win:
                _g, won = _replay_runs(seed, level_idx, runs + [prog])
                if won:
                    return runs + [prog]
        # Otherwise treat confirmed hole-landings as new checkpoints.
        for prog, _ns, is_win in cands:
            if is_win:
                continue
            g2, won = _replay_runs(seed, level_idx, runs + [prog])
            if won:
                return runs + [prog]
            P2 = _player(g2)
            ncp = _blk(P2.ravxreuqho)
            if ncp not in seen:
                seen.add(ncp)
                nhv = tuple(h.is_visible for h in P2.cbbgkmkxku)
                q.append((ncp, nhv, runs + [prog]))
    return None


# ── discovery: learn the mechanics ON CAMERA ──────────────────────────────────────
def _discover(sim, rng):
    """Play the level's teaching material, in the recording, and learn each code from it.

    Returns ``(clicks, alpha, effects, solved, dead)`` where ``clicks`` are the display-space
    `Action`s already driven against ``sim``.

    Two cases, both of which spend real clicks against the timer:

      * **A demo panel exists.** Click every action box.  Each one loads a taught program onto
        the read-only demo tablet and plays it on the demo block, so the recorded animation
        shows the program's bars, a highlight walking the executing column, and the block's
        response frame by frame.  The boxes are independent (each reseats the demo block from
        the level's own data before running), so they are clicked in a random order for
        trajectory variety.  The alphabet is the union of the codes the demo tablet DISPLAYS --
        not ``current_level.get_data("Programs")``, which would also hand over the codes of
        boxes the agent never clicked.
      * **No demo panel** (level index 0).  Nothing is taught, so the agent experiments on its
        own tablet: enter each enterable code alone in column 0 and press run.  That level has
        no walls, holes or hazards, so every probe animates the effect and then snaps back --
        the state is untouched and only the timer is spent.

    Either way the effect table the search runs on is derived from exactly the frames the
    demonstration contains.  Nothing is learned off-camera."""
    ctx = sim.tsflfunycx
    D, P = ctx.kmhfvaisxu, _player(sim)
    clicks: list = []
    effects: dict = {}
    alpha: set = {0}

    def click(gx, gy, watch):
        solved, dead, (dx, dy), states = _drive_click(sim, gx, gy, watch=watch)
        clicks.append(Action(CLICK_ACTION, (dy, dx)))   # click_rc = (row=y, col=x)
        return solved, dead, states

    boxes = list(ctx.mcxkhvobyv)
    if boxes:
        order = list(range(len(boxes)))
        rng.shuffle(order)
        for i in order:
            b = boxes[i]
            solved, dead, states = click(b.x + b.width // 2, b.y + b.height // 2, D)
            if solved or dead:
                return clicks, sorted(alpha), _effects_final(effects, alpha), solved, dead
            codes = list(D.tlwkpfljid.ylczjoyapu)       # what the demo tablet now shows
            alpha.update(codes)
            _learn_from_run(codes, states, effects)
        return clicks, sorted(alpha), _effects_final(effects, alpha), False, False

    # Demo-less level: experiment on our own tablet.
    ncols = len(P.tlwkpfljid.sfdxmndvyn)
    alpha.update(range(1, _enterable_limit(P) + 1))
    for code in sorted(c for c in alpha if c):
        probe = [code] + [0] * (ncols - 1)
        for gx, gy in _run_click_cells(P, probe):       # toggles, then the run button
            solved, dead, states = click(gx, gy, P)
            if solved or dead:
                return clicks, sorted(alpha), _effects_final(effects, alpha), solved, dead
        _learn_from_run(probe, states, effects)         # states from the run-button click
    return clicks, sorted(alpha), _effects_final(effects, alpha), False, False


# ── click generation ──────────────────────────────────────────────────────────────
def _run_click_cells(P, program):
    """Grid cells to click to enter ``program`` on the LIVE player tablet and press run.  A mark is
    toggled iff its desired bit differs from its current state; the run button is clicked last."""
    cells = []
    tablet = P.tlwkpfljid
    for c, code in enumerate(program):
        column = tablet.sfdxmndvyn[c]
        for i, mark in enumerate(column.puakvdstpr):
            want = bool((code >> i) & 1)
            if mark.hokejgzome != want:
                cells.append((mark.x + mark.width // 2, mark.y + mark.height // 2))
    run_btn = P.owdgwmdfzu
    cells.append((run_btn.x + run_btn.width // 2, run_btn.y + run_btn.height // 2))
    return cells


# ── solver ─────────────────────────────────────────────────────────────────────
class Tn36Solver(BaseSolver):
    game_id = "tn36"
    # ``solve_from`` is a pure function of (seed, level) -- it rebuilds a throwaway sim and
    # returns the winning click sequence regardless of the LIVE board -- so it is valid from a
    # level's INITIAL state only.  That is exactly the reset-recovery contract: explore the
    # epsilon prefix, then ONE RESET back to the initial state, then replay the cached clicks.
    supports_recovery = True
    recovery_mode = "reset"
    step_guard = _STEP_GUARD

    def make_game(self, seed: int):
        return Tn36(seed=seed)

    def available_actions(self, game) -> list[int]:
        return [CLICK_ACTION]

    def solve_episode(self, seed: int, explore: bool = True):
        """All-or-nothing multi-level playthrough, one FRESH instance per level (the original
        model).  TN36 levels don't advance in place -- the engine is re-derived per (seed, level)
        -- so a single instance would carry a prior level's queued ``_next_level`` and make the
        base's driver report a spurious solve on the first click.  Fresh-per-level avoids that.

        Exploration (the epsilon prefix + RESET-recovery) is wired exactly like the base
        ``solve_episode``: the ``EpsilonSchedule`` and ``ExplorationPolicy`` are built ONCE per
        episode (so the ignorant-then-informed arc spans the whole playthrough, not each level)
        and handed to every ``record_level`` call.  RESET restores the level's initial state via
        the default ``reset_level`` (re-clone the clean template + re-run ``set_level``), which
        matches the fresh-instance state ``solve_from`` plans against."""
        game0 = self.make_game(seed)
        n = len(game0._levels)
        do_explore = explore and self.supports_recovery
        schedule = (EpsilonSchedule(self.rng, center=self._explore_center,
                                    jitter=self._explore_jitter)
                    if do_explore else None)
        exploration = (ExplorationPolicy(self.available_actions(game0), self.rng)
                       if do_explore else None)
        levels: list[dict] = []
        for level_idx in range(n):
            game = self.make_game(seed)
            try:
                obs, acts = self.record_level(game, level_idx, seed,
                                              schedule=schedule, exploration=exploration)
            except Exception:                    # noqa: BLE001 -- a bad seed just fails
                return False, levels
            if obs is None:
                return False, levels
            levels.append({"level_id": level_idx, "observations": obs, "actions": acts})
        return True, levels

    def optimal_set_from(self, game, level_idx: int, seed: int):
        """Left at the base default (``None``) -- a single-element (canonical-head) target.

        STOCHASTIC OPTIMAL note (DESIGN.md): the base's multi-action ``optimal_set_from``
        sampling works by taking one action from the returned set and then RE-CALLING
        ``solve_from`` on the perturbed live game to recover the remainder.  That requires a
        ``replan``-mode, state-reading ``solve_from``.  TN36's ``solve_from`` is state-BLIND
        (it re-derives the whole click sequence from ``(seed, level)`` on a throwaway sim and
        is only valid from a level's INITIAL state -- hence ``recovery_mode = "reset"``), so a
        multi-element set here would make the base re-issue an already-toggled mark and corrupt
        the code.  The genuine order-freedom TN36 does have -- the marks of ONE run commute
        (each just sets a bit; nothing executes until the run button) -- is therefore exposed
        the safe way instead: ``solve_from`` shuffles each run's toggle clicks with ``self.rng``
        (run button kept last).  Every permutation wins at the identical length, so trajectories
        vary while staying exactly optimal.  Verified length-stable across seeds x levels."""
        return None

    def solve_from(self, game, level_idx: int, seed: int):
        """Solve the level (outer/inner BFS over a fresh sim) and return a winning click sequence
        in display-pixel space.  The clicks for run k depend on the tablet state AFTER run k-1, so
        they are precomputed against a throwaway sim driven click-by-click; the live recording
        instance (driven by the base) evolves identically, so the coordinates match.

        STOCHASTIC OPTIMAL: within a single run the mark-toggle clicks COMMUTE -- each merely
        flips a mark bit and nothing executes until the run button -- so any order enters the
        same program and wins at the identical length.  Each run's toggles are therefore shuffled
        with ``self.rng`` (the run button, ``_run_click_cells``'s last cell, is kept last), giving
        a different equally-optimal trajectory per episode without ever lengthening the plan."""
        sim = _make_level(seed, level_idx)   # throwaway: reproduces game's initial state exactly
        P = _player(sim)

        # PHASE 1 -- DISCOVERY, recorded. Spend real clicks watching the level teach its codes;
        # the effect table below is read out of those very frames (see `_discover`).
        actions, alpha, effects, solved, dead = _discover(sim, self.rng)
        if dead:
            return []
        if solved:                           # a probe/demo somehow won outright
            return actions

        # PHASE 2 -- SOLVE, using only what phase 1 observed.
        runs = solve_level(seed, level_idx, alpha, effects)
        if not runs:
            return []
        for prog in runs:
            cells = _run_click_cells(P, prog)            # read live tablet each run
            toggles, run_btn = cells[:-1], cells[-1:]    # run button is always last
            self.rng.shuffle(toggles)                    # commuting toggles -> order-free
            for gx, gy in toggles + run_btn:
                solved, dead, (dx, dy), _st = _drive_click(sim, gx, gy)
                actions.append(Action(CLICK_ACTION, (dy, dx)))  # click_rc = (row=y, col=x)
                if dead:
                    return []                # winning runs never die; guard anyway
                if solved:
                    return actions
        return actions


if __name__ == "__main__":
    sys.exit(Tn36Solver.main())
