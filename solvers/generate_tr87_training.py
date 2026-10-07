"""Generate Phase-1 training data for the TR87 game (simple actions only).

TR87 is a **rule-table translation puzzle**. Glyphs are 5x5 sprites named
``nxkictbbvzt<FAM><d>`` (FAM in A/B/C, d in 1..7). An ``iqrduxrukrk`` separator bar
defines each rule -- the LHS walks LEFT from the bar, the RHS walks RIGHT -- and the
rule list ``cifzvbcuwqe`` is ordered by separator (y, x). Below the ``background``
divider the TOP row is the INPUT and the rest is the OUTPUT (answer) row. The win
check ``bsqsshqpox`` is a **greedy first-match parse** of the input: scan left->right,
fire the FIRST rule whose (name-wise) LHS matches, require its RHS to match the output
at the running offset, consume both, repeat. Everything keys on sprite *names*, never
colours, so the per-seed colour/rotation augmentation is solvability-free.

Two player modes (available_actions = [1, 2, 3, 4], all simple):

  * **normal levels 1-4 (idx 0-3)** -- ACTION3/4 move a cursor over OUTPUT cells,
    ACTION1/2 cycle that cell's digit within its family. The rule table is read-only,
    so the level is solvable iff the greedy parse covers the input and the required
    output (sum of translated-RHS lengths) fits the answer row. Because the output
    *content* never decides which rule fires, the target answer row is a pure O(n)
    function of (input, rules, translation flags) -- no search. Level 4 sets
    ``double_translation`` (each RHS is re-looked-up as another rule's LHS).

  * **alter levels 5-6 (idx 4-5)** -- ACTION1/2 instead shift a whole rule SET (one
    LHS list or one RHS list) UNIFORMLY; the input and output rows are fixed. The
    player must shift the rule sets so the fixed input parses to the fixed output.
    Level 6 also sets ``tree_translation`` (each RHS glyph expands via the first rule
    whose LHS[0] matches) + ``double_translation`` (tree takes precedence). A shift
    only changes a set's DIGITS, never its families, and every glyph in a set moves
    together -- so a set matches a target run iff the families line up and one shift
    reproduces every digit. We solve this with a small DFS over the parse structure
    (input is <=5 glyphs) that assigns each set's shift lazily and verifies every
    candidate against a pure parse model that is asserted equal to the engine's
    ``bsqsshqpox`` (see ``--verify``). The winning shift vector is **seed-independent**
    -- the per-seed relabel is a per-family *cyclic* offset applied to rules AND the
    input/output alike, so it cancels out of the (digit-difference) parse constraint --
    so each alter level is solved ONCE and the plan replayed for every seed.

Determinism
-----------
``Tr87`` takes no ``seed=`` ctor arg (the ls20/mx01/nu01 family): the jumble+relabel
augmentation is driven by ``self._rng`` and the palette by a *local* unseeded
``random.Random()`` inside ``on_set_level``. This generator makes both deterministic
per (seed, level): it seeds ``game._rng`` with ``tr87:<seed>:<level>`` and pins the
palette by installing, only for the duration of ``set_level``, a scoped ``random.Random``
factory keyed by ``tr87-colors:<seed>:<level>`` (no-arg calls -> seeded; seeded calls
pass through). A given ``--start-seed`` is reproducible byte-for-byte, and every emitted
episode is an engine-verified win.

Rotation
--------
Episodes are recorded at the level's REAL display rotation, NOT pinned to k=0. Live play
draws a rotation per level, so a corpus recorded only at k=0 would leave ~3/4 of the
boards a policy meets in an orientation it has never seen. ``Tr87`` is an
``AugmentedGame`` built as ``Tr87(seed=seed)``, so every level's rotation IS
``random_rotation_k(seed, level_index)`` -- k stays uniform over {0,1,2,3} but an episode
remains a pure function of its seed (this is also exactly how the other seeded games pick
k). Rotation is a pure DISPLAY transform, so the
plan -- computed over sprite names and positions -- is unaffected; what must change is
the *input*, because ``step`` maps a press through ``remap_action(action, k)`` before
using it. So each planned game action is converted to the SCREEN action a player would
press via ``inverse_remap_action_full``, and that screen action is what gets issued and
recorded.

Engine-driving trap: ``is_action_complete()`` is ``not _next_level and _action_complete``,
so after a win it stays False forever; and ``next_level()`` only sets ``_next_level`` for
non-last levels -- the LAST level (L6) calls ``win()`` instead (``_state = WIN``). So the
win-animation loop must break on ``_next_level OR _state != NOT_FINISHED`` or it steps
past the animation's end and dies with IndexError in ``pvgetmhmhgk[...]``.

Schema (multi-level, same as nu01 / maze / flood):

    {"game_id": "tr87",
     "levels": [{"level_id": k,
                 "observations": [[[c,...],...], ...],   # [T, H, W] palette idx
                 "actions":      [{"type":"simple","index":i}, ...]}]}   # length T

actions[0] is the RESET that produced obs[0]; actions[i] (i>=1) is the simple action
taking obs[i-1] -> obs[i]. Emitted indices are in 0..4.

Usage (run from the repo root, with the ARC-AGI-3 conda python):
    python solvers/generate_tr87_training.py --episodes 1000
    python solvers/generate_tr87_training.py --verify
"""

from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

# Repo root (parent of solvers/) -- that's where the games/ package lives.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import games.tr87.tr87 as _tr87mod  # noqa: E402
from arcengine import ActionInput, GameAction, GameState  # noqa: E402
from games.tr87.tr87 import Tr87  # noqa: E402
from solvers.base_solver import BaseSolver, DriveResult, _ID_TO_GAMEACTION  # noqa: E402
from utils.rotation import inverse_remap_action_full  # noqa: E402

GAME_ID = "tr87"
_STEP_GUARD = 5000

A1 = GameAction.ACTION1  # cycle digit -1  /  shift set -1
A2 = GameAction.ACTION2  # cycle digit +1  /  shift set +1
A3 = GameAction.ACTION3  # cursor -1
A4 = GameAction.ACTION4  # cursor +1
_NDIGITS = 7  # kjgicbtgrt


# ── glyph / shift primitives (a glyph is (family, digit)) ───────────────────────
def _pd(sprite):
    return (sprite.name[-2], int(sprite.name[-1]))


def _sh(glyph, k):
    f, d = glyph
    return (f, (d - 1 + k) % _NDIGITS + 1)


def _shl(glyphs, k):
    return tuple(_sh(g, k) for g in glyphs)


def _model(game):
    """Extract the pure parse model from the current level's live sprites."""
    rules = [([_pd(s) for s in l], [_pd(s) for s in r]) for l, r in game.cifzvbcuwqe]
    inp = [_pd(s) for s in game.zvojhrjxxm]
    out = [_pd(s) for s in game.ztgmtnnufb]
    tree = bool(game.current_level.get_data("tree_translation"))
    double = bool(game.current_level.get_data("double_translation"))
    return rules, inp, out, tree, double


# ── pure parse model (asserted == engine bsqsshqpox by --verify) ────────────────
def parse_ok(rules, sv, inp, out, tree, double):
    """True iff the greedy first-match parse of ``inp`` under rules shifted by ``sv``
    reproduces a prefix of ``out``. ``sv[2r]`` = LHS shift, ``sv[2r+1]`` = RHS shift.
    Mirrors bsqsshqpox: commit to the first LHS match (with tree/double expansion
    success), then the (expanded) RHS must match the output or the whole parse fails."""
    n, m = len(inp), len(out)
    i = o = 0
    while i < n:
        matched = False
        for ridx, (lhs, rhs) in enumerate(rules):
            L = len(lhs)
            if i + L > n:
                continue
            slhs = _shl(lhs, sv[2 * ridx])
            if tuple(inp[i:i + L]) != slhs:
                continue
            srhs = _shl(rhs, sv[2 * ridx + 1])
            if tree:
                exp, ok = [], True
                for gph in srhs:
                    for r2 in range(len(rules)):
                        l2, r2r = rules[r2]
                        if _sh(l2[0], sv[2 * r2]) == gph:
                            exp.extend(_shl(r2r, sv[2 * r2 + 1]))
                            break
                    else:
                        ok = False
                        break
                if not ok:
                    continue
                eff = exp
            elif double:
                eff = None
                for r2 in range(len(rules)):
                    l2, r2r = rules[r2]
                    sl2 = _shl(l2, sv[2 * r2])
                    if all(a == b for a, b in zip(srhs, sl2)):
                        eff = list(_shl(r2r, sv[2 * r2 + 1]))
                        break
                if eff is None:
                    continue
            else:
                eff = list(srhs)
            if o + len(eff) > m or list(out[o:o + len(eff)]) != eff:
                return False
            i += L
            o += len(eff)
            matched = True
            break
        if not matched:
            return False
    return True


# ── normal-level solver (target answer row + digit-cycling plan) ────────────────
def _target_output(game):
    """The names the answer row must show, mirroring bsqsshqpox's input-side parse.
    Returns a list of (family, digit), or None if the input cannot be parsed."""
    rules, inp, out, tree, double = _model(game)
    res = []
    i = 0
    while i < len(inp):
        for lhs, rhs in rules:
            L = len(lhs)
            if i + L > len(inp) or list(inp[i:i + L]) != list(lhs):
                continue
            if tree:
                exp, ok = [], True
                for gph in rhs:
                    for l2, r2 in rules:
                        if l2[0] == gph:
                            exp.extend(r2)
                            break
                    else:
                        ok = False
                        break
                if not ok:
                    continue
                eff = exp
            elif double:
                eff = None
                for l2, r2 in rules:
                    if all(a == b for a, b in zip(rhs, l2)):
                        eff = list(r2)
                        break
                if eff is None:
                    continue
            else:
                eff = list(rhs)
            res.extend(eff)
            i += L
            break
        else:
            return None
    return res


def _cursor_route(plan, cursor, target, ncells):
    """Append the shorter of forward (A4) / backward (A3) cursor moves to reach
    ``target`` on a cyclic track of ``ncells`` cells. Returns the new cursor."""
    fwd = (target - cursor) % ncells
    bwd = (cursor - target) % ncells
    plan.extend([A4] * fwd if fwd <= bwd else [A3] * bwd)
    return target


def _digit_edits(current, target):
    """(action, count) taking digit ``current`` -> ``target`` the short way, or None."""
    if current == target:
        return None
    up = (target - current) % _NDIGITS      # A2 (+1)
    down = (current - target) % _NDIGITS    # A1 (-1)
    return (A2, up) if up <= down else (A1, down)


def solve_normal(game):
    """Plan for a read-only-rules level: cursor to each answer cell that is wrong and
    cycle it to the target digit. The win is checked after every ACTION1/2, so the
    final digit-cycle (of the highest-index edited cell) triggers it once the whole
    prefix matches. Returns a list of GameActions, or None if unsolvable."""
    target = _target_output(game)
    if target is None:
        return None
    out = game.ztgmtnnufb
    ncells = len(out)
    if len(target) > ncells:
        return None
    edits = []
    for i, (tf, td) in enumerate(target):
        cf, cd = _pd(out[i])
        if cf != tf:            # family is fixed (digit cycling only) -> unsolvable
            return None
        e = _digit_edits(cd, td)
        if e is not None:
            edits.append((i, *e))
    if not edits:
        return [A2] * _NDIGITS  # already correct: loop cell 0 back to fire the check
    plan, cursor = [], game.qvtymdcqear_index  # start from the LIVE cursor, not 0
    for i, act, cnt in edits:
        cursor = _cursor_route(plan, cursor, i, ncells)
        plan.extend([act] * cnt)
    return plan


# ── alter-level solver (DFS over the parse -> seed-independent shift vector) ─────
def search_shifts(game):
    """Find a per-set shift vector that makes the fixed input parse to the fixed
    output. DFS assigns shifts lazily along the parse (input is short), fills any
    unconstrained set, and accepts only vectors the pure model confirms. Returns the
    shift vector (length 2*num_rules) or None."""
    rules, inp, out, tree, double = _model(game)
    nsets = 2 * len(rules)
    N, M = len(inp), len(out)

    def expand_match(glyphs, o, shifts):
        # Consume ``glyphs`` (a rule's shifted RHS) as output starting at o.
        if not tree:
            k = len(glyphs)
            if o + k <= M and list(out[o:o + k]) == list(glyphs):
                yield shifts, o + k
            return

        def rec(gi, oo, sh_):
            if gi == len(glyphs):
                yield sh_, oo
                return
            gph = glyphs[gi]
            for r2 in range(len(rules)):
                l2, r2r = rules[r2]
                if l2[0][0] != gph[0]:
                    continue
                need = (gph[1] - l2[0][1]) % _NDIGITS
                li2 = 2 * r2
                if li2 in sh_ and sh_[li2] != need:
                    continue
                ri2 = 2 * r2 + 1
                sr_opts = [sh_[ri2]] if ri2 in sh_ else range(_NDIGITS)
                for sr2 in sr_opts:
                    cexp = _shl(r2r, sr2)
                    k = len(cexp)
                    if oo + k > M or list(out[oo:oo + k]) != list(cexp):
                        continue
                    s2 = dict(sh_)
                    s2[li2] = need
                    s2[ri2] = sr2
                    yield from rec(gi + 1, oo + k, s2)
                break  # greedy: first rule whose LHS family fits; model re-checks
        yield from rec(0, o, shifts)

    def dfs(i, o, shifts):
        if i == N:
            yield dict(shifts)
            return
        for ridx, (lhs, rhs) in enumerate(rules):
            L = len(lhs)
            if i + L > N or any(lhs[k][0] != inp[i + k][0] for k in range(L)):
                continue
            need = {(inp[i + k][1] - lhs[k][1]) % _NDIGITS for k in range(L)}
            if len(need) != 1:
                continue
            sl = need.pop()
            li = 2 * ridx
            if li in shifts and shifts[li] != sl:
                continue
            s1 = dict(shifts)
            s1[li] = sl
            ri = 2 * ridx + 1
            sr_opts = [shifts[ri]] if ri in shifts else range(_NDIGITS)
            for sr in sr_opts:
                s2 = dict(s1)
                s2[ri] = sr
                for s3, newo in expand_match(_shl(rhs, sr), o, s2):
                    yield from dfs(i + L, newo, s3)

    for cand in dfs(0, 0, {}):
        free = [j for j in range(nsets) if j not in cand]
        for combo in itertools.product(range(_NDIGITS), repeat=min(len(free), 5)):
            sv = tuple(cand.get(j, 0) for j in range(nsets))
            if free:
                sv = list(sv)
                for jf, v in zip(free, combo):
                    sv[jf] = v
                sv = tuple(sv)
            if parse_ok(rules, sv, inp, out, tree, double):
                return sv
            if len(free) > 5:
                break
    return None


def plan_from_shifts(game, sv):
    """Cursor over the 2*num_rules sets (ACTION3/4) and shift each by ``sv[j]`` with
    ACTION1/2. Returns a list of GameActions."""
    nsets = len(sv)
    edits = [(j, *_shift_edits(sv[j])) for j in range(nsets) if sv[j] % _NDIGITS]
    if not edits:
        return [A2] * _NDIGITS
    plan, cursor = [], game.qvtymdcqear_index  # start from the LIVE cursor, not 0
    for j, act, cnt in edits:
        cursor = _cursor_route(plan, cursor, j, nsets)
        plan.extend([act] * cnt)
    return plan


def _shift_edits(s):
    s %= _NDIGITS
    up, down = s, (_NDIGITS - s) % _NDIGITS  # A2 (+1) s times / A1 (-1) (7-s) times
    return (A2, up) if up <= down else (A1, down)


def solve_alter(game, level_idx):
    """Re-plan from the LIVE rule table. ``search_shifts`` reads the CURRENT set
    digits and returns the shift DELTA needed *from here* -- so a perturbation that
    already shifted some sets is absorbed automatically. (The fresh-state delta is
    seed-independent, but we must NOT cache it as an absolute shift: once a burst has
    partially shifted the rules, the fresh-state vector is stale, so recompute the
    delta against the currently-shifted rules every call. It is a ~0.1 ms DFS.)"""
    sv = search_shifts(game)
    if sv is None:
        return None
    return plan_from_shifts(game, sv)


def solve_level(game, level_idx):
    if game.current_level.get_data("alter_rules"):
        return solve_alter(game, level_idx)
    return solve_normal(game)


# ── engine driving ─────────────────────────────────────────────────────────────
def _screen_action(action, game):
    """Game action -> the SCREEN action to press so that this level's rotation maps it
    back to ``action``. ``step`` applies ``remap_action(pressed, k)``, so pressing the
    plan's action verbatim on a rotated board would move the cursor / cycle in the wrong
    direction. Identity at k=0; non-directional actions pass through unchanged."""
    return inverse_remap_action_full(action, game._rotation_k)


class Tr87Solver(BaseSolver):
    #: Plans in SCREEN space: `_screen_action` converts each move with
    #: ``inverse_remap_action_full`` and that screen action is what gets issued.
    #: So the base must not convert again when recording, and ``drive`` has to
    #: de-rotate before ``_set_action`` (which bypasses the rotation wrapper).
    plans_in_screen_space = True
    game_id = GAME_ID
    step_guard = _STEP_GUARD
    # A digit-cycle / rule-shift puzzle under a per-level action budget whose win
    # check fires mid-plan. ``solve_level`` RE-PLANS from the LIVE state: the normal
    # levels read each answer cell's current digit AND the live cursor index, and the
    # alter levels recompute the shift DELTA against the currently-shifted rules (and
    # likewise route from the live cursor) -- so a perturbation burst that partially
    # cycles a cell or shifts a rule set is absorbed by the next ``solve_from`` rather
    # than needing a RESET back to the initial state.
    supports_recovery = True
    recovery_mode = "replan"

    def make_game(self, seed: int):
        self._seed = seed
        return Tr87(seed=seed)

    def available_actions(self, game) -> list[int]:
        return [1, 2, 3, 4]

    def set_level(self, game, level_idx: int) -> None:
        """Make level ``level_idx`` deterministic in (seed, level): seed ``_rng``
        (jumble+relabel) and install, only for the duration of set_level, a scoped
        seeded ``random.Random`` factory for the palette. The display rotation is
        drawn by AugmentedGame.set_level as ``random_rotation_k(seed, level)``."""
        game._rng = random.Random(f"tr87:{self._seed}:{level_idx}")
        palette_rng = random.Random(f"tr87-colors:{self._seed}:{level_idx}")
        orig_random = _tr87mod.random.Random

        def factory(*a):
            return orig_random(*a) if a else orig_random(palette_rng.random())

        _tr87mod.random.Random = factory
        try:
            game.set_level(level_idx)
        finally:
            _tr87mod.random.Random = orig_random

    def solve_from(self, game, level_idx: int, seed: int):
        """Plan (in game coordinates) for the level ``game`` is on, then convert
        each move to the SCREEN action this level's rotation maps back into it --
        the base records/issues that screen press and the engine inverse-rotates
        it. Returns [] (fail) when unsolvable or the plan exceeds the step budget."""
        plan = solve_level(game, level_idx)
        if plan is None or len(plan) > game.upmkivwyrxz:
            return []
        return [_screen_action(a, game) for a in plan]

    def drive(self, game, action) -> DriveResult:
        """Apply one SCREEN action against the real (rendering) engine. Breaks on
        _next_level OR a terminal _state: the LAST level wins via win() (which never
        sets _next_level), and after a win ``is_action_complete()`` stays False
        forever, so a naive loop would step past the win animation and die.

        ``_set_action`` BYPASSES the rotation wrapper (only ``perform_action``
        de-rotates), so the screen action has to be mapped back to game space here --
        `_to_core` does it. The game used to invert the rotation itself; it no longer
        does, and without this the plan drove the wrong direction at k != 0."""
        self._note_rotation(game)
        ga = _ID_TO_GAMEACTION[self._to_core(action).action_id]
        game._full_reset = False
        game._set_action(ActionInput(id=ga))
        game.step()
        frame = self.render(game)
        if game.yfetxjexviz >= 0:  # win check passed -> colour-remap animation queued
            guard = 0
            while (not game._next_level and game._state == GameState.NOT_FINISHED
                   and guard < self.step_guard):
                game.step()
                guard += 1
            return DriveResult(np.asarray(frame), True, False)
        guard = 0
        while (not game.is_action_complete() and not game._next_level
               and game._state == GameState.NOT_FINISHED and guard < self.step_guard):
            game.step()
            frame = self.render(game)
            guard += 1
        return DriveResult(np.asarray(frame), False,
                           game._state == GameState.GAME_OVER)


if __name__ == "__main__":
    sys.exit(Tr87Solver.main())
