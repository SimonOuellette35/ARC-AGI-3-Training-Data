"""Generate Phase-1 training data for the PuzzleScript game ps:sokobaiogenesis
("Sokobaiogenesis" by Bagenzo).

The harness -- the rotation contract, the trajectory recorder, the RESET-recovery
prefix and the BaseSolver plumbing -- lives in `solvers/common/ps_astar.py`,
shared with the other ps: generators. This file is the game-specific part: the
mechanic notes, the native model of the turn, the two solvers, and the
optimal-action labeller.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_sokobaiogenesis",
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
presented view, so replaying the recorded actions reproduces the recorded frames
exactly.

The game
--------
A sokoban with no crates in it. Every level is two rooms joined by a column of
DNA: a LAB on the left where the player walks and shoves crates onto targets,
and a VOID on the right holding loose NUCLEOTIDES (C, G, T, A). Crates do not
exist until you make one, and you make one by parking two complementary bases
next to each other -- C beside G becomes a CG crate on the Lab tile, T beside A
becomes a TA crate. The win is the usual ``All <colour>Target on <matching
crate>``, plus ``Some Player``.

What makes it a puzzle is how the bases are steered:

  * **The player's COLOUR is a remote control.** ``[moving PlayerRed]
    [stationary CNucleo] -> [... moving CNucleo]`` and its three siblings mean
    every nucleotide matching the player's colour takes the SAME step the player
    does, on the same press, across the DNA wall. Red drives C, Blue G, Yellow
    T, Green A. There is no other way to move a base while big.
  * **so the puzzle is DE-SYNCING.** Two bases that both move with every press
    keep their offset forever; they only close on each other when one of them is
    clipped -- by the board edge, a VoidWall, another base -- while the other
    keeps going. Every level is a shape built to make exactly the needed
    clipping possible.
  * **``require_player_movement`` means there is no wait.** If no player ends the
    turn on a new cell the WHOLE turn is reverted, nucleotides included. Pressing
    into a wall is an exact no-op, and so is ACTION (no rule reads the action
    force) -- `selfcheck` measures both against the interpreter rather than
    reading them off the rules section.
  * **The eyes repaint you, once each.** ``late [PlayerAlive EyeYellow] ->
    [PlayerYellow]`` drops the Eye from the RHS, so an eye is a single-use
    colour change, and the repaint happens in the late pass AFTER that press's
    nucleotide movement.
  * **Shrink / Grow are the other way in.** Walking onto a Shrink tile in the DNA
    column replaces the player with a tiny one on the Void cell beside it, where
    it pushes bases by hand, one at a time, sokoban-style, and drives none of
    them by colour. Walking a tiny player onto the Grow tile puts it back in the
    lab as PlayerBlue on the Growdestin cell. A BIG player that steps on Grow is
    bounced straight back to Growdestin, so the tiny room has exactly one door.
  * **The Splitter is a two-way exchange**, and it only takes guanine: a G that
    steps onto it becomes a G crate on the lab side, and a CG or G crate shoved
    onto it becomes a G nucleotide in the void. It is how the two G-target levels
    are fed. ``[> NonGuanine|Splitter]`` and ``[> Player|Splitter]`` refuse
    everything else.
  * **Frankenstein.** Pair C-G while a TA crate is sitting on the Lab (or T-A
    while a CG crate is) and instead of a crate you get a FRANKENPLAYER on the
    Lab -- a SECOND body that takes every press with you and drives no
    nucleotide at all. Its only job is ``late [FrankenPlayer Lab]
    [EyeGreenClosed] -> [... EyeGreen]``: it opens the green eye, which is the
    only source of PlayerGreen and therefore the only way to move adenine.
  * **Synthesising a crate on top of yourself KILLS you.** The rule places the
    crate on the Lab cell and `_cell_add` keeps one object per collision layer,
    so a player standing on the Lab when a pair completes is simply deleted,
    ``Some Player`` fails, and the level is over. The fuzz hits it 504 times.
  * **ACTION IS A WAIT, not a no-op.** It grants no force and no rule reads it,
    so nothing moves -- but the late pass still runs, and one late rule can only
    fire on a LATER turn than the one that set it up: a Frankenstein needs
    ``[Lab TACrate]`` and the Lab object is gone for the two ``again`` ticks the
    LabAni animation takes, so a pair that completes on the same press the crate
    was made has to wait. `selfcheck` found that; assuming the rules section
    instead is what cost ps:ouroboros a 26-press plan where a 6-press one
    existed.

Model + search
--------------
`_Board.step` re-implements one turn natively -- the twelve main rules in file
order, the chain force resolution, the twenty-one late rules, then
``require_player_movement``. ``--selfcheck`` drives 349k presses through BOTH
the interpreter and the model across all thirteen levels and compares the whole
state (every player and its kind, all four nucleotide sets, all three crate
sets, every eye) plus the win flag after each one, and prints WHICH mechanics it
exercised. It has three modes and needs all three: uniform play never pairs a
base; a GUIDED mode steers at the rare rules; and a CROWDED mode fills the void
with extra bases so that the interpreter has to CHOOSE which complementary pair
to eat. ``--moves`` covers with purpose-built boards the one refusal rule no
level can reach.

**The one thing the model cannot compute is which pair that is.** When two
complementary pairs are adjacent in the same expansion direction, the
interpreter takes the first one `_candidate_positions` iterates -- a ``set`` of
``(row, col)`` tuples copied out of a position index that `_cell_add` /
`_cell_discard` maintain INCREMENTALLY across the whole episode, so its order is
a fact about the mutation history and not about the board. `_Board._pair`
reproduces the set and takes its first match, which is right the large majority
of the time and flags `amb` when the choice was real; `solve_level` then
CERTIFIES every plan by replaying it on the interpreter and, on a disagreement,
pins the engine's own answer into ``board.override`` and re-searches. So a plan
is never shipped on a transition the interpreter disagrees with, and an optimal
set never offers a sibling press whose outcome was a guess.

The model runs at ~35k presses/s against the interpreter's ~1.7k, which is what
makes the exact tier below affordable at all.

Three tiers, strongest first (`_solve_once`):

  * **EXACT** -- a layered forward BFS from the level start, stopped at the end
    of the first layer that contains a win, plus a reverse BFS over the edges it
    collected. That gives distance-to-win on the shortest-path subgraph, so the
    plan is SHORTEST and every optimal-action set is MEASURED
    (``dist(succ) == dist - 1``), not inferred. The exactness argument is the
    usual one: a state at depth ``g`` with true remaining ``h`` has its whole
    shortest route inside the ball whenever ``g + h <= d*``, which every state a
    shortest plan touches satisfies by construction, and a state outside can
    only be OVER-priced because a truncated subgraph omits edges rather than
    inventing them.
  * **BEAM** -- a width-capped breadth-first beam ordered by `_Guide.score`:
    crate-to-target push distances off exact reverse-BFS tables, plus a term for
    how far the void is from producing the next crate (base-pair proximity, the
    walk to whichever eye grants the needed colour, and the tiny player's walk
    to a pushing stance).
  * **MILESTONES** (`staged`) -- for the levels whose plan runs through several
    crate deliveries, the same beam re-aimed at "one more target satisfied",
    else "one more crate on the board", with backtracking over the candidates
    and two soundness tests on each: `_Guide.alive` (is every unmet target still
    servable with the OTHER crates treated as walls) and `dead_end` (is this
    state's whole reachable component closed and win-free).

Beam and milestone plans WIN and are interpreter-verified but are not certified
shortest; ``--plans`` says which tier answered each level.

**Eleven of the thirteen levels solve, nine of them PROVED SHORTEST** (levels
0-3, 5-7, 9, 10 at 16/43/41/42/43/60/27/68/34 presses, exact; levels 4 and 11 at
49 and 141, beam), 564 presses in all. Levels 8 and 12 are in
`SokobaiogenesisSolver.skip_levels` -- see the comment there for why, and note it
is a search limit and not a model one.

Optimal-action targets and recovery
-----------------------------------
`optimal_for` reads the SET the tier computed for that step; no expert step ever
ships unlabelled. On an EXACT level the set is the measured tie set at the live
board; on the others it is proved by EXHIBITION -- an alternative press is
offered only when replaying the plan's own suffix from it still reaches the win.
``epsilon = 0``: recovery data comes from the episode-wide exploration prefix
that opens every level and is undone by ONE RESET back to the level start
(`BaseSolver._reset_prefix`), which is the state the cached plan was solved
from. An epsilon detour would need a re-plan from an arbitrary board, which on a
beam level is a fresh minutes-long search, so it is left off -- the same choice
every push game in this family makes.

The prefix cannot win a level outright here -- it is at most 15 presses and the
shortest level is 16 -- so this game does NOT need the local `_reset_prefix`
override that ps:snakeoban and ps:slippy_penguin do, and no level tapes a run of
unlabelled explore steps.

Rendering
---------
The sprites in data/puzzlescript_games/Sokobaiogenesis.txt were REDRAWN; the
header comment there is the full list, and ``--audit`` is the check. The short
version is that at the cell sizes these boards render at (3px and 4px) an opaque
crate hid the target under it, so the winning board was pixel-identical to any
other, and the player's colour -- which IS the mechanic -- was on a sprite row
cell_px 3 does not sample. Every layer-4 occupant now leaves the top-left 2x2 of
its cell transparent and every ground object paints its identity there.

Augmentation
------------
Per (seed, level) the adapter draws a frame rotation (0..3). ``Sokobaiogenesis``
is NOT in `PuzzleScriptAdapter._FLIP_GAMES`: several of its late rules resolve a
choice by the interpreter's fixed direction order (which adjacent base pair is
consumed, which side of the Splitter the new crate lands on, which Void cell a
shrinking player drops into), so the mechanic is chiral in engine space.
``--symmetry`` measures it rather than arguing it.

Usage (run from the repo root):
    python solvers/generate_sokobaiogenesis_training.py --episodes 200 \
        --out data/training_multi_level/sokobaiogenesis
    python solvers/generate_sokobaiogenesis_training.py --selfcheck
    python solvers/generate_sokobaiogenesis_training.py --moves
    python solvers/generate_sokobaiogenesis_training.py --plans
    python solvers/generate_sokobaiogenesis_training.py --verify
    python solvers/generate_sokobaiogenesis_training.py --audit
    python solvers/generate_sokobaiogenesis_training.py --symmetry
"""

from __future__ import annotations

import itertools
import random
import sys
import numpy as np

from array import array
from collections import Counter, deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.puzzlescript_adapter import (                     # noqa: E402
    PuzzleScriptAdapter, _render_frame)
from solvers.common.ps_astar import (                           # noqa: E402
    PSAStarSolver, PSExpert, Plan, restore, snapshot)

GAME_NAME = "Sokobaiogenesis"

_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
#: The engine expands an undirected rule over its four directions in THIS
#: order (`_DIR_NAMES`), running each to a fixpoint, so a late rule with
#: several possible bindings takes the first one this order finds.
_DIRS = ("up", "down", "left", "right")
#: The presses a search branches on. ACTION carries no force and no rule reads
#: it, so it moves nothing -- but the LATE rules still run on it, and one of
#: them fires only on a turn AFTER the board changed: a Frankenstein is born
#: from ``[Lab TACrate]`` and the Lab object does not come back until the
#: LabAni animation is over, so a pair that completes the same press the crate
#: was made has to wait for the next one. That makes ACTION a real WAIT here,
#: not a no-op, which `selfcheck` measures rather than assumes (ps:ouroboros is
#: the game where assuming it cost a 26-press plan where 6 existed).
_ACTS = _DIRS + ("action",)

# Player kinds.
RED, BLUE, YELLOW, GREEN, FRANKEN, SMALL, FSMALL = range(7)
#: A colour player drags every nucleotide of the matching type; the kind index
#: IS the nucleotide type for the four alive colours.
ALIVE = (RED, BLUE, YELLOW, GREEN)
BIG = (RED, BLUE, YELLOW, GREEN, FRANKEN)

# Nucleotide types.
NC, NG, NT, NA = range(4)
# Crate types.
KCG, KTA, KG = range(3)

_PLAYER_OBJ = {"playerred": RED, "playerblue": BLUE, "playeryellow": YELLOW,
               "playergreen": GREEN, "frankenplayer": FRANKEN,
               "playersmall": SMALL, "frankensmall": FSMALL}
_NUC_OBJ = {"cnucleo": NC, "gnucleo": NG, "tnucleo": NT, "anucleo": NA}
_CRATE_OBJ = {"cgcrate": KCG, "tacrate": KTA, "gcrate": KG}
# Eye kinds: 0 yellow, 1 red, 2 green-closed, 3 green.
EYE_Y, EYE_R, EYE_EC, EYE_E = range(4)
_EYE_OBJ = {"eyeyellow": EYE_Y, "eyered": EYE_R,
            "eyegreenclosed": EYE_EC, "eyegreen": EYE_E}
#: Win-condition targets: object name -> the crate type that satisfies it
#: (None = any crate).
_TARGET_OBJ = {"whitetarget": None, "cgtarget": KCG,
               "tatarget": KTA, "gtarget": KG}

#: Per-mechanic firing counts, filled by `_Board.step` when `TRACE` is on.
#: `selfcheck` reports them so a clean run is a statement about WHAT was
#: exercised, not just about how many presses agreed -- random play here seals
#: itself into the walking half of the game and never pairs a base, and both
#: halves of that were true of the first fuzz written for this file.
COUNTS: Counter = Counter()
TRACE = False


def _hit(name):
    if TRACE:
        COUNTS[name] += 1


# ---------------------------------------------------------------------------
# The level, natively
# ---------------------------------------------------------------------------

class _Board:
    """One level's static geometry plus a native re-implementation of a turn.

    A state is ``(players, nucs, crates, eyes)``:

      * ``players``  sorted tuple of ``(cell, kind)``
      * ``nucs``     four sorted tuples of cells, indexed by nucleotide type
      * ``crates``   three sorted tuples of cells, indexed by crate type
      * ``eyes``     sorted tuple of ``(cell, eye kind)``

    Everything else on the board is static: the Walls, the VoidWalls, the DNA
    columns, the Void, the Lab, the Shrink/Grow/Growdestin/Splitter tiles and
    the Targets. The Lab is momentarily replaced by its LabAni animation when a
    crate is synthesised, but the animation puts it back inside the same press
    (the ``again`` loop), so it is not state -- see `_late`.
    """

    __slots__ = ("h", "w", "n", "blocked", "dna", "void", "lab", "shrink",
                 "grow", "growdest", "splitter", "targets", "start", "_nx",
                 "amb", "override")

    def __init__(self, h, w, blocked, dna, void, lab, shrink, grow, growdest,
                 splitter, targets, start):
        self.h, self.w, self.n = h, w, h * w
        self.blocked = frozenset(blocked)     # static layer-5 obstacles
        self.dna = frozenset(dna)             # the DNA columns, which `_first`
        #                                       (and only `_first`) singles out
        self.void = frozenset(void)
        self.lab = lab                        # cell or None
        self.shrink = frozenset(shrink)
        self.grow = frozenset(grow)
        self.growdest = frozenset(growdest)
        self.splitter = frozenset(splitter)
        self.targets = dict(targets)          # cell -> crate type or None
        self.start = start
        #: Set by the LAST `step`: True when the interpreter had a genuine
        #: CHOICE of complementary pair to consume and the model guessed. See
        #: `_pair`.
        self.amb = False
        #: ``(packed state, press) -> packed successor``, pinned from the
        #: INTERPRETER by `certify` whenever a replay caught the guess above
        #: being wrong. `solve_level` re-searches after each pin, so a plan is
        #: never shipped on a transition the interpreter disagrees with.
        self.override: dict = {}

        # ``_nx[d][cell]`` is the cell one step along d, or -1 off the board.
        self._nx = {}
        for d, (dr, dc) in _DELTA.items():
            table = [-1] * self.n
            for r in range(h):
                for c in range(w):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        table[r * w + c] = nr * w + nc
            self._nx[d] = table

    # -- win ------------------------------------------------------------------
    def won(self, st) -> bool:
        """``All <colour>Target on <matching crate>`` for the four target
        classes, plus ``Some Player``."""
        players, nucs, crates, _eyes = st
        if not players:
            return False
        if not self.targets:
            return True
        by_cell = {}
        for t in range(3):
            for cell in crates[t]:
                by_cell[cell] = t
        for cell, want in self.targets.items():
            got = by_cell.get(cell)
            if got is None:
                return False
            if want is not None and got != want:
                return False
        return True

    # -- one turn -------------------------------------------------------------
    def step(self, st, d: str):
        """One press, natively. Returns the settled state.

        Mirrors the interpreter's turn: assign the pressed force to every
        Player, run the main rules in file order (which is what cancels a force
        or grants one to a crate / a nucleotide), resolve the forces as chains,
        run the late rules, then apply ``require_player_movement`` -- which
        reverts the WHOLE turn, nucleotides included, when no player ended up
        on a new cell.
        """
        self.amb = False
        if self.override:
            pinned = self.override.get((pack(st), d))
            if pinned is not None:
                return unpack(pinned)
        players, nucs, crates, eyes = st
        plist = list(players)
        nsets = [set(x) for x in nucs]
        csets = [set(x) for x in crates]

        if d == "action":
            # No force is granted and no rule reads the action, so nothing
            # MOVES -- but the late pass still runs, and that is not nothing.
            st2 = self._late((plist, nsets, csets, dict(eyes)))
            if {c for c, _k in st2[0]} == {c for c, _k in players}:
                _hit("turn_reverted")
                return st
            return self._freeze(st2)

        nx = self._nx[d]

        pcell = {cell: i for i, (cell, _k) in enumerate(plist)}
        ncell = {}
        for t in range(4):
            for cell in nsets[t]:
                ncell[cell] = t
        ccell = {}
        for t in range(3):
            for cell in csets[t]:
                ccell[cell] = t

        pf = [True] * len(plist)
        nforce: set = set()
        cforce: set = set()

        # M1  [ > Player | Player ] -> [ Player | Player ]
        for i, (cell, _k) in enumerate(plist):
            n = nx[cell]
            if n >= 0 and n in pcell:
                pf[i] = False
                _hit("m1_player_bump")
        # M2  [> Player|Crate|Tech] -> cancel   (a crate may not be shoved onto
        #     a Shrink or a Grow tile)
        for i, (cell, _k) in enumerate(plist):
            if not pf[i]:
                continue
            n = nx[cell]
            if n >= 0 and n in ccell:
                n2 = nx[n]
                if n2 >= 0 and (n2 in self.shrink or n2 in self.grow):
                    pf[i] = False
                    _hit("m2_crate_onto_tech_refused")
        # M3  [> Player|TACrate|Splitter] -> cancel
        for i, (cell, _k) in enumerate(plist):
            if not pf[i]:
                continue
            n = nx[cell]
            if n >= 0 and ccell.get(n) == KTA:
                n2 = nx[n]
                if n2 >= 0 and n2 in self.splitter:
                    pf[i] = False
                    _hit("m3_tacrate_onto_splitter_refused")
        # M4  [> Player | Crate] -> push
        # M5  [> Player | Nucleo] -> push
        for i, (cell, _k) in enumerate(plist):
            if not pf[i]:
                continue
            n = nx[cell]
            if n < 0:
                continue
            if n in ccell:
                cforce.add(n)
                _hit("m4_push_crate")
            if n in ncell:
                nforce.add(n)
                _hit("m5_push_nucleo")
        # M6-M9  [moving Player<Colour>] [stationary <Nucleo>] -> drag them all
        for i, (cell, k) in enumerate(plist):
            if pf[i] and k < 4:
                if nsets[k]:
                    _hit("m6_telekinesis")
                nforce |= nsets[k]
        # M10  [ > Nucleo | Tech] -> cancel  (nucleotides cannot leave the void)
        for cell in list(nforce):
            n = nx[cell]
            if n >= 0 and (n in self.shrink or n in self.grow):
                nforce.discard(cell)
                _hit("m10_nucleo_onto_tech_refused")
        # M11  [> NonGuanine | Splitter] -> cancel
        for cell in list(nforce):
            if ncell[cell] != NG:
                n = nx[cell]
                if n >= 0 and n in self.splitter:
                    nforce.discard(cell)
                    _hit("m11_nonguanine_onto_splitter_refused")
        # M12  [> Player | Splitter] -> cancel
        for i, (cell, _k) in enumerate(plist):
            if pf[i]:
                n = nx[cell]
                if n >= 0 and n in self.splitter:
                    pf[i] = False
                    _hit("m12_player_onto_splitter_refused")

        # -- resolve: every force points the same way, so the movers are
        # chains and a chain moves only if the cell past its head is free.
        occupied = set(self.blocked)
        occupied |= set(pcell)
        occupied |= set(ncell)
        occupied |= set(ccell)
        forced = set(nforce) | set(cforce)
        for i, (cell, _k) in enumerate(plist):
            if pf[i]:
                forced.add(cell)

        verdict: dict = {}

        def can_move(cell):
            got = verdict.get(cell)
            if got is not None:
                return got
            verdict[cell] = False              # guard (chains cannot cycle)
            n = nx[cell]
            if n < 0:
                out = False
            elif n not in occupied:
                out = True
            elif n in forced:
                out = can_move(n)
            else:
                out = False
            verdict[cell] = out
            return out

        movers = {cell for cell in forced if can_move(cell)}

        if movers:
            plist = [((nx[cell] if (pf[i] and cell in movers) else cell), k)
                     for i, (cell, k) in enumerate(plist)]
            for t in range(4):
                nsets[t] = {(nx[c] if c in movers else c) for c in nsets[t]}
            for t in range(3):
                csets[t] = {(nx[c] if c in movers else c) for c in csets[t]}

        st2 = self._late((plist, nsets, csets, dict(eyes)))

        # require_player_movement: the SET of player cells must have changed.
        if {c for c, _k in st2[0]} == {c for c, _k in players}:
            _hit("turn_reverted")
            return st
        if not st2[0]:
            _hit("player_destroyed")
        return self._freeze(st2)

    # -- late rules -----------------------------------------------------------
    def _late(self, st):
        """The late-rule pass, in file order. ``st`` is mutable
        ``(list players, list of sets, list of sets, dict eyes)``."""
        plist, nsets, csets, eyes = st
        lab = self.lab

        crate_at_lab = None
        if lab is not None:
            for t in range(3):
                if lab in csets[t]:
                    crate_at_lab = t
                    break

        # L1-L4: base pairing at the Lab. Exactly one of the four can fire in a
        # pass -- L1/L2 consume the Lab object (it comes back through the
        # LabAni animation inside the same press) and L3/L4 consume the crate
        # the pass needed to match.
        if lab is not None:
            fired = False
            for a, b, made in ((NC, NG, KCG), (NT, NA, KTA)):
                if crate_at_lab is None and self._pair(nsets, a, b):
                    _hit("synth_cg" if made == KCG else "synth_ta")
                    if any(e is not None and e[0] == lab for e in plist):
                        _hit("synth_killed_the_player")
                    self._evict(plist, nsets, csets, lab)
                    csets[made].add(lab)
                    crate_at_lab = made
                    fired = True
                    break
            if not fired:
                for a, b, need in ((NC, NG, KTA), (NT, NA, KCG)):
                    if crate_at_lab == need and self._pair(nsets, a, b):
                        _hit("franken_born")
                        csets[need].discard(lab)
                        crate_at_lab = None
                        self._evict(plist, nsets, csets, lab)
                        plist.append((lab, FRANKEN))
                        break

        # L5-L7: an eye repaints the player that steps on it, and is CONSUMED.
        for want, kind in ((EYE_Y, YELLOW), (EYE_R, RED), (EYE_E, GREEN)):
            for i, ent in enumerate(plist):
                if ent is None:
                    continue
                cell, k = ent
                if k in ALIVE and eyes.get(cell) == want:
                    _hit(f"eye_{want}")
                    plist[i] = (cell, kind)
                    del eyes[cell]
        # L8: a FrankenPlayer standing in the Lab opens every closed green eye.
        if any(e is not None and e[1] == FRANKEN and e[0] == lab for e in plist):
            for cell in [c for c, e in eyes.items() if e == EYE_EC]:
                _hit("green_eye_opened")
                eyes[cell] = EYE_E

        # L9-L11: the Shrink tiles. L9 keeps a FrankenSmall small, L10 shrinks a
        # FrankenPlayer, L11 shrinks everything else -- all into the first Void
        # neighbour in `_DIRS` order.
        for match, become in ((FSMALL, FSMALL), (FRANKEN, FSMALL), (None, SMALL)):
            for i, ent in enumerate(plist):
                if ent is None:
                    continue
                cell, k = ent
                if cell not in self.shrink:
                    continue
                if match is None:
                    if k in (FSMALL,):          # already handled by L9/L10
                        continue
                elif k != match:
                    continue
                dest = self._first(cell, self.void)
                if dest is not None:
                    _hit(f"shrink_{k}_to_{become}")
                    self._evict(plist, nsets, csets, dest)
                    plist[i] = (dest, become)
        # L12-L14: the Grow tiles.
        for match, become in ((FSMALL, FRANKEN), (SMALL, BLUE)):
            for i, ent in enumerate(plist):
                if ent is None:
                    continue
                cell, k = ent
                if k == match and cell in self.grow:
                    dest = self._grow_dest(cell)
                    if dest is not None:
                        _hit(f"grow_{k}_to_{become}")
                        self._evict(plist, nsets, csets, dest)
                        plist[i] = (dest, become)
        if self.growdest:
            dest = min(self.growdest)
            for i, ent in enumerate(plist):
                if ent is None:
                    continue
                cell, k = ent
                if k in BIG and cell in self.grow:
                    _hit("big_bounced_off_grow")
                    self._evict(plist, nsets, csets, dest)
                    plist[i] = (dest, k)

        # L15-L17: the Splitter.
        for t in (KCG, KG):
            for cell in list(csets[t]):
                if cell in self.splitter:
                    dest = self._first(cell, self.void)
                    if dest is not None:
                        _hit("splitter_crate_to_guanine")
                        csets[t].discard(cell)
                        self._evict(plist, nsets, csets, dest)
                        nsets[NG].add(dest)
        for cell in list(nsets[NG]):
            if cell in self.splitter:
                dest = self._first(cell, None)   # any neighbour, no DNA wall
                if dest is not None:
                    _hit("splitter_guanine_to_crate")
                    nsets[NG].discard(cell)
                    self._evict(plist, nsets, csets, dest)
                    csets[KG].add(dest)
        return ([e for e in plist if e is not None], nsets, csets, eyes)

    # -- late-rule helpers ----------------------------------------------------
    @staticmethod
    def _evict(plist, nsets, csets, cell):
        """Clear ``cell`` on the Player/Wall/Crate/Nucleo collision layer.

        A late rule that PLACES an object there does exactly this in the
        interpreter -- `_cell_add` keeps one object per layer per cell -- and
        the case it exists for is lethal: synthesising a crate while standing on
        the Lab DELETES the player, which fails ``Some Player`` and ends the
        level's chances. `dead` treats a playerless board as lost.

        Evicted players are blanked rather than removed, because every caller is
        part-way through ``enumerate(plist)`` and a deletion would renumber the
        entry it is about to write back -- which is exactly the IndexError this
        replaced. `_late` drops the blanks on its way out.
        """
        for i, ent in enumerate(plist):
            if ent is not None and ent[0] == cell:
                plist[i] = None      # blanked, not deleted: `_late` is iterating
        for s in nsets:
            s.discard(cell)
        for s in csets:
            s.discard(cell)

    def _pair(self, nsets, a, b) -> bool:
        """Consume one adjacent ``a``-``b`` pair, and FLAG the turn when there
        was more than one to choose from.

        WHICH PAIR IS NOT A DETAIL. A crowded void routinely has two
        complementary pairs adjacent at once and only one crate comes out, so
        picking the other one leaves a different board -- level 9's shortest
        plan diverged from the interpreter at press 11 over exactly this.

        And the interpreter's choice is NOT reproducible from the board.
        `_execute_late_single` walks the rule's four expanded directions in
        ``up, down, left, right`` order and `_apply_multi_group_late_rule`
        anchors on `_candidate_positions`, which copies
        ``PSEngine._object_positions[CNucleo]`` -- a ``set`` of ``(row, col)``
        tuples that is maintained INCREMENTALLY by `_cell_add` / `_cell_discard`
        across the whole episode. Its iteration order therefore depends on the
        mutation history, not on the contents, and no model can mirror that
        without re-implementing CPython's set.

        So the model REBUILDS the same set of ``(row, col)`` tuples and takes
        its first match in ``up, down, left, right`` order -- which is exactly
        right whenever only one pair matches in the first matching direction
        (the common case) and measures ~96% right when several do -- and sets
        `amb` when the choice was real. Three things then hang off that flag,
        and between them nothing is ever shipped on a guess:

          * `solve_level` CERTIFIES each finished plan on the interpreter and
            pins the engine's own successor into `override` on a disagreement;
          * `_Level.answer` drops an ambiguous SIBLING from an optimal set, so a
            label only ever offers the plan's own certified press;
          * `selfcheck` re-syncs from the interpreter on such a press and counts
            it, rather than reporting a mismatch it cannot fix.

        Dropping the successor outright was tried first and is worse: level 7's
        START state has four complementary pairs, so it would have no plan at
        all.
        """
        w = self.w
        hit = None
        found = 0
        for d in _DIRS:
            nx = self._nx[d]
            anchors = {divmod(cell, w): cell for cell in nsets[a]}
            for rc in set(anchors):          # a fresh set of (row, col) TUPLES
                cell = anchors[rc]
                n = nx[cell]
                if n >= 0 and n in nsets[b]:
                    found += 1
                    if hit is None:
                        hit = (cell, n)
            if hit is not None:
                break
        if hit is None:
            return False
        if found > 1:
            self.amb = True
            _hit("ambiguous_pair")
        nsets[a].discard(hit[0])
        nsets[b].discard(hit[1])
        return True

    def _first(self, cell, want):
        """The first neighbour of ``cell`` in `_DIRS` order that is in ``want``.

        ``want=None`` is the Splitter rule's ``| Background no DNAWall``: every
        cell in a PuzzleScript level carries Background, so that pattern means
        exactly "not a DNA wall" and nothing more -- a Wall or a VoidWall
        neighbour would match it too. `build_board` asserts no shipped Splitter
        has one, so the crate always lands on open lab floor."""
        for d in _DIRS:
            n = self._nx[d][cell]
            if n < 0:
                continue
            if want is None:
                if n not in self.dna:
                    return n
            elif n in want:
                return n
        return None

    def _grow_dest(self, cell):
        """``[Growdestin|Grow]``: a Growdestin cell with this Grow beside it."""
        for d in _DIRS:
            for g in sorted(self.growdest):
                if self._nx[d][g] == cell:
                    return g
        return None

    # -- state plumbing -------------------------------------------------------
    @staticmethod
    def _freeze(st):
        plist, nsets, csets, eyes = st
        return (tuple(sorted(plist)),
                tuple(tuple(sorted(s)) for s in nsets),
                tuple(tuple(sorted(s)) for s in csets),
                tuple(sorted(eyes.items())))


# ---------------------------------------------------------------------------
# Reading the interpreter
# ---------------------------------------------------------------------------

def read_state(g, eng):
    """The model state for the engine's current grid."""
    idx = g.obj_name_to_idx
    w = eng.width
    players, nucs, crates, eyes = [], [set() for _ in range(4)], \
        [set() for _ in range(3)], {}
    inv = {}
    for name, kind in _PLAYER_OBJ.items():
        inv[idx[name]] = ("p", kind)
    for name, t in _NUC_OBJ.items():
        inv[idx[name]] = ("n", t)
    for name, t in _CRATE_OBJ.items():
        inv[idx[name]] = ("c", t)
    for name, t in _EYE_OBJ.items():
        if name in idx:
            inv[idx[name]] = ("e", t)
    for r, row in enumerate(eng.grid):
        for c, cell in enumerate(row):
            for o in cell:
                got = inv.get(o)
                if got is None:
                    continue
                what, t = got
                pos = r * w + c
                if what == "p":
                    players.append((pos, t))
                elif what == "n":
                    nucs[t].add(pos)
                elif what == "c":
                    crates[t].add(pos)
                else:
                    eyes[pos] = t
    return _Board._freeze((players, nucs, crates, eyes))


def build_board(g, eng) -> _Board:
    """The level's static geometry, read off the engine's current grid."""
    idx = g.obj_name_to_idx
    h, w = eng.height, eng.width
    blocked_ids = {idx[n] for n in ("wall", "voidwall", "dnarb", "dnabr")}
    dna_ids = {idx[n] for n in ("dnarb", "dnabr")}
    blocked, dna, void, shrink, grow, growdest, splitter, targets = \
        set(), set(), set(), set(), set(), set(), set(), {}
    lab = None
    for r, row in enumerate(eng.grid):
        for c, cell in enumerate(row):
            pos = r * w + c
            if cell & blocked_ids:
                blocked.add(pos)
            if cell & dna_ids:
                dna.add(pos)
            if idx["void"] in cell:
                void.add(pos)
            if idx["growdestin"] in cell:
                growdest.add(pos)
            if idx["lab"] in cell:
                lab = pos
            if idx["shrink"] in cell:
                shrink.add(pos)
            if idx["grow"] in cell:
                grow.add(pos)
            if idx["splitter"] in cell:
                splitter.add(pos)
            for name, t in _TARGET_OBJ.items():
                if idx[name] in cell:
                    targets[pos] = t
    return _Board(h, w, blocked, dna, void, lab, shrink, grow, growdest,
                  splitter, targets, read_state(g, eng))


def check_geometry(board: _Board) -> None:
    """Assert the three places where a late rule picks a NEIGHBOUR by direction
    order have only one candidate to pick, so the choice is never a guess.

    `_Board._first` mirrors ``| Background no DNAWall`` faithfully, and that
    pattern would happily choose a Wall cell for the Splitter's output; the
    Shrink and Grow rules likewise take the first Void / Growdestin side they
    find. Every shipped level has exactly one candidate for each, which is what
    keeps the model exact -- so it is asserted rather than assumed. (The
    purpose-built `--moves` boards are not held to it: they exist to exercise a
    rule, not to be shippable levels.)
    """
    w = board.w
    for sp in board.splitter:
        dest = board._first(sp, None)
        assert dest is None or dest not in board.blocked, \
            f"the Splitter at {divmod(sp, w)} would deliver onto a wall"
    for cell in board.shrink:
        n = sum(1 for d in _DIRS
                if board._nx[d][cell] >= 0 and board._nx[d][cell] in board.void)
        assert n == 1, f"the Shrink at {divmod(cell, w)} has {n} Void neighbours"
    for cell in board.grow:
        n = sum(1 for d in _DIRS if board._nx[d][cell] >= 0
                and board._nx[d][cell] in board.growdest)
        assert n == 1, f"the Grow at {divmod(cell, w)} has {n} Growdestin sides"
    return board


# ---------------------------------------------------------------------------
# State packing (the exact tier keeps one key per state, so it has to be small)
# ---------------------------------------------------------------------------

def pack(st) -> bytes:
    """A state as ~30 canonical bytes.

    Every cell id fits in one byte (the widest board is 9x17 = 153 cells) and
    every kind/type is a small int, so the whole board is one short ``bytes`` --
    which is also the exact-tier's dict key. Storing the tuple form instead
    costs roughly ten Python objects per state and puts the 1.4M-state levels
    into gigabytes; this keeps them in a few hundred megabytes."""
    players, nucs, crates, eyes = st
    out = bytearray()
    out.append(len(players))
    for cell, kind in players:
        out.append(cell)
        out.append(kind)
    for group in (nucs, crates):
        for cells in group:
            out.append(len(cells))
            out.extend(cells)
    out.append(len(eyes))
    for cell, kind in eyes:
        out.append(cell)
        out.append(kind)
    return bytes(out)


def unpack(key: bytes):
    """`pack` inverted."""
    i = 0
    n = key[i]; i += 1
    players = tuple((key[i + 2 * j], key[i + 2 * j + 1]) for j in range(n))
    i += 2 * n
    groups = []
    for count in (4, 3):
        got = []
        for _ in range(count):
            m = key[i]; i += 1
            got.append(tuple(key[i:i + m]))
            i += m
        groups.append(tuple(got))
    n = key[i]; i += 1
    eyes = tuple((key[i + 2 * j], key[i + 2 * j + 1]) for j in range(n))
    return (players, groups[0], groups[1], eyes)


# ---------------------------------------------------------------------------
# Tier 1: the exact distance-to-win field over the shortest-path ball
# ---------------------------------------------------------------------------

class _Field:
    """Exact distance-to-win on the subgraph a shortest plan can live in.

    Built by a layered forward BFS from the level start, stopped at the END of
    the first layer that contains a win, keeping every edge it traversed; then
    one reverse BFS from the winning states over those edges.

    WHY THAT IS EXACT (and not just "a plan that happens to win"): a state at
    depth ``g`` whose true remaining distance is ``h`` has its entire shortest
    route inside the ball whenever ``g + h <= d*``, and every state a shortest
    plan passes through satisfies that by construction. A state the truncation
    leaves half-explored can only be priced too HIGH -- a missing edge never
    invents a shortcut -- so `plan` is shortest and `optimal` is the exact tie
    set, both from any state the ball contains. States it does not contain
    answer None, which is what makes an epsilon detour that wanders off the
    shortest-path subgraph a rejected detour rather than a wrong label.
    """

    __slots__ = ("index", "succ", "dist", "dstar", "states")

    def __init__(self, index, succ, dist, dstar, states):
        self.index = index          # packed key -> id
        self.succ = succ            # array('i'), 4 per state, -1 = terminal
        self.dist = dist            # array('i'), -1 = cannot reach a win here
        self.dstar = dstar
        self.states = states        # id -> packed key

    # -- reading it -----------------------------------------------------------
    def distance(self, st):
        j = self.index.get(pack(st))
        if j is None or self.dist[j] < 0:
            return None
        return self.dist[j]

    def plan(self, st):
        """A SHORTEST press sequence from ``st``, or None.

        Descends the field taking the first tied press in `_ACTS` order, so the
        plan is a pure function of the board."""
        j = self.index.get(pack(st))
        if j is None or self.dist[j] < 0:
            return None
        out = []
        while self.dist[j] > 0:
            row = self.succ
            for k in range(5):
                nxt = row[5 * j + k]
                if nxt >= 0 and self.dist[nxt] == self.dist[j] - 1:
                    out.append(_ACTS[k])
                    j = nxt
                    break
            else:                                        # pragma: no cover
                raise AssertionError("the field has no descent")
        return out

    def optimal(self, st):
        """Every press that keeps the game on a SHORTEST route to the win."""
        j = self.index.get(pack(st))
        if j is None or self.dist[j] <= 0:
            return []
        here = self.dist[j]
        return [_ACTS[k] for k in range(5)
                if self.succ[5 * j + k] >= 0
                and self.dist[self.succ[5 * j + k]] == here - 1]


def build_field(board: _Board, cap: int = 1_600_000) -> "_Field | None":
    """The exact field for ``board``, or None when the ball exceeds ``cap``.

    Winning states are TERMINAL and are not expanded: the adapter ends the level
    there, so their outgoing presses are not part of the game.
    """
    start = board.start
    key = pack(start)
    index = {key: 0}
    states = [key]
    succ = array("i")
    layer = [0]
    depth = 0
    winners: list[int] = []
    while layer:
        for j in layer:
            if board.won(unpack(states[j])):
                winners.append(j)
        if winners:
            break
        if len(index) > cap:
            return None
        nxt: list[int] = []
        for j in layer:
            st = unpack(states[j])
            base = len(succ)
            succ.extend((-1, -1, -1, -1, -1))
            for k, d in enumerate(_ACTS):
                nk = pack(board.step(st, d))
                i = index.get(nk)
                if i is None:
                    i = len(states)
                    index[nk] = i
                    states.append(nk)
                    nxt.append(i)
                succ[base + k] = i
            assert base == 5 * j
        depth += 1
        layer = nxt
    if not winners:
        return None
    n = len(states)
    while len(succ) < 5 * n:
        succ.extend((-1, -1, -1, -1, -1))

    # Reverse BFS over the collected edges, through a CSR predecessor map -- a
    # list-of-lists here is ~100 MB of empty lists before it holds anything.
    counts = array("i", bytes(4 * (n + 1)))
    for e in succ:
        if e >= 0:
            counts[e + 1] += 1
    for i in range(n):
        counts[i + 1] += counts[i]
    fill = array("i", counts[:n])
    pred = array("i", bytes(4 * counts[n]))
    for j in range(n):
        for k in range(5):
            e = succ[5 * j + k]
            if e >= 0:
                pred[fill[e]] = j
                fill[e] += 1

    dist = array("i", [-1]) * n
    queue = deque()
    for j in winners:
        dist[j] = 0
        queue.append(j)
    while queue:
        j = queue.popleft()
        for p in range(counts[j], counts[j + 1]):
            src = pred[p]
            if dist[src] < 0:
                dist[src] = dist[j] + 1
                queue.append(src)
    return _Field(index, succ, dist, dist[0], states)


# ---------------------------------------------------------------------------
# Tier 2: a progress-ordered beam over the same model
# ---------------------------------------------------------------------------

INF = 1 << 30
_OPP = {"up": "down", "down": "up", "left": "right", "right": "left"}
#: Which two bases a crate type is made of.
_RECIPE = {KCG: (NC, NG), KTA: (NT, NA)}
#: Which player colour drives which base.
_DRIVER = {NC: RED, NG: BLUE, NT: YELLOW, NA: GREEN}
_EYE_GRANTS = {EYE_Y: YELLOW, EYE_R: RED, EYE_E: GREEN}


class _Guide:
    """The distance tables the beam is ordered by. Static per level.

    Everything here is an exact reverse BFS on the level's static geometry --
    what varies is which of them the score adds up.
    """

    def __init__(self, board: _Board):
        self.b = board
        n = board.n
        # A crate may sit on any lab-side cell that is not a wall, a Tech tile
        # (`[> Player|Crate|Tech]` refuses the shove) or a Splitter (which eats
        # it), and never in the void.
        self.crate_ok = [(c not in board.blocked and c not in board.shrink
                          and c not in board.grow and c not in board.splitter
                          and c not in board.void) for c in range(n)]
        self.stand_ok = [c not in board.blocked for c in range(n)]
        self.void_ok = [c in board.void for c in range(n)]
        self.pushd = {t: self._push_table(t) for t in board.targets}
        self.vd = {c: self._flood(c, self.void_ok)
                   for c in range(n) if self.void_ok[c]}
        self._walk: dict = {}

    def _push_table(self, target):
        """Presses-to-target for a crate at each cell: reverse BFS over
        ``crate at p, player behind it`` moves, static geometry only."""
        b = self.b
        d = [INF] * b.n
        if not self.crate_ok[target]:
            return d
        d[target] = 0
        q = deque([target])
        while q:
            c = q.popleft()
            for dd in _DIRS:
                back = b._nx[_OPP[dd]]
                p = back[c]
                if p < 0 or not self.crate_ok[p]:
                    continue
                s = back[p]
                if s < 0 or not self.stand_ok[s]:
                    continue
                if d[p] > d[c] + 1:
                    d[p] = d[c] + 1
                    q.append(p)
        return d

    def _flood(self, src, ok):
        d = {src: 0}
        q = deque([src])
        while q:
            c = q.popleft()
            for dd in _DIRS:
                nb = self.b._nx[dd][c]
                if nb >= 0 and ok[nb] and nb not in d:
                    d[nb] = d[c] + 1
                    q.append(nb)
        return d

    def walk(self, src):
        got = self._walk.get(src)
        if got is None:
            got = self._walk[src] = self._flood(src, self.stand_ok)
        return got

    # -- the terms ------------------------------------------------------------
    def drive(self, st, ntype, pw):
        """The walk the party still owes before it can move bases of ``ntype``:
        zero when it already wears that colour or is small (a tiny player shoves
        anything by hand), else the way to the eye that grants the colour, or to
        a Shrink tile."""
        players, _nucs, _crates, eyes = st
        want = _DRIVER[ntype]
        if any(k == want for _c, k in players):
            return 0
        if any(k in (SMALL, FSMALL) for _c, k in players):
            return 0
        best = INF
        for cell, kind in eyes:
            if _EYE_GRANTS.get(kind) == want:
                best = min(best, pw.get(cell, INF))
        for cell in self.b.shrink:
            best = min(best, pw.get(cell, INF))
        return best if best < INF else 60

    def stance(self, st, cells):
        """The void walk a TINY player still owes to get behind one of
        ``cells``. Zero while big -- telekinesis needs no stance, which is why
        this term only ever shows up in the shrunken half of a level."""
        players = st[0]
        tiny = [c for c, k in players if k in (SMALL, FSMALL)]
        if not tiny or not cells:
            return 0
        best = INF
        for p in tiny:
            d = self.vd.get(p)
            if d is None:
                continue
            for x in cells:
                for dd in _DIRS:
                    nb = self.b._nx[dd][x]
                    if nb >= 0:
                        best = min(best, d.get(nb, INF))
        return best if best < INF else 30

    def recipe_cost(self, st, t, pw):
        """How far the void is from producing one crate of type ``t``."""
        _players, nucs, crates, _eyes = st
        b = self.b
        if t == KG:
            if not b.splitter:
                return INF
            best = INF
            for x in nucs[NG]:
                for sp in b.splitter:
                    for dd in _DIRS:
                        nb = b._nx[dd][sp]
                        if nb >= 0 and self.void_ok[nb]:
                            best = min(best, self.vd.get(x, {}).get(nb, INF))
            if best == INF:
                return INF
            return (6 * best + 3 * self.drive(st, NG, pw)
                    + 3 * self.stance(st, nucs[NG]))
        a, c = _RECIPE[t]
        pair = INF
        for x in nucs[a]:
            for y in nucs[c]:
                pair = min(pair, self.vd.get(x, {}).get(y, INF))
        if pair == INF:
            return INF
        return (6 * max(0, pair - 1)
                + 3 * min(self.drive(st, a, pw), self.drive(st, c, pw))
                + 2 * self.stance(st, tuple(nucs[a]) + tuple(nucs[c])))

    def score(self, st):
        """Lower is closer to a win. Not admissible -- it is a beam ordering,
        not an A* heuristic -- but every term in it is an exact distance."""
        b = self.b
        players, _nucs, crates, _eyes = st
        if not players:
            return INF
        pool = [[c, t] for t in range(3) for c in crates[t]]
        todo = []
        for cell, want in b.targets.items():
            hit = None
            for e in pool:
                if e[0] == cell and (want is None or e[1] == want):
                    hit = e
                    break
            if hit is not None:
                pool.remove(hit)
                continue
            todo.append((cell, want))
        if not todo:
            return 0
        pw = self.walk(players[0][0])
        cost = 0
        for cell, want in todo:
            best, chosen = INF, None
            for e in pool:
                if want is not None and e[1] != want:
                    continue
                d = self.pushd[cell][e[0]]
                if d < best:
                    best, chosen = d, e
            if chosen is not None:
                pool.remove(chosen)
                cost += 10 * best + 40 + min(pw.get(chosen[0], 60), 60)
                continue
            cost += 10 * self.pushd[cell][b.lab] + 120
            cand = (want,) if want is not None else (KCG, KTA, KG)
            sub = min(self.recipe_cost(st, t, pw) for t in cand)
            cost += sub if sub < INF else 400
        return cost

    # -- the deadlock test ----------------------------------------------------
    def _push_table_now(self, target, crates_at, blockers):
        """`_push_table` again, but with the OTHER crates standing in the way.

        The static table is what the beam is ordered by (cheap, and monotone as
        crates move); this one is what says whether a target is still SERVABLE,
        and it is only ever asked at a milestone."""
        b = self.b
        if target in blockers or not self.crate_ok[target]:
            return {}
        d = {target: 0}
        q = deque([target])
        while q:
            c = q.popleft()
            for dd in _DIRS:
                back = b._nx[_OPP[dd]]
                p = back[c]
                if p < 0 or not self.crate_ok[p] or p in blockers:
                    continue
                s = back[p]
                if s < 0 or not self.stand_ok[s] or s in blockers:
                    continue
                if p not in d:
                    d[p] = d[c] + 1
                    q.append(p)
        return d

    def makeable(self, st, t) -> bool:
        """Can one more crate of type ``t`` still be built, and WHERE would it
        appear? Returns the list of birth cells (empty = not makeable).

        CG and TA crates are born on the Lab. A G crate is born on the lab-side
        neighbour of a Splitter, out of a guanine that walked into it -- and the
        guanine may itself still have to come out of a CG crate fed back through
        the same Splitter, which is why `crates[KCG]` counts as a source."""
        _players, nucs, crates, _eyes = st
        if t == KG:
            if not self.b.splitter or not (nucs[NG] or crates[KCG]):
                return []
            out = []
            for sp in self.b.splitter:
                cell = self.b._first(sp, None)
                if cell is not None:
                    out.append(cell)
            return out
        a, c = _RECIPE[t]
        if not (nucs[a] and nucs[c]) or self.b.lab is None:
            return []
        return [self.b.lab]

    def alive(self, st) -> bool:
        """False when some unmet target can no longer be served.

        A crate is delivered by SHOVING it, so a target is servable only while
        some source -- an existing crate of an acceptable type, or the Lab if
        one can still be built there -- has a finite push distance to it with
        the OTHER crates treated as walls. That is the check the greedy
        milestone search needs and the static tables cannot make: on level 12
        both G targets sit in one column above the Splitter's delivery cell, so
        filling the NEAR one first walls the far one off forever, and nothing
        about that shows up in a distance that ignores crates.
        """
        players, _nucs, crates, _eyes = st
        if not players:
            return False
        b = self.b
        occupied = {c: t for t in range(3) for c in crates[t]}
        for cell, want in b.targets.items():
            got = occupied.get(cell)
            if got is not None and (want is None or got == want):
                continue
            sources = [c for c, t in occupied.items()
                       if want is None or t == want]
            types = (KCG, KTA, KG) if want is None else (want,)
            for t in types:
                sources.extend(self.makeable(st, t))
            ok = False
            for src in sources:
                blockers = set(occupied)
                blockers.discard(src)
                if src in self._push_table_now(cell, occupied, blockers):
                    ok = True
                    break
            if not ok:
                return False
        return True


def _progress(board: _Board, st):
    """``(targets satisfied, crates on the board)`` -- the milestone ladder."""
    crates = st[2]
    occupied = {c: t for t in range(3) for c in crates[t]}
    sat = 0
    for cell, want in board.targets.items():
        got = occupied.get(cell)
        if got is not None and (want is None or got == want):
            sat += 1
    return (sat, sum(len(crates[t]) for t in range(3)))


#: How many states a milestone candidate's own reachable component may be
#: enumerated to before the check gives up and calls it "unknown". Sized from
#: the measured components: a level-12 branch that has stranded itself closes
#: at ~7k states, and the live branches run past this in well under a second.
DEAD_CAP = 40_000


def dead_end(board: _Board, st, cap: int = DEAD_CAP) -> bool:
    """True when ``st``'s own reachable component is CLOSED and holds no win.

    A proof, not a guess: it only answers True after enumerating the whole
    component, and returns False (i.e. "not known to be dead") the moment it
    passes ``cap``. It is what `_Guide.alive` cannot see -- alive asks whether
    every target is still servable in principle, and the way a branch of this
    game usually dies is subtler than that: the player walls ITSELF in with the
    crates it has delivered, or grows back into a lab room it can no longer
    leave, and the whole component collapses to a few thousand states with no
    win anywhere in it.
    """
    seen = {st}
    frontier = [st]
    while frontier:
        nxt = []
        for cur in frontier:
            for d in _ACTS:
                n = board.step(cur, d)
                if n in seen:
                    continue
                if board.won(n):
                    return False
                seen.add(n)
                if len(seen) > cap:
                    return False
                nxt.append(n)
        frontier = nxt
    return True


def _to_milestone(board, guide, start, reached, width, depth, keep):
    """Up to ``keep`` states satisfying ``reached``, shallowest first, each with
    the presses that get there. Same beam as `beam`, stopped at a milestone."""
    parent = {start: (None, None)}
    layer = [start]
    found = []
    for _depth in range(depth):
        nxt = []
        for st in layer:
            for d in _ACTS:
                n = board.step(st, d)
                if n == st or n in parent:
                    continue
                parent[n] = (st, d)
                if not n[0]:
                    continue                 # the player was destroyed
                if board.won(n) or reached(n):
                    if not board.won(n) and (not guide.alive(n)
                                             or dead_end(board, n)):
                        continue             # a milestone that strands the level
                    out = []
                    cur = n
                    while parent[cur][0] is not None:
                        cur, d2 = parent[cur]
                        out.append(d2)
                    found.append((list(reversed(out)), n))
                    if len(found) >= keep:
                        return found
                    continue
                nxt.append(n)
        if found or not nxt:
            return found
        nxt.sort(key=guide.score)
        layer = nxt[:width]
    return found


def staged(board: _Board, guide: _Guide, width: int, depth: int,
           keep: int = 3, max_stages: int = 14, budget: float = 300.0,
           verbose: bool = False):
    """Plan one MILESTONE at a time, with backtracking.

    The plain beam climbs a level's score fine until the last crate and then
    stalls, because a hundred-press plan through four crate deliveries is far
    past where a width-capped frontier keeps the right basin. Searching to the
    next milestone instead -- "one more target satisfied", else "one more crate
    on the board" -- makes each search 15-60 presses deep, which the same beam
    handles easily.

    Two things stop it being naive greed:

      * every milestone candidate must pass `_Guide.alive`, so a delivery that
        walls off a target it has not served yet is not taken; and
      * it keeps ``keep`` candidates per milestone and BACKTRACKS, so an order
        that dead-ends later is undone rather than being the answer.

    Deliveries are tried before syntheses: making crates is always available and
    always looks like progress, and a search that takes it first fills the lab
    room with crates it can no longer place.
    """
    import time as _time
    t0 = _time.monotonic()

    def rec(st, presses, stage):
        if board.won(st):
            return presses
        if stage >= max_stages or _time.monotonic() - t0 > budget:
            return None
        sat, ncr = _progress(board, st)
        for label, reached in (
                ("deliver", lambda n: _progress(board, n)[0] > sat),
                ("make", lambda n: _progress(board, n)[1] > ncr)):
            for i, (sub, st2) in enumerate(
                    _to_milestone(board, guide, st, reached, width, depth,
                                  keep)):
                if verbose:
                    print(f"      stage {stage} {label}#{i}: +{len(sub)} "
                          f"presses -> {_progress(board, st2)}", flush=True)
                got = rec(st2, presses + sub, stage + 1)
                if got is not None:
                    return got
                if _time.monotonic() - t0 > budget:
                    return None
        # No milestone left to reach, or none of them led anywhere: the win may
        # still be a plain beam away. This is the ENDGAME case and it is the
        # reason the milestone ladder is not enough on its own -- the last
        # delivery of a level often needs a crate SHUFFLED off one target onto
        # another, which raises no milestone in between.
        endgame = beam(board, guide, width, BEAM_DEPTH - len(presses), verbose,
                       start=st)
        if endgame is not None and verbose:
            print(f"      stage {stage} endgame: +{len(endgame)} presses",
                  flush=True)
        return None if endgame is None else presses + endgame

    return rec(board.start, [], 0)


def beam(board: _Board, guide: _Guide, width: int, max_depth: int,
         verbose: bool = False, start=None):
    """Width-capped breadth-first search ordered by `_Guide.score`.

    A beam rather than weighted A*: the score is loose by a large constant over
    the middle of a level (it costs a base pair by how far apart the two bases
    ARE, and closing that gap needs a de-sync detour it cannot see), so an f =
    g + w*h ordering spends the whole budget on the shallow states while the
    beam keeps climbing. Same reasoning as ps:idols_to_the_burnt_god -- order
    the probe by PROGRESS, not by an admissible bound.
    """
    if start is None:
        start = board.start
    parent: dict = {start: (None, None)}
    layer = [start]
    for depth in range(max_depth):
        nxt = []
        for st in layer:
            for d in _ACTS:
                n = board.step(st, d)
                if n == st or n in parent:
                    continue
                parent[n] = (st, d)
                if not n[0]:
                    continue                 # the player was destroyed
                if board.won(n):
                    out = []
                    cur = n
                    while parent[cur][0] is not None:
                        cur, d2 = parent[cur]
                        out.append(d2)
                    return list(reversed(out))
                nxt.append(n)
        if not nxt:
            return None
        nxt.sort(key=guide.score)
        layer = nxt[:width]
        if verbose and depth % 20 == 0:
            print(f"      depth {depth:3d}  beam {len(layer):6d}  "
                  f"best {guide.score(layer[0]):5d}  seen {len(parent)}",
                  flush=True)
    return None


# ---------------------------------------------------------------------------
# One level's answer
# ---------------------------------------------------------------------------

class _Level:
    """A level's board plus the best answer either tier could give it.

    ``tier`` is ``"exact"`` (proved shortest, MEASURED optimal sets, and a plan
    from any state on the shortest-path subgraph) or ``"beam"`` (a winning plan
    whose optimal sets are proved by exhibition, and no answer away from it).
    """

    __slots__ = ("board", "field", "plan", "optsets", "tier", "states",
                 "amb_steps", "pins")

    def __init__(self, board):
        self.board = board
        self.field = None
        self.plan = None
        self.optsets = None
        self.tier = "unsolved"
        self.states = 0
        #: How many of the plan's own presses are ones where the interpreter had
        #: a real choice of base pair. They are certified by replay, so they are
        #: right -- this is how many of them there were.
        self.amb_steps = 0
        #: How many transitions `certify` had to pin from the interpreter before
        #: the plan replayed clean.
        self.pins = 0

    # -- answering ------------------------------------------------------------
    def answer(self, st):
        """``(plan, optsets)`` from ``st``, or ``(None, None)``."""
        if self.field is not None:
            p = self.field.plan(st)
            if p is None:
                return None, None
            board = self.board
            walk = [st]
            for d in p:
                walk.append(board.step(walk[-1], d))
            sets = []
            for i, here in enumerate(walk[:-1]):
                keep = []
                for alt in self.field.optimal(here):
                    board.step(here, alt)
                    # An alternative whose outcome the model GUESSED (see
                    # `_Board._pair`) is dropped from the label: only the
                    # plan's own press is certified by replay, so offering a
                    # sibling would be training a coin flip. Dropping can only
                    # shrink a set, never make one wrong.
                    if not board.amb or alt == p[i]:
                        keep.append(alt)
                sets.append(keep or [p[i]])
            return p, sets
        if st == self.board.start and self.plan is not None:
            return list(self.plan), [list(x) for x in self.optsets]
        return None, None


def suffix_optsets(board: _Board, plan) -> list:
    """Optimal-action sets for a beam plan, proved by EXHIBITION.

    An alternative press is offered only when replaying the plan's OWN SUFFIX
    from the board it lands on still reaches the win inside the presses the
    recorded step leaves. That is a demonstration, not an inference: it never
    claims a tie it cannot show, and because the beam's plans are not certified
    shortest it is the only claim available. Ties are rare here and that is the
    mechanic -- every press moves the player AND every base of its colour, so
    two different presses almost never land on the same board.
    """
    walk = [board.start]
    for d in plan:
        walk.append(board.step(walk[-1], d))
    out = []
    for i, taken in enumerate(plan):
        best = {taken}
        for alt in _ACTS:
            if alt == taken:
                continue
            cur = board.step(walk[i], alt)
            if board.amb or cur == walk[i]:
                continue    # refused, or a press whose outcome is a GUESS
            ok = board.won(cur)
            for d in plan[i + 1:]:
                if ok:
                    break
                cur = board.step(cur, d)
                ok = board.won(cur)
            if ok:
                best.add(alt)
        out.append([d for d in _ACTS if d in best])
    return out


#: Levels whose exact ball is small enough to enumerate; everything else falls
#: through to the beam. Sized from the MEASURED balls, not from patience: every
#: queued state is one packed key in a dict plus five successor ints, so this is
#: a memory budget -- about 700 MB at the cap, and 0.4 GB at the largest level
#: that actually fits.
EXACT_CAP = 2_500_000
#: The two tier-2 searches, tried in order until one returns. The plain beam is
#: first because it is what solves level 4 outright; the milestone search below
#: it is the deeper (and slower) one. Both are bounded so that a level no tier
#: reaches costs a fixed few minutes once, and never again -- it is remembered
#: as unsolvable in ``data/sokobaiogenesis_plans.json`` and named in
#: `SokobaiogenesisSolver.skip_levels`.
BEAM_WIDTH = 3_000
#: The adapter GAME_OVERs at 200 actions per level and the exploration prefix
#: spends some of them, so a plan longer than this is no use even if found.
BEAM_DEPTH = 150
STAGED_LADDER = ((5_000, 80, 3, 420.0),)


#: How many transitions `solve_level` will pin from the interpreter before
#: giving a level up. Each pin costs a whole re-search, and a shipped plan has
#: needed at most one.
MAX_PINS = 3


def solve_level(board: _Board, certify=None, verbose: bool = False) -> _Level:
    """Solve a level and, when ``certify`` is given, PROVE the plan on the
    interpreter before returning it.

    ``certify(board, plan)`` replays the plan through the real engine and
    returns ``(won, i, state, truth)`` -- ``won`` when it reached the win with
    the model agreeing at every press, otherwise the index of the first press
    the model got wrong, the state it was taken from, and what the interpreter
    actually did. That only ever happens at a press where the interpreter had a
    genuine choice of which base pair to consume and the model guessed (see
    `_Board._pair`); the truth is pinned into ``board.override`` and the whole
    search is re-run, so a plan is never shipped on a transition the engine
    disagrees with.
    """
    for attempt in range(MAX_PINS + 1):
        lvl = _solve_once(board, verbose)
        if lvl.plan is None or certify is None:
            return lvl
        lvl.pins = attempt
        won, i, st, truth = certify(board, lvl.plan)
        if won:
            for j, d in enumerate(lvl.plan):
                board.step(_walk_to(board, lvl.plan, j), d)
                if board.amb:
                    lvl.amb_steps += 1
            return lvl
        if i is None:                      # replayed clean but did not win
            lvl.plan = None
            return lvl
        board.override[(pack(st), lvl.plan[i])] = pack(truth)
        if verbose:
            print(f"      pinned press {i} ({lvl.plan[i]}) from the "
                  f"interpreter; re-searching", flush=True)
    lvl.plan = None
    lvl.tier = "unsolved"
    return lvl


def _walk_to(board: _Board, plan, i):
    """The state ``plan``'s i-th press is taken from."""
    st = board.start
    for d in plan[:i]:
        st = board.step(st, d)
    return st


def _solve_once(board: _Board, verbose: bool = False) -> _Level:
    """Strongest tier first: the exact field, then the beam, then milestones."""
    lvl = _Level(board)
    field = build_field(board, EXACT_CAP)
    if field is not None and field.dstar >= 0:
        lvl.field = field
        lvl.states = len(field.states)
        lvl.plan = field.plan(board.start)
        lvl.optsets = lvl.answer(board.start)[1]
        lvl.tier = "exact"
        return lvl
    guide = _Guide(board)
    plan = beam(board, guide, BEAM_WIDTH, BEAM_DEPTH, verbose)
    tier = "beam"
    if plan is None:
        for width, depth, keep, budget in STAGED_LADDER:
            plan = staged(board, guide, width, depth, keep=keep,
                          budget=budget, verbose=verbose)
            if plan is not None:
                tier = "staged"
                break
    if plan is not None:
        lvl.plan = plan
        lvl.optsets = suffix_optsets(board, plan)
        lvl.tier = tier
    return lvl


def certify(g, eng, board: _Board, plan):
    """Replay ``plan`` through the INTERPRETER from the level start, comparing
    the model at every press. Returns ``(won, i, state, truth)``.

    The engine is left where it was. Both this and `record_level` start from a
    freshly (re)built position index -- ``restore`` marks it dirty and
    ``set_level`` reloads the level -- which is what makes the replay here and
    the recording later follow the same branch at a press where the interpreter
    is choosing between two base pairs.
    """
    snap = snapshot(eng)
    try:
        st = board.start
        for i, d in enumerate(plan):
            want = board.step(st, d)
            eng.step(d)
            got = read_state(g, eng)
            if got != want:
                return False, i, st, got
            st = want
        return eng.check_win(), None, None, None
    finally:
        restore(eng, snap)


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class SokobaiogenesisExpert(PSExpert):
    """Planner over the native model (see `_Board`); the interpreter is stepped
    only by the recorder, which is also what verifies every plan.

    `PSExpert` supplies the plan memo, the snapshot/restore discipline and the
    on-disk plan cache; `_search` swaps in `solve_level`'s two tiers. The disk
    cache is what makes a second run (and every `parallelize_generator` shard)
    free: the searches are seed-independent and cost ~5 minutes cold.
    """

    #: All five presses, ACTION included -- see `_ACTS` for why it is a real
    #: WAIT here and not a no-op.
    directions = list(_ACTS)

    plan_cache_path = (Path(__file__).resolve().parent.parent
                       / "data" / "sokobaiogenesis_plans.json")

    def setup(self) -> None:
        self._levels: dict[int, _Level] = {}
        self._cur: int | None = None

    # -- the level's answer ---------------------------------------------------
    def level(self, eng, level: int) -> _Level:
        """The level's `_Level`, solved once and cached for every later seed.

        Whichever state happens to build it describes the same level: the walls,
        the DNA columns, the Tech tiles and the targets are static, and the
        search always starts from the board this reads."""
        got = self._levels.get(level)
        if got is None:
            board = build_board(self.g, eng)
            got = self._levels[level] = solve_level(
                board, certify=lambda b, p: certify(self.g, eng, b, p))
        return got

    def state(self, eng):
        return read_state(self.g, eng)

    # -- planning -------------------------------------------------------------
    def plan(self, eng, level: int | None = None):
        self._cur = level
        return super().plan(eng, level)

    def heuristic(self, eng) -> int:                     # pragma: no cover
        raise AssertionError(
            "SokobaiogenesisExpert plans with solve_level; heuristic is unused")

    def _search(self, eng):
        lvl = self.level(eng, self._cur)
        presses, sets = lvl.answer(self.state(eng))
        return None if presses is None else Plan(presses, sets)

    def optimal_dirs(self, eng, level: int | None, plan, pi: int):
        """Every press that is at least as good as the one about to be taken,
        read from the LIVE engine state.

        On an exact level that is the measured tie set at whatever board the
        engine is on -- which is what makes the label right after an epsilon
        detour too. On a beam level the field does not exist, so it falls back
        to the sets `suffix_optsets` proved for the recorded plan."""
        lvl = self._levels.get(level)
        if lvl is not None and lvl.field is not None:
            best = lvl.field.optimal(self.state(eng))
            if best:
                return best
        sets = getattr(plan, "optsets", None)
        if sets is not None and pi < len(sets) and sets[pi]:
            return list(sets[pi])
        return [plan[pi]] if pi < len(plan) else None


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class SokobaiogenesisSolver(PSAStarSolver):
    game_id = "puzzlescript_sokobaiogenesis"
    game_name = GAME_NAME
    expert_cls = SokobaiogenesisExpert

    #: The two levels no tier reaches, skipped up front so a cold start does not
    #: spend ~13 minutes each rediscovering it. Both are two-crate levels whose
    #: lab room is three cells wide, so the plan has to shrink into the void,
    #: hand-push bases into a pair, grow back, deliver, and do it all again --
    #: measured at over a hundred presses before the first delivery, which is
    #: past where a width-capped frontier keeps the right basin, and their exact
    #: balls pass 2.5M states well before the first win. Level 8 also has the
    #: narrowest lab in the game (cols 0-2 with walls in it), which is what
    #: makes its deliveries interfere. This is a SEARCH limit, not a model one:
    #: `--selfcheck` covers both levels exactly like the other eleven.
    skip_levels = frozenset({8, 12})

    #: Unused -- the expert never calls the base A* -- but left at the family
    #: default so a future subclass that does is not silently starved.
    node_cap = 400_000
    weight = 1
    #: The adapter GAME_OVERs at 200 actions per level; the longest plan here is
    #: well inside that, leaving the rest of the budget to the exploration
    #: prefix.
    max_steps = 200

    #: Recovery is the RESET prefix, not an epsilon detour. A detour needs a
    #: re-plan from an arbitrary board: on the nine EXACT levels that is a dict
    #: lookup and would be safe, but on the four beam levels it is a fresh
    #: minute-long search per press, and `record_level` takes one epsilon for
    #: the whole run. Same choice as every other push game in this family.
    epsilon = 0.0

    def prepare_expert(self, game, expert) -> None:
        """Fetch every level's plan up front.

        Pure front-loading -- discovery would pay for exactly these searches on
        its first call -- but it means the cost is one clearly-attributed
        startup pass rather than a mystery stall inside the first episode. It
        goes through `PSExpert.plan`, NOT through `level`, so a process that
        finds ``data/sokobaiogenesis_plans.json`` already written pays nothing:
        the searches are the whole cost of generation (~20 minutes cold) and are
        seed-independent, so without that every `parallelize_generator` shard
        would re-derive all eleven. Delete the file after changing the model,
        the guide or a tier -- it caches FAILURES too.

        `skip_levels` is honoured here and not only in `_ensure`: the two levels
        no tier reaches would otherwise cost ~13 minutes EACH on every cold
        start, discovering the same nothing."""
        for level in range(game.n_levels):
            if level in self.skip_levels:
                continue
            game.set_level(level)
            expert.plan(game._engine, level)

    def optimal_for(self, expert, level: int, plan: list, pi: int):
        return expert.optimal_dirs(expert.game._engine, level, plan, pi)


# ---------------------------------------------------------------------------
# selfcheck
# ---------------------------------------------------------------------------

def _probe(board, st, d):
    """``(next state, events fired)`` for one press, without disturbing the
    global counters. This is what the guided fuzz steers on."""
    global TRACE, COUNTS
    saved_trace, saved = TRACE, COUNTS
    TRACE, COUNTS = True, Counter()
    try:
        nxt = board.step(st, d)
        return nxt, COUNTS
    finally:
        TRACE, COUNTS = saved_trace, saved


#: Mechanics a uniform random walk essentially never reaches: they need a
#: nucleotide steered across the void, a crate shoved into a Splitter, or a
#: player walked onto a Tech tile. The guided fuzz weights presses that fire
#: one of them, which is the difference between "62400 presses agreed" and
#: "the model was exercised".
_RARE = ("synth_cg", "synth_ta", "franken_born", "green_eye_opened",
         "splitter_crate_to_guanine", "splitter_guanine_to_crate",
         "m2_crate_onto_tech_refused", "m3_tacrate_onto_splitter_refused",
         "m11_nonguanine_onto_splitter_refused",
         "m12_player_onto_splitter_refused", "m4_push_crate",
         "m5_push_nucleo", "eye_0", "eye_1", "eye_3", "player_destroyed",
         "big_bounced_off_grow")


def _scatter(g, eng, rng, extra: int):
    """Drop ``extra`` random nucleotides of each type into free Void cells.

    The third fuzz mode exists because of a bug the other two could not see.
    Which adjacent base pair the Lab consumes when SEVERAL are complementary at
    once is decided by the interpreter's expansion order and its position-index
    set order (see `_Board._pair`), and a shipped level rarely has two pairs
    lined up under random play -- but a shortest PLAN lines bases up on purpose,
    so level 9's diverged from the interpreter at press 11 while 228k fuzz
    presses had reported the model clean. Crowding the void makes the
    several-pairs case the common case instead of the rare one.
    """
    idx = g.obj_name_to_idx
    void = idx["void"]
    free = []
    for r, row in enumerate(eng.grid):
        for c, cell in enumerate(row):
            if void in cell and len(cell) == 2:      # bare Void + Background
                free.append((r, c))
    rng.shuffle(free)
    for name in ("cnucleo", "gnucleo", "tnucleo", "anucleo"):
        for _ in range(extra):
            if not free:
                return
            r, c = free.pop()
            eng.grid[r][c].add(idx[name])
    eng._position_index_dirty = True


def selfcheck(trials: int = 40, steps: int = 120, guided: int = 80,
              gsteps: int = 160, crowded: int = 60, verbose: bool = True) -> int:
    """Drive presses through BOTH the interpreter and `_Board.step`, comparing
    the WHOLE state (players and their kinds, all four nucleotide sets, all
    three crate sets, every eye) and the win flag after every one.

    Three modes, because none of them alone is enough here.

      * RANDOM. A uniform walk spends the game shuffling the player around the
        lab half: it pairs a base only by accident, and it never shoves a crate
        into a Splitter or walks onto a Tech tile at all.
      * GUIDED. Each of the four presses is scored by the rare mechanics it
        would fire (`_RARE`) and one of those is taken when it can be, which is
        what covers the synthesis, the Frankenstein birth, the green eye, both
        Splitter conversions, and the lethal case where a crate is synthesised
        on top of the player.
      * CROWDED. The guided walk again, on a board whose void has been filled
        with extra bases (`_scatter`), so SEVERAL complementary pairs are
        adjacent at once and the interpreter has to choose between them. That
        choice is real state, and it is the one thing the first two modes do not
        reach often enough to test.

    ``action`` is in the press alphabet in every mode even though the search
    excludes it: "no rule reads it" is exactly the claim that should be measured
    against the interpreter rather than read off a rules section (see
    ps:ouroboros, whose rules section does not mention ``action`` either and
    whose ACTION is a shrink button).

    Walks start COLD, from the level as loaded, with no warm-up press.
    """
    global TRACE
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    g, eng = game._game, game._engine
    alphabet = _ACTS
    total = 0
    ambiguous = 0
    COUNTS.clear()
    TRACE = True
    for level in range(game.n_levels):
        rng = random.Random(f"sokobaiogenesis:selfcheck:{level}")
        bad = presses = 0
        first = None
        for run in range(trials + guided + crowded):
            steer = run >= trials
            game.set_level(level)
            if run >= trials + guided:
                _scatter(g, eng, rng, rng.randint(1, 3))
            board = build_board(g, eng)
            st = board.start
            for _ in range(gsteps if steer else steps):
                if steer and rng.random() < 0.75:
                    scored = []
                    for cand in _ACTS:
                        nxt, ev = _probe(board, st, cand)
                        if nxt != st:
                            scored.append((sum(ev[k] for k in _RARE), cand))
                    rare = [c for h, c in scored if h]
                    live = [c for _h, c in scored]
                    d = (rng.choice(rare) if rare
                         else (rng.choice(live) if live
                               else rng.choice(alphabet)))
                else:
                    d = rng.choice(alphabet)
                want = board.step(st, d)
                amb = board.amb
                eng.step(d)
                presses += 1
                got = read_state(g, eng)
                if amb:
                    # The interpreter had a genuine choice of which base pair
                    # to consume, and the model says so rather than pretending
                    # to know (see `_Board._pair`). Re-sync and carry on -- no
                    # search ever takes such a press, so it is not a mismatch.
                    ambiguous += 1
                    st = got
                    if eng.check_win():
                        break
                    continue
                if want != got or board.won(want) != eng.check_win():
                    bad += 1
                    if first is None:
                        first = (d, st, want, got,
                                 board.won(want), eng.check_win())
                    break
                st = want
                if eng.check_win():
                    break
        total += bad
        if verbose:
            print(f"  L{level}: {'OK' if not bad else f'{bad} MISMATCHES'} "
                  f"({presses} presses vs the interpreter)", flush=True)
            if first is not None:
                d, st, want, got, ww, gw = first
                print(f"      press={d} win model={ww} engine={gw}")
                print(f"      from  {st}")
                print(f"      model {want}")
                print(f"      engine{got}")
    TRACE = False
    if verbose:
        print(f"  {ambiguous} presses re-synced: the interpreter had a real "
              f"choice of base pair, which `_Board._pair` guesses and "
              f"`solve_level` certifies")
        print("  mechanics exercised:")
        for name in sorted(COUNTS):
            print(f"    {name:38s} {COUNTS[name]}")
        missing = [k for k in _RARE if not COUNTS[k]]
        if missing:
            print(f"    not reached by the fuzz (--moves witnesses these): "
                  f"{', '.join(missing)}")
    return total


# ---------------------------------------------------------------------------
# --moves: purpose-built witnesses for the mechanics no shipped level reaches
# ---------------------------------------------------------------------------

#: ``(name, level rows, {cell: [extra object names]}, presses, mechanic)``.
#:
#: `selfcheck`'s guided fuzz covers every mechanic the thirteen shipped levels
#: can reach, which is not all of them: three REFUSAL rules need a geometry no
#: level ships (a crate lined up with a Tech tile, a TA crate lined up with a
#: Splitter, a non-guanine base lined up with a Splitter), and the
#: FrankenSmall->FrankenPlayer growth needs a Frankenstein that has been shrunk.
#: Each row below is a board built to hit exactly one of them, checked press by
#: press against the interpreter like everything else -- so a mechanic the
#: corpus cannot reach is still a mechanic the model is proved on. This is the
#: `--moves` half of the ps:silver_lungs discipline: classify the space by
#: mechanic, then top the fuzz up with a purpose-built witness for each branch
#: the shipped levels leave out.
_WITNESSES = [
    ("a crate may not be shoved onto a Tech tile",
     ["........", "..$.-iii", "........"], {(1, 3): ["cgcrate"]},
     ["right"], "m2_crate_onto_tech_refused"),
    ("a TA crate may not be shoved onto a Splitter",
     ["....2iii", "..$.Siii", "....2iii"], {(1, 3): ["tacrate"]},
     ["right"], "m3_tacrate_onto_splitter_refused"),
    ("a CG crate MAY be shoved onto a Splitter, and becomes guanine",
     ["....2iii", "..$.Siii", "....2iii"], {(1, 3): ["cgcrate"]},
     ["right"], "splitter_crate_to_guanine"),
    ("a non-guanine base may not enter the Splitter",
     ["....2iii", "..P.Sizi", "....2iii"], {},
     ["left", "left"], "m11_nonguanine_onto_splitter_refused"),
    ("guanine DOES enter the Splitter and leaves as a crate",
     ["....2iii", "..$.Sixi", "....2iii"], {},
     ["left", "left"], "splitter_guanine_to_crate"),
    ("a Frankenstein shrinks, and grows back a Frankenstein",
     ["..\\+iiii", "....-iii", "....2iii"], {(1, 3): ["frankenplayer"]},
     ["right", "up", "left"], "grow_6_to_4"),
    ("synthesising a crate on the Lab you are standing on kills you",
     ["..L.2iii", "..$.1izi", "....2ixi"], {},
     ["up"], "synth_killed_the_player"),
]


def moves(verbose: bool = True) -> int:
    """Drive each `_WITNESSES` board through both the interpreter and the model
    and require the named mechanic to fire. Returns the number of problems."""
    global TRACE
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    g, eng = game._game, game._engine
    bad = 0
    covered = set()
    for name, rows, extra, presses, mechanic in _WITNESSES:
        eng.load_level(g._build_level(rows))
        for (r, c), objs in extra.items():
            for o in objs:
                eng.grid[r][c].add(g.obj_name_to_idx[o])
        eng._position_index_dirty = True
        board = build_board(g, eng)
        st = board.start
        TRACE, saved = True, COUNTS.copy()
        COUNTS.clear()
        for d in presses:
            st = board.step(st, d)
            eng.step(d)
            if st != read_state(g, eng):
                bad += 1
                if verbose:
                    print(f"  {name}: MODEL DIVERGED on {d}")
                break
        fired = COUNTS[mechanic]
        COUNTS.clear()
        COUNTS.update(saved)
        TRACE = False
        covered.add(mechanic)
        if not fired:
            bad += 1
        if verbose:
            print(f"  {'OK ' if fired else 'MISS'} {name}"
                  f"  ({mechanic} x{fired})")
    if verbose:
        print(f"moves: {len(covered)} mechanics witnessed, {bad} problems")
    return bad


# ---------------------------------------------------------------------------
# --plans
# ---------------------------------------------------------------------------

def _levels():
    """``(solver, game, expert)`` with every level solved."""
    solver = SokobaiogenesisSolver()
    game, expert, _solvable = solver._ensure(0)
    return solver, game, expert


def _report() -> int:
    """Solve every level, replay its plan on the INTERPRETER, and print what
    each tier could prove -- the quick "is this game still solved" check."""
    solver, game, expert = _levels()
    eng = game._engine
    bad = 0
    exact = 0
    total = 0
    for level in range(game.n_levels):
        game.set_level(level)
        if level in solver.skip_levels:
            print(f"  L{level}: {eng.height}x{eng.width}  SKIPPED "
                  f"(no tier reaches it -- see skip_levels)")
            continue
        lvl = expert.level(eng, level)
        check_geometry(lvl.board)
        if lvl.plan is None:
            bad += 1
            print(f"  L{level}: {eng.height}x{eng.width}  UNSOLVED ({lvl.tier})")
            continue
        for d in lvl.plan:
            eng.step(d)
        won = eng.check_win()               # read BEFORE anything reloads
        bad += not won
        exact += lvl.tier == "exact"
        total += len(lvl.plan)
        ties = sum(len(s) for s in lvl.optsets) / len(lvl.optsets)
        print(f"  L{level}: {eng.height}x{eng.width}  {len(lvl.plan):3d} presses  "
              f"win={won}  {lvl.tier:6s}"
              f"{f'  {lvl.states:7d} states' if lvl.states else ' ' * 15}"
              f"  {ties:.2f} optimal presses/step"
              f"  {lvl.amb_steps} guessed pair{'s' if lvl.amb_steps != 1 else ''}"
              f", {lvl.pins} pinned", flush=True)
    solved = game.n_levels - len(solver.skip_levels)
    print(f"  {solved}/{game.n_levels} levels solved, {exact} of them PROVED "
          f"SHORTEST, {total} presses in all, {bad} problems")
    return 1 if bad else 0


# ---------------------------------------------------------------------------
# --verify
# ---------------------------------------------------------------------------

def _verify() -> int:
    """Double-entry check of every field and of every label this ships.

    The training targets ARE the optimal sets, so "the plan wins" is not enough
    of a check. Two passes that fail differently:

      * **Bellman fixpoint** over each exact field's successor table -- relax
        ``d(s) = 1 + min d(succ)`` with no predecessor map, nothing shared with
        the reverse sweep that built the field, and require agreement
        everywhere plus ``d == 0`` exactly at the winning states.
      * **Executable labels** -- walk each plan on the INTERPRETER and actually
        press every direction the label calls optimal. The board it lands on
        must be the model's successor, and on an exact level its distance must
        be exactly one less. That takes the labels out of the model and puts
        them through the real thing.
    """
    _solver, game, expert = _levels()
    eng = game._engine
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        if level in _solver.skip_levels:
            continue
        lvl = expert.level(eng, level)
        board = lvl.board
        if lvl.plan is None:
            print(f"  L{level}: unsolved, nothing to verify")
            continue

        checked = 0
        if lvl.field is not None:
            f = lvl.field
            n = len(f.states)
            win = np.fromiter(
                (board.won(unpack(f.states[j])) for j in range(n)),
                dtype=bool, count=n)
            succ = np.array(f.succ, dtype=np.int64).reshape(n, 5)
            valid = succ >= 0
            safe = np.where(valid, succ, 0)
            big = 1 << 29
            check = np.where(win, 0, -1).astype(np.int64)
            # Bellman to a fixpoint, vectorised -- the biggest field here is
            # 1.3M states and a per-state Python relaxation would take longer
            # than the search that built it.
            while True:
                got = np.where(valid, check[safe], -1)
                got = np.where(got >= 0, got, big)
                best = got.min(axis=1) + 1
                nxt = np.where((best < big) & ((check < 0) | (best < check)),
                               best, check)
                nxt = np.where(win, 0, nxt)
                if np.array_equal(nxt, check):
                    break
                check = nxt
            have = np.array(f.dist, dtype=np.int64)
            diff = int((check != have).sum())
            if diff:
                bad += 1
                print(f"  L{level}: the field disagrees with its Bellman "
                      f"fixpoint at {diff} states")
            if not np.array_equal(have == 0, win):
                bad += 1
                print(f"  L{level}: distance 0 is not exactly the winning states")
            checked = n

        st = board.start
        for i, press in enumerate(lvl.plan):
            best = lvl.optsets[i]
            if press not in best:
                bad += 1
                print(f"  L{level}: step {i} is not in its own optimal set")
            here = lvl.field.distance(st) if lvl.field is not None else None
            before = snapshot(eng)
            for alt in best:
                eng.step(alt)
                landed = read_state(game._game, eng)
                want = board.step(st, alt)
                if landed != want:
                    bad += 1
                    print(f"  L{level}: step {i} label {alt}: the interpreter "
                          f"and the model disagree")
                if here is not None:
                    there = lvl.field.distance(landed)
                    if there is None or there != here - 1:
                        bad += 1
                        print(f"  L{level}: step {i} labels {alt} optimal, but "
                              f"the interpreter does not land one press closer")
                restore(eng, before)
            eng.step(press)
            st = board.step(st, press)
        if not eng.check_win():
            bad += 1
            print(f"  L{level}: the interpreter does not report a win")
        print(f"  L{level}: {checked:7d} states relaxed, {len(lvl.plan):3d} plan "
              f"steps, {sum(len(s) for s in lvl.optsets):3d} labels pressed on "
              f"the interpreter -- {'OK' if not bad else 'see above'}",
              flush=True)
    print(f"verify: {bad} problems")
    return 1 if bad else 0


# ---------------------------------------------------------------------------
# --audit
# ---------------------------------------------------------------------------

#: Every cell COMPOSITION the thirteen levels can produce: a ground tile, and
#: whatever the Player/Wall/Crate/Nucleo collision layer can be holding on it.
def _compositions():
    big = ["playerred", "playerblue", "playeryellow", "playergreen",
           "frankenplayer"]
    tiny = ["playersmall", "frankensmall"]
    crates = ["cgcrate", "tacrate", "gcrate"]
    nucs = ["cnucleo", "gnucleo", "tnucleo", "anucleo"]
    targets = ["whitetarget", "cgtarget", "tatarget", "gtarget"]
    comps = [(), ("wall",), ("voidwall",), ("dnarb",), ("dnabr",), ("void",),
             ("growdestin",), ("lab",), ("shrink",), ("grow",), ("splitter",),
             ("eyeyellow",), ("eyered",), ("eyegreenclosed",), ("eyegreen",)]
    comps += [(t,) for t in targets]
    for p in big:
        comps += [(p,), ("lab", p), ("growdestin", p), ("eyegreenclosed", p)]
        comps += [(t, p) for t in targets]
    comps += [("void", p) for p in tiny]
    for c in crates:
        comps += [(c,), ("lab", c), ("growdestin", c)]
        comps += [(t, c) for t in targets]
    comps += [("void", n) for n in nucs]
    return comps


def _audit() -> int:
    """Assert every composition the game can show is distinct in the frame, at
    every cell size the thirteen levels render at.

    The pair this exists for is ``<target>+<crate>`` against ``<crate>``: the
    win condition is ``All <colour>Target on <matching crate>``, so a crate that
    hides the target under it makes the WINNING board indistinguishable from any
    other -- which is exactly what shipped. See the art note at the top of
    data/puzzlescript_games/Sokobaiogenesis.txt.

    Whole frames are compared, not cell crops: `_render_frame` upscales the
    board to fill 64x64 and letterboxes it, so the output's cell grid is not
    ``cell_px``-aligned and an arithmetic crop reads the wrong window.

    It also checks the LabAni frames, which the interpreter can strand on the
    board for good when a press both wins and pairs a base: they must render
    the settled Lab-under-crate exactly, or the winning frame of such a level
    would show an animation cell no other frame ever shows.
    """
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx

    # The window scheme assumes the ground layers are mutually exclusive, which
    # is a fact about the shipped levels rather than about the collision layers
    # -- so it is asserted, not assumed.
    ground = [idx[n] for n in ("lab", "shrink", "grow", "splitter", "eyeyellow",
                               "eyered", "eyegreenclosed", "eyegreen")]
    targets = [idx[n] for n in ("whitetarget", "cgtarget", "tatarget",
                                "gtarget")]
    for level in range(game.n_levels):
        game.set_level(level)
        for row in eng.grid:
            for cell in row:
                assert not (any(o in cell for o in ground)
                            and any(o in cell for o in targets)), \
                    "a level ships a Target under a Lab/Tech/Eye tile"
        check_geometry(build_board(g, eng))

    sizes: dict = {}
    for level in range(game.n_levels):
        game.set_level(level)
        sizes.setdefault((eng.height, eng.width), []).append(level)

    comps = _compositions()
    clashes_total = 0
    for (h, w), levels in sorted(sizes.items()):
        def shot(objs):
            eng.height, eng.width = h, w
            eng.grid = [[{idx["background"]} | {idx[o] for o in objs}
                         for _ in range(w)] for _ in range(h)]
            eng._position_index_dirty = True
            return np.asarray(_render_frame(eng, g)).copy()

        shots = {objs: shot(objs) for objs in comps}
        clashes = [(a, b) for a, b in itertools.combinations(comps, 2)
                   if np.array_equal(shots[a], shots[b])]
        # ...and the animation frames must be INVISIBLE -- each must render
        # the settled Lab-under-crate exactly.
        visible = []
        for crate in ("cgcrate", "tacrate", "gcrate"):
            settled = shots[("lab", crate)]
            for ani in ("labani", "labani2", "labani4", "labani5"):
                if not np.array_equal(settled, shot((crate, ani))):
                    visible.append(f"{crate}+{ani}")
        clashes_total += len(clashes) + len(visible)
        name = lambda t: "+".join(t) if t else "floor"           # noqa: E731
        print(f"{h}x{w} board (level{'s' if len(levels) > 1 else ''} "
              f"{', '.join(str(x) for x in levels)}), "
              f"cell {max(1, min(64 // h, 64 // w))}px: {len(comps)} "
              f"compositions, {len(clashes)} indistinguishable pairs, "
              f"{len(visible)} visible animation frames")
        for a, b in clashes:
            print(f"  IDENTICAL: {name(a)} == {name(b)}")
        for v in visible:
            print(f"  VISIBLE ANIMATION: {v} does not render as lab+crate")
    print("audit clean" if not clashes_total
          else f"AUDIT FAILED: {clashes_total} problems")
    return 0 if not clashes_total else 1


# ---------------------------------------------------------------------------
# --symmetry
# ---------------------------------------------------------------------------

def _transform(k: int, mirror: bool):
    """``(cell_fn, dims_fn, direction_map)`` for one element of the 8-group:
    mirror left-right first, then turn clockwise ``k`` times."""
    def cell(rc, hw):
        r, c = rc
        h, w = hw
        if mirror:
            c = w - 1 - c
        for _ in range(k):
            r, c, h, w = c, h - 1 - r, w, h
        return (r, c)

    def dims(hw):
        h, w = hw
        return (w, h) if k % 2 else (h, w)

    dmap = {}
    for nm, (dr, dc) in _DELTA.items():
        if mirror:
            dc = -dc
        for _ in range(k):
            dr, dc = dc, -dr
        dmap[nm] = next(n for n, d in _DELTA.items() if d == (dr, dc))
    dmap["action"] = "action"          # non-directional, and plans contain it
    return cell, dims, dmap


def _symmetry(walk: int = 300) -> int:
    """MEASURE whether the eight turns and mirrors of a board are a symmetry of
    the mechanic -- which is what would entitle this game to the extra
    `PuzzleScriptAdapter._FLIP_GAMES` presentations.

    It is expected to FAIL, and the point is the number. Several late rules
    resolve a choice by the interpreter's fixed ``up, down, left, right``
    expansion order -- which adjacent base pair the Lab consumes, which Void
    cell a shrinking player drops into, which side of the Splitter a new crate
    lands on -- so the mechanic is chiral in ENGINE space. The frame
    augmentation is presentation-only (the engine always plays the same board),
    so this costs nothing in correctness; it is a statement about how much of
    the mechanic a policy can read off the screen, and the reason the game is
    left out of `_FLIP_GAMES`.
    """
    _solver, game, expert = _levels()
    eng, g = game._engine, game._game
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        if level in _solver.skip_levels:
            continue
        lvl = expert.level(eng, level)
        layout = g.levels[level]
        hw = (len(layout), len(layout[0]))
        rng = random.Random(f"sokobaiogenesis:symmetry:{level}")
        runs = {"plan": list(lvl.plan or []),
                "walk": [rng.choice(_ACTS) for _ in range(walk)]}
        notes = []
        for kind, presses in runs.items():
            if not presses:
                continue
            eng.load_level(layout)
            ref = [read_state(g, eng)]
            for d in presses:
                eng.step(d)
                ref.append(read_state(g, eng))
            for k, mirror in itertools.product(range(4), (False, True)):
                if (k, mirror) == (0, False):
                    continue
                cell, dims, dmap = _transform(k, mirror)
                th, tw = dims(hw)
                turned = [[set() for _ in range(tw)] for _ in range(th)]
                for r, row in enumerate(layout):
                    for c, objs in enumerate(row):
                        tr, tc = cell((r, c), hw)
                        turned[tr][tc] = set(objs)
                eng.load_level(turned)
                for i, d in enumerate(presses):
                    eng.step(dmap[d])
                    def move(x):
                        r, c = divmod(x, hw[1])
                        tr, tc = cell((r, c), hw)
                        return tr * tw + tc
                    want = (tuple(sorted((move(x), kk) for x, kk in ref[i + 1][0])),
                            tuple(tuple(sorted(move(x) for x in s))
                                  for s in ref[i + 1][1]),
                            tuple(tuple(sorted(move(x) for x in s))
                                  for s in ref[i + 1][2]),
                            tuple(sorted((move(x), kk)
                                         for x, kk in ref[i + 1][3])))
                    if read_state(g, eng) != want:
                        notes.append(f"{kind}:rot{k}{'m' if mirror else ''}"
                                     f"@press{i}")
                        break
        bad += len(notes)
        print(f"  L{level}: {'SYMMETRIC' if not notes else 'CHIRAL ' + ', '.join(notes[:4])}")
    print("symmetry clean" if not bad else
          f"{bad} chiral presentations -- Sokobaiogenesis stays out of "
          f"_FLIP_GAMES (see the docstring)")
    return 0


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        bad = selfcheck() + moves()
        print(f"selfcheck: {bad} problems")
        sys.exit(1 if bad else 0)
    if "--moves" in sys.argv:
        sys.exit(1 if moves() else 0)
    if "--plans" in sys.argv:
        sys.exit(_report())
    if "--verify" in sys.argv:
        sys.exit(_verify())
    if "--audit" in sys.argv:
        sys.exit(_audit())
    if "--symmetry" in sys.argv:
        sys.exit(_symmetry())
    sys.exit(SokobaiogenesisSolver.main())
