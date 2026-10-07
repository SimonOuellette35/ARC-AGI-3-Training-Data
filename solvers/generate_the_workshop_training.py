"""Generate Phase-1 training data for the PuzzleScript game ps:the_workshop
("The Workshop", bregehr).

The harness -- the rotation contract, the trajectory recorder and the
`BaseSolver` plumbing -- lives in `solvers/common/ps_astar.py`, shared with the
other ps: generators. This file is the game-specific part: the measured
mechanic, a native model of it, the two searches that model is solved with, the
proof of shortestness where there is one, and the render audit.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_the_workshop",
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
Every expert step carries a set of equally-optimal presses -- the COMPLETE,
measured set on every level solved by `_field`, and a sound but partial one
(`_replay_ties`) on the three that only `_beam` reaches.

The game
--------
A workshop with three materials -- STONE, WOOD and IRON -- and three kinds of
floor MARKER, one per material. The win is ``all stonemarker on stone`` and the
same for the other two, i.e. every marker must end up under a block of its own
material. Blocks are never destroyed and markers are never moved, so a level's
marker layout is static and the whole puzzle is where the blocks end up.

Blocks are shoved the way sokoban crates are, with one difference that changes
every level: a push propagates through a WHOLE RUN of blocks
(``[> pushable | pushable]``), so one press can move three blocks at once. And
because the player has to stand behind the run, a block on the outermost ring of
a room can only ever slide ALONG that ring -- the square the player would have to
occupy to push it off is the wall. Sending a block to the ring is usually how a
level is lost.

The tool is the CUTTER: a rotary saw with a red HANDLE and a white BLADE, drawn
in four orientations that are an exact quarter-turn orbit of each other. Cutters
live on their own collision layer, so they pass over blocks and walls, and they
do three things.

  * **They cut.** ``[ > player | cutters | wall] -> [> player | | ]``: shove a
    cutter into a wall and the wall AND THE CUTTER are both destroyed, and the
    player steps into the square the cutter vacated. Walls are the only
    destructible terrain in the game and a cutter is single-use, so the number
    of cutters a level ships is the number of walls it can ever open.

  * **They refuse to cut sideways.** ``UP [ > player | cutters no cuttersup |
    wall] -> cancel`` and its three siblings: shoving a cutter into a wall when
    the BLADE is not pointing that way cancels the whole turn -- not "the push
    is blocked", the turn. Nothing at all happens, including everything else the
    press would have done elsewhere on the board.

  * **They turn.** Pressing ACTION with exactly one cutter's HANDLE against you
    swings it a quarter turn CLOCKWISE around you: the cutter above you (a
    cuttersup, whose handle is its bottom edge) becomes a cuttersright to your
    right. That is the only way to re-aim a cutter, and it is why the game stops
    to say "Press X at the handle to control the cutter" on level 8.

Five more rules that only the interpreter will tell you, all measured against it
in ``--fuzz`` (20 000 random boards x 25 presses = 500 000 transitions, plus
the eleven shipped levels x 400 presses, whole board compared after every one;
0 mismatches):

  * **A chain of cutters cuts, and only the front one dies.**
    ``[ > cutters | cutters | wall]`` has no orientation test at all, so a cutter
    shoved into a second cutter that is against a wall destroys THAT cutter and
    the wall while itself surviving. Two cutters in a row are worth one cut and
    one surviving cutter; two cutters used one at a time are worth two cuts.

  * **A cutter can be pushed ONTO a wall or a block and dies there NEXT turn.**
    ``[cutters wall] -> [wall]`` and ``[cutters pushable] -> [pushable]`` run in
    the rule pass, which happens BEFORE movement resolution -- so an overlap
    created by this turn's movement survives until the next press. Anything can
    be shoved into a cutter (a block chain, another cutter) and the cutter is
    simply deleted; only the PLAYER's own push cuts.

  * **A cutter that ends up on the player's square is spat out on every press.**
    ``RIGHT [player cuttersright | ] -> [player | cuttersright ]`` carries no
    ``action`` keyword, so it is not part of the turn mechanic at all: it fires
    on arrow presses too. It is what makes the quarter turn look like one motion
    (the cutter is placed on the player and immediately ejected), and it is the
    reason a cutter turned toward the grid EDGE gets stuck riding the player
    forever -- there is no square to eject it into.

  * **Two handles cancel.** The ``beside`` object exists only to count them:
    with a second cutter presenting its handle, ACTION does nothing whatsoever.
    Zero handles is also nothing. Exactly one is the turn.

  * **The rule ORDER is the mechanic, not an implementation detail.**
    ``[> pushable | pushable]`` is listed before ``[> cutters | pushable]``, and
    each rule runs to fixpoint before the next one starts, so a block that got
    its force FROM a cutter never propagates that force to the block in front of
    it. A player can shove a run of three blocks; a cutter can shove exactly one.

Levels are 3x7 to 13x12, with one to six blocks and zero to four cutters. The
first six have no cutter at all and are pure typed sokoban; the last five are
the tool game.

Level numbers below are the 0-based indices every report here prints, so the
eleven rooms are 0 to 10.

Expert solver
-------------
A NATIVE model of the mechanic above (`_Board`), and TWO tiers on top of it. The
interpreter is not searched: it runs at 520-3 000 presses/s on these boards
against 180 000-370 000/s for the model -- 100x to 460x, and it is the big boards
where the gap is widest (``--speed`` measures it) -- while level 7's reachable
space alone is 1.5 million states and level 8's is past 3 million by depth 27,
against a win at 62.

TIER 1, the levels whose whole reachable ball fits `space_cap`: an EXACT
distance field, in two sweeps over the same space and with no successor graph
kept between them.

  * **forward.** Layered BFS from the level start, recording each state's depth,
    stopping at the first layer that contains a winning press. That depth is
    ``d*`` and it is shortest by construction.
  * **backward.** Re-expand the layers from ``d* - 1`` down to 0, marking a state
    GOOD when some press of it lands on a state that is one layer deeper and
    already GOOD (at ``d* - 1``, when some press wins outright). A state is GOOD
    exactly when it can still finish in the moves its own layer leaves, so the
    optimal SET at any state on the plan is every press that lands on a GOOD
    successor -- measured, not inferred.

Keeping only ``depth`` between the sweeps is what makes the field affordable: a
successor graph over 1.5 million states with five edges each is gigabytes, while
one dict of packed keys is ~220 MB. The second sweep costs one more expansion of
the same states, i.e. the field is 2x the BFS and not 2x the memory.

TIER 2, the levels that do not fit: a width-capped layered BEAM over the same
model, ordered by a matched PUSH-distance heuristic (`_Geo`), then SHORTENED by
deleting model-verified blocks longest-first. Its plan is a genuine WIN, replayed
press by press through the interpreter like every other; what it is NOT is proved
shortest, and every report here says so where such a level appears. Its optimal
sets come from `_replay_ties` -- the presses that still finish in the same number
of moves USING THIS PLAN'S OWN TAIL, which is sound but cannot see a tie that
leaves the route.

Where that lands, and it is the whole result (``--plans``, and ``--engine``
replays all ten through the interpreter):

    L0   3x7   1 block   0 cutters    3 presses  PROVED SHORTEST (4 states)
    L1   7x5   3 blocks  0 cutters   11 presses  PROVED SHORTEST (675)
    L2   6x8   2 blocks  0 cutters   13 presses  PROVED SHORTEST (2 059)
    L3   7x7   6 blocks  0 cutters   22 presses  PROVED SHORTEST (2 554 581)
    L4   7x6   6 blocks  0 cutters   24 presses  PROVED SHORTEST (619 142)
    L5   9x7   9 blocks  0 cutters   46 presses  beam
    L6   9x12  1 block   1 cutter    24 presses  PROVED SHORTEST (3 930)
    L7  11x12  1 block   4 cutters   73 presses  PROVED SHORTEST (1 497 030)
    L8   8x12  3 blocks  2 cutters   62 presses  beam
    L9   9x20  3 blocks  4 cutters       UNSOLVED
    L10 13x12  1 block   4 cutters  111 presses  beam

``--ties`` re-derives ``d*`` and every optimal set a SECOND way -- the whole
successor graph plus a reverse BFS from the winning edges, which shares no code
with `_field` beyond `_Board.step` -- and agrees on every level whose FULL space
fits its cap (0, 1, 2 and 6; the rest are bigger than the ball `_field` needed,
because a reverse BFS cannot stop at ``d*``).

LEVEL 9 IS NOT SOLVED and is dropped from every episode rather than taped as a
loss. ``--space 9`` is what is known: 3 055 084 states swept with no win found,
which proves only that no solution of 103 presses or fewer exists. The reason it
is believed to have none at all is an accounting one. Its three blocks
sit in a pocket whose only exits are the three wall squares above them, and a
wall costs one cutter; of the four cutters on the board, the one at row 6 has a
wall on three sides and a wall behind the fourth, so the ONLY press it can ever
take is the shove that spends it on that wall, and two more sit in a sealed room
that itself costs a cutter to open. Three cuts are needed and at most two can be
paid for. That is an argument, not a proof, and it is written down here because
it is the thing a future search would have to defeat.

Rendering
---------
The win condition for iron was INVISIBLE: ``iron`` (``#d7dfed #b9c1ce``) and
``ironmarker`` (``#b7b7b7 #cecccc``) both quantize to ARC 1 in every one of
their pixels, so an iron block sitting on an iron marker -- half of what levels
4, 5, 8 and 10 are about -- rendered as exactly the same flat block of colour 1
as the bare marker. ``Wall`` (``#010c21``) was ARC 5, which is also the colour
`_render_frame` letterboxes with, so the game's one DESTRUCTIBLE terrain had no
seam against the edge of the picture. Both fixes, and the four more objects
repainted to keep every material distinct from the floor it stands on, are in
the header comment of ``data/puzzlescript_games/The_Workshop.txt``.

``--audit`` is the check: it renders all 105 cell COMPOSITIONS a board of this
game can hold (each marker x each body x each cutter, plus cutter-on-wall) as a
whole 64x64 frame of a uniform board, at every cell size the eleven levels
render at (3, 4, 5, 7, 8 and 9 px), and requires them pairwise distinct and none
of them equal to the letterbox pad. Whole frames rather than one cell sliced out
of a mixed board: `_render_frame` upscales and centre-pads, so slicing by
``cell_px`` arithmetic reads the wrong pixels (the ps:explod lesson).

Augmentation
------------
Rotation, and NOT the flips. The quarter turn ACTION applies to a cutter is
CLOCKWISE, and a mirror takes clockwise to counter-clockwise while leaving the
cutter art alone -- ``cuttersup`` is its own horizontal mirror, and
``cuttersleft``/``cuttersright`` are each other's. So a mirrored presentation
would show the identical picture (a cuttersup above the player) for two
different outcomes (the cutter appears on the right, or on the left), with
nothing in the frame to say which, and a board that is itself mirror-symmetric
makes that ambiguity exact rather than theoretical. Rotation has no such
problem: a quarter turn maps clockwise to clockwise and maps each cutter's art
exactly onto its neighbour's in the same orbit, which ``--audit``'s second pass
measures pixel for pixel. ``--symmetry`` measures the whole claim end to end,
replaying every plan and a seeded random walk at all four rotations and
requiring each frame to be the exact transform of the unrotated one.

Usage (run from the repo root):
    python solvers/generate_the_workshop_training.py --episodes 200 \
        --out data/training_multi_level/the_workshop

    python solvers/generate_the_workshop_training.py --plans      # level report
    python solvers/generate_the_workshop_training.py --fuzz 20000 # model vs engine
    python solvers/generate_the_workshop_training.py --engine     # plans replayed
    python solvers/generate_the_workshop_training.py --ties       # optimal sets
    python solvers/generate_the_workshop_training.py --audit      # rendering
    python solvers/generate_the_workshop_training.py --symmetry   # augmentation
    python solvers/generate_the_workshop_training.py --space 9    # a level's space
    python solvers/generate_the_workshop_training.py --speed      # why native

The eleven searches take ~10 minutes and ~750 MB at a COLD start and are then
memoised to ``data/the_workshop_plans.json``; delete that file to re-derive
them.
"""

from __future__ import annotations

import heapq
import itertools
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arcengine import ActionInput, GameState                          # noqa: E402
from solvers.base_solver import _ID_TO_GAMEACTION                     # noqa: E402
from solvers.common.ps_astar import (PSAStarSolver, PSExpert, Plan,   # noqa: E402
                                     screen_action)

_REPO_ROOT = Path(__file__).resolve().parent.parent

GAME_NAME = "The_Workshop"

#: Engine actions, in the order ties are broken -- which is what makes a
#: re-derived plan byte-identical across processes.
DIRS: tuple[str, ...] = ("up", "down", "left", "right", "action")

UP, DOWN, LEFT, RIGHT, ACT = 0, 1, 2, 3, 4

#: ``ACTION`` swings the cutter a quarter turn CLOCKWISE around the player: the
#: one above (handle down, against you) ends up on your right, facing right.
CW: tuple[int, ...] = (RIGHT, LEFT, UP, DOWN)

#: Materials are indexed 0/1/2 by these tuples, and a marker's index is its
#: material's -- so ``push[cell] == markers[cell]`` IS the win test for a square.
_MATERIAL = ("stone", "wood", "iron")
_MARKER = ("stonemarker", "woodmarker", "ironmarker")
_CUTTER = ("cuttersup", "cuttersdown", "cuttersleft", "cuttersright")

#: Disk cache of every level's start plan AND its optimal-action sets. Load
#: bearing here: the exact field for level 7 alone is ~100 s and ~250 MB, and
#: without a file on disk every `parallelize_generator` shard would re-derive
#: all eleven. Delete the file to re-derive it.
PLAN_CACHE: Path = _REPO_ROOT / "data" / "the_workshop_plans.json"


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

class _Board:
    """One level's state, plus the mechanic. Cells are flat ``r * W + c``.

    Four things move or vanish: the player, the WALLS (a cutter destroys them),
    the pushable BLOCKS (never destroyed, so their per-material counts are fixed
    -- `_Codec` leans on that) and the CUTTERS, each carrying the direction its
    blade faces. Markers never change and are shared by reference between every
    clone of a level.

    A cutter is on its own collision layer, so ``cut`` may hold a cell that is
    also a wall, also a block, or also the player: all three overlaps are
    reachable and all three mean something different, which is why they are
    separate dicts rather than one occupancy map.

    Every method below is a transcription of one rule of
    ``data/puzzlescript_games/The_Workshop.txt`` in the order the interpreter
    applies them, because in this game that order IS the mechanic (see the
    module docstring). `_step_move` names each rule it is implementing.
    """

    __slots__ = ("H", "W", "markers", "player", "walls", "push", "cut")

    def __init__(self, H, W, markers, player, walls, push, cut):
        self.H, self.W = H, W
        self.markers = markers            # {cell: material}, static
        self.player = player              # cell
        self.walls = walls                # set[cell]
        self.push = push                  # {cell: material}
        self.cut = cut                    # {cell: facing}

    def clone(self) -> "_Board":
        return _Board(self.H, self.W, self.markers, self.player,
                      set(self.walls), dict(self.push), dict(self.cut))

    def key(self):
        return (self.player, frozenset(self.walls),
                frozenset(self.push.items()), frozenset(self.cut.items()))

    def won(self) -> bool:
        """``all <m>marker on <m>`` for all three materials. Spare blocks are
        allowed; a marker with the WRONG material on it is not."""
        push = self.push
        for cell, material in self.markers.items():
            if push.get(cell) != material:
                return False
        return True

    def ahead(self, cell, d):
        """The neighbour of ``cell`` in direction ``d``, or None off-grid."""
        W = self.W
        if d == UP:
            return cell - W if cell >= W else None
        if d == DOWN:
            nxt = cell + W
            return nxt if nxt < self.H * W else None
        if d == LEFT:
            return cell - 1 if cell % W else None
        nxt = cell + 1
        return nxt if nxt % W else None

    # -- the turn ---------------------------------------------------------
    def step(self, d) -> bool:
        """Apply one press. Returns True when the board changed."""
        return self._step_action() if d == ACT else self._step_move(d)

    def _eject(self, fcut) -> None:
        """``RIGHT [player cuttersright | ] -> [player | cuttersright]`` (x4).

        No ``action`` keyword, so this fires on EVERY press: a cutter sharing the
        player's square is shoved one cell along its blade, overwriting whatever
        cutter was there. At the grid edge there is no square to shove it into
        and the pattern simply does not match, which is how a cutter gets stuck
        riding the player.

        A force in the interpreter is keyed on ``(row, col, object index)``, so
        overwriting a cutter that this turn's rules had already FORCED discards
        that force -- the object it belonged to is gone -- unless the newcomer
        faces the same way and is therefore the same object index, in which case
        the key still matches and the force is inherited. Both halves are
        measured: `_fuzz` found the first one (a cutter ejected onto a forced
        cuttersleft, which then travelled on a force that no longer had an owner).
        """
        p = self.player
        face = self.cut.get(p)
        if face is None:
            return
        tgt = self.ahead(p, face)
        if tgt is not None:
            del self.cut[p]
            if self.cut.get(tgt, face) != face:
                fcut.discard(tgt)
            self.cut[tgt] = face

    def _cleanup(self, fcut) -> None:
        """``[cutters wall] -> [wall]`` then ``[cutters pushable] -> [pushable]``.

        The last two rules of the file, so they run BEFORE this turn's movement:
        an overlap that movement is about to create survives one full press.
        """
        cut, walls, push = self.cut, self.walls, self.push
        for cell in [c for c in cut if c in walls]:
            del cut[cell]
            fcut.discard(cell)
        for cell in [c for c in cut if c in push]:
            del cut[cell]
            fcut.discard(cell)

    def _step_move(self, d) -> bool:
        before = self.key()
        walls, push, cut = self.walls, self.push, self.cut
        p = self.player
        n1 = self.ahead(p, d)
        n2 = self.ahead(n1, d) if n1 is not None else None
        fpush, fcut = set(), set()

        if n1 is not None and n1 in push:                      # R1  player->block
            fpush.add(n1)
        if n1 is not None and n1 in cut:                       # R2  player->cutter
            fcut.add(n1)
        x = n1                                                 # R3  cutter->cutter
        while x is not None and x in fcut:
            y = self.ahead(x, d)
            if y is not None and y in cut and y not in fcut:
                fcut.add(y)
                x = y
            else:
                break

        # R4: shoving a cutter whose blade does NOT point that way into a wall
        # cancels the entire turn.
        if n1 is not None and n1 in cut and cut[n1] != d and n2 is not None \
                and n2 in walls:
            return False
        # R5: otherwise that cutter eats itself and the wall.
        if n1 is not None and n1 in cut and n2 is not None and n2 in walls:
            del cut[n1]
            fcut.discard(n1)
            walls.discard(n2)
        # R6: a forced cutter shoving a second cutter into a wall destroys THAT
        # cutter and the wall, whatever either of them is facing, and survives.
        again = True
        while again:
            again = False
            for x in sorted(fcut):
                if x not in cut:
                    continue
                y = self.ahead(x, d)
                if y is None or y not in cut:
                    continue
                z = self.ahead(y, d)
                if z is None or z not in walls:
                    continue
                del cut[y]
                fcut.discard(y)
                walls.discard(z)
                again = True
                break

        x = n1                                                 # R7  block->block
        while x is not None and x in fpush:
            y = self.ahead(x, d)
            if y is not None and y in push and y not in fpush:
                fpush.add(y)
                x = y
            else:
                break
        for x in sorted(fpush):                                # R8  block->cutter
            y = self.ahead(x, d)
            if y is not None and y in cut:
                fcut.add(y)
        # R9 block-behind-a-cutter runs AFTER R7 has already reached fixpoint, so
        # the force it hands out never propagates further: a cutter shoves one
        # block, never a run.
        for x in sorted(fcut):                                 # R9  cutter->block
            y = self.ahead(x, d)
            if y is not None and y in push:
                fpush.add(y)

        self._eject(fcut)                                      # R14
        self._cleanup(fcut)                                    # R15, R16
        self._move(d, fpush, fcut)
        return self.key() != before

    def _move(self, d, fpush, fcut) -> None:
        """Force resolution. Every force in this game points the same way, so
        each collision layer is a set of disjoint runs and one front-to-back
        sweep resolves them exactly -- the frontmost mover is the only one whose
        target can be occupied by something that will not move."""
        walls, push, cut = self.walls, self.push, self.cut
        front = (lambda c: c) if d in (UP, LEFT) else (lambda c: -c)

        p0 = self.player
        for cell in sorted(fpush | {p0}, key=front):     # layer 2
            tgt = self.ahead(cell, d)
            if tgt is None or tgt in walls or tgt in push:
                continue
            if cell == p0:
                self.player = tgt
            else:
                push[tgt] = push.pop(cell)
        for cell in sorted(fcut, key=front):             # layer 3
            if cell not in cut:
                continue
            tgt = self.ahead(cell, d)
            if tgt is None or tgt in cut:
                continue
            cut[tgt] = cut.pop(cell)

    def _step_action(self) -> bool:
        before = self.key()
        cut = self.cut
        p = self.player
        # R10/R11: the `beside` object counts handles. Two cancels the turn.
        handles = [(dd, n) for dd in (UP, DOWN, LEFT, RIGHT)
                   for n in (self.ahead(p, dd),)
                   if n is not None and cut.get(n) == dd]
        if len(handles) >= 2:
            return False
        if handles:
            dd, src = handles[0]
            del cut[src]
            cut[p] = CW[dd]                   # R13: onto the player's own square
        self._eject(set())                    # R14: and straight back off it
        self._cleanup(set())                  # R15, R16
        return self.key() != before


class _Codec:
    """Packed, DECODABLE state key.

    Blocks are never destroyed, so their per-material counts are fixed and the
    block section has a known layout; cutters and walls both shrink, so they get
    a ``255`` sentinel each. A state is ~15-30 bytes here against ~700 for the
    tuple-of-frozensets `_Board.key` returns, which is the difference between a
    1.5-million-state field fitting in memory and not.

    ``unpack`` is exactly ``pack``'s inverse -- `_field` re-seats every state it
    expands by decoding rather than by keeping a snapshot -- so ``--fuzz``
    round-trips every board it builds through both.
    """

    def __init__(self, b0: _Board):
        self.H, self.W = b0.H, b0.W
        self.markers = b0.markers
        self.types = sorted(set(b0.push.values()) | set(b0.markers.values()))
        self.counts = [sum(1 for t in b0.push.values() if t == ty)
                       for ty in self.types]
        assert b0.H * b0.W < 255, "cell index must fit in a byte"

    def pack(self, b: _Board) -> bytes:
        out = [b.player]
        for ty in self.types:
            out.extend(sorted(c for c, t in b.push.items() if t == ty))
        out.append(255)
        for c in sorted(b.cut):
            out.append(c)
            out.append(b.cut[c])
        out.append(255)
        out.extend(sorted(b.walls))
        return bytes(out)

    def unpack(self, k: bytes) -> _Board:
        player, i, push = k[0], 1, {}
        for ty, n in zip(self.types, self.counts):
            for c in k[i:i + n]:
                push[c] = ty
            i += n
        i += 1                                           # sentinel
        cut = {}
        while k[i] != 255:
            cut[k[i]] = k[i + 1]
            i += 2
        return _Board(self.H, self.W, self.markers, player,
                      set(k[i + 1:]), push, cut)


# ---------------------------------------------------------------------------
# Tier 1: the exact distance field
# ---------------------------------------------------------------------------

def _field(b0: _Board, codec: _Codec, cap: int, report=None):
    """``(presses, optsets, states, d*)`` -- a provably shortest plan and the
    COMPLETE optimal set at each of its steps -- or None past ``cap`` states.

    Two sweeps, no successor graph kept between them; see the module docstring
    for why that is the whole trick. A level whose ball is exhausted without a
    winning press is UNSOLVABLE and that is a proof, not a timeout: it returns
    ``(None, None, states, None)``.
    """
    k0 = codec.pack(b0)
    dist = {k0: 0}
    layers = [[k0]]
    dstar = None
    while layers[-1]:
        depth = len(layers) - 1
        nxt = []
        for k in layers[depth]:
            b = codec.unpack(k)
            for d in range(5):
                nb = b.clone()
                if not nb.step(d):
                    continue
                if nb.won():
                    dstar = depth + 1
                    break
                nk = codec.pack(nb)
                if nk in dist:
                    continue
                # Checked per STATE, not per layer: a layer of this game roughly
                # DOUBLES, so a layer-granular check overshoots the memory
                # budget by the whole branching factor before it fires.
                if len(dist) >= cap:
                    return None
                dist[nk] = depth + 1
                nxt.append(nk)
            if dstar is not None:
                break
        if dstar is not None:
            break
        layers.append(nxt)
        if report:
            report(f"      forward depth {depth + 1}: {len(dist)} states")
    if dstar is None:
        return None, None, len(dist), None

    # Backward: GOOD = "can still finish in the moves this layer leaves".
    good: set[bytes] = set()
    for depth in range(dstar - 1, -1, -1):
        for k in layers[depth]:
            b = codec.unpack(k)
            for d in range(5):
                nb = b.clone()
                if not nb.step(d):
                    continue
                if nb.won():
                    if depth + 1 == dstar:
                        good.add(k)
                        break
                    continue
                nk = codec.pack(nb)
                if dist.get(nk) == depth + 1 and nk in good:
                    good.add(k)
                    break
        if report:
            report(f"      backward depth {depth}: {len(good)} good")

    presses, optsets = [], []
    b = b0.clone()
    for depth in range(dstar):
        best = []
        for d in range(5):
            nb = b.clone()
            if not nb.step(d):
                continue
            if nb.won():
                if depth + 1 == dstar:
                    best.append(d)
                continue
            nk = codec.pack(nb)
            if dist.get(nk) == depth + 1 and nk in good:
                best.append(d)
        assert best, "a state on a shortest path always has an optimal press"
        presses.append(best[0])
        optsets.append(best)
        b.step(best[0])
    assert b.won()
    return presses, optsets, len(dist), dstar


# ---------------------------------------------------------------------------
# Tier 2: the beam
# ---------------------------------------------------------------------------

BIG = 200


class _Geo:
    """Push-distance tables for the beam's heuristic, memoised per WALL SET.

    ``tables()[m][c]`` is the cost of bringing a block from ``c`` onto marker
    ``m``, ignoring the other blocks: one per push, plus `WALL_COST` for each
    wall the route would have to pass through -- either under the block or under
    the square the player has to stand on to shove it. Charging walls instead of
    forbidding them is what makes the tables usable on the cutter levels: with
    walls impassable, level 10's single iron marker is simply UNREACHABLE from
    its block until the right two walls are cut, so the heuristic is a flat
    constant over the entire search and the beam has nothing to steer with (it
    wanders for 240 layers and 1.4 M states, measured). With them charged, the
    beam is pulled toward the block's real route AND rewarded the moment a cut
    opens it.

    The player-square term is what tells the beam that a block on a room's outer
    ring is finished travelling -- the square it would be shoved from is the
    wall -- which is the deadlock that loses most of these levels.

    Walls only ever disappear, and only a handful of times per level (once per
    cutter), so the memo holds a handful of entries and the tables stay honest
    about the board the state is actually on.
    """

    #: Presses a cut is worth, roughly: fetch a cutter, aim it, shove it. Only
    #: the ORDER it induces matters, so it is not tuned finely.
    WALL_COST = 8

    def __init__(self, b0: _Board):
        H, W = b0.H, b0.W
        self.n = H * W
        self.nb = [[-1] * 4 for _ in range(self.n)]
        for c in range(self.n):
            r, cc = divmod(c, W)
            for d, (dr, dc) in enumerate(((-1, 0), (1, 0), (0, -1), (0, 1))):
                nr, nc = r + dr, cc + dc
                if 0 <= nr < H and 0 <= nc < W:
                    self.nb[c][d] = nr * W + nc
        self.markers = b0.markers
        self.by_type: dict[int, list[int]] = {}
        for m, t in self.markers.items():
            self.by_type.setdefault(t, []).append(m)
        self._tabs: dict[frozenset, dict] = {}

    def tables(self, walls):
        key = frozenset(walls)
        got = self._tabs.get(key)
        if got is None:
            got = {m: self._push_table(m, walls) for m in self.markers}
            self._tabs[key] = got
        return got

    def _push_table(self, target, walls):
        """Backward Dijkstra over pushes, walls charged rather than forbidden."""
        nb, opp, cost = self.nb, (DOWN, UP, RIGHT, LEFT), self.WALL_COST
        dist = {target: 0}
        pq = [(0, target)]
        while pq:
            g, c = heapq.heappop(pq)
            if g > dist.get(c, 1 << 30):
                continue
            for d in range(4):
                src = nb[c][opp[d]]              # a block here, pushed d, lands on c
                if src < 0:
                    continue
                behind = nb[src][opp[d]]         # the player has to stand here
                if behind < 0:
                    continue
                ng = g + 1 + cost * ((src in walls) + (behind in walls))
                if ng < dist.get(src, 1 << 30):
                    dist[src] = ng
                    heapq.heappush(pq, (ng, src))
        return dist

    def h(self, b: _Board) -> int:
        """Best matching of unsatisfied markers to blocks of their material,
        brute-forced per material (at most 4! per level). Not admissible -- a
        press moves a whole run of blocks at once -- so it orders the beam and
        nothing else."""
        tabs = self.tables(b.walls)
        total = 0
        for t, ms in self.by_type.items():
            todo = [m for m in ms if b.push.get(m) != t]
            if not todo:
                continue
            blocks = [c for c, ct in b.push.items() if ct == t]
            best = None
            for perm in itertools.permutations(blocks, len(todo)):
                v = sum(tabs[m].get(c, BIG) for m, c in zip(todo, perm))
                if best is None or v < best:
                    best = v
            total += (BIG * len(todo) if best is None else best) + len(todo)
        return total


def _beam(b0: _Board, codec: _Codec, geo: _Geo, width: int, max_depth: int,
          report=None):
    """Width-capped layered beam. Returns a winning press list or None."""
    frontier = [(b0, [])]
    seen = {codec.pack(b0)}
    for depth in range(max_depth):
        scored = []
        for b, path in frontier:
            for d in range(5):
                nb = b.clone()
                if not nb.step(d):
                    continue
                k = codec.pack(nb)
                if k in seen:
                    continue
                seen.add(k)
                if nb.won():
                    return path + [d]
                scored.append((geo.h(nb), nb, path + [d]))
        if not scored:
            return None
        scored.sort(key=lambda x: x[0])
        frontier = [(b, p) for _, b, p in scored[:width]]
        if report and depth % 20 == 0:
            report(f"      beam depth {depth}: h={scored[0][0]} "
                   f"{len(seen)} states")
    return None


def _replay_ties(b0: _Board, presses, window: int = 10):
    """Per-step tie sets for a plan nothing has proved shortest.

    Two tests, unioned, both of which finish the level in the SAME number of
    presses as the recorded plan (or fewer) and are therefore sound labels
    relative to it:

    * **substitute.** ``a`` changes the board and the plan's own remaining
      presses still win from where ``a`` lands.
    * **reorder.** Some later press of the plan can be brought FORWARD to this
      step and the rest replayed behind it. This is the one that finds the
      order-free stretches -- a walk to a cell can interleave its two axes any
      way at all -- and the substitute test structurally cannot, because the
      plan's tail there begins with the very press the alternative just used.

    What neither can find is a tie that leaves this plan's route entirely, which
    is why these are not `_field`'s measured sets and every report says so. The
    press the expert took is always in the set (the optimal-target contract in
    `BaseSolver`).

    Costs ``len(plan) * (5 + window) * len(plan)`` model steps -- a hundred
    thousand or so, i.e. nothing.
    """
    out = []
    b = b0.clone()
    for i, taken in enumerate(presses):
        tail = presses[i + 1:]
        best = {taken}
        for d in range(5):
            nb = b.clone()
            if not nb.step(d):
                continue
            if nb.won() or _wins(nb, tail):
                best.add(d)
        for j in range(i + 1, min(i + 1 + window, len(presses))):
            if presses[j] in best:
                continue
            cand = [presses[j]] + presses[i:j] + presses[j + 1:]
            if _wins(b, cand):
                best.add(presses[j])
        out.append(sorted(best))
        b.step(taken)
    return out


def _wins(b0: _Board, presses) -> bool:
    b = b0.clone()
    for d in presses:
        b.step(d)
        if b.won():
            return True
    return False


def _shorten(b0: _Board, presses, report=None):
    """Delete the longest block of presses the plan can do without, and repeat.

    A beam plan wanders -- it is steered by a heuristic that cannot see a whole
    detour -- and a deleted block is verified against the model, so what comes
    back is still a genuine win. It is not shortest and nothing here pretends it
    is."""
    presses = list(presses)
    improved = True
    while improved:
        improved = False
        for size in range(len(presses) - 1, 0, -1):
            for i in range(len(presses) - size + 1):
                cand = presses[:i] + presses[i + size:]
                if _wins(b0, cand):
                    presses = cand
                    improved = True
                    break
            if improved:
                if report:
                    report(f"      shortened by {size} -> {len(presses)}")
                break
    return presses


# ---------------------------------------------------------------------------
# The expert
# ---------------------------------------------------------------------------

class WorkshopExpert(PSExpert):
    """`PSExpert`'s plan memo, disk cache and snapshot discipline around the two
    tiers above. `_search` is replaced wholesale -- the interpreter is never
    stepped by the planner, only by the reports that certify what it produced --
    so `heuristic` is never called and asserts rather than returning a number
    nothing would use.
    """

    directions = list(DIRS)
    plan_cache_path = PLAN_CACHE

    #: States `_field` may hold before it gives up and the beam takes over. At
    #: ~250 bytes a state (the packed key, its dict slot and its layer slot) this
    #: is ~750 MB, which is the real budget; level 3 needs 2 554 581 of it and is
    #: the largest level that fits, while level 8 is still growing at 3 M by
    #: depth 27 against a 62-press beam win, which is the reason there is a
    #: second tier at all. The cap is per STATE rather than per layer: these
    #: layers roughly double, so a layer-granular check overshoots by that much
    #: -- a 12 M cap on level 9 took a 32 GB machine down before that fix.
    space_cap: int = 3_000_000

    beam_width: int = 8_000
    beam_depth: int = 400

    #: `PuzzleScriptAdapter` gives a level 200 presses before GAME_OVER, and the
    #: `set_level` that ends the exploration prefix restarts that count, so the
    #: plan gets all 200. A plan that does not fit is not a plan: the level is
    #: reported unsolved rather than taped into a losing episode.
    press_budget: int = 195

    def __init__(self, *args, **kwargs):
        self.method: dict[int, str] = {}
        self.stats: dict[int, dict] = {}
        self._level: int | None = None
        self.verbose: bool = False
        super().__init__(*args, **kwargs)

    # -- reading the engine ------------------------------------------------
    def setup(self) -> None:
        idx = self.g.obj_name_to_idx
        self._i_wall = idx["wall"]
        self._i_player = idx["player"]
        self._i_material = {idx[n]: i for i, n in enumerate(_MATERIAL)}
        self._i_marker = {idx[n]: i for i, n in enumerate(_MARKER)}
        self._i_cutter = {idx[n]: i for i, n in enumerate(_CUTTER)}

    def read(self, eng) -> _Board:
        """The engine grid as a `_Board`. Every object of this game is one of
        five kinds and none of them is hidden bookkeeping (`beside` exists only
        inside a rule pass and is gone before any frame is drawn), so this is a
        lossless read."""
        W = eng.width
        markers, walls, push, cut = {}, set(), {}, {}
        player = None
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                at = r * W + c
                for o in cell:
                    if o in self._i_marker:
                        markers[at] = self._i_marker[o]
                    elif o == self._i_wall:
                        walls.add(at)
                    elif o == self._i_player:
                        player = at
                    elif o in self._i_material:
                        push[at] = self._i_material[o]
                    elif o in self._i_cutter:
                        cut[at] = self._i_cutter[o]
        assert player is not None, "no player on the grid"
        return _Board(eng.height, W, markers, player, walls, push, cut)

    def heuristic(self, eng) -> int:
        raise AssertionError(
            "WorkshopExpert solves a native model; heuristic is unused")

    # -- planning ----------------------------------------------------------
    def plan(self, eng, level: int | None = None):
        """`PSExpert.plan` plus two lines of bookkeeping: remember which level
        `_search` is working on (it is handed only the engine) and keep the tier
        that produced each plan in the disk cache, so ``--plans`` can report
        "proved shortest" or "beam" from a warm cache without re-deriving it."""
        self._level = level
        got = super().plan(eng, level)
        entry = self._disk.get(level)
        if entry is not None:
            mine = self.method.get(level)
            if mine and entry.get("method") != mine:
                entry["method"] = mine
                entry["states"] = self.stats.get(level, {}).get("states")
                self._save_disk()
            elif entry.get("method"):
                self.method.setdefault(level, entry["method"])
                self.stats.setdefault(level, {"states": entry.get("states")})
        return got

    def _report(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def _search(self, eng):
        level = self._level
        b0 = self.read(eng)
        codec = _Codec(b0)
        t0 = time.time()

        got = _field(b0, codec, self.space_cap, self._report)
        if got is not None:
            presses, optsets, states, dstar = got
            self.stats[level] = {"states": states, "secs": time.time() - t0}
            if presses is None:
                self.method[level] = "unsolvable"
                self._report(f"      level {level}: PROVED UNSOLVABLE "
                             f"({states} states)")
                return None
            self.method[level] = "field"
            self._report(f"      level {level}: shortest {dstar} presses "
                         f"({states} states, {time.time() - t0:.0f}s)")
            if len(presses) > self.press_budget:
                self.method[level] = "over-budget"
                return None
            return Plan([DIRS[d] for d in presses],
                        [[DIRS[x] for x in s] for s in optsets])

        geo = _Geo(b0)
        presses = _beam(b0, codec, geo, self.beam_width, self.beam_depth,
                        self._report)
        if presses is None:
            self.method[level] = "beam-failed"
            self.stats[level] = {"secs": time.time() - t0}
            return None
        presses = _shorten(b0, presses, self._report)
        self.method[level] = "beam"
        self.stats[level] = {"secs": time.time() - t0}
        self._report(f"      level {level}: beam win in {len(presses)} presses "
                     f"({time.time() - t0:.0f}s)")
        if len(presses) > self.press_budget:
            self.method[level] = "over-budget"
            return None
        # Nothing PROVED the ties on a beam plan, but the cheap half of the
        # question -- which other presses this plan's own tail still finishes
        # from -- is worth asking, and it is never None (see the optimal-target
        # contract in `BaseSolver`).
        sets = _replay_ties(b0, presses)
        return Plan([DIRS[d] for d in presses],
                    [[DIRS[x] for x in st] for st in sets])


# ---------------------------------------------------------------------------
# The generator
# ---------------------------------------------------------------------------

class WorkshopSolver(PSAStarSolver):
    game_id = "puzzlescript_the_workshop"
    game_name = GAME_NAME
    expert_cls = WorkshopExpert

    #: `games/ps:the_workshop/ps:the_workshop.py` is a plain passthrough -- it
    #: builds the adapter and nothing else, and the rendering fix is in the .txt,
    #: which both paths read. Set this if that wrapper ever grows a patch.
    game_module_id = ""

    #: Unused: `WorkshopExpert._search` never calls `_astar`. Left at the base
    #: values so nothing reads a lie off them; `WorkshopExpert.space_cap` and
    #: ``beam_width`` are the knobs that bound the searches.
    weight = 1
    node_cap = 400_000

    #: Room for the longest plan plus the re-plans the recorder might ask for.
    #: The adapter's own 200-press per-level budget is separate and is restarted
    #: by the `set_level` that ends the exploration prefix, so the plan starts it
    #: from zero -- `WorkshopExpert.press_budget` is what keeps a plan inside it.
    max_steps = 260

    #: Nothing is skipped up front; `discover_solvable` finds the subset the two
    #: tiers can win, once, and it is seed-independent.
    skip_levels = frozenset()

    #: NOT raised above zero, unlike the small ps: games. `record_level`'s
    #: epsilon detour probes each alternative by asking the expert to re-plan
    #: from the state it lands in, and here that is a fresh exact field (a minute
    #: and hundreds of megabytes on level 7) from a state the disk cache has
    #: never seen. Recovery comes from the RESET prefix instead, which is what
    #: ``recovery_mode = "reset"`` is for.
    epsilon = 0.0

    def prepare_expert(self, game, expert) -> None:
        """Plan every level before `discover_solvable` asks for it -- the same
        work either way, but it fills the disk cache in one pass and makes the
        startup cost visible as startup."""
        expert.verbose = True
        for level in range(game.n_levels):
            game.set_level(level)
            expert.plan(game._engine, level)
        expert.verbose = False


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _new(seed: int = 0):
    """The solver, its adapter and an expert -- without planning anything.

    The reports below each want a different slice (``--audit`` needs no plan at
    all, ``--fuzz`` needs only the model), so planning is left to whoever asks
    for it rather than paid on every entry point."""
    solver = WorkshopSolver()
    game = solver.make_game(seed)
    return solver, game, WorkshopExpert(game)


def _ascii(b: _Board) -> str:
    """A model board as ASCII, so a failing check can print what it failed on.

    Upper case is a body on bare floor, lower case the same body on a marker;
    ``*``/``%``/``x`` are a bare stone / wood / iron marker, ``UDLR`` a cutter
    (drawn over whatever it shares the square with) and ``@`` a cutter riding
    the player."""
    out = []
    for r in range(b.H):
        line = ""
        for c in range(b.W):
            at = r * b.W + c
            marked = at in b.markers
            if at in b.cut and at == b.player:
                ch = "@"
            elif at in b.cut:
                ch = "UDLR"[b.cut[at]]
            elif at == b.player:
                ch = "p" if marked else "P"
            elif at in b.walls:
                ch = "#"
            elif at in b.push:
                ch = "OWI"[b.push[at]] if not marked else "owi"[b.push[at]]
            elif marked:
                ch = "*%x"[b.markers[at]]
            else:
                ch = "."
            line += ch
        out.append(line)
    return "\n".join(out)


def _plans(verbose: bool = True) -> int:
    """Every level: which tier solved it, how long the plan is and whether that
    length is PROVED shortest. Returns the number of unsolved levels."""
    solver, game, expert = _new()
    expert.verbose = verbose
    unsolved = 0
    print(f"{GAME_NAME}: {game.n_levels} levels, {solver.max_steps}-press "
          f"recorder budget, {game._max_steps}-press adapter budget")
    for level in range(game.n_levels):
        game.set_level(level)
        b0 = expert.read(game._engine)
        t0 = time.time()
        plan = expert.plan(game._engine, level)
        method = expert.method.get(level, "cached")
        states = (expert.stats.get(level) or {}).get("states")
        shape = f"{b0.H}x{b0.W}"
        pieces = (f"{len(b0.push)} blocks, {len(b0.cut)} cutters, "
                  f"{len(b0.markers)} markers")
        if plan is None:
            unsolved += 1
            print(f"  L{level:<2d} {shape:>6s} {pieces:<38s} UNSOLVED ({method})")
            continue
        claim = ("PROVED SHORTEST" if method == "field" else
                 "a win, NOT proved shortest" if method == "beam" else
                 "shortest (cached)" if _all_sets(plan) else "cached")
        room = "ok" if len(plan) <= expert.press_budget else "OVER BUDGET"
        extra = f", {states} states" if states else ""
        print(f"  L{level:<2d} {shape:>6s} {pieces:<38s} "
              f"{len(plan):3d} presses  {claim}{extra}  [{room}] "
              f"{time.time() - t0:.1f}s")
    return unsolved


def _all_sets(plan) -> bool:
    """True when a cached plan carries a measured tie somewhere -- the only tell
    that separates a `_field` plan from a `_beam` one once both are on disk and
    the ``method`` key predates this run."""
    sets = getattr(plan, "optsets", None)
    return bool(sets) and any(len(s) > 1 for s in sets)


def _to_engine(b: _Board, eng, idx) -> None:
    """Seat a model board on the interpreter (``--fuzz`` drives both)."""
    bg = idx["background"]
    eng.height, eng.width = b.H, b.W
    grid = [[{bg} for _ in range(b.W)] for _ in range(b.H)]

    def at(cell):
        return grid[cell // b.W][cell % b.W]

    for cell, t in b.markers.items():
        at(cell).add(idx[_MARKER[t]])
    for cell in b.walls:
        at(cell).add(idx["wall"])
    for cell, t in b.push.items():
        at(cell).add(idx[_MATERIAL[t]])
    for cell, f in b.cut.items():
        at(cell).add(idx[_CUTTER[f]])
    at(b.player).add(idx["player"])
    eng.load_level(grid)


def _random_board(rng, H, W, wall_p, dense) -> _Board:
    while True:
        cells = list(range(H * W))
        walls = {c for c in cells if rng.random() < wall_p}
        free = [c for c in cells if c not in walls]
        if len(free) < 6:
            continue
        rng.shuffle(free)
        player, i = free[0], 1
        push, cut, markers = {}, {}, {}
        hi = 6 if dense else 3
        for _ in range(min(rng.randint(0, hi), len(free) - i)):
            push[free[i]] = rng.randrange(3)
            i += 1
        for _ in range(min(rng.randint(0, hi), len(free) - i)):
            cut[free[i]] = rng.randrange(4)
            i += 1
        for _ in range(min(rng.randint(0, 2), len(free) - i)):
            markers[free[i]] = rng.randrange(3)
            i += 1
        # cutters that START on a wall or a block: both are reachable states
        # (movement makes the overlap and the next rule pass clears it), and
        # both are grabbable by ACTION before that pass, so they have to be
        # fuzzed rather than assumed away.
        if rng.random() < 0.2 and walls:
            cut[rng.choice(sorted(walls))] = rng.randrange(4)
        if rng.random() < 0.2 and push:
            cut[rng.choice(sorted(push))] = rng.randrange(4)
        return _Board(H, W, markers, player, walls, push, cut)


def _fuzz(boards: int = 5000, presses: int = 25, verbose: bool = True) -> int:
    """The model against the interpreter, whole board compared after EVERY
    press, on the eleven shipped levels and on random boards.

    Random play alone would never reach the states that decide this game -- a
    cutter riding the player, a cutter sitting on a wall, two handles at once --
    so `_random_board` seeds them directly and the density is swept from sparse
    to nearly solid walls. The codec is round-tripped on every state as well:
    `_field` re-seats states by DECODING them, so a codec that is not exactly
    its own inverse would silently search a different game.
    """
    _solver, game, expert = _new()
    eng, idx = game._engine, game._game.obj_name_to_idx
    rng = random.Random("the_workshop:fuzz")
    bad = 0

    def compare(b0: _Board, seq) -> int:
        _to_engine(b0, eng, idx)
        model, prev = b0.clone(), b0.clone()
        codec = _Codec(b0)
        for d in seq:
            eng.step(DIRS[d])
            model.step(d)
            ref = expert.read(eng)
            if ref.key() != model.key() or ref.won() != model.won():
                print(f"    MISMATCH on press {'UDLRA'[d]}")
                print("    before:\n" + _ascii(prev))
                print("    engine:\n" + _ascii(ref))
                print("    model:\n" + _ascii(model))
                return 1
            if codec.unpack(codec.pack(model)).key() != model.key():
                print("    CODEC round-trip failed on:\n" + _ascii(model))
                return 1
            prev = ref
        return 0

    for level in range(game.n_levels):
        game.set_level(level)
        b0 = expert.read(eng)
        seq = [rng.randrange(5) for _ in range(400)]
        bad += compare(b0, seq)
    if verbose:
        print(f"  {game.n_levels} shipped levels x 400 random presses: "
              f"{'OK' if not bad else 'FAILED'}")

    mism = 0
    for i in range(boards):
        b0 = _random_board(rng, rng.randint(3, 8), rng.randint(3, 9),
                           rng.choice([0.1, 0.25, 0.4, 0.55]), i % 2 == 0)
        mism += compare(b0, [rng.randrange(5) for _ in range(presses)])
    if verbose:
        print(f"  {boards} random boards x {presses} presses "
              f"({boards * presses} transitions): {mism} mismatched")
    return bad + mism


def _engine(verbose: bool = True) -> int:
    """Replay every plan through the real interpreter, press by press, and
    require ``GameState.WIN`` at the end and nowhere before it.

    This is what makes the native model a planning device rather than a claim:
    whatever `_field` and `_beam` believe, the shipped trajectory is the one the
    interpreter produces."""
    solver, game, expert = _new()
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        plan = expert.plan(game._engine, level)
        if plan is None:
            continue
        game.set_level(level)
        won_at = None
        for i, d in enumerate(plan):
            fd = game.perform_action(
                ActionInput(id=screen_action(d, game._rotation_k, game._hflip,
                                             game._vflip)))
            if fd.state == GameState.WIN and won_at is None:
                won_at = i + 1
            if fd.state == GameState.GAME_OVER:
                print(f"    L{level}: GAME_OVER after {i + 1} presses")
                bad += 1
                break
        if won_at != len(plan):
            print(f"    L{level}: won at press {won_at}, plan is "
                  f"{len(plan)} long")
            bad += 1
        elif verbose:
            print(f"  L{level}: interpreter WIN in {len(plan)} presses")
    return bad


def _ties(cap: int = 400_000, verbose: bool = True) -> int:
    """Re-derive ``d*`` and the optimal SETS a SECOND way and require agreement.

    `_field` gets its answer from forward layers plus a backward GOOD sweep that
    never materialises an edge. This check does the opposite: it builds the whole
    successor graph, runs a reverse BFS from the winning edges to get the true
    distance-to-win of EVERY state, and reads the sets straight off it. Same
    answer, no shared code path beyond `_Board.step`. It runs on the levels whose
    space fits ``cap``; the ones it skips say so."""
    _solver, game, expert = _new()
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        b0 = expert.read(game._engine)
        plan = expert.plan(game._engine, level)
        if plan is None or expert.method.get(level) not in ("field", None):
            if verbose:
                print(f"  L{level}: skipped ({expert.method.get(level)})")
            continue
        codec = _Codec(b0)
        k0 = codec.pack(b0)
        succ: dict[bytes, dict[int, bytes | None]] = {}
        q, seen = deque([k0]), {k0}
        over = False
        while q:
            k = q.popleft()
            b = codec.unpack(k)
            edges: dict[int, bytes | None] = {}
            for d in range(5):
                nb = b.clone()
                if not nb.step(d):
                    continue
                if nb.won():
                    edges[d] = None                # None is the WIN sink
                    continue
                nk = codec.pack(nb)
                if nk not in seen:
                    if len(seen) >= cap:
                        over = True
                        break
                    seen.add(nk)
                    q.append(nk)
                edges[d] = nk
            succ[k] = edges
            if over:
                break
        if over:
            if verbose:
                print(f"  L{level}: skipped (space over {cap})")
            continue
        rev: dict[bytes, list[bytes]] = {}
        dtw: dict[bytes, int] = {}
        frontier = []
        for k, edges in succ.items():
            for d, nk in edges.items():
                if nk is None:
                    if k not in dtw:
                        dtw[k] = 1
                        frontier.append(k)
                else:
                    rev.setdefault(nk, []).append(k)
        q = deque(frontier)
        while q:
            k = q.popleft()
            for p in rev.get(k, ()):
                if p not in dtw:
                    dtw[p] = dtw[k] + 1
                    q.append(p)
        if dtw.get(k0) != len(plan):
            print(f"    L{level}: reverse BFS says {dtw.get(k0)}, plan is "
                  f"{len(plan)}")
            bad += 1
            continue
        k, sets = k0, getattr(plan, "optsets", None)
        for i, taken in enumerate(plan):
            want = sorted(DIRS[d] for d, nk in succ[k].items()
                          if (1 if nk is None else 1 + dtw.get(nk, 1 << 30))
                          == dtw[k])
            got = sorted(sets[i]) if sets else [taken]
            if want != got:
                print(f"    L{level} step {i}: sets {got} != measured {want}")
                bad += 1
                break
            k = succ[k][DIRS.index(taken)]
        else:
            if verbose:
                print(f"  L{level}: d*={len(plan)} and every optimal set "
                      f"confirmed over {len(succ)} states")
    return bad


#: Every cell STACK a board of this game can hold: one marker layer, one
#: body layer (the player or a block), one cutter layer -- plus the wall, which
#: shares the body layer with everything except a cutter. ``wall + marker`` and
#: ``wall + block`` are unreachable (they are on one collision layer and no level
#: places a marker under a wall), so they are not listed.
def _compositions() -> list[tuple[str, ...]]:
    out: list[tuple[str, ...]] = [(), ("wall",)]
    for m in ((),) + tuple((x,) for x in _MARKER):
        for body in ((),) + (("player",),) + tuple((x,) for x in _MATERIAL):
            for cut in ((),) + tuple((x,) for x in _CUTTER):
                comp = m + body + cut
                if comp and comp not in out:
                    out.append(comp)
    out.extend(("wall", c) for c in _CUTTER)
    return out


def _comp_name(c) -> str:
    return "floor" if not c else "+".join(c)


def _audit(verbose: bool = True) -> int:
    """Two passes over every reachable cell COMPOSITION.

    **Pass 1 -- distinctness**, at every cell size the eleven boards use (3, 4,
    5, 7, 8 and 9 px; the size list is derived from the levels rather than
    assumed, so an edited level is covered). Whole 64x64 frames of uniform boards
    are compared rather than one cell out of a mixed board: `_render_frame`
    upscales and centre-pads, so slicing a cell by ``cell_px`` arithmetic reads
    the wrong pixels (the ps:explod lesson). It also asserts that no composition
    renders as the letterbox PAD colour, which a pairwise matrix structurally
    cannot catch -- the other half of that pair is not a composition (the
    ps:stand_iii lesson, and it is exactly what `Wall` was doing here).

    This is the pass that fails on the shipped .txt: ``iron`` and ``ironmarker``
    were the same flat block of ARC 1, so an iron marker looked identical
    satisfied and unsatisfied -- five clashes at every one of the eleven board
    shapes, all of them that one identity. See the header comment in
    ``data/puzzlescript_games/The_Workshop.txt``.

    **Pass 2 -- the quarter-turn orbit.** This game takes the rotation
    augmentation (and not the flips -- see the module docstring), so a board is
    drawn at any of four presentations. Every composition's art may turn into art
    the game owns, and for the CUTTERS it must: a quarter turn has to map
    ``cuttersup`` exactly onto ``cuttersright``, because that is what the same
    turn does to the mechanic. Any other composition whose rotation lands on a
    DIFFERENT composition would be a frame that lies, so the pass lists the
    orbit it measured and counts anything outside it. It runs on SQUARE boards,
    because a transform of a non-square board also moves the letterbox and every
    comparison would pass for the wrong reason.
    """
    from adapters.puzzlescript_adapter import _render_frame            # noqa: PLC0415

    _solver, game, _expert = _new()
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx

    sizes: dict = {}
    for level in range(game.n_levels):
        game.set_level(level)
        sizes.setdefault((eng.height, eng.width), []).append(level)
    comps = _compositions()

    def shoot(h, w, comp):
        eng.height, eng.width = h, w
        cell = {idx["background"]} | {idx[o] for o in comp}
        eng.grid = [[set(cell) for _ in range(w)] for _ in range(h)]
        eng._position_index_dirty = True
        return np.asarray(_render_frame(eng, g)).copy()

    bad = 0
    for (h, w), levels in sorted(sizes.items()):
        shots = {c: shoot(h, w, c) for c in comps}
        clashes = [(a, b) for a, b in itertools.combinations(comps, 2)
                   if np.array_equal(shots[a], shots[b])]
        pad = [c for c in comps
               if len(np.unique(shots[c])) == 1 and shots[c][0][0] == 5]
        bad += len(clashes) + len(pad)
        if verbose:
            print(f"  {h:2d}x{w:2d} (cell {min(64 // h, 64 // w)}px, levels "
                  f"{','.join(str(x) for x in levels)}): {len(shots)} "
                  f"compositions, {'OK' if not clashes and not pad else 'CLASH'}")
        for a, b in clashes:
            print(f"      IDENTICAL: {_comp_name(a)}  ==  {_comp_name(b)}")
        for c in pad:
            print(f"      PAD COLOUR: {_comp_name(c)} is the letterbox")

    #: The quarter turn permutes exactly these -- a cutter's art onto the next
    #: cutter's, in every stack a cutter can sit in. ``np.rot90(k=1)`` turns the
    #: array counter-clockwise, which is the sense `_present_frame` uses, so a
    #: cuttersup lands on the cuttersLEFT art.
    turn = {"cuttersup": "cuttersleft", "cuttersleft": "cuttersdown",
            "cuttersdown": "cuttersright", "cuttersright": "cuttersup"}

    def expected(comp, k):
        out = list(comp)
        for _ in range(k % 4):
            out = [turn.get(o, o) for o in out]
        return tuple(out)

    #: The compositions whose NON-cutter part is rotation invariant: solid
    #: markers, the solid wall, bare floor. Only for these can a turned frame be
    #: required to land on another composition's art -- the player and the three
    #: block sprites are little pictures with a top and a bottom, so their turned
    #: art is art the game does not own, which is fine (the whole frame turns
    #: together) as long as it is not some OTHER composition's art.
    invariant = [c for c in comps
                 if not ({"player", *_MATERIAL} & set(c))]

    for cell in sorted({min(64 // h, 64 // w) for h, w in sizes}):
        n = 64 // cell
        shots = {c: shoot(n, n, c) for c in comps}
        wrong = orbit = 0
        for a in comps:
            for k in (1, 2, 3):
                turned = np.ascontiguousarray(np.rot90(shots[a], k=k))
                want = expected(a, k)
                hits = [b for b in comps if np.array_equal(turned, shots[b])]
                if a in invariant:
                    # A closed orbit is REQUIRED here: a quarter turn maps the
                    # mechanic cuttersup -> cuttersleft, so it has to map the art
                    # the same way or a turned board shows the wrong tool.
                    if hits != [want]:
                        wrong += 1
                        print(f"      {_comp_name(a)} turned {k}x should be "
                              f"exactly {_comp_name(want)}, matched "
                              f"{[_comp_name(x) for x in hits]}")
                    else:
                        orbit += want != a
                elif hits not in ([], [want]):
                    # A turned rock is a turned rock; it just must not be some
                    # other composition.
                    wrong += 1
                    print(f"      {_comp_name(a)} turned {k}x collides with "
                          f"{[_comp_name(x) for x in hits]}")
        bad += wrong
        if verbose:
            print(f"  cell {cell}px ({n}x{n}): quarter turn is a closed orbit "
                  f"on {len(invariant)} invariant compositions "
                  f"({orbit} moved), {wrong} wrong")
    return bad


def _symmetry(walk_presses: int = 150, verbose: bool = True) -> int:
    """Replay every level through the ADAPTER at every presentation it can draw
    and require the frames to be exactly the transform of the unrotated ones.

    This is the evidence for the rotation augmentation. The structural argument
    is in the module docstring (and the reason the FLIPS are refused);
    ps:gobble_rush's chirality hid inside exactly that kind of argument, so it is
    measured. Both the PLANS and a seeded random walk are replayed: the walk
    reaches the boards a plan never visits -- a cutter shoved flat into a wall it
    cannot cut, ACTION pressed with two handles or none, a block wedged on a
    room's outer ring -- and it presses every button, not only the ones a plan
    uses."""
    solver, game, expert = _new()
    plans = {}
    for level in range(game.n_levels):
        game.set_level(level)
        plans[level] = expert.plan(game._engine, level) or []

    def drive(g, level, presses):
        g.set_level(level)
        out = [np.asarray(g._current_frame)]
        for act in presses:
            fd = g.perform_action(ActionInput(id=act))
            out.append(np.asarray(fd.frame[-1] if fd.frame else g._current_frame))
        return out

    ref, seen, bad = {}, set(), 0
    for seed in range(200):
        if len(seen) == 4 and seed > 40:
            break
        g = solver.make_game(seed)
        for level in range(g.n_levels):
            g.set_level(level)
            k = (g._rotation_k, g._hflip, g._vflip)
            seen.add(k)
            if k[1] or k[2]:
                print(f"    seed {seed} L{level}: a FLIP was presented; this "
                      f"game must not be in _FLIP_GAMES")
                bad += 1
                continue
            rng = random.Random(f"the_workshop:symmetry:{level}")
            walk = [_ID_TO_GAMEACTION[rng.randrange(1, 6)]
                    for _ in range(walk_presses)]
            from utils.rotation import inverse_remap_action_full        # noqa: PLC0415
            for tag, presses in (
                    ("plan", [screen_action(d, *k) for d in plans[level]]),
                    ("walk", [inverse_remap_action_full(a, *k) for a in walk])):
                frames = drive(g, level, presses)
                if tag == "plan" and plans[level] and g._state != GameState.WIN:
                    print(f"    seed {seed} L{level} {k}: plan did not win")
                    bad += 1
                key = (level, tag)
                if key not in ref:
                    if k != (0, False, False):
                        continue          # nothing to compare against yet
                    ref[key] = frames
                    continue
                if any(not np.array_equal(np.ascontiguousarray(
                        np.rot90(a, k=k[0])), b)
                        for a, b in zip(ref[key], frames)):
                    print(f"    seed {seed} L{level} {k}: {tag} frames are not "
                          f"the rotation of the unrotated ones")
                    bad += 1
    if verbose:
        print(f"  {len(seen)} presentations drawn: {sorted(seen)}")
    return bad


def _space(levels=None, cap: int = 3_000_000, verbose: bool = True) -> int:
    """Sweep a level's whole reachable space, layer by layer, and report where it
    got to. The evidence behind "this level does not fit `_field`", and the only
    honest thing to say about level 9.

    It is `_field`'s forward sweep with no backward pass, so what it prints is a
    measurement anyone can re-take rather than an assertion. A sweep that
    EXHAUSTS without a winning press proves the level unwinnable; one that stops
    at ``cap`` proves only that no win exists at or below the depth it reached.

    ``cap`` IS A MEMORY BUDGET and it is the whole safety of this report -- ~250
    bytes a state, so the default is ~750 MB and a cap of 12 million took a
    32 GB machine down. Raise it deliberately or not at all."""
    _solver, game, expert = _new()
    for level in (levels if levels else range(game.n_levels)):
        game.set_level(level)
        b0 = expert.read(game._engine)
        codec = _Codec(b0)
        t0 = time.time()
        dist = {codec.pack(b0): 0}
        frontier = [codec.pack(b0)]
        depth, dstar = 0, None
        while frontier and dstar is None and len(dist) <= cap:
            nxt = []
            for k in frontier:
                b = codec.unpack(k)
                for d in range(5):
                    nb = b.clone()
                    if not nb.step(d):
                        continue
                    if nb.won():
                        dstar = depth + 1
                        break
                    nk = codec.pack(nb)
                    if nk in dist:
                        continue
                    dist[nk] = depth + 1
                    nxt.append(nk)
                if dstar is not None:
                    break
            if dstar is not None:
                break
            depth += 1
            frontier = nxt
            if len(dist) > cap:
                break
        verdict = (f"WIN at depth {dstar}" if dstar is not None else
                   f"EXHAUSTED at depth {depth} -- PROVABLY UNWINNABLE"
                   if not frontier else
                   f"stopped at the {cap}-state cap, depth {depth} reached "
                   f"(so no win exists in {depth} presses or fewer)")
        if verbose:
            print(f"  L{level}: {len(dist)} states, {verdict}, "
                  f"{time.time() - t0:.0f}s")
    return 0


def _speed(presses: int = 1500, verbose: bool = True) -> int:
    """Interpreter presses/s against model presses/s, per level.

    The reason this generator carries a native model at all. Measure it before
    copying `PSExpert`'s engine-blackbox A* into a new ps: game -- the
    entrepotphage rule -- because a mechanically identical game whose rules
    repaint the board every turn is 50x slower and needs a different tier."""
    _solver, game, expert = _new()
    eng = game._engine
    for level in range(game.n_levels):
        game.set_level(level)
        b0 = expert.read(eng)
        rng = random.Random(level)
        seq = [rng.randrange(5) for _ in range(presses)]
        t0 = time.time()
        for d in seq:
            eng.step(DIRS[d])
        eng_rate = presses / max(1e-9, time.time() - t0)
        b = b0.clone()
        t0 = time.time()
        for d in seq:
            b.step(d)
        mod_rate = presses / max(1e-9, time.time() - t0)
        if verbose:
            print(f"  L{level:<2d} interpreter {eng_rate:8.0f}/s   "
                  f"model {mod_rate:8.0f}/s   x{mod_rate / eng_rate:.0f}")
    return 0


if __name__ == "__main__":
    if "--plans" in sys.argv:
        sys.exit(_plans())
    if "--fuzz" in sys.argv:
        n = [int(a) for a in sys.argv[sys.argv.index("--fuzz") + 1:]
             if a.isdigit()]
        violations = _fuzz(n[0] if n else 5000)
        print(f"fuzz: {violations} mismatches")
        sys.exit(1 if violations else 0)
    if "--engine" in sys.argv:
        violations = _engine()
        print(f"engine replay: {violations} disagreements")
        sys.exit(1 if violations else 0)
    if "--ties" in sys.argv:
        args = [int(a) for a in sys.argv[sys.argv.index("--ties") + 1:]
                if a.isdigit()]
        violations = _ties(args[0] if args else 400_000)
        print(f"ties: {violations} disagreements")
        sys.exit(1 if violations else 0)
    if "--audit" in sys.argv:
        collisions = _audit()
        print(f"audit: {collisions} colliding compositions")
        sys.exit(1 if collisions else 0)
    if "--symmetry" in sys.argv:
        violations = _symmetry()
        print(f"symmetry: {violations} violations")
        sys.exit(1 if violations else 0)
    if "--space" in sys.argv:
        args = [int(a) for a in sys.argv[sys.argv.index("--space") + 1:]
                if a.isdigit()]
        sys.exit(_space(args or None))
    if "--speed" in sys.argv:
        sys.exit(_speed())
    sys.exit(WorkshopSolver.main())
