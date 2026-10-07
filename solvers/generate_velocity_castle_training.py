"""Generate Phase-1 training data for the PuzzleScript game ps:velocity_castle
("Velocity Castle", Tobin Mollett, ``data/puzzlescript_games/Velocity_Castle.txt``).

You are wearing magical shoes that never stop moving. One press launches you in
that direction and you keep going until something stops you; then you press
again. Fourteen levels of castle, and the way out of each is a gate that only
comes down when every switch on the level is down.

The harness -- the rotation contract, the trajectory recorder and the
`BaseSolver` plumbing -- lives in `solvers/common/ps_astar.py`, shared with the
other ps: generators. This file is the game-specific part: a native model of the
mechanic that was fuzzed against the interpreter, an ORACLE that reads an exact
distance field over it, the honest partially-observed expert that is what
actually gets recorded, and the reports that back all of it up.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_velocity_castle",
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
view, so replaying the recorded actions reproduces the recorded frames exactly
-- which ``--replay`` checks, frame for frame, on a freshly built adapter.

Level indices here and in every report are 0-BASED, i.e. one less than the
number in the game's own "Level N" messages.


THE MECHANIC, AS MEASURED
-------------------------
Everything below was measured against the interpreter and is re-checked by
``--model``, which fuzzes the native model against it (33,600 random presses,
2,400 per level, comparing the whole board after every one).

* **A press is a whole ROLL.** ``[PlayerUp] -> [UP PlayerUp] again`` re-fires to
  a fixpoint inside a single ``eng.step``, so the branching factor is four and
  the states are the squares you can come to REST on.
* **You cannot push from a standstill.** ``require_player_movement`` reverts the
  WHOLE turn when the player ends where it started -- and ``[ > Player |
  Movable ] -> [ PlayerRest | > Movable ]`` does not move the player. So a bump
  from an adjacent square is not a weak push, it is *nothing at all*: the statue
  does not move, the switch does not flip, the turn is undone. Every push in
  this game is a ram taken at speed. It is the mechanic the title is about.
* **A rammed statue moves ONE square; a rammed boulder ROLLS** until something
  stops it, and it stops for the same things you do. A rolling boulder flips
  switches, smashes treasure, shoves statues one square and hands its force to
  the next boulder in line -- which is how a chain crosses a room in one press.
* **Spike traps are one-shot.** ``[Weight SpikeTrap no TriggerSpikes]`` arms the
  trap the tick after something stands on it and ``[TriggerSpikes no Weight]``
  turns it into permanent Spikes the tick after that thing leaves. So crossing a
  trap builds a wall BEHIND you, two ticks later, and a trap you come to rest on
  is armed but harmless until you go. Nothing here kills you -- spikes are just
  wall -- but they are irreversible, and half of what makes a level lost is a
  corridor you have already spiked shut.
* **The gate wants EVERY switch.** The three `late` gate rules are an AND over
  the whole board (mark all gates if ANY switch is down, unmark them if ANY
  switch is still up, delete what is left marked), so one switch of three is no
  progress you can see. Gates never come back.
* **The force fields want EVERY panel**, by the same three-rule shape, and they
  are NOT permanent: step off the panel and ``[FloorPanel no Weight]
  [OpenForceField] -> [ForceField]`` writes the field back into the same
  collision layer as whatever is standing on that square -- **deleting it**.
  Rolling through a doorway that closes behind you is how you lose the player
  outright, and it is why the model carries a "which fields are open" mask
  rather than deriving it: a level's authored start need not be settled (level
  11 ships eight open fields and two closed ones with the panel already held).
* **Treasure is not the win** -- ``Some Player on Exit`` is -- but it is not
  decoration either: a treasure STOPS a roll and is destroyed doing it, so on
  level 13 the scattered treasure in the void is the only thing to aim off.
* **The win fires MID-ROLL.** The engine breaks the ``again`` cascade the moment
  the player stands on the exit, which leaves the winning frame showing a
  ROLLING player sprite on the exit square.
* **A roll with nothing to stop it freezes you.** Off the board the move fails
  but the rolling sprite is never converted back, so the player keeps its
  direction and no press can turn it: stuck forever. No reachable state of any
  shipped level does this (``--space`` counts them: zero), but the model
  reproduces it rather than assuming the walls are always there.
* One press is capped at 50 ``again`` iterations. The worst press in the game
  takes 29, so no cascade is cut short -- but the model reproduces the cap, and
  a cascade that WAS cut short leaves the roll to be resumed by the next press,
  not abandoned.


THE MODEL, AND WHY THE WHOLE GAME ENUMERATES
--------------------------------------------
A state is ``(player, roll direction, statues, boulders, rolling boulders,
switches up, treasures, armed traps, spikes, open fields)``; everything no rule
can change (walls, blocks, the entrance arch, the exit squares, which cells hold
a switch, a panel, a field or a trap) is board geometry. `_Board.step` is one
press: the whole cascade, tick by tick, in the interpreter's own order.

At 6-25us a press against the interpreter's 3-19ms -- three orders of magnitude
-- the exact field is affordable everywhere. ``--space`` enumerates every state
every level can reach: 37, 108, 66395, 256, 9778, 1861503, 1349, 3186, 592,
535274, 34239, 912, 698 and 271, the whole game in under two minutes, and EVERY
level has a winning state -- so all fourteen are winnable, proved by
enumeration rather than by a search that stopped. ``--plans`` reads the shortest win out of `_Board.field` (a forward
sweep to the first win, keeping predecessors because almost every press here is
irreversible, then a backward sweep from the wins): 10 to 54 presses, and every
one of them replays to a WIN on the real interpreter.


WHY THE ORACLE IS NOT THE EXPERT: ``flickscreen 16x16``
-------------------------------------------------------
The camera shows one 16x16 screen -- the one the player is standing in -- and
six of the fourteen levels are two, three or four screens across. At the start
of level 4 the third of its three switches has never been rendered; at the start
of level 12, three of its four. An expert that planned over the whole engine
grid would emit a beeline to a switch the frames do not show: a target no
observation-conditioned learner can reproduce (the badge_placement failure mode
in its pure form). So `VelocityCastleExpert` reads exactly the window
`_render_frame` draws, keeps a map of what it has been shown, and plans over
that map and nothing else -- exploiting when the map already contains a way out,
exploring the cheapest press whose outcome the map cannot predict when it does
not, and pressing R when a roll into an unseen room has stranded the level.

On the eight single-screen levels the first frame shows everything, so the two
modes collapse into the oracle's exact field and those levels come out provably
SHORTEST. ``--honest`` measures the rest against the oracle:

    level  0-3, 6, 8, 9   one screen    15/18/21/10, 25, 32, 18   = shortest
    level 11              one screen    44  (shortest 42)
    level  4              4 screens     35  (shortest 35)
    level  5              2 screens     24  (shortest 22)
    level  7              2 screens     28  (shortest 28)
    level 12              4 screens     36  (shortest 33)
    level 13              3 screens     18  (shortest 18)

Thirteen of fourteen, and the gap where there is one is the exploration itself.
``--ties`` prices that exactly: of the 324 presses the expert takes across the
thirteen levels, **319 carry an optimal-action set that is exactly the FULL
game's** -- five are off-path, and those five are the presses spent finding
something the frames had not shown yet.

Level 10 is the one it cannot win inside the adapter's 200-press budget, and it
is `skip_levels` rather than a mystery: see the note there.


THE HYPOTHESES, AND HOW THEY GET REFUTED
-----------------------------------------
Nothing the expert plans on is a read of hidden state. What it cannot see it
GUESSES, cheapest guess first, and every guess is dropped the moment a frame
contradicts it (`VelocityCastleExpert.observe` / `_refute`):

* *the switches I have seen are all there are*, so flipping them opens the gate
  -- refuted by seeing a gate still standing with all of them down (level 4
  does this on purpose);
* *the panels I have seen are all there are*, so standing on them opens the
  fields -- refuted by seeing a field still shut while they are all held;
* *nothing has moved off screen since I last looked* -- corrected by the next
  look, and `tried` remembers what a press REALLY did whenever a prediction was
  wrong, so a wrong guess costs one press and cannot loop;
* *a restart puts every room back as I first saw it* -- the weakest of the four,
  because the press that first shows a room has already crossed its traps and
  shoved its statues. `stale_screens` is the repair: after a restart, rooms not
  seen since are worth LOOKING at again, and `base` keeps each cell's earliest
  sighting rather than its first.


RENDERING: FOUR THINGS THE FRAME DID NOT SHOW
----------------------------------------------
Every board is cropped to 16x16, so every cell is 4px, so a 5x5 sprite is
sampled at rows/cols {0,1,3,4} -- **the middle row and column are never read**.
Four of this game's sprites carried their whole meaning there, and
``data/puzzlescript_games/Velocity_Castle.txt`` was edited (art only, no rules):

* **SwitchUp vs SwitchDown** differed in the centre pixel alone. The one bit of
  state the gate depends on was invisible; the marker moved one column out.
* **FloorPanel** was a diamond on the centre row and column: a panel rendered as
  bare floor, so the squares the force fields depend on were not in the picture.
  Given corner studs, which also show through the player's transparent corners.
* **OpenForceField** was two centre-row pixels: an open doorway rendered as bare
  floor and, under the player, as nothing at all. Redrawn on rows 1 and 3.
* **PlayerUp vs PlayerDown** differed in one centre pixel, and both differed
  from each other in a colour (``#fcf695`` and ``#fea1e7``) that quantizes to
  the same ARC index anyway -- so the WINNING frame could not say which way you
  came in. Repainted in the sprite's own red.

``--audit`` renders every cell composition a settled frame can show, on uniform
boards, and asserts pairwise distinctness. One pair is allowed to collide: Wall
and Block ship the same palette and are both `Immovable`, so nothing a press
does depends on telling them apart -- and `observe` folds them into one
observable class rather than pretending the difference is legible.


RECOVERY
--------
``recovery_mode = "reset"`` from `PSAStarSolver`: an epsilon-decayed exploration
prefix flails first and ONE RESET restores the level start, from which the
expert plays. Here that is not a convention borrowed from the other ps: games,
it is the game's own advice -- the third level's message tells the player to
press R -- and the expert reaches for the same key itself when a roll has
stranded a level (`_restart_or_probe`). `ps_astar.DIR_TO_ACTION` maps the
``"reset"`` plan entry onto `GameAction.RESET` so `record_level` drives and
records it like any other press; it does not refund the level's press budget.

The knowledge map deliberately SURVIVES the prefix's RESET: the agent has seen
those frames, and what it learned by flailing is exactly what it should still
know afterwards.


CLI
---
    --plans        the oracle's shortest win per level, replayed on the engine
    --honest       what the expert actually plays, against that yardstick
    --ties         every optimal-action label, checked against the full game
    --space        the exhaustive reachable-state count per level
    --model        fuzz the native model against the interpreter
    --audit        assert every cell composition renders distinctly
    --replay DIR   replay recorded episodes frame for frame
"""

from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solvers.common.ps_astar import (                        # noqa: E402
    PSAStarSolver, PSExpert, Plan,
)

GAME_NAME = "Velocity_Castle"

#: How many states the exact field may hold. Every level in the game closes far
#: under it -- the largest whole space is level 5's 1.86M and the field stops at
#: the first win's depth, well short of that -- so this is a runaway guard on an
#: edited level, not a tuning dial.
FIELD_CAP = 3_000_000

#: The same, for the searches taken inside the recording loop -- one per press,
#: so they have to stay small. Only reached when an exit is on the map and no
#: route to it exists: the sweep then has to enumerate the whole known
#: component before it can say so.
FIELD_CAP_FAST = 200_000

#: A plan longer than this cannot be replayed: the adapter cuts a level off at
#: 200 presses, counted from the `set_level` that ends the exploration prefix.
#: Only the ORACLE is bounded by it -- the expert decides one press at a time.
MAX_PLAN = 185

#: The four presses, in the order ties are broken. ``noaction`` is declared and
#: no rule reads the ACTION key, so branching on it would be pure waste.
DIRS: tuple[str, ...] = ("up", "down", "left", "right")

_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

#: `PSEngine.step`'s ``max_again``. One press runs the whole roll inside a
#: single ``step`` as an ``again`` cascade, and the cascade is CUT OFF after
#: this many iterations -- which leaves whatever was rolling still rolling, to
#: be resumed by the next press. The model reproduces the cap rather than
#: assuming rolls always finish; see `_Board.step`.
MAX_AGAIN = 50

#: `_Board.step`'s answer when the roll enters a cell the camera has never
#: shown: the outcome is not "unknown to the model", it is unknowable from the
#: frames, and the expert's whole exploration policy is built on chasing these.
#: A board with nothing hidden (``unknown == 0``) can never produce one.
UNKNOWN = "unknown"

#: The plan entry that means "press R". `ps_astar.DIR_TO_ACTION` maps it to
#: `GameAction.RESET`, which the adapter answers with a level restart, so
#: `record_level` drives and records it like any other press. It is never
#: searched over -- it is what `VelocityCastleExpert._restart_or_probe` reaches
#: for when a roll into an unseen screen has stranded the level.
RESET_PRESS = "reset"

#: Object names the model reads. Resolved once and asserted, so a renamed
#: object in the .txt is a crash rather than a silent mis-read.
_OBJ_NAMES = (
    "background", "floor", "exit", "gate", "entrance", "wall", "block",
    "switchup", "switchdown", "playerrest", "playerup", "playerdown",
    "playerleft", "playerright", "statue", "statueeyes", "boulderrest",
    "boulderup", "boulderdown", "boulderright", "boulderleft", "boulderlights",
    "floorpanel", "forcefield", "openforcefield", "spiketrap", "spikes",
    "triggerspikes", "treasure",
)


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

class _Board:
    """One level's static geometry, plus the mechanic.

    Cells are flat ``r * w + c`` indices. A STATE is the 9-tuple

        (player, pdir, statues, boulders, rolling, switches_up,
         treasures, triggered, spikes, open_fields)

    with ``player`` a cell (or -1 once a closing force field has deleted it),
    ``open_fields`` the force fields currently standing OPEN -- a state bit
    rather than a derived one, because a level's authored start is not
    necessarily settled (level 11 ships eight open fields and two closed ones
    with the panel already held, and it takes one turn of `late` rules to make
    them agree),
    ``pdir`` the direction it is still rolling in (-1 at rest), ``rolling`` a
    sorted tuple of ``(cell, direction)`` for boulders still in motion, and the
    rest plain cell bitmasks.

    Everything that no rule can change -- walls, blocks, the entrance arch, the
    exit squares, which cells hold a switch, a panel, a force field or a spike
    trap -- lives here rather than in the state.
    """

    __slots__ = ("h", "w", "n", "nbr", "wallish", "exits", "gates",
                 "switches", "panels", "fields", "traps", "n_switch",
                 "unknown", "extra_switch", "extra_panel")

    def __init__(self, h, w, wallish, exits, gates, switches, panels, fields,
                 traps, unknown=0, extra_switch=False, extra_panel=False):
        self.h, self.w = h, w
        self.n = h * w
        self.wallish = wallish          # wall / block / entrance: never changes
        self.exits = exits              # cells the win condition reads
        self.gates = gates              # cells that hold a Gate at level start
        self.switches = switches        # every switch cell (up or down)
        self.panels = panels            # every FloorPanel cell
        self.fields = fields            # every ForceField / OpenForceField cell
        self.traps = traps              # every SpikeTrap cell at level start
        #: Cells the camera has never shown. Zero for a fully-informed board;
        #: see `VelocityCastleExpert` for the partially observed one.
        self.unknown = unknown
        #: Refutation: a gate was SEEN standing while every known switch was
        #: already down, so a switch exists that has not been found yet and
        #: "all my switches are down" no longer predicts an open gate.
        self.extra_switch = extra_switch
        #: The same refutation for the panels: a force field was SEEN closed
        #: while every known panel was held, so a panel exists that has not
        #: been found and "all my panels are held" no longer predicts an open
        #: field.
        self.extra_panel = extra_panel
        self.n_switch = bin(switches).count("1")
        self.nbr = []
        for i in range(self.n):
            r, c = divmod(i, w)
            row = []
            for d in DIRS:
                dr, dc = _DELTA[d]
                nr, nc = r + dr, c + dc
                row.append(nr * w + nc if 0 <= nr < h and 0 <= nc < w else -1)
            self.nbr.append(tuple(row))

    # -- derived board facts -------------------------------------------------
    def gates_gone(self, swu) -> bool:
        """The three `late` gate rules, which are an AND over the whole board.

        ``[SwitchDown][Gate] -> [Gate ShadowGate]`` marks every gate when ANY
        switch is down; ``[SwitchUp][ShadowGate] -> [no ShadowGate]`` unmarks
        them again when ANY switch is still up; only then does
        ``[Gate ShadowGate] -> [no Gate]`` delete them. So the gates open when
        EVERY switch on the level is down -- and stay open, because no rule ever
        raises a switch."""
        return self.n_switch > 0 and swu == 0 and not self.extra_switch

    def field_update(self, fopn, weights):
        """The force fields standing open at the end of a turn, given the ones
        that were open at the start of it and where the weights are.

        The `late` rules have the same all-board shape as the gates:
        ``[Weight FloorPanel][ForceField]`` marks, ``[FloorPanel no Weight]
        [ShadowField]`` unmarks, so the fields open exactly while EVERY panel
        carries a weight (player, statue or boulder) -- and unlike the gates it
        is not permanent: step off and ``[FloorPanel no Weight]
        [OpenForceField] -> [ForceField]`` puts every field back, on top of
        whatever is standing there.

        The two clauses that are not the shipped rule are what a PARTIALLY
        observed board needs, and both are no-ops on a fully observed one
        (``panels`` empty means the level really has no panel, and then no rule
        can fire; ``extra_panel`` is only ever set from an observation):

        * with NO panel known, nothing the agent knows about can move a field,
          so they stay exactly as they were last seen -- which is how an open
          field on a screen whose panel is held two screens away stays open in
          the model instead of being predicted shut;
        * with a panel known to be UNFOUND (`extra_panel`), holding every panel
          on the map is not enough to conclude anything, so again they stay as
          they were seen.
        """
        if not self.panels:
            return fopn
        if self.panels & ~weights:
            return 0
        if self.extra_panel:
            return fopn
        return self.fields

    def won(self, state) -> bool:
        """``Some Player on Exit``."""
        p = state[0]
        return p >= 0 and (self.exits >> p) & 1

    # -- the mechanic --------------------------------------------------------
    def step(self, state, di):
        """The state after pressing ``DIRS[di]``, or None if the press is a
        no-op.

        A press is a no-op when the player cannot move at all: the engine's
        ``require_player_movement`` reverts the WHOLE turn (including a statue
        that a bump would have shifted) whenever the player ends where it
        started, and a player the board has deleted has nothing to revert.
        """
        p, pdir, sta, bou, roll, swu, tre, trg, spk, fopn = state
        if p < 0:
            return None
        start_p = p
        # PlayerRest takes the pressed direction; a player still rolling keeps
        # its own (``[PlayerUp] -> [UP PlayerUp]`` overrides the input force),
        # which is what makes a cascade cut off by MAX_AGAIN resumable.
        pdir = pdir if pdir >= 0 else di
        roll = list(roll)
        won = False

        for _tick in range(MAX_AGAIN):
            changed = False
            gates_gone = self.gates_gone(swu)
            blocked = self.wallish | spk | tre | sta | bou
            if not gates_gone:
                blocked |= self.gates
            blocked |= self.switches                 # up or down, both Immovable
            for cell, _d in roll:
                blocked |= 1 << cell
            weights = sta | bou
            if p >= 0:
                weights |= 1 << p
            for cell, _d in roll:
                weights |= 1 << cell
            blocked |= self.fields & ~fopn      # a closed field is Immovable

            # -- rules: every entity in motion, player first (rule order) ----
            moves = []            # (from, to, kind) kind: 0 player, 1 boulder
            new_roll = []
            statue_moves = []     # (from, dir)
            queue = []
            if pdir >= 0 and p >= 0:
                queue.append((p, pdir, 0))
            for cell, d in roll:
                queue.append((cell, d, 1))
            qi = 0
            while qi < len(queue):
                cell, d, kind = queue[qi]
                qi += 1
                t = self.nbr[cell][d]
                if t >= 0 and (self.unknown >> t) & 1:
                    return UNKNOWN            # the roll leaves what has been shown
                if t < 0:
                    # Off the board: the move fails but the rolling sprite is
                    # never converted back, so the entity stays in motion
                    # forever -- it is stuck against the edge.
                    if kind == 0:
                        pass                      # pdir kept: player is stuck
                    else:
                        bou &= ~(1 << cell)
                        new_roll.append((cell, d))
                    continue
                bt = 1 << t
                if swu & bt:                      # SwitchUp: flip it and stop
                    swu &= ~bt
                    changed = True
                    stopped = True
                elif (sta | bou) & bt:            # Movable: stop and pass force
                    if sta & bt:
                        statue_moves.append((t, d))
                    else:
                        queue.append((t, d, 1))
                    stopped = True
                elif any(c == t for c, _ in roll):
                    stopped = True                # a boulder already in motion
                elif tre & bt:                    # Treasure: destroy it and stop
                    tre &= ~bt
                    changed = True
                    stopped = True
                elif blocked & bt:                # Immovable / closed field
                    stopped = True
                else:
                    stopped = False
                if stopped:
                    if kind == 0:
                        pdir = -1                 # -> PlayerRest
                    else:
                        bou |= 1 << cell          # -> BoulderRest
                else:
                    moves.append((cell, t, kind, d))

            # -- spike traps, read at the positions the rules see -------------
            live = self.traps & ~spk
            if live:
                on = live & weights
                if on & ~trg:
                    trg |= on
                    changed = True
                off = trg & ~weights
                if off:
                    trg &= ~off
                    spk |= off
                    changed = True

            # -- force resolution --------------------------------------------
            taken = blocked                       # cells that stay occupied
            for frm, to, kind, d in moves:
                taken &= ~(1 << frm)
            for frm, to, kind, d in moves:
                bt = 1 << to
                if taken & bt:
                    # Contested: leave the entity where it is, still in motion.
                    if kind == 0:
                        pass
                    else:
                        new_roll.append((frm, d))
                    continue
                taken |= bt
                changed = True
                if kind == 0:
                    p = to
                else:
                    bou &= ~(1 << frm)
                    new_roll.append((to, d))
            for frm, d in statue_moves:
                to = self.nbr[frm][d]
                if to >= 0 and (self.unknown >> to) & 1:
                    return UNKNOWN            # the statue is shoved off screen
                if to < 0:
                    continue
                bt = 1 << to
                if taken & bt:
                    continue
                taken |= bt
                sta = (sta & ~(1 << frm)) | bt
                changed = True
            roll = new_roll

            # -- late rules ---------------------------------------------------
            weights = sta | bou
            if p >= 0:
                weights |= 1 << p
            for cell, _d in roll:
                weights |= 1 << cell
            was_open = fopn
            fopn = self.field_update(fopn, weights)
            if fopn != was_open:
                changed = True
            if was_open & ~fopn:
                # Every field closes, replacing whatever stands on its square:
                # the ForceField is written into the same collision layer, so
                # the player, a statue or a boulder caught there is DELETED.
                hit = was_open & ~fopn & weights
                if hit:
                    if p >= 0 and (hit >> p) & 1:
                        p, pdir = -1, -1
                    if sta & hit:
                        sta &= ~hit
                    if bou & hit:
                        bou &= ~hit
                    if roll:
                        roll = [(c, d) for c, d in roll if not (hit >> c) & 1]
                    changed = True

            if p >= 0 and (self.exits >> p) & 1:
                won = True
                break
            if not changed:
                break

        state = (p, pdir, sta, bou, tuple(sorted(roll)), swu, tre, trg, spk,
                 fopn)
        if p == start_p and p >= 0 and not won:
            return None                # require_player_movement reverts the turn
        return state

    # -- the exact field -----------------------------------------------------
    def field(self, state, cap: int):
        """``(dist, d_star, status)`` -- presses-to-win for every state on a
        shortest path from ``state``, the length of that path, and one of
        ``"exact"`` / ``"dead"`` / ``"capped"``.

        Sweep 1 is a forward BFS stopped at the depth of the first win, which
        fixes ``d_star`` and collects the states a shortest path can pass
        through; it records PREDECESSORS as it goes, because nearly every press
        here is irreversible (a crossed trap turns to spikes, a flipped switch
        never rises, a rolled boulder cannot be pulled back) so there is no
        backward move generator to sweep with. Sweep 2 is a backward BFS from
        the winning states over those recorded predecessors.

        Distances are exact for every state ON a shortest start-to-win path,
        which is all `solve` and the tie sets ever read: such a state's whole
        future has forward distance at most ``d_star``, so the truncated sweep
        1 contains it. States on no shortest path can come out too high or be
        missing entirely.

        ``"dead"`` is a PROOF of unwinnability -- the queue ran dry having
        generated every state reachable from here, and none of them wins.
        """
        if self.won(state):
            return {state: 0}, 0, "exact"
        seen = {state: 0}
        preds: dict = {}
        queue = deque([state])
        d_star = None
        while queue:
            cur = queue.popleft()
            d = seen[cur]
            if d_star is not None and d >= d_star:
                break
            for di in range(4):
                nxt = self.step(cur, di)
                if nxt is None or nxt is UNKNOWN or nxt == cur:
                    continue
                if nxt in seen:
                    if seen[nxt] == d + 1:
                        preds[nxt].append(cur)
                    continue
                seen[nxt] = d + 1
                preds[nxt] = [cur]
                if self.won(nxt):
                    if d_star is None:
                        d_star = d + 1
                    continue                    # a won board is terminal
                queue.append(nxt)
            if len(seen) > cap:
                return None, None, "capped"
        if d_star is None:
            return None, None, "dead"
        dist = {}
        bq = deque()
        for s in seen:
            if self.won(s):
                dist[s] = 0
                bq.append(s)
        while bq:
            cur = bq.popleft()
            d = dist[cur]
            for prev in preds.get(cur, ()):
                if prev in dist:
                    continue
                dist[prev] = d + 1
                bq.append(prev)
        return dist, d_star, "exact"

    def field_optimal(self, dist, state):
        """``[(direction index, successor), ...]``: every press on a shortest
        path from ``state``. Ties come out in `DIRS` order, which is what makes
        a re-derived plan byte-identical across processes."""
        rest = dist.get(state)
        if not rest:
            return []
        out = []
        for di in range(4):
            nxt = self.step(state, di)
            if (nxt is not None and nxt is not UNKNOWN and nxt != state
                    and dist.get(nxt, -1) == rest - 1):
                out.append((di, nxt))
        return out


# ---------------------------------------------------------------------------
# Engine <-> model
# ---------------------------------------------------------------------------

def _bits(mask: int):
    """The set cells of a cell bitmask, ascending."""
    out = []
    while mask:
        low = mask & -mask
        out.append(low.bit_length() - 1)
        mask ^= low
    return out


def _weights(state):
    """The cells a `_Board`'s ``Weight`` group occupies in ``state`` -- the
    player, every statue and every boulder, rolling or not."""
    p, _pdir, sta, bou, roll, *_rest = state
    mask = sta | bou | (0 if p < 0 else 1 << p)
    for cell, _d in roll:
        mask |= 1 << cell
    return mask

class _Ids:
    """The object indices the reader and the writer need, resolved once."""

    __slots__ = ("bg", "floor", "exit", "gate", "entrance", "wall", "block",
                 "swup", "swdown", "prest", "pdirs", "statue", "eyes",
                 "brest", "bdirs", "lights", "panel", "field", "openfield",
                 "trap", "spikes", "trig", "treasure", "player_all",
                 "boulder_all", "dynamic")

    def __init__(self, g):
        idx = g.obj_name_to_idx
        missing = [n for n in _OBJ_NAMES if n not in idx]
        if missing:
            raise AssertionError(f"{GAME_NAME}.txt is missing object(s) {missing}")
        self.bg = idx["background"]
        self.floor = idx["floor"]
        self.exit = idx["exit"]
        self.gate = idx["gate"]
        self.entrance = idx["entrance"]
        self.wall = idx["wall"]
        self.block = idx["block"]
        self.swup = idx["switchup"]
        self.swdown = idx["switchdown"]
        self.prest = idx["playerrest"]
        #: Indexed by `DIRS`, so ``pdirs[di]`` is the sprite for that roll.
        self.pdirs = tuple(idx["player" + d] for d in DIRS)
        self.statue = idx["statue"]
        self.eyes = idx["statueeyes"]
        self.brest = idx["boulderrest"]
        self.bdirs = tuple(idx["boulder" + d] for d in DIRS)
        self.lights = idx["boulderlights"]
        self.panel = idx["floorpanel"]
        self.field = idx["forcefield"]
        self.openfield = idx["openforcefield"]
        self.trap = idx["spiketrap"]
        self.spikes = idx["spikes"]
        self.trig = idx["triggerspikes"]
        self.treasure = idx["treasure"]
        self.player_all = frozenset((self.prest,) + self.pdirs)
        self.boulder_all = frozenset((self.brest,) + self.bdirs)
        #: Everything a rule can add or remove -- what `_seat` clears before
        #: writing a state, and what the expert's `_key` reads.
        self.dynamic = frozenset(
            self.player_all | self.boulder_all
            | {self.gate, self.swup, self.swdown, self.statue, self.eyes,
               self.lights, self.field, self.openfield, self.trap,
               self.spikes, self.trig, self.treasure})


def _read(eng, ids):
    """``(board, state)`` for the engine's current grid."""
    h, w = eng.height, eng.width
    wallish = exits = gates = switches = panels = fields = traps = 0
    sta = bou = swu = tre = trg = spk = fopn = 0
    roll = []
    p, pdir = -1, -1
    for r, row in enumerate(eng.grid):
        for c, cell in enumerate(row):
            i = r * w + c
            b = 1 << i
            if cell & {ids.wall, ids.block, ids.entrance}:
                wallish |= b
            if ids.exit in cell:
                exits |= b
            if ids.gate in cell:
                gates |= b
            if ids.swup in cell or ids.swdown in cell:
                switches |= b
                if ids.swup in cell:
                    swu |= b
            if ids.panel in cell:
                panels |= b
            if ids.field in cell or ids.openfield in cell:
                fields |= b
                if ids.openfield in cell:
                    fopn |= b
            if cell & {ids.trap, ids.trig, ids.spikes}:
                traps |= b
                if ids.trig in cell:
                    trg |= b
                if ids.spikes in cell:
                    spk |= b
            if ids.treasure in cell:
                tre |= b
            if ids.statue in cell:
                sta |= b
            if ids.brest in cell:
                bou |= b
            else:
                for di, oid in enumerate(ids.bdirs):
                    if oid in cell:
                        roll.append((i, di))
                        break
            if ids.prest in cell:
                p, pdir = i, -1
            else:
                for di, oid in enumerate(ids.pdirs):
                    if oid in cell:
                        p, pdir = i, di
                        break
    board = _Board(h, w, wallish, exits, gates, switches, panels, fields, traps)
    state = (p, pdir, sta, bou, tuple(sorted(roll)), swu, tre, trg, spk, fopn)
    return board, state


def _seat(eng, board, state, ids) -> None:
    """Write a model state onto the engine's grid, leaving the static scenery
    (walls, blocks, floor, exits, panels, background) exactly as it is.

    The two decorative markers are DERIVED here rather than carried in the
    state: ``[StatueEyes] -> [no StatueEyes]`` wipes them every turn and
    ``late [Statue FloorPanel] -> [Statue StatueEyes FloorPanel]`` re-adds them,
    so at the end of any settled turn a statue (boulder) wears its marker
    exactly while it stands on a panel."""
    p, pdir, sta, bou, roll, swu, tre, trg, spk, fopn = state
    w = board.w
    for row in eng.grid:
        for cell in row:
            cell -= ids.dynamic
    def put(mask, obj):
        m = mask
        while m:
            low = m & -m
            i = low.bit_length() - 1
            eng.grid[i // w][i % w].add(obj)
            m ^= low
    put(sta, ids.statue)
    put(sta & board.panels, ids.eyes)
    put(bou, ids.brest)
    put(swu, ids.swup)
    put(board.switches & ~swu, ids.swdown)
    put(tre, ids.treasure)
    put(spk, ids.spikes)
    put(trg, ids.trig)
    put(board.traps & ~(spk | trg), ids.trap)
    put(tre, ids.treasure)
    if not board.gates_gone(swu):
        put(board.gates, ids.gate)
    put(fopn, ids.openfield)
    put(board.fields & ~fopn, ids.field)
    for cell, d in roll:
        eng.grid[cell // w][cell % w].add(ids.bdirs[d])
    put((bou | sum(1 << c for c, _ in roll)) & board.panels, ids.lights)
    if p >= 0:
        eng.grid[p // w][p % w].add(ids.prest if pdir < 0 else ids.pdirs[pdir])
    eng._position_index_dirty = True
    eng._rule_noop_cache.clear()


# ---------------------------------------------------------------------------
# The oracle (fully informed) -- the yardstick, not the expert
# ---------------------------------------------------------------------------

def oracle_plan(board, state, cap: int = FIELD_CAP):
    """The provably SHORTEST win from ``state``, with the exact optimal SET at
    every step, read off `_Board.field`.

    This is what the game looks like to something that can see the whole map at
    once -- which the agent cannot, on the six levels that are more than one
    ``flickscreen`` across. It is the yardstick ``--plans`` measures the honest
    expert against, and on the eight single-screen levels it is exactly what the
    expert plays, because there the first frame shows everything.

    Returns ``(Plan, how)``; ``how`` is "field", "won", "dead" (a PROOF -- the
    enumeration ran dry) or "capped".
    """
    if state[0] < 0:
        return None, "dead"
    if board.won(state):
        return Plan([], []), "won"
    dist, d_star, status = board.field(state, cap)
    if status != "exact":
        return None, status
    if d_star > MAX_PLAN:
        return None, "too-long"
    presses, optsets = [], []
    cur = state
    for _ in range(d_star):
        best = board.field_optimal(dist, cur)
        if not best:
            return None, "broken-field"
        presses.append(DIRS[best[0][0]])
        optsets.append([DIRS[di] for di, _ in best])
        cur = best[0][1]
    return Plan(presses, optsets), "field"


# ---------------------------------------------------------------------------
# The honest partial-observation expert
# ---------------------------------------------------------------------------

class VelocityCastleExpert(PSExpert):
    """Plans from the screens the camera has SHOWN, one press at a time.

    WHY IT IS NOT `oracle_plan`. ``flickscreen 16x16`` crops every board to the
    one 16x16 screen the player is standing in, and six of the fourteen levels
    are two to four screens across: at the start of level 4 the third switch,
    at the start of level 12 three of the four, are on screens that have never
    been rendered. An expert that planned over the whole engine grid would
    emit a beeline to a switch the frames do not show -- a target no
    observation-conditioned learner can reproduce (the badge_placement failure
    mode in its pure form). So this one reads exactly the window
    `_render_frame` draws and plans over that map and nothing else.

    THE DECISION, re-derived from the knowledge state at every press. Each rung
    is a shortest path over the KNOWN board and is only reached when the one
    above it has nothing to offer:

    1. **exploit** (`decide`) -- an exit has been seen and the mapped part of
       the castle already contains a route to it: take a shortest such route.
    2. **explore** (`decide`, `stale_screens`) -- reach, in as few presses as
       possible, either a press whose OUTCOME the map cannot predict (a roll
       that leaves the screens seen so far) or a room not seen since the last
       restart. Both end with the camera showing something, so the map grows.
    3. **retry** (`_retry_or_restart`) -- the castle is fully seen and the map
       says there is no way out, which usually means the MAP is wrong: take a
       press this run has not taken, because a room seen from a state the last
       run never reached is a reading the map cannot already have wrong.
    4. **restart** (`_restart_or_probe`) -- press R. The map survives it and
       `tried` survives it, so the next run reaches further before it runs out
       of castle. Guarded so a run that learned nothing cannot press it twice.
    5. **probe** (`_probe`) -- at the level's start with all of the above
       exhausted: press something and see. This is what settles a hypothesis
       the board itself contradicts (level 11's unsettled opening frame).

    Every rung hands back the full set of equally good presses rather than one
    arbitrary choice. On a single-screen level the first frame shows
    everything, the map is complete from the first press, and rung 1 IS
    `oracle_plan`'s exact field -- which is why those eight levels come out
    provably shortest, and why ``--ties`` finds 319 of 324 labels across the
    whole game to be exactly the full game's optimal set.

    THE HYPOTHESES it plays on, and how they are refuted (nothing here is a
    read of hidden state -- each is the cheapest guess consistent with what has
    been shown, and each is dropped the moment a frame contradicts it):

    * *the switches I have seen are all the switches there are*, so flipping
      them opens the gate. Level 4 refutes it: two of its three switches are on
      the starting screen, and the gate is still standing when they are both
      down. `observe` sees that and sets `_Board.extra_switch`, which stops the
      model predicting an open gate until a new switch turns up.
    * *the panels I have seen are all the panels there are*, so standing on
      them opens the force fields. Refuted the same way, by seeing a field
      still closed.
    * *nothing has moved off screen since I last looked*. A boulder rolled out
      of the window is remembered where it was; the next frame that shows it
      corrects the map, and `tried` remembers what a press REALLY did whenever
      the prediction was wrong, so a wrong guess costs one press and cannot
      loop.
    * *a restart puts every room back as I first saw it*. The weakest of the
      four, because the press that first rolls into a room has already crossed
      its traps and shoved its statues -- see `stale_screens`, which is the
      repair, and `skip_levels`, which is the level where the repair is not
      enough.

    There is no plan cache and no disk cache: the answer depends on what has
    been seen, not only on where the pieces are, so a memo keyed on the board
    would serve the wrong press.
    """

    directions = list(DIRS)

    #: There is no plan cache and no disk cache (see the class docstring), so
    #: the base class's is switched off rather than left to key on a board.
    plan_cache_path = None

    #: How many states one decision's sweep may visit.
    survey_cap = FIELD_CAP_FAST

    def setup(self) -> None:
        self.ids = _Ids(self.g)
        #: ``flickscreen`` is (width, height). No flickscreen would mean the
        #: whole level is one screen, and the expert would then degenerate to a
        #: fully-informed shortest-path player -- the correct reading of "the
        #: frame shows everything".
        self.scr_w, self.scr_h = self.g.flickscreen or (1 << 30, 1 << 30)
        self._episode = None
        self._forget()

    def heuristic(self, eng) -> int:            # pragma: no cover
        raise AssertionError(
            "VelocityCastleExpert plans over its map, not by engine-side A*")

    # -- knowledge ----------------------------------------------------------
    def _forget(self) -> None:
        #: cell -> the objects the camera showed there, last time it did.
        self.mem: dict = {}
        #: cell -> the objects it held the FIRST time this episode showed it,
        #: which is what the map goes back to when the level is restarted. It
        #: assumes nothing the player did changed a cell before the player ever
        #: saw it -- true unless a boulder rolled off screen into a room not yet
        #: visited, and self-correcting when it is not.
        self.base: dict = {}
        #: How many presses into its run each `base` entry was seen.
        self.base_age: dict = {}
        #: Presses since the last restart.
        self.age = 0
        #: Where the player was standing the first time this episode was shown
        #: to the expert -- the square a restart puts it back on.
        self.base_pos = None
        self.ppos, self.pdir = -1, -1
        #: Cells the camera has shown SINCE the last restart. Everything else is
        #: stale: a restart puts the castle back, so a memory of a room from
        #: before it is a guess about a room that has since been rebuilt.
        self.fresh: set = set()
        self.h = self.w = 0
        #: Bumped whenever a cell's remembered contents CHANGE, so a decision
        #: is cached against the map it was made from.
        self._epoch = 0
        #: (state, direction index) -> the state the press REALLY reached, or
        #: None for a press that changed nothing. An observed outcome always
        #: beats a predicted one.
        self.tried: dict = {}
        #: Every ``(state, direction)`` this run has actually pressed. What is
        #: left is what there is still to learn once the map has gone wrong.
        self.taken: set = set()
        self._pending = None       # the decision whose outcome is not in yet
        self._restarting = False   # the press just handed out was an R
        self._state = None         # the state the last `observe` left us in
        self._board = None         # `_Board` for the current map (rebuilt lazily)
        self._board_sig = None
        self.explored = 0          # presses spent looking rather than winning
        self.restarts = 0          # times it pressed R and started the run over
        self._start = None         # the state a restart goes back to
        self.extra_switch = False
        self.extra_panel = False
        #: The map epoch R was last pressed at. A second restart is only worth
        #: a press once the map has CHANGED since -- the cycle has to have
        #: learned something or it will simply replay itself.
        self._reset_seen = -1

    def observe(self, eng, restarted: bool = False) -> None:
        """Fold ONE presented frame into the map. The only engine read there is.

        It takes exactly the window `_render_frame` renders -- the screen
        holding the player -- and reads the objects in it. Nothing outside the
        window is touched, which is what makes every decision reproducible by a
        viewer of the frames. Two things are folded specially:

        * **Wall and Block are one class.** They render identically at this
          game's one cell size AND behave identically (both are Immovable), so
          the difference is not in the picture and does not need to be; see
          ``--audit``.
        * **The player is not stored in the map at all**, it is tracked
          separately. There is exactly one of it and the camera is centred on
          it, so "where is the player" is the one thing every frame answers --
          whereas leaving it in the map would remember a player in every cell it
          was ever seen in, and a plan built from that board is a plan for a
          castle full of ghosts.

        ``restarted`` says the press that produced this frame was the expert's
        own R. A restart puts the WHOLE castle back, including the rooms the
        camera is not showing, so the map goes back to what each cell held the
        first time this episode showed it (`base`). The recovery prefix's RESET
        does not announce itself that way and is recognised instead: the frame
        is the one the episode opened on -- same screen contents, player back on
        its starting square -- while the map still remembers a mess elsewhere.
        (A player that rolls back onto its own starting square with its screen
        untouched reads the same way; the belief that costs is a memory of
        another screen, and the next look at that screen corrects it.)
        """
        self.h, self.w = eng.height, eng.width
        ids = self.ids
        pr = pc = 0
        pdir = -1
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                got = cell & ids.player_all
                if got:
                    pr, pc = r, c
                    if ids.prest not in got:
                        for di, oid in enumerate(ids.pdirs):
                            if oid in got:
                                pdir = di
                                break
                    break
            else:
                continue
            break
        sr = (pr // self.scr_h) * self.scr_h
        sc = (pc // self.scr_w) * self.scr_w
        er, ec = min(sr + self.scr_h, self.h), min(sc + self.scr_w, self.w)
        window = []
        for r in range(sr, er):
            row = eng.grid[r]
            for c in range(sc, ec):
                i = r * self.w + c
                seen = frozenset(o for o in row[c]
                                 if o != ids.bg and o != ids.floor
                                 and o not in ids.player_all)
                if ids.block in seen:
                    seen = (seen - {ids.block}) | {ids.wall}
                window.append(i)
                self.fresh.add(i)
                if self.age <= self.base_age.get(i, 1 << 30):
                    # The best claim about what a restart puts back is the
                    # EARLIEST sighting of the cell in any run: spikes only
                    # appear, treasures only vanish, switches only go down, and
                    # a piece has had fewer presses to be shoved. The very
                    # first sighting is not good enough on its own -- the press
                    # that first rolls into a room has already crossed its
                    # traps and shoved its statues, so the opening frame of a
                    # room is the room AFTER the entrance.
                    self.base[i], self.base_age[i] = seen, self.age
                if self.mem.get(i) != seen:
                    self.mem[i] = seen
                    self._epoch += 1
        self.ppos, self.pdir = pr * self.w + pc, pdir
        if self.base_pos is None:
            self.base_pos = self.ppos
        self._refute(window)
        if self._restarting:
            restarted, self._restarting = True, False
        if not restarted:
            restarted = (self.ppos == self.base_pos
                         and all(self.mem[i] == self.base[i] for i in window))
        if restarted:
            self.fresh = set(window)
            self.age = 0
            for i in window:                # age 0: this frame IS the restart
                self.base[i], self.base_age[i] = self.mem[i], 0
            if any(v != self.base[i] for i, v in self.mem.items()):
                self.mem = dict(self.base)
                self._epoch += 1
        board, state = self.board_state()
        if self._pending is not None:
            prev, di, predicted = self._pending
            if state != predicted:
                self.tried[(prev, di)] = None if state == prev else state
            self._pending = None
        if self._start is None:
            self._start = state
        self._state = state

    def _refute(self, window) -> None:
        """Update the two hypotheses from the window that was just folded in.

        Both are read off cells the camera is showing RIGHT NOW, never off the
        map: a remembered gate may have come down, and a remembered force field
        may have opened, while they were off screen -- believing a stale memory
        over the rules is what had the expert concluding a level was lost while
        it was standing on the panel that opens it.

        Each flag is kept until the window contradicts it, because a screen that
        is not on show cannot contradict anything.
        """
        ids = self.ids
        gate_seen = door_seen = shut_seen = open_seen = False
        for i in window:
            seen = self.mem[i]
            if ids.gate in seen:
                gate_seen = True
            elif ids.exit in seen:
                door_seen = True           # the gate is down: the exit is bare
            if ids.field in seen:
                shut_seen = True
            if ids.openfield in seen:
                open_seen = True
        state = self._state_from_mem()
        board_panels = self._panels()
        all_down = self._n_switch() > 0 and state[5] == 0
        held = bool(board_panels) and not (board_panels & ~_weights(state))
        was = (self.extra_switch, self.extra_panel)
        if gate_seen and all_down:
            self.extra_switch = True       # a switch I have not found exists
        elif door_seen or not all_down:
            self.extra_switch = False
        if held and shut_seen:
            self.extra_panel = True        # a panel I have not found exists
        elif held and open_seen:
            self.extra_panel = False
        if was != (self.extra_switch, self.extra_panel):
            self._epoch += 1               # the flags are part of the map

    def stale_screens(self) -> set:
        """Screens holding a cell the camera has not shown since the last
        restart -- never-seen ones included.

        Re-LOOKING is exploration too. `base`, the map a restart falls back on,
        is what each cell held the first time this episode showed it, and the
        press that first shows a screen is the press that rolled into it: it
        has already crossed that screen's traps and shoved its statues, so what
        the first frame of a room shows is the room AFTER the entrance, not the
        room a restart puts back. That gap is what had the expert reading a
        perfectly winnable board as dead. It cannot be reasoned away from the
        frames -- so it is looked at instead."""
        out = set()
        for sr in range(0, self.h, self.scr_h):
            for sc in range(0, self.w, self.scr_w):
                for r in range(sr, min(sr + self.scr_h, self.h)):
                    row = r * self.w
                    if any(row + c not in self.fresh
                           for c in range(sc, min(sc + self.scr_w, self.w))):
                        out.add((sr // self.scr_h, sc // self.scr_w))
                        break
        return out

    def _panels(self) -> int:
        return self._mask(self.ids.panel)

    def _n_switch(self) -> int:
        ids = self.ids
        return sum(1 for seen in self.mem.values()
                   if seen & {ids.swup, ids.swdown})

    def _mask(self, obj) -> int:
        mask = 0
        for i, seen in self.mem.items():
            if obj in seen:
                mask |= 1 << i
        return mask

    def board_state(self):
        """``(_Board, state)`` built from the knowledge map alone.

        The grid's DIMENSIONS come from the engine, which is geometry rather
        than knowledge: every cell outside the screens the camera has shown is
        ``unknown``, and every level is walled at its boundary, so the extent is
        not something a press could ever be different about.

        The two refutations are DERIVED here rather than remembered, which is
        what keeps them self-correcting:

        * a gate still standing in the map while every known switch is already
          down means a switch exists that has not been found (or, harmlessly,
          that the gate came down while off screen and the memory is stale) --
          either way the honest reading is "flipping what I know about will not
          open it", and the moment the map is corrected the flag goes away;
        * a force field remembered CLOSED while every known panel is held means
          a panel exists that has not been found. Level 11's start is the
          benign case: it ships two closed fields among eight open ones with
          the panel already held, so the flag is set on the very first frame
          and clears one press later, when the `late` rules have made them
          agree.
        """
        state = self._state_from_mem()
        if self._board is not None and self._board_sig == self._epoch:
            return self._board, self._settle(self._board, state)
        ids = self.ids
        wallish = exits = gates = switches = panels = fields = traps = 0
        unknown = 0
        for i in range(self.h * self.w):
            seen = self.mem.get(i)
            b = 1 << i
            if seen is None:
                unknown |= b
                continue
            if seen & {ids.wall, ids.entrance}:
                wallish |= b
            if ids.exit in seen:
                exits |= b
            if ids.gate in seen:
                gates |= b
            if seen & {ids.swup, ids.swdown}:
                switches |= b
            if ids.panel in seen:
                panels |= b
            if seen & {ids.field, ids.openfield}:
                fields |= b
            if seen & {ids.trap, ids.trig, ids.spikes}:
                traps |= b
        board = _Board(self.h, self.w, wallish, exits, gates, switches, panels,
                       fields, traps, unknown, self.extra_switch,
                       self.extra_panel)
        self._board, self._board_sig = board, self._epoch
        return board, self._settle(board, state)

    @staticmethod
    def _settle(board, state):
        """Bring the remembered force fields up to date with where the weights
        are now.

        The map records what the camera last SAW; the `late` rules run every
        turn, on screen or off, and the agent knows them -- so what a field is
        doing now is the rule applied to the last sighting, not the sighting
        itself. Without this the expert believes a field it opened two screens
        ago is still shut, which is exactly the state that used to make it
        conclude the level was lost and press R forever.
        """
        fopn = board.field_update(state[9], _weights(state))
        return state if fopn == state[9] else state[:9] + (fopn,)

    def _state_from_mem(self):
        """The dynamic half of the state, from the same map plus the player the
        window put on screen."""
        ids = self.ids
        sta = bou = swu = tre = trg = spk = fopn = 0
        roll = []
        for i, seen in self.mem.items():
            if not seen:
                continue
            b = 1 << i
            if ids.statue in seen:
                sta |= b
            if ids.brest in seen:
                bou |= b
            else:
                for di, oid in enumerate(ids.bdirs):
                    if oid in seen:
                        roll.append((i, di))
                        break
            if ids.swup in seen:
                swu |= b
            if ids.treasure in seen:
                tre |= b
            if ids.trig in seen:
                trg |= b
            if ids.spikes in seen:
                spk |= b
            if ids.openfield in seen:
                fopn |= b
        return (self.ppos, self.pdir, sta, bou, tuple(sorted(roll)), swu, tre,
                trg, spk, fopn)

    # -- planning over the map ----------------------------------------------
    def succ(self, board, state, di):
        """Where ``di`` leads from ``state``: what was OBSERVED if this press
        has been taken from this state before, else what the map predicts,
        else `UNKNOWN`."""
        key = (state, di)
        if key in self.tried:
            got = self.tried[key]
            return state if got is None else got
        return board.step(state, di)

    def _screen(self, cell) -> tuple:
        return ((cell // self.w) // self.scr_h, (cell % self.w) // self.scr_w)

    def _survey(self, board, state, goal):
        """Forward BFS over presses the map can predict, from ``state``.

        ``goal`` is "win" or "blind"; the sweep stops at the depth of the first
        state that meets it, which both fixes the answer's length and keeps it
        off the rest of the component. Returns ``(preds, targets, depth)``, or
        ``(None, None, None)`` when nothing qualifying is reachable (or the cap
        bit)."""
        stale = self.stale_screens() if goal == "blind" else ()
        seen = {state: 0}
        preds: dict = {}
        queue = deque([state])
        targets: list = []
        depth = None
        while queue:
            cur = queue.popleft()
            d = seen[cur]
            if depth is not None and d >= depth:
                break
            blind = False
            for di in range(4):
                nxt = self.succ(board, cur, di)
                if nxt is UNKNOWN:
                    blind = True
                    continue
                if nxt is None or nxt == cur:
                    continue
                if nxt in seen:
                    if seen[nxt] == d + 1:
                        preds[nxt].append(cur)
                    continue
                seen[nxt] = d + 1
                preds[nxt] = [cur]
                if goal == "win" and board.won(nxt):
                    if depth is None:
                        depth = d + 1
                    targets.append(nxt)
                    continue
                queue.append(nxt)
            if goal == "untried" and depth is None and any(
                    (cur, di) not in self.taken
                    and self.succ(board, cur, di) not in (None, cur)
                    for di in range(4)):
                depth = d + 1
                targets.append(cur)
            if goal == "blind" and depth is None:
                if blind:
                    depth = d + 1              # the blind press itself costs one
                    targets.append(cur)
                elif d and self._screen(cur[0]) in stale:
                    depth = d                  # standing there IS the look
                    targets.append(cur)
            if len(seen) > self.survey_cap:
                return None, None, None
        if depth is None:
            return None, None, None
        return preds, targets, depth

    @staticmethod
    def _back(preds, targets, base: int) -> dict:
        """Presses-to-target for every state, by BFS backwards over ``preds``."""
        dist = {t: base for t in targets}
        queue = deque(targets)
        while queue:
            cur = queue.popleft()
            d = dist[cur]
            for prev in preds.get(cur, ()):
                if prev in dist:
                    continue
                dist[prev] = d + 1
                queue.append(prev)
        return dist

    def decide(self):
        """``(press, optimal set)`` from the current knowledge, or
        ``(None, None)`` when the map offers neither a way out nor anything new
        to look at."""
        board, state = self.board_state()
        if state[0] < 0 or board.won(state):
            return None, None

        if board.exits:                        # an exit has been shown
            preds, targets, depth = self._survey(board, state, "win")
            if targets:
                dist = self._back(preds, targets, 0)
                here = dist.get(state)
                if here:
                    best = [DIRS[di] for di in range(4)
                            if dist.get(self.succ(board, state, di), -1)
                            == here - 1]
                    if best:
                        return best[0], best

        preds, targets, depth = self._survey(board, state, "blind")
        if not targets:
            return self._retry_or_restart(board, state)
        self.explored += 1
        dist = self._back(preds, targets, 1)
        here = dist.get(state)
        if here is None:
            return None, None
        if here == 1:
            best = [DIRS[di] for di in range(4)
                    if self.succ(board, state, di) is UNKNOWN
                    or dist.get(self.succ(board, state, di), -1) == 0]
        else:
            best = [DIRS[di] for di in range(4)
                    if dist.get(self.succ(board, state, di), -1) == here - 1]
        if not best:
            return self._retry_or_restart(board, state)
        return best[0], best

    def _retry_or_restart(self, board, state):
        """The map says the castle is fully seen AND that there is no way out.

        One of those is wrong, and it is nearly always the map: what the frames
        showed was folded in honestly, but a restart puts back rooms whose only
        recorded sighting was taken AFTER the press that rolled into them (see
        `stale_screens`). So before giving up on the level, go and take a press
        this run has not taken -- a different way into the same room shows it
        from a state the last run never reached, and the reading it gives is
        one the map cannot already have wrong.
        """
        preds, targets, depth = self._survey(board, state, "untried")
        if targets:
            self.explored += 1
            dist = self._back(preds, targets, 1)
            here = dist.get(state)
            if here == 1:
                best = [DIRS[di] for di in range(4)
                        if (state, di) not in self.taken
                        and self.succ(board, state, di) not in (None, state)]
                if best:
                    return best[0], best
            elif here:
                best = [DIRS[di] for di in range(4)
                        if dist.get(self.succ(board, state, di), -1) == here - 1]
                if best:
                    return best[0], best
        return self._restart_or_probe(board, state)

    def _restart_or_probe(self, board, state):
        """Nothing to head for and nothing new to look at: press R, or -- if
        that would change nothing -- try something and see.

        RESTARTING is the move this game is built around: a roll cannot be
        taken back, the castle is only visible one screen at a time, and the
        opening message tells the player in so many words to press R. So an
        exploration that strands the level is not a failure, it is how the map
        gets drawn: the restart puts every piece back while the MAP keeps
        everything the camera has shown, and `tried` keeps what each press
        really did, so the presses that were blind last time are predictable
        this time and the search reaches further before it runs out of castle.
        Each cycle therefore explores strictly more, which is what makes the
        loop finite.

        A restart from a board that is already the start state would change
        nothing, and neither would a second one at the same map, so those fall
        through to `_probe`.
        """
        if state != self._start and self._epoch != self._reset_seen:
            self._reset_seen = self._epoch
            self.restarts += 1
            return RESET_PRESS, [RESET_PRESS]
        return self._probe(board, state)

    def _probe(self, board, state):
        """The fallback when the map offers neither a route out nor anything
        new to look at: TRY something and see.

        That situation means the map is complete (nothing unseen is reachable)
        and, under the hypotheses, no win is left -- so either the level really
        is lost, or one of the hypotheses is wrong, and the only way to tell
        them apart is a press. A level's very first frame is the standing case:
        the `late` rules have not run yet, so level 11 ships two force fields
        closed with their panel already held, the panel hypothesis reads that as
        "there is a panel I have not found", and one press is what settles it.

        Presses already taken from this exact state are tried last -- they have
        already been observed, so they are the ones with nothing left to say.
        """
        fresh, stale = [], []
        for di in range(4):
            nxt = self.succ(board, state, di)
            if nxt is None or nxt == state:
                continue
            (stale if (state, di) in self.tried else fresh).append(di)
        order = fresh + stale
        if not order:
            return None, None
        self.explored += 1
        best = [DIRS[di] for di in order[:1]]
        return best[0], best

    # -- PSExpert interface --------------------------------------------------
    def plan(self, eng, level: int | None = None):
        """ONE press, carrying its optimal set, so `record_level` re-derives the
        decision from the live frame at every step -- which is what an adaptive
        policy is."""
        episode = (getattr(self.game, "_seed", 0), level)
        if episode != self._episode:
            # A new (seed, level): forget the castle and re-read the frame the
            # harness has already presented (the `observe` before this call
            # belongs to the level that just ended).
            self._episode = episode
            self._forget()
            self.observe(eng)
        d, best = self.decide()
        if d is None:
            return None
        if d == RESET_PRESS:
            self._pending = None       # a restart is not a transition to learn
            self._restarting = True
            return Plan([d], [best])

        board, state = self.board_state()
        di = DIRS.index(d)
        self._pending = (state, di, self.succ(board, state, di))
        self.taken.add((state, di))
        self.age += 1
        return Plan([d], [best])


class VelocityCastleSolver(PSAStarSolver):
    game_id = "puzzlescript_velocity_castle"
    game_name = GAME_NAME
    expert_cls = VelocityCastleExpert

    #: Record against the GAME FOLDER's adapter. ``games/ps:velocity_castle``
    #: is a plain passthrough today; it is named anyway so a sprite patch or a
    #: step cap added there later cannot make this generator tape a game nobody
    #: plays.
    game_module_id = "ps:velocity_castle"

    #: The expert is ADAPTIVE, so the base class's one-shot "can the expert win
    #: from the start" discovery pass would only be measuring its first press.
    #: Require the real thing: every level must WIN for the seed to count.
    require_all_levels = True

    #: Level 10 (the game's own "Level 11"), and only it, is beyond the honest
    #: expert inside the adapter's 200-press budget -- see the module docstring.
    #: It is three screens wide with a 54-press shortest solution, and the only
    #: way into its west screen is a roll down row 7 that RAMS the statue
    #: standing there. So the first frame the agent ever gets of that room
    #: already has the statue one square along, the map it restarts on carries
    #: that (and the spikes the same run made), and the exploration cannot buy
    #: the room back cheaply enough to leave room for a 54-press win.
    #: ``--honest`` re-measures this: it is 200 presses and 12 restarts short.
    skip_levels = frozenset({10})

    #: Unused: the expert plans over its map rather than by the base class's
    #: engine-side A*. Left at the base values so nothing reads a lie off them.
    weight = 1
    node_cap = 400_000

    #: The stock adapter budget. Exploring and then winning costs 15 to 92
    #: presses (``--honest``) and the recovery prefix at most ~25 more, so the
    #: game needs no raised cap -- but it does need most of one.
    max_steps = 200

    def available_actions(self, game) -> list[int]:
        return [1, 2, 3, 4]                    # ACTION5 fires no rule here


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

#: The 9-tuple's field names, for readable ``--model`` diffs.
_STATE_FIELDS = ("player", "pdir", "statues", "boulders", "rolling",
                 "switches_up", "treasures", "triggered", "spikes",
                 "open_fields")


def _show(value, w):
    """A state field as (row, col) cells where that is what it is."""
    if isinstance(value, tuple):
        return [(divmod(c, w), DIRS[d]) for c, d in value]
    if isinstance(value, int) and value > 4:
        cells, m = [], value
        while m:
            low = m & -m
            cells.append(divmod(low.bit_length() - 1, w))
            m ^= low
        return cells
    return value


def _new(seed: int = 0):
    """``(game, engine, ids)`` -- the raw adapter, for the model-only reports."""
    from adapters.puzzlescript_adapter import PuzzleScriptAdapter
    game = PuzzleScriptAdapter(GAME_NAME, seed=seed)
    return game, game._engine, _Ids(game._game)


def _oracle(level: int, game=None, eng=None, ids=None):
    """``(board, state, Plan | None, how)`` for a level's start, fully informed."""
    if game is None:
        game, eng, ids = _new()
    game.set_level(level)
    board, state = _read(eng, ids)
    plan, how = oracle_plan(board, state)
    return board, state, plan, how


def _replay(game, level: int, plan) -> str:
    """Drive ``plan`` press by press on the REAL interpreter from the level's
    start. Returns "win", or what went wrong -- the one check that the model's
    answer is a genuine win path and not a story about a game of its own."""
    game.set_level(level)
    eng = game._engine
    for i, d in enumerate(plan):
        before = [[frozenset(c) for c in row] for row in eng.grid]
        eng.step(d)
        if [[frozenset(c) for c in row] for row in eng.grid] == before:
            return f"press {i} ({d}) was a no-op on the engine"
        if eng.check_win():
            return "win" if i == len(plan) - 1 else f"won early at press {i}"
    return "did not win"


def _plans(verbose: bool = True) -> int:
    """The ORACLE's answer for every level: the provably shortest win, its tie
    coverage, and a replay of it against the interpreter.

    This is the yardstick, not what the expert plays -- see ``--honest`` for
    that, and `VelocityCastleExpert` for why the two differ on the six levels
    that are more than one screen across."""
    game, eng, ids = _new()
    bad = total = tied = 0
    for level in range(game.n_levels):
        t0 = time.time()
        board, _state, plan, how = _oracle(level, game, eng, ids)
        dt = time.time() - t0
        head = f"level {level:2d} {board.h:2d}x{board.w:2d}"
        if plan is None:
            print(f"{head}: NO PLAN ({how})")
            bad += 1
            continue
        sets = getattr(plan, "optsets", []) or []
        ties = sum(1 for x in sets if len(x) > 1)
        total += len(plan)
        tied += ties
        verdict = _replay(game, level, plan)
        if verdict != "win":
            bad += 1
        if verbose:
            print(f"{head}: shortest {len(plan):3d} presses ({how}), "
                  f"{ties:3d} steps with a tie "
                  f"({ties / max(1, len(plan)):3.0%}), {dt:6.2f}s "
                  f"-- replay {verdict}")
    print(f"total {total} presses, {tied} tie steps, "
          + ("all plans replay to a WIN" if not bad else f"{bad} LEVELS BAD"))
    return 0 if not bad else 1


def _honest(verbose: bool = True, seed: int = 0) -> int:
    """Play every level with the expert the generator actually records: one
    press at a time, from the screens the camera has shown.

    Reports what it cost against the oracle's shortest -- the gap IS the
    exploration, i.e. the presses spent finding switches and doors that were
    off screen -- and whether it won inside the adapter's 200-press budget.

    `skip_levels` is played too rather than skipped: that level's line is the
    measurement the skip rests on, and it is marked instead of counted.
    """
    solver = VelocityCastleSolver()
    game = solver.make_game(seed)
    eng = game._engine
    expert = VelocityCastleExpert(game)
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board, _state, opt, _how = _oracle(level, game, eng, expert.ids)
        game.set_level(level)
        expert.observe(eng)
        presses = tied = 0
        seen0 = None
        while presses < solver.max_steps:
            plan = expert.plan(eng, level)
            if plan is None:
                break
            if seen0 is None:
                seen0 = len(expert.mem)
            if len(plan.optsets[0]) > 1:
                tied += 1
            if plan[0] == RESET_PRESS:
                game.level_reset()
            else:
                eng.step(plan[0])
            presses += 1
            expert.observe(eng)
            if eng.check_win():
                break
        won = eng.check_win()
        skipped = level in solver.skip_levels
        if not won and not skipped:
            bad += 1
        shortest = len(opt) if opt is not None else None
        screens = len(expert.mem) / (expert.scr_w * expert.scr_h)
        if verbose:
            print(f"level {level:2d} {board.h:2d}x{board.w:2d} "
                  f"({board.h * board.w // (expert.scr_w * expert.scr_h)} "
                  f"screens): {'WIN ' if won else 'LOST'}"
                  f"{' [skip_levels]' if skipped else ''} in {presses:3d} "
                  f"presses (shortest {shortest}, "
                  f"{expert.explored:3d} exploring, {expert.restarts} R), "
                  f"{tied:3d} tie steps,"
                  f" saw {screens:.2f} screens, first frame showed "
                  f"{seen0 or 0} cells")
    print("every level won" if not bad else f"{bad} LEVELS LOST")
    return 0 if not bad else 1


def _ties(verbose: bool = True) -> int:
    """Check the expert's optimal-action labels against the FULL game.

    At every press the expert takes, the oracle's exact field is derived from
    the interpreter's live board and its optimal set compared with the one the
    expert labelled the step with. Three outcomes, and all three are meaningful:

    * **exact** -- the label IS the full game's optimal set. Every step of the
      eight single-screen levels should be this: the first frame shows the whole
      board, so planning over what has been shown and planning over everything
      are the same search.
    * **subset** -- the label is a non-empty part of it. Honest and shortest,
      just not the whole tie set: two presses are equally good in the full game
      and the map can only see that one of them is.
    * **off-path** -- the label contains a press that is NOT on a shortest path
      of the full game. That is the exploration, and it is expected on the
      levels that are more than one screen across -- a press spent finding a
      switch is not on any shortest path, because a shortest path already knows
      where the switch is.
    """
    solver = VelocityCastleSolver()
    game = solver.make_game(0)
    eng = game._engine
    expert = VelocityCastleExpert(game)
    ids = expert.ids
    bad = 0
    for level in range(game.n_levels):
        if level in solver.skip_levels:
            print(f"level {level:2d}: skipped (see skip_levels)")
            continue
        game.set_level(level)
        expert.observe(eng)
        exact = subset = off = dead = 0
        for _ in range(solver.max_steps):
            plan = expert.plan(eng, level)
            if plan is None:
                break
            claimed = set(plan.optsets[0])
            if RESET_PRESS in claimed:
                off += 1                      # a restart is never on a shortest path
            else:
                board, state = _read(eng, ids)
                dist, _d, status = board.field(state, FIELD_CAP_FAST)
                if status != "exact":
                    dead += 1
                else:
                    best = {DIRS[di]
                            for di, _n in board.field_optimal(dist, state)}
                    if claimed == best:
                        exact += 1
                    elif claimed <= best:
                        subset += 1
                    else:
                        off += 1
            if plan[0] == RESET_PRESS:
                game.level_reset()
            else:
                eng.step(plan[0])
            expert.observe(eng)
            if eng.check_win():
                break
        if not eng.check_win():
            bad += 1
        total = exact + subset + off + dead
        if verbose:
            print(f"level {level:2d}: {total:3d} labels -- {exact:3d} exact, "
                  f"{subset:3d} subset, {off:3d} off-path, {dead:3d} from a "
                  f"board with no win left"
                  + ("" if eng.check_win() else "   [LEVEL LOST]"))
    print("every recorded level won" if not bad
          else f"{bad} LEVELS LOST")
    return 0 if not bad else 1


def _replay_episode(path, verbose: bool = True) -> int:
    """Replay a recorded episode against a fresh adapter, frame for frame.

    The end-to-end check that the tape is a real game: every recorded action is
    pressed on a newly built adapter at the episode's own seed, and every frame
    it renders must equal the frame the file stores. It is what says the SCREEN
    actions (rotation-remapped) and the RESET presses were recorded the way an
    agent would have to press them, and that the level really ends in a WIN.
    """
    import json

    import numpy as np

    from arcengine import ActionInput, GameState

    from solvers.base_solver import _ID_TO_GAMEACTION

    data = json.loads(Path(path).read_text())
    seed = int(Path(path).stem.split("seed")[-1])
    solver = VelocityCastleSolver()
    game = solver.make_game(seed)
    bad = 0
    for entry in data["levels"]:
        level = entry["level_id"]
        obs, acts = entry["observations"], entry["actions"]
        game._seed = seed
        game.set_level(level)
        frames = [np.asarray(game._current_frame)]
        for act in acts[1:]:
            fd = game.perform_action(
                ActionInput(id=_ID_TO_GAMEACTION[act["index"]]))
            got = list(fd.frame) if act.get("n_obs") else [
                fd.frame[-1] if fd.frame else game._current_frame]
            frames.extend(np.asarray(f) for f in got)
        mism = sum(1 for a, b in zip(frames, obs)
                   if not np.array_equal(a, np.asarray(b)))
        # Every EXPERT step must carry an optimal-action target: a step without
        # one stays in the context and contributes nothing to the loss, which is
        # silent. The exploration prefix and the level's opening RESET are the
        # standing exceptions -- `train_policy` supervises neither.
        blank = sum(1 for a in acts
                    if a.get("phase") == "expert" and not a.get("optimal"))
        ok = (len(frames) == len(obs) and not mism and not blank
              and game._state == GameState.WIN)
        bad += 0 if ok else 1
        if verbose and not ok:
            print(f"  level {level}: {len(frames)} frames vs {len(obs)} "
                  f"recorded, {mism} differ, {blank} expert steps unlabelled, "
                  f"final state {game._state}")
    if verbose:
        steps = sum(len(e["actions"]) for e in data["levels"])
        labelled = sum(1 for e in data["levels"] for a in e["actions"]
                       if a.get("optimal"))
        print(f"{Path(path).name}: {len(data['levels'])} levels, {steps} "
              f"actions ({labelled} labelled), "
              + ("replays frame for frame to a WIN" if not bad
                 else f"{bad} LEVELS DO NOT REPLAY"))
    return bad


def _replays(out_dir, verbose: bool = True) -> int:
    """`_replay_episode` over every episode JSON in a directory."""
    files = sorted(Path(out_dir).glob("episode_*.json"))
    if not files:
        print(f"no episodes in {out_dir}")
        return 1
    bad = sum(_replay_episode(f, verbose) for f in files)
    print("all episodes replay" if not bad else f"{bad} LEVELS BAD")
    return 0 if not bad else 1


def _model(trials: int = 40, presses: int = 60, verbose: bool = True) -> int:
    """Fuzz the native model against the interpreter.

    Random presses from every level's start, driving BOTH, comparing the whole
    board after every one: the model's next state must match what `_read` sees
    on the engine, and `_seat` of that state must reproduce the engine's grid
    cell for cell (which is what checks the derived gate / force-field /
    statue-eye / boulder-light spellings, not just the mechanic). A press the
    model calls a no-op must leave the engine's grid untouched.
    """
    import random

    game, eng, ids = _new()
    bad = 0
    for level in range(game.n_levels):
        mism = 0
        walked = 0
        for trial in range(trials):
            rng = random.Random(1000 * level + trial)
            game.set_level(level)
            board, state = _read(eng, ids)
            for _ in range(presses):
                di = rng.randrange(4)
                before = [[frozenset(c) for c in row] for row in eng.grid]
                expect = board.step(state, di)
                eng.step(DIRS[di])
                walked += 1
                after = [[frozenset(c) for c in row] for row in eng.grid]
                if expect is None:
                    if after != before:
                        mism += 1
                        if verbose and mism <= 3:
                            print(f"  level {level} trial {trial}: model says "
                                  f"no-op for {DIRS[di]}, engine moved")
                        break
                    continue
                _got_board, got = _read(eng, ids)
                if got != expect:
                    mism += 1
                    if verbose and mism <= 3:
                        print(f"  level {level} trial {trial}: {DIRS[di]} at "
                              f"{divmod(state[0], board.w)}")
                        for name, a, b in zip(_STATE_FIELDS, expect, got):
                            if a == b:
                                continue
                            print(f"    {name}: model {_show(a, board.w)} "
                                  f"engine {_show(b, board.w)}")
                    break
                _seat(eng, board, expect, ids)
                seated = [[frozenset(c) for c in row] for row in eng.grid]
                if seated != after:
                    mism += 1
                    if verbose and mism <= 3:
                        diff = [(r, c, sorted(after[r][c] ^ seated[r][c]))
                                for r in range(eng.height)
                                for c in range(eng.width)
                                if after[r][c] != seated[r][c]]
                        print(f"  level {level} trial {trial}: seat mismatch "
                              f"after {DIRS[di]}: {diff[:6]}")
                    break
                state = expect
                if board.won(state):
                    if not eng.check_win():
                        mism += 1
                        if verbose:
                            print(f"  level {level} trial {trial}: model won, "
                                  f"engine did not")
                    break
                if eng.check_win():
                    mism += 1
                    if verbose:
                        print(f"  level {level} trial {trial}: engine won, "
                              f"model did not")
                    break
        bad += mism
        if verbose:
            print(f"level {level:2d}: {walked:5d} presses, "
                  f"{'clean' if not mism else f'{mism} MISMATCHES'}")
    if verbose:
        print("model clean" if not bad else f"MODEL FAILED: {bad} mismatches")
    return bad


def _space(cap: int = 4_000_000) -> int:
    """Enumerate every state each level can reach, with the model.

    Not what the planner runs -- `_Board.field` stops at the first win -- but
    it is what says whether a level's whole space is small enough for the field
    to be the WHOLE story, and it is the report that proves a level winnable or
    not without appeal to a search budget."""
    import time

    game, eng, ids = _new()
    for level in range(game.n_levels):
        game.set_level(level)
        board, start = _read(eng, ids)
        t0 = time.time()
        seen = {start}
        queue = deque([start])
        wins = dead = 0
        capped = False
        while queue:
            cur = queue.popleft()
            for di in range(4):
                nxt = board.step(cur, di)
                if nxt is None or nxt == cur:
                    continue
                if nxt in seen:
                    continue
                if board.won(nxt):
                    wins += 1
                    seen.add(nxt)
                    continue
                if nxt[0] < 0 or nxt[1] >= 0:
                    dead += 1
                seen.add(nxt)
                queue.append(nxt)
            if len(seen) > cap:
                capped = True
                break
        print(f"level {level:2d} {board.h:2d}x{board.w:2d}: "
              f"{len(seen):9d} states{' (CAPPED)' if capped else ''}, "
              f"{wins} winning, {dead} stranded, {time.time() - t0:6.1f}s",
              flush=True)
    return 0


#: Every cell composition a settled frame of this game can show, as the object
#: names that stack in one cell. Derived from the model's own semantics rather
#: than guessed at:
#:
#: * walls, blocks, switches, the entrance arch and the gate/exit squares carry
#:   NO Floor -- the level legend maps ``#``, ``+``, ``U``, ``*`` and ``X`` to
#:   the object alone -- so they are listed without one;
#: * a spike trap under something is always ``triggerspikes``, never
#:   ``spiketrap``: the tick after a weight lands on it
#:   ``[Weight SpikeTrap no TriggerSpikes]`` has already fired, and the cascade
#:   does not settle until it has;
#: * a statue or a boulder standing on a floor panel always wears its marker
#:   (``late [Statue FloorPanel] -> [Statue StatueEyes FloorPanel]`` re-adds it
#:   every turn), so the bare ``statue + floorpanel`` pair cannot be seen;
#: * the four rolling player sprites are here because the WIN frame is one --
#:   the engine stops the cascade the moment the player is on the exit, mid-roll.
_AUDIT_CASES: dict = {
    "void": (),
    "floor": ("floor",),
    "wall": ("wall",),
    "block": ("block",),
    "entrance": ("entrance",),
    "gate_on_exit": ("exit", "gate"),
    "exit": ("exit",),
    "switch_up": ("switchup",),
    "switch_down": ("switchdown",),
    "panel": ("floor", "floorpanel"),
    "field_closed": ("floor", "forcefield"),
    "field_open": ("floor", "openforcefield"),
    "trap": ("floor", "spiketrap"),
    "trap_triggered": ("floor", "triggerspikes"),
    "spikes": ("floor", "spikes"),
    "treasure": ("floor", "treasure"),
    "treasure_void": ("treasure",),
    "statue": ("floor", "statue"),
    "statue_on_panel": ("floor", "floorpanel", "statue", "statueeyes"),
    "boulder": ("floor", "boulderrest"),
    "boulder_on_panel": ("floor", "floorpanel", "boulderrest", "boulderlights"),
    "player": ("floor", "playerrest"),
    "player_up": ("floor", "playerup"),
    "player_down": ("floor", "playerdown"),
    "player_left": ("floor", "playerleft"),
    "player_right": ("floor", "playerright"),
    "player_on_panel": ("floor", "floorpanel", "playerrest"),
    "player_on_open_field": ("floor", "openforcefield", "playerrest"),
    "player_on_trap": ("floor", "triggerspikes", "playerrest"),
    "player_on_exit": ("exit", "playerright"),
}

#: Compositions that render alike and are ALLOWED to, with the reason. These
#: are the game's own design, not a defect of the port: the gate is drawn with
#: the entrance's sprite and palette on purpose (you cannot tell the door you
#: came in through from the door you are trying to open until the switches
#: bring it down), and Wall and Block share a palette. Both members of every
#: pair here behave identically under the mechanic -- all four are Immovable --
#: so a viewer who cannot tell them apart is never misled about what a press
#: will do; see `VelocityCastleExpert.observe`, which folds each pair into one
#: observable class rather than pretending the difference is legible.
_AUDIT_ALLOWED: set = {
    frozenset({"wall", "block"}),
}


def _audit(verbose: bool = True) -> int:
    """Assert every cell composition a settled frame can show is distinct at the
    ONE cell size this game ever renders at.

    ``flickscreen 16x16`` crops every board -- 16x16, 32x32, 16x48 and 48x16
    alike -- to a 16x16 window, and every level's height and width is a multiple
    of 16, so the window is always exactly 16x16 and the cell is always 4px.
    There is no second size to check.

    Whole 64x64 frames of UNIFORM boards are compared rather than one cell out
    of a mixed board: a cell block is not simply indexable once the frame is
    padded or scaled, and two uniform boards render identically iff their cells
    do (the ps:explod lesson).
    """
    import itertools

    import numpy as np

    from adapters.puzzlescript_adapter import _render_frame

    game, eng, ids = _new()
    parsed = game._game
    idx = parsed.obj_name_to_idx
    size = parsed.flickscreen[0]
    assert parsed.flickscreen == (size, size), parsed.flickscreen
    for level in range(game.n_levels):
        game.set_level(level)
        assert eng.height % size == 0 and eng.width % size == 0, (
            f"level {level} is {eng.height}x{eng.width}: not whole screens")

    shots = {}
    for name, objs in _AUDIT_CASES.items():
        eng.height = eng.width = size
        eng.grid = [[{idx["background"]} | {idx[o] for o in objs}
                     for _ in range(size)] for _ in range(size)]
        eng._position_index_dirty = True
        shots[name] = np.asarray(_render_frame(eng, parsed)).copy()

    clashes = [frozenset({a, b})
               for a, b in itertools.combinations(sorted(shots), 2)
               if np.array_equal(shots[a], shots[b])]
    unexpected = [c for c in clashes if c not in _AUDIT_ALLOWED]
    if verbose:
        print(f"{len(shots)} compositions at cell_px={64 // size}, "
              f"{len(clashes)} render alike "
              f"({len(clashes) - len(unexpected)} known-benign)")
        for c in clashes:
            a, b = sorted(c)
            tag = "ok" if c in _AUDIT_ALLOWED else "UNEXPECTED"
            print(f"  {a} == {b}  [{tag}]")
        print("audit clean" if not unexpected
              else f"AUDIT FAILED: {len(unexpected)} unexpected pairs")
    return len(unexpected)


if __name__ == "__main__":
    if "--model" in sys.argv:
        sys.exit(1 if _model() else 0)
    if "--space" in sys.argv:
        sys.exit(_space())
    if "--plans" in sys.argv:
        sys.exit(_plans())
    if "--honest" in sys.argv:
        sys.exit(_honest())
    if "--ties" in sys.argv:
        sys.exit(_ties())
    if "--replay" in sys.argv:
        sys.exit(_replays(sys.argv[sys.argv.index("--replay") + 1]))
    if "--audit" in sys.argv:
        sys.exit(1 if _audit() else 0)
    sys.exit(VelocityCastleSolver.main())
