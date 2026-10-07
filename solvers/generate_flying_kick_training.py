"""Generate Phase-1 training data for the PuzzleScript game ps:flying_kick
("Flying Kick", Aaron Steed -- the ninja platformer where the only way to move
sideways through the air is to kick, and the kick does not stop until it hits
something).

The harness -- the rotation contract, the trajectory recorder, the plan cache
and the BaseSolver plumbing -- lives in `solvers/common/ps_astar.py`, shared
with the other ps: generators. This file is the game-specific part: a native
model of the ruleset and the exhaustive search over it.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_flying_kick",
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
Every step also carries a set of equally-optimal presses.

The game
--------
Reach the Exit. Four buttons, no action key (the game declares ``noaction``),
and the whole thing is one two-stroke move: UP to jump, then LEFT or RIGHT to
kick. Every level is that move repeated up a staircase of ledges.

THE MOVEMENT MODEL, which is nothing like a normal platformer:

  * A JUMP RISES EXACTLY TWO CELLS AND THEN HANGS THERE FOREVER. Gravity is two
    rules and both name ``PlayerGround`` or ``PlayerFall``; ``PlayerJump`` is in
    neither, so a hovering ninja waits, indefinitely, for the next press. Under
    a low ceiling the jump rises one cell instead (the second rise is a separate
    rule that simply fails to match), and with a solid cell directly overhead UP
    is a no-op. DOWN drops you out of a hover.
  * THE FLYING KICK IS THE ONLY AIR MOVE, AND IT IS UNSTEERABLE. LEFT/RIGHT
    while hovering spends two ticks winding up and then flies the ninja along
    that ROW, one cell per tick, until something on the Item layer stops it --
    and then drops it straight down from the cell in front of the obstacle. You
    do not choose where you land; the geometry does. That is the whole puzzle:
    every plan here reads as "jump, kick into a wall, land on the ledge that
    wall was holding up, jump again".
  * A KICK PASSES STRAIGHT THROUGH THE EXIT, AND THAT COUNTS. Exit is on the
    scenery layer, not the Item layer, so it never stops a kick -- but the win
    condition is tested after every tick of the animation and the adapter
    freezes the ``again`` loop the moment it holds. All sixteen levels here are
    won in mid-flight, with the ninja drawn as a kick sprite on the exit.
  * KICKING A CRATE COSTS YOU THE FLIGHT. ``[PlayerKickRight | CrateGround]``
    launches the crate, and the very next rule -- ``[PlayerKickRight | Item]``,
    which the launched crate still satisfies -- drops the ninja on the spot. So
    a kick into a crate is a kick that ends where it started. The crate flies on
    alone, one cell per tick, hands its flight to the next resting crate it
    meets, and falls when it stops.
  * CRATES ONLY EVER GO SIDEWAYS AND DOWN. No rule lifts one, so the set of
    heights a crate can reach is fixed at level load. On the ground you also
    PUSH them, one cell per press, which is the other half of every crate
    puzzle: a crate is walked to the edge of a ledge, dropped down a shaft, and
    walked along the corridor below onto a switch.
  * A KICK INTO A BREAKWALL DESTROYS IT, and so does a flying crate. That is the
    only thing that removes scenery.
  * A SWITCH IS HELD DOWN BY ANY ITEM -- a crate, or you -- and a gate is shut
    only while EVERY switch on the board is held: ``[No Item Switch]
    [GateClosed] -> [GateOpen]`` re-opens it the moment one is free. A shut gate
    lands on the Item layer, so it is a floor you can stand on and jump from,
    and it DESTROYS whatever is standing in the doorway when it shuts.
  * SPIKES KILL ON CONTACT, including in mid-flight. The death is not a
    GAME_OVER: blood spreads for a few ticks and a rule then RESTARTs the level
    (which the ADAPTER performs, on the press after the one that killed you).
    So dying only wastes the episode budget -- the search drops those states.

The expert
----------
A NATIVE model of the ruleset (`_Board`) and an exhaustive layered BFS over it.
The model exists because the interpreter runs this game at 150-450 presses/s --
one kick is a 10-30 tick ``again`` chain and every tick rescans the grid for all
43 rules -- where the model runs at 31k-133k. `_Board._tick` is a faithful
TICK-level simulation (main rules in file order, then force resolution, then the
late rules, repeating while a rule asked for ``again`` AND the grid changed),
because the settled outcome of a press really does depend on that interleaving:
a crate kicked out from under a stack is still in flight while the crate that
was resting on it starts to fall.

Two things the .txt does not say and a model has to get right anyway:

  * WHICH RULES KEEP THE PLAYER'S FORCE. A press forces the player once, on the
    first tick only. The force normally dies with the sprite that carried it,
    but it SURVIVES an in-place swap performed by a rule that mentions an
    OR-GROUP, because that takes the rule off the adapter's trivial
    per-cell-substitution fast path. ``[PlayerKickRight | Item] -> [PlayerFall |
    Item]`` therefore drops you one cell PAST the wall you kicked, while its
    concrete-object sibling ``[PlayerKickRight | BreakWall] -> [PlayerFall |
    BreakDebris1]`` drops you straight down. Same split decides whether a crate
    you were pushing keeps the shove. See `_Board.step`.
  * TWO SEPARATE FALL RULES. ``[PlayerFall | No Item]`` moves you and
    ``[PlayerFall | Item]`` lands you, in that order in one tick, so a ninja
    that falls into its last free cell also LANDS on the same tick and the
    frame shows a standing ninja, not a falling one.

``--fuzz`` plays boards press-for-press against the real interpreter and
compares the player (cell AND sprite), every crate, the breakwalls, the gate and
the win flag after every press -- first on random boards built to hit what the
shipped levels barely do (breakwalls beside crates, gates with several switches,
crates stacked so kicking the bottom one drops the top one, spikes everywhere),
then on the shipped boards themselves, whose geometry random rectangles do not
reproduce. It was accepted on 500k+ random-board and 15k on-level transitions,
with zero mismatches.

THE ONE PLACE THE ENGINE IS NOT DETERMINISTIC. ``random down [CrateFall | No
Item]`` advances ONE randomly chosen falling crate per tick, so the moment two
crates are in the air together the schedule is a coin flip -- and the ninja may
be flying through the columns they are falling down. Dropping every press that
merely HAD a choice costs seven of the sixteen levels, so `_Board.step` instead
proves the press: `_all_outcomes` simulates the SET of possible boards, tick by
tick, deduping the frontier (schedules reconverge almost immediately -- two
crates falling down different columns are in the same place two ticks later
whichever moved first), and only a press whose outcome really does depend on the
roll is dropped. It is 0-3% of transitions on the shipped levels and no plan
needs one. `--verify` replays every plan against the real interpreter twenty
times, which is the empirical half of the same claim.

The search
----------
Every press costs 1, so a breadth-first search is already optimal, and the
layered sweep `_Board.bfs` runs afterwards hands back the EXACT set of
equally-shortest presses at every step for free (same argument as ps:flood: a
cross edge to a shallower layer is already too long to lie on a shortest path).
All sixteen plans are therefore certified SHORTEST, and 15 of their 321 steps
carry a genuine tie. ``--ties`` re-derives every one of those sets with an
independent BFS from each alternative press: 1,284 alternatives re-solved over
all sixteen levels, zero disagreements.

The levels
----------
Aaron Steed's sixteen boards in shipped order (the ``message`` screens between
them are not levels and the adapter does not count them). ``states`` is what the
BFS held; the whole run is ~90 s cold and cached in ``data/flying_kick_plans.json``.

    level  size    crates  plan   states   ties
    0      11x8      0        8       22    0
    1      12x14     5       13      108    0
    2      12x13     2       10       34    0
    3      12x9      2       25      190    0
    4      12x9      0       20      153    0
    5      10x14     6       12      479    1
    6      12x11     3       20      427    0
    7      12x20    14       24  247,817    0
    8      10x9      7       22    1,203    2
    9      10x13     6       18   26,221    0
    10     14x13    17       27    2,619    2
    11     11x15     6       22  145,054    3
    12     11x17    12       23  107,035    3
    13     12x17     6       31   28,982    3
    14     11x17    10       16      400    0
    15     11x12     4       30    1,893    1

321 presses over the sixteen, every one replayed through the real interpreter to
a WIN by ``--verify``, and all inside the adapter's 200-step per-level budget
with room for the exploration prefix.

Level 3 is the one worth reading the trace of: it looks unwinnable, because the
exit can only be reached by hovering along row 2, which needs solid ground on
row 4, and the only cell of row 5 that can ever hold an Item is the gate -- whose
one switch sits in a dead-end corridor roofed by wall on every column a crate
could fall down. The answer is that a crate does not have to FALL onto the
switch: it is pushed off the ledge above down the one unroofed shaft, and then
pushed along the corridor onto the switch, which shuts the gate for good and
turns row 5 into the missing floor.

Rendering
---------
Three cell compositions collide once a board is downsampled, and all three are
fixed by editing the sprites in ``data/puzzlescript_games/Flying_Kick.txt``
(which is what every consumer of this game parses):

  * GateOpen and GateClosed were IDENTICAL at cell_px=3 -- levels 7, 12 and 13 --
    because the renderer samples sprite rows/cols {0,2,4} and the two sprites
    agree on all nine of those pixels. Whether the gate is shut is the entire
    puzzle on those boards. GateClosed's centre pixel is now filled in.
  * A ninja STANDING ON THE EXIT was invisible at cell_px=3: the Exit sprite is
    black wherever the player sprite is black. Two of the Exit's pixels are now
    its yellow, so the ninja shows against them -- and that composition is the
    WIN frame of every level.
  * A crate on a SWITCH and a crate on a SPIKE rendered identically at every
    cell size: the crate sprite is opaque except at its four corners, and both
    scenery sprites happened to put the same ARC palette maroon there. SpikeUp's
    bottom corners are now its white.

``--audit`` renders every composition at every cell size in use and asserts they
are pairwise distinct.

CLI
---
    --plans    derive (and cache) every level's plan; prints the table above
    --verify   replay each plan on the real interpreter, 20 times, require WIN
    --ties     re-derive every optimal-set label with an independent BFS
    --audit    the rendering check above
    --fuzz     the differential model check above
    (no flag)  generate episodes: --episodes N --out DIR
"""

from __future__ import annotations

import itertools
import random
import sys
from array import array
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.puzzlescript_adapter import _render_frame              # noqa: E402
from solvers.common.ps_astar import PSAStarSolver, PSExpert, Plan    # noqa: E402

GAME_NAME = "Flying_Kick"

#: The four buttons. The game declares ``noaction``, so there is no fifth.
PRESSES = ("up", "down", "left", "right")

PLAN_CACHE = (Path(__file__).resolve().parent.parent
              / "data" / "flying_kick_plans.json")

# Player modes, in the order the rules mention them.
PG, PF, PJS, PJ, KR, KRB, KRBW, KL, KLB, KLBW = range(10)
# Crate modes.
CG, CF, CR, CL = range(4)

_PLAYER_MODE = {
    "playerground": PG, "playerfall": PF, "playerjumpstart": PJS,
    "playerjump": PJ, "playerkickright": KR, "playerkickrightback": KRB,
    "playerkickrightbackwait": KRBW, "playerkickleft": KL,
    "playerkickleftback": KLB, "playerkickleftbackwait": KLBW,
}
_CRATE_MODE = {"crateground": CG, "cratefall": CF,
               "crateright": CR, "crateleft": CL}


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

class _Board:
    """A tick-level re-implementation of Flying Kick's 43 rules.

    Static geometry (walls, spikes, exits, switches, gate cells) lives on the
    instance; everything that moves lives in the state tuple

        (player cell, player mode, crates, breakwalls, gate shut)

    with ``crates`` a sorted tuple of ``(cell, mode)`` and ``breakwalls`` a
    frozenset of cells. ``player cell`` is -1 once the ninja has been killed --
    by a spike, or by a gate shutting on it -- which is a state no plan may
    pass through.

    WHY TICK-LEVEL AND NOT A CLOSED FORM. A press is an ``again`` chain of up
    to 50 ticks and the pieces genuinely interleave: the ninja flies one cell
    per tick while a crate it kicked out from under a stack flies one cell per
    tick beside it and the crate that was resting on that one falls one cell
    per tick behind them. Writing "where does the kick end" as arithmetic gets
    the common case right and the interesting case wrong.
    """

    __slots__ = ("h", "w", "n", "walls", "spikes", "exits", "switches",
                 "gates", "has_gate")

    def __init__(self, h, w, walls, spikes, exits, switches, gates):
        self.h, self.w, self.n = h, w, h * w
        self.walls = frozenset(walls)
        self.spikes = frozenset(spikes)
        self.exits = frozenset(exits)
        self.switches = frozenset(switches)
        self.gates = frozenset(gates)
        self.has_gate = bool(gates and switches)

    def signature(self):
        return (self.h, self.w, self.walls, self.spikes, self.exits,
                self.switches, self.gates)

    def won(self, state):
        return state[0] >= 0 and state[0] in self.exits

    # -- one press -----------------------------------------------------------
    #: Boards `_all_outcomes` may hold before it gives up and the press is
    #: called ambiguous. The shipped levels never pass 20, so this is headroom
    #: against a pathological fuzz board rather than a real budget.
    outcome_cap = 2_000

    def step(self, state, press):
        """``(next state, won, ambiguous)`` after one button press.

        ``ambiguous`` is True when the interpreter's own RNG would have decided
        this press. ``random down [CrateFall | No Item]`` advances ONE randomly
        chosen falling crate per tick, so as soon as two crates are in the air
        together the schedule is a coin flip -- and the ninja may be flying
        through the columns they are falling down.

        Two crates falling in different columns settle identically whatever the
        order, which is the common case and must not be thrown away (drop every
        press that merely HAD a choice and seven of the sixteen levels stop
        being solvable). So a press that had a choice is PROVED, by simulating
        every board the choices admit (`_all_outcomes`) and calling it ambiguous
        only if two of them disagree. That is a proof, not a sample, and it is
        free on the presses that have no choice at all: they run once and the
        empty ``counts`` says so.

        THE FORCE RULE, which is a third of this function and is in neither the
        .txt nor the PuzzleScript documentation. A press puts a force on the
        player ONCE, on the first tick; the ``again`` continuations run with no
        input at all. Within that first tick the force normally dies with the
        player sprite that carried it -- but it SURVIVES the three rules that
        swap one player sprite for another in place and mention an OR-GROUP:

            random down [PlayerFall | Item]      -> [PlayerGround | Item]
            right       [PlayerKickRight | Item] -> [PlayerFall | Item]

        An or-group anywhere in a rule takes it off the adapter's trivial
        per-cell-substitution fast path (which pops the force with the object
        it removes) and onto the generic path, which deliberately transfers a
        force across a same-layer swap of two objects sharing an or-group.
        Their concrete-object siblings -- ``[PlayerKickRight | BreakWall] ->
        [PlayerFall | BreakDebris1]``, ``[PlayerKickRightBack] ->
        [PlayerKickRight]`` -- are trivial and drop it. So pressing LEFT on a
        ninja whose kick is blocked by a WALL does not merely drop it, it drops
        it one cell to the left first; blocked by a BREAKWALL, it drops
        straight down. The difference is worth two cells of landing position
        and it is entirely an artefact of which rule spells out `Item`.
        """
        out, won, counts = self._run(state, press, ())
        if not counts:
            return out, won, False
        outcomes = self._all_outcomes(state, press)
        return out, won, outcomes is None or len(outcomes) > 1

    def _run(self, state, press, sched):
        """One press under ONE crate schedule -- the fast path.

        ``sched[i]`` is which of the tied candidates the i-th ``random`` rule
        match takes (candidates sorted by cell; positions past the end of
        ``sched`` take the first). Returns ``(state, won, counts)``, where
        ``counts`` is how many candidates each choice point actually had, so an
        empty ``counts`` is the proof that the press was deterministic and
        nothing further needs simulating.
        """
        wk = (state[0], state[1], state[2], state[3], state[4], ())
        counts: list[int] = []
        for tick in range(50):
            nwk, again, cnt = self._tick(wk, press if tick == 0 else None,
                                         sched[len(counts):])
            counts.extend(cnt)
            done = not again or nwk == wk
            wk = nwk
            if done or (wk[0] >= 0 and wk[0] in self.exits):
                break
        out = wk[:5]
        return out, (out[0] >= 0 and out[0] in self.exits), counts

    def _all_outcomes(self, state, press):
        """Every state this press can settle in, or None if it blew a cap.

        A BREADTH-first simulation over the SET of possible boards rather than
        a replay per schedule: the schedules reconverge almost immediately (two
        crates falling down different columns are in the same place two ticks
        later whichever moved first), so deduping the frontier every tick turns
        an exponential enumeration into a handful of states. Replaying whole
        presses instead costs 88 runs on level 10's key kick -- all of which
        agree -- and thousands on the bigger boards.
        """
        live = {(state[0], state[1], state[2], state[3], state[4], ())}
        finals = set()
        for tick in range(50):
            nxt = set()
            for wk in live:
                branches = self._tick_all(wk, press if tick == 0 else None)
                if branches is None:
                    return None
                for nwk, again in branches:
                    if (not again or nwk == wk
                            or (nwk[0] >= 0 and nwk[0] in self.exits)):
                        finals.add(nwk[:5])
                    else:
                        nxt.add(nwk)
            if not nxt:
                return finals
            if len(nxt) + len(finals) > self.outcome_cap:
                return None
            live = nxt
        return finals | {w[:5] for w in live}            # hit the tick cap

    def _tick_all(self, wk, pforce):
        """Every ``(next wk, again)`` one tick can produce, over its choices.

        None if the tick's own choice tree is wider than ``outcome_cap`` --
        with several crates tied for every one of the three ``random`` rules
        the product is large, and a press that wide is not one to plan through
        anyway."""
        nwk, again, cnt = self._tick(wk, pforce, ())
        if not cnt:
            return ((nwk, again),)
        out = {(nwk, again)}
        seen = {()}
        stack = [((), cnt)]
        while stack:
            prefix, cnts = stack.pop()
            padded = prefix + (0,) * max(0, len(cnts) - len(prefix))
            for i, k in enumerate(cnts):
                for j in range(1, k):
                    alt = padded[:i] + (j,)
                    if alt in seen:
                        continue
                    seen.add(alt)
                    if len(seen) > self.outcome_cap:
                        return None
                    o, a, c = self._tick(wk, pforce, alt)
                    out.add((o, a))
                    stack.append((alt, c))
        return out

    def _tick(self, wk, pforce, sched):
        """One ``again`` iteration: the main rules in file order, then force
        resolution, then the late rules. ``(next wk, a rule asked for again,
        the choice counts this tick hit)``."""
        h, w, n = self.h, self.w, self.n
        walls, gates, spikes = self.walls, self.gates, self.spikes
        switches, has_gate = self.switches, self.has_gate

        pcell, pmode, crates_t, breaks_t, gate, debris_t = wk
        crates = dict(crates_t)
        breaks = set(breaks_t)
        debris = dict(debris_t)
        counts: list[int] = []
        nsched = len(sched)

        def pick(cand):
            if len(cand) == 1:
                return cand[0]
            i = len(counts)
            counts.append(len(cand))
            return sorted(cand)[sched[i] if i < nsched else 0]

        def item(i):
            return (i in walls or i in breaks or i in crates
                    or i == pcell or (gate and i in gates))

        again = False

        # -- debris decay (rules 2-4) ------------------------------------
        if debris:
            nxt = {}
            for cell, stage in debris.items():
                if stage < 3:
                    nxt[cell] = stage + 1
                    again = True
            debris = nxt

        # -- input rules (5-8): all four match `[<dir> PlayerJump]`, so the
        #    force is consumed by the match whatever it does.
        if pmode == PJ and pforce is not None:
            if pforce == "right":
                pmode = KRBW
            elif pforce == "left":
                pmode = KLBW
            elif pforce == "down":
                pmode, again = PF, True
            pforce = None

        # -- jump (rules 9-11) -------------------------------------------
        if pmode == PJS:
            up = pcell - w
            if up >= 0 and not item(up):
                pcell = up                               # rule 9
            pmode, pforce = PJ, None

        elif pmode == PG and pforce == "up":
            up = pcell - w
            if up >= 0 and not item(up):
                pcell, pmode, pforce, again = up, PJS, None, True

        # -- falling (rules 12-13). Two SEPARATE rules, so a ninja that
        #    falls into its last free cell also lands in the same tick.
        if pmode == PF:
            dn = pcell + w
            if dn < n and not item(dn):
                pcell, pforce, again = dn, None, True
        if pmode == PF:
            dn = pcell + w
            if dn < n and item(dn):
                pmode = PG                               # landed: force kept

        # -- the flying kick (rules 14-25) -------------------------------
        elif pmode == KR or pmode == KL:
            d = 1 if pmode == KR else -1
            if 0 <= (pcell % w) + d < w:
                nxt = pcell + d
                # Launching a crate leaves it an Item in front of the
                # ninja, so `[PlayerKickRight | Item]` fires straight after
                # and the flight ends where it started.
                if crates.get(nxt) == CG:
                    crates[nxt] = CR if d > 0 else CL
                    pmode, again = PF, True
                elif nxt in breaks:
                    breaks.discard(nxt)
                    debris[nxt] = 1
                    pmode, pforce, again = PF, None, True
                elif item(nxt):
                    pmode, again = PF, True
                else:
                    pcell, pforce, again = nxt, None, True
        elif pmode == KRB:
            pmode, pforce, again = KR, None, True
        elif pmode == KRBW:
            pmode, pforce, again = KRB, None, True
        elif pmode == KLB:
            pmode, pforce, again = KL, None, True
        elif pmode == KLBW:
            pmode, pforce, again = KLB, None, True

        # -- the push (rule 26), the only rule that forces a crate -------
        #
        # ``pushed`` is the cell of the crate now carrying that force. The
        # crate rules below can take it away again -- same trivial-rule
        # split as the player's force: the ones naming a concrete object
        # (`[CrateRight | CrateGround]`, `[CrateRight | BreakWall]`) drop
        # the crate's force with the sprite they remove, and the ones
        # naming an or-group (`[CrateRight | Item]`, `[CrateFall | Item]`)
        # hand it on. That is why kicking a crate into a breakwall leaves
        # the ninja standing where it was, while kicking one into a WALL
        # shoves it along.
        pushed = -1
        if pcell >= 0 and pforce in ("left", "right"):
            d = 1 if pforce == "right" else -1
            if 0 <= (pcell % w) + d < w and (pcell + d) in crates:
                pushed = pcell + d

        # -- crate gravity (rules 27-28), both `random`: one match each --
        drop = [c for c, m in crates.items()
                if m == CF and c + w < n and not item(c + w)]
        if drop:
            c0 = pick(drop)
            del crates[c0]
            crates[c0 + w] = CF
            if pushed == c0:
                pushed = -1
            again = True
        land = [c for c, m in crates.items()
                if m == CF and c + w < n and item(c + w)]
        if land:
            crates[pick(land)] = CG               # no `again`

        # -- crate flight (rules 29-36) ----------------------------------
        for mode, d in ((CR, 1), (CL, -1)):
            # 29: a flying crate that reaches a resting one hands the
            #     flight over to it and drops on the spot -- so a row of
            #     crates propagates the kick along in a single tick.
            while True:
                hit = None
                for c0, m in crates.items():
                    if (m == mode and 0 <= (c0 % w) + d < w
                            and crates.get(c0 + d) == CG):
                        hit = c0
                        break
                if hit is None:
                    break
                crates[hit] = CF
                crates[hit + d] = mode
                if pushed in (hit, hit + d):
                    pushed = -1
                again = True
            for c0 in [c for c, m in crates.items() if m == mode]:
                if 0 <= (c0 % w) + d < w and (c0 + d) in breaks:
                    breaks.discard(c0 + d)
                    debris[c0 + d] = 1
                    crates[c0] = CF
                    if pushed == c0:
                        pushed = -1
                    again = True
            for c0 in [c for c, m in crates.items() if m == mode]:
                if 0 <= (c0 % w) + d < w and item(c0 + d):
                    crates[c0] = CF
                    again = True
            fly = [c for c, m in crates.items()
                   if m == mode and 0 <= (c % w) + d < w
                   and not item(c + d)]
            if fly:
                c0 = pick(fly)
                del crates[c0]
                crates[c0 + d] = mode
                if pushed == c0:
                    pushed = -1
                again = True

        # -- force resolution --------------------------------------------
        if pcell >= 0 and pforce is not None:
            d = {"up": -w, "down": w, "left": -1, "right": 1}[pforce]
            tgt = pcell + d
            ok = 0 <= tgt < n
            if ok and pforce in ("left", "right"):
                ok = 0 <= (pcell % w) + d < w
            if ok:
                if tgt == pushed and tgt in crates:
                    ct = tgt + d
                    if (0 <= ct < n and 0 <= (tgt % w) + d < w
                            and not item(ct)):
                        crates[ct] = crates.pop(tgt)
                        pcell = tgt
                elif not item(tgt):
                    pcell = tgt

        # -- late rules ---------------------------------------------------
        if pcell >= 0 and pmode == PG:
            dn = pcell + w
            if dn < n and not item(dn):
                pmode, again = PF, True
        for cell in [c for c, m in crates.items() if m == CG]:
            dn = cell + w
            if dn < n and not item(dn):
                crates[cell] = CF
                again = True
        if pcell >= 0 and pcell in spikes:
            pcell, pmode, again = -1, PG, True
        if has_gate:
            occupied = [item(s) for s in switches]
            if not gate and all(occupied):
                gate = True
                for cell in gates:            # a shutting gate crushes
                    crates.pop(cell, None)
                    if pcell == cell:
                        pcell, pmode = -1, PG
            elif gate and not all(occupied):
                gate = False

        return ((pcell, pmode, tuple(sorted(crates.items())),
                 frozenset(breaks), gate, tuple(sorted(debris.items()))),
                again, counts)



    # -- the exhaustive search ----------------------------------------------
    def bfs(self, state, cap=400_000):
        """Layered breadth-first search over the four presses.

        Returns ``(plan, optsets, states)`` -- a SHORTEST press sequence to a
        win and, for every step, the exact set of presses that are equally
        shortest -- or ``(None, None, states)`` when the level cannot be won or
        ``cap`` states are exhausted.

        Every press costs 1, so BFS is already the optimal search; what it also
        buys is the labelling. A state ``j`` can only be an optimal successor
        of ``i`` when ``depth[j] == depth[i] + 1``: a cross edge to a shallower
        ``j`` would need ``depth[j] + f(j) >= d*`` with ``depth[j] <
        depth[i] + 1``, i.e. ``f(j) > d* - depth[i] - 1``, which is already too
        long to lie on a shortest path. So "can still win in exactly the
        presses remaining" propagates backwards through the depth layers of the
        forward edges alone -- one sweep, no reverse adjacency, no re-solve.
        """
        k = len(PRESSES)
        states = [state]
        index = {state: 0}
        depth = [0]
        # succ[k * i + j]: the successor of press j, -1 for a WIN, -2 for a
        # press that killed the ninja, rolled the dice, or was never expanded.
        succ = array("i", [-2] * k)
        frontier = [0]
        won_depth = None
        while frontier and won_depth is None:
            nxt_frontier = []
            for i in frontier:
                s = states[i]
                base = k * i
                for j, press in enumerate(PRESSES):
                    ns, won, amb = self.step(s, press)
                    if amb:
                        continue             # the interpreter would roll a die
                    if won:
                        succ[base + j] = -1
                        won_depth = depth[i] + 1
                        continue
                    if ns[0] < 0:
                        continue             # spiked, or crushed by the gate
                    t = index.get(ns)
                    if t is None:
                        if len(states) >= cap:
                            return None, None, len(states)
                        t = len(states)
                        index[ns] = t
                        states.append(ns)
                        depth.append(depth[i] + 1)
                        succ.extend([-2] * k)
                        nxt_frontier.append(t)
                    succ[base + j] = t
            frontier = nxt_frontier
        if won_depth is None:
            return None, None, len(states)

        # tight[i]: a win is still reachable from i in exactly
        # ``won_depth - depth[i]`` presses.
        tight = bytearray(len(states))
        layers: list[list[int]] = [[] for _ in range(won_depth + 1)]
        for i, dep in enumerate(depth):
            layers[dep].append(i)
        for i in layers[won_depth - 1]:
            base = k * i
            if any(succ[base + j] == -1 for j in range(k)):
                tight[i] = 1
        for dep in range(won_depth - 2, -1, -1):
            for i in layers[dep]:
                base = k * i
                for j in range(k):
                    t = succ[base + j]
                    if t >= 0 and depth[t] == dep + 1 and tight[t]:
                        tight[i] = 1
                        break

        plan, optsets = [], []
        i = 0
        for dep in range(won_depth):
            base = k * i
            best = []
            for j, press in enumerate(PRESSES):
                t = succ[base + j]
                if dep == won_depth - 1:
                    if t == -1:
                        best.append((press, t))
                elif t >= 0 and depth[t] == dep + 1 and tight[t]:
                    best.append((press, t))
            plan.append(best[0][0])
            optsets.append([pr for pr, _t in best])
            i = best[0][1]
        return plan, optsets, len(states)


# ---------------------------------------------------------------------------
# Reading a board out of the interpreter
# ---------------------------------------------------------------------------

def read_board(eng, g):
    """``(_Board, state)`` for the interpreter's current grid."""
    inv = {v: k for k, v in g.obj_name_to_idx.items()}
    w = eng.width
    walls, spikes, exits, switches, gates, breaks = (set() for _ in range(6))
    crates: dict[int, int] = {}
    pcell, pmode, gate = -1, PG, False
    for r, row in enumerate(eng.grid):
        for c, cell in enumerate(row):
            i = r * w + c
            for o in cell:
                name = inv[o]
                if name in ("walltop", "wallbase"):
                    walls.add(i)
                elif name in ("spikeup", "mine"):
                    spikes.add(i)
                elif name == "exit":
                    exits.add(i)
                elif name == "switch":
                    switches.add(i)
                elif name == "gateopen":
                    gates.add(i)
                elif name == "gateclosed":
                    gates.add(i)
                    gate = True
                elif name == "breakwall":
                    breaks.add(i)
                elif name in _CRATE_MODE:
                    crates[i] = _CRATE_MODE[name]
                elif name in _PLAYER_MODE:
                    pcell, pmode = i, _PLAYER_MODE[name]
    board = _Board(eng.height, w, walls, spikes, exits, switches, gates)
    state = (pcell, pmode, tuple(sorted(crates.items())),
             frozenset(breaks), gate)
    return board, state


# ---------------------------------------------------------------------------
# The expert
# ---------------------------------------------------------------------------

class FlyingKickExpert(PSExpert):
    """Plans read off the native model; `PSExpert` supplies the in-memory memo,
    the on-disk start-plan cache and the snapshot discipline, so only `_search`
    is overridden."""

    directions = PRESSES
    plan_cache_path = PLAN_CACHE

    #: States the exhaustive BFS may hold. The heaviest level (7) needs 248k,
    #: so this is modest headroom and still bounds a runaway to under a minute
    #: and a few hundred MB -- which matters, because `parallelize_generator`
    #: runs several shards at once.
    bfs_cap = 400_000

    def setup(self) -> None:
        self._board = None
        self._board_sig = None
        #: States the last `_search` held, for `_report`. None when the plan
        #: came off the disk cache and no search ran.
        self.last_states = None

    def heuristic(self, eng) -> int:                 # pragma: no cover
        raise NotImplementedError("the model owns the search; see _Board.bfs")

    def _board_for(self, board):
        """Reuse one `_Board` per level so nothing static is rebuilt per call."""
        sig = board.signature()
        if sig != self._board_sig:
            self._board, self._board_sig = board, sig
        return self._board

    def _search(self, eng) -> list | None:
        board, state = read_board(eng, self.g)
        if state[0] < 0:
            return None                              # no ninja on the board
        board = self._board_for(board)
        plan, optsets, self.last_states = board.bfs(state, self.bfs_cap)
        if plan is None:
            return None
        return Plan(plan, optsets)


class FlyingKickSolver(PSAStarSolver):
    game_id = "puzzlescript_flying_kick"
    game_name = GAME_NAME
    expert_cls = FlyingKickExpert

    #: Record against the game folder's adapter -- the one `game_envs` hands a
    #: live agent -- so a later step limit or patch there cannot silently
    #: diverge from what is taped here.
    game_module_id = "ps:flying_kick"

    #: All sixteen levels are winnable, so nothing is skipped and the startup
    #: discovery pass finds a plan for every one of them (from the disk cache,
    #: after the first process has paid for the searches).
    skip_levels = frozenset()

    #: The longest plan is level 13's 31 presses; the rest is room for the
    #: exploration prefix and its RESET. Stays inside the adapter's 200-step
    #: per-level budget.
    max_steps = 120


# ---------------------------------------------------------------------------
# Reports and checks
# ---------------------------------------------------------------------------

def _levels(solver=None):
    solver = solver or FlyingKickSolver()
    game = solver.make_game(0)
    return game, FlyingKickExpert(game)


def _report() -> int:
    """Derive (and cache) every level's plan; print the module's level table.

    ``states`` is blank for a level whose plan came straight off the disk
    cache -- there was no search to count. Delete
    ``data/flying_kick_plans.json`` for the cold numbers."""
    game, expert = _levels()
    eng = game._engine
    total = solved = 0
    for level in range(game.n_levels):
        game.set_level(level)
        _board, state = read_board(eng, game._game)
        crates = len(state[2])
        if level in FlyingKickSolver.skip_levels:
            print(f"level {level:2d}: SKIPPED")
            continue
        expert.last_states = None
        found = expert.plan(eng, level)
        if found is None:
            print(f"level {level:2d}: NO PLAN "
                  f"({expert.last_states} states closed)")
            continue
        sets = getattr(found, "optsets", []) or []
        ties = sum(1 for s in sets if len(s) > 1)
        total += len(found)
        solved += 1
        states = ("" if expert.last_states is None
                  else f"{expert.last_states:>8,} states,")
        room = "" if len(found) < game._max_steps else " OVER BUDGET"
        print(f"level {level:2d}: {eng.height:2d}x{eng.width:2d} "
              f"{crates:2d} crates {states:>16} "
              f"{len(found):3d} presses, {ties:2d} with a tie set{room}")
    print(f"total {total} presses over {solved} solved levels "
          f"({game.n_levels} shipped)")
    return 0


def _verify(repeats: int = 20) -> int:
    """Replay every level's plan on the REAL interpreter and require a WIN.

    The plans come off a model; this is the line that says the model and the
    interpreter agree about the boards that actually ship. Repeated, because
    the interpreter's crate gravity is a ``random`` rule: a plan that only
    happens to work on one roll would record a level that fails to replay.
    """
    game, expert = _levels()
    eng = game._engine
    bad = 0
    for level in range(game.n_levels):
        if level in FlyingKickSolver.skip_levels:
            continue
        game.set_level(level)
        plan = expert.plan(eng, level)
        if plan is None:
            print(f"level {level:2d}: NO PLAN")
            bad += 1
            continue
        wins = 0
        for _ in range(repeats):
            game.set_level(level)
            for d in plan:
                eng.step(d)
                if eng.check_win():
                    break
            wins += eng.check_win()
        bad += wins != repeats
        print(f"level {level:2d}: {len(plan):3d} presses -> "
              f"{wins}/{repeats} WIN")
    print("verify clean" if not bad else f"VERIFY FAILED on {bad} levels")
    return 0 if not bad else 1


def _ties(levels=None) -> int:
    """Re-derive every step's tie set with a fresh BFS and compare.

    An independent check of the layered argument `_Board.bfs` labels with: at
    each step of the plan, press everything, BFS from each successor for its
    true distance to a win, and assert that the presses reaching one in the
    moves remaining are exactly the labelled set. Both an unlabelled press that
    ties (the label claims too little) and a labelled press that does not (it
    claims too much) fail.

    Cost is one full BFS per alternative per step, so it is seconds for most
    levels and tens of minutes for the four whose search holds 100k+ states
    (7, 9, 11, 12). Pass level numbers to check a subset:
    ``--ties 0 1 2 3 4 5 6 8 10 13 14 15`` is the fast half.
    """
    game, expert = _levels()
    eng = game._engine
    bad = tested = 0
    for level in range(game.n_levels):
        if level in FlyingKickSolver.skip_levels:
            continue
        if levels and level not in levels:
            continue
        game.set_level(level)
        board, state = read_board(eng, game._game)
        board = expert._board_for(board)
        plan = expert.plan(eng, level)
        if plan is None:
            continue
        sets = getattr(plan, "optsets", None) or [[p] for p in plan]
        for i, press in enumerate(plan):
            remaining = len(plan) - i
            measured = []
            for alt in PRESSES:
                ns, won, amb = board.step(state, alt)
                tested += 1
                if amb:
                    continue
                if won:
                    if remaining == 1:
                        measured.append(alt)
                    continue
                if ns[0] < 0:
                    continue
                sub, _o, _n = board.bfs(ns, expert.bfs_cap)
                if sub is not None and 1 + len(sub) == remaining:
                    measured.append(alt)
            if sorted(measured) != sorted(sets[i]):
                bad += 1
                print(f"level {level:2d} step {i:3d}: labelled {sets[i]} "
                      f"but measured {measured}")
            state, _won, _amb = board.step(state, press)
        print(f"level {level:2d}: {len(plan):3d} steps checked", flush=True)
    print(f"ties: {tested} alternatives re-solved, {bad} disagreements")
    return 0 if not bad else 1


# ---------------------------------------------------------------------------
# Rendering audit
# ---------------------------------------------------------------------------

def _audit() -> int:
    """Assert every cell COMPOSITION is distinct at every cell size in use.

    Composition, not object: a crate ON a switch is what holds a gate shut and
    has to differ from a crate on the floor, and the ninja standing on one has
    to differ from the ninja standing on grass.

    Rendering is per-cell independent, so filling the WHOLE board with one
    composition and comparing whole frames is the same test as cropping a cell
    -- done where `_render_frame`'s upscale-to-64 cannot make the crop land in
    the wrong place. See the ps:explod note on that.
    """
    game = FlyingKickSolver().make_game(0)
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    comps = {
        "floor": (), "walltop": ("walltop",), "wallbase": ("wallbase",),
        "breakwall": ("breakwall",), "spike": ("spikeup",), "mine": ("mine",),
        "exit": ("exit",), "switch": ("switch",),
        "gate_open": ("gateopen",), "gate_shut": ("gateclosed",),
        "crate": ("crateground",),
        "crate_on_switch": ("switch", "crateground"),
        "crate_on_exit": ("exit", "crateground"),
        "crate_on_spike": ("spikeup", "crateground"),
        "crate_on_gate_open": ("gateopen", "crateground"),
        "ninja": ("playerground",), "ninja_hover": ("playerjump",),
        "ninja_on_exit": ("exit", "playerground"),
        "ninja_on_switch": ("switch", "playerground"),
        "ninja_on_gate_open": ("gateopen", "playerground"),
        "hover_on_exit": ("exit", "playerjump"),
        "kick_on_exit": ("exit", "playerkickleft"),
    }

    sizes: dict[tuple[int, int], list[int]] = {}
    for level in range(game.n_levels):
        game.set_level(level)
        sizes.setdefault((eng.height, eng.width), []).append(level)

    def shoot(h, w, objs):
        eng.height, eng.width = h, w
        eng.grid = [[{idx["background"]} | {idx[o] for o in objs}
                     for _ in range(w)] for _ in range(h)]
        eng._position_index_dirty = True
        return np.asarray(_render_frame(eng, g))

    bad = 0
    for (h, w), levels in sorted(sizes.items()):
        shots = {name: shoot(h, w, objs) for name, objs in comps.items()}
        clashes = [(a, b) for a, b in itertools.combinations(shots, 2)
                   if np.array_equal(shots[a], shots[b])]
        bad += len(clashes)
        print(f"{h:2d}x{w:2d} (cell {64 // max(h, w)}px, levels "
              f"{','.join(str(x) for x in levels)}): "
              f"{'OK' if not clashes else 'IDENTICAL ' + str(clashes)}")
    print("audit clean" if not bad else f"AUDIT FAILED: {bad} problems")
    return 0 if not bad else 1


# ---------------------------------------------------------------------------
# Differential fuzz
# ---------------------------------------------------------------------------

def _fuzz(n_boards: int = 1500, n_steps: int = 25, seed: int = 0) -> int:
    """Random boards played press-for-press against the real interpreter.

    This is the only thing standing between `_Board` and a silently wrong
    corpus, so the boards are built to hit what the shipped levels barely do:
    breakwalls beside crates, gates with several switches, crates stacked so
    kicking the bottom one drops the top one, and spikes everywhere.

    A board is abandoned as soon as the ninja dies (blood spreads for several
    ticks and then a rule RESTARTs the level, which the ADAPTER handles, not
    ``eng.step``) or the model reports an ambiguous transition (the
    interpreter's crate gravity rolled a die that the model cannot mirror).
    """
    game = FlyingKickSolver().make_game(0)
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    rng = random.Random(seed)
    mismatches = steps = 0
    for _ in range(n_boards):
        h, w = rng.randint(4, 8), rng.randint(4, 8)
        cells = [(r, c) for r in range(h) for c in range(w)]
        rng.shuffle(cells)
        n = len(cells)
        take = 0

        def grab(k):
            nonlocal take
            out = set(cells[take:take + k])
            take += k
            return out

        walls = grab(rng.randint(0, n // 4))
        breaks = grab(rng.randint(0, n // 6))
        crates = grab(rng.randint(0, n // 4))
        # Scenery shares a collision layer, so a cell gets at most one.
        spikes = grab(rng.randint(0, n // 8))
        exits = grab(rng.randint(0, 2))
        switches = grab(rng.randint(0, 2))
        gates = grab(rng.randint(0, 3)) if switches else set()
        shut = bool(gates) and rng.random() < 0.4
        rest = cells[take:]
        player = rest[0] if rest else None
        pname = rng.choice(["playerground", "playerground", "playerjump",
                            "playerjump", "playerfall", "playerkickright",
                            "playerkickleft", "playerjumpstart"])

        grid = []
        for r in range(h):
            row = []
            for c in range(w):
                cell = {idx["background"]}
                p = (r, c)
                if p in spikes:
                    cell.add(idx["spikeup"])
                elif p in exits:
                    cell.add(idx["exit"])
                elif p in switches:
                    cell.add(idx["switch"])
                elif p in gates and not shut:
                    cell.add(idx["gateopen"])
                if p in walls:
                    cell.add(idx["walltop"])
                elif p in breaks:
                    cell.add(idx["breakwall"])
                elif p in crates:
                    cell.add(idx["crateground"])
                elif p in gates and shut:
                    cell.add(idx["gateclosed"])
                elif p == player:
                    cell.add(idx[pname])
                row.append(cell)
            grid.append(row)
        eng.grid, eng.height, eng.width = grid, h, w
        eng._position_index_dirty = True
        eng._rule_noop_cache.clear()
        eng._rule_win = False

        board, state = read_board(eng, g)
        if state[0] < 0:
            continue
        for _s in range(n_steps):
            press = rng.choice(PRESSES)
            eng.step(press)
            truth_won = eng.check_win()
            _b, truth = read_board(eng, g)
            state, won, amb = board.step(state, press)
            steps += 1
            if amb:
                break                       # the engine rolled a die; move on
            if state != truth or won != truth_won:
                mismatches += 1
                print(f"MISMATCH press={press} board {h}x{w}\n"
                      f"  walls {sorted(walls)} breaks {sorted(breaks)}\n"
                      f"  crates {sorted(crates)} spikes {sorted(spikes)}\n"
                      f"  exits {sorted(exits)} switches {sorted(switches)} "
                      f"gates {sorted(gates)} shut={shut}\n"
                      f"  player {player} as {pname}\n"
                      f"  model  {state} won={won}\n"
                      f"  engine {truth} won={truth_won}")
                if mismatches > 4:
                    return 1
                break
            if state[0] < 0 or won:
                break
    print(f"fuzz: {steps} random-board transitions, {mismatches} mismatches")

    # The shipped boards, walked at random. Random rectangles do not reproduce
    # this game's actual geometry -- five-high crate stacks against a ledge,
    # a switch down a roofed corridor -- and it is the shipped boards the
    # plans are derived on.
    lsteps = lbad = 0
    for level in range(game.n_levels):
        for _run in range(30):
            game.set_level(level)
            _b, state = read_board(eng, g)
            board, _s = read_board(eng, g)
            for _ in range(40):
                press = rng.choice(PRESSES)
                eng.step(press)
                truth_won = eng.check_win()
                _b2, truth = read_board(eng, g)
                state, won, amb = board.step(state, press)
                lsteps += 1
                if amb:
                    break
                if state != truth or won != truth_won:
                    lbad += 1
                    print(f"MISMATCH level {level} press={press}\n"
                          f"  model  {state} won={won}\n"
                          f"  engine {truth} won={truth_won}")
                    break
                if state[0] < 0 or won:
                    break
    print(f"fuzz: {lsteps} on-level transitions, {lbad} mismatches")
    return 0 if not (mismatches or lbad) else 1


if __name__ == "__main__":
    if "--plans" in sys.argv:
        sys.exit(_report())
    if "--verify" in sys.argv:
        sys.exit(_verify())
    if "--ties" in sys.argv:
        sys.exit(_ties([int(x) for x in sys.argv[sys.argv.index("--ties") + 1:]
                        if x.isdigit()]))
    if "--audit" in sys.argv:
        sys.exit(_audit())
    if "--fuzz" in sys.argv:
        sys.exit(_fuzz())
    sys.exit(FlyingKickSolver.main())
