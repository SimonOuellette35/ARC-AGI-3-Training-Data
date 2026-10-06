"""PuzzleScript game: slidyyyyyyy (mokesmoe's "Slidyyyyyyy")

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Slidyyyyyyy.txt

Play with: python solver_client.py ps:slidyyyyyyy

You cannot take a step. One press launches you and you slide until a wall stops
you; you win by coming to rest ON the target, which means the target is useless
unless there is a wall directly behind it. Five levels.

ONE PATCH: the slide's 5-frames-per-cell ANIMATION is collapsed to the move it
animates
------------------------------------------------------------------------------
The source spells the slide as a cel animation. ``[ > player | no wall ] ->
[pl1 | pr4 marker] again`` deletes the player, draws two half-player sprites
across the two cells and drops an invisible ``marker`` on the destination;
three more rules walk that pair through ``pl2/pr3``, ``pl3/pr2``, ``pl4/pr1``;
a fifth puts the player down on the destination and drops a marker on the cell
BEYOND it; and ``[player | marker] -> [ > player | marker]`` then relaunches
him at that marker, so the whole thing loops one cell at a time until
``[marker wall] -> [wall]`` eats the marker at a wall and the slide stops. With
``again_interval 0.02`` that is 21 of the game's 22 rules spent on making one
slide take a fifth of a second on screen.

None of it is visible here. `PuzzleScriptAdapter.perform_action` renders one
frame per press -- the settled board -- and captures the in-step animation only
for ACTION presses on play-button games, so an agent never sees a ``pl3``, a
``pr1`` or a marker no matter what. What the animation DOES do is cost five
``again`` iterations per cell travelled, and the interpreter caps an ``again``
chain at **50 iterations** (``PSEngine.step``'s ``max_again``, a runaway guard).
Ten cells is therefore the longest slide this engine can finish, and level 5's
row 3 is a thirteen-cell corridor:

    #################
    #.....#..o..#...#
    #.....#.........#
    #.............#.#      <- 13 cells wide
    ##..#........#..#
    #P...#....#..#..#
    #......###......#
    #################

Crossing it truncates. The chain runs out with the player parked two cells
short and a live marker still on the board, and because the marker's sprite is
five rows of ``.....`` that state is pixel-identical to an honest one. The
consequences are all silent: ``no marker`` is half the win condition, so a
truncated slide onto the target would not count as a win; and the next press --
ANY of the four -- resumes the interrupted slide instead of doing what it says,
because ``[player | marker] -> [ > player | marker]`` is the first rule in the
file and fires before the input is read. Two of level 5's board states are
reachable only that way.

So the patch replaces all 22 rules with the one move they animate:

    [ > player | no wall ] -> [ | > player ]

which the interpreter re-fires to a fixpoint inside a single pass -- one press,
one whole slide, no ``again``, no cap. This is a restatement, not a change: the
animation has no side effect on anything except the marker it also cleans up,
it never touches Target (no rule in the file mentions it, which
`_assert_animation_ruleset` checks), and the two win conditions still read the
same board. ``No marker`` becomes trivially true because no marker is ever
created.

That it is a restatement is MEASURED, not argued: ``generate_slidyyyyyyy_
training.py --rules`` enumerates the entire reachable state space of all five
levels under both rulesets and compares every transition. Levels 1-4 agree on
all 392 of them; level 5 differs on exactly the 2 that overrun ``max_again``,
and there the patched engine finishes the slide the truncated one abandoned.
The collapse is also 14x faster per press, which is why the generator's whole
enumeration costs 0.1s.

(The sixteen ``pl*``/``pr*``/``pu*``/``pd*`` part objects and ``marker`` stay
declared and simply never appear. Leaving them keeps the object indices, the
collision layers and the rendering identical to the file, so the sprite audit
in the generator still audits the game as shipped.)
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter

GAME_NAME = "Slidyyyyyyy"

#: The move the 22 animation rules animate: keep going while the next cell is
#: not a wall. The interpreter re-applies a rule until it stops matching within
#: one pass, so this walks the player to the end of the corridor in a single
#: ``eng.step`` and the leftover force is then refused by the wall.
_SLIDE_RULE = "[ > player | no wall ] -> [ | > player ]"

#: Names the animation is allowed to mention. If a rule ever mentions anything
#: else -- Target above all -- the slide is not pure player-vs-wall geometry and
#: collapsing it would drop a mechanic.
_ANIMATION_NAMES = frozenset(
    {"player", "marker", "wall", "part", "left", "right", "up", "down"}
    | {f"p{axis}{i}" for axis in "lrud" for i in range(1, 5)}
)


def _rule_names(rule) -> set:
    """Every object/legend name a rule mentions, on either side.

    A parsed cell is a list of ``(modifier, name, resolved_indices)``; the name
    is what matters here."""
    return {item[1].lower()
            for groups in (rule.groups_lhs, rule.groups_rhs)
            for group in groups
            for cell in group
            for item in cell}


def _is_launch_rule(rule) -> bool:
    """True for one of the four ``DIR [ > player | no wall ] -> ...`` rules --
    the place where "slide while the next cell is not a wall" is written down,
    and hence the move `_SLIDE_RULE` restates."""
    if len(rule.groups_lhs) != 1 or len(rule.patterns_lhs) != 2:
        return False
    first, second = rule.patterns_lhs
    return ([(item[0], item[1]) for item in first] == [(">", "player")]
            and [(item[0], item[1]) for item in second] == [("no", "wall")])


def _assert_animation_ruleset(game) -> None:
    """Fail loudly unless the ruleset is the cel animation this file replaces.

    Three premises, each one a reason the collapse is sound rather than a
    coincidence that happened to hold once:

    * every rule is an ordinary (non-late) rule over the player, the marker,
      the wall and the sixteen part sprites -- in particular **no rule mentions
      Target**, so nothing about the slide depends on, or changes, the thing the
      win condition reads;
    * the four launch rules are present in their ``[ > player | no wall ]``
      shape, which is the move `_SLIDE_RULE` restates -- that is where "slide
      while the next cell is not a wall" is actually written down;
    * ``marker`` and the parts exist only inside this ruleset, so once it is
      gone none of them can ever be placed.
    """
    if not game.rules:
        raise AssertionError("Slidyyyyyyy: no rules parsed at all.")

    stray = {n for rule in game.rules for n in _rule_names(rule)
             if n not in _ANIMATION_NAMES}
    if stray:
        raise AssertionError(
            f"Slidyyyyyyy: rules mention {sorted(stray)}, which the slide "
            "animation never did -- this ruleset is no longer the one the "
            "patch in games/ps:slidyyyyyyy is written for.")

    if any(rule.is_late for rule in game.rules):
        raise AssertionError(
            "Slidyyyyyyy: a `late` rule appeared; the collapse assumes the "
            "whole ruleset runs in the main pass.")

    launched = sorted({d for rule in game.rules if _is_launch_rule(rule)
                       for d in rule.directions})
    if launched != ["down", "left", "right", "up"]:
        raise AssertionError(
            "Slidyyyyyyy: `[ > player | no wall ]` is written for "
            f"{launched} and not for all four directions, so it is not the "
            "launch rule the patch restates.")


def _collapse_slide_animation(game) -> int:
    """Swap the animation for `_SLIDE_RULE`. Returns the number of rules
    dropped.

    ``_parse_rules`` appends to ``game.rules``, so the list is cleared first;
    ``_compile_patterns`` then rebuilds every derived cache from the new rule
    (per-cell index metadata, feature flags, the trivial-rule fast-path tables),
    so nothing downstream keeps a view of the old ones. `PSEngine` caches
    nothing about rules at construction, so the live engine needs no rebuild --
    only a reset, which `make_game` does.
    """
    dropped = len(game.rules)
    game.rules.clear()
    game._parse_rules([_SLIDE_RULE])
    if len(game.rules) != 1:
        raise AssertionError(
            f"Slidyyyyyyy: {_SLIDE_RULE!r} parsed to {len(game.rules)} rules.")
    game._compile_patterns()
    return dropped


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    adapter = PuzzleScriptAdapter(GAME_NAME, seed=seed)
    _assert_animation_ruleset(adapter._game)
    _collapse_slide_animation(adapter._game)
    adapter._do_reset()
    return adapter
