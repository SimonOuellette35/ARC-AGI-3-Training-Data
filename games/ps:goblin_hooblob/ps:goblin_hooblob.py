"""PuzzleScript game: goblin_hooblob (Evan Kuhn and Michael Franklin's
"Goblin Hooblob")

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Goblin_Hooblob.txt

Play with: python solver_client.py ps:goblin_hooblob

Walk the goblin to his gold. Three things are in the way: CratePush, the
ordinary sokoban crate; CratePull, the sticky one, which cannot be shoved at
all and instead FOLLOWS you when you step away from it; and the hoomans, who
each take one step in their own facing every time you move, turn around at any
wall or crate in front of them, and kill you on contact in either direction.

Two facts about the mechanic that the rule listing does not show, and that
every plan here turns on:

  * **There is no wait move.** A press the rules refuse -- into a wall, into a
    pull crate, into a crate with no room behind it -- also matches none of the
    hooman rules (they all require the goblin's destination to be free of
    Objects), so the whole turn is a no-op and the patrols do not advance. The
    only way to spend time is to walk somewhere and back.
  * **A dead goblin can never win.** The second win condition, ``No Splatter on
    Background``, reads as a lose condition: nothing removes a Splatter once a
    hooman has made one.

No per-level step limit is needed here, and that is a checked fact rather than
an omission: `PuzzleScriptAdapter` gives every level 200 actions and flips to
GAME_OVER on the 201st, silently, and the expert plans are

    level      0   1   2   3   4   5
    presses   13  22  18  31  24  36

all proved shortest. See `solvers/generate_goblin_hooblob_training.py --plans`,
which prints the budget column for exactly this check, and `games/ps:cancel`
for the shape this file would take if a level ever stopped fitting.

The two rendering fixes this game needed are in the game FILE, not here -- the
four hooman facings shipped as pixel-identical sprites (so the one piece of
state that decides whether a press is fatal was not on screen at all), and both
crates were solid blocks that hid the gold they were parked on. See the comment
at the top of data/puzzlescript_games/Goblin_Hooblob.txt and
``generate_goblin_hooblob_training.py --audit``.
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Goblin_Hooblob", seed=seed)
