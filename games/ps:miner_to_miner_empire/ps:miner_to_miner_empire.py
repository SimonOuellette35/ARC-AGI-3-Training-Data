"""PuzzleScript game: miner_to_miner_empire

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Miner_To_Miner_Empire.txt

Play with: python solver_client.py ps:miner_to_miner_empire

Walk every miner onto a target. One arrow key moves ALL of them at once -- there
is no select-a-character key -- and the miners share the wall's collision layer,
so the only way a formation ever breaks is by walking part of it into scenery.
The RULES section is empty; everything the game does is the interpreter's own
movement resolution, which settles a queue of miners front-first, so a blocked
leader freezes exactly the miners lined up behind it and nobody else.

Eight levels: the five the file names, plus the four identical copies of the last
one ("Industrial", 11x24, six miners) that its money-counter messages separate.

No per-level step limits are needed here, and that is a checked fact rather than
an omission: `PuzzleScriptAdapter` gives every level 200 actions and flips to
GAME_OVER on the 201st, silently, and the shortest plans are

    level      0   1   2   3   4   5   6   7
    presses   19  26  32  39  39  39  39  39

so even the longest leaves the whole exploration budget spare.
`solvers/generate_miner_to_miner_empire_training.py --plans` prints that column.

The one sprite fix this game needed is in the game FILE, not here: the Player's
legs moved off the two sprite columns that survive the ``cell_px`` 2 sampling the
last four levels render at, because a miner used to cover the target he was
standing on completely -- i.e. on half the game the win condition painted
nothing. See the comment at the top of
data/puzzlescript_games/Miner_To_Miner_Empire.txt and ``--audit``.
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Miner_To_Miner_Empire", seed=seed)
