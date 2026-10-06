"""PuzzleScript game: g2020 (bregehr's "2020")

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/2020.txt

Play with: python solver_client.py ps:g2020

Push the ``2`` and ``0`` blocks until they spell 2020 -- left-to-right or
top-to-bottom -- and get EVERY block on the board into one. The four cells under
a spelled 2020 are marked permanently, so what the win really asks is that every
block stands on a cell that has at some point been inside a pattern; level 5
(six blocks, four cells to a pattern) is won by using that.

No per-level step limits here, and that is a checked fact rather than an
omission: `PuzzleScriptAdapter` gives every level 200 actions and flips to
GAME_OVER on the 201st, silently, and the expert plans are

    level     1   2   3   4   5
    presses   2  21  72  22  58

so the longest is 58 and even 3x the expert's route fits inside the default. See
`games/ps:cancel` for the shape this would take if it ever stopped fitting, and
`solvers/generate_g2020_training.py --plans`, which prints the budget column for
exactly this check.

The one sprite fix this game needed is in the game FILE, not here: ``win``, the
marker object the win condition reads, shipped ``transparent`` and is now drawn
as three yellow corner pips. See the comment at the top of
data/puzzlescript_games/2020.txt and the rendering note in
`solvers/generate_g2020_training.py`.
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("2020", seed=seed)
