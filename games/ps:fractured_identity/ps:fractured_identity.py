"""PuzzleScript game: fractured_identity

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Fractured_Identity.txt

Play with: python solver_client.py ps:fractured_identity

No per-level step limits here, and that is a checked fact rather than an
omission: `PuzzleScriptAdapter` gives every level 200 actions and flips to
GAME_OVER on the 201st, silently, and the expert plans are

    level    0   1   2   3   4   5   6   7   8   9  10
    presses 15  22  31  31  47  28  24  31   -  41  11

(level 8 has no plan -- see `FracturedIdentitySolver.skip_levels`). The longest
is 47, so even 3x the expert's route fits inside the default. See
`games/ps:cancel` for the shape this would take if it ever stopped fitting, and
`solvers/generate_ps_fractured_identity_training.py --plans`, which prints the
budget column for exactly this check.

The sprite fixes this game needed are in the game FILE, not here: they are
transparency and one recolor, which the parser reads directly. See the comment
at the top of data/puzzlescript_games/Fractured_Identity.txt.
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Fractured_Identity", seed=seed)
