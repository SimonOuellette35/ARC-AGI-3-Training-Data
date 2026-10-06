"""PuzzleScript game: entrepotphage_demake

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/EntrepotPhage_Demake.txt

Play with: python solver_client.py ps:entrepotphage_demake
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter

#: Per-level step limit, keyed by level index. EntrepotPhage's later boards are
#: full-size warehouse Sokoban -- 6 to 12 crates that have to be routed round
#: each other -- and several of them cannot be finished inside the adapter's
#: 200-step default at all: the expert's shortest known plans are
#:
#:     level   0   1   2   3   4   5   6   7   8   9  10  11  12  13  14
#:     presses 17  62  46  37  30  39  60  65  38  70  38  48  35  51 114
#:
#:     level  15  16  17  18  19  20  21  22  23  25  26  27  28
#:     presses 52  33  52 180  58 190  93  82  72  43  46  41  68
#:
#: (levels 24 and 29 are unsolved -- see the generator's ``skip_levels``), so
#: levels 18 and 20 would flip to GAME_OVER within a dozen presses of the end
#: even for a perfect agent, and 14, 21 and 22 leave no room to wander at all.
#: Each limit here is 3x the expert plan rounded up, floored at the 200 default
#: so no level is made TIGHTER than stock -- a real margin for a policy that
#: explores, while still ending an agent that is looping. See `games/ps:cancel`
#: and `games/ps:count_mover` for the same treatment.
_STEP_LIMITS = {9: 210, 14: 342, 18: 540, 20: 570, 21: 279, 22: 246, 23: 216,
                28: 204}

#: Every other level, and any level added past the table.
_DEFAULT_STEPS = 200


class EntrepotPhageAdapter(PuzzleScriptAdapter):
    """EntrepotPhage Demake with per-level step limits; see `_STEP_LIMITS`.
    Exceeding a level's limit without reaching the win state flips the game to
    GameState.GAME_OVER.
    """

    def _do_reset(self):
        super()._do_reset()
        self._max_steps = _STEP_LIMITS.get(self._current_level_index,
                                           _DEFAULT_STEPS)


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return EntrepotPhageAdapter("EntrepotPhage_Demake", seed=seed)
