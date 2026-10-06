"""PuzzleScript game: idols_to_the_burnt_god (Edalcmagal's "IDOLS TO THE
BURNT GOD")

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/IDOLS_TO_THE_BURNT_GOD.txt

Play with: python solver_client.py ps:idols_to_the_burnt_god

Walk the acolyte through the temple and burn every statue to EXACTLY the right
shade. Each statue has a ladder -- IdolCool, Idol, Burn1, Burn2, Burn3, Ash --
and advances one rung for every press that lands the player orthogonally beside
it, so a press can burn up to four statues at once.

The win condition is a pair of opposite demands, and both are exact:

  * ``no Unburnt`` wants every ordinary statue at Burn3, while ``no Ash`` and
    ``no MiniAsh`` forbid the next rung. Nothing turns ash back into a statue,
    so one press too many is a permanent loss, not a setback -- a level with an
    Ash on it can never be won and only R (or the RESET the solver's recovery
    prefix uses) gets out of it.
  * ``no Heretic`` wants the opposite: a Heretic must be burnt all the way THROUGH
    Burn3 into HereticAsh. The heretic ashes are named by no win condition, so
    they are absorbing and free.

Three facts about the mechanic that the rule listing does not show, and that
every plan turns on:

  * **A refused press is a complete no-op.** ``Move`` is an object the player
    leaves behind when it walks, not a flag, and the burn rules are guarded by
    ``[Player Move | ...]``; a press into a wall, into a full-size statue or
    into a jammed push never re-grants it, so nothing burns and nothing ticks.
    (Nothing on this board moves on its own, so no wait move is needed either.)
  * **Mini statues are sokoban crates and full-size ones are walls.** Only the
    ``MiniStatue`` or-group appears in a push rule; the Idol / Heretic objects
    share the player's collision layer and are immovable. A push is also a
    burn, since the shoved mini lands beside the player that shoved it.
  * **A mini can be pushed ONTO a Wall and is then stuck forever.** Only the
    player has a rule cancelling a move into a wall, and that same rule eats
    the force before the push rule can fire, so the crate can never be shoved
    off again. Level 26 has both walls and minis; this is a real way to lose it.

No per-level step limit is needed here, and that is a checked fact rather than
an omission: `PuzzleScriptAdapter` gives every level 200 actions and flips to
GAME_OVER on the 201st, silently, while the longest expert plan over the 27
shipped levels is 38 presses. See
`solvers/generate_idols_to_the_burnt_god_training.py --plans`, which prints the
budget column for exactly this check.

The three rendering fixes this game needed are in the game FILE, not here --
the walls were the same ARC palette index as the floor, and both heretic ashes
were pixel-identical to the ordinary ashes they are the opposite of. See the
comment at the top of data/puzzlescript_games/IDOLS_TO_THE_BURNT_GOD.txt and
``generate_idols_to_the_burnt_god_training.py --audit``.
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("IDOLS_TO_THE_BURNT_GOD", seed=seed)
