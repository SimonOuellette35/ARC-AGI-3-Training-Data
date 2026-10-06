"""PuzzleScript game: santas_great_escape

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Santa's_Great_Escape.txt

Play with: python solver_client.py ps:santas_great_escape

TWO PATCHES, both faithful rewrites of what the source file already says
-----------------------------------------------------------------------
Neither changes the game; each restates one of its own declarations in a form
this interpreter reads correctly. Without them the game is unplayable: every
delivered gift vanishes and no level can ever be won.

1. ``Santa<N>OnGift`` on a rule's RHS placed only Santa<N>.
   The LEGEND declares twelve AND-composites (``Santa0OnGift = Santa0 and
   Gift``) and the twelve delivery rules put one on their RHS:

       late [ Santa1 Target ] -> [ Santa0onGift ]

   ``PSGame._parse_legend`` files an AND-composite in ``legend`` (not in
   ``or_groups``), so ``resolve_object_name`` correctly returns BOTH member
   indices -- but the two rule-apply paths that consume an RHS item treat a
   multi-index name as an OR-group and pick a single member out of it
   (``_apply_late_rule_match``, ``_apply_rule_match_forces``). The first
   member wins, so Santa was placed and the Gift silently dropped: stepping on
   a target consumed it and left bare floor, and no gift ever appeared on the
   board. The fix here splits each composite RHS item into one item per
   member, which is the same thing spelled out. (This is a general engine bug,
   not a quirk of this game -- 10 of the 810 corpus files put an AND-composite
   on a rule RHS -- but fixing it in the interpreter would change behaviour
   for the other nine, so it is done locally here.)

2. ``All Player on Chimney`` is how this game spells "Santa got out".
   COLLISIONLAYERS puts Player and Chimney on the SAME layer, so no state of
   this game can ever have a Santa standing on a Chimney -- the condition is
   satisfiable only when there is no Player left at all, which is exactly what
   the escape rule arranges:

       [ > Santa0 | Chimney ] -> [ ... | Chimney ]

   (an empty RHS cell: Santa0 walks into the chimney and is removed from the
   board). PuzzleScript reads ``All X on Y`` as vacuously true with no X, but
   this interpreter deliberately refuses that reading when X is a player class
   and the board has no player left -- a guard against an engine slip that
   loses the player "solving" a level (see
   ``PSEngine._check_single_win_condition``). That guard is right for every
   game whose player is supposed to survive, and it makes this one -- whose
   whole win IS the player leaving -- permanently unwinnable.

   So the condition is restated as ``No Player``, which is equivalent on this
   game's own collision layers:

     * some Santa on the board -> it is not on a Chimney (same layer), so
       ``All Player on Chimney`` is false, and ``No Player`` is false;
     * no Santa on the board -> ``All Player on Chimney`` is vacuously true,
       and ``No Player`` is true.

   `_assert_player_and_chimney_collide` checks that premise at load time
   rather than trusting this comment, so an edited COLLISIONLAYERS section
   makes the patch raise instead of silently changing the game.
"""

from adapters.puzzlescript_adapter import PSWinCondition, PuzzleScriptAdapter

GAME_NAME = "Santa's_Great_Escape"


def _split_and_composites(game) -> int:
    """Rewrite every RHS ``A and B`` legend name into its members, in place.

    Returns the number of items rewritten. Cell patterns are mutated as lists,
    so ``rule.patterns_rhs`` and ``rule.groups_rhs`` (which share the same
    objects) both see it; ``_compile_patterns`` then rebuilds every derived
    cache from them -- the per-cell index metadata, the feature flags and the
    ``_trivial_*_per_group`` fast-path tables -- so nothing downstream keeps a
    stale view of the old single-item cell.
    """
    composites = {name: members for name, members in game.legend.items()
                  if len(members) > 1 and name not in game.or_groups}
    n = 0
    for rule in game.rules:
        for group in rule.groups_rhs:
            for cell in group:
                for i in range(len(cell) - 1, -1, -1):
                    mod, name = cell[i][0], cell[i][1]
                    members = composites.get(name)
                    if not members or len(game.resolve_object_name(name)) < 2:
                        continue
                    cell[i:i + 1] = [(mod, member) for member in members]
                    n += 1
    if n:
        game._compile_patterns()
    return n


def _assert_player_and_chimney_collide(game) -> None:
    """Fail loudly unless Player and Chimney share a collision layer.

    That is the premise the win-condition rewrite below rests on: it is what
    makes ``All Player on Chimney`` unsatisfiable-except-vacuously, and hence
    what makes ``No Player`` the same condition rather than a weaker one."""
    players = set(game.resolve_object_name("player"))
    chimney = set(game.resolve_object_name("chimney"))
    for layer in game.collision_layers:
        idxs = {i for name in layer
                for i in game.resolve_object_name(name)}
        if idxs & players and idxs & chimney:
            return
    raise AssertionError(
        "Santa's_Great_Escape: Player and Chimney no longer share a collision "
        "layer, so 'All Player on Chimney' is no longer equivalent to "
        "'No Player' -- the win-condition patch in this file is invalid.")


def _rewrite_escape_win_condition(game) -> bool:
    """``All Player on Chimney`` -> ``No Player``. See the module docstring."""
    for i, wc in enumerate(game.win_conditions):
        if (wc.quantifier == "all" and wc.on_obj == "chimney"
                and set(game.resolve_object_name(wc.obj_name))
                == set(game.resolve_object_name("player"))):
            _assert_player_and_chimney_collide(game)
            game.win_conditions[i] = PSWinCondition("no", wc.obj_name)
            return True
    return False


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    adapter = PuzzleScriptAdapter(GAME_NAME, seed=seed)
    game = adapter._game

    if not _split_and_composites(game):
        raise AssertionError(
            "Santa's_Great_Escape: no AND-composite RHS item found -- the "
            "twelve Santa<N>OnGift delivery rules are gone, so this patch is "
            "no longer describing the game it was written for.")
    if not _rewrite_escape_win_condition(game):
        raise AssertionError(
            "Santa's_Great_Escape: no 'All Player on Chimney' win condition "
            "found -- see this file's docstring for why it is rewritten.")

    adapter._do_reset()
    return adapter
