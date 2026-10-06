"""Diagnostic-only visual validity checks for supported fixed-layout grid games.

Enumerates rendered player positions for the current level. Inconsistency with
this bank certifies a board cannot be rendered by that level (apart from ignored
HUD pixels). It does NOT certify absence from the model's training distribution.
No policy or live planner imports this helper.
"""
import numpy as np


def legal_maze_boards(env, game_name):
    if game_name == 'synthetic_geodesic':
        from utils.synthetic import render_frame
        g=env._game
        return np.stack([render_frame(g.level.grid,g.level.goal,(int(r),int(c)),g.frame_size)[0]
                         for r,c in zip(*np.where(g.level.grid==0))]).astype(np.uint8)
    if game_name.startswith('gymgw:'):
        from gym_gridworlds.gridworld import Gridworld
        from adapters.gymgridworlds_adapter import _build_arc_frame
        adapter=getattr(getattr(env,'wrapper',None),'_adapter',None)
        engine=getattr(adapter,'_env',None)
        underlying=getattr(engine,'unwrapped',None)
        # Subclasses such as Taxi can change tiles; this bank assumes fixed tiles.
        if type(underlying) is not Gridworld:return None
        grid=underlying.grid
        # Include every cell, even walls: this is a conservative render superset.
        # No state is mutated and rotations use the actual case's adapter.
        return np.stack([adapter._rotate_frame(_build_arc_frame(grid,(r,c)))
                         for r in range(grid.shape[0]) for c in range(grid.shape[1])]).astype(np.uint8)
    if game_name != 'maze':
        return None
    import game_envs
    game=game_envs._get_underlying_game(env.wrapper)
    level=game.current_level
    player=level.get_sprites_by_tag('player')[0]
    position=(player.x,player.y)
    walls={(s.x,s.y) for s in level.get_sprites() if s.name=='wall'}
    width,height=level.grid_size
    boards=[]
    try:
        for y in range(height):
            for x in range(width):
                if (x,y) in walls:continue
                player.set_position(x,y)
                boards.append(np.asarray(game.camera.render(level.get_sprites())).copy())
    finally:
        player.set_position(*position)
    return np.stack(boards).astype(np.uint8)


def minimum_legal_error(board,bank):
    if bank is None:return None
    # Native maze's step-counter HUD lies on an outer edge and can rotate.
    return int(np.count_nonzero(bank[:,1:-1,1:-1]!=np.asarray(board)[None,1:-1,1:-1],axis=(1,2)).min())
