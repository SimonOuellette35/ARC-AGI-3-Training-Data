# MIT License
#
# Copyright (c) 2026 ARC Prize Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import random as _random_module
from typing import Any

import numpy as np
from arcengine import (
    ActionInput,
    ARCBaseGame,
    Camera,
    GameAction,
    Level,
    RenderableUserDisplay,
    Sprite,
)
from utils.arc_game import AugmentedGame

sprites = {
    "0": Sprite(
        pixels=[
            [10],
        ],
        name="0",
        visible=True,
        collidable=True,
        tags=["0", "fruit"],
        layer=10,
    ),
    "1": Sprite(
        pixels=[
            [6, 6],
            [6, 6],
        ],
        name="1",
        visible=True,
        collidable=True,
        tags=["fruit", "1"],
        layer=9,
    ),
    "2": Sprite(
        pixels=[
            [15, 15, 15],
            [15, 15, 15],
            [15, 15, 15],
        ],
        name="2",
        visible=True,
        collidable=True,
        tags=["2", "fruit"],
    ),
    "3": Sprite(
        pixels=[
            [11, 11, 11, 11],
            [11, 11, 11, 11],
            [11, 11, 11, 11],
            [11, 11, 11, 11],
        ],
        name="3",
        visible=True,
        collidable=True,
        tags=["3", "fruit"],
    ),
    "4": Sprite(
        pixels=[
            [12, 12, 12, 12, 12],
            [12, 12, 12, 12, 12],
            [12, 12, 12, 12, 12],
            [12, 12, 12, 12, 12],
            [12, 12, 12, 12, 12],
        ],
        name="4",
        visible=True,
        collidable=True,
        tags=["fruit", "4"],
    ),
    "5": Sprite(
        pixels=[
            [8, 8, 8, 8, 8, 8, 8],
            [8, 8, 8, 8, 8, 8, 8],
            [8, 8, 8, 8, 8, 8, 8],
            [8, 8, 8, 8, 8, 8, 8],
            [8, 8, 8, 8, 8, 8, 8],
            [8, 8, 8, 8, 8, 8, 8],
            [8, 8, 8, 8, 8, 8, 8],
        ],
        name="5",
        visible=True,
        collidable=True,
        tags=["fruit", "5"],
    ),
    "6": Sprite(
        pixels=[
            [9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9],
        ],
        name="6",
        visible=True,
        collidable=True,
        tags=["fruit", "6"],
    ),
    "7": Sprite(
        pixels=[
            [7, 7, 7, 7, 7, 7, 7, 7, 7],
            [7, 7, 7, 7, 7, 7, 7, 7, 7],
            [7, 7, 7, 7, 7, 7, 7, 7, 7],
            [7, 7, 7, 7, 7, 7, 7, 7, 7],
            [7, 7, 7, 7, 7, 7, 7, 7, 7],
            [7, 7, 7, 7, 7, 7, 7, 7, 7],
            [7, 7, 7, 7, 7, 7, 7, 7, 7],
            [7, 7, 7, 7, 7, 7, 7, 7, 7],
            [7, 7, 7, 7, 7, 7, 7, 7, 7],
        ],
        name="7",
        visible=True,
        collidable=True,
        tags=["7", "fruit"],
    ),
    "8": Sprite(
        pixels=[
            [14, 14, 14, 14, 14, 14, 14, 14, 14, 14],
            [14, 14, 14, 14, 14, 14, 14, 14, 14, 14],
            [14, 14, 14, 14, 14, 14, 14, 14, 14, 14],
            [14, 14, 14, 14, 14, 14, 14, 14, 14, 14],
            [14, 14, 14, 14, 14, 14, 14, 14, 14, 14],
            [14, 14, 14, 14, 14, 14, 14, 14, 14, 14],
            [14, 14, 14, 14, 14, 14, 14, 14, 14, 14],
            [14, 14, 14, 14, 14, 14, 14, 14, 14, 14],
            [14, 14, 14, 14, 14, 14, 14, 14, 14, 14],
            [14, 14, 14, 14, 14, 14, 14, 14, 14, 14],
        ],
        name="8",
        visible=True,
        collidable=True,
        tags=["8", "fruit"],
        layer=2,
    ),
    "avvxfurrqu": Sprite(
        pixels=[
            [-1, -1, 9, 9, 9, 9, 9, -1, -1],
            [-1, 9, 9, 9, 9, 9, 9, 9, -1],
            [9, 9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9, 9],
            [-1, 9, 9, 9, 9, 9, 9, 9, -1],
            [-1, -1, 9, 9, 9, 9, 9, -1, -1],
        ],
        name="avvxfurrqu",
        visible=True,
        collidable=True,
        tags=["goal"],
        layer=-3,
    ),
    "dawnpfnkpy": Sprite(
        pixels=[
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [5, 7, 7, 5, 5, 14, 14, 5, 5, 13, 13, 5],
            [5, 7, 7, 5, 5, 14, 14, 5, 5, 13, 13, 5],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        ],
        name="dawnpfnkpy",
        visible=True,
        collidable=True,
    ),
    "eifgovhtsm": Sprite(
        pixels=[
            [15, 15, 15],
            [15, 15, 15],
            [15, 15, 15],
        ],
        name="eifgovhtsm",
        visible=True,
        collidable=True,
        tags=["key"],
    ),
    "enemy": Sprite(
        pixels=[
            [-2, -2, 7, -2, -2],
            [-2, 7, -2, 7, -2],
            [7, -2, -2, -2, 7],
            [-2, 7, 7, 7, -2],
        ],
        name="enemy",
        visible=True,
        collidable=True,
        tags=["enemy"],
        layer=-1,
    ),
    "enemy2": Sprite(
        pixels=[
            [-1, -1, 14, -1, -1],
            [-1, 14, -1, 14, -1],
            [14, -1, 14, -1, 14],
            [-1, 14, -1, 14, -1],
        ],
        name="enemy2",
        visible=True,
        collidable=True,
        tags=["enemy2"],
        layer=-1,
    ),
    "enemy3": Sprite(
        pixels=[
            [13, -2, 13, -2, 13],
            [-2, 13, -2, 13, -2],
            [13, 13, 13, 13, 13],
            [-2, 13, 13, 13, -2],
        ],
        name="enemy3",
        visible=True,
        collidable=True,
        tags=["enemy3"],
        layer=-1,
    ),
    "ezepymlzep": Sprite(
        pixels=[
            [-2, -2, 0, -2],
            [-2, 0, 0, 0],
            [-2, -2, 0, -2],
            [-2, -2, -2, -2],
        ],
        name="ezepymlzep",
        visible=True,
        collidable=True,
        tags=["hint"],
        layer=5,
    ),
    "fkfegqgsqk": Sprite(
        pixels=[
            [1, 1, 0, 1, 1],
            [1, 1, 0, 1, 1],
            [0, 0, 0, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ],
        name="fkfegqgsqk",
        visible=True,
        collidable=True,
    ),
    "gcpqtwbmkp": Sprite(
        pixels=[
            [0],
        ],
        name="gcpqtwbmkp",
        visible=True,
        collidable=True,
    ),
    "jlwzvvgvqo": Sprite(
        pixels=[
            [4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4],
        ],
        name="jlwzvvgvqo",
        visible=True,
        collidable=True,
    ),
    "jpjwahlikp": Sprite(
        pixels=[
            [12, 12, 12, 12, 12],
            [12, 12, 12, 12, 12],
            [12, 12, 12, 12, 12],
            [12, 12, 12, 12, 12],
            [12, 12, 12, 12, 12],
        ],
        name="jpjwahlikp",
        visible=True,
        collidable=True,
        tags=["key"],
    ),
    "nswgtbwgsz": Sprite(
        pixels=[
            [4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4],
        ],
        name="nswgtbwgsz",
        visible=True,
        collidable=True,
    ),
    "nsxshyalyp": Sprite(
        pixels=[
            [-1, -1, -1, -1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, -1, -1, -1, -1],
            [4, 4, 4, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 4, 4, 4],
            [4, 4, 4, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 4, 4, 4],
            [-1, -1, -1, -1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, -1, -1, -1, -1],
        ],
        name="nsxshyalyp",
        visible=True,
        collidable=True,
    ),
    "nyvfnpgcbv": Sprite(
        pixels=[
            [-2, -2, 7, -2, -2],
            [-2, 7, -2, 7, -2],
            [7, -2, -2, -2, 7],
            [-2, 7, 7, 7, -2],
        ],
        name="nyvfnpgcbv",
        visible=True,
        collidable=True,
        tags=["key"],
        layer=-1,
    ),
    "oosgzctbjt": Sprite(
        pixels=[
            [6, 6],
            [6, 6],
        ],
        name="oosgzctbjt",
        visible=True,
        collidable=True,
    ),
    "pgizszwemp": Sprite(
        pixels=[
            [-1, -1, 10, -1],
            [10, 10, 10, 10],
            [-1, -1, 10, -1],
        ],
        name="pgizszwemp",
        visible=True,
        collidable=True,
    ),
    "pkqaggtppt": Sprite(
        pixels=[
            [4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4],
        ],
        name="pkqaggtppt",
        visible=True,
        collidable=True,
    ),
    "pzqkrtozkk": Sprite(
        pixels=[
            [13, -2, 13, -2, 13],
            [-2, 13, -2, 13, -2],
            [13, 13, 13, 13, 13],
            [-2, 13, 13, 13, -2],
        ],
        name="pzqkrtozkk",
        visible=True,
        collidable=True,
        tags=["key"],
        layer=-1,
    ),
    "qakdkhhaxs": Sprite(
        pixels=[
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
        ],
        name="qakdkhhaxs",
        visible=True,
        collidable=True,
    ),
    "rcknhqdhzc": Sprite(
        pixels=[
            [4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4],
        ],
        name="rcknhqdhzc",
        visible=True,
        collidable=True,
    ),
    "recfijsnol": Sprite(
        pixels=[
            [11, 11, 11, 11],
            [11, 11, 11, 11],
            [11, 11, 11, 11],
            [11, 11, 11, 11],
        ],
        name="recfijsnol",
        visible=True,
        collidable=True,
        tags=["key"],
    ),
    "spnivuaouo": Sprite(
        pixels=[
            [-1, -1, 9, 9, 9, 9, 9, -1, -1],
            [-1, 9, 9, 9, 9, 9, 9, 9, -1],
            [9, 9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9, 9],
            [-1, 9, 9, 9, 9, 9, 9, 9, -1],
            [-1, -1, 9, 9, 9, 9, 9, -1, -1],
        ],
        name="spnivuaouo",
        visible=True,
        collidable=True,
        tags=["goal"],
        layer=-3,
    ),
    "tixakbqato": Sprite(
        pixels=[
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
        ],
        name="tixakbqato",
        visible=True,
        collidable=True,
        layer=-2,
    ),
    "tltqnwoiek": Sprite(
        pixels=[
            [-1, -1, 14, -1, -1],
            [-1, 14, -1, 14, -1],
            [14, -1, 14, -1, 14],
            [-1, 14, -1, 14, -1],
        ],
        name="tltqnwoiek",
        visible=True,
        collidable=True,
        tags=["key"],
    ),
    "uhtbmsrkmc": Sprite(
        pixels=[
            [1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1],
        ],
        name="uhtbmsrkmc",
        visible=True,
        collidable=True,
    ),
    "vjbztqdvzs": Sprite(
        pixels=[
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        ],
        name="vjbztqdvzs",
        visible=True,
        collidable=True,
        layer=-2,
    ),
    "wmivicdntp": Sprite(
        pixels=[
            [10, -1, 6, -1, 15, -1, 11, -1, 12, -1, 8],
        ],
        name="wmivicdntp",
        visible=True,
        collidable=True,
    ),
    "wovupizsya": Sprite(
        pixels=[
            [-1, -1, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, -1, -1],
            [-1, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, -1],
            [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
            [-1, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, -1],
            [-1, -1, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, -1, -1],
        ],
        name="wovupizsya",
        visible=True,
        collidable=True,
        tags=["goal"],
        layer=-2,
    ),
    "xjbvgededw": Sprite(
        pixels=[
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
        ],
        name="xjbvgededw",
        visible=True,
        collidable=True,
        layer=-1,
    ),
    "xlmseladmx": Sprite(
        pixels=[
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
        ],
        name="xlmseladmx",
        visible=True,
        collidable=True,
        layer=-2,
    ),
    "xozbukeser": Sprite(
        pixels=[
            [10, 6, 15, 11, 12, 8, 9, 7, 14],
        ],
        name="xozbukeser",
        visible=True,
        collidable=True,
    ),
    "zjbjphqtno": Sprite(
        pixels=[
            [-1, -1, -1, 0, 0, 0, 0, 0, -1, -1, -1],
            [-1, -1, 0, -1, -1, -1, -1, -1, 0, -1, -1],
            [-1, 0, -1, -1, -1, -1, -1, -1, -1, 0, -1],
            [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0],
            [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0],
            [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0],
            [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0],
            [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0],
            [-1, 0, -1, -1, -1, -1, -1, -1, -1, 0, -1],
            [-1, -1, 0, -1, -1, -1, -1, -1, 0, -1, -1],
            [-1, -1, -1, 0, 0, 0, 0, 0, -1, -1, -1],
        ],
        name="zjbjphqtno",
        visible=True,
        collidable=True,
        tags=["hint"],
    ),
    "zvhznxdaay": Sprite(
        pixels=[
            [-1, -1, 9, 9, 9, 9, -1, -1],
            [-1, 9, 9, 9, 9, 9, 9, -1],
            [9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9],
            [-1, 9, 9, 9, 9, 9, 9, -1],
            [-1, -1, 9, 9, 9, 9, -1, -1],
        ],
        name="zvhznxdaay",
        visible=True,
        collidable=True,
        layer=-1,
    ),
}
levels = [
    # Level 1
    Level(
        sprites=[
            sprites["2"].clone().set_position(3, 58),
            sprites["avvxfurrqu"].clone().set_position(44, 11),
            sprites["eifgovhtsm"].clone().set_position(30, 4),
            sprites["ezepymlzep"].clone().set_position(8, 52),
            sprites["qakdkhhaxs"].clone(),
            sprites["tixakbqato"].clone(),
            sprites["vjbztqdvzs"].clone().set_position(6, 15),
            sprites["xjbvgededw"].clone().set_position(47, 14),
        ],
        grid_size=(64, 64),
        data={
            "goal": [2, 1],
            "steps": 32,
        },
    ),
    # Level 2
    Level(
        sprites=[
            sprites["0"].clone().set_position(41, 37),
            sprites["0"].clone().set_position(18, 37),
            sprites["0"].clone().set_position(37, 40),
            sprites["0"].clone().set_position(16, 41),
            sprites["0"].clone().set_position(14, 55),
            sprites["0"].clone().set_position(16, 57),
            sprites["0"].clone().set_position(49, 54),
            sprites["0"].clone().set_position(47, 56),
            sprites["avvxfurrqu"].clone().set_position(29, 23),
            sprites["pkqaggtppt"].clone().set_position(16, 0),
            sprites["recfijsnol"].clone().set_position(30, 3),
            sprites["tixakbqato"].clone(),
            sprites["wmivicdntp"].clone().set_position(1, 1).set_scale(2),
        ],
        grid_size=(64, 64),
        data={
            "goal": [3, 1],
            "steps": 32,
        },
    ),
    # Level 3
    Level(
        sprites=[
            sprites["0"].clone().set_position(55, 23),
            sprites["0"].clone().set_position(61, 23),
            sprites["0"].clone().set_position(31, 22),
            sprites["0"].clone().set_position(31, 15),
            sprites["0"].clone().set_position(12, 23),
            sprites["0"].clone().set_position(8, 28),
            sprites["1"].clone().set_position(46, 22),
            sprites["1"].clone().set_position(30, 32),
            sprites["1"].clone().set_position(18, 16),
            sprites["avvxfurrqu"].clone().set_position(5, 46),
            sprites["avvxfurrqu"].clone().set_position(19, 46),
            sprites["eifgovhtsm"].clone().set_position(36, 4),
            sprites["nswgtbwgsz"].clone().set_position(16, 0),
            sprites["recfijsnol"].clone().set_position(30, 3),
            sprites["tixakbqato"].clone(),
            sprites["wmivicdntp"].clone().set_position(1, 1).set_scale(2),
        ],
        grid_size=(64, 64),
        data={
            "goal": [[3, 1], [2, 1]],
            "steps": 48,
        },
    ),
    # Level 4
    Level(
        sprites=[
            sprites["0"].clone().set_position(5, 26),
            sprites["0"].clone().set_position(11, 26),
            sprites["0"].clone().set_position(31, 27),
            sprites["0"].clone().set_position(36, 29),
            sprites["0"].clone().set_position(33, 47),
            sprites["0"].clone().set_position(30, 51),
            sprites["0"].clone().set_position(12, 47),
            sprites["0"].clone().set_position(8, 41),
            sprites["avvxfurrqu"].clone().set_position(1, 53),
            sprites["enemy"].clone().set_position(52, 19),
            sprites["jlwzvvgvqo"].clone().set_position(16, 0),
            sprites["recfijsnol"].clone().set_position(30, 3),
            sprites["tixakbqato"].clone(),
            sprites["wmivicdntp"].clone().set_position(1, 1).set_scale(2),
        ],
        grid_size=(64, 64),
        data={
            "goal": [3, 1],
            "steps": 48,
        },
    ),
    # Level 5
    Level(
        sprites=[
            sprites["0"].clone().set_position(58, 59),
            sprites["0"].clone().set_position(44, 53),
            sprites["0"].clone().set_position(3, 60),
            sprites["0"].clone().set_position(14, 54),
            sprites["1"].clone().set_position(14, 28),
            sprites["1"].clone().set_position(53, 26),
            sprites["1"].clone().set_position(6, 25),
            sprites["1"].clone().set_position(42, 26),
            sprites["avvxfurrqu"].clone().set_position(28, 11),
            sprites["enemy"].clone().set_position(4, 37),
            sprites["enemy"].clone().set_position(46, 37),
            sprites["rcknhqdhzc"].clone().set_position(16, 0),
            sprites["recfijsnol"].clone().set_position(30, 3),
            sprites["tixakbqato"].clone(),
            sprites["wmivicdntp"].clone().set_position(1, 1).set_scale(2),
        ],
        grid_size=(64, 64),
        data={
            "goal": [3, 1],
            "steps": 32,
        },
    ),
    # Level 6
    Level(
        sprites=[
            sprites["5"].clone().set_position(33, 32),
            sprites["avvxfurrqu"].clone().set_position(2, 12),
            sprites["avvxfurrqu"].clone().set_position(52, 53),
            sprites["enemy"].clone().set_position(16, 34),
            sprites["nyvfnpgcbv"].clone().set_position(36, 3),
            sprites["recfijsnol"].clone().set_position(30, 3),
            sprites["tixakbqato"].clone(),
            sprites["wmivicdntp"].clone().set_position(1, 1).set_scale(2),
        ],
        grid_size=(64, 64),
        data={
            "goal": [[3, 1], ["vnjbdkorwc", 1]],
            "steps": 32,
        },
    ),
    # Level 7
    Level(
        sprites=[
            sprites["1"].clone().set_position(9, 25),
            sprites["1"].clone().set_position(20, 35),
            sprites["1"].clone().set_position(6, 35),
            sprites["1"].clone().set_position(30, 37),
            sprites["5"].clone().set_position(51, 46),
            sprites["avvxfurrqu"].clone().set_position(19, 13),
            sprites["avvxfurrqu"].clone().set_position(40, 18),
            sprites["enemy"].clone().set_position(12, 51),
            sprites["enemy"].clone().set_position(52, 56),
            sprites["recfijsnol"].clone().set_position(30, 3),
            sprites["recfijsnol"].clone().set_position(36, 3),
            sprites["tixakbqato"].clone(),
            sprites["wmivicdntp"].clone().set_position(1, 1).set_scale(2),
        ],
        grid_size=(64, 64),
        data={
            "goal": [3, 2],
            "steps": 32,
        },
    ),
    # Level 8
    Level(
        sprites=[
            sprites["3"].clone().set_position(13, 42),
            sprites["3"].clone().set_position(3, 40),
            sprites["5"].clone().set_position(20, 24),
            sprites["avvxfurrqu"].clone().set_position(52, 15),
            sprites["avvxfurrqu"].clone().set_position(3, 15),
            sprites["avvxfurrqu"].clone().set_position(52, 51),
            sprites["avvxfurrqu"].clone().set_position(3, 51),
            sprites["dawnpfnkpy"].clone().set_position(0, 5),
            sprites["enemy"].clone().set_position(43, 31),
            sprites["enemy"].clone().set_position(29, 53),
            sprites["enemy"].clone().set_position(47, 48),
            sprites["jpjwahlikp"].clone().set_position(30, 2),
            sprites["jpjwahlikp"].clone().set_position(37, 2),
            sprites["tixakbqato"].clone(),
            sprites["tltqnwoiek"].clone().set_position(44, 3),
            sprites["wmivicdntp"].clone().set_position(1, 1).set_scale(2),
        ],
        grid_size=(64, 64),
        data={
            "goal": [[4, 2], ["yckgseirmu", 1]],
            "steps": 48,
        },
    ),
    # Level 9
    Level(
        sprites=[
            sprites["1"].clone().set_position(18, 46),
            sprites["1"].clone().set_position(23, 52),
            sprites["5"].clone().set_position(35, 48),
            sprites["avvxfurrqu"].clone().set_position(7, 37),
            sprites["avvxfurrqu"].clone().set_position(49, 51),
            sprites["avvxfurrqu"].clone().set_position(7, 51),
            sprites["dawnpfnkpy"].clone().set_position(0, 5),
            sprites["eifgovhtsm"].clone().set_position(37, 4),
            sprites["enemy"].clone().set_position(51, 13),
            sprites["enemy"].clone().set_position(14, 12),
            sprites["enemy"].clone().set_position(15, 22),
            sprites["enemy"].clone().set_position(54, 33),
            sprites["jpjwahlikp"].clone().set_position(30, 2),
            sprites["pzqkrtozkk"].clone().set_position(42, 3),
            sprites["tixakbqato"].clone(),
            sprites["wmivicdntp"].clone().set_position(1, 1).set_scale(2),
        ],
        grid_size=(64, 64),
        data={
            "goal": [[4, 1], ["vptxjilzzk", 1], [2, 1]],
            "steps": 48,
        },
    ),
]


class musowtfgnt(RenderableUserDisplay):
    def __init__(self, wncfdksibg: "Su15", sukopjjenv: int):
        self.sukopjjenv = sukopjjenv
        self.current_steps = sukopjjenv
        self.wncfdksibg = wncfdksibg
        self.penalty: int = 0

    def nycdrqbsxe(self, kgyxpqvylp: int) -> None:
        self.current_steps = max(0, min(kgyxpqvylp, self.sukopjjenv))

    def kelgufkvkh(self) -> bool:
        if self.current_steps > 0:
            self.current_steps -= 1
        return self.current_steps > 0

    def myafvtlbfr(self) -> bool:
        if self.current_steps > 0:
            self.current_steps -= 2 + self.penalty * 2
            self.penalty += 1
        return self.current_steps > 0

    def walglvjgdc(self) -> None:
        self.current_steps = self.sukopjjenv
        self.penalty = 0

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        oazbqthsrx: float = self.current_steps / self.sukopjjenv
        start_y: int = 63
        ipapfhrgsf = int(np.ceil(oazbqthsrx * 64))
        frame[start_y, 0:ipapfhrgsf] = qdlaeohrgy
        return frame


BACKGROUND_COLOR = 5
PADDING_COLOR = 3
usszepkrvw: int = -1
qdlaeohrgy: int = 0
ifrliiwrop: int = 1
fqcvrfmltv: int = 2
gfyrbolbko: int = 3
ipniyheohy: int = 4
vxrqbiqdnp: int = 5
okftqhwnyx: int = 6
vjyttwdlop: int = 7
eyajamelvr: int = 8
upwpwbdgjx: int = 9
wjofpgosjz: int = 10
opjbyuzoqh: int = 11
pjthgydsvz: int = 12
oybzapqazr: int = 13
mqhwfaprma: int = 14
mvwodydglr: int = 15
cuseckyghy: list[int] = [
    wjofpgosjz,
    okftqhwnyx,
    mvwodydglr,
    opjbyuzoqh,
    pjthgydsvz,
    eyajamelvr,
    upwpwbdgjx,
    vjyttwdlop,
    mqhwfaprma,
]
mgxfziiwqq: list[str] = ["0", "1", "2", "3", "4", "5", "6", "7", "8"]
bjquaudhya: str = "fruit"
geenhzgokh: str = "goal"
oqscouauxq: str = ""
gulrbtyssc: str = "vnjbdkorwc"
wfcfzowdju: str = "yckgseirmu"
dzxwbnerru: str = "vptxjilzzk"
gxqglqjgnq: str = "enemy"
ocvccofipr: str = "enemy2"
zvcltwpfvq: str = "enemy3"
okwzbiftnr: int = 8
aevhcnismc: int = 4
vprzfcbjwy: int = 4
ftaeelrvuu: int = 16
gnexwlqinp: int = 10
ncfmodluov: int = 63


class Su15(AugmentedGame):
    def __init__(self, seed: int | None = None) -> None:
        self.step_counter_ui = musowtfgnt(self, 128)
        self._init_augmentation(seed)
        camera = Camera(
            background=BACKGROUND_COLOR,
            letter_box=PADDING_COLOR,
            interfaces=[self.step_counter_ui],
        )
        self.oaihjsiof: int = 64
        self.kamls: int = 64
        self.hmeulfxgy: list[Sprite] = []
        self.peiiyyzum: list[Sprite] = []
        self.rqdsgrklq: list[Sprite] = []
        self.hirdajbmj: dict[Sprite, str] = {}
        self.amnmgwpkeb: dict[Sprite, int] = {}
        self.oqruhyvee: np.ndarray = np.zeros((0, 2), dtype=np.int64)
        self.xmuutlucr: np.ndarray = np.zeros((0,), dtype=np.int64)
        self.szyyvpgcv: tuple[int, int] = (0, 0)
        self.nhxemszsx: list[Sprite] = []
        self.cqmgrggak: list[Sprite] = []
        self.ackguicmt: int = 0
        self.bjetwxoaq: int = aevhcnismc
        self.qjlubdgly: int = okwzbiftnr
        self.stqbquzms: int = vprzfcbjwy
        self.xmtnegqli: bool = False
        self.itlxknnsz: dict[Sprite, tuple[float, float]] = {}
        self.rzfgsshuk: dict[Sprite, tuple[float, float]] = {}
        self.npdhdupen: set[Sprite] = set()
        self.nbnfqojis: dict[Sprite, int] = {}
        self.zwgbpzcgq: dict[Sprite, tuple[float, float]] = {}
        self.mdomdbjdo: dict[Sprite, tuple[float, float]] = {}
        self.dwehbjuln: dict[Sprite, tuple[float, float]] = {}
        self.tfaferyux: int = 4
        self.djoqfdlzu: int = 4
        self.gylidxxtq: float = 10.0
        self.zrggshnlg: set[Sprite] = set()
        self.citbwsczl: dict[Sprite, int] = {}
        self.xmcssnhit: dict[Sprite, int] = {}
        self.zhlackwpo: Sprite | None = None
        self.ycxcikasw: float = float(self.qjlubdgly)
        self.gzmqvzsoi: float = float(self.qjlubdgly)
        self.qwmmebmhb: np.ndarray | None = None
        self.sqehcqhws: np.ndarray | None = None
        self.eauclepne: list[Sprite] = []
        self.lpbtgxyij: dict[Sprite, np.ndarray] = {}
        self.ilcarpaut: int = 0
        self.nkwbdgqdb: int = ftaeelrvuu
        self.anibpvotxtvdating: bool = False
        self.inhlatxex: str = "none"
        self.reqbygadvzmjired: list[Any] | None = None
        self.dqxbwefew: list[list[tuple[str, int, int, int]]] = []
        self._color_lut: np.ndarray = np.arange(16, dtype=np.int64)
        self._gray_color: int = fqcvrfmltv
        super().__init__(game_id="su15", levels=levels, camera=camera, available_actions=[6, 7])

    def jsthicgwqm(self) -> None:
        oifmnnjbvl = self.current_level.get_data("steps")
        if oifmnnjbvl:
            self.step_counter_ui.sukopjjenv = oifmnnjbvl
            self.step_counter_ui.walglvjgdc()

    def _randomize_colors(self) -> None:
        rng = self.level_rng("colors")
        available = list(range(1, 16))
        shuffled = rng.sample(available, len(available))
        color_map = dict(zip(available, shuffled))
        lut = np.arange(16, dtype=np.int64)
        for old_c, new_c in color_map.items():
            lut[old_c] = new_c
        self._color_lut = lut
        for sprite in self.current_level.get_sprites():
            px = sprite.pixels
            if px is not None:
                mask = px >= 1
                px[mask] = lut[px[mask]]
        self.camera.background = color_map[BACKGROUND_COLOR]
        self.camera.letter_box = color_map[PADDING_COLOR]
        self._gray_color = int(lut[fqcvrfmltv])

    def _remap_sprite(self, sprite: Sprite) -> Sprite:
        px = sprite.pixels
        if px is not None:
            mask = px >= 1
            px[mask] = self._color_lut[px[mask]]
        return sprite

    def on_set_level(self, level: Level) -> None:
        self._levels[self._current_level_index] = self._clean_levels[self._current_level_index].clone()
        self._randomize_colors()
        self.actions: list[ActionInput] = []
        for i in range(16):
            for ituakcbkov in range(14):
                y = 10 + ituakcbkov * 4
                self.actions.append(ActionInput(id=GameAction.ACTION6.value, data={"x": i * 4, "y": y}))
        self.jsthicgwqm()
        oiasidva = self.current_level.grid_size
        if oiasidva is not None:
            self.oaihjsiof = oiasidva[0]
            self.kamls = oiasidva[1]
        self.reqbygadvzmjired = self.current_level.get_data("goal")
        self.koprtgesg = self.current_level.get_sprites_by_tag("key")
        self.liesgkdzq: Sprite | None = None
        sgxxybqgsv = self.current_level.get_sprites_by_tag("hint")
        if self.level_index == 0:
            self.liesgkdzq = sgxxybqgsv[0]
        self.grayed = False
        self.hmeulfxgy = self.current_level.get_sprites_by_tag(bjquaudhya)
        lrxgzrrgul: list[Sprite] = self.current_level.get_sprites_by_tag(gxqglqjgnq)
        dflvyelvmq: list[Sprite] = self.current_level.get_sprites_by_tag(ocvccofipr)
        eiqlefvkeq: list[Sprite] = self.current_level.get_sprites_by_tag(zvcltwpfvq)
        self.peiiyyzum = []
        self.peiiyyzum.extend(lrxgzrrgul)
        self.peiiyyzum.extend(dflvyelvmq)
        self.peiiyyzum.extend(eiqlefvkeq)
        self.rqdsgrklq = self.current_level.get_sprites_by_tag(geenhzgokh)
        self.hirdajbmj = {}
        for vxxrprjnav in lrxgzrrgul:
            self.hirdajbmj[vxxrprjnav] = gulrbtyssc
        for vxxrprjnav in dflvyelvmq:
            self.hirdajbmj[vxxrprjnav] = wfcfzowdju
        for vxxrprjnav in eiqlefvkeq:
            self.hirdajbmj[vxxrprjnav] = dzxwbnerru
        self.amnmgwpkeb.clear()
        for sprite in self.hmeulfxgy:
            kcacvefoot = self.bwcllmldzc(sprite)
            self.amnmgwpkeb[sprite] = kcacvefoot
        self.cmnpjwkivs()
        self.nhxemszsx = []
        self.cqmgrggak = []
        self.itlxknnsz = {}
        self.rzfgsshuk = {}
        self.szyyvpgcv = (0, 0)
        self.ackguicmt = 0
        self.ycxcikasw = float(self.qjlubdgly)
        if self.bjetwxoaq > 1:
            self.gzmqvzsoi = float(self.qjlubdgly) / float(self.bjetwxoaq - 1)
        else:
            self.gzmqvzsoi = float(self.qjlubdgly)
        self.npdhdupen = set()
        self.nbnfqojis = {}
        self.zwgbpzcgq = {}
        self.mdomdbjdo = {}
        self.dwehbjuln = {}
        self.citbwsczl = {}
        self.xmcssnhit = {}
        self.zrggshnlg = set()
        self.zhlackwpo = sprites["gcpqtwbmkp"].clone()
        self.zhlackwpo.pixels = np.full((64, 64), usszepkrvw, dtype=np.int64)
        self.current_level.add_sprite(self.zhlackwpo)
        wskvwndunw = self.zhlackwpo.pixels
        canvas_height, canvas_width = wskvwndunw.shape
        gwxlhgjbac = np.arange(canvas_height, dtype=np.float32)
        afgzzcccpo = np.arange(canvas_width, dtype=np.float32)
        self.sqehcqhws, self.qwmmebmhb = np.meshgrid(gwxlhgjbac, afgzzcccpo, indexing="ij")
        self.jcoktfacgw()
        self.eauclepne = []
        self.lpbtgxyij = {}
        self.ilcarpaut = 0
        self.anibpvotxtvdating = False
        self.inhlatxex = "none"
        self.xmtnegqli = False
        self.win_frame = 0
        self.dqxbwefew = []
        self.sebayllgbc()

    def step(self) -> None:
        if self.anibpvotxtvdating:
            if self.inhlatxex == "win":
                index = 0
                for k in self.koprtgesg:
                    mpalcyrcyw = k.pixels >= 0
                    k.pixels[mpalcyrcyw] = qdlaeohrgy
                    index += 1
                for dmanzruhiz in self.rqdsgrklq:
                    mpalcyrcyw = dmanzruhiz.pixels >= 0
                    dmanzruhiz.pixels[mpalcyrcyw] = qdlaeohrgy
                self.win_frame += 1
                if self.win_frame > 10:
                    self.next_level()
                    self.complete_action()
                return
            elif self.inhlatxex == "vacuum":
                kjmljovana = self.lyaaynsyhw()
                if not self.xmtnegqli and self.ackguicmt >= self.bjetwxoaq:
                    self.ivbqcpwjdw()
                    self.xmtnegqli = True
                    if self.inhlatxex == "flash":
                        return
                if self.kouxmshyjy():
                    self.jcoktfacgw()
                    self.anibpvotxtvdating = True
                    self.inhlatxex = "win"
                    return
                if kjmljovana:
                    return
                self.anibpvotxtvdating = False
                self.inhlatxex = "none"
                if not self.auvdgqbzgb():
                    self.grayed = True
                    for k in self.koprtgesg:
                        mpalcyrcyw = k.pixels >= 0
                        k.pixels[mpalcyrcyw] = self._gray_color
                    for dmanzruhiz in self.rqdsgrklq:
                        mpalcyrcyw = dmanzruhiz.pixels >= 0
                        dmanzruhiz.pixels[mpalcyrcyw] = self._gray_color
                if self.level_index == 0 and (self.hmeulfxgy[0]._x != 3 or self.hmeulfxgy[0]._y != 58):
                    if self.liesgkdzq:
                        self.liesgkdzq._x = 500
                if self.kouxmshyjy():
                    self.anibpvotxtvdating = True
                    self.inhlatxex = "win"
                    return
                if len(self.current_level.get_sprites_by_tag("fruit")) == 0:
                    self.lose()
                if not self.step_counter_ui.kelgufkvkh():
                    self.lose()
                self.complete_action()
                return
            if self.inhlatxex == "flash":
                if self.kpphfqcyzs():
                    return
                self.inhlatxex = "none"
                self.anibpvotxtvdating = False
                self.jennkqcauc()
                if not self.step_counter_ui.myafvtlbfr():
                    self.lose()
                    self.complete_action()
                    return
                self.ikzmrdjxry()
                self.complete_action()
                return
            self.anibpvotxtvdating = False
            self.inhlatxex = "none"
            self.complete_action()
            return
        if self.action.id == GameAction.ACTION6:
            wqolgdvhew = self.action.data.get("x", 0)
            vpudlxuvpd = self.action.data.get("y", 0)
            wqolgdvhew, vpudlxuvpd = self.screen_click_to_game(wqolgdvhew, vpudlxuvpd)
            zeyfpujhqd = self.camera.display_to_grid(wqolgdvhew, vpudlxuvpd)
            if zeyfpujhqd is not None:
                grid_x, grid_y = zeyfpujhqd
                self.sebayllgbc()
                if self.level_index == 0:
                    rzwdpukxyy = self.current_level.get_sprite_at(grid_x, grid_y, "hint")
                    if rzwdpukxyy is not None:
                        rzwdpukxyy._x = 500
                self.ctohhyezgx(grid_x, grid_y)
                if self.anibpvotxtvdating:
                    return
            self.complete_action()
            return
        if self.action.id == GameAction.ACTION7:
            self.ikzmrdjxry()
            self.complete_action()
            return
        self.complete_action()

    def ctohhyezgx(self, grid_x: int, grid_y: int) -> None:
        if grid_y <= gnexwlqinp - 1 or grid_y >= ncfmodluov:
            return
        self.cmnpjwkivs()
        gvflqukvtu = self.qjlubdgly
        self.npdhdupen = set()
        self.nbnfqojis = {}
        self.zwgbpzcgq = {}
        self.mdomdbjdo = {}
        self.dwehbjuln = {}
        self.citbwsczl = {}
        self.xmcssnhit = {}
        self.zrggshnlg = set()
        self.nhxemszsx = []
        for vgtpakybrc in self.hmeulfxgy:
            if self.yrufkxnmou(grid_x, grid_y, gvflqukvtu, vgtpakybrc):
                self.nhxemszsx.append(vgtpakybrc)
        self.cqmgrggak = []
        self.itlxknnsz = {}
        self.rzfgsshuk = {}
        if self.peiiyyzum:
            for vxxrprjnav in self.peiiyyzum:
                if not self.yrufkxnmou(grid_x, grid_y, gvflqukvtu, vxxrprjnav):
                    continue
                self.cqmgrggak.append(vxxrprjnav)
                tjgvnkouuz = self.hirdajbmj.get(vxxrprjnav, gulrbtyssc)
                cptzxsjeyh = self.cqjufwqvag(tjgvnkouuz)
                if cptzxsjeyh == 1:
                    enemy_center_x, enemy_center_y = self.qmecbepbyz(vxxrprjnav)
                    vpnrfpppjj = float(grid_x) - float(enemy_center_x)
                    flhganpjxt = float(grid_y) - float(enemy_center_y)
                    frslhefhml = vpnrfpppjj * vpnrfpppjj + flhganpjxt * flhganpjxt
                    if frslhefhml > 0.0:
                        smdamdpvbs = frslhefhml**0.5
                        okhjylumdn = vpnrfpppjj / smdamdpvbs
                        sfxtvvwind = flhganpjxt / smdamdpvbs
                    else:
                        okhjylumdn = 0.0
                        sfxtvvwind = 0.0
                    self.itlxknnsz[vxxrprjnav] = (
                        okhjylumdn,
                        sfxtvvwind,
                    )
                    self.rzfgsshuk[vxxrprjnav] = (
                        float(vxxrprjnav.x),
                        float(vxxrprjnav.y),
                    )
        self.szyyvpgcv = (grid_x, grid_y)
        self.ackguicmt = 0
        self.ycxcikasw = float(self.qjlubdgly)
        if self.bjetwxoaq > 1:
            self.gzmqvzsoi = float(self.qjlubdgly) / float(self.bjetwxoaq - 1)
        else:
            self.gzmqvzsoi = float(self.qjlubdgly)
        self.xmtnegqli = False
        self.anibpvotxtvdating = True
        self.inhlatxex = "vacuum"
        self.xbmjodcifm()

    def xbmjodcifm(self) -> None:
        if self.zhlackwpo is None:
            return
        if self.zhlackwpo.pixels is None:
            return
        if self.qwmmebmhb is None or self.sqehcqhws is None:
            return
        pixels = self.zhlackwpo.pixels
        pixels[:, :] = usszepkrvw
        gvflqukvtu = self.ycxcikasw
        if gvflqukvtu <= 0.0:
            return
        umyncnjoxc, fbynmfmnkb = self.szyyvpgcv
        dx = self.qwmmebmhb - float(umyncnjoxc)
        dy = self.sqehcqhws - float(fbynmfmnkb)
        jojfalzkuj = dx * dx + dy * dy
        nsplbotuaq = gvflqukvtu + 0.5
        hlliyirjkc = max(0.0, gvflqukvtu - 0.5)
        ajgubownbe = nsplbotuaq * nsplbotuaq
        jfnwcbjlft = hlliyirjkc * hlliyirjkc
        gcpsvytzxh = (jojfalzkuj <= ajgubownbe) & (jojfalzkuj >= jfnwcbjlft)
        pixels[gcpsvytzxh] = qdlaeohrgy

    def jcoktfacgw(self) -> None:
        if self.zhlackwpo is None:
            return
        if self.zhlackwpo.pixels is None:
            return
        self.zhlackwpo.pixels[:, :] = usszepkrvw

    def ivbqcpwjdw(self) -> None:
        self.fzolkosujg()
        if self.inhlatxex == "flash":
            return
        if not self.nhxemszsx:
            return
        self.cmnpjwkivs()
        nuekotmpxi = len(self.nhxemszsx)
        if nuekotmpxi < 2:
            return
        ukrrnnuysj = [self.amnmgwpkeb.get(sprite, 0) for sprite in self.nhxemszsx]
        rrsxgaznjn = list(range(nuekotmpxi))

        def ikmvvzofng(index: int) -> int:
            while rrsxgaznjn[index] != index:
                rrsxgaznjn[index] = rrsxgaznjn[rrsxgaznjn[index]]
                index = rrsxgaznjn[index]
            return index

        def icliwhbnqx(thavbderam: int, ekdpitzaju: int) -> None:
            vdvfmqirba = ikmvvzofng(thavbderam)
            cbtkkwnuig = ikmvvzofng(ekdpitzaju)
            if vdvfmqirba == cbtkkwnuig:
                return
            rrsxgaznjn[cbtkkwnuig] = vdvfmqirba

        for i in range(nuekotmpxi):
            uypdqbxdnt = self.nhxemszsx[i]
            for ituakcbkov in range(i + 1, nuekotmpxi):
                dpcwawzfil = self.nhxemszsx[ituakcbkov]
                if not self.rukauvoumh(uypdqbxdnt, dpcwawzfil):
                    continue
                lztrvkkkzz = ukrrnnuysj[i]
                aggcncmcfq = ukrrnnuysj[ituakcbkov]
                if lztrvkkkzz != aggcncmcfq:
                    self.qcnvceoxkw([uypdqbxdnt, dpcwawzfil])
                    return
                icliwhbnqx(i, ituakcbkov)
        ixerilvqvr: dict[int, list[int]] = {}
        for index in range(nuekotmpxi):
            jtwusevbhw = ikmvvzofng(index)
            if jtwusevbhw not in ixerilvqvr:
                ixerilvqvr[jtwusevbhw] = []
            ixerilvqvr[jtwusevbhw].append(index)
        sqkcnvjkcn: list[list[Sprite]] = []
        for mncdxcqyyu in ixerilvqvr.values():
            if len(mncdxcqyyu) < 2:
                continue
            qwjzfehqwx: list[Sprite] = []
            for index in mncdxcqyyu:
                qwjzfehqwx.append(self.nhxemszsx[index])
            sqkcnvjkcn.append(qwjzfehqwx)
        if not sqkcnvjkcn:
            return
        self.eagtlyxico(sqkcnvjkcn)
        self.cmnpjwkivs()

    def fzolkosujg(self) -> None:
        ecbrzskltw = len(self.peiiyyzum)
        if ecbrzskltw < 2:
            return
        qvguicvytj: list[str] = [self.hirdajbmj.get(vxxrprjnav, gulrbtyssc) for vxxrprjnav in self.peiiyyzum]
        rrsxgaznjn = list(range(ecbrzskltw))

        def ikmvvzofng(index: int) -> int:
            while rrsxgaznjn[index] != index:
                rrsxgaznjn[index] = rrsxgaznjn[rrsxgaznjn[index]]
                index = rrsxgaznjn[index]
            return index

        def icliwhbnqx(thavbderam: int, ekdpitzaju: int) -> None:
            vdvfmqirba = ikmvvzofng(thavbderam)
            cbtkkwnuig = ikmvvzofng(ekdpitzaju)
            if vdvfmqirba == cbtkkwnuig:
                return
            rrsxgaznjn[cbtkkwnuig] = vdvfmqirba

        for i in range(ecbrzskltw):
            jsypthuvlv = self.peiiyyzum[i]
            for ituakcbkov in range(i + 1, ecbrzskltw):
                wddpazkfuz = self.peiiyyzum[ituakcbkov]
                if not self.rukauvoumh(jsypthuvlv, wddpazkfuz):
                    continue
                jdnnpadnur = qvguicvytj[i]
                ccvpmdiwjt = qvguicvytj[ituakcbkov]
                if jdnnpadnur != ccvpmdiwjt:
                    self.qcnvceoxkw([jsypthuvlv, wddpazkfuz])
                    return
                icliwhbnqx(i, ituakcbkov)
        ixerilvqvr: dict[int, list[int]] = {}
        for index in range(ecbrzskltw):
            jtwusevbhw = ikmvvzofng(index)
            if jtwusevbhw not in ixerilvqvr:
                ixerilvqvr[jtwusevbhw] = []
            ixerilvqvr[jtwusevbhw].append(index)
        sqkcnvjkcn: list[list[Sprite]] = []
        for mncdxcqyyu in ixerilvqvr.values():
            if len(mncdxcqyyu) < 2:
                continue
            qwjzfehqwx: list[Sprite] = []
            for index in mncdxcqyyu:
                qwjzfehqwx.append(self.peiiyyzum[index])
            sqkcnvjkcn.append(qwjzfehqwx)
        if not sqkcnvjkcn:
            return
        self.vwucsjocjy(sqkcnvjkcn)

    def eagtlyxico(self, sqkcnvjkcn: list[list[Sprite]]) -> None:
        nobhaarcjr = len(mgxfziiwqq) - 1

        def mdknvglful(sprite: Sprite) -> None:
            if sprite in self.npdhdupen:
                self.npdhdupen.discard(sprite)
            if sprite in self.nbnfqojis:
                del self.nbnfqojis[sprite]
            if sprite in self.zwgbpzcgq:
                del self.zwgbpzcgq[sprite]
            if sprite in self.mdomdbjdo:
                del self.mdomdbjdo[sprite]
            if sprite in self.dwehbjuln:
                del self.dwehbjuln[sprite]
            if sprite in self.xmcssnhit:
                del self.xmcssnhit[sprite]
            if sprite in self.zrggshnlg:
                self.zrggshnlg.discard(sprite)

        for qwjzfehqwx in sqkcnvjkcn:
            if not qwjzfehqwx:
                continue
            kcacvefoot = self.amnmgwpkeb.get(qwjzfehqwx[0], 0)
            if kcacvefoot >= nobhaarcjr:
                for sprite in qwjzfehqwx:
                    mdknvglful(sprite)
                    self.current_level.remove_sprite(sprite)
                    if sprite in self.hmeulfxgy:
                        self.hmeulfxgy.remove(sprite)
                    if sprite in self.amnmgwpkeb:
                        del self.amnmgwpkeb[sprite]
                    if sprite in self.nhxemszsx:
                        self.nhxemszsx.remove(sprite)
                continue
            fqjuxupjdi = kcacvefoot + 1
            bwtdecdvlt = mgxfziiwqq[fqjuxupjdi]
            eacpgvdnya = sprites[bwtdecdvlt]
            isvnxzxwks = eacpgvdnya.clone()
            self._remap_sprite(isvnxzxwks)
            rcfwyjbamh = 0
            jfwksyrdzk = 0
            for sprite in qwjzfehqwx:
                umyncnjoxc, fbynmfmnkb = self.qmecbepbyz(sprite)
                rcfwyjbamh += umyncnjoxc
                jfwksyrdzk += fbynmfmnkb
            umyncnjoxc = rcfwyjbamh // len(qwjzfehqwx)
            fbynmfmnkb = jfwksyrdzk // len(qwjzfehqwx)
            if isvnxzxwks.pixels is not None:
                sprite_height, sprite_width = isvnxzxwks.pixels.shape
            else:
                sprite_height, sprite_width = (1, 1)
            qfcgjyyogx = umyncnjoxc - sprite_width // 2
            tvfxhjorfu = fbynmfmnkb - sprite_height // 2
            if qfcgjyyogx < 0:
                qfcgjyyogx = 0
            max_x = self.oaihjsiof - sprite_width
            if qfcgjyyogx > max_x:
                qfcgjyyogx = max_x
            if tvfxhjorfu < gnexwlqinp:
                tvfxhjorfu = gnexwlqinp
            if tvfxhjorfu > ncfmodluov:
                tvfxhjorfu = ncfmodluov
            max_y = self.kamls - sprite_height
            if tvfxhjorfu > max_y:
                tvfxhjorfu = max_y
            isvnxzxwks.set_position(qfcgjyyogx, tvfxhjorfu)
            self.current_level.add_sprite(isvnxzxwks)
            self.hmeulfxgy.append(isvnxzxwks)
            self.amnmgwpkeb[isvnxzxwks] = fqjuxupjdi
            for sprite in qwjzfehqwx:
                mdknvglful(sprite)
                self.current_level.remove_sprite(sprite)
                if sprite in self.hmeulfxgy:
                    self.hmeulfxgy.remove(sprite)
                if sprite in self.amnmgwpkeb:
                    del self.amnmgwpkeb[sprite]
                if sprite in self.nhxemszsx:
                    self.nhxemszsx.remove(sprite)

    def vwucsjocjy(self, sqkcnvjkcn: list[list[Sprite]]) -> None:
        for qwjzfehqwx in sqkcnvjkcn:
            if not qwjzfehqwx:
                continue
            nvgwiuywco = qwjzfehqwx[0]
            tjgvnkouuz = self.hirdajbmj.get(nvgwiuywco, gulrbtyssc)
            zketwqvgup = self.xeufnojhrt(tjgvnkouuz)
            if zketwqvgup is None:
                for vxxrprjnav in qwjzfehqwx:
                    self.ygojzhrjdl(vxxrprjnav)
                continue
            cgmmpmplah = self.evazcmpxah(zketwqvgup)
            yagfbqunge = cgmmpmplah.clone()
            self._remap_sprite(yagfbqunge)
            rcfwyjbamh = 0
            jfwksyrdzk = 0
            for vxxrprjnav in qwjzfehqwx:
                umyncnjoxc, fbynmfmnkb = self.qmecbepbyz(vxxrprjnav)
                rcfwyjbamh += umyncnjoxc
                jfwksyrdzk += fbynmfmnkb
            umyncnjoxc = rcfwyjbamh // len(qwjzfehqwx)
            fbynmfmnkb = jfwksyrdzk // len(qwjzfehqwx)
            if yagfbqunge.pixels is not None:
                sprite_height, sprite_width = yagfbqunge.pixels.shape
            else:
                sprite_height, sprite_width = (1, 1)
            qfcgjyyogx = umyncnjoxc - sprite_width // 2
            tvfxhjorfu = fbynmfmnkb - sprite_height // 2
            if qfcgjyyogx < 0:
                qfcgjyyogx = 0
            max_x = self.oaihjsiof - sprite_width
            if qfcgjyyogx > max_x:
                qfcgjyyogx = max_x
            if tvfxhjorfu < gnexwlqinp:
                tvfxhjorfu = gnexwlqinp
            if tvfxhjorfu > ncfmodluov:
                tvfxhjorfu = ncfmodluov
            max_y = self.kamls - sprite_height
            if tvfxhjorfu > max_y:
                tvfxhjorfu = max_y
            yagfbqunge.set_position(qfcgjyyogx, tvfxhjorfu)
            self.current_level.add_sprite(yagfbqunge)
            self.peiiyyzum.append(yagfbqunge)
            self.hirdajbmj[yagfbqunge] = zketwqvgup
            if self.anibpvotxtvdating and self.inhlatxex == "vacuum":
                self.citbwsczl[yagfbqunge] = self.bjetwxoaq
            for vxxrprjnav in qwjzfehqwx:
                self.ygojzhrjdl(vxxrprjnav)

    def qcnvceoxkw(self, vuzbgsqdwt: list[Sprite]) -> None:
        self.jcoktfacgw()
        self.eauclepne = []
        self.lpbtgxyij = {}
        for sprite in vuzbgsqdwt:
            if sprite.pixels is None:
                continue
            if sprite in self.lpbtgxyij:
                continue
            self.eauclepne.append(sprite)
            self.lpbtgxyij[sprite] = sprite.pixels.copy()
        for sprite in self.eauclepne:
            pixels = sprite.pixels
            mpalcyrcyw = pixels > -1
            pixels[mpalcyrcyw] = qdlaeohrgy
        self.ilcarpaut = 0
        self.anibpvotxtvdating = True
        self.inhlatxex = "flash"

    def kpphfqcyzs(self) -> bool:
        if not self.eauclepne:
            return False
        wykwcvbijt = self.ilcarpaut // 2 % 2 == 0
        for sprite in self.eauclepne:
            if sprite.pixels is None:
                continue
            pixels = sprite.pixels
            if sprite not in self.lpbtgxyij:
                continue
            if wykwcvbijt:
                mpalcyrcyw = pixels > -1
                pixels[mpalcyrcyw] = qdlaeohrgy
            else:
                shobdbrmxz = self.lpbtgxyij[sprite]
                pixels[:, :] = shobdbrmxz
        self.ilcarpaut += 1
        if self.ilcarpaut < self.nkwbdgqdb:
            return True
        return False

    def jennkqcauc(self) -> None:
        for sprite, shobdbrmxz in self.lpbtgxyij.items():
            if sprite.pixels is None:
                continue
            sprite.pixels[:, :] = shobdbrmxz
        self.eauclepne = []
        self.lpbtgxyij = {}

    def lyaaynsyhw(self) -> bool:
        umyncnjoxc, fbynmfmnkb = self.szyyvpgcv
        zittpgpamq: list[Sprite] = []
        if self.nhxemszsx:
            zittpgpamq.extend(self.nhxemszsx)
        if self.cqmgrggak:
            zittpgpamq.extend(self.cqmgrggak)
        qfkdcxgkgj = self.xvnayzzvfx()
        for sprite in zittpgpamq:
            if sprite in self.npdhdupen:
                continue
            if sprite in self.peiiyyzum and qfkdcxgkgj:
                continue
            qhffhgeifo = sprite in self.itlxknnsz and sprite in self.rzfgsshuk
            if qhffhgeifo:
                okhjylumdn, sfxtvvwind = self.itlxknnsz[sprite]
                hwnebnrvfl, svxmgnbxtk = self.rzfgsshuk[sprite]
                cmfhziahuk = float(self.stqbquzms * 0.85)
                hwnebnrvfl += okhjylumdn * cmfhziahuk
                svxmgnbxtk += sfxtvvwind * cmfhziahuk
                qfcgjyyogx = int(round(hwnebnrvfl))
                tvfxhjorfu = int(round(svxmgnbxtk))
            else:
                sprite_center_x, sprite_center_y = self.qmecbepbyz(sprite)
                dx = umyncnjoxc - sprite_center_x
                dy = fbynmfmnkb - sprite_center_y
                tewydphtje = 0
                rzcfxufysa = 0
                if dx > 0:
                    tewydphtje = min(self.stqbquzms, dx)
                elif dx < 0:
                    tewydphtje = max(-self.stqbquzms, dx)
                if dy > 0:
                    rzcfxufysa = min(self.stqbquzms, dy)
                elif dy < 0:
                    rzcfxufysa = max(-self.stqbquzms, dy)
                qfcgjyyogx = sprite.x + tewydphtje
                tvfxhjorfu = sprite.y + rzcfxufysa
            if sprite.pixels is not None:
                sprite_height, sprite_width = sprite.pixels.shape
            else:
                sprite_height, sprite_width = (1, 1)
            if qfcgjyyogx < 0:
                qfcgjyyogx = 0
            max_x = self.oaihjsiof - sprite_width
            if qfcgjyyogx > max_x:
                qfcgjyyogx = max_x
            if tvfxhjorfu < gnexwlqinp:
                tvfxhjorfu = gnexwlqinp
            if tvfxhjorfu > ncfmodluov:
                tvfxhjorfu = ncfmodluov
            max_y = self.kamls - sprite_height
            if tvfxhjorfu > max_y:
                tvfxhjorfu = max_y
            if qhffhgeifo:
                self.rzfgsshuk[sprite] = (
                    float(qfcgjyyogx),
                    float(tvfxhjorfu),
                )
            sprite.set_position(qfcgjyyogx, tvfxhjorfu)
        self.wwvazosegn()
        if self.inhlatxex != "vacuum":
            self.jcoktfacgw()
            return True
        self.ackguicmt += 1
        self.ycxcikasw = max(0.0, self.ycxcikasw - self.gzmqvzsoi)
        self.xbmjodcifm()
        if self.ackguicmt < self.bjetwxoaq:
            return True
        if self.npdhdupen:
            self.jcoktfacgw()
            return True
        self.jcoktfacgw()
        return False

    def wwvazosegn(self) -> None:
        myvsuxlxul = self.xvnayzzvfx()
        if self.npdhdupen:
            self.thxkmeyhjg()
        if self.citbwsczl:
            akmfbjzmzl: list[Sprite] = []
            for vxxrprjnav, wenjougvog in self.citbwsczl.items():
                if wenjougvog > 0:
                    self.citbwsczl[vxxrprjnav] = wenjougvog - 1
                if self.citbwsczl[vxxrprjnav] <= 0:
                    akmfbjzmzl.append(vxxrprjnav)
            for vxxrprjnav in akmfbjzmzl:
                del self.citbwsczl[vxxrprjnav]
        if self.ackguicmt >= self.bjetwxoaq:
            return
        if not self.peiiyyzum:
            return
        if not self.hmeulfxgy:
            return
        if myvsuxlxul:
            return
        plklcqdime = set(self.cqmgrggak)
        scfdcrilps = set(self.citbwsczl.keys())
        for vxxrprjnav in self.peiiyyzum:
            if vxxrprjnav in plklcqdime:
                continue
            if vxxrprjnav in scfdcrilps:
                continue
            emjjkgbxif = False
            for vgtpakybrc in self.hmeulfxgy:
                if vgtpakybrc not in self.npdhdupen:
                    emjjkgbxif = True
                    break
            if not emjjkgbxif:
                break
            enemy_center_x, enemy_center_y = self.qmecbepbyz(vxxrprjnav)
            gnrmbrgloh: Sprite | None = None
            yzqonqsxci: int | None = None
            for vgtpakybrc in self.hmeulfxgy:
                if vgtpakybrc in self.npdhdupen:
                    continue
                fruit_center_x, fruit_center_y = self.qmecbepbyz(vgtpakybrc)
                dx = fruit_center_x - enemy_center_x
                dy = fruit_center_y - enemy_center_y
                jojfalzkuj = dx * dx + dy * dy
                if yzqonqsxci is None or jojfalzkuj < yzqonqsxci:
                    yzqonqsxci = jojfalzkuj
                    gnrmbrgloh = vgtpakybrc
            if gnrmbrgloh is None:
                continue
            target_center_x, target_center_y = self.qmecbepbyz(gnrmbrgloh)
            tjgvnkouuz = self.hirdajbmj.get(vxxrprjnav, gulrbtyssc)
            lvkjczbkpm = 2 if tjgvnkouuz == dzxwbnerru else 1
            cvdguhkpob = 0
            msscqgccsv = 0
            if target_center_x > enemy_center_x:
                cvdguhkpob = lvkjczbkpm
            elif target_center_x < enemy_center_x:
                cvdguhkpob = -lvkjczbkpm
            if target_center_y > enemy_center_y:
                msscqgccsv = lvkjczbkpm
            elif target_center_y < enemy_center_y:
                msscqgccsv = -lvkjczbkpm
            qfcgjyyogx = vxxrprjnav.x + cvdguhkpob
            tvfxhjorfu = vxxrprjnav.y + msscqgccsv
            if vxxrprjnav.pixels is not None:
                enemy_height, enemy_width = vxxrprjnav.pixels.shape
            else:
                enemy_height, enemy_width = (1, 1)
            if qfcgjyyogx < 0:
                qfcgjyyogx = 0
            max_x = self.oaihjsiof - enemy_width
            if qfcgjyyogx > max_x:
                qfcgjyyogx = max_x
            if tvfxhjorfu < gnexwlqinp:
                tvfxhjorfu = gnexwlqinp
            if tvfxhjorfu > ncfmodluov:
                tvfxhjorfu = ncfmodluov
            max_y = self.kamls - enemy_height
            if tvfxhjorfu > max_y:
                tvfxhjorfu = max_y
            vxxrprjnav.set_position(qfcgjyyogx, tvfxhjorfu)
        if not self.hmeulfxgy:
            return
        for vxxrprjnav in self.peiiyyzum:
            if vxxrprjnav in scfdcrilps:
                continue
            if vxxrprjnav in plklcqdime:
                continue
            for vgtpakybrc in self.hmeulfxgy:
                if vgtpakybrc in self.npdhdupen:
                    continue
                if not self.rukauvoumh(vxxrprjnav, vgtpakybrc):
                    continue
                self.sbfzybbszx(vxxrprjnav, vgtpakybrc)
        self.cmnpjwkivs()
        self.fzolkosujg()
        if self.inhlatxex == "flash":
            return

    def xvnayzzvfx(self) -> bool:
        if not self.npdhdupen:
            return False
        if self.djoqfdlzu <= 0:
            return False
        for vgtpakybrc in self.npdhdupen:
            if vgtpakybrc in self.zrggshnlg:
                continue
            syelanegrq = self.nbnfqojis.get(vgtpakybrc, 0)
            if syelanegrq >= self.tfaferyux:
                jzcfcenqzz = self.tfaferyux + self.djoqfdlzu
                if syelanegrq < jzcfcenqzz:
                    return True
        return False

    def sbfzybbszx(self, vxxrprjnav: Sprite, vgtpakybrc: Sprite) -> None:
        if vgtpakybrc not in self.hmeulfxgy:
            return
        if vgtpakybrc in self.npdhdupen:
            return
        amnmgwpkeb = self.amnmgwpkeb.get(vgtpakybrc, 0)
        rzdkhogqmi = amnmgwpkeb <= 0
        if not rzdkhogqmi:
            biawrretvq = amnmgwpkeb - 1
            self.xmcssnhit[vgtpakybrc] = biawrretvq
        fruit_center_x, fruit_center_y = self.qmecbepbyz(vgtpakybrc)
        enemy_center_x, enemy_center_y = self.qmecbepbyz(vxxrprjnav)
        okhjylumdn = float(fruit_center_x - enemy_center_x)
        sfxtvvwind = float(fruit_center_y - enemy_center_y)
        frslhefhml = okhjylumdn * okhjylumdn + sfxtvvwind * sfxtvvwind
        if frslhefhml > 0.0:
            smdamdpvbs = frslhefhml**0.5
            okhjylumdn /= smdamdpvbs
            sfxtvvwind /= smdamdpvbs
        else:
            okhjylumdn = 0.0
            sfxtvvwind = -1.0
        if okhjylumdn == 0.0 and sfxtvvwind == 0.0:
            sfxtvvwind = -1.0
        self.npdhdupen.add(vgtpakybrc)
        self.nbnfqojis[vgtpakybrc] = 0
        eemwekvrse = float(vgtpakybrc.x)
        bctbqavhmy = float(vgtpakybrc.y)
        self.zwgbpzcgq[vgtpakybrc] = (eemwekvrse, bctbqavhmy)
        self.mdomdbjdo[vgtpakybrc] = (okhjylumdn, sfxtvvwind)
        self.dwehbjuln[vgtpakybrc] = (eemwekvrse, bctbqavhmy)
        if rzdkhogqmi:
            self.zrggshnlg.add(vgtpakybrc)
        jzcfcenqzz = self.tfaferyux + self.djoqfdlzu + 1
        self.citbwsczl[vxxrprjnav] = jzcfcenqzz

    def thxkmeyhjg(self) -> None:
        if not self.npdhdupen:
            return
        fxhuvimusc: list[Sprite] = []
        jzcfcenqzz = self.tfaferyux + self.djoqfdlzu
        for vgtpakybrc in list(self.npdhdupen):
            if vgtpakybrc not in self.nbnfqojis:
                fxhuvimusc.append(vgtpakybrc)
                continue
            syelanegrq = self.nbnfqojis[vgtpakybrc]
            tijhbemqxz = self.zwgbpzcgq.get(vgtpakybrc)
            hfgsbvmopf = self.mdomdbjdo.get(vgtpakybrc)
            lvwhnuzrsl = self.dwehbjuln.get(vgtpakybrc)
            if tijhbemqxz is None or hfgsbvmopf is None or lvwhnuzrsl is None:
                fxhuvimusc.append(vgtpakybrc)
                continue
            kcxfklpxii = jzcfcenqzz
            if vgtpakybrc in self.zrggshnlg:
                kcxfklpxii = self.tfaferyux
            eemwekvrse, bctbqavhmy = tijhbemqxz
            okhjylumdn, sfxtvvwind = hfgsbvmopf
            hwnebnrvfl, svxmgnbxtk = lvwhnuzrsl
            if syelanegrq < self.tfaferyux:
                dzvmzsimej = syelanegrq % 4
                aieskpwdyr = 0.0
                if dzvmzsimej == 1:
                    aieskpwdyr = -1.0
                elif dzvmzsimej == 3:
                    aieskpwdyr = 1.0
                hwnebnrvfl = eemwekvrse
                svxmgnbxtk = bctbqavhmy + aieskpwdyr
                if self.djoqfdlzu == 0 and syelanegrq + 1 == self.tfaferyux and (vgtpakybrc in self.xmcssnhit):
                    biawrretvq = self.xmcssnhit[vgtpakybrc]
                    self.amnmgwpkeb[vgtpakybrc] = biawrretvq
                    bwtdecdvlt = mgxfziiwqq[biawrretvq]
                    cgmmpmplah = sprites[bwtdecdvlt]
                    if cgmmpmplah.pixels is not None:
                        vgtpakybrc.pixels = cgmmpmplah.pixels.copy()
                        _m = vgtpakybrc.pixels >= 1
                        vgtpakybrc.pixels[_m] = self._color_lut[vgtpakybrc.pixels[_m]]
                    del self.xmcssnhit[vgtpakybrc]
            else:
                if vgtpakybrc in self.xmcssnhit:
                    biawrretvq = self.xmcssnhit[vgtpakybrc]
                    self.amnmgwpkeb[vgtpakybrc] = biawrretvq
                    bwtdecdvlt = mgxfziiwqq[biawrretvq]
                    cgmmpmplah = sprites[bwtdecdvlt]
                    if cgmmpmplah.pixels is not None:
                        vgtpakybrc.pixels = cgmmpmplah.pixels.copy()
                        _m = vgtpakybrc.pixels >= 1
                        vgtpakybrc.pixels[_m] = self._color_lut[vgtpakybrc.pixels[_m]]
                    del self.xmcssnhit[vgtpakybrc]
                if self.djoqfdlzu > 0:
                    qowlxepxpm = self.gylidxxtq / float(self.djoqfdlzu)
                else:
                    qowlxepxpm = self.gylidxxtq
                hwnebnrvfl += okhjylumdn * qowlxepxpm
                svxmgnbxtk += sfxtvvwind * qowlxepxpm
                self.dwehbjuln[vgtpakybrc] = (hwnebnrvfl, svxmgnbxtk)
            qfcgjyyogx = int(round(hwnebnrvfl))
            tvfxhjorfu = int(round(svxmgnbxtk))
            if vgtpakybrc.pixels is not None:
                fruit_height, fruit_width = vgtpakybrc.pixels.shape
            else:
                fruit_height, fruit_width = (1, 1)
            if qfcgjyyogx < 0:
                qfcgjyyogx = 0
            max_x = self.oaihjsiof - fruit_width
            if qfcgjyyogx > max_x:
                qfcgjyyogx = max_x
            if tvfxhjorfu < gnexwlqinp:
                tvfxhjorfu = gnexwlqinp
            if tvfxhjorfu > ncfmodluov:
                tvfxhjorfu = ncfmodluov
            max_y = self.kamls - fruit_height
            if tvfxhjorfu > max_y:
                tvfxhjorfu = max_y
            vgtpakybrc.set_position(qfcgjyyogx, tvfxhjorfu)
            syelanegrq += 1
            self.nbnfqojis[vgtpakybrc] = syelanegrq
            if syelanegrq >= kcxfklpxii:
                fxhuvimusc.append(vgtpakybrc)
        if not fxhuvimusc:
            return
        for vgtpakybrc in fxhuvimusc:
            self.npdhdupen.discard(vgtpakybrc)
            if vgtpakybrc in self.nbnfqojis:
                del self.nbnfqojis[vgtpakybrc]
            if vgtpakybrc in self.zwgbpzcgq:
                del self.zwgbpzcgq[vgtpakybrc]
            if vgtpakybrc in self.mdomdbjdo:
                del self.mdomdbjdo[vgtpakybrc]
            if vgtpakybrc in self.dwehbjuln:
                del self.dwehbjuln[vgtpakybrc]
            if vgtpakybrc in self.xmcssnhit:
                del self.xmcssnhit[vgtpakybrc]
            if vgtpakybrc in self.zrggshnlg:
                self.zrggshnlg.discard(vgtpakybrc)
                self.current_level.remove_sprite(vgtpakybrc)
                if vgtpakybrc in self.hmeulfxgy:
                    self.hmeulfxgy.remove(vgtpakybrc)
                if vgtpakybrc in self.amnmgwpkeb:
                    del self.amnmgwpkeb[vgtpakybrc]
                if vgtpakybrc in self.nhxemszsx:
                    self.nhxemszsx.remove(vgtpakybrc)

    def bwcllmldzc(self, sprite: Sprite) -> int:
        for tag in sprite.tags:
            if tag == "fruit":
                continue
            return int(tag)
        return 0

    def cmnpjwkivs(self) -> None:
        qplkqujnlq = len(self.hmeulfxgy)
        xqefcotwls = np.zeros((qplkqujnlq, 2), dtype=np.int64)
        ukrrnnuysj = np.zeros((qplkqujnlq,), dtype=np.int64)
        for i, sprite in enumerate(self.hmeulfxgy):
            umyncnjoxc, fbynmfmnkb = self.qmecbepbyz(sprite)
            xqefcotwls[i, 0] = umyncnjoxc
            xqefcotwls[i, 1] = fbynmfmnkb
            ukrrnnuysj[i] = self.amnmgwpkeb.get(sprite, 0)
        self.oqruhyvee = xqefcotwls
        self.xmuutlucr = ukrrnnuysj

    def qmecbepbyz(self, sprite: Sprite) -> tuple[int, int]:
        if sprite.pixels is not None:
            sprite_height, sprite_width = sprite.pixels.shape
        else:
            sprite_height, sprite_width = (1, 1)
        umyncnjoxc = sprite.x + sprite_width // 2
        fbynmfmnkb = sprite.y + sprite_height // 2
        return (umyncnjoxc, fbynmfmnkb)

    def rukauvoumh(self, uypdqbxdnt: Sprite, dpcwawzfil: Sprite) -> bool:
        if uypdqbxdnt.pixels is not None:
            height_a, width_a = uypdqbxdnt.pixels.shape
        else:
            height_a, width_a = (1, 1)
        if dpcwawzfil.pixels is not None:
            height_b, width_b = dpcwawzfil.pixels.shape
        else:
            height_b, width_b = (1, 1)
        oldwyeshdz = uypdqbxdnt.x
        fuhwxypsmw = uypdqbxdnt.y
        dgcogkqmbd = uypdqbxdnt.x + width_a
        rrznqaqqkg = uypdqbxdnt.y + height_a
        zabmerqpsb = dpcwawzfil.x
        ujlimziild = dpcwawzfil.y
        camvkzoytz = dpcwawzfil.x + width_b
        feinskarfd = dpcwawzfil.y + height_b
        if dgcogkqmbd <= zabmerqpsb:
            return False
        if camvkzoytz <= oldwyeshdz:
            return False
        if rrznqaqqkg <= ujlimziild:
            return False
        if feinskarfd <= fuhwxypsmw:
            return False
        return True

    def yrufkxnmou(self, umyncnjoxc: int, fbynmfmnkb: int, gvflqukvtu: int, sprite: Sprite) -> bool:
        if sprite.pixels is not None:
            sprite_height, sprite_width = sprite.pixels.shape
        else:
            sprite_height, sprite_width = (1, 1)
        obgidvqngh = sprite.x
        xzewyycdue = sprite.y
        zwzbcbwpyh = sprite.x + sprite_width
        khfytdaqqf = sprite.y + sprite_height
        if umyncnjoxc < obgidvqngh:
            pormmcwelo = obgidvqngh
        elif umyncnjoxc > zwzbcbwpyh - 1:
            pormmcwelo = zwzbcbwpyh - 1
        else:
            pormmcwelo = umyncnjoxc
        if fbynmfmnkb < xzewyycdue:
            bcwunhexxc = xzewyycdue
        elif fbynmfmnkb > khfytdaqqf - 1:
            bcwunhexxc = khfytdaqqf - 1
        else:
            bcwunhexxc = fbynmfmnkb
        dx = umyncnjoxc - pormmcwelo
        dy = fbynmfmnkb - bcwunhexxc
        return dx * dx + dy * dy <= gvflqukvtu * gvflqukvtu

    def cqjufwqvag(self, tjgvnkouuz: str) -> int:
        if tjgvnkouuz == gulrbtyssc:
            return 1
        if tjgvnkouuz == wfcfzowdju:
            return 2
        if tjgvnkouuz == dzxwbnerru:
            return 3
        return 1

    def varjjfmuhc(self, level: int) -> str:
        if level <= 1:
            return gulrbtyssc
        if level == 2:
            return wfcfzowdju
        return dzxwbnerru

    def xeufnojhrt(self, tjgvnkouuz: str) -> str | None:
        if tjgvnkouuz == gulrbtyssc:
            return wfcfzowdju
        if tjgvnkouuz == wfcfzowdju:
            return dzxwbnerru
        return None

    def evazcmpxah(self, tjgvnkouuz: str) -> Sprite:
        if tjgvnkouuz == gulrbtyssc:
            return sprites[gxqglqjgnq]
        if tjgvnkouuz == wfcfzowdju:
            return sprites[ocvccofipr]
        if tjgvnkouuz == dzxwbnerru:
            return sprites[zvcltwpfvq]
        return sprites[gxqglqjgnq]

    def ygojzhrjdl(self, vxxrprjnav: Sprite) -> None:
        self.current_level.remove_sprite(vxxrprjnav)
        if vxxrprjnav in self.peiiyyzum:
            self.peiiyyzum.remove(vxxrprjnav)
        if vxxrprjnav in self.hirdajbmj:
            del self.hirdajbmj[vxxrprjnav]
        if vxxrprjnav in self.cqmgrggak:
            self.cqmgrggak.remove(vxxrprjnav)
        if vxxrprjnav in self.citbwsczl:
            del self.citbwsczl[vxxrprjnav]
        if vxxrprjnav in self.itlxknnsz:
            del self.itlxknnsz[vxxrprjnav]
        if vxxrprjnav in self.rzfgsshuk:
            del self.rzfgsshuk[vxxrprjnav]

    def bxzwtdtrud(self) -> bool:
        if not self.rqdsgrklq:
            return False
        if self.reqbygadvzmjired is None:
            return False
        ifeuhzhcxr = self.reqbygadvzmjired
        krdxxrzrlz = ifeuhzhcxr[0]
        gmjltxweel: list[tuple[str, int]]
        if isinstance(krdxxrzrlz, (list, tuple)):
            gmjltxweel = []
            for mvrnytgvvx, aqfhmtmxhi in ifeuhzhcxr:
                gmjltxweel.append((str(mvrnytgvvx), int(aqfhmtmxhi)))
        else:
            mvrnytgvvx = str(ifeuhzhcxr[0])
            aqfhmtmxhi = int(ifeuhzhcxr[1])
            gmjltxweel = [(mvrnytgvvx, aqfhmtmxhi)]
        nhefdqizsw: dict[int, int] = {}
        xbxnoponrg: dict[str, int] = {}
        for vgtpakybrc in self.hmeulfxgy:
            naojwrjpdq = False
            for coitlkmvpm in self.rqdsgrklq:
                if self.rukauvoumh(coitlkmvpm, vgtpakybrc):
                    naojwrjpdq = True
                    break
            if not naojwrjpdq:
                continue
            thujeoyajx = self.amnmgwpkeb.get(vgtpakybrc, 0)
            if thujeoyajx in nhefdqizsw:
                nhefdqizsw[thujeoyajx] += 1
            else:
                nhefdqizsw[thujeoyajx] = 1
        for vxxrprjnav in self.peiiyyzum:
            naojwrjpdq = False
            for coitlkmvpm in self.rqdsgrklq:
                if self.rukauvoumh(coitlkmvpm, vxxrprjnav):
                    naojwrjpdq = True
                    break
            if not naojwrjpdq:
                continue
            tjgvnkouuz = self.hirdajbmj.get(vxxrprjnav, gulrbtyssc)
            if tjgvnkouuz in xbxnoponrg and xbxnoponrg is not None:
                xbxnoponrg[tjgvnkouuz] += 1
            else:
                xbxnoponrg[tjgvnkouuz] = 1
        for xecuihaefu, aqfhmtmxhi in gmjltxweel:
            if xecuihaefu in (gulrbtyssc, wfcfzowdju, dzxwbnerru):
                if xbxnoponrg.get(xecuihaefu, 0) != aqfhmtmxhi:
                    return False
            else:
                thujeoyajx = int(xecuihaefu)
                if nhefdqizsw.get(thujeoyajx, 0) != aqfhmtmxhi:
                    return False
        return True

    def epvtlqtczz(self, x: int, y: int, sprite: Sprite) -> bool:
        if sprite.pixels is not None:
            sprite_height, sprite_width = sprite.pixels.shape
        else:
            sprite_height, sprite_width = (1, 1)
        obgidvqngh = sprite.x
        xzewyycdue = sprite.y
        zwzbcbwpyh = sprite.x + sprite_width
        khfytdaqqf = sprite.y + sprite_height
        if x < obgidvqngh:
            return False
        if x >= zwzbcbwpyh:
            return False
        if y < xzewyycdue:
            return False
        if y >= khfytdaqqf:
            return False
        return True

    def kouxmshyjy(self) -> bool:
        if not self.rqdsgrklq:
            return False
        if self.reqbygadvzmjired is None:
            return False
        ifeuhzhcxr = self.reqbygadvzmjired
        krdxxrzrlz = ifeuhzhcxr[0]
        gmjltxweel: list[tuple[str, int]]
        if isinstance(krdxxrzrlz, (list, tuple)):
            gmjltxweel = []
            for mvrnytgvvx, aqfhmtmxhi in ifeuhzhcxr:
                gmjltxweel.append((str(mvrnytgvvx), int(aqfhmtmxhi)))
        else:
            mvrnytgvvx = str(ifeuhzhcxr[0])
            aqfhmtmxhi = int(ifeuhzhcxr[1])
            gmjltxweel = [(mvrnytgvvx, aqfhmtmxhi)]
        nhefdqizsw: dict[int, int] = {}
        xbxnoponrg: dict[str, int] = {}
        for vgtpakybrc in self.hmeulfxgy:
            umyncnjoxc, fbynmfmnkb = self.qmecbepbyz(vgtpakybrc)
            naojwrjpdq = False
            for coitlkmvpm in self.rqdsgrklq:
                if self.epvtlqtczz(umyncnjoxc, fbynmfmnkb, coitlkmvpm):
                    naojwrjpdq = True
                    break
            if not naojwrjpdq:
                continue
            thujeoyajx = self.amnmgwpkeb.get(vgtpakybrc, 0)
            nhefdqizsw[thujeoyajx] = nhefdqizsw.get(thujeoyajx, 0) + 1
        for vxxrprjnav in self.peiiyyzum:
            umyncnjoxc, fbynmfmnkb = self.qmecbepbyz(vxxrprjnav)
            naojwrjpdq = False
            for coitlkmvpm in self.rqdsgrklq:
                if self.epvtlqtczz(umyncnjoxc, fbynmfmnkb, coitlkmvpm):
                    naojwrjpdq = True
                    break
            if not naojwrjpdq:
                continue
            tjgvnkouuz = self.hirdajbmj.get(vxxrprjnav, gulrbtyssc)
            xbxnoponrg[tjgvnkouuz] = xbxnoponrg.get(tjgvnkouuz, 0) + 1
        for xecuihaefu, aqfhmtmxhi in gmjltxweel:
            if xecuihaefu in (gulrbtyssc, wfcfzowdju, dzxwbnerru):
                if xbxnoponrg.get(xecuihaefu, 0) != aqfhmtmxhi:
                    return False
            else:
                thujeoyajx = int(xecuihaefu)
                if nhefdqizsw.get(thujeoyajx, 0) != aqfhmtmxhi:
                    return False
        return True

    def sebayllgbc(self) -> None:
        juvlspuhfl: list[tuple[str, int, int, int]] = []
        for vgtpakybrc in self.hmeulfxgy:
            kcacvefoot = self.amnmgwpkeb.get(vgtpakybrc, 0)
            juvlspuhfl.append(("fruit", int(kcacvefoot), int(vgtpakybrc.x), int(vgtpakybrc.y)))
        for vxxrprjnav in self.peiiyyzum:
            maybkcbmig = self.hirdajbmj.get(vxxrprjnav, gulrbtyssc)
            level = self.cqjufwqvag(maybkcbmig)
            juvlspuhfl.append(("enemy", int(level), int(vxxrprjnav.x), int(vxxrprjnav.y)))
        self.dqxbwefew.append(juvlspuhfl)

    def ikzmrdjxry(self) -> None:
        self.jcoktfacgw()
        if not self.dqxbwefew:
            return
        ivsyxrygbg = self.dqxbwefew.pop()
        for vgtpakybrc in list(self.hmeulfxgy):
            self.current_level.remove_sprite(vgtpakybrc)
        self.hmeulfxgy = []
        self.amnmgwpkeb = {}
        for vxxrprjnav in list(self.peiiyyzum):
            self.ygojzhrjdl(vxxrprjnav)
        self.peiiyyzum = []
        self.hirdajbmj = {}
        for hzlnmxhwpt, xcuvllvbjg, x, y in ivsyxrygbg:
            if hzlnmxhwpt == "fruit":
                thujeoyajx = xcuvllvbjg
                if thujeoyajx < 0:
                    thujeoyajx = 0
                if thujeoyajx >= len(mgxfziiwqq):
                    thujeoyajx = len(mgxfziiwqq) - 1
                bwtdecdvlt = mgxfziiwqq[thujeoyajx]
                cgmmpmplah = sprites[bwtdecdvlt]
                vgtpakybrc = cgmmpmplah.clone()
                self._remap_sprite(vgtpakybrc)
                vgtpakybrc.set_position(x, y)
                self.current_level.add_sprite(vgtpakybrc)
                self.hmeulfxgy.append(vgtpakybrc)
                self.amnmgwpkeb[vgtpakybrc] = thujeoyajx
            elif hzlnmxhwpt == "enemy":
                cptzxsjeyh = xcuvllvbjg
                tjgvnkouuz = self.varjjfmuhc(cptzxsjeyh)
                nvgwiuywco = self.evazcmpxah(tjgvnkouuz)
                vxxrprjnav = nvgwiuywco.clone()
                self._remap_sprite(vxxrprjnav)
                vxxrprjnav.set_position(x, y)
                self.current_level.add_sprite(vxxrprjnav)
                self.peiiyyzum.append(vxxrprjnav)
                self.hirdajbmj[vxxrprjnav] = tjgvnkouuz
        self.cmnpjwkivs()
        if self.grayed and self.auvdgqbzgb():
            self.grayed = False
            for k in self.koprtgesg:
                k.pixels = sprites[k.name].pixels.copy()
                _m = k.pixels >= 1
                k.pixels[_m] = self._color_lut[k.pixels[_m]]
            for dmanzruhiz in self.rqdsgrklq:
                dmanzruhiz.pixels = sprites[dmanzruhiz.name].pixels.copy()
                _m = dmanzruhiz.pixels >= 1
                dmanzruhiz.pixels[_m] = self._color_lut[dmanzruhiz.pixels[_m]]

    def qupyysmnmi(self, thujeoyajx: int) -> int:
        if thujeoyajx < 0:
            thujeoyajx = 0
        return 1 << thujeoyajx

    def auvdgqbzgb(self) -> bool:
        if self.reqbygadvzmjired is None:
            return False
        ifeuhzhcxr = self.reqbygadvzmjired
        krdxxrzrlz = ifeuhzhcxr[0]
        lzepagmnxr: list[tuple[int, int]] = []
        if isinstance(krdxxrzrlz, (list, tuple)):
            for mvrnytgvvx, aqfhmtmxhi in ifeuhzhcxr:
                xecuihaefu = str(mvrnytgvvx)
                if xecuihaefu in (gulrbtyssc, wfcfzowdju, dzxwbnerru):
                    continue
                lzepagmnxr.append((int(xecuihaefu), int(aqfhmtmxhi)))
        else:
            mvrnytgvvx = ifeuhzhcxr[0]
            aqfhmtmxhi = ifeuhzhcxr[1]
            xecuihaefu = str(mvrnytgvvx)
            if xecuihaefu not in (gulrbtyssc, wfcfzowdju, dzxwbnerru):
                lzepagmnxr.append((int(xecuihaefu), int(aqfhmtmxhi)))
        if not lzepagmnxr:
            return True
        yaakzbpcpp: int = 0
        for thujeoyajx, vbcanllumj in lzepagmnxr:
            eqyvskhdxh = self.qupyysmnmi(thujeoyajx)
            yaakzbpcpp += eqyvskhdxh * vbcanllumj
        tvuutcmlpj: int = 0
        for vgtpakybrc in self.hmeulfxgy:
            thujeoyajx = self.amnmgwpkeb.get(vgtpakybrc, 0)
            value = self.qupyysmnmi(thujeoyajx)
            tvuutcmlpj += value
        return tvuutcmlpj >= yaakzbpcpp
