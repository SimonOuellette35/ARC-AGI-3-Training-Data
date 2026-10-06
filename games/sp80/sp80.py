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

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import random as _random_module
from utils.arc_game import AugmentedGame
from arcengine import (
    ActionInput,
    ARCBaseGame,
    Camera,
    GameAction,
    Level,
    RenderableUserDisplay,
    Sprite,
)

sprites = {
    "adbrqflmwi": Sprite(
        pixels=[
            [8, 8, 8, 4, 8, 8, 8],
        ],
        name="adbrqflmwi",
        visible=True,
        collidable=True,
        tags=["ksmzdcblcz", "syaipsfndp"],
    ),
    "jgfvrvnkaz": Sprite(
        pixels=[
            [8, 8, 8, 8, 8],
        ],
        name="jgfvrvnkaz",
        visible=True,
        collidable=True,
        tags=["ksmzdcblcz", "sys_click"],
    ),
    "mdhkebfsmg": Sprite(
        pixels=[
            [8],
            [8],
            [8],
            [8],
        ],
        name="mdhkebfsmg",
        visible=True,
        collidable=True,
        tags=["ksmzdcblcz", "sys_click"],
    ),
    "nadtnzkesz": Sprite(
        pixels=[
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
            [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
            [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
            [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
            [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
            [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
            [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
            [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
            [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
            [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
            [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
            [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
            [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
            [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
            [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
            [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
            [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
            [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
            [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
            [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
            [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
            [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
            [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
            [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
            [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
            [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
            [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
            [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
            [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
            [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
            [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
            [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        name="nadtnzkesz",
        visible=True,
        collidable=True,
    ),
    "nkrtlkykwe": Sprite(
        pixels=[
            [6],
        ],
        name="nkrtlkykwe",
        visible=True,
        collidable=True,
        tags=["nkrtlkykwe"],
    ),
    "nvzozwqarf": Sprite(
        pixels=[
            [8, 8, 8, 8, 8, 8, 8, 8],
        ],
        name="nvzozwqarf",
        visible=True,
        collidable=True,
        tags=["ksmzdcblcz", "sys_click"],
    ),
    "odioorqnkn": Sprite(
        pixels=[
            [8, 8, 8, 8, 8, 8],
        ],
        name="odioorqnkn",
        visible=True,
        collidable=True,
        tags=["ksmzdcblcz", "sys_click"],
    ),
    "qwsmjdrvqj": Sprite(
        pixels=[
            [15, -1],
            [15, 15],
        ],
        name="qwsmjdrvqj",
        visible=True,
        collidable=True,
        tags=["hfjpeygkxy", "sys_click"],
    ),
    "syaipsfndp": Sprite(
        pixels=[
            [4],
        ],
        name="syaipsfndp",
        visible=True,
        collidable=True,
        tags=["syaipsfndp"],
    ),
    "trurgcakbj": Sprite(
        pixels=[
            [8, 8, 8, 8],
        ],
        name="trurgcakbj",
        visible=True,
        collidable=True,
        tags=["ksmzdcblcz", "sys_click"],
    ),
    "ttkatugvbk": Sprite(
        pixels=[
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
            [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
            [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
            [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
            [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
            [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
            [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
            [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
            [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
            [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
            [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
            [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
            [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
            [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
            [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
            [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
            [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
            [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
            [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
            [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        name="ttkatugvbk",
        visible=True,
        collidable=True,
    ),
    "uihgaxtzkm": Sprite(
        pixels=[
            [8, 8, 8, 8, 8, 8, 8],
        ],
        name="uihgaxtzkm",
        visible=True,
        collidable=True,
        tags=["ksmzdcblcz"],
    ),
    "untfxhpddv": Sprite(
        pixels=[
            [8, 8, 8],
        ],
        name="untfxhpddv",
        visible=True,
        collidable=True,
        tags=["ksmzdcblcz", "sys_click"],
    ),
    "uzunfxpwmd": Sprite(
        pixels=[
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        name="uzunfxpwmd",
        visible=True,
        collidable=True,
        tags=["uzunfxpwmd"],
    ),
    "uzvelihpxo": Sprite(
        pixels=[
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
            [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
            [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
            [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
            [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
            [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
            [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
            [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
            [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
            [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
            [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
            [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
            [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
            [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
            [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
            [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        name="uzvelihpxo",
        visible=True,
        collidable=True,
    ),
    "vkwijvqdla": Sprite(
        pixels=[
            [-1, 15],
            [15, 15],
        ],
        name="vkwijvqdla",
        visible=True,
        collidable=True,
        tags=["hfjpeygkxy", "sys_click"],
    ),
    "xsrqllccpx": Sprite(
        pixels=[
            [11, -1, 11],
            [11, 11, 11],
        ],
        name="xsrqllccpx",
        visible=True,
        collidable=True,
        tags=["xsrqllccpx"],
    ),
    "zgsbadjnjn": Sprite(
        pixels=[
            [8, 8, 4, 8, 8],
        ],
        name="zgsbadjnjn",
        visible=True,
        collidable=True,
        tags=["ksmzdcblcz", "syaipsfndp", "sys_click"],
    ),
}
# Data augmentation: how many rows a paddle's resting row may be nudged from the
# row it was authored on. A level overrides this with a "paddle_jitter" data key;
# PADDLE_JITTER_FULL_BAND exceeds every grid height, which lets a paddle roam its
# whole legal band (legality does the clamping). See _randomize_paddle_heights.
PADDLE_HEIGHT_JITTER = 2
PADDLE_JITTER_FULL_BAND = 20

levels = [
    # Level 1
    Level(
        sprites=[
            sprites["jgfvrvnkaz"].clone().set_position(3, 4),
            sprites["nkrtlkykwe"].clone().set_position(9, 1),
            sprites["syaipsfndp"].clone().set_position(9, 0),
            sprites["uzunfxpwmd"].clone().set_position(0, 15),
            sprites["uzvelihpxo"].clone().set_position(-1, -1),
            sprites["xsrqllccpx"].clone().set_position(4, 13),
            sprites["xsrqllccpx"].clone().set_position(10, 13),
        ],
        grid_size=(16, 16),
        data={
            "steps": 30,
            "rotation": 0,
            # The lone paddle wins from column 6 on ANY row of its legal band
            # (3-11, verified exhaustively), and reaching column 6 costs the same
            # 3 moves from every one of them -- so the row is free here and the
            # nudge costs no steps at all.
            "paddle_jitter": PADDLE_JITTER_FULL_BAND,
        },
    ),
    # Level 2
    Level(
        sprites=[
            sprites["jgfvrvnkaz"].clone().set_position(6, 6),
            sprites["nkrtlkykwe"].clone().set_position(5, 1),
            sprites["syaipsfndp"].clone().set_position(5, 0),
            sprites["untfxhpddv"].clone().set_position(6, 9),
            sprites["untfxhpddv"].clone().set_position(11, 11),
            sprites["uzunfxpwmd"].clone().set_position(0, 15),
            sprites["uzvelihpxo"].clone().set_position(-1, -1),
            sprites["xsrqllccpx"].clone().set_position(2, 13),
            sprites["xsrqllccpx"].clone().set_position(6, 13),
            sprites["xsrqllccpx"].clone().set_position(10, 13),
        ],
        grid_size=(16, 16),
        data={
            "steps": 45,
            "rotation": 180,
        },
    ),
    # Level 3
    Level(
        sprites=[
            sprites["jgfvrvnkaz"].clone().set_position(1, 8),
            sprites["nkrtlkykwe"].clone().set_position(1, 1),
            sprites["nkrtlkykwe"].clone().set_position(14, 1),
            sprites["nkrtlkykwe"].clone().set_position(6, 1),
            sprites["odioorqnkn"].clone().set_position(8, 7),
            sprites["odioorqnkn"].clone().set_position(1, 5),
            sprites["syaipsfndp"].clone().set_position(1, 0),
            sprites["syaipsfndp"].clone().set_position(14, 0),
            sprites["syaipsfndp"].clone().set_position(6, 0),
            sprites["trurgcakbj"].clone().set_position(10, 10),
            sprites["uzunfxpwmd"].clone().set_position(0, 15),
            sprites["uzvelihpxo"].clone().set_position(-1, -1),
            sprites["xsrqllccpx"].clone().set_position(1, 13),
            sprites["xsrqllccpx"].clone().set_position(12, 13),
            sprites["xsrqllccpx"].clone().set_position(7, 13),
        ],
        grid_size=(16, 16),
        data={
            "steps": 100,
            "rotation": 180,
        },
    ),
    # Level 4
    Level(
        sprites=[
            sprites["adbrqflmwi"].clone().set_position(2, 9),
            sprites["jgfvrvnkaz"].clone().set_position(12, 5),
            sprites["jgfvrvnkaz"].clone().set_position(5, 5),
            sprites["nkrtlkykwe"].clone().set_position(7, 1),
            sprites["syaipsfndp"].clone().set_position(7, 0),
            sprites["trurgcakbj"].clone().set_position(12, 13),
            sprites["trurgcakbj"].clone().set_position(14, 10),
            sprites["ttkatugvbk"].clone().set_position(-1, -1),
            sprites["uzunfxpwmd"].clone().set_position(0, 19),
            sprites["xsrqllccpx"].clone().set_position(2, 17),
            sprites["xsrqllccpx"].clone().set_position(16, 17),
            sprites["xsrqllccpx"].clone().set_position(8, 17),
            sprites["xsrqllccpx"].clone().set_position(12, 17),
        ],
        grid_size=(20, 20),
        data={
            "steps": 120,
            "rotation": 0,
        },
    ),
    # Level 5
    Level(
        sprites=[
            sprites["jgfvrvnkaz"].clone().set_position(2, 9),
            sprites["nkrtlkykwe"].clone().set_position(5, 1),
            sprites["nkrtlkykwe"].clone().set_position(13, 1),
            sprites["qwsmjdrvqj"].clone().set_position(8, 5),
            sprites["syaipsfndp"].clone().set_position(5, 0),
            sprites["syaipsfndp"].clone().set_position(13, 0),
            sprites["trurgcakbj"].clone().set_position(7, 13),
            sprites["ttkatugvbk"].clone().set_position(-1, -1),
            sprites["untfxhpddv"].clone().set_position(11, 9),
            sprites["uzunfxpwmd"].clone().set_position(0, 19),
            sprites["uzunfxpwmd"].clone().set_position(19, 0).set_rotation(90),
            sprites["uzunfxpwmd"].clone().set_position(-1, 0).set_rotation(90),
            sprites["xsrqllccpx"].clone().set_position(17, 6).set_rotation(270),
            sprites["xsrqllccpx"].clone().set_position(2, 17),
            sprites["xsrqllccpx"].clone().set_position(6, 17),
            sprites["xsrqllccpx"].clone().set_position(12, 17),
        ],
        grid_size=(20, 20),
        data={
            "steps": 100,
            "rotation": 180,
        },
    ),
    # Level 6
    Level(
        sprites=[
            sprites["mdhkebfsmg"].clone().set_position(14, 4),
            sprites["nkrtlkykwe"].clone().set_position(9, 1),
            sprites["qwsmjdrvqj"].clone().set_position(9, 5),
            sprites["syaipsfndp"].clone().set_position(9, 0),
            sprites["ttkatugvbk"].clone().set_position(-1, -1),
            sprites["uzunfxpwmd"].clone().set_position(0, 19),
            sprites["uzunfxpwmd"].clone().set_position(19, 0).set_rotation(90),
            sprites["uzunfxpwmd"].clone().set_rotation(90),
            sprites["vkwijvqdla"].clone().set_position(9, 14),
            sprites["xsrqllccpx"].clone().set_position(17, 9).set_rotation(270),
            sprites["xsrqllccpx"].clone().set_position(8, 17),
            sprites["xsrqllccpx"].clone().set_position(1, 11).set_rotation(90),
            sprites["xsrqllccpx"].clone().set_position(1, 6).set_rotation(90),
            sprites["zgsbadjnjn"].clone().set_position(7, 10),
        ],
        grid_size=(20, 20),
        data={
            "steps": 120,
            "rotation": 0,
        },
    ),
]
BACKGROUND_COLOR = 12
PADDING_COLOR = 1


class gxetqmbwgi(RenderableUserDisplay):
    """."""

    def __init__(self, pxfqncpydm: int = 0):
        self.pxfqncpydm = pxfqncpydm
        self.current_steps = pxfqncpydm
        super().__init__()

    def mmboppqpvb(self, fagavvtwpi: int) -> None:
        self.current_steps = max(0, min(fagavvtwpi, self.pxfqncpydm))

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        if self.pxfqncpydm == 0:
            return frame
        bmzfzpbuit = self.current_steps / self.pxfqncpydm
        imathbgwdx = round(64 * bmzfzpbuit)
        for x in range(64):
            if x < imathbgwdx:
                frame[0, x] = 14
            else:
                frame[0, x] = 0
        return frame


class hbwuwfezbg(RenderableUserDisplay):
    """."""

    def __init__(self, rotation: int = 0):
        self._k = rotation // 90 % 4

    def set_rotation(self, rotation: int) -> None:
        self._k = rotation // 90 % 4

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        if self._k == 0:
            return frame
        return np.rot90(frame, k=self._k).copy()


class Sp80(AugmentedGame):
    """."""

    wdxitozphu = {
        1: {
            GameAction.ACTION1: GameAction.ACTION4,
            GameAction.ACTION2: GameAction.ACTION3,
            GameAction.ACTION3: GameAction.ACTION1,
            GameAction.ACTION4: GameAction.ACTION2,
        },
        2: {
            GameAction.ACTION1: GameAction.ACTION2,
            GameAction.ACTION2: GameAction.ACTION1,
            GameAction.ACTION3: GameAction.ACTION4,
            GameAction.ACTION4: GameAction.ACTION3,
        },
        3: {
            GameAction.ACTION1: GameAction.ACTION3,
            GameAction.ACTION2: GameAction.ACTION4,
            GameAction.ACTION3: GameAction.ACTION2,
            GameAction.ACTION4: GameAction.ACTION1,
        },
    }
    qxlcnqsvsf = {
        1: {
            GameAction.ACTION1: GameAction.ACTION3,
            GameAction.ACTION2: GameAction.ACTION4,
            GameAction.ACTION3: GameAction.ACTION2,
            GameAction.ACTION4: GameAction.ACTION1,
        },
        2: {
            GameAction.ACTION1: GameAction.ACTION2,
            GameAction.ACTION2: GameAction.ACTION1,
            GameAction.ACTION3: GameAction.ACTION4,
            GameAction.ACTION4: GameAction.ACTION3,
        },
        3: {
            GameAction.ACTION1: GameAction.ACTION4,
            GameAction.ACTION2: GameAction.ACTION3,
            GameAction.ACTION3: GameAction.ACTION1,
            GameAction.ACTION4: GameAction.ACTION2,
        },
    }
    mlgebkvsmt: str
    dpkgglmdup: Optional[Sprite]
    pksbqruoge: List[Tuple[Sprite, int, int]]
    shpilcvwbs: List[Sprite]
    enlvswjeov: bool
    epilwznfbr: bool
    srwrqoodsc: Set[Sprite]
    szbmtoxgbd: Set[Sprite]
    jmbhqnxkkc: int
    jvjkymhjfc: gxetqmbwgi
    awpmaspsfp: bool

    def __init__(self, seed: Optional[int] = None) -> None:
        self._init_augmentation(seed)   # FIRST: the base draws this level's rotation
        # Set before super().__init__(): it calls set_level(0) -> on_set_level, which
        # consumes both RNGs.
        #
        # Two RNGs with different scopes. self._rng is per (seed, level): it picks the
        # paddle rows, which are level geometry. self._look_rng is per EPISODE and is
        # level-INDEPENDENT: the palette and the display rotation are drawn once from it
        # and cached in self._look, so every level of one playthrough shares a single
        # consistent look instead of re-rolling between levels.
        #
        # seed=None keeps the original behaviour: both draw from fresh entropy, so live
        # play is a new board every run. Sp80(seed=S) is a pure function of (S, level).
        self._board_seed = seed
        self._rng = self.level_rng("layout")
        self._look_rng = self._new_look_rng()
        self._look: Optional[Tuple[List[int], int]] = None
        # Per-instance template for the drops a spill spawns. It used to be the shared
        # module-level sprite, which made the liquid colour a global: two live games would
        # overwrite each other's, and whichever rolled last coloured both of their drops.
        self._drop_proto = sprites["nkrtlkykwe"].clone()
        self.mlgebkvsmt = "change"
        self.dpkgglmdup = None
        self.pksbqruoge = []
        self.shpilcvwbs = []
        self.enlvswjeov = False
        self.epilwznfbr = False
        self.srwrqoodsc = set()
        self.szbmtoxgbd = set()
        self.jmbhqnxkkc = 0
        self.jvjkymhjfc = gxetqmbwgi(pxfqncpydm=0)
        self.awpmaspsfp = False
        self.hypjnwsut = False
        self.tunzhnhfa = 0
        self.zzocrmvox = 0
        self.sywpxxgfq = 0
        self.nmcpyttlk = hbwuwfezbg(0)
        self._bg_color = BACKGROUND_COLOR
        self._liquid_color = 6
        self._paddle_color = 8
        self._basket_color = 11
        camera = Camera(
            background=BACKGROUND_COLOR,
            letter_box=PADDING_COLOR,
            interfaces=[self.jvjkymhjfc, self.nmcpyttlk],
        )
        super().__init__(
            game_id="sp80",
            levels=levels,
            camera=camera,
            available_actions=[1, 2, 3, 4, 5, 6],
        )

    def _new_look_rng(self) -> "_random_module.Random":
        """RNG for the episode's look. Level-independent, so every level of one
        playthrough draws the same palette and rotation."""
        return _random_module.Random(None if self._board_seed is None else f"sp80-look:{self._board_seed}")

    def _episode_look(self) -> Tuple[List[int], int]:
        """The episode's palette and display rotation, drawn once and then reused.

        Cached rather than re-rolled per level: a playthrough should keep one consistent
        look. full_reset() clears the cache, so a new episode draws a new one.
        """
        if self._look is None:
            pool = [2, 3, 5, 6, 7, 8, 10, 11, 12]
            chosen = self._look_rng.sample(pool, 4)
            self._look = (chosen, self._look_rng.randint(0, 3))
        return self._look

    def full_reset(self) -> None:
        # A new episode gets a new look. For a seeded game that redraws the same one.
        self._look = None
        self._look_rng = self._new_look_rng()
        super().full_reset()

    def _randomize_colors(self) -> None:
        chosen, _ = self._episode_look()
        self._bg_color = chosen[0]
        self._liquid_color = chosen[1]
        self._paddle_color = chosen[2]
        self._basket_color = chosen[3]
        self.camera.background = self._bg_color
        self._drop_proto.pixels[self._drop_proto.pixels >= 0] = self._liquid_color

    def _paddle_row_free(self, paddle: Sprite, y: int, through_paddles: bool) -> bool:
        """Whether `paddle` could stand on row y at its current column.

        Mirrors the two tests the move code in step() applies: aqltiyljgy (off
        the spouts, clear of the baskets) and a collision sweep. With
        through_paddles set, overlapping another paddle is tolerated -- step()
        force-moves through paddle-only collisions, so those never block a move.
        """
        if not self.aqltiyljgy(paddle, paddle.x, y):
            return False
        original_y = paddle.y
        paddle.set_position(paddle.x, y)
        try:
            for other in self.current_level.get_sprites():
                if other is paddle or not paddle.collides_with(other):
                    continue
                if through_paddles and ("ksmzdcblcz" in other.tags or "hfjpeygkxy" in other.tags):
                    continue
                return False
        finally:
            paddle.set_position(paddle.x, original_y)
        return True

    def _randomize_paddle_heights(self) -> None:
        """Nudge each paddle up or down a few rows, preserving solvability.

        A paddle's row is not part of the puzzle: the player retunes every paddle
        by hand before spilling, and which rows a paddle can occupy depends only
        on static geometry (the y<3 spout margin, the baskets, the frame), never
        on where the other paddles sit. So the set of layouts a player can build
        is the connected component of the paddle's legal rows containing its
        start -- move the start within that component and the reachable layouts,
        and therefore the winning ones, are exactly the same. A nudge of dy costs
        at most |dy| extra moves to undo, which every level's step budget covers
        many times over.

        The two conditions that keep this true are enforced below: each row on
        the straight path from the authored row to the new one must be a legal
        paddle position (so the nudge is a move sequence the player could have
        made, and could reverse), and the new row must be clear of every other
        sprite (so the board still reads as separate pieces).

        How far a paddle may stray is the level's "paddle_jitter" (default
        PADDLE_HEIGHT_JITTER). That bound is only about the step budget -- undoing
        a nudge costs |dy| moves -- never about solvability, so a level whose
        solution is known to be row-independent can open it right up.
        """
        jitter = self.current_level.get_data("paddle_jitter")
        if jitter is None:
            jitter = PADDLE_HEIGHT_JITTER
        for paddle in self.current_level.get_sprites_by_tag("ksmzdcblcz"):
            authored_y = paddle.y
            candidates: List[int] = []
            for y in range(authored_y - jitter, authored_y + jitter + 1):
                step = 1 if y >= authored_y else -1
                path = range(authored_y, y + step, step)
                if not all((self._paddle_row_free(paddle, row, True) for row in path)):
                    continue
                if not self._paddle_row_free(paddle, y, False):
                    continue
                candidates.append(y)
            if candidates:
                paddle.set_position(paddle.x, self._rng.choice(candidates))

    def on_set_level(self, level: Level) -> None:
        # Rebuild from the clean descriptor first, so the paddle nudge below is
        # measured from the authored rows instead of compounding across resets.
        self._levels[self._current_level_index] = self._clean_levels[self._current_level_index].clone()
        level = self.current_level
        # Re-seed per (seed, level) so a level's paddle rows depend only on which level it
        # is, never on how many levels were visited before it.
        if self._board_seed is not None:
            self._rng = _random_module.Random(f"sp80-paddles:{self._board_seed}:{self._current_level_index}")
        self.mlgebkvsmt = "change"
        self.dpkgglmdup = None
        self.pksbqruoge = []
        self.shpilcvwbs = []
        self.enlvswjeov = False
        self.epilwznfbr = False
        self.srwrqoodsc = set()
        self.szbmtoxgbd = set()
        self.awpmaspsfp = False
        self.hypjnwsut = False
        self.zzocrmvox = 0
        self._randomize_colors()
        self._randomize_paddle_heights()
        for hnakeekms in level.get_sprites_by_tag("nkrtlkykwe"):
            hnakeekms.pixels[hnakeekms.pixels >= 0] = self._liquid_color
        for hnakeekms in level.get_sprites_by_tag("ksmzdcblcz"):
            hnakeekms.pixels[(hnakeekms.pixels >= 0) & (hnakeekms.pixels != 4)] = self._paddle_color
        for hnakeekms in level.get_sprites():
            if hnakeekms.name == "xsrqllccpx":
                hnakeekms.pixels[hnakeekms.pixels >= 0] = self._basket_color
        for hnakeekms in level.get_sprites_by_tag("uzunfxpwmd"):
            hnakeekms.pixels[hnakeekms.pixels >= 0] = 1
        fagavvtwpi = level.get_data("steps") or 50
        self.tunzhnhfa = fagavvtwpi
        self.jvjkymhjfc.pxfqncpydm = fagavvtwpi
        self.jvjkymhjfc.mmboppqpvb(fagavvtwpi)
        qmctwztjyb = self.ckahxkcgfi()
        if qmctwztjyb:
            self.gchfqtwjap(qmctwztjyb)
        _, self.sywpxxgfq = self._episode_look()
        self.nmcpyttlk.set_rotation(self.sywpxxgfq * 90)

    def rxjmwfcjyw(self) -> List[Sprite]:
        return [s for s in self.current_level.get_sprites_by_tag("ksmzdcblcz")] + [s for s in self.current_level.get_sprites_by_tag("hfjpeygkxy")]

    def mdtzyuabwe(self) -> List[Sprite]:
        return [s for s in self.current_level.get_sprites_by_tag("nkrtlkykwe")]

    def mldlhgjtqi(self) -> List[Sprite]:
        return [s for s in self.current_level.get_sprites_by_tag("xsrqllccpx")]

    def cycphutjqn(self) -> List[Sprite]:
        return [s for s in self.current_level.get_sprites_by_tag("uzunfxpwmd")]

    def ckahxkcgfi(self) -> Optional[Sprite]:
        zurqbcwssv = self.rxjmwfcjyw()
        if not zurqbcwssv:
            return None
        return min(zurqbcwssv, key=lambda futcsxmviu: futcsxmviu.x**2 + futcsxmviu.y**2)

    def aqltiyljgy(self, rpilpsmmjr: Sprite, x: int, y: int) -> bool:
        """."""
        lybdljomvc = rpilpsmmjr.width
        trvtwyuduw = rpilpsmmjr.height
        if y < 3:
            return False
        for rxocuufmgq in self.mldlhgjtqi():
            rx, ry = (rxocuufmgq.x, rxocuufmgq.y)
            ajqdzrqbmm = rxocuufmgq.width
            veahmazqui = rxocuufmgq.height
            if x < rx + ajqdzrqbmm + 1 and x + lybdljomvc > rx - 1 and (y < ry + veahmazqui + 1) and (y + trvtwyuduw > ry - 1):
                return False
        return True

    def ccgagqcmlv(self, ralthebbsm: Sprite, ehobpbwgqv: Sprite) -> bool:
        ax1, ay1, ax2, ay2 = (
            ralthebbsm.x,
            ralthebbsm.y,
            ralthebbsm.x + ralthebbsm.width,
            ralthebbsm.y + ralthebbsm.height,
        )
        bx1, by1, bx2, by2 = (
            ehobpbwgqv.x,
            ehobpbwgqv.y,
            ehobpbwgqv.x + ehobpbwgqv.width,
            ehobpbwgqv.y + ehobpbwgqv.height,
        )
        return ax1 < bx2 and ax2 > bx1 and (ay1 < by2) and (ay2 > by1)

    def gchfqtwjap(self, rpilpsmmjr: Sprite) -> None:
        if self.dpkgglmdup is not None:
            self.dgtqqipvxj()
        self.dpkgglmdup = rpilpsmmjr
        if rpilpsmmjr is not None:
            rpilpsmmjr.pixels[(rpilpsmmjr.pixels >= 0) & (rpilpsmmjr.pixels != 4)] = 9
            rpilpsmmjr.set_layer(1)
        self.awpmaspsfp = True

    def dgtqqipvxj(self) -> None:
        if self.dpkgglmdup is not None:
            fxxuvnwtvi = 15 if "hfjpeygkxy" in self.dpkgglmdup.tags else self._paddle_color
            self.dpkgglmdup.pixels[(self.dpkgglmdup.pixels >= 0) & (self.dpkgglmdup.pixels != 4)] = fxxuvnwtvi
            self.dpkgglmdup.set_layer(0)
        self.dpkgglmdup = None

    def tadqvfdobr(self) -> None:
        self.mlgebkvsmt = "spill"
        self.dgtqqipvxj()
        self.pksbqruoge = [(coalstikmk, 0, 1) for coalstikmk in self.mdtzyuabwe()]
        self.shpilcvwbs = []
        self.enlvswjeov = False
        self.epilwznfbr = False
        self.srwrqoodsc = set()
        self.szbmtoxgbd = set()
        for mgqvpwjovi in self.current_level.get_sprites_by_tag("syaipsfndp"):
            px, py = (mgqvpwjovi.x, mgqvpwjovi.y)
            arr = mgqvpwjovi.pixels
            for y in range(arr.shape[0]):
                for x in range(arr.shape[1]):
                    if int(arr[y, x]) == 4:
                        dwpbeschlm, irfmxaorsf = (px + x, py + y)
                        tcrfiyjopc = self.current_level.get_sprite_at(dwpbeschlm, irfmxaorsf + 1)
                        ubhgljcvxu = tcrfiyjopc and "nkrtlkykwe" in getattr(tcrfiyjopc, "tags", [])
                        if not ubhgljcvxu and self.current_level.get_sprite_at(dwpbeschlm, irfmxaorsf + 1) is None:
                            ackqyairgy = self._drop_proto.clone().set_position(dwpbeschlm, irfmxaorsf + 1)
                            self.current_level.add_sprite(ackqyairgy)
                            self.pksbqruoge.append((ackqyairgy, 0, 1))
                            self.shpilcvwbs.append(ackqyairgy)

    def yxidiymutj(self) -> None:
        for s in self.shpilcvwbs:
            self.current_level.remove_sprite(s)
        self.shpilcvwbs = []
        for s in self.mldlhgjtqi():
            s.pixels[s.pixels >= 0] = self._basket_color
        for s in self.cycphutjqn():
            s.pixels[s.pixels >= 0] = 1
        self.srwrqoodsc = set()
        self.szbmtoxgbd = set()
        self.mlgebkvsmt = "change"
        self.enlvswjeov = False
        self.epilwznfbr = False
        self.pksbqruoge = []
        self.zzocrmvox += 1
        qmctwztjyb = self.ckahxkcgfi()
        if qmctwztjyb:
            self.gchfqtwjap(qmctwztjyb)

    def step(self) -> None:
        axxkvkila, kmqfjqint = self.lnqtlqefzv()
        if self.mlgebkvsmt == "change":
            if axxkvkila != GameAction.RESET:
                self.rpnnowtzay(1)
            if axxkvkila == GameAction.ACTION6:
                ndacyzjzp = kmqfjqint.get("x", 0)
                cuovqxuvp = kmqfjqint.get("y", 0)
                wjotjhfpqt = self.camera.display_to_grid(ndacyzjzp, cuovqxuvp)
                if wjotjhfpqt:
                    dwpbeschlm, irfmxaorsf = wjotjhfpqt
                    pmghxbqvjk: Optional[Sprite] = None
                    for futcsxmviu in self.rxjmwfcjyw():
                        if futcsxmviu.x <= dwpbeschlm < futcsxmviu.x + futcsxmviu.width and futcsxmviu.y <= irfmxaorsf < futcsxmviu.y + futcsxmviu.height:
                            pmghxbqvjk = futcsxmviu
                            break
                    if pmghxbqvjk:
                        self.gchfqtwjap(pmghxbqvjk)
                        self.hypjnwsut = True
                        self.complete_action()
                        return
            if self.dpkgglmdup is not None:
                worqgwkuyx, patlrtoom = (0, 0)
                if axxkvkila == GameAction.ACTION1:
                    patlrtoom = -1
                elif axxkvkila == GameAction.ACTION2:
                    patlrtoom = 1
                elif axxkvkila == GameAction.ACTION3:
                    worqgwkuyx = -1
                elif axxkvkila == GameAction.ACTION4:
                    worqgwkuyx = 1
                if worqgwkuyx != 0 or patlrtoom != 0:
                    pceslewgef = self.dpkgglmdup.x + worqgwkuyx
                    mcpssfghmx = self.dpkgglmdup.y + patlrtoom
                    if self.aqltiyljgy(self.dpkgglmdup, pceslewgef, mcpssfghmx):
                        collisions = self.try_move_sprite(self.dpkgglmdup, worqgwkuyx, patlrtoom)
                        if len(collisions) > 0 and all(["ksmzdcblcz" in c.tags or "hfjpeygkxy" in c.tags for c in collisions]):
                            self.dpkgglmdup.move(worqgwkuyx, patlrtoom)
                    self.hypjnwsut = False
                    self.complete_action()
                    self.awpmaspsfp = False
                    return
            if axxkvkila == GameAction.ACTION5:
                if self.zzocrmvox >= 4:
                    self.lose()
                    self.complete_action()
                    return
                self.tadqvfdobr()
                self.awpmaspsfp = False
                self.hypjnwsut = False
                return
            self.complete_action()
            return
        elif self.mlgebkvsmt == "spill":
            if self.epilwznfbr:
                dojowtxhlx = self.mldlhgjtqi()
                inszeuniyy = all((r in self.srwrqoodsc for r in dojowtxhlx))
                if self.enlvswjeov or not inszeuniyy:
                    if self.jmbhqnxkkc < 6:
                        for ymuguhctww in self.szbmtoxgbd:
                            ymuguhctww.pixels[ymuguhctww.pixels >= 0] = 14 if self.jmbhqnxkkc % 2 == 1 else 1
                        if self.jmbhqnxkkc < 5:
                            for rxocuufmgq in self.mldlhgjtqi():
                                if rxocuufmgq not in self.srwrqoodsc:
                                    rxocuufmgq.pixels[rxocuufmgq.pixels >= 0] = 0 if self.jmbhqnxkkc % 2 == 0 else self._basket_color
                        self.jmbhqnxkkc += 1
                    else:
                        self.yxidiymutj()
                        if self.tunzhnhfa <= 0:
                            self.lose()
                        self.complete_action()
                else:
                    self.complete_action()
                    self.next_level()
                return
            mapivzldnw: List[Tuple[Sprite, int, int]] = []
            for coalstikmk, worqgwkuyx, patlrtoom in self.pksbqruoge:
                mvswvwdul, ggdytkpvw = (coalstikmk.x, coalstikmk.y)
                adjx1, adjx2 = (-1, 1) if patlrtoom != 0 else (0, 0)
                adjy1, adjy2 = (-1, 1) if patlrtoom == 0 else (0, 0)
                slczixltov = [(adjx1, adjy1), (adjx2, adjy2)]
                tcrfiyjopc = self.current_level.get_sprite_at(mvswvwdul + worqgwkuyx, ggdytkpvw + patlrtoom)
                if tcrfiyjopc is None:
                    ackqyairgy = self._drop_proto.clone().set_position(mvswvwdul + worqgwkuyx, ggdytkpvw + patlrtoom)
                    self.current_level.add_sprite(ackqyairgy)
                    mapivzldnw.append((ackqyairgy, worqgwkuyx, patlrtoom))
                    self.shpilcvwbs.append(ackqyairgy)
                    continue
                if "nkrtlkykwe" in tcrfiyjopc.tags:
                    mapivzldnw.append((tcrfiyjopc, worqgwkuyx, patlrtoom))
                if "ksmzdcblcz" in tcrfiyjopc.tags:
                    for eisjmhtvre, acsawzwkti in slczixltov:
                        if self.current_level.get_sprite_at(mvswvwdul + eisjmhtvre, ggdytkpvw + acsawzwkti) is None:
                            ackqyairgy = self._drop_proto.clone().set_position(mvswvwdul + eisjmhtvre, ggdytkpvw + acsawzwkti)
                            self.current_level.add_sprite(ackqyairgy)
                            mapivzldnw.append((ackqyairgy, worqgwkuyx, patlrtoom))
                            self.shpilcvwbs.append(ackqyairgy)
                    continue
                if "xsrqllccpx" in tcrfiyjopc.tags:
                    tfilpikkqk = self.current_level.get_sprite_at(mvswvwdul + adjx1, ggdytkpvw + adjy1)
                    lkppwwuqyh = self.current_level.get_sprite_at(mvswvwdul + adjx2, ggdytkpvw + adjy2)
                    if tfilpikkqk is tcrfiyjopc and lkppwwuqyh is tcrfiyjopc:
                        tcrfiyjopc.pixels[tcrfiyjopc.pixels >= 0] = 13
                        self.srwrqoodsc.add(tcrfiyjopc)
                        continue
                    else:
                        for eisjmhtvre, acsawzwkti in slczixltov:
                            if self.current_level.get_sprite_at(mvswvwdul + eisjmhtvre, ggdytkpvw + acsawzwkti) is None:
                                ackqyairgy = self._drop_proto.clone().set_position(mvswvwdul + eisjmhtvre, ggdytkpvw + acsawzwkti)
                                self.current_level.add_sprite(ackqyairgy)
                                mapivzldnw.append((ackqyairgy, worqgwkuyx, patlrtoom))
                                self.shpilcvwbs.append(ackqyairgy)
                        continue
                if "hfjpeygkxy" in tcrfiyjopc.tags:
                    tfilpikkqk = self.current_level.get_sprite_at(mvswvwdul + adjx1, ggdytkpvw + adjy1)
                    lkppwwuqyh = self.current_level.get_sprite_at(mvswvwdul + adjx2, ggdytkpvw + adjy2)
                    if tfilpikkqk is tcrfiyjopc and lkppwwuqyh is None:
                        chvgmceunz = patlrtoom
                        zazslqutzx = -worqgwkuyx
                        ackqyairgy = self._drop_proto.clone().set_position(mvswvwdul + chvgmceunz, ggdytkpvw + zazslqutzx)
                        self.current_level.add_sprite(ackqyairgy)
                        mapivzldnw.append((ackqyairgy, chvgmceunz, zazslqutzx))
                        self.shpilcvwbs.append(ackqyairgy)
                    if lkppwwuqyh is tcrfiyjopc and tfilpikkqk is None:
                        chvgmceunz = -patlrtoom
                        zazslqutzx = worqgwkuyx
                        ackqyairgy = self._drop_proto.clone().set_position(mvswvwdul + chvgmceunz, ggdytkpvw + zazslqutzx)
                        self.current_level.add_sprite(ackqyairgy)
                        mapivzldnw.append((ackqyairgy, chvgmceunz, zazslqutzx))
                        self.shpilcvwbs.append(ackqyairgy)
                    else:
                        for eisjmhtvre, acsawzwkti in slczixltov:
                            if self.current_level.get_sprite_at(mvswvwdul + eisjmhtvre, ggdytkpvw + acsawzwkti) is None:
                                ackqyairgy = self._drop_proto.clone().set_position(mvswvwdul + eisjmhtvre, ggdytkpvw + acsawzwkti)
                                self.current_level.add_sprite(ackqyairgy)
                                mapivzldnw.append((ackqyairgy, worqgwkuyx, patlrtoom))
                                self.shpilcvwbs.append(ackqyairgy)
                        continue
                if "uzunfxpwmd" in tcrfiyjopc.tags:
                    tcrfiyjopc.pixels[tcrfiyjopc.pixels >= 0] = 14
                    self.szbmtoxgbd.add(tcrfiyjopc)
                    self.enlvswjeov = True
                    continue
            self.pksbqruoge = mapivzldnw
            if not self.pksbqruoge:
                self.epilwznfbr = True
                self.jmbhqnxkkc = 0
            return

    def udeubouzyp(self, olvhpcnbsh: int, mviqpduaav: int) -> Tuple[int, int]:
        """."""
        k = self.sywpxxgfq
        if k == 0:
            return (olvhpcnbsh, mviqpduaav)
        if k == 1:
            return (63 - mviqpduaav, olvhpcnbsh)
        if k == 2:
            return (63 - olvhpcnbsh, 63 - mviqpduaav)
        return (mviqpduaav, 63 - olvhpcnbsh)

    def fewyrfijcb(self, dwpbeschlm: int, irfmxaorsf: int) -> Tuple[int, int]:
        """."""
        k = self.sywpxxgfq
        if k == 0:
            return (dwpbeschlm, irfmxaorsf)
        if k == 1:
            return (irfmxaorsf, 63 - dwpbeschlm)
        if k == 2:
            return (63 - dwpbeschlm, 63 - irfmxaorsf)
        return (63 - irfmxaorsf, dwpbeschlm)

    def lnqtlqefzv(self) -> Tuple[GameAction, Dict[str, Any]]:
        """."""
        k = self.sywpxxgfq
        ehmbtpreks = self.action.id
        data = self.action.data
        if k == 0:
            return (ehmbtpreks, data)
        if ehmbtpreks in self.wdxitozphu.get(k, {}):
            return (self.wdxitozphu[k][ehmbtpreks], data)
        if ehmbtpreks == GameAction.ACTION6:
            olvhpcnbsh = data.get("x", 0)
            mviqpduaav = data.get("y", 0)
            dwpbeschlm, irfmxaorsf = self.udeubouzyp(olvhpcnbsh, mviqpduaav)
            return (ehmbtpreks, {"x": dwpbeschlm, "y": irfmxaorsf})
        return (ehmbtpreks, data)

    def rpnnowtzay(self, znmndtybio: int) -> None:
        """."""
        self.tunzhnhfa = max(0, self.tunzhnhfa - znmndtybio)
        self.jvjkymhjfc.mmboppqpvb(self.tunzhnhfa)
        if self.tunzhnhfa <= 0:
            self.lose()
            self.complete_action()
            return
