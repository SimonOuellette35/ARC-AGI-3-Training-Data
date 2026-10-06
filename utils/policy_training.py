#!/usr/bin/env python
"""
Shared policy/dynamics training pipeline
=======================================

Both train_policy_dynamics.py and train_policy_dynamics_V2.py use this module.
The data, losses and training behavior below are shared; the packed model layout
sections describe V1. See POLICY_DYNAMICS_V2.md for V2's spatial/temporal network.

Train an *in-context* transformer policy on the multi-level demonstration data
under ``data/training_multi_level/`` -- jointly with a shared-trunk **dynamics
head** that predicts the next settled board from the current state plus the
action taken. Both heads train by DEFAULT; ``--no-dynamics`` trains the policy
alone. See "Auxiliary dynamics head" below.

The network itself lives in ``policy_model.py`` (``InContextPolicy`` +
``ModelConfig`` + the board/action constants), which the inference hosts
(``policy_runtime.py`` -> ``solver.py`` / ``kaggle_agent.py``) import directly so
they never pull in this training script. This file is only the pipeline:
preprocessing, the dataset, the losses and the training loop. A checkpoint
trained with ``--no-dynamics`` is a plain policy checkpoint and loads in
``policy_runtime.py`` unchanged.

Data layout
-----------
``data/training_multi_level/<game>/episode_*.{json,ep.zst}`` where each episode
is::

    {
      "game_id": str,
      "levels": [
        {
          "level_id": int,
          "observations": [F][64][64] ints in [0, 15],   # FLAT frame stream
          "actions":      [T]{...},                      # T <= F; see below
        },
        ...
      ]
    }

Action records (the format this trainer consumes)
-------------------------------------------------
Each entry of ``actions`` is::

    {"type": "simple"|"mouse", "index": 0..6, "data": {"x": .., "y": ..}?,
     "phase":   "reset" | "expert" | "explore" | "burst",
     "n_obs":   int,     # how many frames of `observations` this action produced
     "changed": bool,    # null-transition flag; unused by the policy
     "optimal": [{"type", "index", "data"?}, ...] | None}

Three properties of that record drive everything below.

**(a) taken vs optimal.** The top-level ``type``/``index``/``data`` is the action
the recorder actually TOOK; ``optimal`` is the SET of equally-optimal actions at
the state that action was decided from. They differ on exploration/burst steps,
where the taken action is a deliberate random detour and ``optimal`` is the
expert's relabelled answer at that same (off-policy) state. **The policy is
trained only against** ``optimal``: the taken action is fed back in as context, so
the model sees what actually happened, but it is never a target. Steps with
``optimal == None`` (the leading RESET, plus steps a generator could not label)
carry no target -- they stay in context and are masked out of the loss. Because
``optimal`` is a set, the type head is trained against a UNIFORM distribution over
it rather than against one arbitrary tie-break, and a prediction counts as correct
when it lands anywhere in the set (same for the pointer head over the set's click
locations).

**(b) exploratory + RESET transitions in context.** A trajectory is no longer a
pure optimal replay: it can open with an exploration prefix, contain perturbation
bursts, and contain RESET actions (``index == 0``) where the recorder restored the
level's initial state after stranding itself. All of them stay in the context
stream verbatim -- an off-policy detour followed by an expert-labelled recovery is
exactly the behaviour an in-context policy has to learn. RESET is part of the
discrete vocabulary, and a ``phase == "reset"`` step carries ``optimal ==
[RESET]``, so "when stuck, reset" is itself a trained target. Which phases are
*supervised* is configurable (``--skip-target-phases``); by default every step
with an ``optimal`` set is, off-policy ones included, since a relabelled expert
target at a state the policy would never reach on-path is precisely DAgger data.

**(c) multi-frame observations.** ``observations`` is a FLAT stream of frames and
each action owns a contiguous SPAN of it, ``n_obs`` frames long -- the whole
animation the engine rendered for that one submitted action (``FrameData.frame``
is a list). Via ``common_utils.action_spans``::

    start(i) = sum(a["n_obs"] for a in actions[:i])
    actions[i]  produced  observations[start(i) : start(i) + n_obs(i)]

``actions[0]`` is the leading RESET producing ``observations[0]``. The state that
``actions[i+1]`` was decided from is the LAST (settled) frame of span ``i``; the
earlier frames of that span are the in-flight animation, which for an animating
game is where the mechanic is actually visible (tn36's failed program run animates
across 8 frames and then snaps back, so the settled frame alone hides the whole
experiment). Every frame of a span is tokenised, in order, so the transformer sees
the animation and not just its result. Spans longer than ``--max-frames`` are
subsampled with both endpoints kept, so the settled frame always survives. When
every ``n_obs == 1`` this collapses exactly to the old 1:1 layout.

Action space (simple + mouse-click)
-----------------------------------
Both action kinds carry the engine's ``index`` (== ``GameAction`` enum value)::

    {"type": "simple", "index": 0..5}                            # RESET + ACTION1..5
    {"type": "mouse",  "index": 6, "data": {"x": .., "y": ..}}   # ACTION6 click

So the discrete action vocabulary is ``0..6`` (``CLICK_INDEX == 6``), and a click
additionally carries a pixel coordinate ``(x, y)`` in ``[0, 63]``. The policy
therefore has TWO output heads at each state position:

  * a **type head** over the 7 discrete actions (``NUM_ACTION_TYPES``), and
  * a **spatial pointer head** -- a ``POINTER_GRID x POINTER_GRID`` distribution
    over click locations, supervised ONLY on steps whose OPTIMAL set contains a
    click (target = uniform over that set's click cells).

The taken click coordinate is also fed BACK IN as part of the action token (so a
later state's prediction can condition on where the previous click landed -- e.g.
r11l's "place conditions on pick"). Non-click actions carry a learned no-click
vector instead.

Sequence layout
---------------
One training item is a sequence of *steps*. Step ``t`` bundles the frame span of
action ``t`` with the action that follows it::

    frames(span_0) -> a_1 -> frames(span_1) -> a_2 -> ... -> frames(span_{S-1}) -> a_S

The readout for step ``t`` sits after span ``t``'s frames and before the action
token for ``a_{t+1}``, so under causal masking it sees this state's whole
animation plus the entire preceding trajectory -- but not its own action or the
future -- and predicts ``optimal(a_{t+1})``. The trailing frame span (which has no
following action) is dropped, exactly as the old code dropped the last state; the
leading RESET is likewise never a target. The transformer meta-learns, in-context,
to infer the policy of *this* game/level from the context rather than memorising a
single game.

Spatial patch tokens (Option C)
-------------------------------
A 64x64 board with 16 colors is large; naively one-hot/embedding every cell for
every state blows up memory. Rather than compress a board to ONE token (which
throws position away -- see the frog_crossing gap ~= 0 that motivated this), a
small strided CNN turns each frame into a 4x4 conv feature map emitted as **16
separate patch tokens**, each tagged with a learned 2D position. The *main*
transformer attends over those patches across time, so absolute position is
preserved end-to-end and can interact across timesteps. A step therefore costs
``n_frames * 16 + 2`` tokens (its animation's patches, then a READOUT, then the
ACTION token) -- 18 for the common single-frame step. Because ``n_frames`` varies
per step, the token stream is PACKED (variable length, no per-step padding) and
positions are factorised into ``step x frame-slot x patch x token-type``
embeddings instead of one absolute table, so a 12-frame span costs 12 frames of
tokens and its neighbours still cost one.

FIFO context cap
----------------
To keep sequence length bounded the dataset caps each trajectory to the most
recent ``--max-states`` steps as a FIFO -- the OLDEST steps are evicted first (we
keep the *suffix*, not the prefix) -- and caps each step's animation to
``--max-frames`` frames. Sequence length is then <= (16*max_frames + 2) *
max_states tokens and typically much less; if you OOM, lower ``--batch-size``,
``--max-states`` or ``--max-frames``.

Suffix augmentation (training)
------------------------------
On the TRAIN split each fetch trains on a fresh *random suffix* ``s_j..s_{L-1}``
of the level's trajectory rather than the whole thing. Any suffix is itself a
correct, solvable trajectory from a different start state -- the remainder of the
recording is the proof -- so this widens the policy's start-state distribution at
zero data cost (the one universally-safe form of "exploring starts"). The start
``j`` is uniform in ``[0, L-1]`` and redrawn every epoch, so the same level
teaches many distinct start states. Validation is left as the full deterministic
trajectory so val loss stays comparable. Disable with ``--no-suffix-aug``.

Multi-level context (training)
------------------------------
The in-context policy is meant to carry what it learned on one level into the
next instead of re-deriving the game's rules each level, so training must show it
multi-level contexts -- not just isolated levels. On the TRAIN split each item is
a random SUBSET of the episode's levels, concatenated in random ORDER into one
continuous context (e.g. ``L5 -> L3 -> L7``); ``k=1`` reproduces the isolated
single-level item, which is itself a useful augmentation. Levels are drawn from
the SAME episode (the only place cross-level transfer is meaningful) and the
index level is always included as an anchor, so every level is still covered once
per epoch. Order is randomised because the transfer skill should not depend on
the canonical level order. The concatenation is FIFO-capped to ``--max-states``
steps total, and at most ``--max-levels`` levels are joined. Disable with
``--no-multi-level``.

Action-identity shuffle (training)
-----------------------------------
Every game's corpus is recorded under one fixed action mapping (ACTION1 always
means whatever it means in that game's engine), so a large enough model can
just memorize id->effect per game instead of reading the transition history to
figure out what an action does. On the TRAIN split each item draws a fresh
random permutation of the 5 non-RESET, non-click ids (1..5) and relabels that
item's ``act``/``opt`` through it consistently across the WHOLE item (every
concatenated level, if ``multi_level`` is on) -- one permutation per fetch, not
per step, since a mapping that changed mid-context would carry no learnable
signal at all. RESET (0) and the mouse click (``CLICK_INDEX``) are left fixed:
RESET's meaning does not vary by game, and a click's "effect" is the (x, y) it
carries, not a discrete id, so relabelling it would not do anything. This is
the same trick as the rotation augmentation's per-(seed, level) action remap
(see ``games/`` ``AugmentedGame``), generalised from the 4-element rotation
group to a full random permutation, and applied at fetch time instead of
inside the engine since it never needs to touch the observed frames. Disable
with ``--no-action-shuffle``.

Train / val split -- the (game, level) holdout (default)
-------------------------------------------------------
Val must measure the thing deployment needs: solving a level of a KNOWN game
that was never trained on -- what the engine hits every time it auto-advances.
So the default split holds out, per game, a random ``--level-val-frac`` (0.2) of
that game's distinct ``level_id``s and sends EVERY seed of those ``(game,
level)`` puzzles to val; the trainer never sees a held level in any form, not
even as an unsupervised multi-level sibling (``file_levels`` is held-free on
train). Different seeds of one level are near-identical on most games, so a
seed-level split would leak; holding out the whole puzzle does not. Whole games
are never held out here -- that is the separate OOD test set.

  * Games with ``< --level-val-min-levels`` (4) distinct levels keep every level
    in train (holding one out would gut their coverage) and contribute no val.
  * ``--level-val-seeds`` (40) caps how many seeds of each held level enter val,
    since they are near-duplicates; 0 keeps all ~1000.
  * The manifest is persisted to ``<cache-dir>/level_holdout.json`` so the split
    stays STABLE as seeds/games are added later (a reshuffle would move a
    trained level into val). Delete it to rebuild after changing the flags.
  * ``val`` = each held level ALONE from step 0. ``val-ml`` (the
    ``--select-on multi`` default, i.e. what picks ``best.pt``) =
    DEPLOYMENT-FAITHFUL: the episode's earlier non-held levels as in-context
    history, then the held level last as the only supervised segment.
  * ``--no-level-holdout`` restores the legacy split (a random ``--val-frac`` of
    episode FILES; leaks same-puzzle different-seed episodes across the split).

Error-driven resampling (poor man's DAgger, training)
-----------------------------------------------------
On by default (opt out with ``--no-dagger``). After a warmup of
``--resample-warmup`` plain epochs, and then every ``--resample-every`` epochs,
the current policy is run
**teacher-forced** over the whole TRAIN corpus (one no-grad forward pass, no GPU
in any data-generation loop) and every step where its prediction *disagrees with
the oracle* is recorded. The oracle is now the step's stored ``optimal`` SET, so
the test is the exact one -- "the policy's action is not among the optimal ones"
-- rather than the old conservative "differs from the one demonstrated action"
proxy. For click steps the pointer bin must also land on one of the optimal
click cells. Unsupervised steps (no ``optimal``) never count as errors.

Those error steps (dilated by a ``--resample-neighbourhood`` radius so the
*neighbourhood* of each mistake is included) then drive the NEXT rounds'
sampling two ways, both bounded and both keeping a base weight so clean on-path
data is never dropped -- DAgger's ``D <- D u D_new`` aggregation, not
replacement:

  * **Level level** -- a ``WeightedRandomSampler`` favours levels with a higher
    error fraction (weight ``1 + alpha * mean(dilated_errors)`` in ``[1, 1+a]``).
  * **State level** -- inside a level, the random suffix start is biased toward
    error steps (start ``j`` drawn ~ ``1 + alpha * dilated_errors[j]``), so a
    mistake and its neighbourhood land at the FRONT of many training contexts
    and are trained on far more often, while every state keeps a base share.

This buys DAgger's targeting -- effort concentrated on the errors the policy
*actually* makes -- as a pure sampler change against the existing corpus; no
regeneration, no policy inference in the generator. On by default; disable with
``--no-dagger``.

Dynamics head / world model (on by default; ``--no-dynamics`` to disable)
--------------------------------------------------------------------------
A second readout on the SAME trunk that models the environment instead of the
policy: given the settled board of step ``t`` and the action ``a_{t+1}`` taken
from it, predict the settled board of step ``t+1``. This is meant as a WORLD
MODEL -- the intended consumer is future planning/lookahead built on top of the
policy, not merely a training-time regularizer -- though no inference host
wires it up for rollout yet (``solver.py`` / ``policy_runtime.py`` /
``kaggle_agent.py`` never pass ``return_dynamics=True``). It trains jointly
with the policy by default; its gradient is scaled by ``--dynamics-weight``
(default 1.0). As a side effect it also forces the trunk to encode game
mechanics, which helps the policy head too -- but that is a bonus, not the
reason the head exists. ``--no-dynamics`` drops it and its extra token
entirely, giving a plain policy checkpoint.

Warm-starting from a policy-only checkpoint. ``--resume <policy.pt> --dynamics``
(the default) copies the trunk and the policy head across and starts ONLY the
dynamics head (``dyn_readout`` / ``dyn_norm`` / ``board_decoder`` + the 4th
``type_embed`` row) from init, with a fresh optimizer / schedule / epoch clock --
see ``warm_start_state_dict``.

Layout. Each step gains ONE token, a ``DYN_READOUT`` placed AFTER the action
token (so under causal masking it sees the state AND the action, unlike the
policy readout which sits before the action and must not). Its transformer output
is combined with the settled frame's 16 patch-token outputs -- the readout
carries "what the action does", the patch tokens carry "where everything is now"
-- and a transposed-conv decoder (a mirror of ``PatchEncoder``) upsamples that
``grid x grid`` seed back to a full ``64 x 64`` board. Two per-cell heads:

  * a **colour head**, 16-way per cell, trained with cross-entropy against the
    next board; and
  * a **change head**, one logit per cell, trained (BCE) to predict which cells
    differ next. The change SHAPE is highly predictable even in partially
    observable games where the newly-revealed *contents* are not (the view
    window moves rigidly with the agent), so this is the planner-useful "does
    this action reveal / disturb anything, and where" signal.

Partial observability. The colour loss up-weights cells that actually change
(``--dynamics-changed-weight``, default 5.0) since most of the board is static.
For a fog-of-war game pass ``--dynamics-fog-color C`` (the sentinel colour an
unexplored cell renders as, e.g. 5 for dark_maze_3): cells that are fog in BOTH
the input and the target carry zero colour loss (nothing to learn), and cells
revealed for the first time (fog -> not fog) are down-weighted to
``--dynamics-fog-reveal-weight`` (default 0.25) because their contents are only
partly knowable from context. With no sentinel the head simply pays a bounded,
honest loss on unpredictable reveals and learns to hedge there.

Alignment. Each action's recorded settled outcome is a dynamics target,
including the terminal action. The terminal observation is a target only, never
an extra policy input. ``dyn_continue`` marks whether an imagined continuation
can advance to the next step of the same level; a terminal action may end a
rollout but cannot cross into another level. Existing caches already contain
all of these frames, so no re-preprocess is needed.

Metrics (reported as ``dyn`` on every epoch line, over supervised steps only):
``cell`` per-cell colour accuracy, ``chg`` accuracy restricted to cells that
changed, ``exact`` fraction of steps whose whole board (minus both-fog cells) is
predicted correctly, and ``iou`` the change head's mask IoU. ``best_policy.pt``
uses policy loss alone; ``best_dynamics.pt`` uses one-step dynamics loss without
the outer dynamics weight. ``best.pt`` aliases the best policy. Checkpoints
record the actual validation membership/manifest and evaluation settings;
resuming with a changed or missing identity resets both selection thresholds.

Validation also reports deterministic/uncertain/unknown game groups, moving and
no-op fidelity, and unweighted color NLL, Brier score, entropy, and calibration
error. ``--eval-game-groups`` accepts explicit game classifications; conservative
built-in rules leave other games unknown. Fixed rollout depths default to
``--eval-rollout-depths 1,2,4``, independent of the training curriculum, on up to
``--eval-rollout-max-batches 32`` evenly spaced batches per validation view
(0 = all). All one-step validation batches receive grouped diagnostics.

Rollout curriculum (``--dynamics-unroll-max-k``, default 20; 1 = off). The head
above trains PURELY single-step: it always conditions on the true board at
``t``, never on its own prior prediction. That is fine if the head only ever
sees real frames, but not for a world model meant to be unrolled for planning
-- one that has never seen its own mistakes as input will drift once errors
compound across several imagined steps. This curriculum trains it for that:

  * every batch still pays the ordinary single-step term above;
  * on 1 of every ``--dynamics-unroll-every`` batches (default 4), pick one
    random ``depth``-long window per item, replay ``depth - 1`` hops of the
    dynamics head under ``torch.no_grad()``, feeding its own (change-gated in V2,
    detached) board prediction back in as the NEXT hop's sole input frame.
    The observed prefix retains its real animation spans; imagined steps are
    repacked as one frame each, with no real future animation pixels or lengths.
    Then run one more forward WITH gradients. Score the final dynamics hop
    against the real next-board target. From that SAME causal forward, also
    train the policy on every imagined input step against its recorded
    optimal-action set (including click coordinates). The real prefix/root
    and steps without optimal labels are excluded from this policy term.
    Earlier predicted boards remain detached; no gradients propagate through
    their generation (see ``policy_dynamics_rollout_term``);
  * ``depth`` ramps from 1 up to ``--dynamics-unroll-max-k`` by
    ``+1`` every ``--dynamics-unroll-ramp-epochs`` epochs (default 20 and 1 ->
    full depth at epoch 20);
  * ``--dynamics-unroll-weight`` scales the final dynamics term independently
    of ``--dynamics-weight``; ``--policy-unroll-weight`` (default 1) scales the
    imagined-step action loss, which also respects ``--pointer-weight``.
    Set ``--policy-unroll-weight 0`` for dynamics-only rollout training.

Reported as ``rollout(k=N)`` on the epoch line once ``depth > 1``.
``--dynamics-unroll-max-k 1`` reproduces the exact single-step behaviour above
bit-for-bit -- this is a strict addition, never a replacement, so it is safe
to turn off entirely if it is not paying for itself.

Usage
-----
    # 1. build the compact cache (frames -> uint8 + per-action arrays). Resumable.
    #    NOTE: the cache schema is versioned; a cache built by an older revision
    #    is refused, so re-run this with --overwrite after upgrading.
    python train_policy_dynamics.py preprocess --overwrite

    # 2. train -- policy + dynamics heads jointly (the default)
    python train_policy_dynamics.py train --epochs 10 --batch-size 16
    #    ... for a fog-of-war game, name the unexplored-cell colour:
    python train_policy_dynamics.py train --dynamics-fog-color 5
    #    ... policy head only:
    python train_policy_dynamics.py train --no-dynamics

    # 2b. add a dynamics head to an existing policy-only checkpoint
    python train_policy_dynamics.py train --resume runs/policy_C/best.pt

    # or preprocess + train in one go
    python train_policy_dynamics.py all --epochs 10

    # both commands use every episode on disk by default; pass a positive
    # --episodes-per-game only if you deliberately want to cap the corpus.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import zlib
from dataclasses import fields
from glob import glob

import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import common_utils
# The network itself and the board / action constants live in policy_model so the
# inference hosts (policy_runtime -> solver.py / kaggle_agent.py) never import
# this training script. Everything here is re-exported, so importers that used to
# read these names off train_policy keep working after the rename.
from policy_model import (  # noqa: F401  (re-exported for downstream importers)
    BOARD_H, BOARD_W, NUM_COLORS, CLICK_INDEX, NUM_ACTION_TYPES, NUM_ACTIONS,
    POINTER_GRID, PATCH_GRID, PATCHES_PER_FRAME, NO_COORD, NO_ACTION,
    coord_to_bin, coord_to_bin_flat, bin_flat_to_coord, _span_keep,
    PatchEncoder, ActionIO, BoardDecoder, ModelConfig, InContextPolicy,
)
from utils.policy_checkpoint import checkpoint_architecture, policy_classes
from utils.policy_rollout import dynamics_rollout_predictions
from utils.dynamics_decoding import (
    change_threshold, configured_change_threshold, decode_dynamics_board,
)
from utils.policy_evaluation import (
    DynamicsEvaluation, load_game_groups, rollout_batch_indices, format_evaluation,
)
from utils.policy_training_checkpoint import (
    validation_context, restore_selection, update_selection, save_training_checkpoints,
    optimizer_state_with_spatial_pointer,
)
from utils.training_resume import keep_training_config

# ----------------------------------------------------------------------------
# Constants describing the CACHE / dataset (the board + action constants and the
# coord<->bin helpers are imported from policy_model above)
# ----------------------------------------------------------------------------
# --- Optimal-action sets ---------------------------------------------------
# Targets are the step's `optimal` SET, stored densely as (T, MAX_OPTIMAL) with
# NO_ACTION padding. 16 is far above anything the corpus emits today (the widest
# observed tie set is 8) and costs ~48 bytes/step next to 4 KB/frame, so the cap
# never binds in practice; a step that somehow exceeds it keeps its first 16
# optimal actions and preprocess reports the truncation.
MAX_OPTIMAL = 16

# --- Recording phases ------------------------------------------------------
# Every action record carries a `phase` (see BaseSolver._encode_step). It is kept
# in the cache so training can (optionally) restrict which steps are SUPERVISED
# without ever dropping them from the context. Unknown/absent -> "other".
PHASES = ("reset", "expert", "explore", "burst", "other")
PHASE_ID = {p: i for i, p in enumerate(PHASES)}
OTHER_PHASE = PHASE_ID["other"]
EXPERT_PHASE = PHASE_ID["expert"]

# Cache layout version. Bumped when the .npz schema changes so a stale cache is
# refused loudly instead of being read as garbage. v1 was the single-frame,
# taken-action-as-target layout; v2 adds frame spans, optimal sets and phases;
# v3 adds `lid_<i>` -- the source `level_id` of each cached level, so the
# train/val split can hold out whole (game, level) puzzles (see the module
# docstring's "Train / val split").
CACHE_VERSION = 3

DATA_DIR = "data/training_multi_level"
CACHE_DIR = "data/cache_policy"   # compact per-episode .npz cache lives here
# Persisted level-holdout manifest (see build_level_holdout). Lives inside the
# cache dir so it travels with the cache it describes.
LEVEL_HOLDOUT_FILE = "level_holdout.json"


# ----------------------------------------------------------------------------
# Stage 1: preprocessing -- JSON -> compact .npz cache
# ----------------------------------------------------------------------------
def _coord_of(a: dict) -> tuple[int, int] | None:
    """The ``(x, y)`` pixel coordinate carried by one action record, or ``None``.

    Accepts either discriminator the corpus uses -- ``type == "mouse"`` or
    ``index == CLICK_INDEX`` -- since ``optimal`` entries and the flat taken
    action are written by the same encoder but read back from two schemas.
    """
    data = a.get("data") or {}
    if "x" not in data or "y" not in data:
        return None
    if a.get("type") != "mouse" and int(a.get("index", -1)) != CLICK_INDEX:
        return None
    return int(data["x"]), int(data["y"])


def _encode_level(lv: dict, src_idx: int = -1) -> dict[str, np.ndarray] | None:
    """One level's JSON record -> the cache's per-level arrays, or ``None`` if
    the level is malformed (callers skip it).

    Returns arrays over the level's ``T`` actions, all index-aligned to
    ``actions``:

      ``obs``  (F,64,64) uint8   the FLAT frame stream
      ``nobs`` (T,)      uint16  frames produced by each action (its span)
      ``act``  (T,)      uint8   the TAKEN action index (context, never a target)
      ``axy``  (T,2)     uint8   taken click coord, ``NO_COORD`` if not a click
      ``opt``  (T,K)     int16   the OPTIMAL action set, ``NO_ACTION``-padded
      ``oxy``  (T,K,2)   uint8   each optimal action's click coord
      ``pha``  (T,)      uint8   phase code (see ``PHASES``)

    plus a scalar ``lid`` -- the source ``level_id`` (``src_idx`` when the record
    omits it), which keys the (game, level) holdout split. It is NOT positional:
    a few generators drop an unsolved level for some seeds, so position ``i`` is
    not a stable puzzle identity across seeds.

    A step with ``optimal == None`` gets an all-``NO_ACTION`` row, which is how
    "in context but not supervised" is represented downstream.
    """
    acts = lv.get("actions") or []
    obs = np.asarray(lv.get("observations", []), dtype=np.uint8)
    # Need >= 2 actions: the leading RESET's span is the first state, and the
    # action after it is the first target. One action alone yields no pair.
    if obs.ndim != 3 or obs.shape[0] == 0 or len(acts) < 2:
        return None
    # `action_spans` is the single source of truth for the frame/action
    # alignment and raises if the spans don't tile the stream exactly -- a
    # corrupted or `n_obs`-stripped level, which must not be trained on.
    spans = common_utils.action_spans(acts, obs.shape[0])

    T = len(acts)
    nobs = np.asarray([b - a for a, b in spans], dtype=np.uint16)
    act = np.asarray([int(a["index"]) for a in acts], dtype=np.uint8)
    axy = np.full((T, 2), NO_COORD, dtype=np.uint8)
    opt = np.full((T, MAX_OPTIMAL), NO_ACTION, dtype=np.int16)
    oxy = np.full((T, MAX_OPTIMAL, 2), NO_COORD, dtype=np.uint8)
    pha = np.asarray([PHASE_ID.get(a.get("phase") or "other", OTHER_PHASE)
                      for a in acts], dtype=np.uint8)
    truncated = 0
    for i, a in enumerate(acts):
        xy = _coord_of(a)
        if xy is not None:
            axy[i] = xy
        for k, oa in enumerate((a.get("optimal") or [])[:MAX_OPTIMAL]):
            opt[i, k] = int(oa["index"])
            oxy_k = _coord_of(oa)
            if oxy_k is not None:
                oxy[i, k] = oxy_k
        truncated += max(0, len(a.get("optimal") or []) - MAX_OPTIMAL)
    lid = lv.get("level_id")
    lid = int(lid) if lid is not None else int(src_idx)
    return {"obs": obs, "nobs": nobs, "act": act, "axy": axy, "opt": opt,
            "oxy": oxy, "pha": pha, "lid": np.asarray(lid, dtype=np.int64),
            "_truncated": truncated}


def preprocess(data_dir: str, cache_dir: str, episodes_per_game: int | None,
               overwrite: bool = False) -> None:
    """Convert episode files into compact ``.npz`` caches.

    We keep only what training needs -- the frame stream as ``uint8`` plus the
    per-action arrays of `_encode_level` -- discarding the large legacy
    ``labels`` blob. One ``.npz`` per episode holds every level, keyed
    ``obs_<i>`` / ``nobs_<i>`` / ``act_<i>`` / ``axy_<i>`` / ``opt_<i>`` /
    ``oxy_<i>`` / ``pha_<i>`` / ``lid_<i>`` plus ``n_levels`` and
    ``cache_version``. ``i`` is the position among a given episode's
    *successfully encoded* levels; ``lid_<i>`` carries the stable source
    ``level_id`` the (game, level) holdout split keys on. The step is resumable:
    existing outputs are skipped unless ``--overwrite`` is given.

    Unlike the previous version this no longer refuses multi-frame levels: an
    action's whole animation is cached and ``nobs`` records the span, so the
    frame/action alignment survives to the dataset (see the module docstring).
    """
    os.makedirs(cache_dir, exist_ok=True)
    games = sorted(d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d)))
    print(f"[preprocess] {len(games)} games found under {data_dir}")

    total_written = 0
    for game in games:
        files = common_utils.episode_files(os.path.join(data_dir, game))
        if episodes_per_game is not None:
            files = files[:episodes_per_game]
        out_game_dir = os.path.join(cache_dir, game)
        os.makedirs(out_game_dir, exist_ok=True)

        written = 0
        n_trunc = 0
        t0 = time.time()
        for fpath in files:
            # episode_00000_seed0.ep.zst / .json -> episode_00000_seed0
            base = os.path.basename(str(fpath)).split(".", 1)[0]
            out_path = os.path.join(out_game_dir, base + ".npz")
            if os.path.exists(out_path) and not overwrite:
                continue
            try:
                d = common_utils.load_episode(fpath)
            except (json.JSONDecodeError, OSError, ValueError) as e:
                print(f"[preprocess] skipping unreadable {fpath}: {e}")
                continue

            payload: dict[str, np.ndarray] = {}
            n_levels = 0
            for src_idx, lv in enumerate(d["levels"]):
                try:
                    enc = _encode_level(lv, src_idx)
                except ValueError as e:      # spans don't tile the frame stream
                    print(f"[preprocess] SKIPPING corrupt level in {fpath} "
                          f"(level_id {lv.get('level_id')}): {e}")
                    continue
                if enc is None:
                    continue
                n_trunc += enc.pop("_truncated")
                for key, arr in enc.items():
                    payload[f"{key}_{n_levels}"] = arr
                n_levels += 1

            if n_levels == 0:
                continue
            payload["n_levels"] = np.asarray(n_levels, dtype=np.int64)
            payload["cache_version"] = np.asarray(CACHE_VERSION, dtype=np.int64)
            np.savez_compressed(out_path, **payload)
            written += 1

        total_written += written
        dt = time.time() - t0
        extra = (f"  ({n_trunc} optimal actions past MAX_OPTIMAL dropped)"
                 if n_trunc else "")
        print(f"[preprocess] {game}: wrote {written} new / {len(files)} "
              f"episodes in {dt:.1f}s{extra}")

    print(f"[preprocess] done. {total_written} new episode files written to "
          f"{cache_dir}")


# ----------------------------------------------------------------------------
# Stage 1b: the (game, level) train / val holdout
# ----------------------------------------------------------------------------
def _game_of(npz_path: str) -> str:
    """``.../cache_policy/<game>/episode_*.npz`` -> ``<game>``."""
    return os.path.basename(os.path.dirname(npz_path))


def _seed_of(npz_path: str) -> int | None:
    """``episode_00042_seed42.npz`` -> ``42`` (``None`` if the name has no seed)."""
    m = re.search(r"seed(\d+)", os.path.basename(npz_path))
    return int(m.group(1)) if m else None


def build_level_holdout(cache_dir: str, seed: int, level_val_frac: float,
                        min_levels: int, val_seeds: int,
                        discover_per_game: int = 64) -> dict:
    """Reserve a per-game random subset of each game's LEVELS for validation.

    The unit held out is a ``(game, level_id)`` puzzle, across EVERY seed of it:
    the trainer never sees a held level in any form, so val measures "solve an
    unseen level of a game you know" -- what the engine's auto-advance actually
    hits -- rather than a re-seeded / re-rotated copy of a level already trained
    on (different seeds of one level are near-identical on most games). Whole
    games are NOT held out here; that is the separate OOD test set.

    Returns ``{game: {"held_levels": [int, ...], "val_seeds": [int, ...] | None}}``
    where ``val_seeds`` caps which seeds of the held levels appear in val
    (``None`` == all). A game with fewer than ``min_levels`` distinct level_ids
    keeps every level in train (holding one out would gut its coverage) and gets
    an empty ``held_levels``.

    The level_id set per game is discovered from up to ``discover_per_game``
    sampled episodes; the actual train/val filtering later reads each episode's
    own ``lid_<i>`` so a level missing from some seeds is still handled exactly.
    """
    files = sorted(glob(os.path.join(cache_dir, "*", "*.npz")))
    by_game: dict[str, list[str]] = {}
    for f in files:
        by_game.setdefault(_game_of(f), []).append(f)

    out: dict = {}
    for game, gfiles in sorted(by_game.items()):
        rng_g = random.Random(f"{seed}:{game}")
        sample = gfiles if len(gfiles) <= discover_per_game else \
            rng_g.sample(gfiles, discover_per_game)
        lids: set[int] = set()
        for f in sample:
            with np.load(f) as z:
                n = int(z["n_levels"])
                if "lid_0" in z.files:
                    lids.update(int(z[f"lid_{i}"]) for i in range(n))
                else:
                    lids.update(range(n))
        lids_sorted = sorted(lids)
        # rng_g was partly consumed by the discovery sample above; re-seed so the
        # held set depends only on (seed, game), not on how many files existed.
        rng_g = random.Random(f"{seed}:{game}:hold")
        if len(lids_sorted) < min_levels:
            held: list[int] = []
        else:
            k = max(1, round(len(lids_sorted) * level_val_frac))
            k = min(k, len(lids_sorted) - 1)          # never hold out every level
            held = sorted(rng_g.sample(lids_sorted, k))

        vs: list[int] | None = None
        if held and val_seeds and val_seeds > 0:
            seeds_present = sorted(
                s for s in (_seed_of(f) for f in gfiles) if s is not None)
            if len(seeds_present) > val_seeds:
                vs = sorted(random.Random(f"{seed}:{game}:seeds")
                            .sample(seeds_present, val_seeds))
        out[game] = {"held_levels": held, "val_seeds": vs}
    return out


def load_or_build_level_holdout(cache_dir: str, seed: int, level_val_frac: float,
                                min_levels: int, val_seeds: int) -> dict:
    """Load ``<cache_dir>/level_holdout.json`` or build + persist it.

    Persisted so the split is STABLE as the corpus grows: adding seeds or games
    later must not reshuffle which levels are val (that would move a level the
    model already trained on into val, or vice versa). Delete the file to force
    a rebuild. Returns the manifest with the list fields turned into ``set``s.
    """
    path = os.path.join(cache_dir, LEVEL_HOLDOUT_FILE)
    params = {"seed": seed, "level_val_frac": level_val_frac,
              "min_levels": min_levels, "val_seeds": val_seeds}

    def _as_sets(games: dict) -> dict:
        return {g: {"held_levels": set(v["held_levels"]),
                    "val_seeds": (set(v["val_seeds"])
                                  if v["val_seeds"] is not None else None)}
                for g, v in games.items()}

    if os.path.exists(path):
        with open(path) as fh:
            blob = json.load(fh)
        if blob.get("params") != params:
            print(f"[data] WARNING: {path} was built with {blob.get('params')} "
                  f"!= requested {params}; REUSING the existing manifest so the "
                  f"split stays stable. Delete the file to rebuild.")
        return _as_sets(blob["games"])

    print(f"[data] building level-holdout manifest -> {path}")
    games = build_level_holdout(cache_dir, seed, level_val_frac, min_levels,
                                val_seeds)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump({"params": params, "games": games}, fh, indent=1,
                  sort_keys=True)
    os.replace(tmp, path)
    n_pairs = sum(len(v["held_levels"]) for v in games.values())
    n_games = sum(1 for v in games.values() if v["held_levels"])
    print(f"[data] holdout: {n_pairs} (game, level) puzzles across "
          f"{n_games}/{len(games)} games reserved for val "
          f"(min_levels={min_levels}, level_val_frac={level_val_frac}, "
          f"val_seeds={val_seeds or 'all'})")
    return _as_sets(games)


# ----------------------------------------------------------------------------
# Stage 2: dataset over cached levels
# ----------------------------------------------------------------------------
def _cap_frames(frames: np.ndarray, nfr: np.ndarray, max_frames: int):
    """Apply `_span_keep` to every step of a (frames, per-step span length) pair.

    ``frames`` is the flat stream, ``nfr[i]`` the length of step ``i``'s span.
    Returns the subsampled stream and the new span lengths."""
    if max_frames <= 0 or nfr.size == 0 or int(nfr.max()) <= max_frames:
        return frames, nfr
    starts = np.concatenate([[0], np.cumsum(nfr)])[:-1]
    keep = [s + _span_keep(int(k), max_frames) for s, k in zip(starts, nfr)]
    out_n = np.asarray([len(idx) for idx in keep], dtype=np.int64)
    return frames[np.concatenate(keep)], out_n


class LevelSequenceDataset(Dataset):
    """One item == a demonstration context, either one level or several.

    An item is a sequence of ``S`` *steps*. Step ``t`` carries the frame span of
    one action (its animation) and is supervised with the OPTIMAL action set at
    that state -- the action the recorder actually took is fed back in as context
    only. See the module docstring for the format and the alignment rule. The
    returned dict is::

        frames     (F,64,64) int64   flat frame stream of the whole context
        frame_step (F,)      int64   which step each frame belongs to
        frame_slot (F,)      int64   position of the frame within its own span
        nframes    (S,)      int64   frames per step (sums to F)
        act        (S,)      int64   TAKEN action fed in after step t's frames
        xy         (S,2)     int64   its click coord (NO_COORD if not a click)
        opt        (S,K)     int64   OPTIMAL action set (NO_ACTION padded)
        oxy        (S,K,2)   int64   each optimal action's click coord
        phase      (S,)      int64   phase code of the taken action
        length     int                == S

    A step whose ``opt`` row is entirely ``NO_ACTION`` is in context but carries
    no target (an unlabelled step, or one whose phase is in
    ``skip_target_phases``); the loss masks it out.

    Three train-only augmentations widen the distribution the in-context policy
    is trained on (all are no-ops for the deterministic val split):

    * ``suffix_augment`` -- draw a fresh RANDOM suffix of each level's trajectory
      every fetch (different, still-solvable start state; see ``_prep_level``).
    * ``multi_level`` -- concatenate a random SUBSET of the episode's levels, in
      random ORDER, into one continuous context so the policy learns to carry
      experience across level boundaries (see ``__getitem__``). Levels come from
      the SAME episode (the only place cross-level transfer is meaningful). The
      index level acts as an anchor that is always included, so every level is
      still covered once per epoch; ``k=1`` reproduces the single-level item.
    * ``action_shuffle`` -- relabel action ids 1..5 through a fresh random
      permutation each fetch, held fixed across the whole item, so the policy
      cannot memorize a fixed id->effect mapping and must read it off the
      in-context transition history instead (see the module docstring's
      "Action-identity shuffle" section and ``__getitem__``).
    """

    def __init__(self, cache_dir: str, split: str = "train",
                 val_frac: float = 0.05, max_states: int = 128, seed: int = 0,
                 suffix_augment: bool = False, multi_level: bool = False,
                 action_shuffle: bool = False,
                 max_levels: int = 8, return_index: bool = False,
                 resample_alpha: float = 0.0, max_frames: int = 4,
                 skip_target_phases: tuple[str, ...] = (),
                 taken_action_fallback: bool = True,
                 sampling_seed: int | None = None, dynamics: bool = False,
                 holdout: dict | None = None):
        self.max_states = max_states
        self.max_frames = max_frames
        # When set, __getitem__ also emits per-step (dyn_in, dyn_tgt, dyn_ok):
        # the settled board before / after step t's action, and whether that
        # pair is supervised, including a level's terminal action outcome.
        self.dynamics = dynamics
        self.suffix_augment = suffix_augment
        self.multi_level = multi_level
        self.action_shuffle = action_shuffle
        self.max_levels = max_levels
        # Phases whose steps stay in context but are NOT supervised. Empty by
        # default: an `optimal` set relabelled at an off-policy (explore/burst)
        # state is exactly the recovery target the policy needs, so dropping it
        # throws away the most valuable part of the corpus.
        bad = [p for p in skip_target_phases if p not in PHASE_ID]
        if bad:
            raise ValueError(f"unknown phase(s) {bad}; known: {list(PHASES)}")
        self.skip_phase_ids = np.asarray(
            sorted(PHASE_ID[p] for p in skip_target_phases), dtype=np.int64)
        # See _prep_level: supervise an oracle-less EXPERT step with its taken
        # action, so a step-wise adaptive generator is not silently unsupervised.
        self.taken_action_fallback = taken_action_fallback
        # See _choose_levels: pins the multi-level subset/order per item so a
        # multi_level VAL split is reproducible epoch to epoch. None == resample
        # every fetch (what training wants).
        self.sampling_seed = sampling_seed
        # Error-driven resampling state (see set_error_profile). When
        # return_index is set, __getitem__ tags each item with its (fpath, li)
        # so a teacher-forced scan can map per-step errors back to their level.
        self.return_index = return_index
        self.resample_alpha = resample_alpha
        # (fpath, level_idx) -> per-step dilated error weight in {0, 1}; None
        # until the first scan populates it. Keyed by (file, level) rather than
        # dataset index so multi_level sibling levels share the same profile.
        self.error_profile: dict[tuple[str, int], np.ndarray] | None = None
        # (game, level) holdout manifest (see load_or_build_level_holdout). When
        # given it REPLACES the legacy episode-level split: val is every seed of
        # a per-game random subset of levels, none of which is ever trained on.
        self.holdout = holdout
        self.holdout_val = holdout is not None and split == "val"
        # Index every (episode_file, positional level index) pair without loading
        # obs. `file_lids[f][pos]` is that level's source level_id.
        print(f"[data] listing cached episodes under {cache_dir}", flush=True)
        files = sorted(glob(os.path.join(cache_dir, "*", "*.npz")))
        if not files:
            raise FileNotFoundError(
                f"No cached .npz under {cache_dir}. Run `preprocess` first.")

        self.index: list[tuple[str, int]] = []
        # Per-episode level list, so multi_level can sample sibling levels of an
        # anchor without re-reading n_levels on every fetch. Under a holdout,
        # the TRAIN split's list excludes held levels (so they never enter a
        # training context, not even as an unsupervised sibling).
        self.file_levels: dict[str, list[int]] = {}
        self.file_lids: dict[str, list[int]] = {}

        def _open(f):
            with np.load(f) as z:
                ver = int(z["cache_version"]) if "cache_version" in z.files else 1
                if ver != CACHE_VERSION:
                    raise RuntimeError(
                        f"{f} is cache version {ver}, this trainer needs "
                        f"v{CACHE_VERSION} (adds `lid_<i>` for the (game, level) "
                        f"holdout split). Re-run "
                        f"`python train_policy_dynamics.py preprocess --overwrite`.")
                n = int(z["n_levels"])
                lids = ([int(z[f"lid_{i}"]) for i in range(n)]
                        if "lid_0" in z.files else list(range(n)))
            return n, lids

        if holdout is None:
            # Legacy: deterministic train/val split at the *episode* level so no
            # level of a val episode leaks into training. Kept for
            # --no-level-holdout and external callers that pass no manifest.
            rng = random.Random(seed)
            files_shuf = files[:]
            rng.shuffle(files_shuf)
            n_val = max(1, int(len(files_shuf) * val_frac))
            val_files = set(files_shuf[:n_val])
            in_split = ((lambda f: f in val_files) if split == "val"
                        else (lambda f: f not in val_files))
            n_eps = 0
            for f in tqdm(files, desc=f"[data] indexing {split}", unit="file",
                          dynamic_ncols=True):
                if not in_split(f):
                    continue
                n_eps += 1
                n, lids = _open(f)
                self.file_levels[f] = list(range(n))
                self.file_lids[f] = lids
                for li in range(n):
                    self.index.append((f, li))
        else:
            # (game, level) holdout. TRAIN keeps only non-held levels; VAL keeps
            # only held levels, optionally restricted to the manifest's val
            # seeds. `file_levels` (multi_level sibling pool) is held-free on
            # train; on val it is the full positional list so a deployment-
            # faithful item can prepend the episode's earlier non-held levels.
            n_eps = 0
            for f in tqdm(files, desc=f"[data] indexing {split}", unit="file",
                          dynamic_ncols=True):
                g = _game_of(f)
                hv = holdout.get(g)
                held = hv["held_levels"] if hv else set()
                vseeds = hv["val_seeds"] if hv else None
                n, lids = _open(f)
                self.file_lids[f] = lids
                if split == "train":
                    keep_pos = [i for i in range(n) if lids[i] not in held]
                    if not keep_pos:
                        continue
                    n_eps += 1
                    self.file_levels[f] = keep_pos
                    for i in keep_pos:
                        self.index.append((f, i))
                else:  # val
                    held_pos = [i for i in range(n) if lids[i] in held]
                    if not held_pos:
                        continue
                    if vseeds is not None and (_seed_of(f) not in vseeds):
                        continue
                    n_eps += 1
                    self.file_levels[f] = list(range(n))
                    for i in held_pos:
                        self.index.append((f, i))

        kind = "level-holdout" if holdout is not None else "episode-split"
        print(f"[data] {kind} split={split}: {len(self.index)} level-sequences "
              f"from {n_eps} episodes", flush=True)

    def __len__(self) -> int:
        return len(self.index)

    @staticmethod
    def _load_level(z, li: int):
        """Return the raw per-level cache arrays for level ``li`` (see
        `_encode_level` for their meaning), all still index-aligned to the
        level's ``T`` actions."""
        return (z[f"obs_{li}"], z[f"nobs_{li}"], z[f"act_{li}"], z[f"axy_{li}"],
                z[f"opt_{li}"], z[f"oxy_{li}"], z[f"pha_{li}"])

    def _prep_level(self, z, li: int, fpath: str | None = None,
                    supervise: bool = True):
        """Load level ``li`` and return its aligned per-step arrays, with the
        frame cap and suffix augmentation applied.

        ``supervise=False`` (deployment-faithful val only) keeps the level's
        frames and taken actions as pure in-context history but strips every
        target, so the val metric is scored on the held anchor level alone.

        Alignment: the cache stores everything indexed by ACTION, where action
        ``i`` produced frame span ``i`` of the flat stream and ``optimal[i]`` is
        the optimal set at the state that action was decided from -- i.e. at the
        settled (last) frame of span ``i-1``. So step ``t`` of a training item
        pairs the frames of span ``t`` with the target/context action ``t+1``:
        drop the trailing span (no following action) and the leading RESET (which
        only produced the first span and is never a target). With every span one
        frame long this is exactly the old ``(obs[t], act[t+1])`` pairing.

        Returns ``(frames, nfr, act, xy, opt, oxy, pha, dyn_tgt)``, or ``None``
        for a level too short to yield a single step. Dynamics targets retain
        the outcome of the terminal action without exposing it as input.
        """
        obs, nobs, act, axy, opt, oxy, pha = self._load_level(z, li)
        T = act.shape[0]
        S = T - 1                 # steps: span t paired with action t+1
        if S < 1:
            return None
        nobs = nobs.astype(np.int64)
        # Exclusive prefix sums -> where each span starts in the flat stream.
        starts = np.concatenate([[0], np.cumsum(nobs)])          # (T+1,)
        frames = obs[:starts[S]]  # spans 0..S-1; the trailing span is dropped
        nfr = nobs[:S]
        dyn_tgt = obs[starts[2:] - 1] if self.dynamics else None
        act = act[1:T].astype(np.int64)   # action taken AT step t's settled frame
        xy = axy[1:T].astype(np.int64)    # its click coord (NO_COORD if not one)
        opt = opt[1:T].astype(np.int64)   # the optimal SET at that same state
        oxy = oxy[1:T].astype(np.int64)
        pha = pha[1:T].astype(np.int64)
        if self.taken_action_fallback:
            # An EXPERT step with no oracle target supervises its TAKEN action.
            #
            # v2 supervises `optimal` only, which silently unsupervised every
            # generator whose expert is a step-wise adaptive policy rather than a
            # plan -- those legitimately have no oracle and record
            # `optimal=None` (solvers/common/mm.py, generate_re86_training.py).
            # The result was 6 games (re86, mm01..mm05) at 100% unlabelled EXPERT
            # steps: present in every context, contributing nothing to the loss.
            # v1 trained them fine because its target WAS the taken action, so
            # this restores exactly that for the steps that have no better label.
            #
            # Deliberately EXPERT-phase only: falling back on an explore/burst
            # step would clone a deliberately-random action as if it were expert.
            miss = (pha == EXPERT_PHASE) & (opt < 0).all(axis=1)
            if miss.any():
                opt, oxy = opt.copy(), oxy.copy()
                opt[miss, 0] = act[miss]
                oxy[miss, 0] = xy[miss]
        if self.skip_phase_ids.size:
            # Keep the step in context, strip its target.
            opt = opt.copy()
            opt[np.isin(pha, self.skip_phase_ids)] = NO_ACTION
        if not supervise:
            # Context-only level in a deployment-faithful val item: no targets.
            opt = np.full_like(opt, NO_ACTION)
        frames, nfr = _cap_frames(frames, nfr, self.max_frames)
        L = S                     # steps in this level (>=1)
        if self.suffix_augment and L > 1:
            # Suffix augmentation (training only): instead of the whole
            # trajectory, train on a random suffix s_j .. s_{L-1}. Any suffix is
            # itself a correct, solvable trajectory from a *different* start
            # state -- the remainder of the recording is the proof -- so this
            # widens the start-state distribution at zero data cost, and is the
            # one universally-safe form of "exploring starts" (see DESIGN.md
            # "Suffix-only training"). Uniform start j in [0, L-1] gives a suffix
            # length in [1, L]; a fresh j is drawn on every fetch, so each epoch
            # sees a different start state for the same level.
            #
            # Error-driven resampling biases this start toward the states the
            # policy gets wrong: with a per-step error profile in hand, draw j ~
            # (1 + alpha * dilated_errors[j]) instead of uniform, so a mistake
            # and its neighbourhood land near the FRONT of the suffix (where the
            # whole remaining suffix is trained on it) far more often. The +1
            # base keeps every start reachable, so clean states are never
            # dropped -- DAgger's D <- D u D_new aggregation. Falls back to
            # uniform when there is no profile or the level has no errors.
            prof = None
            if (self.error_profile is not None and fpath is not None
                    and self.resample_alpha > 0):
                prof = self.error_profile.get((fpath, li))
            if prof is not None and len(prof) == L and float(prof.sum()) > 0:
                w = [1.0 + self.resample_alpha * float(prof[t]) for t in range(L)]
                j = random.choices(range(L), weights=w, k=1)[0]
            else:
                j = random.randint(0, L - 1)
            # Slicing steps also slices the FLAT frame stream -- the suffix keeps
            # only the frames of the steps it keeps.
            frames = frames[int(nfr[:j].sum()):]
            nfr, act, xy = nfr[j:], act[j:], xy[j:]
            opt, oxy, pha = opt[j:], oxy[j:], pha[j:]
            if dyn_tgt is not None:
                dyn_tgt = dyn_tgt[j:]
        return frames, nfr, act, xy, opt, oxy, pha, dyn_tgt

    def set_error_profile(self, profile: dict[tuple[str, int], np.ndarray] | None):
        """Install the per-(file, level) dilated error profile from a scan (see
        ``scan_errors`` / ``dilate_errors``). Drives both the suffix-start bias
        (``_prep_level``) and the level-level weights (``item_sampling_weights``).
        Pass ``None`` to revert to uniform sampling."""
        self.error_profile = profile

    def item_sampling_weights(self) -> torch.Tensor:
        """Per-dataset-index weight for a WeightedRandomSampler, aligned to
        ``self.index``. A level's weight is ``1 + alpha * mean(dilated_errors)``,
        bounded in ``[1, 1 + alpha]`` -- error-heavy levels are drawn more often
        while every level keeps at least a ``1/(1+alpha)`` share so clean data is
        never starved. Uniform (all ones) until a profile is installed."""
        w = np.ones(len(self.index), dtype=np.float64)
        if self.error_profile is not None and self.resample_alpha > 0:
            for i, (fpath, li) in enumerate(self.index):
                prof = self.error_profile.get((fpath, li))
                if prof is not None and prof.size:
                    w[i] = 1.0 + self.resample_alpha * float(prof.mean())
        return torch.as_tensor(w, dtype=torch.double)

    def _choose_levels(self, fpath: str, anchor: int) -> list[int]:
        """Level indices for this item: just ``[anchor]`` unless multi_level, in
        which case a subset of the episode's levels (always containing the
        anchor) in shuffled order -- see the class docstring.

        With ``sampling_seed`` set the subset and order are a DETERMINISTIC
        function of ``(seed, file, anchor)`` rather than a fresh global draw. That
        is what makes a multi-level VALIDATION split usable: the item is identical
        every epoch, so the metric is comparable across epochs instead of moving
        because a different level combination happened to be sampled. Training
        leaves it None -- there the per-epoch resampling is the augmentation.

        Deployment-faithful VAL (``holdout_val``): the anchor is a HELD-OUT
        level. With ``multi_level`` it is returned last, preceded by the
        episode's EARLIER non-held levels in ``level_id`` order -- the in-context
        history the live roll-out has when the engine auto-advances into an
        unseen level. Those earlier levels are context only (``_prep_level
        supervise=False``); the metric is scored on the held anchor alone.
        Without ``multi_level`` the val item is the held level in isolation."""
        if self.holdout_val:
            lids = self.file_lids[fpath]
            if not self.multi_level:
                return [anchor]
            held = self.holdout.get(_game_of(fpath), {}).get(
                "held_levels", set())
            a_lid = lids[anchor]
            ctx = sorted((i for i in range(len(lids))
                          if lids[i] not in held and lids[i] < a_lid),
                         key=lambda i: lids[i])
            keep_ctx = max(0, self.max_levels - 1)   # anchor takes one slot
            return (ctx[len(ctx) - keep_ctx:] if keep_ctx else []) + [anchor]
        if not self.multi_level:
            return [anchor]
        rng = random
        if self.sampling_seed is not None:
            rng = random.Random(
                zlib.crc32(f"{self.sampling_seed}:{_game_of(fpath)}/"
                           f"{os.path.basename(fpath)}:{anchor}".encode()))
        levels = self.file_levels[fpath]
        k = rng.randint(1, min(len(levels), self.max_levels))
        others = [l for l in levels if l != anchor]
        chosen = [anchor] + rng.sample(others, k - 1)
        rng.shuffle(chosen)       # anchor is not necessarily first
        return chosen

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        fpath, anchor = self.index[i]
        chosen = self._choose_levels(fpath, anchor)
        # Deployment-faithful val: every chosen level EXCEPT the last (the held
        # anchor) is in-context history only -- its targets are stripped so the
        # metric scores the unseen level alone.
        ctx_only = self.holdout_val and self.multi_level and len(chosen) > 1
        last_pos = len(chosen) - 1
        with np.load(fpath) as z:
            raw = [self._prep_level(z, li, fpath,
                                    supervise=not (ctx_only and pos != last_pos))
                   for pos, li in enumerate(chosen)]
        parts, sup_flags = [], []
        for pos, p in enumerate(raw):
            if p is None:
                continue
            parts.append(p)
            sup_flags.append(not (ctx_only and pos != last_pos))
        # A terminal action has a dynamics target, but its prediction must not
        # be fed into the following level as if that were the same trajectory.
        level_end = np.concatenate([
            (np.arange(len(p[1])) == len(p[1]) - 1) for p in parts])
        # Per-step "this level is supervised" mask (all True unless a
        # deployment-faithful val item has context-only prefix levels). Gates the
        # dynamics targets; the policy targets are already stripped in opt.
        sup_level = np.concatenate([
            np.full(len(p[1]), s, dtype=bool) for p, s in zip(parts, sup_flags)])
        # Concatenate the chosen levels into one continuous context. Across a
        # level boundary the "next" state token is the following level's start --
        # exactly what the live roll-out sees when a level is won and the engine
        # auto-advances, so causal attention learns to reuse earlier levels'
        # context. Each level's targets stay that level's own optimal actions.
        frames = np.concatenate([p[0] for p in parts], axis=0)
        nfr = np.concatenate([p[1] for p in parts], axis=0)
        act = np.concatenate([p[2] for p in parts], axis=0)
        xy = np.concatenate([p[3] for p in parts], axis=0)
        opt = np.concatenate([p[4] for p in parts], axis=0)
        oxy = np.concatenate([p[5] for p in parts], axis=0)
        pha = np.concatenate([p[6] for p in parts], axis=0)
        if self.action_shuffle:
            # ONE random permutation of {1..5} for the whole item -- see the
            # module docstring's "Action-identity shuffle" section. RESET (0)
            # and CLICK_INDEX (6) are left fixed (identity in `remap`).
            ids = list(range(1, CLICK_INDEX))
            shuffled = ids[:]
            random.shuffle(shuffled)
            remap = np.arange(NUM_ACTION_TYPES)
            remap[ids] = shuffled
            act = remap[act]
            opt = np.where(opt >= 0, remap[np.clip(opt, 0, None)], opt)
        if self.dynamics:
            dyn_tgt = np.concatenate([p[7] for p in parts], axis=0)
        L = nfr.shape[0]
        cap = self.max_states
        if L > cap:
            # FIFO: drop the OLDEST steps and keep the most recent `cap` (their
            # frames go with them). The live roll-out likewise truncates to its
            # most recent window, so training on the suffix matches the
            # distribution seen at inference.
            drop = L - cap
            frames = frames[int(nfr[:drop].sum()):]
            nfr, act, xy = nfr[drop:], act[drop:], xy[drop:]
            opt, oxy, pha = opt[drop:], oxy[drop:], pha[drop:]
            level_end = level_end[drop:]
            sup_level = sup_level[drop:]
            if self.dynamics:
                dyn_tgt = dyn_tgt[drop:]
            L = cap
        # Per-frame provenance: which step a frame belongs to, and its slot
        # within that step's span. The model turns these into token positions
        # (see InContextPolicy.forward), so the packing is computed once here
        # rather than per forward pass.
        frame_step = np.repeat(np.arange(L, dtype=np.int64), nfr)
        span_start = np.concatenate([[0], np.cumsum(nfr)])[:-1]
        frame_slot = np.arange(frames.shape[0], dtype=np.int64) \
            - np.repeat(span_start, nfr)
        item = {
            "frames": torch.from_numpy(frames.astype(np.int64)),
            "frame_step": torch.from_numpy(frame_step),
            "frame_slot": torch.from_numpy(frame_slot),
            "nframes": torch.from_numpy(nfr.astype(np.int64)),
            "act": torch.from_numpy(act.astype(np.int64)),
            "xy": torch.from_numpy(xy.astype(np.int64)),
            "opt": torch.from_numpy(opt.astype(np.int64)),
            "oxy": torch.from_numpy(oxy.astype(np.int64)),
            "phase": torch.from_numpy(pha.astype(np.int64)),
            "length": L,
            "game_id": _game_of(fpath),
        }
        if self.dynamics:
            settled = span_start + nfr - 1                       # (L,) into frames
            item["dyn_in"] = torch.from_numpy(frames[settled].astype(np.uint8))
            item["dyn_tgt"] = torch.from_numpy(dyn_tgt.astype(np.uint8))
            item["dyn_ok"] = torch.from_numpy(sup_level)
            item["dyn_continue"] = torch.from_numpy(~level_end & sup_level)
        if self.return_index:
            # Tag the item with its (file, level) so a teacher-forced scan can
            # attribute per-step errors back to the right level. Only meaningful
            # for the single-level scan view (multi_level off), where the item is
            # exactly the anchor level's full trajectory.
            item["fpath"] = fpath
            item["li"] = anchor
        return item


def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Pad a batch of variable-length contexts.

    Two independent ragged axes: STEPS (padded to ``Lmax``, flagged by ``mask``)
    and FRAMES (padded to ``Fmax``, flagged by ``frame_mask``). They are separate
    because a step owns a whole animation, so frames per item is not a multiple
    of steps per item -- see the module docstring on multi-frame observations."""
    B = len(batch)
    lengths = torch.tensor([b["length"] for b in batch], dtype=torch.long)
    Lmax = int(lengths.max())
    Fmax = max(int(b["frames"].shape[0]) for b in batch)
    K = int(batch[0]["opt"].shape[1])

    frames = torch.zeros(B, Fmax, BOARD_H, BOARD_W, dtype=torch.long)
    frame_step = torch.zeros(B, Fmax, dtype=torch.long)
    frame_slot = torch.zeros(B, Fmax, dtype=torch.long)
    frame_mask = torch.zeros(B, Fmax, dtype=torch.bool)
    nframes = torch.zeros(B, Lmax, dtype=torch.long)
    act = torch.zeros(B, Lmax, dtype=torch.long)
    # Pad click coords with the NO_COORD sentinel (padded steps are non-click).
    xy = torch.full((B, Lmax, 2), NO_COORD, dtype=torch.long)
    # Pad optimal sets with NO_ACTION == "no target here".
    opt = torch.full((B, Lmax, K), NO_ACTION, dtype=torch.long)
    oxy = torch.full((B, Lmax, K, 2), NO_COORD, dtype=torch.long)
    phase = torch.full((B, Lmax), OTHER_PHASE, dtype=torch.long)
    mask = torch.zeros(B, Lmax, dtype=torch.bool)   # True == valid step
    for i, b in enumerate(batch):
        L, Fi = b["length"], int(b["frames"].shape[0])
        frames[i, :Fi] = b["frames"]
        frame_step[i, :Fi] = b["frame_step"]
        frame_slot[i, :Fi] = b["frame_slot"]
        frame_mask[i, :Fi] = True
        nframes[i, :L] = b["nframes"]
        act[i, :L] = b["act"]
        xy[i, :L] = b["xy"]
        opt[i, :L] = b["opt"]
        oxy[i, :L] = b["oxy"]
        phase[i, :L] = b["phase"]
        mask[i, :L] = True
    out = {"frames": frames, "frame_step": frame_step, "frame_slot": frame_slot,
           "frame_mask": frame_mask, "nframes": nframes, "act": act, "xy": xy,
           "opt": opt, "oxy": oxy, "phase": phase, "mask": mask,
           "lengths": lengths,
           "game_id": [b.get("game_id", "unknown") for b in batch]}
    if "dyn_tgt" in batch[0]:                     # auxiliary dynamics targets
        dyn_in = torch.zeros(B, Lmax, BOARD_H, BOARD_W, dtype=torch.uint8)
        dyn_tgt = torch.zeros(B, Lmax, BOARD_H, BOARD_W, dtype=torch.uint8)
        dyn_ok = torch.zeros(B, Lmax, dtype=torch.bool)
        dyn_continue = torch.zeros_like(dyn_ok)
        for i, b in enumerate(batch):
            L = b["length"]
            dyn_in[i, :L] = b["dyn_in"]
            dyn_tgt[i, :L] = b["dyn_tgt"]
            dyn_ok[i, :L] = b["dyn_ok"]
            dyn_continue[i, :L] = b.get("dyn_continue", b["dyn_ok"])
        out.update(dyn_in=dyn_in, dyn_tgt=dyn_tgt, dyn_ok=dyn_ok,
                   dyn_continue=dyn_continue)
    if "fpath" in batch[0]:                       # error-scan identity tags
        out["fpath"] = [b["fpath"] for b in batch]
        out["li"] = [b["li"] for b in batch]
    return out


def model_inputs(batch: dict, device) -> dict[str, torch.Tensor]:
    """The subset of a collated batch that ``InContextPolicy.forward`` consumes,
    moved to ``device``. One place to keep the call sites (train loop, eval,
    error scan, downstream importers) in sync with the model signature."""
    keys = ("frames", "act", "xy", "mask", "nframes", "frame_step",
            "frame_slot", "frame_mask")
    return {k: batch[k].to(device, non_blocking=True) for k in keys}


def optimal_targets(opt: torch.Tensor, oxy: torch.Tensor, mask: torch.Tensor,
                    grid: int):
    """Turn a step's OPTIMAL ACTION SET into soft targets for the two heads.

    ``opt`` (B,L,K) holds the optimal action indices with ``NO_ACTION`` padding
    and ``oxy`` (B,L,K,2) their click coordinates; a row that is entirely
    ``NO_ACTION`` means "in context, no target" (unlabelled step, or a phase
    excluded via ``--skip-target-phases``). Returns::

        type_tgt  (B,L,NUM_ACTION_TYPES)  uniform over the set's action types
        ptr_tgt   (B,L,grid*grid)         uniform over the set's click cells
        tgt_mask  (B,L)                   step has >=1 optimal action
        ptr_mask  (B,L)                   its optimal set contains a click

    A UNIFORM distribution over the set is the right target because the actions
    in it are by construction equally optimal -- training against one arbitrary
    tie-break would penalise the model for preferring a different, equally
    correct move (see DESIGN.md, *optimal-action sets*). Duplicate entries would
    simply weight that action more, which is why the mass is normalised by the
    number of entries rather than by the number of distinct ones."""
    B, L, _K = opt.shape
    valid = (opt >= 0) & mask[..., None]                         # (B,L,K)
    n_opt = valid.sum(-1)                                        # (B,L)
    tgt_mask = n_opt > 0

    type_tgt = opt.new_zeros((B, L, NUM_ACTION_TYPES), dtype=torch.float32)
    type_tgt.scatter_add_(-1, opt.clamp(min=0), valid.to(type_tgt.dtype))
    type_tgt = type_tgt / n_opt.clamp(min=1).unsqueeze(-1)

    is_click = valid & (opt == CLICK_INDEX)                      # (B,L,K)
    n_click = is_click.sum(-1)
    ptr_mask = n_click > 0
    cells = grid * grid
    bins = coord_to_bin_flat(oxy, grid).clamp(0, cells - 1)      # (B,L,K)
    ptr_tgt = opt.new_zeros((B, L, cells), dtype=torch.float32)
    ptr_tgt.scatter_add_(-1, bins, is_click.to(ptr_tgt.dtype))
    ptr_tgt = ptr_tgt / n_click.clamp(min=1).unsqueeze(-1)
    return type_tgt, ptr_tgt, tgt_mask, ptr_mask


def _soft_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-position cross-entropy against a probability distribution target.
    Computed in fp32 so it is safe under autocast."""
    return -(target * F.log_softmax(logits.float(), dim=-1)).sum(-1)


def action_loss(type_logits: torch.Tensor, ptr_logits: torch.Tensor,
                opt: torch.Tensor, oxy: torch.Tensor, mask: torch.Tensor,
                grid: int, pointer_weight: float = 1.0):
    """Combined type + pointer loss and metrics, supervised by OPTIMAL actions.

    ``type_logits`` (B,L,NUM_ACTION_TYPES), ``ptr_logits`` (B,L,grid*grid),
    ``opt``/``oxy`` the optimal-action sets (see `optimal_targets`). The type head
    is trained on every step that HAS an optimal set -- never on the action the
    recorder merely took, so exploratory detours and bursts contribute their
    relabelled expert target instead of teaching the mistake. The pointer head is
    trained only on steps whose optimal set contains a click.

    Accuracy is set membership, not equality: a prediction is correct if it is
    among the optimal actions (and, for clicks, on one of their cells). Returns
    ``(loss, stats)`` where ``stats`` has scalar sums for logging:
    ``type_correct``, ``n_steps`` (SUPERVISED steps), ``click_correct``,
    ``n_clicks`` and the two component losses."""
    type_tgt, ptr_tgt, tgt_mask, ptr_mask = optimal_targets(opt, oxy, mask, grid)

    n_sup = tgt_mask.sum()
    tl = _soft_ce(type_logits, type_tgt)                         # (B,L)
    type_loss = (tl * tgt_mask).sum() / n_sup.clamp(min=1)

    n_clicks = int(ptr_mask.sum().item())
    if n_clicks > 0:
        pl = _soft_ce(ptr_logits, ptr_tgt)                       # (B,L)
        ptr_loss = (pl * ptr_mask).sum() / ptr_mask.sum()
    else:
        # Keep the pointer head in the graph (zero grad) so DDP/AMP stay happy
        # even in a batch with no clicks.
        ptr_loss = ptr_logits.sum() * 0.0

    loss = type_loss + pointer_weight * ptr_loss
    with torch.no_grad():
        pred = type_logits.argmax(-1)                            # (B,L)
        hit = (opt == pred.unsqueeze(-1)) & (opt >= 0)           # in the set?
        type_correct = int((hit.any(-1) & tgt_mask).sum().item())
        ptr_pred = ptr_logits.argmax(-1)                         # (B,L)
        ptr_hit = ((coord_to_bin_flat(oxy, grid) == ptr_pred.unsqueeze(-1))
                   & (opt == CLICK_INDEX) & (opt >= 0))
        click_correct = int((ptr_hit.any(-1) & ptr_mask).sum().item())
    stats = {
        "type_correct": type_correct, "n_steps": int(n_sup.item()),
        "click_correct": click_correct, "n_clicks": n_clicks,
        "type_loss": float(type_loss.item()), "ptr_loss": float(ptr_loss.item()),
    }
    return loss, stats


def cap_true_per_row(mask: torch.Tensor, cap: int,
                     generator: torch.Generator | None = None) -> torch.Tensor:
    """Return a copy of the (B,L) bool ``mask`` with at most ``cap`` True per row.

    When a row has more than ``cap`` True entries a uniform random ``cap`` of them
    are kept (the rest cleared). ``cap <= 0`` is a no-op. Used to bound how many
    steps per training item the dynamics decoder runs on -- a full-resolution
    64x64 decode per step is too large to do for every step of a long context.
    """
    if cap <= 0 or mask.numel() == 0:
        return mask
    out = mask.clone()
    over = mask.sum(1) > cap
    for b in over.nonzero(as_tuple=False).flatten().tolist():
        idx = mask[b].nonzero(as_tuple=False).flatten()
        perm = torch.randperm(idx.numel(), generator=generator, device=mask.device)
        out[b, idx[perm[cap:]]] = False
    return out


def dynamics_loss(col_logits: torch.Tensor, chg_logits: torch.Tensor,
                  dec_bl: torch.Tensor, dyn_in: torch.Tensor,
                  dyn_tgt: torch.Tensor, fog: int = -1,
                  changed_weight: float = 5.0, fog_reveal_weight: float = 0.25,
                  change_head_weight: float = 1.0):
    """Next-board prediction loss for the auxiliary dynamics head + metrics.

    ``col_logits`` (Nd,C,64,64), ``chg_logits`` (Nd,64,64) and ``dec_bl`` (Nd,2)
    come straight from ``InContextPolicy.forward(return_dynamics=True)`` -- one
    row per DECODED step, ``dec_bl[k] == (batch, step)``. ``dyn_in`` / ``dyn_tgt``
    (B,L,64,64) uint8 are the settled boards before / after each step's action;
    they are gathered down to the decoded rows here.

    * colour head -- per-cell cross-entropy against ``dyn_tgt``, with cells that
      actually change up-weighted (``changed_weight``) since the board is mostly
      static. With a ``fog`` sentinel: both-fog cells carry zero weight (nothing
      to learn) and first-time-revealed cells (fog -> not fog) are down-weighted
      to ``fog_reveal_weight`` (their contents are only partly knowable).
    * change head -- per-cell BCE predicting which cells differ next.

    Returns ``(loss, stats)`` with SUMS for exact epoch-level aggregation:
    ``col_correct`` / ``col_cells`` (per-cell colour acc over evaluated cells),
    ``chg_correct`` / ``chg_cells`` (over cells that changed), ``exact`` /
    ``sup_steps`` (whole-board hits), ``iou_i`` / ``iou_u`` (change-mask IoU),
    plus the two component losses.
    """
    Nd, C, H, W = col_logits.shape
    if Nd == 0:                                    # nothing to supervise
        z = col_logits.sum() * 0.0 + chg_logits.sum() * 0.0
        return z, {k: 0 for k in DYN_SUM_KEYS} | {
            "colour_loss": 0.0, "change_loss": 0.0}
    bi, li = dec_bl[:, 0], dec_bl[:, 1]
    inb = dyn_in[bi, li].long()                                 # (Nd,H,W)
    tgt = dyn_tgt[bi, li].long()
    changed = inb != tgt

    w = torch.where(changed, col_logits.new_tensor(changed_weight),
                    col_logits.new_tensor(1.0))                 # (Nd,H,W) float
    eval_cell = torch.ones_like(changed)
    if fog >= 0:
        both_fog = (inb == fog) & (tgt == fog)
        revealed = (inb == fog) & (tgt != fog)
        w = torch.where(both_fog, col_logits.new_tensor(0.0), w)
        w = torch.where(revealed, col_logits.new_tensor(fog_reveal_weight), w)
        eval_cell = ~both_fog

    ce = F.cross_entropy(col_logits.float(), tgt, reduction="none")  # (Nd,H,W)
    colour_loss = (ce * w).sum() / w.sum().clamp(min=1)

    chg_tgt = changed.float()
    bce = F.binary_cross_entropy_with_logits(
        chg_logits.float(), chg_tgt, reduction="none")          # (Nd,H,W)
    n_cell = eval_cell.sum().clamp(min=1)
    change_loss = (bce * eval_cell).sum() / n_cell

    loss = colour_loss + change_head_weight * change_loss
    with torch.no_grad():
        pred = col_logits.argmax(1)                             # (Nd,H,W)
        cell_ok = (pred == tgt) & eval_cell
        col_correct = int(cell_ok.sum().item())
        col_cells = int(eval_cell.sum().item())
        chg_eval = changed & eval_cell
        chg_correct = int((cell_ok & chg_eval).sum().item())
        chg_cells = int(chg_eval.sum().item())
        board_bad = ((pred != tgt) & eval_cell).any(-1).any(-1)  # (Nd,)
        exact = int((~board_bad).sum().item())
        cp = (chg_logits > 0) & eval_cell
        ct = changed & eval_cell
        iou_i = int((cp & ct).sum().item())
        iou_u = int((cp | ct).sum().item())
    stats = {
        "col_correct": col_correct, "col_cells": col_cells,
        "chg_correct": chg_correct, "chg_cells": chg_cells,
        "exact": exact, "sup_steps": int(Nd),
        "iou_i": iou_i, "iou_u": iou_u,
        "colour_loss": float(colour_loss.item()),
        "change_loss": float(change_loss.item()),
    }
    return loss, stats


# ----------------------------------------------------------------------------
# Stage 4: training
# ----------------------------------------------------------------------------
# Keys `dynamics_loss` returns as running SUMS, aggregated then turned into
# ratios by `dyn_ratios`. Kept in one place so the train loop and `run_epoch`
# accumulate identically.
DYN_SUM_KEYS = ("col_correct", "col_cells", "chg_correct", "chg_cells",
                "exact", "sup_steps", "iou_i", "iou_u")


def dyn_ratios(acc: dict) -> dict:
    """Running dynamics sums -> ``{cell, chg, exact, iou}`` ratio metrics."""
    r = lambda a, b: (a / b) if b else float("nan")
    return {"cell": r(acc["col_correct"], acc["col_cells"]),
            "chg": r(acc["chg_correct"], acc["chg_cells"]),
            "exact": r(acc["exact"], acc["sup_steps"]),
            "iou": r(acc["iou_i"], acc["iou_u"])}


def fmt_dyn(m: dict) -> str:
    return (f"cell {m['cell']:.3f} chg {m['chg']:.3f} "
            f"exact {m['exact']:.3f} iou {m['iou']:.3f}")


def build_dyn_decode(batch, mask, dyn_cfg: dict, generator=None) -> torch.Tensor:
    """(B,L) bool of which steps the dynamics decoder should run on this batch:
    valid steps that have a next board (``dyn_ok``), capped per item to
    ``dyn_cfg['max_steps']`` uniformly at random (a full-res 64x64 decode per
    step is too large to do for every step of a long context)."""
    dok = batch["dyn_ok"].to(mask.device, non_blocking=True) & mask
    return cap_true_per_row(dok, dyn_cfg.get("max_steps", 0), generator=generator)


def dynamics_term(col_logits, chg_logits, dec_bl, batch, dyn_cfg: dict,
                  weight: float | None = None):
    """``(scaled_loss_term, stats)`` for the dynamics head from its logits.

    ``scaled_loss_term`` is already multiplied by ``dyn_cfg['weight']`` (or
    ``weight``, when given -- ``dynamics_rollout_term`` uses this to apply
    ``dyn_cfg['unroll_weight']`` instead) and ready to add to the policy loss.
    Factored out so the train loop and ``run_epoch`` build the auxiliary term
    the same way from one ``return_dynamics=True`` call.
    """
    dev = col_logits.device
    dloss, dstats = dynamics_loss(
        col_logits, chg_logits, dec_bl,
        batch["dyn_in"].to(dev, non_blocking=True),
        batch["dyn_tgt"].to(dev, non_blocking=True),
        fog=dyn_cfg["fog"], changed_weight=dyn_cfg["changed_weight"],
        fog_reveal_weight=dyn_cfg["fog_reveal_weight"],
        change_head_weight=dyn_cfg["change_head_weight"])
    w = dyn_cfg["weight"] if weight is None else weight
    return w * dloss, dstats


def dynamics_rollout_term(model, batch: dict, inp: dict, dyn_cfg: dict,
                          depth: int):
    """Loss on the final hop of a same-level, fully imagined continuation."""
    if depth <= 1:
        return None, None
    result = dynamics_rollout_predictions(model, batch, inp, depth)
    if result is None:
        return None, None
    return dynamics_term(*result, batch, dyn_cfg,
                         weight=dyn_cfg.get("unroll_weight", 1.0))


def policy_dynamics_rollout_term(model, batch: dict, inp: dict, dyn_cfg: dict,
                                 depth: int, pointer_weight: float = 1.0):
    """Train policy on all imagined steps and dynamics on the final transition.

    The final causal forward supplies policy logits at every imagined step;
    earlier roll-in passes and their predicted boards remain detached. Actions
    follow the recording, while policy targets use the original optimal sets.
    Real prefix/root steps, padding and missing optimal sets are not supervised.
    Returns combined loss, dynamics stats, and policy stats (None when off).
    """
    if dyn_cfg.get("policy_unroll_weight", 1.0) == 0:
        loss, stats = dynamics_rollout_term(model, batch, inp, dyn_cfg, depth)
        return loss, stats, None
    if depth <= 1:
        return None, None, None
    result = dynamics_rollout_predictions(model, batch, inp, depth, return_policy=True)
    if result is None:
        return None, None, None
    dynamics, (types, pointers, rows, imagined_mask) = result
    dyn_loss, dyn_stats = dynamics_term(
        *dynamics, batch, dyn_cfg, weight=dyn_cfg.get("unroll_weight", 1.0))
    length = imagined_mask.shape[1]
    opt = batch["opt"].to(types.device, non_blocking=True)[rows, :length]
    oxy = batch["oxy"].to(types.device, non_blocking=True)[rows, :length]
    policy_loss, policy_stats = action_loss(
        types, pointers, opt, oxy, imagined_mask, model.cfg.pointer_grid, pointer_weight)
    return (dyn_loss + dyn_cfg.get("policy_unroll_weight", 1.0) * policy_loss,
            dyn_stats, policy_stats)


def run_epoch(model, loader, device, optimizer=None, scaler=None,
              grad_clip: float = 1.0, log_every: int = 50,
              pointer_weight: float = 1.0, dyn_cfg: dict | None = None,
              return_loss_parts: bool = False, eval_config: dict | None = None):
    """One pass over ``loader``. Train if ``optimizer`` is given, else eval.

    Returns ``(loss, type_acc, click_acc, dyn_metrics)``, the action accuracies
    measured as membership in the step's OPTIMAL action set and averaged over
    SUPERVISED steps only. ``click_acc`` is over steps whose optimal set contains
    a click (``nan`` if the split has none). ``dyn_metrics`` is ``None`` unless
    ``dyn_cfg`` is given and the model has a dynamics head, else the
    ``dyn_ratios`` dict for the auxiliary next-board head. With
    ``return_loss_parts``, append independent policy/dynamics losses. An
    ``eval_config`` enables grouped one-step and fixed-depth rollout diagnostics
    under ``dyn_metrics['evaluation']`` during validation only."""
    train = optimizer is not None
    model.train(train)
    grid = model.cfg.pointer_grid
    use_dyn = dyn_cfg is not None and getattr(model.cfg, "dynamics", False)
    tot_loss = tot_type_correct = tot_count = 0
    tot_click_correct = tot_clicks = 0
    dyn_acc = {k: 0 for k in DYN_SUM_KEYS}
    policy_total = dynamics_total = dynamics_count = 0
    seed = (eval_config or {}).get("seed", 0)
    decode_generator = None if train else torch.Generator(device=device).manual_seed(seed)
    detailed = use_dyn and not train and eval_config is not None
    if detailed:
        def accumulator(raw_scores=True):
            return DynamicsEvaluation(eval_config.get("game_groups"), fog=dyn_cfg["fog"],
                                      raw_scores=raw_scores)
        one_step = accumulator()
        threshold = configured_change_threshold(model.cfg)
        one_step_gated = accumulator(False) if threshold >= 0 else None
        depths = eval_config.get("rollout_depths", (1, 2, 4))
        rollouts = {k: accumulator() for k in depths}
        rollouts_gated = {k: accumulator(False) for k in depths} if threshold >= 0 else {}
        rollout_generators = {k: torch.Generator(device=device).manual_seed(seed + 1000 + k)
                              for k in depths}
        rollout_batches = rollout_batch_indices(len(loader), eval_config.get("rollout_max_batches", 32))

        def observe(acc, colors, decoded, batch, prediction=None):
            bi, li = decoded[:, 0].cpu(), decoded[:, 1].cpu()
            games = batch.get("game_id", ["unknown"] * len(batch["mask"]))
            acc.update(colors, batch["dyn_in"][bi, li], batch["dyn_tgt"][bi, li],
                       [games[i] for i in bi.tolist()], prediction=prediction)
    t0 = time.time()

    for it, batch in enumerate(loader):
        inp = model_inputs(batch, device)
        opt_t = batch["opt"].to(device, non_blocking=True)
        oxy_t = batch["oxy"].to(device, non_blocking=True)

        with torch.set_grad_enabled(train), \
                torch.autocast(device_type=device.type,
                               enabled=(device.type == "cuda")):
            if use_dyn:
                dyn_decode = build_dyn_decode(batch, inp["mask"], dyn_cfg, decode_generator)
                type_logits, ptr_logits, col_logits, chg_logits, dec_bl = model(
                    **inp, return_dynamics=True, dyn_decode=dyn_decode)
            else:
                type_logits, ptr_logits = model(**inp)
            loss, stats = action_loss(type_logits, ptr_logits, opt_t, oxy_t,
                                      inp["mask"], grid, pointer_weight)
            policy_total += loss.item() * stats["n_steps"]
            if use_dyn:
                dyn_term, dstats = dynamics_term(col_logits, chg_logits, dec_bl,
                                                 batch, dyn_cfg)
                loss = loss + dyn_term
                for k in DYN_SUM_KEYS:
                    dyn_acc[k] += dstats[k]
                dynamics_total += (dstats["colour_loss"] + dyn_cfg["change_head_weight"]
                                   * dstats["change_loss"]) * dstats["sup_steps"]
                dynamics_count += dstats["sup_steps"]

        if detailed:
            observe(one_step, col_logits, dec_bl, batch)
            if one_step_gated is not None:
                predicted = decode_dynamics_board(col_logits, chg_logits, inp, dec_bl, threshold)
                observe(one_step_gated, col_logits, dec_bl, batch, predicted)
            if it in rollout_batches:
                with torch.no_grad(), torch.autocast(device_type=device.type,
                                                    enabled=device.type == "cuda"):
                    for depth, acc in rollouts.items():
                        result = dynamics_rollout_predictions(
                            model, batch, inp, depth, rollout_generators[depth],
                            return_prediction=threshold >= 0)
                        if result is not None:
                            observe(acc, result[0], result[2], batch)
                            if threshold >= 0:
                                observe(rollouts_gated[depth], result[0], result[2], batch, result[3])

        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

        tot_type_correct += stats["type_correct"]
        tot_count += stats["n_steps"]
        tot_click_correct += stats["click_correct"]
        tot_clicks += stats["n_clicks"]
        tot_loss += loss.item() * stats["n_steps"]

        if train and (it % log_every == 0):
            acc = tot_type_correct / max(1, tot_count)
            print(f"    it {it:4d}/{len(loader)}  loss {loss.item():.4f}  "
                  f"acc {acc:.3f}  ({(time.time()-t0):.0f}s)")

    n = max(1, tot_count)
    click_acc = tot_click_correct / tot_clicks if tot_clicks else float("nan")
    dyn_metrics = dyn_ratios(dyn_acc) if use_dyn else None
    if detailed:
        dyn_metrics["evaluation"] = {
            "one_step": one_step.result(),
            "rollouts": {str(k): v.result() for k, v in rollouts.items()},
            "rollout_batch_indices": sorted(rollout_batches), "seed": seed,
            "change_threshold": threshold,
            "rollout_decoder": "change_gate" if threshold >= 0 else "color_argmax",
            "raw_metrics": "raw final color logits on configured rollout inputs"}
        if one_step_gated is not None:
            dyn_metrics["evaluation"].update(
                one_step_gated=one_step_gated.result(),
                rollouts_gated={str(k): v.result() for k, v in rollouts_gated.items()})
    result = (tot_loss / n, tot_type_correct / n, click_acc, dyn_metrics)
    if return_loss_parts:
        return (*result, {"policy_loss": policy_total / tot_count if tot_count else float("nan"),
                          "dynamics_loss": dynamics_total / dynamics_count
                          if dynamics_count else None})
    return result


def dilate_errors(err: np.ndarray, radius: int) -> np.ndarray:
    """Spread a 0/1 per-step error mask over a +-``radius`` neighbourhood.

    Each mistake marks itself and up to ``radius`` steps on either side, so
    error-driven resampling oversamples not just the exact off-policy state but
    the states around it (the *neighbourhood*). ``radius <= 0`` is a no-op copy.
    Returns a float array in {0, 1} the same length as ``err``.
    """
    err = err.astype(np.float32)
    if radius <= 0:
        return err.copy()
    L = err.shape[0]
    out = np.zeros(L, dtype=np.float32)
    for t in np.nonzero(err > 0)[0]:
        out[max(0, t - radius): min(L, t + radius + 1)] = 1.0
    return out


@torch.no_grad()
def scan_errors(model, loader, device, grid: int, radius: int):
    """Teacher-force ``model`` over ``loader`` and return the dilated error
    profile plus the overall teacher-forced error rate.

    ``loader`` must yield single-level items tagged with ``fpath``/``li`` (build
    it over a ``return_index=True`` dataset with suffix/multi-level OFF). A step
    counts as an error when the predicted action type is NOT in that step's
    optimal set, or -- when the optimal set contains a click -- when the
    predicted pointer bin lands on none of its click cells. Unsupervised steps
    (no optimal set) are never errors. Returns ``(profile, err_rate)`` where
    ``profile[(fpath, li)]`` is the per-step mask dilated by ``radius``.
    """
    model.eval()
    profile: dict[tuple[str, int], np.ndarray] = {}
    n_steps = n_err = 0
    for batch in tqdm(loader, desc="scan", leave=False):
        inp = model_inputs(batch, device)
        mask = inp["mask"]
        opt_t = batch["opt"].to(device, non_blocking=True)
        oxy_t = batch["oxy"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type,
                            enabled=(device.type == "cuda")):
            type_logits, ptr_logits = model(**inp)
        is_opt = opt_t >= 0                                       # (B,L,K)
        tgt_mask = is_opt.any(-1) & mask                          # supervised
        pred = type_logits.argmax(-1)                             # (B,L)
        wrong = ~((opt_t == pred.unsqueeze(-1)) & is_opt).any(-1)
        opt_click = is_opt & (opt_t == CLICK_INDEX)               # (B,L,K)
        ptr_pred = ptr_logits.argmax(-1)                          # (B,L)
        ptr_hit = ((coord_to_bin_flat(oxy_t, grid) == ptr_pred.unsqueeze(-1))
                   & opt_click).any(-1)
        click_wrong = opt_click.any(-1) & ~ptr_hit
        step_err = ((wrong | click_wrong) & tgt_mask
                    ).to(torch.uint8).cpu().numpy()
        valid = mask.to(torch.int64).cpu().numpy()
        for i, (fpath, li) in enumerate(zip(batch["fpath"], batch["li"])):
            L = int(valid[i].sum())
            raw = step_err[i, :L]
            profile[(fpath, li)] = dilate_errors(raw, radius)
            n_steps += L
            n_err += int(raw.sum())
    err_rate = n_err / max(1, n_steps)
    return profile, err_rate


def warm_start_state_dict(model: nn.Module, sd: dict,
                          allowed_missing_prefixes: tuple[str, ...] | None = None) -> tuple[int, int]:
    """Load every tensor of ``sd`` that matches ``model`` by name AND shape;
    leave the rest of ``model`` at its init. Returns ``(adopted, held)`` --
    tensors copied in, and model tensors left untouched.

    Used to continue a policy-only checkpoint as a policy+dynamics model: the
    trunk (encoder, transformer, position tables) and the policy heads
    (``action_io``, ``readout``) copy straight over; the dynamics head
    (``dyn_readout`` / ``dyn_norm`` / ``board_decoder``) and the added 4th row of
    ``type_embed.weight`` have no counterpart in ``sd`` and stay random.

    A name/shape mismatch that is NOT one of those expected dynamics-only
    additions is fatal -- that means the checkpoint genuinely does not fit this
    architecture, which ``--resume`` must not paper over.
    ``allowed_missing_prefixes`` explicitly overrides the permitted additions
    when upgrading other heads, such as V2's optional spatial pointer.
    """
    msd = model.state_dict()
    ADDED = getattr(model, "_dynamics_state_prefixes",
                    ("dyn_readout", "dyn_norm.", "board_decoder."))
    if allowed_missing_prefixes is not None:
        ADDED = allowed_missing_prefixes
    to_load: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if k not in msd:
            sys.exit(f"[train] warm-start: checkpoint tensor {k!r} has no slot "
                     f"in this model -- the checkpoint does not fit.")
        if msd[k].shape == v.shape:
            to_load[k] = v
        elif k == "type_embed.weight" and v.shape[0] < msd[k].shape[0] \
                and v.shape[1:] == msd[k].shape[1:]:
            # 3-row (patch/readout/action) table -> first rows of the 4-row one;
            # the dyn-readout row keeps its init.
            buf = msd[k].clone()
            buf[:v.shape[0]] = v
            to_load[k] = buf
        else:
            sys.exit(f"[train] warm-start: {k!r} shape {tuple(v.shape)} != model "
                     f"{tuple(msd[k].shape)} -- the checkpoint does not fit.")
    held = [k for k in msd if k not in to_load]
    unexpected = [k for k in held if not k.startswith(ADDED)]
    if unexpected:
        sys.exit(f"[train] warm-start: model tensors {unexpected} are missing "
                 f"from the checkpoint and are not an explicitly allowed new head -- "
                 f"refusing to start them from init.")
    model.load_state_dict(to_load, strict=False)
    return len(to_load), len(held)


def train(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    architecture = getattr(args, "architecture", "packed_v1")
    config_class, model_class = policy_classes(architecture)

    # Configuration always comes from current defaults / CLI, including resume.
    # Saved metadata is used only to check compatibility and detect a policy-only
    # warm start; incompatible weight shapes still fail when loaded below.
    cfg = config_class(**{f.name: getattr(args, f.name)
                          for f in fields(config_class) if hasattr(args, f.name)})
    resume_ckpt = None
    warm_start_dynamics = False
    warm_start_pointer = False
    if getattr(args, "resume", None):
        if not os.path.exists(args.resume):
            sys.exit(f"[train] --resume checkpoint not found: {args.resume}")
        resume_ckpt = torch.load(args.resume, map_location="cpu",
                                 weights_only=False)
        saved_architecture = checkpoint_architecture(resume_ckpt)
        if saved_architecture != architecture:
            sys.exit(f"[train] cannot resume {saved_architecture} weights with "
                     f"the {architecture} trainer. Use the matching training "
                     "entry point, or train the new architecture from scratch.")
        warm_start_dynamics = cfg.dynamics and not resume_ckpt["cfg"].get("dynamics", False)
        warm_start_pointer = (getattr(cfg, "spatial_pointer", False)
                              and not resume_ckpt["cfg"].get("spatial_pointer", False))
        print(f"[train] resuming from {args.resume} "
              f"(config from current defaults/CLI"
              + ("; ADDING a fresh dynamics head to a policy-only checkpoint)"
                 if warm_start_dynamics else ")")
              + ("; adding a zero-initialized spatial pointer head" if warm_start_pointer else ""))

    if BOARD_W % cfg.pointer_grid or BOARD_H % cfg.pointer_grid:
        sys.exit(f"[train] --pointer-grid {cfg.pointer_grid} must divide "
                 f"{BOARD_W} (board size) so click pixels bin cleanly.")

    skip_phases = tuple(p for p in args.skip_target_phases.split(",") if p)
    unknown = [p for p in skip_phases if p not in PHASE_ID]
    if unknown:
        sys.exit(f"[train] --skip-target-phases: unknown phase(s) {unknown}; "
                 f"known phases are {list(PHASES)}")
    if skip_phases:
        print(f"[train] phases kept in context but NOT supervised: "
              f"{list(skip_phases)}")

    # Auxiliary dynamics head config (see dynamics_loss). `dyn_cfg` is None when
    # off, so every downstream call is a plain policy step.
    dyn_cfg = None
    if cfg.dynamics:
        dyn_cfg = dict(weight=args.dynamics_weight,
                       fog=args.dynamics_fog_color,
                       changed_weight=args.dynamics_changed_weight,
                       fog_reveal_weight=args.dynamics_fog_reveal_weight,
                       change_head_weight=args.dynamics_change_head_weight,
                       max_steps=args.dynamics_max_steps,
                       unroll_max_k=args.dynamics_unroll_max_k,
                       unroll_ramp_epochs=args.dynamics_unroll_ramp_epochs,
                       unroll_every=args.dynamics_unroll_every,
                       unroll_weight=args.dynamics_unroll_weight,
                       policy_unroll_weight=args.policy_unroll_weight)
        print(f"[train] dynamics head ON: {dyn_cfg}")

    # Train / val split. By default a (game, level) HOLDOUT: every seed of a
    # per-game random subset of levels is val-only, so val measures solving an
    # UNSEEN level of a known game (what the engine's auto-advance hits) rather
    # than a re-seeded copy of a trained level -- different seeds of one level
    # are near-identical on most games. Whole games are never held out here (that
    # is the separate OOD test set). `--no-level-holdout` restores the legacy
    # random-`--val-frac`-of-episode-files split. The manifest is persisted so
    # the split stays stable as the corpus grows.
    holdout = None
    if args.level_holdout:
        holdout = load_or_build_level_holdout(
            args.cache_dir, args.seed, args.level_val_frac,
            args.level_val_min_levels, args.level_val_seeds)

    # Suffix + multi-level augmentation widen the training distribution on TRAIN
    # only. The frame cap matches the currently configured model's embeddings.
    ds_kw = dict(val_frac=args.val_frac, max_states=cfg.max_states,
                 seed=args.seed, max_frames=cfg.max_frames,
                 skip_target_phases=skip_phases,
                 taken_action_fallback=not args.no_taken_action_fallback,
                 holdout=holdout)
    train_ds = LevelSequenceDataset(args.cache_dir, "train",
                                    suffix_augment=not args.no_suffix_aug,
                                    multi_level=not args.no_multi_level,
                                    action_shuffle=not args.no_action_shuffle,
                                    max_levels=args.max_levels,
                                    resample_alpha=args.resample_alpha
                                    if not args.no_dagger else 0.0,
                                    dynamics=cfg.dynamics, **ds_kw)
    # `val_ds`: the held-out level ALONE, from step 0, no cross-level context --
    # a clean "can it solve this unseen level cold" number. (Legacy split: a
    # single full non-held level.) Deterministic across epochs; reported as
    # `val`.
    val_ds = LevelSequenceDataset(args.cache_dir, "val",
                                  suffix_augment=False, multi_level=False,
                                  dynamics=cfg.dynamics, **ds_kw)
    # `val_ml_ds`: DEPLOYMENT-FAITHFUL. Feeds the episode's earlier NON-held
    # levels (same seed) as in-context history, then the held-out level last as
    # the only supervised segment -- exactly what the live roll-out sees when it
    # auto-advances into a level it has never trained on. This is the checkpoint-
    # selection metric (`--select-on multi`, the default); reported as `val-ml`.
    # (Legacy split: a random deterministic subset of non-held levels, pinned by
    # `sampling_seed`.) `suffix_augment` stays off so items start at a level's
    # true beginning.
    val_ml_ds = LevelSequenceDataset(args.cache_dir, "val",
                                     suffix_augment=False, multi_level=True,
                                     max_levels=args.max_levels,
                                     sampling_seed=args.seed,
                                     dynamics=cfg.dynamics, **ds_kw)
    game_groups = load_game_groups(args.eval_game_groups)
    validation = validation_context(args, cfg, val_ds, holdout, dyn_cfg, game_groups)
    eval_config = {"seed": args.seed, "game_groups": game_groups,
                   "rollout_depths": args.eval_rollout_depths,
                   "rollout_max_batches": args.eval_rollout_max_batches}

    def make_train_loader(sampler=None):
        """Build the train DataLoader. With a sampler (error-driven resampling)
        `shuffle` must be off; without one we shuffle. Rebuilt on every resample
        so freshly-forked workers pick up the updated error profile even under
        `persistent_workers`."""
        return DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                          shuffle=(sampler is None),
                          num_workers=args.num_workers, collate_fn=collate,
                          pin_memory=(device.type == "cuda"), drop_last=True,
                          persistent_workers=args.num_workers > 0)

    train_loader = make_train_loader()
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate,
                            pin_memory=(device.type == "cuda"))
    val_ml_loader = DataLoader(val_ml_ds, batch_size=args.batch_size,
                               shuffle=False, num_workers=args.num_workers,
                               collate_fn=collate,
                               pin_memory=(device.type == "cuda"))

    # Error-driven resampling: a single-level, augmentation-off view of the TRAIN
    # split that a periodic teacher-forced scan runs over to locate the states
    # the policy gets wrong (see scan_errors). Built only when enabled.
    scan_loader = None
    if not args.no_dagger:
        scan_ds = LevelSequenceDataset(args.cache_dir, "train",
                                       suffix_augment=False, multi_level=False,
                                       return_index=True, **ds_kw)
        scan_loader = DataLoader(scan_ds, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.num_workers,
                                 collate_fn=collate,
                                 pin_memory=(device.type == "cuda"))

    model = model_class(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] model params: {n_params/1e6:.2f}M  device: {device}")
    if cfg.dynamics:
        threshold = configured_change_threshold(cfg)
        print(f"[train] imagined board decoder: "
              + (f"change gate {threshold:g}" if threshold >= 0 else "raw color argmax"))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay,
                                  betas=(0.9, 0.95))
    # Training runs indefinitely (no fixed horizon), so a cosine decay-to-zero
    # doesn't apply: use warmup-then-constant LR.
    warmup = max(1, args.warmup_steps)

    def lr_lambda(step):
        return min(1.0, (step + 1) / warmup)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

    os.makedirs(args.out_dir, exist_ok=True)
    best, selection_reset = restore_selection(
        None if warm_start_dynamics else resume_ckpt, validation)
    if selection_reset:
        print(f"[train] {selection_reset}")
    global_step = 0
    start_epoch = 0

    if resume_ckpt is not None and (warm_start_dynamics or warm_start_pointer):
        # Copy existing weights and initialize only the newly enabled heads.
        # A pointer-only upgrade also migrates optimizer state below. Adding
        # dynamics retains its separate, existing warm-start behavior.
        allowed = ()
        if warm_start_dynamics:
            allowed += getattr(model, "_dynamics_state_prefixes",
                               ("dyn_readout", "dyn_norm.", "board_decoder."))
        if warm_start_pointer:
            allowed += ("spatial_pointer.",)
        adopted, held = warm_start_state_dict(model, resume_ckpt["model"], allowed)
        print(f"[train] warm-started {adopted} tensors from the "
              f"checkpoint; {held} new-head tensors kept at init.")
        if warm_start_dynamics:
            print("[train] dynamics-head upgrade: fresh optimizer/schedule; epoch clock reset to 0.")
    elif resume_ckpt is not None:
        try:
            model.load_state_dict(resume_ckpt["model"])
        except RuntimeError as e:
            sys.exit(
                f"[train] {args.resume} does not fit this model: {e}\n"
                "[train] Resume uses current defaults/CLI. Specify model-size "
                "flags compatible with the saved weights, or train from scratch.")
    if resume_ckpt is not None and not warm_start_dynamics:
        # Optimizer/scheduler/step state were saved from this script onward; older
        # checkpoints (weights only) resume as a warm start with a fresh schedule.
        if "optimizer" in resume_ckpt:
            with keep_training_config(optimizer, scheduler):
                if warm_start_pointer:
                    optimizer.load_state_dict(optimizer_state_with_spatial_pointer(
                        optimizer, model, resume_ckpt))
                    print(f"[train] preserved Adam state for {len(optimizer.state)} existing "
                          "parameters; new spatial pointer starts with fresh optimizer state")
                else:
                    optimizer.load_state_dict(resume_ckpt["optimizer"])
                scheduler.load_state_dict(resume_ckpt["scheduler"])
            scaler.load_state_dict(resume_ckpt["scaler"])
            global_step = resume_ckpt.get("global_step", 0)
        else:
            print("[train] checkpoint has no optimizer state -- warm-starting "
                  "weights with a fresh optimizer/schedule")
        start_epoch = resume_ckpt.get("epoch", -1) + 1
        print(f"[train] resumed at epoch {start_epoch} "
              f"(best_policy {best['policy']:.4f}, best_dynamics {best['dynamics']:.4f})")

    # Run indefinitely; stop with Ctrl-C. An optional --epochs > 0 caps the run.
    max_epochs = args.epochs if args.epochs and args.epochs > 0 else None
    if max_epochs is not None and start_epoch >= max_epochs:
        print(f"[train] nothing to do: start epoch {start_epoch} >= "
              f"--epochs {max_epochs}. Increase --epochs to continue.")
        return
    print("[train] training indefinitely -- press Ctrl-C to stop "
          + (f"(or until epoch {max_epochs})" if max_epochs else "")
          + f". Checkpoints -> {args.out_dir}/best_policy.pt, best_dynamics.pt "
            "(best.pt aliases best_policy.pt)")

    epoch = start_epoch
    while max_epochs is None or epoch < max_epochs:
        # Error-driven resampling (poor man's DAgger): after `resample_warmup`
        # plain epochs, and every `resample_every` epochs thereafter, teacher-
        # force the current policy over the corpus, find the states it gets
        # wrong, and reweight the next rounds toward those states and their
        # neighbourhoods (see scan_errors + the dataset's suffix-start bias /
        # item weights). A pure sampler change against the existing corpus.
        if (scan_loader is not None and epoch >= args.resample_warmup
                and (epoch - args.resample_warmup) % max(1, args.resample_every)
                == 0):
            print(f"[dagger] epoch {epoch+1}: teacher-forcing the policy over "
                  f"the corpus to locate error states...")
            profile, err_rate = scan_errors(model, scan_loader, device,
                                            cfg.pointer_grid,
                                            args.resample_neighbourhood)
            train_ds.set_error_profile(profile)
            weights = train_ds.item_sampling_weights()
            n_up = int((weights > 1.0 + 1e-9).sum())
            sampler = WeightedRandomSampler(weights, num_samples=len(train_ds),
                                            replacement=True)
            train_loader = make_train_loader(sampler)
            print(f"[dagger] epoch {epoch+1}: teacher-forced err "
                  f"{err_rate:.3f}; upweighted {n_up}/{len(train_ds)} levels "
                  f"(alpha {args.resample_alpha}, radius "
                  f"{args.resample_neighbourhood})")

        # Rollout curriculum depth for this epoch (see module docstring); stays
        # at 1 (no-op) unless --dynamics-unroll-max-k > 1.
        roll_depth = 1
        if dyn_cfg is not None:
            roll_depth = min(dyn_cfg["unroll_max_k"],
                             1 + epoch // max(1, dyn_cfg["unroll_ramp_epochs"]))
            if roll_depth > 1:
                print(f"[train] epoch {epoch+1}: dynamics rollout depth "
                      f"{roll_depth}/{dyn_cfg['unroll_max_k']}")

        # Manual epoch loop so we can step the LR schedule per batch.
        model.train()
        t0 = time.time()
        run_loss = run_corr = run_cnt = 0
        run_click_corr = run_clicks = 0
        dyn_run = {k: 0 for k in DYN_SUM_KEYS}
        roll_run = {k: 0 for k in DYN_SUM_KEYS}
        roll_policy_run = {k: 0 for k in ("type_correct", "n_steps", "click_correct", "n_clicks")}
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}", leave=False)
        for it, batch in enumerate(pbar):
            inp = model_inputs(batch, device)
            opt_t = batch["opt"].to(device, non_blocking=True)
            oxy_t = batch["oxy"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type,
                                enabled=(device.type == "cuda")):
                if dyn_cfg is not None:
                    dyn_decode = build_dyn_decode(batch, inp["mask"], dyn_cfg)
                    (type_logits, ptr_logits, col_logits, chg_logits,
                     dec_bl) = model(**inp, return_dynamics=True,
                                     dyn_decode=dyn_decode)
                else:
                    type_logits, ptr_logits = model(**inp)
                loss, stats = action_loss(type_logits, ptr_logits, opt_t, oxy_t,
                                          inp["mask"], cfg.pointer_grid,
                                          args.pointer_weight)
                if dyn_cfg is not None:
                    dyn_term, dstats = dynamics_term(col_logits, chg_logits,
                                                     dec_bl, batch, dyn_cfg)
                    loss = loss + dyn_term
                    for k in DYN_SUM_KEYS:
                        dyn_run[k] += dstats[k]
                    if roll_depth > 1 and it % dyn_cfg["unroll_every"] == 0:
                        roll_term, rstats, pstats = policy_dynamics_rollout_term(
                            model, batch, inp, dyn_cfg, roll_depth, args.pointer_weight)
                        if roll_term is not None:
                            loss = loss + roll_term
                            for k in DYN_SUM_KEYS:
                                roll_run[k] += rstats[k]
                            if pstats is not None:
                                for k in roll_policy_run:
                                    roll_policy_run[k] += pstats[k]
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1

            run_corr += stats["type_correct"]
            run_cnt += stats["n_steps"]
            run_click_corr += stats["click_correct"]
            run_clicks += stats["n_clicks"]
            run_loss += loss.item() * stats["n_steps"]
            post = dict(loss=f"{run_loss / max(1, run_cnt):.4f}",
                        acc=f"{run_corr / max(1, run_cnt):.3f}",
                        click=f"{run_click_corr / max(1, run_clicks):.3f}",
                        lr=f"{scheduler.get_last_lr()[0]:.2e}")
            if dyn_cfg is not None:
                post["dyn"] = f"{dyn_ratios(dyn_run)['cell']:.3f}"
                if roll_depth > 1:
                    post["roll"] = f"k{roll_depth}:{dyn_ratios(roll_run)['cell']:.3f}"
            pbar.set_postfix(**post)

        tr_loss, tr_acc = run_loss / max(1, run_cnt), run_corr / max(1, run_cnt)
        tr_click = run_click_corr / run_clicks if run_clicks else float("nan")
        tr_dyn = dyn_ratios(dyn_run) if dyn_cfg is not None else None
        tr_roll = dyn_ratios(roll_run) if roll_depth > 1 else None
        val_loss, val_acc, val_click, val_dyn, val_parts = run_epoch(
            model, val_loader, device, pointer_weight=args.pointer_weight,
            dyn_cfg=dyn_cfg, return_loss_parts=True, eval_config=eval_config)
        vml_loss, vml_acc, vml_click, vml_dyn, vml_parts = run_epoch(
            model, val_ml_loader, device, pointer_weight=args.pointer_weight,
            dyn_cfg=dyn_cfg, return_loss_parts=True, eval_config=eval_config)
        dyn_line = ""
        if dyn_cfg is not None:
            dyn_line = (f" | dyn train[{fmt_dyn(tr_dyn)}] "
                        f"val[{fmt_dyn(val_dyn)}] val-ml[{fmt_dyn(vml_dyn)}]")
            if roll_depth > 1:
                dyn_line += f" | rollout(k={roll_depth}) train[{fmt_dyn(tr_roll)}]"
                rp = roll_policy_run
                if rp["n_steps"]:
                    dyn_line += (f" policy acc {rp['type_correct'] / rp['n_steps']:.3f}"
                                 f" click {rp['click_correct'] / rp['n_clicks'] if rp['n_clicks'] else float('nan'):.3f}"
                                 f" steps {rp['n_steps']}")
        print(f"[epoch {epoch+1}] train loss {tr_loss:.4f} "
              f"acc {tr_acc:.3f} click {tr_click:.3f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.3f} click {val_click:.3f} | "
              f"val-ml loss {vml_loss:.4f} acc {vml_acc:.3f} "
              f"click {vml_click:.3f} | "
              f"lr {scheduler.get_last_lr()[0]:.2e} | {time.time()-t0:.0f}s"
              + dyn_line)
        for split, parts, metrics in (("val", val_parts, val_dyn),
                                       ("val-ml", vml_parts, vml_dyn)):
            print(f"  {split} policy_loss={parts['policy_loss']:.6f} "
                  f"dynamics_loss={parts['dynamics_loss']}")
            if metrics is not None:
                print(format_evaluation(split, metrics["evaluation"]))

        # Independent objectives on the same selected validation view. Dynamics
        # loss excludes its outer training weight and the changing curriculum.
        selected = update_selection(best, vml_parts if args.select_on == "multi" else val_parts)
        ckpt = {"model": model.state_dict(), "cfg": cfg.__dict__,
                "architecture": architecture,
                "epoch": epoch, "val_loss": val_loss, "val_acc": val_acc,
                "val_click_acc": val_click,
                "val_ml_loss": vml_loss, "val_ml_acc": vml_acc,
                "val_ml_click_acc": vml_click, "select_on": args.select_on,
                "optimizer": optimizer.state_dict(),
                "optimizer_param_names": [name for name, _ in model.named_parameters()],
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "global_step": global_step, "best_val": best["policy"],
                "best_policy_loss": best["policy"], "best_dynamics_loss": best["dynamics"],
                "val_policy_loss": val_parts["policy_loss"],
                "val_dynamics_loss": val_parts["dynamics_loss"],
                "val_ml_policy_loss": vml_parts["policy_loss"],
                "val_ml_dynamics_loss": vml_parts["dynamics_loss"],
                "validation_context": validation, "selection_reset_reason": selection_reset}
        if dyn_cfg is not None:
            ckpt["dyn_cfg"] = dyn_cfg
            ckpt["val_dyn"] = val_dyn
            ckpt["val_ml_dyn"] = vml_dyn
            ckpt["train_rollout_policy"] = roll_policy_run
        save_training_checkpoints(ckpt, args.out_dir, selected)
        for name in selected:
            print(f"    saved best_{name}.pt ({args.select_on} {name}_loss {best[name]:.6f})")
        epoch += 1

    print(f"\n[train] done. best policy_loss {best['policy']:.4f}, "
          f"best dynamics_loss {best['dynamics']:.4f}. "
          f"checkpoints in {args.out_dir}")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def _rollout_depths(value):
    try:
        depths = tuple(sorted({int(v) for v in value.split(",") if v.strip()}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("rollout depths must be comma-separated integers") from exc
    if any(d < 1 for d in depths):
        raise argparse.ArgumentTypeError("rollout depths must be positive")
    return depths


def build_parser(architecture="packed_v1", description=None) -> argparse.ArgumentParser:
    config_class, _ = policy_classes(architecture)
    p = argparse.ArgumentParser(description=description or __doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.set_defaults(architecture=architecture)
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("preprocess", help="JSON -> compact .npz cache")
    pp.add_argument("--data-dir", default=DATA_DIR)
    pp.add_argument("--cache-dir", default=CACHE_DIR)
    pp.add_argument("--episodes-per-game", type=int, default=-1,
                    help="cap episodes per game (None-like: use -1 for all)")
    pp.add_argument("--overwrite", action="store_true")

    def add_train_args(tp):
        tp.add_argument("--cache-dir", default=CACHE_DIR)
        tp.add_argument("--out-dir", default="runs/policy_C")
        tp.add_argument("--resume", type=str, default=None,
                        help="path to a checkpoint (e.g. runs/policy/last.pt) to "
                             "resume training from; config uses current defaults/CLI, "
                             "so model-size flags must match the saved weights")
        tp.add_argument("--epochs", type=int, default=0,
                        help="0 = train indefinitely (Ctrl-C to stop); a positive "
                             "value caps the number of epochs")
        tp.add_argument("--batch-size", type=int, default=16)
        tp.add_argument("--lr", type=float, default=3e-4)
        tp.add_argument("--warmup-steps", type=int, default=300,
                        help="linear LR warmup steps, then constant LR")
        tp.add_argument("--weight-decay", type=float, default=0.05)
        tp.add_argument("--grad-clip", type=float, default=1.0)
        tp.add_argument("--max-states", type=int, default=100,
                        help="FIFO cap on retained steps. V1 uses patch-grid "
                             "squared tokens per frame; V2 uses temporal-tokens. "
                             "Lower this cap to reduce attention memory")
        tp.add_argument("--max-frames", type=int, default=4,
                        help="cap on frames per action's animation span; longer "
                             "spans are subsampled keeping both endpoints (so "
                             "the settled frame always survives). 1 = settled "
                             "frame only, i.e. the old single-frame behaviour")
        tp.add_argument("--skip-target-phases", type=str, default="",
                        help="comma-separated recording phases whose steps stay "
                             "in context but are NOT supervised, e.g. "
                             "'explore,burst'. Empty (default) trains on every "
                             "step that has an optimal-action set, including the "
                             f"relabelled off-policy ones. Known: {list(PHASES)}")
        tp.add_argument("--select-on", choices=("multi", "single"), default="multi",
                        help="validation view for independent best_policy.pt "
                             "(policy loss) and best_dynamics.pt (one-step dynamics "
                             "loss). best.pt aliases best_policy.pt. 'multi' "
                             "(default) uses the deployment-shaped multi-level "
                             "split -- levels concatenated into one context, as "
                             "the live roll-out sees them on auto-advance. "
                             "'single' uses the legacy single-full-level split, "
                             "which is EASIER than the training distribution "
                             "(val acc can exceed train acc) and never measures "
                             "cross-level transfer. Both are always reported")
        tp.add_argument("--eval-rollout-depths", type=_rollout_depths, default=(1, 2, 4),
                        help="fixed validation rollout depths, independent of the training "
                             "curriculum (default: 1,2,4; empty string disables rollouts)")
        tp.add_argument("--eval-rollout-max-batches", type=int, default=32,
                        help="evaluate rollouts on this many evenly spaced batches per "
                             "validation view; 0 evaluates every batch (default: 32)")
        tp.add_argument("--eval-game-groups", default=None,
                        help="optional JSON mapping corpus game IDs to deterministic, "
                             "uncertain, or unknown. Overrides conservative built-in rules; "
                             "unclassified games remain visible in unknown")
        tp.add_argument("--no-taken-action-fallback", action="store_true",
                        help="do NOT supervise an oracle-less EXPERT step with "
                             "its taken action. The fallback is ON by default: "
                             "without it, generators whose expert is a step-wise "
                             "adaptive policy rather than a plan (re86, "
                             "mm01..mm05) record optimal=None on EVERY step and "
                             "so train on nothing at all, which is how v2 lost "
                             "games v1 had learned")
        tp.add_argument("--no-suffix-aug", action="store_true",
                        help="disable suffix augmentation (train on a random "
                             "trajectory suffix each fetch); on by default")
        tp.add_argument("--no-multi-level", action="store_true",
                        help="disable multi-level context augmentation (train on a "
                             "random subset of the episode's levels, in random "
                             "order); on by default")
        tp.add_argument("--no-action-shuffle", action="store_true",
                        help="disable action-identity shuffle (relabel action ids "
                             "1..5 through a fresh random permutation, held fixed "
                             "per item, so the policy can't memorize a fixed "
                             "id->effect mapping and must read it off the "
                             "in-context history instead); on by default")
        tp.add_argument("--max-levels", type=int, default=8,
                        help="multi-level cap: at most this many levels are "
                             "concatenated into one training context")
        tp.add_argument("--no-dagger", action="store_true",
                        help="disable error-driven resampling (poor man's "
                             "DAgger): by default the policy is periodically "
                             "teacher-forced over the corpus and the states it "
                             "gets wrong (and their neighbourhoods) are "
                             "oversampled in later rounds; pass this to turn it off")
        tp.add_argument("--resample-warmup", type=int, default=1,
                        help="plain epochs before the first error scan "
                             "(needs a partly-trained policy to scan)")
        tp.add_argument("--resample-every", type=int, default=3,
                        help="re-scan errors and reweight every this-many epochs "
                             "(each scan is one extra forward pass over TRAIN)")
        tp.add_argument("--resample-alpha", type=float, default=3.0,
                        help="resampling strength: level weight is 1+alpha*mean "
                             "error, suffix start ~ 1+alpha*error; 0 disables")
        tp.add_argument("--resample-neighbourhood", type=int, default=3,
                        help="+-this-many steps around each error are also "
                             "oversampled (the mistake's neighbourhood)")
        tp.add_argument("--pointer-grid", type=int, default=POINTER_GRID,
                        help="click-distribution resolution GxG (must divide 64)")
        tp.add_argument("--patch-grid", type=int, default=PATCH_GRID,
                        help="board -> GxG spatial patch features. The original "
                             "model sends all G*G tokens per frame through "
                             "history attention; V2 pools them to --temporal-tokens "
                             "while retaining the full grid for dynamics decoding")
        tp.add_argument("--pointer-weight", type=float, default=1.0,
                        help="weight of the pointer (click-location) loss term")
        # --- Auxiliary dynamics head (see module docstring) ---
        tp.add_argument("--no-dynamics", dest="dynamics", action="store_false",
                        help="train the POLICY head only. By DEFAULT the "
                             "auxiliary next-board dynamics head trains jointly "
                             "with the policy; pass this to turn it off (the "
                             "checkpoint is then a plain policy checkpoint and "
                             "loads unchanged in policy_runtime.py)")
        tp.set_defaults(dynamics=True)
        tp.add_argument("--dynamics", dest="dynamics", action="store_true",
                        help="enable joint dynamics training (the default)")
        tp.add_argument("--dynamics-weight", type=float, default=1.0,
                        help="scale on the dynamics loss added to the policy "
                             "loss")
        tp.add_argument("--dynamics-fog-color", type=int, default=-1,
                        help="colour a fog-of-war game renders UNEXPLORED cells "
                             "as (e.g. 5 for dark_maze_3). -1 (default) = not a "
                             "PO game: cells fog in both input and target then "
                             "carry no colour loss, and first-time reveals are "
                             "down-weighted to --dynamics-fog-reveal-weight")
        tp.add_argument("--dynamics-changed-weight", type=float, default=5.0,
                        help="per-cell colour-loss weight for cells that change "
                             "between the input and target board (the board is "
                             "mostly static, so unweighted CE ignores them)")
        tp.add_argument("--dynamics-fog-reveal-weight", type=float, default=0.25,
                        help="per-cell colour-loss weight for cells revealed for "
                             "the first time (fog -> not fog); their contents "
                             "are only partly knowable from context")
        tp.add_argument("--dynamics-change-head-weight", type=float, default=1.0,
                        help="scale on the change-mask BCE head relative to the "
                             "colour CE within the dynamics loss")
        tp.add_argument("--dynamics-change-threshold", type=change_threshold,
                        default=config_class().dynamics_change_threshold,
                        help="copy current pixels unless P(change) exceeds this threshold; "
                             "V2 default .25, V1 default -1 (raw). Saved for planning; "
                             "does not gate supervised losses")
        tp.add_argument("--dynamics-max-steps", type=int, default=16,
                        help="cap on how many steps per training item the 64x64 "
                             "decoder runs on (a uniform random subset when the "
                             "context has more); bounds decoder memory. 0 = "
                             "every valid step (large)")
        tp.add_argument("--dynamics-unroll-max-k", type=int, default=20,
                        help="max autoregressive ROLLOUT depth trained for the "
                             "policy and dynamics heads (see the module docstring's "
                             "'Rollout curriculum'): at depth k it conditions on "
                             "k-1 of its own (configured decoder, detached) predictions "
                             "instead of real frames before being scored again. "
                             "1 = OFF, i.e. only the ordinary single-step term "
                             "runs -- bit-identical to before this flag existed. "
                             "Default 20 with the default --dynamics-unroll-"
                             "ramp-epochs 1 reaches full depth at epoch 20; "
                             "lower either flag to ramp faster")
        tp.add_argument("--dynamics-unroll-ramp-epochs", type=int, default=1,
                        help="epochs per +1 rollout depth on the way to "
                             "--dynamics-unroll-max-k (e.g. 3 -> depth 2 from "
                             "epoch 3, depth 3 from epoch 6, ...)")
        tp.add_argument("--dynamics-unroll-every", type=int, default=4,
                        help="fire the rollout term on 1 of every N training "
                             "batches. Each firing costs depth-1 extra no_grad "
                             "forward passes plus one extra forward+backward, "
                             "so this bounds the average overhead; ignored "
                             "while the curriculum depth is still 1")
        tp.add_argument("--dynamics-unroll-weight", type=float, default=1.0,
                        help="scale on the dynamics rollout loss term, independent of "
                             "--dynamics-weight (which scales the single-step "
                             "term)")
        tp.add_argument("--policy-unroll-weight", type=float, default=1.0,
                        help="scale on optimal-action loss at every imagined input "
                             "step, using the dynamics rollout depth and frequency; "
                             "0 disables policy rollout supervision. Pointer loss "
                             "also uses --pointer-weight (default: 1)")
        tp.add_argument("--d-model", type=int, default=256)
        tp.add_argument("--n-heads", type=int, default=8)
        tp.add_argument("--n-layers", type=int, default=6)
        tp.add_argument("--dropout", type=float, default=0.1)
        tp.add_argument("--val-frac", type=float, default=0.05,
                        help="LEGACY (--no-level-holdout) split only: fraction "
                             "of episode FILES held out for val")
        tp.add_argument("--no-level-holdout", dest="level_holdout",
                        action="store_false",
                        help="use the legacy split (random --val-frac of episode "
                             "FILES) instead of the (game, level) holdout. The "
                             "holdout (default) reserves every seed of a per-game "
                             "random subset of LEVELS for val, so val measures "
                             "solving an unseen level of a known game -- what the "
                             "engine's auto-advance hits -- not a re-seeded copy "
                             "of a trained level. Whole games are never held out "
                             "here (that is the OOD test set)")
        tp.set_defaults(level_holdout=True)
        tp.add_argument("--level-val-frac", type=float, default=0.2,
                        help="(game,level) holdout: fraction of each game's "
                             "distinct levels reserved for val")
        tp.add_argument("--level-val-min-levels", type=int, default=4,
                        help="(game,level) holdout: games with fewer distinct "
                             "levels than this keep ALL levels in train (holding "
                             "one out would gut their coverage) and contribute "
                             "no val levels")
        tp.add_argument("--level-val-seeds", type=int, default=40,
                        help="(game,level) holdout: cap on how many seeds of each "
                             "held level land in val (0 = all ~1000; the default "
                             "keeps val small since seeds of one level are "
                             "near-duplicates). The manifest is persisted to "
                             "<cache-dir>/level_holdout.json -- delete it to "
                             "rebuild after changing these flags")
        tp.add_argument("--num-workers", type=int, default=4)
        tp.add_argument("--log-every", type=int, default=50)
        tp.add_argument("--seed", type=int, default=0)
        if architecture == "spatial_temporal_v2":
            defaults = config_class()
            tp.add_argument("--spatial-pointer", action=argparse.BooleanOptionalAction,
                            default=True,
                            help="history-conditioned pointer over the current spatial grid "
                                 "(V2 default: on); upgrading an old checkpoint preserves "
                                 "initial predictions and existing optimizer/schedule/epoch state")
            tp.add_argument("--spatial-pointer-dim", type=int, default=defaults.spatial_pointer_dim,
                            help="query/key width for the optional spatial pointer (default 64)")
            tp.add_argument("--spatial-layers", type=int, default=defaults.spatial_layers,
                            help="unrestricted per-frame transformer depth")
            tp.add_argument("--decoder-layers", type=int, default=defaults.decoder_layers,
                            help="action-conditioned spatial dynamics decoder depth")
            tp.add_argument("--temporal-tokens", type=int, default=defaults.temporal_tokens,
                            help="learned history summary tokens per frame; full spatial "
                                 "features are also retained for dynamics")
            tp.add_argument("--decode-chunk-size", type=int, default=defaults.decode_chunk_size,
                            help="number of selected transitions decoded together")
            tp.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction,
                            default=defaults.gradient_checkpointing,
                            help="recompute transformer activations during backward "
                                 "to reduce training memory")
            tp.set_defaults(d_model=defaults.d_model, n_layers=defaults.n_layers,
                            n_heads=defaults.n_heads, patch_grid=defaults.patch_grid,
                            max_states=defaults.max_states, batch_size=4,
                            dynamics_max_steps=4, lr=2e-4,
                            out_dir="runs/policy_dynamics_V2")

    tr = sub.add_parser("train", help="train the in-context policy")
    add_train_args(tr)

    al = sub.add_parser("all", help="preprocess then train")
    al.add_argument("--data-dir", default=DATA_DIR)
    al.add_argument("--episodes-per-game", type=int, default=-1,
                    help="cap episodes per game (None-like: use -1 for all)")
    al.add_argument("--overwrite", action="store_true")
    add_train_args(al)
    return p


def main(architecture="packed_v1", description=None) -> None:
    args = build_parser(architecture, description).parse_args()
    if args.cmd in ("preprocess", "all"):
        epg = None if args.episodes_per_game is not None and \
            args.episodes_per_game < 0 else args.episodes_per_game
        preprocess(args.data_dir, args.cache_dir, epg, args.overwrite)
    if args.cmd in ("train", "all"):
        try:
            train(args)
        except KeyboardInterrupt:
            print(f"\n[train] interrupted -- best model kept at "
                  f"{os.path.join(args.out_dir, 'best.pt')}")


if __name__ == "__main__":
    main()
