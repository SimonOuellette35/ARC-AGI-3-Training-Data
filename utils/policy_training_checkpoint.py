"""Validation identity and independent policy/dynamics checkpoint selection."""
import hashlib
import json
import math
import os

import torch

from utils.policy_evaluation import GROUP_RULE_VERSION
from utils.policy_checkpoint import checkpoint_architecture, policy_classes


def optimizer_state_with_spatial_pointer(optimizer, model, checkpoint):
    """Map old Adam state by parameter name, leaving only new pointer state empty.

    Policy training uses one group containing every parameter. Old checkpoints
    lack parameter names; recover their exact registration order by constructing
    the saved architecture on the meta device (no second set of real weights).
    Never align by position in the upgraded model: the pointer is inserted
    before the temporal trunk, so that would shift existing optimizer states.
    """
    saved = checkpoint['optimizer']
    current = optimizer.state_dict()
    if len(saved['param_groups']) != 1 or len(current['param_groups']) != 1:
        raise ValueError('Spatial-pointer optimizer upgrade requires the trainer\'s single parameter group')
    old_names = checkpoint.get('optimizer_param_names')
    if old_names is None:
        config_class, model_class = policy_classes(checkpoint_architecture(checkpoint))
        with torch.device('meta'):
            previous = model_class(config_class(**checkpoint['cfg']))
        old_names = [name for name, _ in previous.named_parameters()]
    old_ids = saved['param_groups'][0]['params']
    if len(old_names) != len(old_ids) or len(set(old_names)) != len(old_names):
        raise ValueError('Checkpoint optimizer parameter names do not match its parameter group')
    if len(set(old_ids)) != len(old_ids) or set(saved['state']) - set(old_ids):
        raise ValueError('Checkpoint optimizer contains unknown or duplicate parameter IDs')
    named = dict(model.named_parameters())
    names_by_object = {id(p): name for name, p in named.items()}
    current_names = [names_by_object[id(p)] for p in optimizer.param_groups[0]['params']]
    new_ids = dict(zip(current_names, current['param_groups'][0]['params']))
    added = set(current_names) - set(old_names)
    if set(old_names) - set(current_names) or any(not n.startswith('spatial_pointer.') for n in added):
        raise ValueError('Optimizer upgrade may only add spatial_pointer parameters')
    migrated = {}
    for name, old_id in zip(old_names, old_ids):
        if name not in checkpoint['model'] or checkpoint['model'][name].shape != named[name].shape:
            raise ValueError(f'Cannot preserve optimizer state for changed parameter {name!r}')
        state = saved['state'].get(old_id)
        if state is None:
            continue  # Unused parameters may not yet have Adam state.
        for key, value in state.items():
            if key != 'step' and torch.is_tensor(value) and value.shape != named[name].shape:
                raise ValueError(f'Optimizer {key} shape does not match {name!r}')
        migrated[new_ids[name]] = state
    return {'state': migrated, 'param_groups': current['param_groups']}


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validation_context(args, cfg, dataset, holdout, dyn_cfg, game_groups):
    """Record actual split membership and evaluation settings, not just CLI intent.

    Relative paths keep the identity portable between machines. The manifest
    records all held puzzles/seeds; the membership hash also catches added or
    removed validation examples with an otherwise unchanged manifest.
    """
    membership = hashlib.sha256()
    for path, li in dataset.index:
        row = [os.path.relpath(path, args.cache_dir), li, dataset.file_lids[path][li]]
        membership.update(json.dumps(row, separators=(",", ":")).encode() + b"\n")
    manifest = None if holdout is None else {
        g: {"held_levels": sorted(v["held_levels"]),
            "val_seeds": sorted(v["val_seeds"]) if v["val_seeds"] is not None else None}
        for g, v in holdout.items()}
    # Also identify earlier levels that may enter a multi-level val context.
    context_levels = hashlib.sha256()
    for path in sorted({p for p, _ in dataset.index}):
        row = [os.path.relpath(path, args.cache_dir), dataset.file_lids[path]]
        context_levels.update(json.dumps(row, separators=(",", ":")).encode() + b"\n")
    split = {"kind": "level_holdout" if holdout is not None else "episode_holdout",
             "seed": args.seed, "episode_val_frac": args.val_frac if holdout is None else None,
             "manifest": manifest, "membership_sha256": membership.hexdigest(),
             "context_levels_sha256": context_levels.hexdigest(), "sequences": len(dataset)}
    settings = {"version": 2, "select_on": args.select_on,
                "policy_metric": "policy_loss", "dynamics_metric": "dynamics_loss",
                "terminal_dynamics_targets": True, "imagined_frames_per_step": 1,
                "max_states": cfg.max_states, "max_frames": cfg.max_frames,
                "pointer_grid": cfg.pointer_grid, "max_levels": args.max_levels,
                "batch_size": args.batch_size, "pointer_weight": args.pointer_weight,
                "skip_target_phases": sorted(dataset.skip_phase_ids.tolist()),
                "taken_action_fallback": dataset.taken_action_fallback,
                "dynamics": None if dyn_cfg is None else {k: dyn_cfg[k] for k in
                    ("fog", "changed_weight", "fog_reveal_weight", "change_head_weight", "max_steps")},
                "rollout_depths": list(args.eval_rollout_depths),
                "dynamics_change_threshold": getattr(cfg, "dynamics_change_threshold", -1),
                "rollout_max_batches": args.eval_rollout_max_batches,
                "game_group_rule_version": GROUP_RULE_VERSION, "game_group_overrides": game_groups}
    context = {"split": split, "settings": settings}
    context["fingerprint"] = _fingerprint(context)
    return context


def restore_selection(checkpoint, context):
    """Legacy/mismatched evaluation metadata cannot supply comparable best losses."""
    fresh = {"policy": float("inf"), "dynamics": float("inf")}
    if checkpoint is None:
        return fresh, None
    previous = checkpoint.get("validation_context")
    if previous is None:
        return fresh, "checkpoint has no validation identity; resetting policy/dynamics selection"
    if previous.get("fingerprint") != context["fingerprint"]:
        return fresh, "validation split or evaluation settings changed; resetting policy/dynamics selection"
    if "best_policy_loss" not in checkpoint or "best_dynamics_loss" not in checkpoint:
        return fresh, "checkpoint lacks separate best losses; resetting policy/dynamics selection"
    return {"policy": checkpoint["best_policy_loss"],
            "dynamics": checkpoint["best_dynamics_loss"]}, None


def update_selection(best, losses):
    """Update independent minima; missing/undefined metrics never select a model."""
    selected = []
    for name in ("policy", "dynamics"):
        value = losses.get(name + "_loss")
        if value is not None and math.isfinite(value) and value < best[name]:
            best[name] = value
            selected.append(name)
    return selected


def save_training_checkpoints(checkpoint, out_dir, selected):
    """All files are resumable; best.pt remains a compatibility alias for policy."""
    paths = ["last.pt"] + [f"best_{name}.pt" for name in selected]
    if "policy" in selected:
        paths.append("best.pt")
    for name in paths:
        path = os.path.join(out_dir, name)
        temporary = path + ".tmp"
        torch.save(checkpoint, temporary)
        os.replace(temporary, path)
