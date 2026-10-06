"""Game-grouped dynamics fidelity and uncertainty diagnostics."""
from collections import defaultdict
import json

import torch


GROUPS = ("deterministic", "uncertain", "unknown")
GROUP_RULE_VERSION = 1


def load_game_groups(path=None):
    """Optional JSON maps exact corpus game IDs to a group; unknown stays visible."""
    if path is None:
        return {}
    with open(path) as fh:
        groups = json.load(fh)
    if not isinstance(groups, dict) or any(
            not isinstance(k, str) or v not in GROUPS for k, v in groups.items()):
        raise ValueError("evaluation game groups must map game IDs to " + ", ".join(GROUPS))
    return groups


def game_group(game, overrides):
    if game in overrides:
        return overrides[game]
    if (game.startswith("minigrid_partial_")
            or (game.startswith("minigrid_") and "dynamic_obstacles" in game)
            or game == "puzzlescript_blind_maze_a1"):
        return "uncertain"
    if (game in {"maze", "sokoban", "puzzlescript_stickyban", "gymgridworlds_empty_10x10"}
            or game.startswith("minigrid_empty_")
            or game.startswith("minigrid_doorkey_") or game == "minigrid_fourrooms"):
        return "deterministic"
    return "unknown"


class DynamicsEvaluation:
    """Streaming raw-argmax metrics and proper scores on unweighted color targets.

    Group names describe known game mechanics, not a proof that every transition
    is observable. Overrides allow explicit corpus annotations. Softmax scores
    remain those of the trained model; change-weighted training can bias them.
    ``raw_scores=False`` accepts an overridden decoded board for fidelity only;
    probability metrics are omitted because a hard gate is not a distribution.
    """

    def __init__(self, overrides=None, fog=-1, raw_scores=True):
        self.overrides = overrides or {}
        self.fog = fog
        self.raw_scores = raw_scores
        self.counts = {g: defaultdict(float) for g in ("all", *GROUPS)}
        self.games = {g: set() for g in self.counts}

    @torch.no_grad()
    def update(self, colors, current, target, games, prediction=None):
        if not len(colors):
            return
        current, target = current.to(colors.device).long(), target.to(colors.device).long()
        if prediction is not None and self.raw_scores:
            raise ValueError("Overridden board predictions require raw_scores=False")
        pred = colors.argmax(1) if prediction is None else prediction
        valid = torch.ones_like(target, dtype=torch.bool)
        if self.fog >= 0:
            valid &= ~((current == self.fog) & (target == self.fog))
        changed = (current != target) & valid
        correct = (pred == target) & valid
        if self.raw_scores:
            logp = colors.float().log_softmax(1)
            prob = logp.exp()
            nll = -logp.gather(1, target[:, None])[:, 0]
            brier = prob.square().sum(1) - 2 * prob.gather(1, target[:, None])[:, 0] + 1
            entropy = -(prob * logp).sum(1)
            confidence = prob.max(1).values
        group_ids = [game_group(g, self.overrides) for g in games]
        for group, totals in self.counts.items():
            indices = [i for i, g in enumerate(group_ids) if group == "all" or g == group]
            if not indices:
                continue
            self.games[group].update(games[i] for i in indices)
            v, ch, ok = valid[indices], changed[indices], correct[indices]
            boards = v.flatten(1).any(1)
            moving = ch.flatten(1).any(1) & boards
            exact = (~(v & ~ok).flatten(1).any(1)) & boards
            counts = {"boards": boards.sum(), "pixels": v.sum(), "correct": ok.sum(),
                      "changed_pixels": ch.sum(), "changed_correct": (ok & ch).sum(),
                      "static_pixels": (v & ~ch).sum(), "static_correct": (ok & ~ch).sum(),
                      "exact": exact.sum(), "moving": moving.sum(),
                      "moving_exact": (exact & moving).sum(), "noop": (boards & ~moving).sum(),
                      "noop_exact": (exact & ~moving).sum(),
                      "moving_pixels": v[moving].sum(), "moving_correct": ok[moving].sum(),
                      "noop_pixels": v[boards & ~moving].sum(),
                      "noop_correct": ok[boards & ~moving].sum()}
            if self.raw_scores:
                counts.update(nll_sum=nll[indices][v].sum(), brier_sum=brier[indices][v].sum(),
                              entropy_sum=entropy[indices][v].sum())
            values = torch.stack([x.float() for x in counts.values()]).cpu().tolist()
            for key, value in zip(counts, values):
                totals[key] += value
            if not self.raw_scores:
                continue
            conf = confidence[indices][v]
            bins = (conf * 10).long().clamp(max=9)
            for prefix, weights in (("count", None), ("confidence", conf),
                                    ("correct", ok[v].float())):
                values = torch.bincount(bins, weights=weights, minlength=10).cpu().tolist()
                for i, value in enumerate(values):
                    totals[f"calibration_{prefix}_{i}"] += value

    def result(self):
        result = {}
        for group, c in self.counts.items():
            def r(a, b):
                return c[a] / c[b] if c[b] else None
            ece = sum(abs(c[f"calibration_correct_{i}"] - c[f"calibration_confidence_{i}"])
                      for i in range(10)) / c["pixels"] if c["pixels"] and self.raw_scores else None
            result[group] = {"games": sorted(self.games[group]), "boards": int(c["boards"]),
                "pixels": int(c["pixels"]), "moving_boards": int(c["moving"]),
                "noop_boards": int(c["noop"]), "cell": r("correct", "pixels"),
                "chg": r("changed_correct", "changed_pixels"),
                "static": r("static_correct", "static_pixels"), "exact": r("exact", "boards"),
                "moving_cell": r("moving_correct", "moving_pixels"),
                "noop_cell": r("noop_correct", "noop_pixels"),
                "moving_exact": r("moving_exact", "moving"), "noop_exact": r("noop_exact", "noop"),
                "nll": r("nll_sum", "pixels") if self.raw_scores else None,
                "brier": r("brier_sum", "pixels") if self.raw_scores else None,
                "entropy": r("entropy_sum", "pixels") if self.raw_scores else None, "ece": ece}
        return result


def rollout_batch_indices(n_batches, limit):
    """Evenly spread a bounded evaluation budget across the sorted loader."""
    if limit <= 0 or limit >= n_batches:
        return set(range(n_batches))
    if limit == 1:
        return {n_batches // 2}
    return {round(i * (n_batches - 1) / (limit - 1)) for i in range(limit)}


def format_evaluation(split, evaluation):
    lines = []
    views = [("one-step raw", evaluation["one_step"]), *
             [(f"rollout k={k} raw-final", v) for k, v in evaluation["rollouts"].items()]]
    if "one_step_gated" in evaluation:
        threshold = evaluation["change_threshold"]
        views += [(f"one-step gate={threshold:g}", evaluation["one_step_gated"]), *
                  [(f"rollout k={k} gate={threshold:g}", v)
                   for k, v in evaluation["rollouts_gated"].items()]]
    for depth, groups in views:
        for group, values in groups.items():
            def f(key):
                value = values[key]
                return "n/a" if value is None else f"{value:.4f}"
            line = (f"  {split} {depth} {group}: boards={values['boards']} "
                         f"games={len(values['games'])} moving={values['moving_boards']} "
                         f"noop={values['noop_boards']} cell={f('cell')} chg={f('chg')} "
                         f"static={f('static')} exact={f('exact')} moving_exact={f('moving_exact')} "
                         f"noop_exact={f('noop_exact')} moving_cell={f('moving_cell')} "
                         f"noop_cell={f('noop_cell')}")
            if "gate=" not in depth:
                line += (f" nll={f('nll')} brier={f('brier')} "
                         f"entropy={f('entropy')} ece={f('ece')}")
            lines.append(line)
    return "\n".join(lines)
