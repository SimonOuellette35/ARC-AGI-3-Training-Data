"""Restore training progress while keeping the current run's hyperparameters."""
from contextlib import contextmanager
from copy import deepcopy


@contextmanager
def keep_training_config(optimizer, scheduler):
    """Wrap state loading for an optimizer and a LambdaLR schedule.

    Adam moments and scheduler counters resume; parameter-group settings and
    the LR function/base rates come from the newly constructed run.
    """
    groups = [deepcopy({k: v for k, v in group.items() if k != "params"})
              for group in optimizer.param_groups]
    schedule = deepcopy(scheduler.state_dict())
    yield
    for group, current in zip(optimizer.param_groups, groups):
        params = group["params"]
        group.clear()
        group.update(current, params=params)
    progress = scheduler.state_dict()
    for key in ("last_epoch", "_step_count"):
        schedule[key] = progress[key]
    scheduler.load_state_dict(schedule)
    # Both policy and LoRA trainers use LambdaLR. Recompute immediately so the
    # first resumed update also uses the current LR and warmup/decay settings.
    rates = [base * fn(scheduler.last_epoch)
             for base, fn in zip(scheduler.base_lrs, scheduler.lr_lambdas)]
    for group, rate in zip(optimizer.param_groups, rates):
        group["lr"] = rate
    scheduler._last_lr = rates


class CurrentConfigTrainerMixin:
    """Transformers Trainer resume hooks, shared by both goal trainers."""

    def _inner_training_loop(self, batch_size=None, args=None, **kwargs):
        # Trainer.train restores a saved batch size before entering this loop.
        return super()._inner_training_loop(
            batch_size=args.train_batch_size, args=args, **kwargs)

    def _load_optimizer_and_scheduler(self, checkpoint):
        if checkpoint is None:
            return super()._load_optimizer_and_scheduler(checkpoint)
        with keep_training_config(self.optimizer, self.lr_scheduler):
            return super()._load_optimizer_and_scheduler(checkpoint)
