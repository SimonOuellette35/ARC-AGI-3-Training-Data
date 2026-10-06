"""Architecture selection shared by training, evaluation and inference."""


def policy_classes(architecture="packed_v1"):
    if architecture == "packed_v1":
        from policy_model import ModelConfig, InContextPolicy
        return ModelConfig, InContextPolicy
    if architecture == "spatial_temporal_v2":
        from policy_model_V2 import SpatialTemporalConfig, SpatialTemporalPolicy
        return SpatialTemporalConfig, SpatialTemporalPolicy
    raise ValueError(f"Unknown policy architecture: {architecture!r}")


def checkpoint_architecture(checkpoint):
    """Old checkpoints have no architecture tag and remain original policies."""
    config_tag = checkpoint["cfg"].get("architecture")
    tag = checkpoint.get("architecture", config_tag or "packed_v1")
    if config_tag is not None and config_tag != tag:
        raise ValueError("Checkpoint architecture disagrees with its config")
    policy_classes(tag)  # validate before attempting to instantiate a model
    return tag


def policy_from_checkpoint(checkpoint, device="cpu"):
    config_class, model_class = policy_classes(checkpoint_architecture(checkpoint))
    cfg = config_class(**checkpoint["cfg"])
    model = model_class(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, cfg
