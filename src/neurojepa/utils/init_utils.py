import logging
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, Optional

import torch
import yaml

from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import (
    MultilabelAUROC, MultilabelAveragePrecision, MultilabelAccuracy, MultilabelF1Score,
    BinaryAUROC, BinaryAveragePrecision,
    MulticlassAUROC, MulticlassAveragePrecision,
)
    
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError, R2Score

import neurojepa.models.predictor as vit_pred
import neurojepa.models.vision_transformer as vit
from neurojepa.utils.checkpoint_loader import robust_checkpoint_loader
from neurojepa.utils.schedulers import CosineWDSchedule, LinearDecaySchedule, WarmupCosineSchedule
from neurojepa.utils.wrappers import MultiSeqWrapper, PredictorMultiSeqWrapper, MultiModalSeqWrapper

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

_DEFAULT_NEUROJEPA_FINETUNE_CONFIG = (
    Path(__file__).resolve().parents[3] / "configs" / "finetune" / "finetune_neurojepa.yaml"
)


def _load_config_file(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _get_config_section(config, key):
    if isinstance(config, Mapping):
        return config.get(key, {})
    return getattr(config, key, {})


def _get_config_value(config, key, default=None):
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _to_attr_namespace(value):
    if isinstance(value, Mapping):
        return SimpleNamespace(**{k: _to_attr_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_attr_namespace(v) for v in value]
    return value


def _build_backbone_kwargs_from_config(config, device):
    model_cfg = _get_config_section(config, "model")
    meta_cfg = _get_config_section(config, "meta")

    backbone_name = _get_config_value(
        model_cfg,
        "backbone_name",
        _get_config_value(model_cfg, "model_name", "vit_base"),
    )

    backbone_kwargs = {
        "in_chans": _get_config_value(model_cfg, "in_chans", 1),
        "out_layers": _get_config_value(model_cfg, "out_layers", None),
        "uniform_power": _get_config_value(model_cfg, "uniform_power", False),
        "patch_size": _get_config_value(model_cfg, "patch_size", [16, 16, 4]),
        "model_name": backbone_name,
        "img_size": _get_config_value(model_cfg, "img_size", [224, 224, 64]),
        "use_sdpa": _get_config_value(meta_cfg, "use_sdpa", _get_config_value(model_cfg, "use_sdpa", False)),
        "use_silu": _get_config_value(model_cfg, "use_silu", False),
        "wide_silu": _get_config_value(model_cfg, "wide_silu", False),
        "use_rope": _get_config_value(model_cfg, "use_rope", False),
        "use_activation_checkpointing": _get_config_value(model_cfg, "use_activation_checkpointing", False),
        "device": device,
    }

    if _get_config_value(model_cfg, "use_moe", False):
        backbone_kwargs["use_moe"] = True
        backbone_kwargs["moe_params"] = _to_attr_namespace(_get_config_value(model_cfg, "moe_params", None))

    return backbone_kwargs


def _resolve_hf_or_local_file(
    model_name_or_path,
    filename,
    revision=None,
    cache_dir=None,
    token=None,
    subfolder=None,
):
    local_path = Path(model_name_or_path)
    if local_path.exists():
        return local_path / filename if local_path.is_dir() else local_path

    from huggingface_hub import hf_hub_download

    download_kwargs = {
        "repo_id": model_name_or_path,
        "filename": filename,
    }
    if revision is not None:
        download_kwargs["revision"] = revision
    if cache_dir is not None:
        download_kwargs["cache_dir"] = cache_dir
    if token is not None:
        download_kwargs["token"] = token
    if subfolder is not None:
        download_kwargs["subfolder"] = subfolder

    return Path(hf_hub_download(**download_kwargs))


def _resolve_hf_or_local_checkpoint(
    model_name_or_path,
    checkpoint_filename=None,
    fallback_checkpoint_filename=None,
    revision=None,
    cache_dir=None,
    token=None,
    subfolder=None,
):
    if checkpoint_filename is not None:
        return _resolve_hf_or_local_file(
            model_name_or_path=model_name_or_path,
            filename=checkpoint_filename,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            subfolder=subfolder,
        )

    checkpoint_filenames = ["model.safetensors"]
    for filename in (fallback_checkpoint_filename, "pytorch_model.bin"):
        if filename is not None and filename not in checkpoint_filenames:
            checkpoint_filenames.append(filename)

    local_path = Path(model_name_or_path)
    if local_path.is_dir():
        for filename in checkpoint_filenames:
            checkpoint_path = local_path / filename
            if checkpoint_path.exists():
                return checkpoint_path
        return local_path / "model.safetensors"
    if local_path.exists():
        return local_path

    last_error = None
    for filename in checkpoint_filenames:
        try:
            return _resolve_hf_or_local_file(
                model_name_or_path=model_name_or_path,
                filename=filename,
                revision=revision,
                cache_dir=cache_dir,
                token=token,
                subfolder=subfolder,
            )
        except Exception as error:
            last_error = error
            logger.info(f"Could not resolve {filename} from {model_name_or_path}: {error}")

    raise last_error


def _load_hf_or_local_checkpoint(checkpoint_path):
    if checkpoint_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(checkpoint_path), device="cpu")

    return robust_checkpoint_loader(str(checkpoint_path), map_location=torch.device("cpu"))


def _load_hf_or_local_config(
    model_name_or_path,
    config_path=None,
    config_filename="config.json",
    revision=None,
    cache_dir=None,
    token=None,
    subfolder=None,
):
    if config_path is not None:
        logger.info(f"Loading Neuro-JEPA backbone config from {config_path}")
        return _load_config_file(config_path)

    local_path = Path(model_name_or_path)
    if local_path.is_dir():
        local_config_path = local_path / config_filename
        if local_config_path.exists():
            logger.info(f"Loading Neuro-JEPA backbone config from {local_config_path}")
            return _load_config_file(local_config_path)
    elif local_path.exists():
        logger.info(f"Loading fallback Neuro-JEPA backbone config from {_DEFAULT_NEUROJEPA_FINETUNE_CONFIG}")
        return _load_config_file(_DEFAULT_NEUROJEPA_FINETUNE_CONFIG)

    config_path = _resolve_hf_or_local_file(
        model_name_or_path=model_name_or_path,
        filename=config_filename,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        subfolder=subfolder,
    )
    logger.info(f"Loading Neuro-JEPA backbone config from {config_path}")
    return _load_config_file(config_path)


def _extract_backbone_state_dict(checkpoint, checkpoint_key=None):
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint must be a mapping or state_dict.")

    if checkpoint_key is not None:
        if checkpoint_key not in checkpoint:
            raise ValueError(f"Checkpoint does not contain key '{checkpoint_key}'.")
        return checkpoint[checkpoint_key]

    for key in ("encoder", "target_encoder", "state_dict", "model"):
        if key in checkpoint and isinstance(checkpoint[key], Mapping):
            return checkpoint[key]

    return checkpoint


def _clean_backbone_state_dict(state_dict):
    cleaned_state_dict = {}
    prefix_order = (
        "student.vision_encoder.",
        "student.encoder.",
        "vision_encoder.",
        "target_encoder.",
        "encoder.",
        "model.",
        "module.",
        "_orig_mod.",
        "backbone.",
    )

    for name, value in state_dict.items():
        clean_name = name.replace("_checkpoint_wrapped_module.", "")
        stripped = True
        while stripped:
            stripped = False
            for prefix in prefix_order:
                if clean_name.startswith(prefix):
                    clean_name = clean_name[len(prefix):]
                    stripped = True
        cleaned_state_dict[clean_name] = value

    return cleaned_state_dict


def init_backbone(
    device,
    patch_size=[16, 16, 4],
    model_name="vit_base",
    img_size=[224, 224, 64],
    in_chans=1,
    out_layers=None,
    uniform_power=False,
    use_sdpa=False,
    use_rope=False,
    use_silu=False,
    use_moe=False,
    moe_params=None,
    wide_silu=False,
    use_activation_checkpointing=False,
):
    backbone = vit.__dict__[model_name](
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        out_layers=out_layers,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
        use_silu=use_silu,
        use_moe=use_moe,
        moe_params=moe_params,
        wide_silu=wide_silu,
        use_activation_checkpointing=use_activation_checkpointing,
        use_rope=use_rope,
    )
    backbone.to(device)
    logger.info(backbone)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Backbone number of parameters: {count_parameters(backbone)}")

    return backbone


def load_backbone_from_hf(
    model_name_or_path: str,
    device: Optional[str] = None,
    config_path: Optional[str] = None,
    config_filename: str = "config.json",
    checkpoint_filename: Optional[str] = None,
    checkpoint_key: Optional[str] = None,
    strict: bool = False,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
    subfolder: Optional[str] = None,
):
    """
    Load a Neuro-JEPA ViT backbone from a HuggingFace Hub checkpoint or local path.

    Hub repo IDs download ``config.json`` from the repo and use it to build the
    backbone architecture before loading weights. Local checkpoint files fall back
    to ``configs/finetune/finetune_neurojepa.yaml`` unless ``config_path`` is set.

    Args:
        model_name_or_path: HuggingFace repo ID, local checkpoint file, or local
            directory containing ``checkpoint_filename``.
        device: Device passed to ``init_backbone``. Defaults to CUDA if available.
        config_path: Local YAML/JSON config used to build the backbone architecture.
        config_filename: Config file to download from the Hub or read from a
            local directory.
        checkpoint_filename: Checkpoint file to download from the Hub or read from a
            local directory. Defaults to ``model.safetensors`` when available,
            otherwise falls back to ``pytorch_model.bin``.
        checkpoint_key: Optional explicit key inside the checkpoint state, e.g.
            ``"encoder"``, ``"target_encoder"``, or ``"state_dict"``.
        strict: Forwarded to ``load_state_dict``.
        revision: Optional HuggingFace Hub revision.
        cache_dir: Optional HuggingFace Hub cache directory.
        token: Optional HuggingFace Hub token for private repos.
        subfolder: Optional HuggingFace Hub subfolder containing the checkpoint.

    Returns:
        torch.nn.Module: Backbone loaded with the checkpoint weights.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    config = _load_hf_or_local_config(
        model_name_or_path=model_name_or_path,
        config_path=config_path,
        config_filename=config_filename,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        subfolder=subfolder,
    )
    hub_cfg = _get_config_section(config, "huggingface")
    fallback_checkpoint_filename = _get_config_value(hub_cfg, "checkpoint_filename", None)
    if checkpoint_key is None:
        checkpoint_key = _get_config_value(hub_cfg, "checkpoint_key", None)

    backbone = init_backbone(**_build_backbone_kwargs_from_config(config, device))

    checkpoint_path = _resolve_hf_or_local_checkpoint(
        model_name_or_path=model_name_or_path,
        checkpoint_filename=checkpoint_filename,
        fallback_checkpoint_filename=fallback_checkpoint_filename,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        subfolder=subfolder,
    )
    logger.info(f"Loading Neuro-JEPA backbone checkpoint from {checkpoint_path}")
    checkpoint = _load_hf_or_local_checkpoint(checkpoint_path)
    state_dict = _extract_backbone_state_dict(checkpoint, checkpoint_key=checkpoint_key)
    state_dict = _clean_backbone_state_dict(state_dict)

    msg = backbone.load_state_dict(state_dict, strict=strict)
    logger.info(f"Loaded Neuro-JEPA backbone from {checkpoint_path} with msg: {msg}")

    del checkpoint
    return backbone


def _normalize_hf_token(token):
    if token in (None, False, "", "false", "False", "none", "None", "null", "Null"):
        return None
    if token in (True, "true", "True"):
        return True
    return token


def load_backbone_weights(backbone, cfg, strict: Optional[bool] = None):
    """
    Load backbone weights from either a local checkpoint or a Hugging Face repo.

    Expected config values under ``meta``:
        load_checkpoint: bool
        checkpoint_source: "local" or "hf" (default: "local")
        load_path_checkpoint: local checkpoint path when source is local
        hf_model_id: Hub repo ID when source is hf
        hf_revision, hf_checkpoint_filename, hf_checkpoint_key, hf_cache_dir,
            hf_token, hf_subfolder: optional Hugging Face loading controls
    """
    meta_cfg = _get_config_section(cfg, "meta")
    if not _get_config_value(meta_cfg, "load_checkpoint", False):
        return backbone

    if strict is None:
        strict = _get_config_value(meta_cfg, "checkpoint_strict", False)

    checkpoint_source = _get_config_value(
        meta_cfg,
        "checkpoint_source",
        _get_config_value(meta_cfg, "load_checkpoint_source", "local"),
    )
    checkpoint_source = str(checkpoint_source).lower()

    if checkpoint_source in {"hf", "hub", "huggingface", "huggingface_hub"}:
        model_name_or_path = _get_config_value(
            meta_cfg,
            "hf_model_id",
            _get_config_value(meta_cfg, "hf_repo_id", None),
        )
        if model_name_or_path is None:
            raise ValueError("Set meta.hf_model_id when meta.checkpoint_source='hf'.")

        checkpoint_path = _resolve_hf_or_local_checkpoint(
            model_name_or_path=model_name_or_path,
            checkpoint_filename=_get_config_value(meta_cfg, "hf_checkpoint_filename", None),
            revision=_get_config_value(meta_cfg, "hf_revision", None),
            cache_dir=_get_config_value(meta_cfg, "hf_cache_dir", None),
            token=_normalize_hf_token(_get_config_value(meta_cfg, "hf_token", None)),
            subfolder=_get_config_value(meta_cfg, "hf_subfolder", None),
        )
    elif checkpoint_source == "local":
        checkpoint_path = _get_config_value(meta_cfg, "load_path_checkpoint", None)
        if checkpoint_path is None:
            raise ValueError("Set meta.load_path_checkpoint when meta.checkpoint_source='local'.")
        checkpoint_path = Path(checkpoint_path)
    else:
        raise ValueError(f"Unsupported checkpoint_source '{checkpoint_source}'. Use 'local' or 'hf'.")

    logger.info(f"Loading backbone weights from {checkpoint_source} checkpoint: {checkpoint_path}")
    checkpoint = _load_hf_or_local_checkpoint(checkpoint_path)
    state_dict = _extract_backbone_state_dict(
        checkpoint,
        checkpoint_key=_get_config_value(meta_cfg, "hf_checkpoint_key", _get_config_value(meta_cfg, "checkpoint_key", None)),
    )
    state_dict = _clean_backbone_state_dict(state_dict)

    msg = backbone.load_state_dict(state_dict, strict=strict)
    logger.info(f"Loaded backbone weights with msg: {msg}")
    del checkpoint
    return backbone


def init_model(
    device,
    patch_size=[16, 16, 4],
    model_name="vit_base",
    img_size=[224, 224, 64],
    in_chans=1,
    pred_depth=6,
    pred_num_heads=None,
    pred_embed_dim=384,
    uniform_power=False,
    use_mask_tokens=False,
    num_mask_tokens=2,
    zero_init_mask_tokens=True,
    use_sdpa=False,
    use_rope=False,
    use_silu=False,
    use_pred_silu=False,
    use_moe=False,
    moe_params=None,
    wide_silu=False,
    use_activation_checkpointing=False,
):
    encoder = vit.__dict__[model_name](
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
        use_silu=use_silu,
        use_moe=use_moe,
        moe_params=moe_params,
        wide_silu=wide_silu,
        use_activation_checkpointing=use_activation_checkpointing,
        use_rope=use_rope,
    )
    encoder = MultiSeqWrapper(encoder)
    predictor = vit_pred.__dict__["vit_predictor"](
        img_size=img_size,
        use_mask_tokens=use_mask_tokens,
        patch_size=patch_size,
        embed_dim=encoder.backbone.embed_dim,
        predictor_embed_dim=pred_embed_dim,
        depth=pred_depth,
        num_heads=encoder.backbone.num_heads if pred_num_heads is None else pred_num_heads,
        uniform_power=uniform_power,
        num_mask_tokens=num_mask_tokens,
        zero_init_mask_tokens=zero_init_mask_tokens,
        use_rope=use_rope,
        use_sdpa=use_sdpa,
        use_silu=use_pred_silu,
        wide_silu=wide_silu,
        use_activation_checkpointing=use_activation_checkpointing,
    )
    predictor = PredictorMultiSeqWrapper(predictor)

    encoder.to(device)
    predictor.to(device)
    logger.info(encoder)
    logger.info(predictor)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Encoder number of parameters: {count_parameters(encoder)}")
    logger.info(f"Predictor number of parameters: {count_parameters(predictor)}")

    return encoder, predictor


def init_opt(
    is_anneal,
    encoder,
    predictor,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    mixed_precision=False,
    ipe_scale=1.25,
    betas=(0.9, 0.999),
    eps=1e-8,
    zero_init_bias_wd=True,
):
    param_groups = [
        {"params": (p for n, p in encoder.named_parameters() if ("bias" not in n) and (len(p.shape) != 1))},
        {"params": (p for n, p in predictor.named_parameters() if ("bias" not in n) and (len(p.shape) != 1))},
        {
            "params": (p for n, p in encoder.named_parameters() if ("bias" in n) or (len(p.shape) == 1)),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
        },
        {
            "params": (p for n, p in predictor.named_parameters() if ("bias" in n) or (len(p.shape) == 1)),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
        },
    ]

    # fused=True merges per-parameter Adam kernels into one CUDA kernel on A100/H100
    # (requires CUDA >= 11.6 and PyTorch >= 2.0; sm >= 7.0 covers Volta and newer)
    _use_fused = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7
    optimizer = torch.optim.AdamW(param_groups, betas=betas, eps=eps, fused=_use_fused)
    if not is_anneal:
        scheduler = WarmupCosineSchedule(
            optimizer,
            warmup_steps=int(warmup * iterations_per_epoch),
            start_lr=start_lr,
            ref_lr=ref_lr,
            final_lr=final_lr,
            T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
        )
    else:
        scheduler = LinearDecaySchedule(
            optimizer,
            ref_lr=ref_lr,
            final_lr=final_lr,
            T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
        )
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
    )
    scaler = torch.amp.GradScaler('cuda') if mixed_precision else None
    return optimizer, scaler, scheduler, wd_scheduler


def init_opt_monitor(
    backbone,
    classifier,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    lr_scale=1.0,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    mixed_precision=False,
    ipe_scale=1.25,
    betas=(0.9, 0.999),
    eps=1e-8,
    zero_init_bias_wd=True,
    is_anneal=False,
):
    if not isinstance(backbone, list):
        backbone_list = [backbone]
    else:
        backbone_list = backbone

    param_groups = [
        {
            "params": (p for bb in backbone_list for n, p in bb.named_parameters() if ("bias" not in n) and (len(p.shape) != 1))
        },
        {
            "params": (p for n, p in classifier.named_parameters() if ("bias" not in n) and (len(p.shape) != 1)),
            "lr_scale": lr_scale,
        },
        {
            "params": (p for bb in backbone_list for n, p in bb.named_parameters() if ("bias" in n) or (len(p.shape) == 1)),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
        },
        {
            "params": (p for n, p in classifier.named_parameters() if ("bias" in n) or (len(p.shape) == 1)),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
            "lr_scale": lr_scale,
        },
    ]

    _use_fused = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7
    optimizer = torch.optim.AdamW(param_groups, betas=betas, eps=eps, fused=_use_fused)
    if not is_anneal:
        scheduler = WarmupCosineSchedule(
            optimizer,
            warmup_steps=int(warmup * iterations_per_epoch),
            start_lr=start_lr,
            ref_lr=ref_lr,
            final_lr=final_lr,
            T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
        )
    else:
        scheduler = LinearDecaySchedule(
            optimizer,
            ref_lr=ref_lr,
            final_lr=final_lr,
            T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
        )
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
    )
    scaler = torch.amp.GradScaler('cuda') if mixed_precision else None
    return optimizer, scaler, scheduler, wd_scheduler


def set_config_vjepa(cfg, device):
    model_cfg = {
        "in_chans": cfg.model.in_chans,
        "uniform_power": cfg.model.uniform_power,
        "use_mask_tokens": cfg.model.use_mask_tokens,
        "num_mask_tokens": int(len(cfg.mask)),
        "zero_init_mask_tokens": cfg.model.zero_init_mask_tokens,
        "patch_size": cfg.model.patch_size,
        "model_name": cfg.model.backbone_name,
        "img_size": cfg.model.img_size,
        "pred_depth": cfg.model.pred_depth,
        "pred_num_heads": cfg.model.pred_num_heads,
        "pred_embed_dim": cfg.model.pred_embed_dim,
        "use_sdpa": cfg.meta.use_sdpa,
        "use_silu": cfg.model.use_silu,
        "use_pred_silu": cfg.model.use_pred_silu,
        "wide_silu": cfg.model.wide_silu,
        "use_rope": cfg.model.use_rope,
        "use_activation_checkpointing": cfg.model.use_activation_checkpointing,
        "device": device,
    }
    
    # Add MoE settings if it is specified in the config
    if hasattr(cfg.model, 'use_moe') and cfg.model.use_moe:
        model_cfg["use_moe"] = cfg.model.use_moe
        model_cfg["moe_params"] = cfg.model.moe_params
    
    return model_cfg


def set_config_vit(cfg, device):
    model_cfg = {
        "in_chans": cfg.model.in_chans,
        "out_layers": cfg.model.get("out_layers", None),
        "uniform_power": cfg.model.uniform_power,
        "patch_size": cfg.model.patch_size,
        "model_name": cfg.model.backbone_name,
        "img_size": cfg.model.img_size,
        "use_sdpa": cfg.meta.use_sdpa,
        "use_silu": cfg.model.use_silu,
        "wide_silu": cfg.model.wide_silu,
        "use_rope": cfg.model.use_rope,
        "use_activation_checkpointing": cfg.model.use_activation_checkpointing,
        "device": device,
    }
    
    # Add MoE settings if it is specified in the config
    if hasattr(cfg.model, 'use_moe') and cfg.model.use_moe:
        model_cfg["use_moe"] = cfg.model.use_moe
        model_cfg["moe_params"] = cfg.model.moe_params
    
    return model_cfg


def build_monitor_metrics(num_labels: int, device):
    """
    Returns a MetricCollection that computes:
      - per-class AUROC, AUPRC, ACC (tensors of shape [C])
      - macro AUROC, AUPRC, ACC (scalars)
    """
    all_metrics = MetricCollection({
        "auroc_perclass": MultilabelAUROC(num_labels=num_labels, average=None),
        "auprc_perclass": MultilabelAveragePrecision(num_labels=num_labels, average=None),
        "acc_perclass":   MultilabelAccuracy(num_labels=num_labels, average=None),
        "f1_perclass": MultilabelF1Score(num_labels=num_labels, average=None),
        "auroc_macro": MultilabelAUROC(num_labels=num_labels, average="macro"),
        "auprc_macro": MultilabelAveragePrecision(num_labels=num_labels, average="macro"),
        "acc_macro":   MultilabelAccuracy(num_labels=num_labels, average="macro"),
        "f1_macro": MultilabelF1Score(num_labels=num_labels, average="macro"),
    }).to(device)
    
    return all_metrics


class BinaryBalancedAccuracy(Metric):
    """Balanced accuracy = (sensitivity + specificity) / 2."""

    higher_is_better = True
    is_differentiable = False
    full_state_update = False

    def __init__(self, threshold: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.add_state("tp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("tn", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        preds_bin = (preds >= self.threshold).long()
        target = target.long()
        self.tp += ((preds_bin == 1) & (target == 1)).sum()
        self.fp += ((preds_bin == 1) & (target == 0)).sum()
        self.tn += ((preds_bin == 0) & (target == 0)).sum()
        self.fn += ((preds_bin == 0) & (target == 1)).sum()

    def compute(self) -> torch.Tensor:
        sensitivity = self.tp.float() / (self.tp + self.fn).clamp(min=1).float()
        specificity = self.tn.float() / (self.tn + self.fp).clamp(min=1).float()
        return (sensitivity + specificity) / 2.0


class MulticlassBalancedAccuracy(Metric):
    """Per-class or macro balanced accuracy = (sensitivity + specificity) / 2 for multiclass."""

    higher_is_better = True
    is_differentiable = False
    full_state_update = False

    def __init__(self, num_classes: int, average: str = 'macro', **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.average = average
        self.add_state("tp", default=torch.zeros(num_classes, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.zeros(num_classes, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("tn", default=torch.zeros(num_classes, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.zeros(num_classes, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        if preds.ndim == 2:
            preds = preds.argmax(dim=1)
        preds = preds.long()
        target = target.long()
        for c in range(self.num_classes):
            pred_c = preds == c
            tgt_c = target == c
            self.tp[c] += (pred_c & tgt_c).sum()
            self.fp[c] += (pred_c & ~tgt_c).sum()
            self.tn[c] += (~pred_c & ~tgt_c).sum()
            self.fn[c] += (~pred_c & tgt_c).sum()

    def compute(self) -> torch.Tensor:
        sensitivity = self.tp.float() / (self.tp + self.fn).clamp(min=1).float()
        specificity = self.tn.float() / (self.tn + self.fp).clamp(min=1).float()
        per_class = (sensitivity + specificity) / 2.0
        if self.average == 'macro':
            return per_class.mean()
        return per_class


def build_downstream_metrics(
    num_classes: int, 
    device: torch.device, 
    prefix: str = None
) -> MetricCollection:
    """
    Builds a torchmetrics.MetricCollection for either binary or multi-class classification.

    Args:
        num_classes: The number of classes. If 2, binary metrics are used.
        device: The device to move the metrics to.
        prefix: An optional prefix to add to the metric names (e.g., 'train/').

    Returns:
        A MetricCollection object ready for use.
    """
    if num_classes == 1:
        # For regression tasks
        metrics = MetricCollection({
            'RootMeanSquaredError': MeanSquaredError(squared=False),
            'MeanAbsoluteError': MeanAbsoluteError(),
            'R2Score': R2Score(),
        })
    elif num_classes == 2:
        # For binary classification, torchmetrics expects a single probability column.
        # The keys 'BinaryAUROC', etc., are handled by the unpack_metrics function.
        metrics = MetricCollection({
            'BinaryAUROC': BinaryAUROC(),
            'BinaryAveragePrecision': BinaryAveragePrecision(),
            'BinaryAccuracy': BinaryBalancedAccuracy(),  # Balanced accuracy = (sensitivity + specificity) / 2
        })
    elif num_classes > 2:
        # For multi-class, we compute both per-class ('none') and macro-averaged metrics.
        metrics = MetricCollection({
            'MulticlassAUROC': MulticlassAUROC(num_classes=num_classes, average='none', validate_args=False),
            'MulticlassAUROC_macro': MulticlassAUROC(num_classes=num_classes, average='macro', validate_args=False),
            'MulticlassAveragePrecision': MulticlassAveragePrecision(num_classes=num_classes, average='none', validate_args=False),
            'MulticlassAveragePrecision_macro': MulticlassAveragePrecision(num_classes=num_classes, average='macro', validate_args=False),
            'MulticlassAccuracy': MulticlassBalancedAccuracy(num_classes=num_classes, average='none'),  # Per-class balanced accuracy
            'MulticlassAccuracy_macro': MulticlassBalancedAccuracy(num_classes=num_classes, average='macro'),  # Macro balanced accuracy
        })
    else:
        raise ValueError("num_classes must be >= 1")
    
    # Add a prefix if provided (e.g., 'train/' or 'val/') and move to the correct device.
    if prefix:
        return metrics.clone(prefix=f"{prefix}/").to(device)
    else:
        return metrics.to(device)
    
