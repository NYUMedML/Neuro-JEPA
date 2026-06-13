import os
import copy
import json
import random
import argparse
import wandb
import hydra
import numpy as np
from neurojepa.utils.logger import create_logger
from omegaconf import OmegaConf

from neurojepa.utils.misc import init_distributed_mode
from neurojepa.utils.misc import init_seed, cleanup
from neurojepa.utils.init_utils import init_backbone, init_opt_monitor, load_backbone_weights, set_config_vit

from neurojepa.data.datasets_mm import get_finetune_dataset
from neurojepa.data.transforms import vit3d_transforms

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

torch.set_float32_matmul_precision('highest')


@hydra.main(version_base=None, config_path="../../configs/finetune")
def main(cfg):
    # Init distributed training
    world_size, rank = init_distributed_mode()

    # Create logger
    logger = create_logger(output_dir=cfg.log.output_dir, dist_rank=dist.get_rank(), name=cfg.log.filename)

    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")

    # Print cfg
    logger.info(f"==> CONFIGURATION:\n{OmegaConf.to_yaml(cfg)}")

    # Seed
    seed = cfg.meta.seed + dist.get_rank()
    init_seed(seed)
    logger.info(f"Seed is set to {seed}")

    # Output cfg settings
    if dist.get_rank() == 0:
        path = os.path.join(cfg.log.output_dir, f"{cfg.log.filename}.json")
        with open(path, "w") as f:
            json.dump(OmegaConf.to_container(cfg, resolve=True), f, indent=4)
        logger.info(f"Full cfg saved to {path}")

    # Init wandb
    wandb_run = None
    if cfg.wandb.wandb_enable and dist.get_rank() == 0:
        wandb_run = wandb.init(
            name=cfg.log.filename,
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataloader
    imtrans, imvals, imtests = vit3d_transforms(cfg, mode='train'), \
        vit3d_transforms(cfg, mode='val'), vit3d_transforms(cfg, mode='test')
    train_loader, val_loader, test_loader = get_finetune_dataset(cfg, augs=[imtrans, imvals, imtests])

    # =========================================================================
    # Model Creation
    # =========================================================================
    num_classes = cfg.data.num_classes
    num_modalities = cfg.data.get('num_modalities', 2)
    model_name = cfg.model.model_name

    def _get_vit_embed_dim(backbone_name):
        if 'large' in backbone_name:
            return 1024
        elif 'base' in backbone_name:
            return 768
        raise ValueError(f"Backbone name {backbone_name} not recognized")

    # ---- Backbone (shared encoder for all modalities) ----
    if model_name in ('vit_poe', 'vit_poe_joint'):
        model_cfg = set_config_vit(cfg, device)
        backbone = init_backbone(**model_cfg)
        embed_dim = _get_vit_embed_dim(cfg.model.backbone_name)
    else:
        raise ValueError(f"Model {model_name} not supported for PoE fusion")

    # ---- Classifier: per-modality heads ----
    if 'vit' in model_name:
        from neurojepa.models.attentive_pooler import AttentiveClassifier
        classifier = nn.ModuleList([
            AttentiveClassifier(
                embed_dim=embed_dim,
                num_heads=16,
                depth=1,
                num_classes=num_classes,
            ) for _ in range(num_modalities)
        ]).to(device)
    else:
        classifier = nn.ModuleList([
            nn.Linear(embed_dim, num_classes) for _ in range(num_modalities)
        ]).to(device)

    # ---- Joint head (only for _poe_joint) ----
    joint_head = None
    if 'poe_joint' in model_name:
        if 'vit' in model_name:
            from neurojepa.models.attentive_pooler import AttentiveClassifier
            # Joint head on sequence-concatenated tokens [B, N*num_mods, D]
            joint_head = AttentiveClassifier(
                embed_dim=embed_dim,
                num_heads=16,
                depth=1,
                num_classes=num_classes,
            ).to(device)
        else:
            joint_head = nn.Linear(embed_dim * num_modalities, num_classes).to(device)

    # Load pretrained backbone checkpoint
    backbone = load_backbone_weights(backbone, cfg)

    # Compile model (optional)
    if cfg.meta.compile_model:
        logger.info("Compiling backbone.")
        torch._dynamo.config.optimize_ddp = False
        backbone.compile()

    # Set distributed data parallel
    backbone = DistributedDataParallel(backbone, static_graph=False, find_unused_parameters=True)

    # Get optimization settings
    cfg_opt = cfg.optimization

    # Set iterations per epoch to full dataset length if -1
    if cfg_opt.ipe == -1:
        cfg_opt.ipe = len(train_loader)
        logger.info(f"Setting iterations per epoch to {cfg_opt.ipe} (full dataset)")

    logger.info(f"iterations per epoch/dataset length: {cfg_opt.ipe}/{len(train_loader)}")

    # Set default start epoch & global step
    start_epoch, global_step = 0, 0
    max_epoch = cfg_opt.num_epochs

    # Combine classifier + joint_head parameters for optimizer
    if joint_head is not None:
        classifier_for_opt = nn.ModuleList([classifier, joint_head])
    else:
        classifier_for_opt = classifier

    # Create optimizer & scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt_monitor(
        backbone=backbone,
        classifier=classifier_for_opt,
        wd=cfg_opt.weight_decay,
        final_wd=cfg_opt.final_weight_decay,
        start_lr=cfg_opt.start_lr,
        ref_lr=cfg_opt.lr,
        final_lr=cfg_opt.final_lr,
        iterations_per_epoch=cfg_opt.ipe,
        warmup=cfg_opt.warmup,
        num_epochs=cfg_opt.num_epochs,
        ipe_scale=cfg_opt.ipe_scale,
        mixed_precision=cfg_opt.mixed_precision,
        betas=cfg_opt.betas,
        eps=cfg_opt.eps,
    )

    logger.info(f"Start epoch: {start_epoch}, global step: {global_step}")

    # Set models to train/eval mode
    freeze = cfg.model.get("freeze_backbone", False)
    if freeze:
        backbone.eval()
        for param in backbone.parameters():
            param.requires_grad = False
    else:
        backbone.train()

    classifier.train()
    if joint_head is not None:
        joint_head.train()
    model = {"backbone": backbone, "classifier": classifier, "joint_head": joint_head}

    # PoE fusion outputs log-probabilities -> use NLLLoss
    from neurojepa.engines.finetune.clf_mm_poe import trainer
    criterion = nn.NLLLoss().to(device)

    # Train model
    save_path = cfg.data.get("save_path", None)

    train_loss = trainer(
        cfg=cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        wd_scheduler=wd_scheduler,
        start_epoch=start_epoch,
        max_epoch=max_epoch,
        logger=logger,
        device=device,
        scaler=scaler,
        wandb_run=wandb_run,
        save_path=save_path,
    )
    logger.info(f"train completed, last train loss: {train_loss:.4f} ")

    cleanup()


if __name__ == "__main__":
    main()
