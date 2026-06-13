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


from neurojepa.models.attentive_pooler import MultiAttentive
from neurojepa.utils.misc import init_distributed_mode
from neurojepa.utils.misc import init_seed, cleanup
from neurojepa.utils.init_utils import init_backbone, init_opt_monitor, load_backbone_weights, set_config_vit

from neurojepa.data.datasets import get_monitor_dataset
from neurojepa.data.transforms import vit3d_transforms

from neurojepa.engines.monitor import *

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

torch.set_float32_matmul_precision('high')


@hydra.main(version_base=None, config_path="../configs/monitor")
def main(cfg):
    # Init distributed training
    world_size, rank = init_distributed_mode()
    
    # Create logger
    logger = create_logger(output_dir=cfg.log.output_dir, dist_rank=dist.get_rank(), name=cfg.log.filename)
    
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")

    # Print cfg
    logger.info(f"==> CONFIGURATION:\n{OmegaConf.to_yaml(cfg)}")
    
    # Retreive seed
    seed = cfg.meta.seed + dist.get_rank()
    init_seed(seed)
    logger.info(f"Seed is set to {seed}")
    
    # Output cfg settings
    if dist.get_rank() == 0:
        path = os.path.join(cfg.log.output_dir, f"{cfg.log.filename}.json")
        with open(path, "w") as f:
            # Use OmegaConf.to_container for correct dict conversion
            json.dump(OmegaConf.to_container(cfg, resolve=True), f, indent=4)
        logger.info(f"Full cfg saved to {path}")
    
    # Init wandb
    wandb_run = None
    if cfg.wandb.wandb_enable and dist.get_rank() == 0:
        wandb_run = wandb.init(
                # Set the project where this run will be logged
                name=cfg.log.filename,
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                # Track hyperparameters and run metadata
                config=OmegaConf.to_container(cfg, resolve=True),
            )
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Dataloader
    imtrans, imvals, imtests = vit3d_transforms(cfg, mode='train'), \
        vit3d_transforms(cfg, mode='val'), vit3d_transforms(cfg, mode='test')
    train_loader, val_loader, test_loader, test_df = get_monitor_dataset(cfg, augs=[imtrans, imvals, imtests])
    
    # Create model
    if cfg.model.model_name == 'vit':
        # Backbone
        model_cfg = set_config_vit(cfg, device)
        backbone = init_backbone(**model_cfg)
        # Classifier
        num_classes = cfg.data.num_classes
        classifiers = MultiAttentive(embed_dim=backbone.embed_dim, num_classes=num_classes, device=device)
    else:
        raise ValueError(f"Model {cfg.model.model_name} not supported")
    
    # Load model & optimizer & scheduler
    backbone = load_backbone_weights(backbone, cfg)
    
    # Compile model (optional)
    if cfg.meta.compile_model:
        logger.info("Compiling backbone.")
        torch._dynamo.config.optimize_ddp = False
        backbone.compile()

    # Set distributed data parallel
    backbone = DistributedDataParallel(backbone, static_graph=True)
    # Freeze backbone for probing
    for p in backbone.parameters():
        p.requires_grad = False
    
    cfg_opt = cfg.optimization
    
    # Set iterations per epoch to full dataset length if -1
    if cfg_opt.ipe == -1:
        cfg_opt.ipe = len(train_loader)
        logger.info(f"Setting iterations per epoch to {cfg_opt.ipe} (full dataset)")
        
    logger.info(f"iterations per epoch/dataset length: {cfg_opt.ipe}/{len(train_loader)}")
    
    # Set default start epoch & global step
    start_epoch, global_step = 0, 0
    max_epoch = cfg_opt.num_epochs
    
    # Create optimizer & scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt_monitor(
        backbone=backbone,
        classifier=classifiers,
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
    
    # Demographic subgroup columns: read from config or auto-detect from the test CSV
    demo_cols = list(cfg.data.get('demo_cols', None) or [])
    if not demo_cols and test_df is not None:
        auto_demo = ['Manufacturer', 'gender', 'age_group', 'race_group']
        demo_cols = [c for c in auto_demo if c in test_df.columns]
        if demo_cols:
            logger.info(f"Auto-detected demographic columns for subgroup analysis: {demo_cols}")

    logger.info(f"Start epoch: {start_epoch}, global step: {global_step}")

    # Set models to train mode
    backbone.eval()
    classifiers.train()
    model = {"backbone": backbone, "classifier": classifiers}
    criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")

    # Train model
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
        test_df=test_df,
        demo_cols=demo_cols,
    )
    logger.info(f"train completed, last train loss: {train_loss:.4f} ")

    cleanup()


if __name__ == "__main__":
    main()
