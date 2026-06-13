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

from neurojepa.data.datasets import get_finetune_dataset
from neurojepa.data.transforms import vit3d_transforms

from neurojepa.engines.inference import *

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

torch.set_float32_matmul_precision('high')


@hydra.main(version_base=None, config_path="../configs/inference")
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
                # Track hyperparameters and run metadata
                config=OmegaConf.to_container(cfg, resolve=True),
            )
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Dataloader
    imtrans, imvals, imtests = vit3d_transforms(cfg, mode='train'), \
        vit3d_transforms(cfg, mode='val'), vit3d_transforms(cfg, mode='test')
    train_loader, val_loader, test_loader = get_finetune_dataset(cfg, augs=[imtrans, imvals, imtests])

    # Create model
    if cfg.model.model_name == 'vit':
        from neurojepa.models.attentive_pooler import AttentiveClassifier
        # Backbone
        model_cfg = set_config_vit(cfg, device)
        backbone = init_backbone(**model_cfg)
        # Classifier
        num_classes = cfg.data.num_classes
        embed_dim = 768
        depth = 1
        num_heads = 16
        classifier = AttentiveClassifier(embed_dim=embed_dim, num_heads=num_heads, depth=depth, 
                                         num_classes=num_classes).to(device)

    # Load model & optimizer & scheduler
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
    
    # Create optimizer & scheduler
    _, scaler, _, _ = init_opt_monitor(
        backbone=backbone,
        classifier=classifier,
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
    
    # Set iterations per epoch to full dataset length if -1
    if cfg_opt.ipe == -1:
        cfg_opt.ipe = len(train_loader)
        logger.info(f"Setting iterations per epoch to {cfg_opt.ipe} (full dataset)")
        
    logger.info(f"iterations per epoch/dataset length: {cfg_opt.ipe}/{len(train_loader)}")

    # Set models to train mode
    backbone.eval()
    model = {"backbone": backbone}

    # Infer model
    inferencer(
        cfg=cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        logger=logger,
        device=device,
        scaler=scaler,
        wandb_run=wandb_run,
    )

    cleanup()


if __name__ == "__main__":
    main()
