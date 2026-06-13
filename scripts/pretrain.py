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

from neurojepa.utils.misc import init_seed, init_distributed_mode, load_checkpoint, cleanup
from neurojepa.utils.init_utils import init_model, init_opt, set_config_vjepa

from neurojepa.data.datasets import get_pretrain_dataloaders
from neurojepa.data.transforms import jepa3d_transforms

from neurojepa.engines.pretrain import *

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from monai.data import set_track_meta
set_track_meta(False)

# Enable TF32 matmul on A100/H100 (significant pretraining speed-up; negligible
# accuracy impact for SSL pretraining).
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

@hydra.main(version_base=None, config_path="../configs/pretrain")
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
    
    # Create model & dataloader
    if cfg.model.model_name == 'jepa':
        # Dataloader
        imtrans = jepa3d_transforms(cfg, mode='train')
        train_loader, mask_collator = get_pretrain_dataloaders(cfg, augs=imtrans)
        # Model
        model_cfg = set_config_vjepa(cfg, device)
        encoder, predictor = init_model(**model_cfg)
        target_encoder = copy.deepcopy(encoder)
    else:
        raise ValueError(f"Model {cfg.model.model_name} not supported")

    encoder = DistributedDataParallel(encoder, static_graph=False, find_unused_parameters=True)
    predictor = DistributedDataParallel(predictor, static_graph=False, find_unused_parameters=True)
    target_encoder = DistributedDataParallel(target_encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False

    # # Set distributed data parallel
    # encoder = DistributedDataParallel(encoder, static_graph=False, find_unused_parameters=True)
    # predictor = DistributedDataParallel(predictor, static_graph=False, find_unused_parameters=True)
    # target_encoder = DistributedDataParallel(target_encoder, static_graph=False, find_unused_parameters=True)
    # # target_encoder is wrapped in DDP as well so buffer names match the encoder during momentum updates.
    # for p in target_encoder.parameters():
    #     p.requires_grad = False
    # if dist.is_initialized() and dist.get_world_size() > 1:
    #     for param in target_encoder.parameters():
    #         dist.broadcast(param.data, src=0)
    #     for buf in target_encoder.buffers():
    #         dist.broadcast(buf.data, src=0)
        
    # Compile model (optional)
    if cfg.meta.compile_model:
        logger.info("Compiling encoder, target_encoder, and predictor.")
        torch._dynamo.config.optimize_ddp = True
        encoder.compile(dynamic=None)
        target_encoder.compile(dynamic=None)
        predictor.compile(dynamic=None)
    
    cfg_opt = cfg.optimization
    
    # Set iterations per epoch to full dataset length if -1
    if cfg_opt.ipe == -1:
        cfg_opt.ipe = len(train_loader)
        logger.info(f"Setting iterations per epoch to {cfg_opt.ipe} (full dataset)")
        
    logger.info(f"iterations per epoch/dataset length: {cfg_opt.ipe}/{len(train_loader)}")
    
    # Create optimizer & scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        is_anneal=cfg_opt.is_anneal,
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
        #dtype=cfg.meta.dtype,
        betas=cfg_opt.betas,
        eps=cfg_opt.eps,
    )
    
    # Create momentum scheduler
    momentum_scheduler = (
        cfg_opt.ema[0] + i * (cfg_opt.ema[1] - cfg_opt.ema[0]) / (cfg_opt.ipe * cfg_opt.num_epochs * cfg_opt.ipe_scale)
        for i in range(int(cfg_opt.ipe * cfg_opt.num_epochs * cfg_opt.ipe_scale) + 1)
    )
    
    # moe bias scheduler
    from neurojepa.utils.schedulers import InverseCosineScheduler
    use_moe = cfg.model.get("use_moe", False)
    if use_moe:
        moe_bias_scheduler = InverseCosineScheduler(
            u_min=cfg.model.moe_params.get("bias_update_rate_min", 1e-4),
            u_max=cfg.model.moe_params.get("bias_update_rate_max", 1e-4),
            total_steps=cfg_opt.ipe * cfg_opt.num_epochs * cfg_opt.ipe_scale,
            warmup_steps=cfg_opt.ipe * cfg_opt.warmup * cfg_opt.ipe_scale,
        )
    
    # Set default start epoch & global step
    start_epoch, global_step = 0, 0
    max_epoch = cfg_opt.num_epochs
    
    # Load model & optimizer & scheduler
    load_path = cfg.meta.load_path_checkpoint
    if cfg.meta.load_checkpoint and load_path is not None:
        is_anneal = cfg_opt.is_anneal
        (
            encoder,
            predictor,
            target_encoder,
            optimizer,
            scaler,
            start_epoch,
            global_step, 
        ) = load_checkpoint(
            r_path=load_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=optimizer,
            scaler=scaler,
            is_anneal=is_anneal,
        )
        # If loading checkpoint, set schedulers to correct step
        if not is_anneal:
            # global_step == (epoch + 1) * ipe
            print(f"Stepping schedulers to global step {global_step}")
            for _ in range(global_step):
                if use_moe:
                    moe_bias_scheduler.step()
                scheduler.step()
                wd_scheduler.step()
                next(momentum_scheduler)
                mask_collator.step()
            start_epoch += 1  # resume from the next epoch

    logger.info(f"Start epoch: {start_epoch}, global step: {global_step}")

    # Set models to train mode
    encoder.train()
    target_encoder.train()
    predictor.train()
    model = {"encoder": encoder, "target_encoder": target_encoder, "predictor": predictor}
    
    # Train model
    moe_params = cfg.model.get("moe_params", None)
    
    train_loss = trainer(
        cfg=cfg,
        model=model,
        loader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        wd_scheduler=wd_scheduler,
        momentum_scheduler=momentum_scheduler,
        moe_bias_scheduler=moe_bias_scheduler if use_moe else None,
        start_epoch=start_epoch,
        max_epoch=max_epoch,
        logger=logger,
        device=device,
        scaler=scaler,
        wandb_run=wandb_run,
        use_moe=use_moe,
        moe_params=moe_params,
    )
    logger.info(f"train completed, last train jepa loss: {train_loss:.4f} ")

    cleanup()


if __name__ == "__main__":
    main()