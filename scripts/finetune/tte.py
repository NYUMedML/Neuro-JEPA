"""
Main script for time-to-event / survival analysis fine-tuning using DeepSurv.

Usage:
    torchrun --nnodes 1 --nproc_per_node 1 ./scripts/finetune_tte.py \
        -cd configs/finetune --config-name=finetune_survival \
        data.train_csv_path=/path/to/train.csv \
        data.val_csv_path=/path/to/val.csv \
        data.test_csv_path=/path/to/test.csv \
        data.img_col=t1w_path \
        data.time_col=OS \
        data.event_col=survival
"""

import os
import json
import wandb
import hydra
from neurojepa.utils.logger import create_logger
from omegaconf import OmegaConf

from neurojepa.utils.misc import init_distributed_mode
from neurojepa.utils.misc import init_seed, cleanup
from neurojepa.utils.init_utils import init_backbone, init_opt_monitor, load_backbone_weights, set_config_vit

from neurojepa.data.datasets import get_survival_dataset
from neurojepa.data.transforms import vit3d_transforms

from neurojepa.loss.cox_loss import CoxPHLoss

from neurojepa.engines.finetune.tte import trainer

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
    
    # Retrieve seed
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
    
    # Dataloader for survival analysis
    imtrans = vit3d_transforms(cfg, mode='train')
    imvals = vit3d_transforms(cfg, mode='val')
    imtests = vit3d_transforms(cfg, mode='test')
    
    train_loader, val_loader, test_loader = get_survival_dataset(
        cfg, 
        augs={'train': imtrans, 'val': imvals, 'test': imtests}
    )

    # Create model
    if cfg.model.model_name == 'vit':
        from neurojepa.models.attentive_pooler import AttentiveClassifier
        # Backbone
        model_cfg = set_config_vit(cfg, device)
        backbone = init_backbone(**model_cfg)
        
        # Determine embedding dimension
        if 'large' in cfg.model.backbone_name:
            embed_dim = 1024
        elif 'base' in cfg.model.backbone_name:
            embed_dim = 768
        else:
            raise ValueError(f"Backbone name {cfg.model.backbone_name} not recognized")
        
        # Classifier outputs a single risk score for Cox PH model
        depth = 1
        num_heads = 16
        classifier = AttentiveClassifier(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            depth=depth, 
            num_classes=1  # Single output for risk score
        ).to(device)
    else:
        raise ValueError(f"Model {cfg.model.model_name} not supported")

    # Load pretrained backbone if specified
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
    
    # Create optimizer & scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt_monitor(
        backbone=backbone,
        classifier=classifier,
        wd=cfg_opt.weight_decay,
        final_wd=cfg_opt.final_weight_decay,
        start_lr=cfg_opt.start_lr,
        ref_lr=cfg_opt.lr,
        final_lr=cfg_opt.final_lr,
        lr_scale=cfg_opt.get("lr_scale", 1.0),
        iterations_per_epoch=cfg_opt.ipe,
        warmup=cfg_opt.warmup,
        num_epochs=cfg_opt.num_epochs,
        ipe_scale=cfg_opt.ipe_scale,
        mixed_precision=cfg_opt.mixed_precision,
        betas=cfg_opt.betas,
        eps=cfg_opt.eps,
    )
    
    logger.info(f"Start epoch: {start_epoch}, global step: {global_step}")

    # Set models to train mode
    backbone.train()
    classifier.train()
    model = {"backbone": backbone, "classifier": classifier}
    
    # Cox Proportional Hazards Loss for survival analysis
    l2_lambda = cfg.optimization.get("l2_lambda", 1e-3)
    criterion = CoxPHLoss(l2_lambda=l2_lambda).to(device)
    logger.info(f"CoxPH Loss with L2 regularization lambda={l2_lambda}")

    # Train model
    use_moe = cfg.model.get("use_moe", False)
    moe_params = cfg.model.get("moe_params", None)
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
        use_moe=use_moe,
        moe_params=moe_params,
        save_path=save_path,
    )
    logger.info(f"Training completed, last train loss: {train_loss:.4f}")

    cleanup()


if __name__ == "__main__":
    main()
