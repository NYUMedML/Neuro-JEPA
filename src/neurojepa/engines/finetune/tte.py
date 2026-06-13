"""
Time-to-Event (Survival Analysis) Engine using DeepSurv / Cox Proportional Hazards Model.

This module provides training and evaluation functions for survival prediction tasks.
"""

import torch
import math
import sys
import logging
import os
import copy
from typing import Any, Dict, Optional

import torch.distributed as dist
import pickle

from neurojepa.utils.misc import all_reduce_mean, MetricLogger, compute_fg_attn_mask
from neurojepa.models.utils.moe import moe_bias_update, calculate_vio_model

from neurojepa.loss.cox_loss import CoxPHLoss, fast_concordance_index

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def log_survival_metrics(loss: float, epoch: int, c_index: float, wandb_run: Any, mode: str = 'train'):
    """Log survival analysis metrics."""
    logger.info(f"{mode.capitalize()} epoch {epoch+1} metrics:")
    logger.info(f"Loss: {loss:.4f} | C-Index: {c_index:.4f}")
    
    if dist.get_rank() == 0 and wandb_run is not None:
        wandb_run.log({
            f"{mode}/loss": loss,
            f"{mode}/c_index": c_index,
            "epoch": epoch + 1,
        })


@torch.no_grad()
def evaluate_epoch(
    cfg: Any,
    model: Dict[str, torch.nn.Module],
    loader: torch.utils.data.DataLoader,
    criterion: CoxPHLoss,
    device: torch.device,
    logger: Optional[logging.Logger] = None,
    use_amp: bool = False,
    save_path: Optional[str] = None,
) -> Dict[str, float]:
    """
    Run evaluation for survival prediction.
    
    Returns:
        Dictionary with 'loss' and 'c_index'
    """
    backbone, classifier = model['backbone'], model['classifier']
    backbone.eval()
    classifier.eval()

    total_loss = 0.0
    all_risks = []
    all_times = []
    all_events = []
    
    for idx, batch in enumerate(loader):
        image, time, event = batch
        image = image.to(device, non_blocking=True)
        time = time.to(device, non_blocking=True).float()
        event = event.to(device, non_blocking=True).float()

        # Compute foreground attention mask (True = valid foreground token)
        fg_attn_mask = None
        if cfg.data.get('use_foreground_mask', False):
            fg_attn_mask = compute_fg_attn_mask(image, cfg.model.patch_size)

        with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.bfloat16):
            # only apply attention mask if it exist as argument in the backbone forward function
            if 'attn_mask' in backbone.forward.__code__.co_varnames:
                feats = backbone(image, attn_mask=fg_attn_mask)
            else:
                feats = backbone(image)
            risk_pred = classifier(feats).squeeze(-1)  # (B,)
            loss = criterion(risk_pred, time, event)

        all_risks.append(risk_pred.cpu())
        all_times.append(time.cpu())
        all_events.append(event.cpu())
        
        total_loss += loss.item()
        
        if (idx + 1) % 50 == 0:
            logger.info(f"Evaluation: [{idx+1}/{len(loader)}]")

    # Concatenate all predictions
    all_risks = torch.cat(all_risks)
    all_times = torch.cat(all_times)
    all_events = torch.cat(all_events)
    
    # Compute C-index
    c_index = fast_concordance_index(all_risks, all_times, all_events)
    
    # Save predictions if requested
    if save_path:
        save_dict = {
            'risk_scores': all_risks.float().numpy(),
            'times': all_times.float().numpy(),
            'events': all_events.float().numpy(),
            'c_index': c_index,
        }
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(save_dict, f)
        logger.info(f"Saved survival predictions to {save_path}")

    avg_loss = all_reduce_mean(torch.tensor(total_loss / len(loader), device=device)).item()

    return {"loss": avg_loss, "c_index": c_index}


def train_one_epoch(
    cfg: Any,
    model: Dict[str, torch.nn.Module],
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    criterion: CoxPHLoss,
    wd_scheduler: Any,
    epoch: int,
    logger: Optional[logging.Logger] = None,
    device: Optional[torch.device] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    wandb_run: Optional[Any] = None,
    use_moe: bool = False,
    moe_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Train the model for one epoch for survival prediction."""
    metric_logger = MetricLogger(delimiter="  ", logger=logger)
    backbone, classifier = model['backbone'], model['classifier']
    backbone.train()
    classifier.train()

    use_amp = cfg.optimization.mixed_precision
    grad_clip = cfg.optimization.get('grad_clip', 0.0)

    all_risks = []
    all_times = []
    all_events = []
        
    for idx, batch_data in enumerate(loader):
        image, time, event = batch_data
        image = image.to(device, non_blocking=True)
        time = time.to(device, non_blocking=True).float()
        event = event.to(device, non_blocking=True).float()

        _new_lr = scheduler.step()
        _new_wd = wd_scheduler.step()

        # Compute foreground attention mask (True = valid foreground token)
        fg_attn_mask = None
        if cfg.data.get('use_foreground_mask', False):
            fg_attn_mask = compute_fg_attn_mask(image, cfg.model.patch_size)
        
        with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.bfloat16):
            # only apply attention mask if it exist as argument in the backbone forward function
            if 'attn_mask' in backbone.forward.__code__.co_varnames:
                feats = backbone(image, attn_mask=fg_attn_mask)
            else:
                feats = backbone(image)
            risk_pred = classifier(feats).squeeze(-1)  # (B,)
            loss = criterion(risk_pred, time, event)

        if not math.isfinite(loss.item()):
            logger.warning(f"Loss is {loss.item()}, skipping step.")
            optimizer.zero_grad()
            continue

        optimizer.zero_grad()
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(backbone.parameters(), grad_clip)
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(backbone.parameters(), grad_clip)
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), grad_clip)
            optimizer.step()
        
        # MoE bias update
        if use_moe:
            if moe_params.moe_type == "bias":
                minvio, maxvio = moe_bias_update(backbone, moe_params.bias_update_rate, used_ddp=True, vit=True)
            else:
                minvio, maxvio = calculate_vio_model(backbone, used_ddp=True, vit=True)
        
        torch.cuda.synchronize()
        
        loss_value = all_reduce_mean(loss).item()
        metric_logger.update(loss=loss_value, lr=_new_lr, wd=_new_wd)
        
        # Collect for C-index computation
        with torch.no_grad():
            all_risks.append(risk_pred.detach().cpu())
            all_times.append(time.cpu())
            all_events.append(event.cpu())

        logger.info(f"Epoch {epoch+1} [{idx+1}/{len(loader)}]  Loss: {loss_value:.4f}")
        if dist.get_rank() == 0 and wandb_run is not None:
            wandb_run.log({
                "train/loss": loss_value,
                "train/lr": _new_lr,
                "train/wd": _new_wd,
            })
            if use_moe:
                wandb_run.log({
                    "train/MoE MinVio": minvio,
                    "train/MoE MaxVio": maxvio,
                })
            
    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats for epoch: {metric_logger}")
    
    # Compute training C-index
    all_risks = torch.cat(all_risks)
    all_times = torch.cat(all_times)
    all_events = torch.cat(all_events)
    train_c_index = fast_concordance_index(all_risks, all_times, all_events)
    
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    stats['c_index'] = train_c_index
    return stats


def trainer(
    cfg: Any,
    model: Dict[str, torch.nn.Module],
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    criterion: CoxPHLoss,
    wd_scheduler: Any,
    start_epoch: int = 0,
    max_epoch: int = 100,
    logger: Optional[logging.Logger] = None,
    device: Optional[torch.device] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    wandb_run: Optional[Any] = None,
    use_moe: bool = False,
    moe_params: Optional[Dict[str, Any]] = None,
    save_path: Optional[str] = None,
) -> float:
    """Main training loop for survival prediction (time-to-event)."""
    
    best_val_c_index = -float("inf")
    best_model_state = None
    best_classifier_state = None

    for epoch in range(start_epoch, max_epoch):
        logger.info(f"Epoch: {epoch+1}/{max_epoch}")
        
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            cfg=cfg, model=model, loader=train_loader, optimizer=optimizer,
            scheduler=scheduler, criterion=criterion, wd_scheduler=wd_scheduler,
            epoch=epoch, logger=logger, device=device, scaler=scaler,
            wandb_run=wandb_run,
            use_moe=use_moe,
            moe_params=moe_params,
        )
        
        log_survival_metrics(train_stats["loss"], epoch, train_stats["c_index"], wandb_run, mode='train')
            
        val_stats = evaluate_epoch(
            cfg=cfg, model=model, loader=val_loader, criterion=criterion, device=device,
            logger=logger, use_amp=cfg.optimization.mixed_precision,
        )
        
        val_c_index = val_stats["c_index"]
        
        if val_c_index > best_val_c_index:
            best_val_c_index = val_c_index
            best_model_state = copy.deepcopy(model['backbone'].state_dict())
            best_classifier_state = copy.deepcopy(model['classifier'].state_dict())
            logger.info(f"*** New best validation C-Index: {best_val_c_index:.4f} at epoch {epoch+1} ***")

        log_survival_metrics(val_stats['loss'], epoch, val_stats['c_index'], wandb_run, mode='val')
    
    # Testing
    logger.info("--- Testing ---")
    if best_classifier_state is not None:
        logger.info(f"Loading best backbone/classifier for testing (Val C-Index: {best_val_c_index:.4f})")
        model['backbone'].load_state_dict(best_model_state)
        model['classifier'].load_state_dict(best_classifier_state)
    else:
        logger.warning("No best model found, using model from last epoch for testing.")
            
    test_stats = evaluate_epoch(
        cfg=cfg, model=model, loader=test_loader, criterion=criterion, device=device,
        logger=logger, use_amp=cfg.optimization.mixed_precision,
        save_path=save_path,
    )
    
    logger.info("Final Test Metrics (using best validation model):")
    log_survival_metrics(test_stats['loss'], max_epoch-1, test_stats['c_index'], wandb_run, mode='test')
    
    logger.info(f"Training Finished after {max_epoch} epochs.")
    return train_stats.get('loss', 0.0)
