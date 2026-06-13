import torch
import math
import time
import logging
import os
from itertools import chain
from typing import Any, Dict, Optional

import torch.distributed as dist
import torch.nn.functional as F

from monai.utils import convert_to_tensor

from neurojepa.models.utils.moe import moe_bias_update, calculate_vio_model
from neurojepa.utils.misc import all_reduce_mean, save_checkpoint, MetricLogger, _update_momentum_encoder
from neurojepa.loss.jepa_loss import loss_fn
from neurojepa.utils.infinite_loader import InfiniteLoader


def train_one_epoch(
    cfg: Any,
    model: torch.nn.Module,
    iter_loader: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    wd_scheduler: Any,
    momentum_scheduler: Any,
    moe_bias_scheduler: Any,
    epoch: int,
    max_epoch: int,
    logger: Optional[logging.Logger] = None,
    device: Optional[torch.device] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    wandb_run: Optional[Any] = None,
    use_moe: bool = False,
    moe_params: Optional[Any] = None,
) -> Dict[str, float]:
    """
    Train the model for one epoch.

    Args:
        cfg (Any): Configuration object.
        model (torch.nn.Module): The main model containing 'encoder', 'predictor', and 'target_encoder'.
        loader (torch.utils.data.DataLoader): DataLoader for training data.
        iter_loader: torch.utils.data.DataLoader,
        optimizer (torch.optim.Optimizer): Model Optimizer.
        scheduler (Any): Learning rate scheduler.
        wd_scheduler (Any): Weight decay scheduler.
        momentum_scheduler (Any): Momentum scheduler.
        epoch (int): The current epoch number.
        max_epoch (int): Maximum number of epochs.
        logger (Optional[Any]): Logger.
        device (Optional[torch.device]): Device to use.
        scaler (Optional[torch.amp.GradScaler]): Gradient scaler for float16 mixed precision training.
        wandb_run (Optional[Any]): Weights and Biases run object.
        use_moe (bool): Whether to use Mixture of Experts (MoE).
        moe_params (Optional[Any]): MoE parameters.
    Returns:
        Dict[str, float]: Dictionary of average metrics.
    """
    metric_logger = MetricLogger(delimiter="  ", logger=logger)
    encoder, target_encoder, predictor = model['encoder'], model['target_encoder'], model['predictor']
    trainable_params = tuple(chain(encoder.parameters(), predictor.parameters()))
    
    use_amp = cfg.optimization.mixed_precision
    grad_clip = cfg.optimization.grad_clip
    ipe = int(cfg.optimization.ipe)
    dtype = cfg.meta.dtype
    amp_dtype = torch.bfloat16 if dtype in ('bfloat16', torch.bfloat16) else torch.float16
    use_grad_scaler = use_amp and amp_dtype == torch.float16
    if use_amp and amp_dtype == torch.bfloat16 and scaler is not None:
        if logger is not None:
            logger.warning("Ignoring GradScaler for bfloat16 mixed precision; bf16 does not use loss scaling.")
        scaler = None
    if use_grad_scaler and scaler is None:
        raise ValueError("float16 mixed precision requires a GradScaler, but scaler=None was provided.")

    def _current_grad_norm(parameters) -> torch.Tensor:
        total_norm_sq = None
        for p in parameters:
            if p.grad is None:
                continue
            grad = p.grad.detach()
            if grad.is_sparse:
                grad = grad.coalesce().values()
            grad_norm_sq = grad.float().pow(2).sum()
            total_norm_sq = (
                grad_norm_sq
                if total_norm_sq is None
                else total_norm_sq + grad_norm_sq.to(total_norm_sq.device)
            )
        if total_norm_sq is None:
            return torch.zeros((), device=device, dtype=torch.float32)
        return total_norm_sq.sqrt()
    
    # Deferred metric logging. Accumulate on GPU, sync every LOG_FREQ steps
    LOG_FREQ = getattr(cfg.log, 'log_freq', 100)
    _loss_acc = torch.zeros(1, device=device, dtype=torch.float32)
    _grad_norm_latest = torch.zeros((), device=device, dtype=torch.float32)
    _acc_count = 0

    for idx in range(ipe):
        try:
            batch_data = next(iter_loader)
        except StopIteration:
            # Should not happen with InfiniteLoader, but guarding just in case of empty loader
            logger.warning("StopIteration caught in train_one_epoch, which implies empty loader or InfiniteLoader failure.")
            return {}, 
        
        # Calculate global step
        global_step = epoch * ipe + idx

        # Unpack batch data (4th element is foreground map, None when disabled)
        all_packed_data, all_masks_enc, all_masks_pred = batch_data[:3]
        fg_map = batch_data[3] if len(batch_data) > 3 else None
        all_data, all_modality = all_packed_data[0], all_packed_data[1]

        # if all data is not list, convert to list
        if all_data is not None and not isinstance(all_data, list):
            all_data = [all_data]

        # Send data to device
        data = [convert_to_tensor(d, track_meta=False).to(device, non_blocking=True) for d in all_data]   # (image input in first element)
        
        # convert metatensor to tensor
        masks_enc = [[m.to(device, non_blocking=True) for m in all_masks_enc]]
        masks_pred = [[m.to(device, non_blocking=True) for m in all_masks_pred]]
        if fg_map is not None:
            fg_map = fg_map.to(device, non_blocking=True)
            
        # forward pass
        with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
            # forward target
            with torch.no_grad():
                h = target_encoder(data)
                h = [F.layer_norm(hi, (hi.size(-1),)) for hi in h]
                
            # forward context (z is output features, s is gate scores if MoE)
            z, s = encoder(data, masks_enc)
            # forward predictor
            z = predictor(z, masks_enc, masks_pred)
            # loss
            loss = loss_fn(z, h, masks_pred, cfg.loss.loss_exp, fg_map=fg_map,
                          bg_weight=getattr(cfg.loss, 'bg_weight', 1.0))  # jepa prediction loss

        # Check for finite loss.
        # GradScaler handles fp16 inf/nan gradients. bf16 does not use loss
        # scaling, so it keeps the explicit finite-loss guard.
        if not use_grad_scaler:
            _loss_check = loss.item()
            if not math.isfinite(_loss_check):
                logger.warning(f"Skipped a step with non-finite loss: {_loss_check}")
                optimizer.zero_grad(set_to_none=True)
                del loss, z, h, s, data, masks_enc, masks_pred
                if fg_map is not None:
                    del fg_map
                torch.cuda.empty_cache()
                continue

        # Apply schedules after the finite-loss guard so skipped bf16/non-AMP
        # steps do not consume LR or WD schedule entries.
        _new_lr = scheduler.step()
        _new_wd = wd_scheduler.step()
        
        # Initialize gradient norm tracking variable
        grad_norm = None

        # backward with/without loss scaling
        if use_grad_scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = _current_grad_norm(trainable_params)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = _current_grad_norm(trainable_params)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
            optimizer.step()

        # Zero gradients immediately after step — set_to_none=True releases
        # gradient memory before the (potentially large) momentum update.
        optimizer.zero_grad(set_to_none=True)

        # MoE bias update and violation calculation
        if use_moe:
            if moe_params.moe_type == "bias":
                bias_clip = moe_params.get("bias_clip", 1.0)
                bias_update_rate = moe_bias_scheduler.step()
                minvio, maxvio = moe_bias_update(encoder, bias_update_rate, used_ddp=True, 
                                                 bias_clip=bias_clip, use_proportional=True)
            else:
                minvio, maxvio = calculate_vio_model(encoder, used_ddp=True)

        # Update teacher (momentum) model
        _new_m = next(momentum_scheduler)
        with torch.no_grad():
            _update_momentum_encoder(encoder, target_encoder, _new_m)
        
        # Accumulate metrics on GPU (no CPU sync or all-reduce per step)
        _loss_acc += loss.detach().float().nan_to_num()
        if grad_norm is not None:
            _grad_norm_latest = grad_norm.detach().float().nan_to_num()
        _acc_count += 1

        # Periodic sync: all-reduce + .item() + log every LOG_FREQ steps
        if (_acc_count >= LOG_FREQ or (idx + 1) == ipe) and _acc_count > 0:
            _reduced_loss = all_reduce_mean(_loss_acc / _acc_count)
            loss_value = _reduced_loss.item() if isinstance(_reduced_loss, torch.Tensor) else float(_reduced_loss)
            _reduced_grad_norm = all_reduce_mean(_grad_norm_latest)
            grad_norm_value = (
                _reduced_grad_norm.item()
                if isinstance(_reduced_grad_norm, torch.Tensor)
                else float(_reduced_grad_norm)
            )

            metric_logger.update(
                loss=loss_value,
                lr=_new_lr,
                wd=_new_wd,
                mom=_new_m,
                grad_norm=grad_norm_value,
                global_step=global_step,
            )

            logger.info(f"Epoch {epoch+1}/{max_epoch} [{idx+1}/{ipe}]  Loss(avg {_acc_count} steps): {loss_value:.4f}")

            if dist.get_rank() == 0 and wandb_run is not None:
                log_dict = {
                    "Training Loss": loss_value,
                    "Training lr":   _new_lr,
                    "Training wd":   _new_wd,
                    "Momentum":      _new_m,
                    "Grad Norm":     grad_norm_value,
                    "Epoch":         epoch + 1,
                }
                if use_moe:
                    log_dict["MoE MinVio"] = minvio
                    log_dict["MoE MaxVio"] = maxvio
                    if moe_params.moe_type == "bias":
                        log_dict["MoE Bias Update Rate"] = bias_update_rate
                wandb_run.log(log_dict, step=global_step)

            # Reset accumulators
            _loss_acc.zero_()
            _grad_norm_latest.zero_()
            _acc_count = 0
            
    # Gather stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def trainer(
    cfg: Any,
    model: Dict[str, torch.nn.Module],
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    wd_scheduler: Any,
    momentum_scheduler: Any,
    moe_bias_scheduler: Any,
    start_epoch: int = 0,
    max_epoch: int = 100,
    logger: Optional[logging.Logger] = None,
    device: Optional[torch.device] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    wandb_run: Optional[Any] = None,
    use_moe: bool = False,
    moe_params: Optional[Any] = None,
) -> float:
    """
    Train the model.

    Args:
        cfg (Any): Configuration object.
        model (Dict[str, torch.nn.Module]): The main model.
        loader (torch.utils.data.DataLoader): DataLoader for training data.
        optimizer (torch.optim.Optimizer): Model Optimizer.
        scheduler (Any): Learning rate scheduler.
        wd_scheduler (Any): Weight decay scheduler.
        momentum_scheduler (Any): Momentum scheduler.
        start_epoch (int): Starting epoch number.
        max_epoch (int): Maximum number of epochs.
        logger (Optional[any]): Logger.
        device (Optional[torch.device]): Device to use.
        scaler (Optional[torch.amp.GradScaler]): Gradient scaler for float16 mixed precision training.
        wandb_run (Optional[Any]): Weights and Biases run object.
        use_moe (bool): Whether to use Mixture of Experts (MoE).
        moe_params (Optional[Any]): MoE parameters.

    Returns:
        float: Last training loss.
    """
    dtype = cfg.meta.dtype
    amp_dtype = torch.bfloat16 if dtype in ('bfloat16', torch.bfloat16) else torch.float16
    if cfg.optimization.mixed_precision and amp_dtype == torch.bfloat16 and scaler is not None:
        if logger is not None:
            logger.warning("Ignoring GradScaler for bfloat16 mixed precision; bf16 does not use loss scaling.")
        scaler = None

    iter_loader = InfiniteLoader(loader, start_epoch=start_epoch)

    for epoch in range(start_epoch, max_epoch):
        logger.info(f"Epoch: {epoch+1}")
        epoch_time = time.time()
        # Train for one epoch
        train_stats = train_one_epoch(
            cfg=cfg,
            model=model,
            iter_loader=iter_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            wd_scheduler=wd_scheduler,
            momentum_scheduler=momentum_scheduler,
            moe_bias_scheduler=moe_bias_scheduler,
            epoch=epoch,
            max_epoch=max_epoch,
            logger=logger,
            device=device,
            scaler=scaler,
            wandb_run=wandb_run,
            use_moe=use_moe,
            moe_params=moe_params,
        )
        
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
        logger.info(f"log_stats: {log_stats}")
        
        if dist.get_rank() == 0:
            logger.info(
                f"Final training stats for epoch {epoch+1}/{max_epoch}: {log_stats}, "
                f"time: {time.time() - epoch_time:.2f}s"
            )
        
        # Calculate global_step for checkpointing at the end of the epoch
        global_step = (epoch + 1) * int(cfg.optimization.ipe)

        # Save last checkpoint
        if dist.get_rank() == 0:
            # check if path exists
            if not os.path.exists(cfg.meta.save_path_checkpoint):
                os.makedirs(cfg.meta.save_path_checkpoint, exist_ok=True)
            
            # Save latest checkpoint
            save_path = os.path.join(cfg.meta.save_path_checkpoint, "latest.pt")
            save_checkpoint(
                model["encoder"], 
                model["target_encoder"], 
                model["predictor"], 
                optimizer=optimizer, 
                scaler=scaler, 
                epoch=epoch, 
                loss=train_stats['loss'],
                path=save_path,
                global_step=global_step,
            )
        
            # Save checkpoint by frequency
            if epoch == 0 or (epoch + 1) % cfg.meta.save_checkpoint_freq == 0:
                save_path = os.path.join(cfg.meta.save_path_checkpoint, f"e{epoch+1}.pt")
                save_checkpoint(
                    model["encoder"], 
                    model["target_encoder"], 
                    model["predictor"], 
                    optimizer=optimizer, 
                    scaler=scaler, 
                    epoch=epoch, 
                    loss=train_stats['loss'],
                    path=save_path,
                    global_step=global_step,
                )
            
    logger.info(f"Training Finished after {max_epoch} epochs.")
    return train_stats['loss']
