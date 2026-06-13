import torch
import math
import sys
import logging
import os
from typing import Any, Dict, Optional, Callable

import torch.distributed as dist
import torch.nn.functional as F
import pickle

from torchmetrics import MetricCollection

from neurojepa.utils.init_utils import build_downstream_metrics
from neurojepa.utils.misc import all_reduce_mean, save_checkpoint, MetricLogger, compute_fg_attn_mask

from neurojepa.models.utils.moe import moe_bias_update, calculate_vio_model

import copy

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def unpack_metrics(results: Dict[str, torch.Tensor]) -> Any:
    """Unpacks metrics from torchmetrics for both multi-class and binary cases."""
    # Handle different key names from torchmetrics for multi-class vs binary
    auroc_perclass = results.get("MulticlassAUROC", results.get("BinaryAUROC"))
    auprc_perclass = results.get("MulticlassAveragePrecision", results.get("BinaryAveragePrecision"))
    acc_perclass = results.get("MulticlassAccuracy", results.get("BinaryAccuracy"))
    
    # For macro metrics, some might not be computed (e.g., accuracy)
    macro_auc = results.get("MulticlassAUROC_macro", results.get("BinaryAUROC", torch.tensor(0.0))).item()
    macro_ap = results.get("MulticlassAveragePrecision_macro", results.get("BinaryAveragePrecision", torch.tensor(0.0))).item()
    macro_acc = results.get("MulticlassAccuracy_macro", results.get("BinaryAccuracy", torch.tensor(0.0))).item()

    # Convert per-class tensors to lists, handling the case where they might be 0-d tensors
    perclass_auc = auroc_perclass.detach().cpu().tolist()
    if not isinstance(perclass_auc, list): perclass_auc = [perclass_auc]
    
    perclass_ap = auprc_perclass.detach().cpu().tolist()
    if not isinstance(perclass_ap, list): perclass_ap = [perclass_ap]

    perclass_acc = acc_perclass.detach().cpu().tolist()
    if not isinstance(perclass_acc, list): perclass_acc = [perclass_acc]

    return perclass_auc, perclass_ap, perclass_acc, macro_auc, macro_ap, macro_acc


def log_metrics(loss, epoch, results, class_names, wandb_run, mode='train'):
    perclass_auc, perclass_ap, perclass_acc, \
        macro_auc, macro_ap, macro_acc = unpack_metrics(results)
    
    logger.info(f"{mode.capitalize()} epoch {epoch+1} metrics:")
    log_msg = f"Loss: {loss:.4f} | Macro AUC: {macro_auc:.3f}, Macro AP: {macro_ap:.3f}, Macro Acc: {macro_acc:.3f}"
    logger.info(log_msg)
    
    # Only log per-class metrics if they exist
    if len(perclass_auc) > 1:
        per_class_log = " | ".join([f"{name}: AUC={perclass_auc[i]:.3f}" for i, name in enumerate(class_names)])
        logger.info(f"Per-class AUCs: {per_class_log}")

    if dist.get_rank() == 0 and wandb_run is not None:
        log_dict = {
            f"{mode}/loss": loss,
            f"{mode}/auc_macro": macro_auc,
            f"{mode}/ap_macro": macro_ap,
            f"{mode}/acc_macro": macro_acc,
            "epoch": epoch + 1,
        }
        # Also make wandb logging conditional
        if len(perclass_auc) > 1:
            log_dict.update({f"{mode}/auc_{class_names[i]}": perclass_auc[i] for i in range(len(class_names))})
        else:
            # For binary, the single AUC is the macro AUC
            log_dict[f"{mode}/auc"] = macro_auc

        wandb_run.log(log_dict)


@torch.no_grad()
def evaluate_epoch(
    cfg: Any,
    model: Dict[str, torch.nn.Module],
    loader: torch.utils.data.DataLoader,
    criterion: Callable,
    device: torch.device,
    metrics: MetricCollection,
    logger: Optional[logging.Logger] = None,
    use_amp: bool = False,
    save_path: Optional[str] = None,
) -> Dict[str, float]:
    """Run evaluation for multi-class classification."""
    backbone, classifier = model['backbone'], model['classifier']
    backbone.eval()
    classifier.eval()

    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    for idx, batch in enumerate(loader):
        image, target = batch
        image = image.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        target = target.long()

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
            logits = classifier(feats)  # [B, num_classes]
            loss = criterion(logits, target)
            
            # Use softmax for multi-class probabilities
            probs = F.softmax(logits, dim=1)

        # Adjust probs shape for binary classification
        num_classes = logits.shape[1]
        if num_classes == 2:
            # For binary metrics, pass only the probability of the positive class
            metrics.update(probs[:, 1], target)
        else:
            metrics.update(probs, target)
        
        # Collect predictions if saving is requested
        if save_path:
            all_preds.append(probs.cpu())
            all_targets.append(target.cpu())

        total_loss += loss.item()
        
        if (idx + 1) % 50 == 0:
            logger.info(f"Evaluation: [{idx+1}/{len(loader)}]")

    if save_path:
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        
        if all_preds.shape[1] == 2:
            preds_to_save = all_preds[:, 1].numpy()
        else:
            preds_to_save = all_preds.numpy()

        save_dict = {
            'preds': preds_to_save,
            'targets': all_targets.numpy()
        }
        
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(save_dict, f)
        logger.info(f"Saved predictions to {save_path}")

    results = metrics.compute()
    metrics.reset()
    avg_loss = all_reduce_mean(torch.tensor(total_loss / len(loader), device=device)).item()

    return {"results": results, "loss": avg_loss}
    

def train_one_epoch(
    cfg: Any,
    model: Dict[str, torch.nn.Module],
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    criterion: Callable,
    wd_scheduler: Any,
    epoch: int,
    logger: Optional[logging.Logger] = None,
    device: Optional[torch.device] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    wandb_run: Optional[Any] = None,
    metrics: MetricCollection = None,
    use_moe: bool = False,
    moe_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Train the model for one epoch for multi-class classification."""
    metric_logger = MetricLogger(delimiter="  ", logger=logger)
    backbone, classifier = model['backbone'], model['classifier']
    
    freeze = cfg.model.get('freeze_backbone', False)
    if freeze:
        backbone.eval()
    else:
        backbone.train()
    classifier.train()

    use_amp = cfg.optimization.mixed_precision
    grad_clip = cfg.optimization.get('grad_clip', 0.0)

    if metrics is None:
        raise ValueError("A MetricCollection must be provided.")
        
    for idx, batch_data in enumerate(loader):
        image, target = batch_data
        image = image.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        target = target.long()

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
            logits = classifier(feats)
            loss = criterion(logits, target)

        if not math.isfinite(loss.item()):
            logger.warning(f"Loss is {loss.item()}, skipping step.")
            optimizer.zero_grad()
            continue

        optimizer.zero_grad()
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if grad_clip > 0:
                # Clip gradients for both backbone and classifier
                torch.nn.utils.clip_grad_norm_(backbone.parameters(), grad_clip)
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                # Clip gradients for both backbone and classifier
                torch.nn.utils.clip_grad_norm_(backbone.parameters(), grad_clip)
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), grad_clip)
            optimizer.step()
        
        # MoE bias update and violation calculation
        if use_moe:
            if moe_params.moe_type == "bias":
                minvio, maxvio = moe_bias_update(backbone, moe_params.bias_update_rate, used_ddp=True, vit=True)
            else:
                minvio, maxvio = calculate_vio_model(backbone, used_ddp=True, vit=True)
        
        torch.cuda.synchronize()
        
        loss_value = all_reduce_mean(loss).item()
        metric_logger.update(loss=loss_value, lr=_new_lr, wd=_new_wd)
        
        with torch.no_grad():
            probs = F.softmax(logits, dim=1)
            # Adjust probs shape for binary classification
            num_classes = logits.shape[1]
            if num_classes == 2:
                # For binary metrics, pass only the probability of the positive class
                metrics.update(probs[:, 1], target)
            else:
                metrics.update(probs, target)

        logger.info(f"Epoch {epoch+1} [{idx+1}/{len(loader)}]  Loss: {loss_value:.4f}")
        if dist.get_rank() == 0 and wandb_run is not None:
            wandb_run.log({
                "train/loss": loss_value,
                "train/lr": _new_lr,
                "train/wd": _new_wd,
            })
            # log moe maxvio
            if use_moe:
                wandb_run.log({
                    "train/MoE MinVio": minvio,
                    "train/MoE MaxVio": maxvio,
                })
                
            
    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats for epoch: {metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def trainer(
    cfg: Any,
    model: Dict[str, torch.nn.Module],
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    criterion: Callable,
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
    """Main training loop for multi-class finetuning."""
    num_classes = cfg.data.num_classes
    train_metrics, val_metrics, test_metrics = build_downstream_metrics(num_classes, device=device), \
        build_downstream_metrics(num_classes, device=device), \
        build_downstream_metrics(num_classes, device=device)

    class_names = [f"class_{i}" for i in range(num_classes)]
    
    best_val_score = -float("inf")
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
            wandb_run=wandb_run, metrics=train_metrics,
            use_moe=use_moe,
            moe_params=moe_params,
        )
        
        train_results = train_metrics.compute()
        train_metrics.reset()
        log_metrics(train_stats["loss"], epoch, train_results, class_names, wandb_run, mode='train')
            
        val_stats = evaluate_epoch(
            cfg=cfg, model=model, loader=val_loader, criterion=criterion, device=device,
            metrics=val_metrics, logger=logger, use_amp=cfg.optimization.mixed_precision,
        )
        
        _, _, _, val_auc_macro, val_ap_macro, val_acc_macro = unpack_metrics(val_stats["results"])
        
        # Composite score: average of macro AUC and AP
        val_score = 0.5 * val_auc_macro + 0.5 * val_ap_macro
        
        if val_score > best_val_score:
            best_val_score = val_score
            best_model_state = copy.deepcopy(model['backbone'].state_dict())
            best_classifier_state = copy.deepcopy(model['classifier'].state_dict())
            logger.info(
                f"*** New best validation composite score: {best_val_score:.4f} "
                f"(AUC={val_auc_macro:.4f}, AP={val_ap_macro:.4f}, Acc={val_acc_macro:.4f}) "
                f"at epoch {epoch+1} ***"
            )

        log_metrics(val_stats['loss'], epoch, val_stats['results'], class_names, wandb_run, mode='val')
    
    logger.info("--- Testing ---")
    if best_classifier_state is not None:
        logger.info(f"Loading best backbone/classifier for testing (Val Composite Score: {best_val_score:.4f})")
        model['backbone'].load_state_dict(best_model_state)
        model['classifier'].load_state_dict(best_classifier_state)
    else:
        logger.warning("No best model found, using model from last epoch for testing.")
            
    test_stats = evaluate_epoch(
        cfg=cfg, model=model, loader=test_loader, criterion=criterion, device=device,
        metrics=test_metrics, logger=logger, use_amp=cfg.optimization.mixed_precision,
        save_path=save_path,
    )
    
    logger.info("Final Test Metrics (using best validation model):")
    log_metrics(test_stats['loss'], max_epoch-1, test_stats['results'], class_names, wandb_run, mode='test')
    
    logger.info(f"Training Finished after {max_epoch} epochs.")
    return train_stats.get('loss', 0.0)