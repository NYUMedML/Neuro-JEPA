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
from neurojepa.utils.misc import all_reduce_mean, MetricLogger

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
    
    # For macro metrics
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
    
    # Only log per-class metrics if they exist (i.e., not binary case)
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
    
    if isinstance(backbone, list):
        for b in backbone:
            b.eval()
    else:
        backbone.eval()
    classifier.eval()

    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    for idx, batch_data in enumerate(loader):
        images_list, target = batch_data
        # Extract validity mask before processing images
        validity_mask = None
        if isinstance(images_list, dict) and "__validity_mask__" in images_list:
            validity_mask = images_list.pop("__validity_mask__").to(device, non_blocking=True)
        num_modalities = len(images_list)
        for key in images_list.keys():
            images_list[key] = images_list[key].to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        target = target.long()

        with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.bfloat16):
            if isinstance(backbone, list):
                feats_list = {}
                for model_idx, key in enumerate(images_list.keys()):
                    image = images_list[key]
                    feats = backbone[model_idx](image)
                    # check if feats is tuple
                    if isinstance(feats, tuple):
                        feats = feats[0]
                    feats_list[key] = feats
            elif cfg.model.model_name == 'vit_mil':
                batch_size = images_list[list(images_list.keys())[0]].shape[0]
                # Extract patch token features per modality
                mod_patch_tokens = []
                num_modalities = len(images_list.keys())
                for key in images_list.keys():
                    image = images_list[key]
                    feats = backbone(image)
                    if isinstance(feats, tuple):
                        feats = feats[0]
                    # feats: [B, num_tokens, dim]
                    mod_patch_tokens.append(feats)
                
                # Concatenate modality tokens per sample: [B, total_tokens, dim]
                feats_combined = torch.cat(mod_patch_tokens, dim=1)
                
                # Zero out tokens for missing modalities before flattening
                if validity_mask is not None:
                    tokens_per_mod = feats_combined.shape[1] // num_modalities
                    mod_mask = validity_mask.repeat_interleave(tokens_per_mod, dim=1).unsqueeze(-1)
                    feats_combined = feats_combined * mod_mask
                
                # Flatten for MIL: [N, dim] where N = B * total_tokens
                tokens_per_sample = feats_combined.shape[1]
                feats_flat = feats_combined.reshape(-1, feats_combined.shape[-1])
                
                # Build cu_seqlens: [0, tok, 2*tok, ..., B*tok]
                cu_seqlens = torch.arange(0, batch_size + 1, device=device, dtype=torch.int32) * tokens_per_sample
            elif cfg.model.model_name == 'vit_late':
                batch_size = images_list[list(images_list.keys())[0]].shape[0]
                # concatenate images along batch dimension
                combined_images = torch.cat([images_list[k] for k in images_list.keys()], dim=0)
                feats = backbone(combined_images)
                if isinstance(feats, tuple):
                    feats = feats[0]
                # reassign feats to each modality
                feats_list = {}
                num_modalities = len(images_list.keys())
                for i, key in enumerate(images_list.keys()):
                    feats_list[key] = feats[i*batch_size:(i+1)*batch_size]
            elif cfg.model.model_name == 'vit_avg':
                batch_size = images_list[list(images_list.keys())[0]].shape[0]
                # Process each modality through shared backbone, get pooled features
                combined_images = torch.cat([images_list[k] for k in images_list.keys()], dim=0)
                feats = backbone(combined_images)
                if isinstance(feats, tuple):
                    feats = feats[0]
                # Pool token sequence -> [B*num_mods, embed_dim] for nn.Linear classifier
                if feats.dim() == 3:
                    feats = feats.mean(dim=1)
                feats_list = {}
                num_modalities = len(images_list.keys())
                for i, key in enumerate(images_list.keys()):
                    feats_list[key] = feats[i*batch_size:(i+1)*batch_size]
            elif cfg.model.model_name == 'vit_early':
                # convert images_list to list of tensors
                images_list = [images_list[k] for k in images_list.keys()]
                feats = backbone(images_list, validity_mask=validity_mask)
                if isinstance(feats, tuple):
                    feats = feats[0]
                # concat feats from each modality on sequence dimension
                feats = torch.cat([feats[i] for i in range(len(images_list))], dim=1)

            # Zero out features for invalid/missing modalities (non-MIL paths)
            if validity_mask is not None and cfg.model.model_name not in ('vit_mil', 'vit_avg'):
                if cfg.model.model_name == 'vit_early':
                    n_tokens_per_mod = feats.shape[1] // num_modalities
                    mod_mask = validity_mask.repeat_interleave(n_tokens_per_mod, dim=1).unsqueeze(-1)
                    feats = feats * mod_mask
                else:
                    for mod_i, key in enumerate(feats_list.keys()):
                        mask = validity_mask[:, mod_i]
                        while mask.dim() < feats_list[key].dim():
                            mask = mask.unsqueeze(-1)
                        feats_list[key] = feats_list[key] * mask

            if cfg.model.model_name == 'vit_mil':
                # MIL classifier returns [B, num_classes]
                logits = classifier(feats_flat, cu_seqlens=cu_seqlens)
            elif cfg.model.model_name == 'vit_late':
                feats_combined = [feats_list[k] for k in feats_list.keys()]
                logits = classifier(feats_combined[0], feats_combined[1])
            elif cfg.model.model_name == 'vit_avg':
                # Logits averaging: each modality classified independently, then averaged
                clf_list = classifier.module if hasattr(classifier, 'module') else classifier
                mod_keys = list(feats_list.keys())
                logits_list = []
                for mod_idx, key in enumerate(mod_keys):
                    logits_list.append(clf_list[mod_idx](feats_list[key]))
                logits_stack = torch.stack(logits_list, dim=0)  # [num_mods, B, num_classes]
                if validity_mask is not None:
                    # validity_mask: [B, num_mods] -> [num_mods, B, 1]
                    mask = validity_mask.t().unsqueeze(-1)
                    logits_stack = logits_stack * mask
                    logits = logits_stack.sum(dim=0) / mask.sum(dim=0).clamp(min=1)
                else:
                    logits = logits_stack.mean(dim=0)
            else:
                raise NotImplementedError(f"Model {cfg.model.model_name} not implemented for combining features.")

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
    
    freeze = cfg.model.get("freeze_backbone", False)
    if isinstance(backbone, list):
        for b in backbone:
            if freeze:
                b.eval()
            else:
                b.train()
    else:
        if freeze:
            backbone.eval()
        else:
            backbone.train()

    use_amp = cfg.optimization.mixed_precision
    grad_clip = cfg.optimization.get('grad_clip', 0.0)

    if metrics is None:
        raise ValueError("A MetricCollection must be provided.")
        
    for idx, batch_data in enumerate(loader):
        images_list, target = batch_data
        # Extract validity mask before processing images
        validity_mask = None
        if isinstance(images_list, dict) and "__validity_mask__" in images_list:
            validity_mask = images_list.pop("__validity_mask__").to(device, non_blocking=True)
        num_modalities = len(images_list)
        for key in images_list.keys():
            images_list[key] = images_list[key].to(device, non_blocking=True)
        #break
        target = target.to(device, non_blocking=True)
        target = target.long()

        _new_lr = scheduler.step()
        _new_wd = wd_scheduler.step()
        
        with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.bfloat16):
            if isinstance(backbone, list):
                feats_list = {}
                for model_idx, key in enumerate(images_list.keys()):
                    image = images_list[key]
                    feats = backbone[model_idx](image)
                    # check if feats is tuple
                    if isinstance(feats, tuple):
                        feats = feats[0]
                    feats_list[key] = feats
            elif cfg.model.model_name == 'vit_mil':
                batch_size = images_list[list(images_list.keys())[0]].shape[0]
                # Extract patch token features per modality
                mod_patch_tokens = []
                num_modalities = len(images_list.keys())
                for key in images_list.keys():
                    image = images_list[key]
                    feats = backbone(image)
                    if isinstance(feats, tuple):
                        feats = feats[0]
                    # feats: [B, num_tokens, dim]
                    mod_patch_tokens.append(feats)
                
                # Concatenate modality tokens per sample: [B, total_tokens, dim]
                feats_combined = torch.cat(mod_patch_tokens, dim=1)
                
                # Zero out tokens for missing modalities before flattening
                if validity_mask is not None:
                    tokens_per_mod = feats_combined.shape[1] // num_modalities
                    mod_mask = validity_mask.repeat_interleave(tokens_per_mod, dim=1).unsqueeze(-1)
                    feats_combined = feats_combined * mod_mask
                
                # Flatten for MIL: [N, dim] where N = B * total_tokens
                tokens_per_sample = feats_combined.shape[1]
                feats_flat = feats_combined.reshape(-1, feats_combined.shape[-1])
                
                # Build cu_seqlens: [0, tok, 2*tok, ..., B*tok]
                cu_seqlens = torch.arange(0, batch_size + 1, device=device, dtype=torch.int32) * tokens_per_sample
            elif cfg.model.model_name == 'vit_late':
                batch_size = images_list[list(images_list.keys())[0]].shape[0]
                # concatenate images along batch dimension
                combined_images = torch.cat([images_list[k] for k in images_list.keys()], dim=0)
                feats = backbone(combined_images)
                if isinstance(feats, tuple):
                    feats = feats[0]
                # reassign feats to each modality
                feats_list = {}
                num_modalities = len(images_list.keys())
                for i, key in enumerate(images_list.keys()):
                    feats_list[key] = feats[i*batch_size:(i+1)*batch_size]
            elif cfg.model.model_name == 'vit_avg':
                batch_size = images_list[list(images_list.keys())[0]].shape[0]
                # Process each modality through shared backbone, get pooled features
                combined_images = torch.cat([images_list[k] for k in images_list.keys()], dim=0)
                feats = backbone(combined_images)
                if isinstance(feats, tuple):
                    feats = feats[0]
                # Pool token sequence -> [B*num_mods, embed_dim] for nn.Linear classifier
                if feats.dim() == 3:
                    feats = feats.mean(dim=1)
                feats_list = {}
                num_modalities = len(images_list.keys())
                for i, key in enumerate(images_list.keys()):
                    feats_list[key] = feats[i*batch_size:(i+1)*batch_size]

            # Zero out features for invalid/missing modalities (non-MIL paths)
            if validity_mask is not None and cfg.model.model_name not in ('vit_mil', 'vit_avg'):
                for mod_i, key in enumerate(feats_list.keys()):
                    mask = validity_mask[:, mod_i]
                    while mask.dim() < feats_list[key].dim():
                        mask = mask.unsqueeze(-1)
                    feats_list[key] = feats_list[key] * mask

            if cfg.model.model_name == 'vit_mil':
                # MIL classifier returns [B, num_classes]
                logits = classifier(feats_flat, cu_seqlens=cu_seqlens)
            elif cfg.model.model_name == 'vit_late':
                feats_combined = [feats_list[k] for k in feats_list.keys()]
                logits = classifier(feats_combined[0], feats_combined[1])
            elif cfg.model.model_name == 'vit_avg':
                # Logits averaging: each modality classified independently, then averaged
                clf_list = classifier.module if hasattr(classifier, 'module') else classifier
                mod_keys = list(feats_list.keys())
                logits_list = []
                for mod_idx, key in enumerate(mod_keys):
                    logits_list.append(clf_list[mod_idx](feats_list[key]))
                logits_stack = torch.stack(logits_list, dim=0)  # [num_mods, B, num_classes]
                if validity_mask is not None:
                    # validity_mask: [B, num_mods] -> [num_mods, B, 1]
                    mask = validity_mask.t().unsqueeze(-1)
                    logits_stack = logits_stack * mask
                    logits = logits_stack.sum(dim=0) / mask.sum(dim=0).clamp(min=1)
                else:
                    logits = logits_stack.mean(dim=0)
            else:
                raise NotImplementedError(f"Model {cfg.model.model_name} not implemented for combining features.")
            
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
                if isinstance(backbone, list):
                    for b in backbone:
                        torch.nn.utils.clip_grad_norm_(b.parameters(), grad_clip)
                else:
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), grad_clip)
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                # Clip gradients for both backbone and classifier
                if isinstance(backbone, list):
                    for b in backbone:
                        torch.nn.utils.clip_grad_norm_(b.parameters(), grad_clip)
                else:
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), grad_clip)
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), grad_clip)
            optimizer.step()
        
        # MoE bias update and violation calculation
        if use_moe:
            if moe_params.moe_type == "bias":
                if isinstance(backbone, list):
                    for b in backbone:
                        minvio, maxvio = moe_bias_update(b, moe_params.bias_update_rate, used_ddp=True, vit=True)
                else:
                    minvio, maxvio = moe_bias_update(backbone, moe_params.bias_update_rate, used_ddp=True, vit=True)
            else:
                if isinstance(backbone, list):
                    for b in backbone:
                        minvio, maxvio = calculate_vio_model(b, used_ddp=True, vit=True)
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
        
        # Composite score: average of macro AUC, AP, and Accuracy
        val_score = 0.5 * val_auc_macro + 0.5 * val_ap_macro
        
        if val_score > best_val_score:
            best_val_score = val_score
            if isinstance(model['backbone'], list):
                best_model_state = [copy.deepcopy(b.state_dict()) for b in model['backbone']]
            else:
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
        if isinstance(model['backbone'], list):
            for i, b in enumerate(model['backbone']):
                b.load_state_dict(best_model_state[i])
        else:
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
