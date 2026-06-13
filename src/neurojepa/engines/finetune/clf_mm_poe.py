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

import copy

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def unpack_metrics(results: Dict[str, torch.Tensor]) -> Any:
    """Unpacks metrics from torchmetrics for both multi-class and binary cases."""
    auroc_perclass = results.get("MulticlassAUROC", results.get("BinaryAUROC"))
    auprc_perclass = results.get("MulticlassAveragePrecision", results.get("BinaryAveragePrecision"))
    acc_perclass = results.get("MulticlassAccuracy", results.get("BinaryAccuracy"))

    macro_auc = results.get("MulticlassAUROC_macro", results.get("BinaryAUROC", torch.tensor(0.0))).item()
    macro_ap = results.get("MulticlassAveragePrecision_macro", results.get("BinaryAveragePrecision", torch.tensor(0.0))).item()
    macro_acc = results.get("MulticlassAccuracy_macro", results.get("BinaryAccuracy", torch.tensor(0.0))).item()

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
        if len(perclass_auc) > 1:
            log_dict.update({f"{mode}/auc_{class_names[i]}": perclass_auc[i] for i in range(len(class_names))})
        else:
            log_dict[f"{mode}/auc"] = macro_auc

        wandb_run.log(log_dict)


def poe_fuse(logits_list, validity_mask=None):
    """Product-of-Experts fusion in log-probability space.

    Args:
        logits_list: list of [B, C] tensors (raw logits per modality head).
        validity_mask: [B, num_mods] float tensor (1=valid, 0=missing).
            Missing modalities contribute a uniform distribution (no info).

    Returns:
        log_probs: [B, C] normalised log-probabilities after PoE fusion.
    """
    num_classes = logits_list[0].shape[1]
    # uniform log-prob used for missing modalities
    uniform = torch.full_like(logits_list[0], math.log(1.0 / num_classes))

    log_softmaxes = []
    for mod_idx, logits in enumerate(logits_list):
        ls = F.log_softmax(logits, dim=-1)
        if validity_mask is not None:
            # mask: [B, 1]
            mask = validity_mask[:, mod_idx].unsqueeze(-1)
            ls = ls * mask + uniform * (1.0 - mask)
        log_softmaxes.append(ls)

    # Sum of log-softmaxes (log of product)
    output_num = torch.stack(log_softmaxes, dim=0).sum(dim=0)  # [B, C]
    # Re-normalise
    output_den = torch.logsumexp(output_num, dim=-1)            # [B]
    log_probs = output_num - output_den.unsqueeze(1)            # [B, C]
    return log_probs


def poe_joint_fuse(logits_list, joint_logits, validity_mask=None):
    """Product-of-Experts fusion with an additional joint/multimodal expert.

    Combines per-modality log-softmax outputs with a joint head that
    operates on concatenated features from all modalities.

    Args:
        logits_list: list of [B, C] tensors (raw logits per unimodal head).
        joint_logits: [B, C] tensor (raw logits from the joint/concat head).
        validity_mask: [B, num_mods] float tensor (1=valid, 0=missing).
            Missing modalities contribute a uniform distribution.

    Returns:
        log_probs: [B, C] normalised log-probabilities after PoE+Joint fusion.
    """
    num_classes = logits_list[0].shape[1]
    uniform = torch.full_like(logits_list[0], math.log(1.0 / num_classes))

    log_softmaxes = []
    for mod_idx, logits in enumerate(logits_list):
        ls = F.log_softmax(logits, dim=-1)
        if validity_mask is not None:
            mask = validity_mask[:, mod_idx].unsqueeze(-1)
            ls = ls * mask + uniform * (1.0 - mask)
        log_softmaxes.append(ls)

    # Add the joint expert (always valid — it sees concatenated features)
    log_softmaxes.append(F.log_softmax(joint_logits, dim=-1))

    output_num = torch.stack(log_softmaxes, dim=0).sum(dim=0)  # [B, C]
    output_den = torch.logsumexp(output_num, dim=-1)            # [B]
    log_probs = output_num - output_den.unsqueeze(1)            # [B, C]
    return log_probs


def _is_poe_joint(cfg):
    return 'poe_joint' in cfg.model.model_name


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
    """Run evaluation for PoE / PoE+Joint multi-modal classification."""
    backbone, classifier = model['backbone'], model['classifier']
    joint_head = model.get('joint_head', None)
    use_joint = _is_poe_joint(cfg)

    backbone.eval()
    classifier.eval()
    if joint_head is not None:
        joint_head.eval()

    total_loss = 0.0
    all_preds = []
    all_targets = []

    for idx, batch_data in enumerate(loader):
        images_list, target = batch_data
        # Extract validity mask
        validity_mask = None
        if isinstance(images_list, dict) and "__validity_mask__" in images_list:
            validity_mask = images_list.pop("__validity_mask__").to(device, non_blocking=True)
        for key in images_list.keys():
            images_list[key] = images_list[key].to(device, non_blocking=True)
        target = target.to(device, non_blocking=True).long()

        with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.bfloat16):
            # --- Backbone: shared encoder, all modalities concatenated ---
            batch_size = images_list[list(images_list.keys())[0]].shape[0]
            combined_images = torch.cat([images_list[k] for k in images_list.keys()], dim=0)
            feats = backbone(combined_images)
            if isinstance(feats, tuple):
                feats = feats[0]

            # Split features back per modality
            feats_list = {}
            for i, key in enumerate(images_list.keys()):
                feats_list[key] = feats[i * batch_size:(i + 1) * batch_size]

            # --- Per-modality classification heads ---
            clf_list = classifier.module if hasattr(classifier, 'module') else classifier
            mod_keys = list(feats_list.keys())
            logits_list = []
            for mod_idx, key in enumerate(mod_keys):
                logits_list.append(clf_list[mod_idx](feats_list[key]))

            # --- Fusion ---
            if use_joint and joint_head is not None:
                feats_cat = torch.cat([feats_list[k] for k in mod_keys], dim=1)
                jh = joint_head.module if hasattr(joint_head, 'module') else joint_head
                joint_logits = jh(feats_cat)
                log_probs = poe_joint_fuse(logits_list, joint_logits, validity_mask=None)
            else:
                log_probs = poe_fuse(logits_list, validity_mask=None)

            loss = criterion(log_probs, target)

            # Probabilities for metrics
            probs = torch.exp(log_probs)

        # Update metrics
        num_classes = probs.shape[1]
        if num_classes == 2:
            metrics.update(probs[:, 1], target)
        else:
            metrics.update(probs, target)

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
) -> Dict[str, float]:
    """Train one epoch with PoE / PoE+Joint multi-modal fusion."""
    metric_logger = MetricLogger(delimiter="  ", logger=logger)
    backbone, classifier = model['backbone'], model['classifier']
    joint_head = model.get('joint_head', None)
    use_joint = _is_poe_joint(cfg)

    freeze = cfg.model.get("freeze_backbone", False)
    if freeze:
        backbone.eval()
    else:
        backbone.train()

    classifier.train()
    if joint_head is not None:
        joint_head.train()

    use_amp = cfg.optimization.mixed_precision
    grad_clip = cfg.optimization.get('grad_clip', 0.0)

    if metrics is None:
        raise ValueError("A MetricCollection must be provided.")

    for idx, batch_data in enumerate(loader):
        images_list, target = batch_data
        # Extract validity mask
        validity_mask = None
        if isinstance(images_list, dict) and "__validity_mask__" in images_list:
            validity_mask = images_list.pop("__validity_mask__").to(device, non_blocking=True)
        for key in images_list.keys():
            images_list[key] = images_list[key].to(device, non_blocking=True)
        target = target.to(device, non_blocking=True).long()

        _new_lr = scheduler.step()
        _new_wd = wd_scheduler.step()

        with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.bfloat16):
            # --- Backbone: shared encoder ---
            batch_size = images_list[list(images_list.keys())[0]].shape[0]
            combined_images = torch.cat([images_list[k] for k in images_list.keys()], dim=0)
            feats = backbone(combined_images)
            if isinstance(feats, tuple):
                feats = feats[0]

            feats_list = {}
            for i, key in enumerate(images_list.keys()):
                feats_list[key] = feats[i * batch_size:(i + 1) * batch_size]

            # --- Per-modality heads ---
            clf_list = classifier.module if hasattr(classifier, 'module') else classifier
            mod_keys = list(feats_list.keys())
            logits_list = []
            for mod_idx, key in enumerate(mod_keys):
                logits_list.append(clf_list[mod_idx](feats_list[key]))

            # --- Fusion ---
            if use_joint and joint_head is not None:
                feats_cat = torch.cat([feats_list[k] for k in mod_keys], dim=1)
                jh = joint_head.module if hasattr(joint_head, 'module') else joint_head
                joint_logits = jh(feats_cat)
                log_probs = poe_joint_fuse(logits_list, joint_logits, validity_mask=None)
            else:
                log_probs = poe_fuse(logits_list, validity_mask=None)

            loss = criterion(log_probs, target)

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
                if joint_head is not None:
                    torch.nn.utils.clip_grad_norm_(joint_head.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(backbone.parameters(), grad_clip)
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), grad_clip)
                if joint_head is not None:
                    torch.nn.utils.clip_grad_norm_(joint_head.parameters(), grad_clip)
            optimizer.step()

        torch.cuda.synchronize()

        loss_value = all_reduce_mean(loss).item()
        metric_logger.update(loss=loss_value, lr=_new_lr, wd=_new_wd)

        with torch.no_grad():
            probs = torch.exp(log_probs)
            num_classes = probs.shape[1]
            if num_classes == 2:
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
    save_path: Optional[str] = None,
) -> float:
    """Main training loop for PoE / PoE+Joint multi-modal finetuning."""
    num_classes = cfg.data.num_classes
    train_metrics = build_downstream_metrics(num_classes, device=device)
    val_metrics = build_downstream_metrics(num_classes, device=device)
    test_metrics = build_downstream_metrics(num_classes, device=device)
    joint_head = model.get('joint_head', None)

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
        )

        train_results = train_metrics.compute()
        train_metrics.reset()
        log_metrics(train_stats["loss"], epoch, train_results, class_names, wandb_run, mode='train')

        val_stats = evaluate_epoch(
            cfg=cfg, model=model, loader=val_loader, criterion=criterion, device=device,
            metrics=val_metrics, logger=logger, use_amp=cfg.optimization.mixed_precision,
        )

        _, _, _, val_auc_macro, val_ap_macro, val_acc_macro = unpack_metrics(val_stats["results"])

        val_score = 0.5 * val_auc_macro + 0.5 * val_ap_macro

        if val_score > best_val_score:
            best_val_score = val_score
            best_model_state = copy.deepcopy(model['backbone'].state_dict())
            best_classifier_state = copy.deepcopy(model['classifier'].state_dict())
            best_joint_state = copy.deepcopy(joint_head.state_dict()) if joint_head is not None else None
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
        if joint_head is not None and best_joint_state is not None:
            joint_head.load_state_dict(best_joint_state)
    else:
        logger.warning("No best model found, using model from last epoch for testing.")

    test_stats = evaluate_epoch(
        cfg=cfg, model=model, loader=test_loader, criterion=criterion, device=device,
        metrics=test_metrics, logger=logger, use_amp=cfg.optimization.mixed_precision,
        save_path=save_path,
    )

    logger.info("Final Test Metrics (using best validation model):")
    log_metrics(test_stats['loss'], max_epoch - 1, test_stats['results'], class_names, wandb_run, mode='test')

    logger.info(f"Training Finished after {max_epoch} epochs.")
    return train_stats.get('loss', 0.0)
