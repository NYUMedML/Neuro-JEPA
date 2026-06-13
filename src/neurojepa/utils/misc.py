import os
import sys
import json
import time
import logging
import random
import datetime
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque

import torch
import torch.nn as nn
import torch.distributed as dist

from torch import Tensor, norm_except_dim
from torch.nn import Module
from torch.nn.parameter import Parameter, UninitializedParameter

from typing import Any, Tuple
from functools import partial
from collections.abc import Callable, Mapping

from neurojepa.utils.checkpoint_loader import robust_checkpoint_loader
from neurojepa.masks.masking import compute_foreground_patches

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def compute_fg_attn_mask(image, patch_size):
    """
    Compute a foreground attention mask for a ViT backbone.

    Uses ``compute_foreground_patches`` to identify which patches contain
    meaningful tissue, then flattens the 3-D grid to a flat patch sequence
    that matches the token dimension of the backbone.

    Args:
        image: float tensor of shape (B, C, H, W, D).
        patch_size: sequence of three ints (pH, pW, pD).

    Returns:
        attn_mask: bool tensor of shape (B, N) where N = nH*nW*nD.
                   True  = foreground (valid token).
                   False = background (should be ignored).
    """
    pH, pW, pD = int(patch_size[0]), int(patch_size[1]), int(patch_size[2])
    fg_map = compute_foreground_patches(image, (pH, pW, pD))  # (B, nH, nW, nD)
    B = fg_map.shape[0]
    return fg_map.reshape(B, -1)                              # (B, N)


def remove_fsdp_compile_names(name: str):
    name = name.replace("_fsdp_wrapped_module.", "")  # Added by FSDP
    name = name.replace("_checkpoint_wrapped_module.", "")  # Added by activation checkpointing for SimpleFSDP
    name = name.replace("parametrizations.", "")  # Added by SimpleFSDP
    name = name.removesuffix(".original")  # Added by SimpleFSDP
    name = name.replace("module.", "")  # Added by SimpleFSDP
    name = name.replace("_orig_mod.", "")  # Added by torch.compile
    name = name.replace("backbone.", "")  # Added by training
    return name


def get_params_groups(
    model,
    patch_embed_lr_mult: float,
    rope_lr_mult: float,
    layernorm_wd_mult: float,
    logger: Any,
):
    params_groups: list[dict[str, Any]] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        name = remove_fsdp_compile_names(name)
        d = {"param": param, "lr_multiplier": 1.0, "wd_multiplier": 1.0, "name": name}
        if name.endswith(".bias") or "gamma" in name:
            d["wd_multiplier"] = 0.0
        if "norm" in name and "weight" in name:
            d["wd_multiplier"] = layernorm_wd_mult
        if "patch_embed" in name:
            d["lr_multiplier"] = d["lr_multiplier"] * patch_embed_lr_mult
        if "rope" in name:
            d["lr_multiplier"] = d["lr_multiplier"] * rope_lr_mult
        params_groups.append(d)
        #logger.info(f"{d['name']}:   " + ", ".join([f"{k}: {v}" for k, v in d.items() if k not in ["param", "name"]]))
    # FU
    fused_params_groups: defaultdict[str, dict[str, Any]] = defaultdict(lambda: {"params": []})
    for d in params_groups:
        keys = sorted(set(d.keys()) - {"param", "name"})
        identifier = "_".join([f"{k}{d[k]}" for k in keys])
        for k in keys:
            fused_params_groups[identifier][k] = d[k]
        fused_params_groups[identifier]["params"].append(d["param"])
    # SION
    params_groups = list(fused_params_groups.values())
    #logger.info("Fused param groups")
    for g in params_groups:
        g["foreach"] = True
    # HA
    return params_groups
    

def send_data(x):
    if isinstance(x, torch.Tensor):
        x = x.cuda(non_blocking=True)
        x.record_stream(torch.cuda.current_stream())
        return x
    if isinstance(x, dict):
        return {k: send_data(v) for k, v in x.items()}
    if isinstance(x, list):
        return [send_data(v) for v in x]
    return x


def pre_send_to_cuda_wrapper(generator):
    """From apex"""
    data = None
    stream: torch.cuda.Stream = torch.cuda.Stream()  # pyright:ignore[reportAssignmentType]
    for next_data in generator:
        # Move to GPU
        with torch.cuda.stream(stream):
            next_data = send_data(next_data)
        if data is not None:
            yield data
        torch.cuda.current_stream().wait_stream(stream)
        data = next_data
        

def save_checkpoint(encoder, target_encoder, predictor, optimizer, scaler, \
    epoch, global_step, loss, path):
    rank = get_rank()
    if rank != 0:
        return

    save_dict = {
        "encoder": encoder.state_dict(),
        "target_encoder": target_encoder.state_dict(),
        "predictor": predictor.state_dict(),
        "opt": optimizer.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "loss": loss,
    }
    try:
        torch.save(save_dict, path)
    except Exception as e:
        logger.info(f"Encountered exception when saving checkpoint: {e}")
    

def robust_load_optimizer(optimizer, saved_state_dict):
    # Load the state for parameters that exist in both
    # The optimizer.state is a dictionary mapping parameter objects to their buffers
    saved_state = saved_state_dict['state']
    
    # Map the saved state IDs to the current parameter objects
    try:
        optimizer.load_state_dict(saved_state_dict)
    except ValueError:
        print("Mismatched keys found! Switching to manual state injection...")
        
        # Manually update the state dictionary for existing parameters
        for i, group in enumerate(optimizer.param_groups):
            if i >= len(saved_state_dict['param_groups']):
                break
            
            saved_group = saved_state_dict['param_groups'][i]
            current_params = group['params']
            saved_params = saved_group['params']
            
            # Case 1: Lengths match or saved is larger (logic unchanged)
            if len(current_params) <= len(saved_params):
                 for p, s_p in zip(current_params, saved_params):
                    if s_p in saved_state:
                         state = saved_state[s_p]
                         if 'exp_avg' in state and state['exp_avg'].shape != p.shape:
                             continue
                         if 'momentum_buffer' in state and state['momentum_buffer'].shape != p.shape:
                             continue
                         
                         # Move state to device of parameter
                         p_device = p.device
                         for k, v in state.items():
                             if isinstance(v, torch.Tensor):
                                 state[k] = v.to(p_device)
                         optimizer.state[p] = state

            # Case 2: Current has more parameters (insertion happened)
            else:
                print(f"Group {i} has {len(current_params)} params vs {len(saved_params)} saved. Attempting Re-alignment...")
                
                # Check directly matching from start
                match_indices = []
                s_idx = 0
                for c_idx, p in enumerate(current_params):
                    if s_idx >= len(saved_params):
                        break
                    
                    s_p = saved_params[s_idx]
                    if s_p not in saved_state:
                        # If saved param has no state, we can't verify match easily, skip strict check
                        s_idx += 1
                        continue

                    state = saved_state[s_p]
                    # Check shape compatibility
                    is_match = True
                    if 'exp_avg' in state and state['exp_avg'].shape != p.shape:
                        is_match = False
                    elif 'momentum_buffer' in state and state['momentum_buffer'].shape != p.shape:
                        is_match = False
                    
                    if is_match:
                        # Move state to device of parameter
                        p_device = p.device
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(p_device)
                        optimizer.state[p] = state
                        s_idx += 1 # Advance saved pointer only on match
                    else:
                        print(f"  Skipping new parameter at index {c_idx} (Shape {p.shape}) - Re-aligning...")
                
                print(f"Group {i} realignment complete.")
        
        print("Loaded matching states; new parameters left with default initialization.")
        

def load_checkpoint(
    r_path,
    encoder,
    predictor,
    target_encoder,
    opt,
    scaler,
    is_anneal=False,
):
    logger.info(f"Loading checkpoint from {r_path}")
    checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))

    epoch = 0
    global_step = 0
    if not is_anneal:
        epoch = checkpoint["epoch"]
        global_step = checkpoint["global_step"]

    # loading encoder
    pretrained_dict = checkpoint["encoder"]
    msg = encoder.load_state_dict(pretrained_dict, strict=False)
    logger.info(f"loaded pretrained encoder from epoch {epoch+1} with msg: {msg}")

    # loading predictor
    pretrained_dict = checkpoint["predictor"]
    msg = predictor.load_state_dict(pretrained_dict, strict=False)
    logger.info(f"loaded pretrained predictor from epoch {epoch+1} with msg: {msg}")

    # loading target_encoder
    if target_encoder is not None:
        logger.info(list(checkpoint.keys()))
        pretrained_dict = checkpoint["target_encoder"]
        msg = target_encoder.load_state_dict(pretrained_dict, strict=False)
        logger.info(f"loaded pretrained target encoder from epoch {epoch+1} with msg: {msg}")

    # loading optimizer
    robust_load_optimizer(opt, checkpoint["opt"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    else:
        logger.info("Skipping AMP scaler restore because checkpoint contains no scaler state.")
    logger.info(f"loaded optimizers from epoch {epoch+1}")
    logger.info(f"read-path: {r_path}")
    del checkpoint

    return (
        encoder,
        predictor,
        target_encoder,
        opt,
        scaler,
        epoch,
        global_step,
    )
    
    
def load_checkpoint_backbone(
    r_path,
    backbone,
):
    logger.info(f"Loading checkpoint from {r_path}")
    checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))
    # loading backbone
    if "encoder" in checkpoint:
        pretrained_dict = checkpoint["encoder"]
    elif "state_dict" in checkpoint:
        pretrained_dict = checkpoint["state_dict"]
    else:
        raise ValueError("No encoder or state_dict found in checkpoint.")
    
    pretrained_dict = {k.replace("backbone.", "").replace("module.", ""): v for k, v in pretrained_dict.items()}
    msg = backbone.load_state_dict(pretrained_dict, strict=False)
    logger.info(f"loaded pretrained backbone with msg: {msg}")
    del checkpoint

    return backbone

    
def is_main_process():
    return get_rank() == 0

    
def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def datafold_read(datalist, basedir, fold=0, key="training"):
    with open(datalist) as f:
        json_data = json.load(f)

    json_data = json_data[key]

    for d in json_data:
        for k in d:
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]
                
    tr = []
    val = []
    for d in json_data:
        if "fold" in d and d["fold"] == fold:
            val.append(d)
        else:
            tr.append(d)

    return tr, val


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)
        

class MetricLogger(object):
    def __init__(self, delimiter="\t", logger=None):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.logger = logger

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def log_metrics(self, prefix="", **kwargs):
        """Log multiple metrics at once with an optional prefix"""
        self.update(**kwargs)
        
        # Create a formatted string with all the metrics
        log_str = prefix
        log_items = []
        for k, v in kwargs.items():
            if v is not None:
                meter_str = f"{k}: {self.meters[k]}"
                log_items.append(meter_str)
        
        if log_items:
            log_str += self.delimiter.join(log_items)
            self.logger.info(log_str)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        self.logger.info('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def all_reduce_mean(x):
    if not is_dist_avail_and_initialized():
        world_size =  1
    else:
        world_size = dist.get_world_size()
        
    if world_size > 1:
        if isinstance(x, torch.Tensor):
            x_reduce = x.detach().clone().float().cuda()
        else:
            x_reduce = torch.tensor(x, dtype=torch.float32).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce
    else:
        return x


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_seed(seed):
    random_seed = seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    np.random.seed(random_seed)
    random.seed(random_seed)
    

def init_distributed_mode():
    """
    Initializes the distributed process group.
    Assumes environment variables are set by torchrun/torch.distributed.launch.
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, 
                                rank=rank, device_id=torch.device(f"cuda:{local_rank}"))
        dist.barrier()
        print(f"Distributed mode initialized. Rank: {rank}, World Size: {world_size}")
    else:
        print('Not using distributed mode')
        world_size, rank = 1, 0
    return world_size, rank

    
def get_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def cleanup():
    dist.destroy_process_group()


def all_gather(tensor):
    return AllGatherFunction.apply(tensor)


def set_requires_grad_false(*models, lora=False):
    if lora:
        for model in models:
            for name, param in model.named_parameters():
                is_trainable = (
                    "lora" in name or
                    "bias" in name or
                    "embeddings" in name or
                    "norm" in name
                )
                param.requires_grad = is_trainable
    else:
        for model in models:
            for param in model.parameters():
                param.requires_grad = False
            
            
def clip_gradients(models, clip):
    """
    Clips gradients of the given models to the specified maximum norm.
    Args:
        models (list of torch.nn.Module): List of models whose gradients need to be clipped.
        clip (float): Maximum allowed norm of the gradients.
    Returns:
        list: List of gradient norms for each parameter."""
    norms = []
    for model in models:
        for name, p in model.named_parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                norms.append(param_norm.item())
                clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
    return norms

            
@torch.no_grad()
def _update_momentum_encoder(model: torch.nn.Module, momentum_model: torch.nn.Module, m: float) -> None:
    """
    Momentum update of the momentum encoder.

    Args:
        model (torch.nn.Module): The main model.
        momentum_model (torch.nn.Module): The momentum model.
        m (float): Momentum coefficient.
    """
    for param_q, param_k in zip(model.parameters(), momentum_model.parameters()):
        param_k.data.mul_(m).add_((1 - m) * param_q.detach().detach().data)

    # Update Buffers (bias in MoE layers)
    # Update buffers with name checking
    for (name_q, buffer_q), (name_k, buffer_k) in zip(model.named_buffers(), \
        momentum_model.named_buffers()):
        if name_q != name_k:
            print(f"Warning: Buffer name mismatch: {name_q} vs {name_k}")

        buffer_k.data.copy_(buffer_q.data)


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


class AllGatherFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, reduce_dtype: torch.dtype = torch.float32):
        ctx.reduce_dtype = reduce_dtype

        output = list(torch.empty_like(tensor) for _ in range(dist.get_world_size()))
        dist.all_gather(output, tensor)
        output = torch.cat(output, dim=0)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_dtype = grad_output.dtype
        input_list = list(grad_output.to(ctx.reduce_dtype).chunk(dist.get_world_size()))
        grad_input = torch.empty_like(input_list[dist.get_rank()])
        dist.reduce_scatter(grad_input, input_list)
        return grad_input.to(grad_dtype)
    

# bf16-compatible weight norm
class WeightNorm:
    name: str
    dim: int

    def __init__(self, name: str, dim: int) -> None:
        if dim is None:
            dim = -1
        self.name = name
        self.dim = dim

    def compute_weight(self, module: Module) -> Any:
        g = getattr(module, self.name + "_g")
        v = getattr(module, self.name + "_v")
        return v * (g / (norm_except_dim(v, dim=self.dim) + 1e-8))

    @staticmethod
    def apply(module, name: str = "weight", dim: int = 0) -> "WeightNorm":
        for hook in module._forward_pre_hooks.values():
            if isinstance(hook, WeightNorm) and hook.name == name:
                raise RuntimeError(f"Double weight_norm hook on the same parameter {name}")

        if dim is None:
            dim = -1

        fn = WeightNorm(name, dim)

        weight = getattr(module, name)
        assert not isinstance(weight, UninitializedParameter)
        # remove w from parameter list
        del module._parameters[name]

        # add g and v as new parameters and express w as g/||v|| * v
        module.register_parameter(name + "_g", Parameter(norm_except_dim(weight, 2, dim).data))
        module.register_parameter(name + "_v", Parameter(weight.data))
        setattr(module, name, fn.compute_weight(module))

        # recompute weight before every forward()
        module.register_forward_pre_hook(fn)

        return fn

    def remove(self, module: Module) -> None:
        weight = self.compute_weight(module)
        delattr(module, self.name)
        del module._parameters[self.name + "_g"]
        del module._parameters[self.name + "_v"]
        setattr(module, self.name, Parameter(weight.data))

    def __call__(self, module: Module, inputs: Any) -> None:
        setattr(module, self.name, self.compute_weight(module))
    

def weight_norm(module: Module, name: str = "weight", dim: int = 0) -> Module:
    WeightNorm.apply(module, name, dim)
    return module
    
    
class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm
    

def log_gradients(model, model_name, logger):
    for name, param in model.named_parameters():
        if param.grad is not None:
            logger.info(f"{model_name} | {name} | Grad Min: {param.grad.min().item():.6f}, "
                        f"Grad Max: {param.grad.max().item():.6f}, Grad Mean: {param.grad.mean().item():.6f}")


def _backup_bn(model):
    bn_backups = {}
    for n, b in model.named_buffers():
        if any(k in n for k in ["running_mean", "running_var", "num_batches_tracked"]):
            bn_backups[n] = b.clone()
    return bn_backups

def _restore_bn(model, bn_backups):
    for n, b in model.named_buffers():
        if n in bn_backups:
            b.data.copy_(bn_backups[n])

def _grads_are_finite(model):
    for p in model.parameters():
        if p.grad is None: 
            continue
        if not torch.isfinite(p.grad).all():
            return False
    return True