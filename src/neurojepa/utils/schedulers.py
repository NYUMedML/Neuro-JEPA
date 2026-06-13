# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math


class InverseCosineScheduler(object):
    """
    Schedules the bias update rate 'u' from u_min to u_max.
    Tracks internal step state so the user doesn't have to pass it.
    """
    def __init__(self, u_min, u_max, total_steps, warmup_steps=0):
        self.u_min = u_min
        self.u_max = u_max
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.current_step = 0 # Internal state tracker

    def step(self):
        step = self.current_step
        
        # Calculate current update rate u_t
        if step < self.warmup_steps:
            u_t = self.u_min
        else:
            # Progress from 0 to 1 after warmup
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(1.0, progress)
            
            # Inverse Cosine curve
            cosine_factor = 0.5 * (1 - math.cos(math.pi * progress))
            u_t = self.u_min + (self.u_max - self.u_min) * cosine_factor
        
        # Increment step for the next call
        self.current_step += 1
        return u_t

    def set_step(self, step):
        """Utility to manually reset or jump to a step (useful for resuming training)"""
        self.current_step = step
    

class WSDSchedule(object):

    def __init__(self, optimizer, warmup_steps, anneal_steps, T_max, start_lr, ref_lr, final_lr=0.0):
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.final_lr = final_lr
        self.anneal_steps = anneal_steps
        self.warmup_steps = warmup_steps
        self.T_max = T_max - warmup_steps - anneal_steps
        self._step = 0.0

    def step(self):
        self._step += 1
        if self._step < self.warmup_steps:
            progress = float(self._step) / float(max(1, self.warmup_steps))
            new_lr = self.start_lr + progress * (self.ref_lr - self.start_lr)
        elif self._step < self.T_max + self.warmup_steps:
            new_lr = self.ref_lr
        else:
            _step = self._step - (self.T_max + self.warmup_steps)
            progress = float(_step) / float(max(1, self.anneal_steps))
            new_lr = self.ref_lr + progress * (self.final_lr - self.ref_lr)

        for group in self.optimizer.param_groups:
            group["lr"] = new_lr
            if "lr_scale" in group:
                group["lr"] *= group["lr_scale"]

        return new_lr


class WarmupCosineSchedule(object):

    def __init__(self, optimizer, warmup_steps, start_lr, ref_lr, T_max, last_epoch=-1, final_lr=0.0):
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.T_max = T_max - warmup_steps
        self._step = 0.0

    def step(self):
        self._step += 1
        if self._step < self.warmup_steps:
            progress = float(self._step) / float(max(1, self.warmup_steps))
            new_lr = self.start_lr + progress * (self.ref_lr - self.start_lr)
        else:
            # -- progress after warmup
            progress = float(self._step - self.warmup_steps) / float(max(1, self.T_max))
            new_lr = max(
                self.final_lr,
                self.final_lr + (self.ref_lr - self.final_lr) * 0.5 * (1.0 + math.cos(math.pi * progress)),
            )

        for group in self.optimizer.param_groups:
            group["lr"] = new_lr
            if "lr_scale" in group:
                group["lr"] *= group["lr_scale"]

        return new_lr


class CosineWDSchedule(object):

    def __init__(self, optimizer, ref_wd, T_max, final_wd=0.0):
        self.optimizer = optimizer
        self.ref_wd = ref_wd
        self.final_wd = final_wd
        self.T_max = T_max
        self._step = 0.0

    def step(self):
        self._step += 1
        progress = self._step / self.T_max
        new_wd = self.final_wd + (self.ref_wd - self.final_wd) * 0.5 * (1.0 + math.cos(math.pi * progress))

        if self.final_wd <= self.ref_wd:
            new_wd = max(self.final_wd, new_wd)
        else:
            new_wd = min(self.final_wd, new_wd)

        for group in self.optimizer.param_groups:
            if ("WD_exclude" not in group) or not group["WD_exclude"]:
                group["weight_decay"] = new_wd
        return new_wd


class LinearDecaySchedule(object):

    def __init__(self, optimizer, ref_lr, T_max, last_epoch=-1, final_lr=0.0):
        self.optimizer = optimizer
        self.ref_lr = ref_lr
        self.final_lr = final_lr
        self.T_max = T_max
        self._step = 0.0

    def step(self):
        self._step += 1
        progress = float(self._step) / float(max(1, self.T_max))
        new_lr = self.ref_lr + progress * (self.final_lr - self.ref_lr)
        for group in self.optimizer.param_groups:
            group["lr"] = new_lr
            if "lr_scale" in group:
                group["lr"] *= group["lr_scale"]

        return new_lr