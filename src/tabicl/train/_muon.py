# taken from https://github.com/fabian-sp/sda/blob/e2f95648ffdaf937adb6d68340b5e2dc9cf7e8a6/sda/optim/muon.py
""" Muon optimizer

Original code copied from:
    https://github.com/toothacher17/Megatron-LM/blob/moonshot/distributedmuon-impl/megatron/core/optimizer/muon.py

Then we apply the following changes:
    * Remove everything related to distributed training (as we run on single GPU)
    * Reshape all weights to 2D, before Newton-Schulz, and then reshape back.
    (ref: https://github.com/KellerJordan/cifar10-airbench/blob/master/airbench94_muon.py)
"""
import math

import torch


# copy from https://github.com/KellerJordan/Muon/tree/master
def zeropower_via_newtonschulz5(G, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G
    if G.size(0) > G.size(1):
        X = X.T

    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X


# adjust LR based on: https://github.com/MoonshotAI/Moonlight
def adjust_lr_wd_for_muon(lr, matched_adamw_rms, param_shape):
    A, B = param_shape  # assumes 2D shape given (!)
    adjusted_ratio = math.sqrt(max(A, B)) * matched_adamw_rms
    adjusted_lr = lr * adjusted_ratio
    return adjusted_lr


# copy from https://github.com/KellerJordan/Muon/tree/master
class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.

    Routing is controlled purely by the per-group ``use_muon`` flag, which is the caller's
    responsibility: groups with ``use_muon=True`` are optimized with Muon (each parameter is
    reshaped to 2D before the Newton-Schulz step), and any other groups use an internal AdamW.
    There is no automatic detection of 1D parameters -- if a 1D parameter (bias, norm scale, ...)
    is placed in a ``use_muon=True`` group it is orthogonalized like any other. In this project all
    parameters are intentionally put in a single ``use_muon=True`` group.

    Arguments:
        param_groups: The parameters to be optimized.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        matched_adamw_rms: The AdamW Update RMS that Muon is designed to match. (0.2~0.4 recommended)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (5 is probably always enough)
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
        adamw_wd: The weight decay for the internal AdamW.
    """

    def __init__(self,
                 param_groups,
                 lr=2e-2,
                 weight_decay=0.1,
                 matched_adamw_rms=0.2,
                 momentum=0.95,
                 nesterov=True,
                 ns_steps=5,
                 adamw_betas=(0.95, 0.95),
                 adamw_eps=1e-8,
                 use_cautious_wd: bool = False):

        defaults = dict(lr=lr, weight_decay=weight_decay,
                        matched_adamw_rms=matched_adamw_rms,
                        momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                        adamw_betas=adamw_betas, adamw_eps=adamw_eps, use_cautious_wd=use_cautious_wd)

        super().__init__(param_groups, defaults)

    def step(self):
        # update muon momentum first
        for group in self.param_groups:

            if not group.get('use_muon', False):
                continue

            lr = group["lr"]
            ns_steps = group["ns_steps"]
            weight_decay = group["weight_decay"]
            use_cautious_wd = group["use_cautious_wd"]
            momentum = group["momentum"]
            matched_adamw_rms = group["matched_adamw_rms"]
            params = group["params"]

            for p in params:
                g = p.grad
                if g is None:  # parameter unused this step; skip like AdamW does
                    continue

                # prepare muon buffer in state
                state = self.state[p]
                if not "momentum_buffer" in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)

                # momentum
                ns_input = g.add(buf, alpha=momentum) if group['nesterov'] else buf
                ns_input = ns_input.reshape(len(g), -1)  # reshape to 2D

                # Reshape + calc update
                assert ns_input.ndim >= 2
                update = zeropower_via_newtonschulz5(
                    ns_input,
                    steps=ns_steps
                ).view(g.shape)

                # apply weight decay
                if use_cautious_wd:
                    wd_mask = (update * p >= 0)
                    p.data.add_(p * wd_mask, alpha=-lr * weight_decay)
                else:
                    p.data.mul_(1 - lr * weight_decay)

                # adjust lr and apply update
                adjusted_lr = adjust_lr_wd_for_muon(lr, matched_adamw_rms, ns_input.shape)
                p.data.add_(update, alpha=-adjusted_lr)

        # use adam for other params
        for group in self.param_groups:

            if group.get('use_muon', False):
                continue

            # init step
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            step = group['step']
            params = group["params"]
            lr = group['lr']
            weight_decay = group['weight_decay']
            use_cautious_wd = group['use_cautious_wd']
            beta1, beta2 = group['adamw_betas']
            eps = group['adamw_eps']

            for p in params:

                g = p.grad
                if g is None:  # parameter unused this step; skip like AdamW does
                    continue
                state = self.state[p]

                if len(state) == 0:
                    state['adamw_exp_avg'] = torch.zeros_like(g)
                    state['adamw_exp_avg_sq'] = torch.zeros_like(g)

                buf1 = state['adamw_exp_avg']
                buf2 = state['adamw_exp_avg_sq']
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                scale = bias_correction1 / bias_correction2 ** 0.5
                # apply weight decay
                if use_cautious_wd:
                    wd_mask = (g * p >= 0)
                    p.data.add_(p * wd_mask, alpha=-lr * weight_decay)
                else:
                    p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)
