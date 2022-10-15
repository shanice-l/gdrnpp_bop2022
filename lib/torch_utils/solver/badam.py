# https://github.com/bonlime/BAdam/blob/master/badam/badam.py
import torch
from torch.optim import Optimizer
from collections import defaultdict


class BAdam(Optimizer):
    r"""BAdam - Better Adam is an optimizer based on Adam with couple important modifications
    1. decoupled weight decay (as in AdamW [2])
    2. epsilon is inside sqrt to avoid NaN in mixed precision
        default value is much larger than in Adam to reduce 'adaptivity' it leads to better and wider optimums [3]
        large epsilon works better than `amsgrad` version of Adam
    3. `exp_avg_sq` inits with large value, rather than with zeros. this removes the need for lr warmup and does the same
        thing as all the tricks from RAdam [4].
    4. Removed bias correction. It's not needed if exp_avg_sq is correctly initialized
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
        avg_sq_init (float, optional): value to use for average square initialization
            smaller values lead to faster convergence at the begining but too small vaules degrade performance
            default should be good enough, no need to tune
    Ref:
        [1] Adam: A Method for Stochastic Optimization
        [2] Decoupled Weight Decay Regularization
        [3] On the Convergence of Adam and Beyond
        [4] On the Variance of the Adaptive Learning Rate and Beyond
    """

    def __init__(self, params, lr=1e-2, betas=(0.9, 0.999), eps=1e-6, weight_decay=1e-2, avg_sq_init=1e-3):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, avg_sq_init=avg_sq_init)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            grads = []
            states = []
            exp_avg = []
            exp_avg_sq = []
            params_with_grad = []

            for p in group["params"]:
                if p.grad is not None:
                    if p.grad.is_sparse:
                        raise RuntimeError("BAdam does not support sparse gradients")

                    # Perform stepweight decay
                    p.mul_(1 - group["lr"] * group["weight_decay"])

                    params_with_grad.append(p)
                    grads.append(p.grad)

            for p in params_with_grad:
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values. Torch init to zeros here
                    state["exp_avg_sq"] = torch.full_like(p, group["avg_sq_init"], memory_format=torch.preserve_format)

                exp_avg.append(state["exp_avg"])
                exp_avg_sq.append(state["exp_avg_sq"])

                state["step"] += 1
                states.append(state)

            beta1, beta2 = group["betas"]

            #
            # Decay the first and second moment running average coefficient
            #
            torch._foreach_mul_(exp_avg, beta1)
            torch._foreach_add_(exp_avg, grads, alpha=1 - beta1)

            torch._foreach_mul_(exp_avg_sq, beta2)
            torch._foreach_addcmul_(exp_avg_sq, grads, grads, 1 - beta2)

            exp_avg_sq_sqrt = torch._foreach_sqrt(torch._foreach_add(exp_avg_sq, group["eps"]))

            torch._foreach_addcdiv_(params_with_grad, exp_avg, exp_avg_sq_sqrt, -group["lr"])

        return loss

    # TODO: refactor to a base class once foreach ops are in a good shape.
    def zero_grad(self, set_to_none: bool = False):
        per_device_and_dtype_grads = defaultdict(lambda: defaultdict(list))
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        if p.grad.grad_fn is not None:
                            p.grad.detach_()
                        else:
                            p.grad.requires_grad_(False)

                        if p.grad.is_sparse:
                            p.grad.zero_()
                        else:
                            per_device_and_dtype_grads[p.grad.device][p.grad.dtype].append(p.grad)

            for _, per_dtype_grads in per_device_and_dtype_grads.items():
                for grads in per_dtype_grads.values():
                    torch._foreach_zero_(grads)
