import torch
from torch.nn.utils import clip_grad

from ..core_hook import HOOKS, Hook


@HOOKS.register_module()
class OptimizerHook(Hook):
    def __init__(self, grad_clip=None, fp16=False):
        self.grad_clip = grad_clip
        self.fp16 = fp16
        self.scaler = torch.cuda.amp.GradScaler()

    def clip_grads(self, params):
        params = list(filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return clip_grad.clip_grad_norm_(params, **self.grad_clip)

    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()
        if self.fp16:
            self.scaler.scale(runner.outputs["loss"]).backward()
        else:
            runner.outputs["loss"].backward()
        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                # Add grad norm to the logger
                runner.log_buffer.update(
                    {"grad_norm": float(grad_norm)}, runner.outputs["num_samples"]
                )

        if self.fp16:
            self.scaler.step(runner.optimizer)
            self.scaler.update()
        else:
            runner.optimizer.step()
