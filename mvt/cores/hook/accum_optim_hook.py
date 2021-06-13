import math

from ..core_hook import HOOKS
from .optimizer import Fp16OptimizerHook
from mvt.utils.parallel_util import get_dist_info


@HOOKS.register_module()
class Fp16GradAccumulateOptimizerHook(Fp16OptimizerHook):
    def __init__(self, *wargs, **kwargs):
        nominal_batch_size = kwargs.pop('nominal_batch_size', None)
        accumulation = kwargs.pop('accumulation', None)
        self.accumulation = 1
        self.nominal_batch_size = None
        if accumulation is not None:
            assert isinstance(accumulation, int) and accumulation > 0
            self.accumulation = accumulation
        elif nominal_batch_size is not None:
            self.accumulation = None
            self.nominal_batch_size = nominal_batch_size

        super(Fp16GradAccumulateOptimizerHook, self).__init__(*wargs, **kwargs)

    def before_train_epoch(self, runner):
        super(Fp16GradAccumulateOptimizerHook, self).before_train_epoch(runner)
        if self.accumulation is None:
            assert self.nominal_batch_size is not None
            samples_per_gpu = runner.data_loader.sampler.samples_per_gpu
            _, word_size = get_dist_info()
            self.accumulation = math.ceil(self.nominal_batch_size /
					(samples_per_gpu * word_size))

    def after_train_iter(self, runner):
        # clear grads of last iteration
        if runner.iter % self.accumulation == 0:
            runner.model.zero_grad()
            runner.optimizer.zero_grad()

        self.loss_scaler.scale(runner.outputs['loss']).backward()

        if (runner.iter + 1) % self.accumulation == 0:
            self.loss_scaler.unscale_(runner.optimizer)
            # grad clip
            if self.grad_clip is not None:
                grad_norm = self.clip_grads(runner.model.parameters())
                if grad_norm is not None:
                    # Add grad norm to the logger
                    runner.log_buffer.update({
			'grad_norm': float(grad_norm),
			'grad_scale': float(self.loss_scaler.get_scale())
			}, runner.outputs['num_samples'])
            # step and update scaler
            self.loss_scaler.step(runner.optimizer)
            self.loss_scaler.update(self._scale_update_param)
