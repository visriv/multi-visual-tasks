from abc import ABCMeta, abstractmethod
from collections import OrderedDict
import torch
import torch.distributed as dist
import torch.nn as nn

from model.mvt.utils.log_util import print_log


class BaseEmbedder(nn.Module, metaclass=ABCMeta):
    """Base class for classifiers"""

    def __init__(self):
        super(BaseEmbedder, self).__init__()
    
    @property
    def with_neck(self):
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_cls_head(self):
        return hasattr(self, 'cls_head') and self.cls_head is not None

    @abstractmethod
    def extract_feat(self, datas):
        pass

    def extract_feats(self, datas):
        assert isinstance(datas, list)
        for data_item in datas:
            yield self.extract_feat(data_item)

    @abstractmethod
    def forward_train(self, datas, **kwargs):
        """
        Args:
            data_item (list[Tensor]): List of tensors.
            kwargs (keyword arguments): Specific to concrete implementation.
        """
        pass

    @abstractmethod
    def simple_test(self, img, **kwargs):
        pass

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            print_log(f'load model from: {pretrained}', logger='root')

    def forward_test(self, datas, **kwargs):
        """
        Args:
            datas (List[Tensor]): the outer list of Tensors in a batch.
        """
        if isinstance(datas, torch.Tensor):
            datas = [datas]

        if len(datas) == 1:
            return self.simple_test(datas[0], **kwargs)
        else:
            raise NotImplementedError('Test has not been implemented')

    def forward(self, datas, return_loss=True, **kwargs):
        """
        Calls either forward_train or forward_test depending on whether
        return_loss=True. Note this setting will change the expected inputs.
        When `return_loss=True`, data_item and data_meta are single-nested (i.e.
        Tensor and List[dict]), and when `resturn_loss=False`, data_item and data_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test.
        """
        if return_loss:
            return self.forward_train(datas, **kwargs)
        else:
            return self.forward_test(datas, **kwargs)

    def _parse_losses(self, losses):
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            elif isinstance(loss_value, dict):
                for name, value in loss_value.items():
                    log_vars[name] = value
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def train_step(self, datas, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating are also defined in
        this method, such as GAN.

        Args:
            datas (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        # with torch.cuda.amp.autocast():
        losses = self(**datas)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(datas['img'].data))

        return outputs

    def val_step(self, datas, optimizer):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        with torch.cuda.amp.autocast():
            losses = self(**datas)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(datas['img'].data))

        return outputs
