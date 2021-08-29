import torch
import torch.nn as nn
import warnings
import torch.nn.functional as F

from .ops_builder import (
    PLUGIN_LAYERS,
    CONV_LAYERS,
    build_activation_layer,
    build_conv_layer,
    build_norm_layer,
    build_padding_layer,
)
from model.mvt.utils.init_util import constant_init, kaiming_init


def _fuse_conv_bn(conv, bn):
    """Fuse conv and bn into one module.
    Args:
        conv (nn.Module): Conv to be fused.
        bn (nn.Module): BN to be fused.
    Returns:
        nn.Module: Fused module.
    """
    conv_w = conv.weight
    conv_b = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)

    factor = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    conv.weight = nn.Parameter(conv_w * factor.reshape([conv.out_channels, 1, 1, 1]))
    conv.bias = nn.Parameter((conv_b - bn.running_mean) * factor + bn.bias)
    return conv


def fuse_conv_bn(module):
    """Recursively fuse conv and bn in a module.
    During inference, the functionary of batch norm layers is turned off
    but only the mean and var alone channels are used, which exposes the
    chance to fuse it with the preceding conv layers to save computations and
    simplify network structures.
    Args:
        module (nn.Module): Module to be fused.
    Returns:
        nn.Module: Fused module.
    """
    last_conv = None
    last_conv_name = None

    for name, child in module.named_children():
        if isinstance(child, (nn.modules.batchnorm._BatchNorm, nn.SyncBatchNorm)):
            if last_conv is None:  # only fuse BN that is after Conv
                continue
            fused_conv = _fuse_conv_bn(last_conv, child)
            module._modules[last_conv_name] = fused_conv
            # To reduce changes, set BN as Identity instead of deleting it.
            module._modules[name] = nn.Identity()
            last_conv = None
        elif isinstance(child, nn.Conv2d):
            last_conv = child
            last_conv_name = name
        else:
            fuse_conv_bn(child)
    return module


def conv_ws_2d(
    input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, eps=1e-5
):
    c_in = weight.size(0)
    weight_flat = weight.view(c_in, -1)
    mean = weight_flat.mean(dim=1, keepdim=True).view(c_in, 1, 1, 1)
    std = weight_flat.std(dim=1, keepdim=True).view(c_in, 1, 1, 1)
    weight = (weight - mean) / (std + eps)
    return F.conv2d(input, weight, bias, stride, padding, dilation, groups)


@PLUGIN_LAYERS.register_module()
class ConvModule(nn.Module):
    """A conv block that bundles conv/norm/activation layers.
    This block simplifies the usage of convolution layers, which are commonly
    used with a norm layer (e.g., BatchNorm) and activation layer (e.g., ReLU).
    It is based upon three build methods: `build_conv_layer()`,
    `build_norm_layer()` and `build_activation_layer()`.
    Besides, we add some additional features in this module.
    1. Automatically set `bias` of the conv layer.
    2. Spectral norm is supported.
    3. More padding modes are supported. Before PyTorch 1.5, nn.Conv2d only
    supports zero and circular padding, and we add "reflect" padding mode.
    Args:
        in_channels (int): Number of channels in the input feature map.
            Same as that in ``nn._ConvNd``.
        out_channels (int): Number of channels produced by the convolution.
            Same as that in ``nn._ConvNd``.
        kernel_size (int | tuple[int]): Size of the convolving kernel.
            Same as that in ``nn._ConvNd``.
        stride (int | tuple[int]): Stride of the convolution.
            Same as that in ``nn._ConvNd``.
        padding (int | tuple[int]): Zero-padding added to both sides of
            the input. Same as that in ``nn._ConvNd``.
        dilation (int | tuple[int]): Spacing between kernel elements.
            Same as that in ``nn._ConvNd``.
        groups (int): Number of blocked connections from input channels to
            output channels. Same as that in ``nn._ConvNd``.
        bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        inplace (bool): Whether to use inplace mode for activation.
            Default: True.
        with_spectral_norm (bool): Whether use spectral norm in conv module.
            Default: False.
        padding_mode (str): If the `padding_mode` has not been supported by
            current `Conv2d` in PyTorch, we will use our own padding layer
            instead. Currently, we support ['zeros', 'circular'] with official
            implementation and ['reflect'] with our own implementation.
            Default: 'zeros'.
        order (tuple[str]): The order of conv/norm/activation layers. It is a
            sequence of "conv", "norm" and "act". Common examples are
            ("conv", "norm", "act") and ("act", "conv", "norm").
            Default: ('conv', 'norm', 'act').
    """

    _abbr_ = "conv_block"

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias="auto",
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=dict(type="ReLU"),
        inplace=True,
        with_spectral_norm=False,
        padding_mode="zeros",
        order=("conv", "norm", "act"),
    ):
        super(ConvModule, self).__init__()
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        assert act_cfg is None or isinstance(act_cfg, dict)
        official_padding_mode = ["zeros", "circular"]
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.inplace = inplace
        self.with_spectral_norm = with_spectral_norm
        self.with_explicit_padding = padding_mode not in official_padding_mode
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 3
        assert set(order) == set(["conv", "norm", "act"])

        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == "auto":
            bias = not self.with_norm
        self.with_bias = bias

        if self.with_norm and self.with_bias:
            warnings.warn("ConvModule has norm and bias at the same time")

        if self.with_explicit_padding:
            pad_cfg = dict(type=padding_mode)
            self.padding_layer = build_padding_layer(pad_cfg, padding)

        # reset padding to 0 for conv module
        conv_padding = 0 if self.with_explicit_padding else padding
        # build convolution layer
        self.conv = build_conv_layer(
            conv_cfg,
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=conv_padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        # export the attributes of self.conv to a higher level for convenience
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = padding
        self.dilation = self.conv.dilation
        self.transposed = self.conv.transposed
        self.output_padding = self.conv.output_padding
        self.groups = self.conv.groups

        if self.with_spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)

        # build normalization layers
        if self.with_norm:
            # norm layer is after conv layer
            if order.index("norm") > order.index("conv"):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            self.norm_name, norm = build_norm_layer(norm_cfg, norm_channels)
            self.add_module(self.norm_name, norm)

        # build activation layer
        if self.with_activation:
            act_cfg_ = act_cfg.copy()
            # nn.Tanh has no 'inplace' argument
            if act_cfg_["type"] not in [
                "Tanh",
                "PReLU",
                "Sigmoid",
                "HSigmoid",
                "Swish",
                "Mish",
            ]:
                act_cfg_.setdefault("inplace", inplace)
            self.activate = build_activation_layer(act_cfg_)

        # Use msra init by default
        self.init_weights()

    @property
    def norm(self):
        return getattr(self, self.norm_name)

    def init_weights(self):
        # 1. It is mainly for customized conv layers with their own
        #    initialization manners, and we do not want ConvModule to
        #    overrides the initialization.
        # 2. For customized conv layers without their own initialization
        #    manners, they will be initialized by this method with default
        #    `kaiming_init`.
        # 3. For PyTorch's conv layers, they will be initialized anyway by
        #    their own `reset_parameters` methods.
        if not hasattr(self.conv, "init_weights"):
            if self.with_activation and self.act_cfg["type"] == "LeakyReLU":
                nonlinearity = "leaky_relu"
                a = self.act_cfg.get("negative_slope", 0.01)
            else:
                nonlinearity = "relu"
                a = 0
            kaiming_init(self.conv, a=a, nonlinearity=nonlinearity)
        if self.with_norm:
            constant_init(self.norm, 1, bias=0)

    def forward(self, x, activate=True, norm=True):
        for layer in self.order:
            if layer == "conv":
                if self.with_explicit_padding:
                    x = self.padding_layer(x)
                x = self.conv(x)
            elif layer == "norm" and norm and self.with_norm:
                x = self.norm(x)
            elif layer == "act" and activate and self.with_activation:
                x = self.activate(x)
        return x


@CONV_LAYERS.register_module("ConvWS")
class ConvWS2d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        eps=1e-5,
    ):
        super(ConvWS2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.eps = eps

    def forward(self, x):
        return conv_ws_2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.eps,
        )


@CONV_LAYERS.register_module(name="ConvAWS")
class ConvAWS2d(nn.Conv2d):
    """AWS (Adaptive Weight Standardization)
    This is a variant of Weight Standardization
    (https://arxiv.org/pdf/1903.10520.pdf)
    It is used in DetectoRS to avoid NaN
    (https://arxiv.org/pdf/2006.02334.pdf)
    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the conv kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of
            the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements.
            Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        bias (bool, optional): If set True, adds a learnable bias to the
            output. Default: True
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.register_buffer("weight_gamma", torch.ones(self.out_channels, 1, 1, 1))
        self.register_buffer("weight_beta", torch.zeros(self.out_channels, 1, 1, 1))

    def _get_weight(self, weight):
        weight_flat = weight.view(weight.size(0), -1)
        mean = weight_flat.mean(dim=1).view(-1, 1, 1, 1)
        std = torch.sqrt(weight_flat.var(dim=1) + 1e-5).view(-1, 1, 1, 1)
        weight = (weight - mean) / std
        weight = self.weight_gamma * weight + self.weight_beta
        return weight

    def forward(self, x):
        weight = self._get_weight(self.weight)
        return F.conv2d(
            x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Override default load function.
        AWS overrides the function _load_from_state_dict to recover
        weight_gamma and weight_beta if they are missing. If weight_gamma and
        weight_beta are found in the checkpoint, this function will return
        after super()._load_from_state_dict. Otherwise, it will compute the
        mean and std of the pretrained weights and store them in weight_beta
        and weight_gamma.
        """

        self.weight_gamma.data.fill_(-1)
        local_missing_keys = []
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            local_missing_keys,
            unexpected_keys,
            error_msgs,
        )
        if self.weight_gamma.data.mean() > 0:
            for k in local_missing_keys:
                missing_keys.append(k)
            return
        weight = self.weight.data
        weight_flat = weight.view(weight.size(0), -1)
        mean = weight_flat.mean(dim=1).view(-1, 1, 1, 1)
        std = torch.sqrt(weight_flat.var(dim=1) + 1e-5).view(-1, 1, 1, 1)
        self.weight_beta.data.copy_(mean)
        self.weight_gamma.data.copy_(std)
        missing_gamma_beta = [
            k
            for k in local_missing_keys
            if k.endswith("weight_gamma") or k.endswith("weight_beta")
        ]
        for k in missing_gamma_beta:
            local_missing_keys.remove(k)
        for k in local_missing_keys:
            missing_keys.append(k)
