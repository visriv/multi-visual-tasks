from mtl.utils.reg_util import Registry, build_module_from_dict

BBOX_ASSIGNERS = Registry('bbox_assigner')
BBOX_SAMPLERS = Registry('bbox_sampler')
BBOX_CODERS = Registry('bbox_coder')
IOU_CALCULATORS = Registry('IoU calculator')


def build_assigner(cfg_dict, **default_args):
    """Builder of box assigner."""
    return build_module_from_dict(cfg_dict, BBOX_ASSIGNERS, default_args)

def build_sampler(cfg_dict, **default_args):
    """Builder of box sampler."""
    return build_module_from_dict(cfg_dict, BBOX_SAMPLERS, default_args)

def build_bbox_coder(cfg_dict, **default_args):
    """Builder of box coder."""
    return build_module_from_dict(cfg_dict, BBOX_CODERS, default_args)

def build_iou_calculator(cfg_dict, default_args=None):
    """Builder of IoU calculator."""
    return build_module_from_dict(cfg_dict, IOU_CALCULATORS, default_args)
