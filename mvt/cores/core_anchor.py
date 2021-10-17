from mvt.utils.reg_util import Registry, build_module_from_dict

ANCHOR_GENERATORS = Registry("Anchor generator")


def build_anchor_generator(cfg_dict, default_args=None):
    return build_module_from_dict(cfg_dict, ANCHOR_GENERATORS, default_args)
