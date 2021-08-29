from mvt.utils.reg_util import Registry, build_module_from_dict

RUNNERS = Registry("runner")


def build_runner(cfg_dict, default_args=None):
    return build_module_from_dict(cfg_dict, RUNNERS, default_args=default_args)
