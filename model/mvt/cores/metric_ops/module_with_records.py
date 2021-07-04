import torch

from . import common_utils as c_u


class ModuleWithRecords(torch.nn.Module):
    def __init__(self, collect_stats=True):
        super().__init__()
        self.collect_stats = collect_stats

    def add_to_recordable_attributes(
        self, name=None, list_of_names=None, is_stat=False
    ):
        if is_stat and not self.collect_stats:
            pass
        else:
            c_u.add_to_recordable_attributes(
                self, name=name, list_of_names=list_of_names, is_stat=is_stat
            )

    def reset_stats(self):
        c_u.reset_stats(self)
