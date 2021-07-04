import torch

from . import common_utils as c_u
from .module_with_records import ModuleWithRecords


class BaseReducer(ModuleWithRecords):
    def forward(self, loss_dict, embeddings, labels):
        self.reset_stats()
        assert len(loss_dict) == 1
        loss_name = list(loss_dict.keys())[0]
        loss_info = loss_dict[loss_name]
        self.add_to_recordable_attributes(name=loss_name, is_stat=True)
        losses, loss_indices, reduction_type, kwargs = self.unpack_loss_info(loss_info)
        loss_val = self.reduce_the_loss(
            losses, loss_indices, reduction_type, kwargs, embeddings, labels
        )
        setattr(self, loss_name, loss_val.item())
        return loss_val

    def unpack_loss_info(self, loss_info):
        return (
            loss_info["losses"],
            loss_info["indices"],
            loss_info["reduction_type"],
            {},
        )

    def reduce_the_loss(
        self, losses, loss_indices, reduction_type, kwargs, embeddings, labels
    ):
        self.set_losses_size_stat(losses)
        if self.input_is_zero_loss(losses):
            return self.zero_loss(embeddings)
        self.assert_sizes(losses, loss_indices, reduction_type)
        reduction_func = self.get_reduction_func(reduction_type)
        return reduction_func(losses, loss_indices, embeddings, labels, **kwargs)

    def already_reduced_reduction(self, losses, loss_indices, embeddings, labels):
        assert losses.ndim == 0 or len(losses) == 1
        return losses

    def element_reduction(self, losses, loss_indices, embeddings, labels):
        raise NotImplementedError

    def pos_pair_reduction(self, losses, loss_indices, embeddings, labels):
        raise NotImplementedError

    def neg_pair_reduction(self, losses, loss_indices, embeddings, labels):
        raise NotImplementedError

    def triplet_reduction(self, losses, loss_indices, embeddings, labels):
        raise NotImplementedError

    def get_reduction_func(self, reduction_type):
        return getattr(self, "{}_reduction".format(reduction_type))

    def assert_sizes(self, losses, loss_indices, reduction_type):
        getattr(self, "assert_sizes_{}".format(reduction_type))(losses, loss_indices)

    def zero_loss(self, embeddings):
        return torch.sum(embeddings * 0)

    def input_is_zero_loss(self, losses):
        if (not torch.is_tensor(losses)) and (losses == 0):
            return True
        return False

    def assert_sizes_already_reduced(self, losses, loss_indices):
        pass

    def assert_sizes_element(self, losses, loss_indices):
        assert torch.is_tensor(losses)
        assert torch.is_tensor(loss_indices)
        assert len(losses) == len(loss_indices)

    def assert_sizes_pair(self, losses, loss_indices):
        assert torch.is_tensor(losses)
        assert c_u.is_list_or_tuple(loss_indices)
        assert len(loss_indices) == 2
        assert all(torch.is_tensor(x) for x in loss_indices)
        assert len(losses) == len(loss_indices[0]) == len(loss_indices[1])

    def assert_sizes_pos_pair(self, losses, loss_indices):
        self.assert_sizes_pair(losses, loss_indices)

    def assert_sizes_neg_pair(self, losses, loss_indices):
        self.assert_sizes_pair(losses, loss_indices)

    def assert_sizes_triplet(self, losses, loss_indices):
        assert torch.is_tensor(losses)
        assert c_u.is_list_or_tuple(loss_indices)
        assert len(loss_indices) == 3
        assert all(len(x) == len(losses) for x in loss_indices)

    def set_losses_size_stat(self, losses):
        if self.collect_stats:
            self.add_to_recordable_attributes(name="losses_size", is_stat=True)
            if not torch.is_tensor(losses) or losses.ndim == 0:
                self.losses_size = 1
            else:
                self.losses_size = len(losses)


class DoNothingReducer(BaseReducer):
    def forward(self, loss_dict, embeddings, labels):
        return loss_dict


class MeanReducer(BaseReducer):
    def element_reduction(self, losses, *_):
        return torch.mean(losses)

    def pos_pair_reduction(self, losses, *args):
        return self.element_reduction(losses, *args)

    def neg_pair_reduction(self, losses, *args):
        return self.element_reduction(losses, *args)

    def triplet_reduction(self, losses, *args):
        return self.element_reduction(losses, *args)


class MultipleReducers(BaseReducer):
    def __init__(self, reducers, default_reducer=None, **kwargs):
        super().__init__(**kwargs)
        self.reducers = torch.nn.ModuleDict(reducers)
        self.default_reducer = (
            MeanReducer() if default_reducer is None else default_reducer
        )

    def forward(self, loss_dict, embeddings, labels):
        self.reset_stats()
        sub_losses = torch.zeros(
            len(loss_dict), dtype=embeddings.dtype, device=embeddings.device
        )
        loss_count = 0
        for loss_name, loss_info in loss_dict.items():
            input_dict = {loss_name: loss_info}
            if loss_name in self.reducers:
                loss_val = self.reducers[loss_name](input_dict, embeddings, labels)
            else:
                loss_val = self.default_reducer(input_dict, embeddings, labels)
            sub_losses[loss_count] = loss_val
            loss_count += 1
        return self.sub_loss_reduction(sub_losses, embeddings, labels)

    def sub_loss_reduction(self, sub_losses, embeddings=None, labels=None):
        return torch.sum(sub_losses)


class ThresholdReducer(BaseReducer):
    def __init__(self, low=None, high=None, **kwargs):
        super().__init__(**kwargs)
        assert (low is not None) or (
            high is not None
        ), "At least one of low or high must be specified"
        self.low = low
        self.high = high
        if self.low is not None:
            self.add_to_recordable_attributes(list_of_names=["low"], is_stat=False)
        if self.high is not None:
            self.add_to_recordable_attributes(list_of_names=["high"], is_stat=False)

    def element_reduction(self, losses, loss_indices, embeddings, labels):
        return self.element_reduction_helper(losses, embeddings, "elements")

    def pos_pair_reduction(self, losses, loss_indices, embeddings, labels):
        return self.element_reduction_helper(losses, embeddings, "pos_pairs")

    def neg_pair_reduction(self, losses, loss_indices, embeddings, labels):
        return self.element_reduction_helper(losses, embeddings, "neg_pairs")

    def triplet_reduction(self, losses, loss_indices, embeddings, labels):
        return self.element_reduction_helper(losses, embeddings, "triplets")

    def element_reduction_helper(self, losses, embeddings, attr_name):
        low_condition = losses > self.low if self.low is not None else True
        high_condition = losses < self.high if self.high is not None else True
        threshold_condition = low_condition & high_condition
        num_past_filter = torch.sum(threshold_condition)
        if num_past_filter >= 1:
            loss = torch.mean(losses[threshold_condition])
        else:
            loss = self.zero_loss(embeddings)
        self.set_stats(low_condition, high_condition, num_past_filter, attr_name)
        return loss

    def set_stats(self, low_condition, high_condition, num_past_filter, attr_name):
        if self.collect_stats:
            curr_attr_name = "{}_past_filter".format(attr_name)
            self.add_to_recordable_attributes(name=curr_attr_name, is_stat=True)
            setattr(self, curr_attr_name, num_past_filter.item())
            with torch.no_grad():
                if self.low is not None:
                    curr_attr_name = "{}_above_low".format(attr_name)
                    self.add_to_recordable_attributes(name=curr_attr_name, is_stat=True)
                    setattr(self, curr_attr_name, torch.sum(low_condition).item())
                if self.high is not None:
                    curr_attr_name = "{}_below_high".format(attr_name)
                    self.add_to_recordable_attributes(name=curr_attr_name, is_stat=True)
                    setattr(self, curr_attr_name, torch.sum(high_condition).item())


class AvgNonZeroReducer(ThresholdReducer):
    def __init__(self, **kwargs):
        super().__init__(low=0, **kwargs)
