from torch import nn


class BaseClsHead(nn.Module):
    """Classification head.
    Args:
        out_channels (int): Number of classes.
        in_channels (int): Number of channels in the input feature map.
        loss (dict): Config of classification loss.
    """

    def __init__(self):
        super(BaseClsHead, self).__init__()

    def simple_test(self, feats):
        """Test without augmentation."""
        x = self(feats)
        x = x.detach().cpu().numpy()
        return x

    def forward_train(self, feats, labels):
        x = self(feats)
        losses = self.loss(x, labels)
        return dict(loss_emb=losses)
