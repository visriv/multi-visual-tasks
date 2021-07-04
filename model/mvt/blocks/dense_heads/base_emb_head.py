from torch import nn


class BaseEmbHead(nn.Module):
    """Linear regressor head.

    Args:
        num_dim (int): Number of regression dimensions.
        in_channels (int): Number of channels in the input feature map.
        loss (dict): Config of classification loss.
    """

    def __init__(self):
        super(BaseEmbHead, self).__init__()
    
    def simple_test(self, feats):
        """Test without augmentation."""
        x = self(feats)        
        x = x.detach().cpu().numpy()
        return x

    def forward_train(self, feats, labels):
        x = self(feats)
        losses = self.loss(x, labels)
        return dict(loss_emb=losses)
    