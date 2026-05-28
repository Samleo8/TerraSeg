import torch
import torch.nn as nn


def replace_bn1d_with_gn(
    model: nn.Module,
    groups_default: int = 32,
) -> nn.Module:
    """
    Recursively replace every :class:`torch.nn.BatchNorm1d` layer with :class:`torch.nn.GroupNorm`.

    Following TerraSeg paper section 3.4: 1D BatchNorm is replaced with GroupNorm to prevent instability
    when training on diverse multi-LiDAR batches. The number of groups is chosen as the largest
    divisor of the channel count that does not exceed ``groups_default`` (with a special case for
    32-channel layers, which use 16 groups).

    Args:
        model (torch.nn.Module) : Input model. Modified in place; returned for chaining.
        groups_default (int) : Default upper bound on the number of GroupNorm groups.

    Returns:
        model (torch.nn.Module) : The same model with BatchNorm1d layers replaced.
    """
    for name, child in list(model.named_children()):
        if isinstance(child, nn.BatchNorm1d):
            channels = child.num_features
            num_groups = 16 if channels == 32 else min(groups_default, channels)
            while channels % num_groups != 0:
                num_groups -= 1

            gn = nn.GroupNorm(
                num_groups=num_groups, num_channels=channels, eps=child.eps, affine=True
            )
            with torch.no_grad():
                gn.weight.copy_(child.weight)
                gn.bias.copy_(child.bias)
            setattr(model, name, gn)
        else:
            replace_bn1d_with_gn(model=child, groups_default=groups_default)
    return model
