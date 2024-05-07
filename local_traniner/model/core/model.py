import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ResNet50_Weights
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


class DRE_net(nn.Module):
    def __init__(self, n_class):
        super(DRE_net, self).__init__()
        self.n_class = n_class

        self.backbone = resnet_fpn_backbone(backbone_name='resnet50', weights=ResNet50_Weights.DEFAULT)
        in_channels_list = [256, 256, 256, 256, 256]

        self.prediction_layers = nn.ModuleDict({
            str(i): nn.Sequential(
                nn.Conv2d(in_channels=in_channels_list[i], out_channels=n_class, kernel_size=1),
                nn.BatchNorm2d(n_class),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1)
            ) for i in range(len(in_channels_list))
        })

        self.attention = nn.Linear(len(in_channels_list) * n_class, len(in_channels_list))

    def forward(self, x):
        feature_maps = self.backbone(x)

        preds = [self.prediction_layers[str(i)](feature_map) for i, feature_map in enumerate(feature_maps.values())]
        # Squeeze each tensor in preds
        preds = [p.view(p.size(0), -1) for p in preds]

        concatenated = torch.cat(preds, dim=1)

        # Predict the attention scores
        attn_scores = self.attention(concatenated)
        attn_weights = F.softmax(attn_scores, dim=1)

        # adjust shapes for both attn_weights and preds
        attn_weights = attn_weights.unsqueeze(-1)
        preds = torch.stack(preds, dim=1)

        # Sum the weighted preds
        output = (attn_weights * preds).sum(dim=1)

        return output
