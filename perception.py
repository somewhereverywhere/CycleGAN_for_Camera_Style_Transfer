# perceptual_loss.py
import torch
import torch.nn as nn
import torchvision.models as models

class PerceptualLoss(nn.Module):
    def __init__(self, layers=['relu3_3'], use_gpu=True):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features
        self.layers = {'relu1_1': 1, 'relu2_1': 6, 'relu3_3': 16, 'relu4_3': 25}
        selected_layers = [self.layers[layer] for layer in layers]

        self.vgg_layers = nn.Sequential(*[vgg[i] for i in range(max(selected_layers) + 1)])
        for param in self.vgg_layers.parameters():
            param.requires_grad = False

        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.vgg_layers = self.vgg_layers.to(self.device)

    def forward(self, x, y):
        # Assumes input x, y are normalized to [0, 1]
        x_vgg = self.vgg_layers(self._normalize(x))
        y_vgg = self.vgg_layers(self._normalize(y))
        return torch.mean((x_vgg - y_vgg) ** 2)

    def _normalize(self, batch):
        # Normalize to ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        batch = (batch + 1) / 2  # Convert from [-1, 1] to [0, 1]
        return (batch - mean) / std
