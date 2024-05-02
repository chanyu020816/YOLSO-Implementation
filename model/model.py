import torch
import torch.nn as nn
from config import model_configs


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bn_act: bool=True, *args, **kwargs) -> None:
        super().__init__(*args)
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels: int, use_residual: bool=True, repeat: bool=1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layers = nn.ModuleList()
        for _ in range(repeat):
            self.layers += [
                nn.Sequential(
                    ConvBlock(channels, channels // 2, kernel_size=1),
                    ConvBlock(channels // 2, channels, kernel_size=3),
                )
            ]
        self.use_residual = use_residual
        self.repeat = repeat

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x) if self.use_residual else layer(x)
        return x

class ScalePrediction(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pred = nn.Sequential(
            ConvBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            ConvBlock(2 * in_channels, (num_classes + 3), bn_act=False, kernel_size=1)
        )
        self.num_classes = num_classes

    def forward(self, x):
        return (
            self.pred(x)
            .reshape(x.shape[0], 1, (self.num_classes + 3), x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 5)
        )

class YOLSO(nn.Module):
    def __init__(self, model_config: list, in_channels: int, num_classes: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = model_config
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.model = self._parse_model()

    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x

    def _parse_model(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels

        for module in self.config:
            module_type = module[2]
            if module_type == 'Conv':
                number = module[1]
                out_channels, kernel_size, stride, padding = module[3]
                layers.extend([
                    ConvBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding
                    )
                    for _ in range(number)
                ])
                in_channels = out_channels
            elif module_type == 'Residual':
                pass
            elif module_type == 'MaxPool':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            elif module_type == 'ScalePrediction':
                pass
        return layers

if __name__ == '__main__':
    image = torch.randn(10, 3, 480, 480)
    model = YOLSO(model_configs['origin_config'], 3, 8)
    output = model(image)
    print(output.shape)