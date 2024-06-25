import torch
import torch.nn as nn
import torch.nn.functional as F

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    # transpose
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)
    return x
  class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        branch_features = oup // 2
        if self.stride == 1:
            assert inp == branch_features * 2

        if self.stride == 2:
            self.branch1 = nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride=2, padding=1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if self.stride > 1 else branch_features, branch_features, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_features, branch_features, 3, stride=self.stride, padding=1, groups=branch_features, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        out = channel_shuffle(out, 2)
        return out

    class ShuffleNetV2(nn.Module):
    def __init__(self, stages_repeats, stages_out_channels, num_classes=1000):
        super(ShuffleNetV2, self).__init__()
        self.num_classes = num_classes

        input_channels = 3
        output_channels = stages_out_channels[0]

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.stages = []
        for repeats, output_channels in zip(stages_repeats, stages_out_channels[1:]):
            seq = [InvertedResidual(input_channels, output_channels, 2)]
            for _ in range(repeats - 1):
                seq.append(InvertedResidual(output_channels, output_channels, 1))
            self.stages.append(nn.Sequential(*seq))
            input_channels = output_channels

        self.stages = nn.Sequential(*self.stages)

        output_channels = stages_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Linear(output_channels, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stages(x)
        x = self.conv5(x)
        x = x.mean([2, 3])
        x = self.fc(x)
        return x

    def shufflenet_v2_x1_0():
    stages_repeats = [4, 8, 4]
    stages_out_channels = [24, 116, 232, 464, 1024]
    return ShuffleNetV2(stages_repeats, stages_out_channels)

model = shufflenet_v2_x1_0()
print(model)
