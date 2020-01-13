import torch
import torch.nn as nn
import torchvision

from torchvision.models.resnet import model_urls

import math
import pdb



class CTNetwork2d(nn.Module):
    def __init__(self, lossfn, inchannels, num_classes, pretrained, cam=False):
        super(CTNetwork2d, self).__init__()

        self.cam = cam
        # Backbone based on resnet18
        
        try:
            self.base_network = torchvision.models.resnet18(pretrained=pretrained)
        except:
            model_urls['resnet18'] = model_urls['resnet18'].replace('https://', 'http://')
            self.base_network = torchvision.models.resnet18(pretrained=pretrained)

        # Change the number of input channels
        self.base_network.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Make the output of layer 4 - 25 X 25 (This might be useful for unsupervised branch / Attention )
        self.base_network.maxpool = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
        self.base_network.fc = nn.Linear(512, num_classes)

        if lossfn == 'focal':
            prior = 0.01
            bias = -math.log((1 - prior) / prior)
            nn.init.constant_(self.base_network.fc.bias, bias)

    def reset_parameters(self):
        # init all params
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.consintat_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        """
        :param x: 4D tensor - N X 128 X 200 X 200
        :return: 3D tensor - N X 128 x 6
        """
        output_map = None
        if self.cam:
            for i in range(x.shape[1]):
                if i==0:
                    # output - N X 6, output_map - N X 25 X 25
                    output, output_map = self.cam_helper(x[:, i:(i+1), :, :])
                    output = torch.unsqueeze(output, 1)
                    output_map = torch.unsqueeze(output_map, 1)
                else:
                    output, output_map = self.cam_helper(x[:, i:(i+1), :, :])
                    output = torch.cat((output, torch.unsqueeze(output, 1)), 1)
                    output_map = torch.cat((output_map, torch.unsqueeze(output_map, 1)), 1)
        else:
            for i in range(x.shape[1]):
                if i==0:
                    output = self.base_network(x[:, i:(i+1), :, :])
                    output = torch.unsqueeze(output, 1)
                else:
                    output = torch.cat((output, torch.unsqueeze(self.base_network(x[:, i:(i+1), :, :]), 1)), 1)

            # TODO : Output map
            return output, output_map

    def cam_helper(self, x):
    """
    A helper function that returns Class Activation Maps (CAM) along with the output predictions.
    """
        x = self.base_network.conv1(x)
        x = self.base_network.bn1(x)
        x = self.base_network.relu(x)
        x = self.base_network.maxpool(x)

        x = self.base_network.layer1(x)
        x = self.base_network.layer2(x)
        x = self.base_network.layer3(x)
        x_map = self.base_network.layer4(x)

        x = self.base_network.avgpool(x_map)
        x = torch.base_network.flatten(x, 1)
        x = self.fc(x)
        return x, x_map


def main():
    input = torch.rand(4,128,200,200)
    model = CTNetwork2d(1, 6, True)
    out = model(input)
    print("output shape = ", out.shape)


if __name__ == "__main__":
    main()
