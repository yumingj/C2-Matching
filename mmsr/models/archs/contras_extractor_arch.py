from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision.models.vgg as vgg


class ContrasExtractorLayer(nn.Module):

    def __init__(self):
        super(ContrasExtractorLayer, self).__init__()

        vgg16_layers = [
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1',
            'relu2_1', 'conv2_2', 'relu2_2', 'pool2', 'conv3_1', 'relu3_1',
            'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3', 'conv4_1',
            'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'pool4',
            'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3',
            'pool5'
        ]
        conv3_1_idx = vgg16_layers.index('conv3_1')
        features = getattr(vgg,
                           'vgg16')(pretrained=True).features[:conv3_1_idx + 1]
        modified_net = OrderedDict()
        for k, v in zip(vgg16_layers, features):
            modified_net[k] = v

        self.model = nn.Sequential(modified_net)
        # the mean is for image with range [0, 1]
        self.register_buffer(
            'mean',
            torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        # the std is for image with range [0, 1]
        self.register_buffer(
            'std',
            torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, batch):
        batch = (batch - self.mean) / self.std
        output = self.model(batch)
        return output


class ContrasExtractorSep(nn.Module):

    def __init__(self):
        super(ContrasExtractorSep, self).__init__()

        self.feature_extraction_image1 = ContrasExtractorLayer()
        self.feature_extraction_image2 = ContrasExtractorLayer()

    def forward(self, image1, image2):
        dense_features1 = self.feature_extraction_image1(image1)
        dense_features2 = self.feature_extraction_image2(image2)

        return {
            'dense_features1': dense_features1,
            'dense_features2': dense_features2
        }
