import logging

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

from .archs.vgg_arch import VGGFeatureExtractor
from .loss_utils import masked_loss

_reduction_modes = ['none', 'mean', 'sum']

logger = logging.getLogger('base')


@masked_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@masked_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


@masked_loss
def charbonnier_loss(pred, target, eps=1e-6):
    return torch.sqrt((pred - target)**2 + eps)


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        if self.reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {self.reduction}. '
                             f'Supported ones are: {_reduction_modes}')

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)


class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        if self.reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {self.reduction}. '
                             f'Supported ones are: {_reduction_modes}')

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)


class CharbonnierLoss(nn.Module):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable
    variant of L1Loss).

    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
        Super-Resolution".

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        eps (float): A value used to control the curvature near zero.
            Default: 1e-12.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-12):
        super(CharbonnierLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps
        if self.reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {self.reduction}. '
                             f'Supported ones are: {_reduction_modes}')

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * charbonnier_loss(
            pred, target, weight, eps=self.eps, reduction=self.reduction)


class MaskedTVLoss(L1Loss):

    def __init__(self, loss_weight=1.0):
        super(MaskedTVLoss, self).__init__(loss_weight=loss_weight)

    def forward(self, pred, mask=None):
        y_diff = super(MaskedTVLoss, self).forward(
            pred[:, :, :-1, :], pred[:, :, 1:, :], weight=mask[:, :, :-1, :])
        x_diff = super(MaskedTVLoss, self).forward(
            pred[:, :, :, :-1], pred[:, :, :, 1:], weight=mask[:, :, :, :-1])

        loss = x_diff + y_diff

        return loss


class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layers_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4 feature
            layer (before relu5_4) will be extracted with weight 1.0 in
            calculting losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        norm_img (bool): If True, the image will be normed to [0, 1]. Note that
            this is different from the `use_input_norm` which norm the input in
            in forward function of vgg according to the statistics of dataset.
            Importantly, the input image must be in range [-1, 1].
        pretrained (str): Path for pretrained weights. Default:
            'torchvision://vgg19'
    """

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 perceptual_weight=1.0,
                 style_weight=0,
                 norm_img=True,
                 criterion='l1'):
        super(PerceptualLoss, self).__init__()
        self.norm_img = norm_img
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm)

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.L2loss()
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(
                f'{criterion} criterion has not been supported in'
                ' this version.')

    def forward(self, x, gt):
        if self.norm_img:
            x = (x + 1.) * 0.5
            gt = (gt + 1.) * 0.5
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(
                        x_features[k] - gt_features[k],
                        p='fro') * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(
                        x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                style_loss += self.criterion(
                    self._gram_mat(x_features[k]),
                    self._gram_mat(gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss

    def _gram_mat(self, x):
        (n, c, h, w) = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram


class PerceptualLossMultiInputs(PerceptualLoss):
    """Perceptual loss with multiple inputs images.

    Args:
        x (Tensor): Input tensor with shape (B, N, C, H, W), where N indicates
            number of images.
        gt (Tensor): GT tensor with shape (B, N, C, H, W).

    Returns:
        list[Tensor]: total perceptual loss and total style loss.
    """

    def forward(self, x, gt):
        assert x.size() == gt.size(
        ), 'The sizes of input and GT should be the same.'

        total_percep_loss, total_style_loss = 0, 0
        for i in range(x.size(1)):
            percep_loss, style_loss = super(PerceptualLossMultiInputs,
                                            self).forward(
                                                x[:, i, :, :, :],
                                                gt[:, i, :, :, :])
            if percep_loss is None:
                total_percep_loss = None
            else:
                total_percep_loss += percep_loss
            if style_loss is None:
                total_style_loss = None
            else:
                total_style_loss += style_loss

        return total_percep_loss, total_style_loss


class GANLoss(nn.Module):
    """Define GAN loss.

    Args:
        gan_type (str): Support 'vanilla', 'lsgan', 'wgan', 'hinge'.
        real_label_val (float): The value for real label. Default: 1.0.
        fake_label_val (float): The value for fake label. Default: 0.0.
        loss_weight (float): Loss weight. Default: 1.0.
            Note that loss_weight is only for generators; and it is always 1.0
            for discriminators.
    """

    def __init__(self,
                 gan_type,
                 real_label_val=1.0,
                 fake_label_val=0.0,
                 loss_weight=1.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type
        self.loss_weight = loss_weight
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan':
            self.loss = self._wgan_loss
        elif self.gan_type == 'hinge':
            self.loss = nn.ReLU()
        else:
            raise NotImplementedError(
                f'GAN type {self.gan_type} is not implemented.')

    def _wgan_loss(self, input, target):
        """wgan loss.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """
        return -input.mean() if target else input.mean()

    def get_target_label(self, input, target_is_real):
        """Get target label.

        Args:
            input (Tensor): Input tensor.
            target_is_real (bool): Whether the targe is real or fake.

        Returns:
            (bool | Tensor): Target tensor. Return bool for wgan, otherwise,
                return Tensor.
        """

        if self.gan_type == 'wgan':
            return target_is_real
        target_val = (
            self.real_label_val if target_is_real else self.fake_label_val)
        return input.new_ones(input.size()) * target_val

    def forward(self, input, target_is_real, is_disc=False):
        """
        Args:
            input (Tensor): The input for the loss module, i.e., the network
                prediction.
            target_is_real (bool): Whether the targe is real or fake.
            is_disc (bool): Whether the loss for discriminators or not.
                Default: False.

        Returns:
            Tensor: GAN loss value.
        """
        target_label = self.get_target_label(input, target_is_real)
        if self.gan_type == 'hinge':
            if is_disc:  # for discriminators in hinge-gan
                input = -input if target_is_real else input
                loss = self.loss(1 + input).mean()
            else:  # for generators in hinge-gan
                loss = -input.mean()
        else:  # other gan types
            loss = self.loss(input, target_label)

        # loss_weight is always 1.0 for discriminators
        return loss if is_disc else loss * self.loss_weight


def gradient_penalty_loss(discriminator, real_data, fake_data, mask=None):
    """Calculate gradient penalty for wgan-gp.

    Args:
        discriminator (nn.Module): Network for the discriminator.
        real_data (Tensor): Real input data.
        fake_data (Tensor): Fake input data.
        mask (Tensor): Masks for inpaitting. Default: None.

    Returns:
        Tensor: A tensor for gradient penalty.
    """

    batch_size = real_data.size(0)
    alpha = real_data.new_tensor(torch.rand(batch_size, 1, 1, 1))

    # interpolate between real_data and fake_data
    interpolates = alpha * real_data + (1. - alpha) * fake_data
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = discriminator(interpolates)
    gradients = autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]

    if mask is not None:
        gradients = gradients * mask
    gradients = gradients.view(gradients.size(0), -1)
    gradients_penalty = ((gradients.norm(2, dim=1) - 1)**2).mean()

    return gradients_penalty


class GradientPenaltyLoss(nn.Module):
    """Gradient penalty loss for wgan-gp.

    Args:
        loss_weight (float): Loss weight. Default: 1.0.
    """

    def __init__(self, loss_weight=1.):
        super(GradientPenaltyLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, discriminator, real_data, fake_data, mask=None):
        """
        Args:
            discriminator (nn.Module): Network for the discriminator.
            real_data (Tensor): Real input data.
            fake_data (Tensor): Fake input data.
            mask (Tensor): Masks for inpaitting. Default: None.

        Returns:
            Tensor: Loss.
        """
        loss = gradient_penalty_loss(
            discriminator, real_data, fake_data, mask=mask)

        return loss * self.loss_weight


class TextureLoss(nn.Module):
    """ Define Texture Loss.

    Args:
        use_weights (bool): If True, the weights computed in swapping will be
            used to scale the features.
            Default: False
        loss_weight (float): Loss weight. Default: 1.0.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        layer_weights (dict): The weight for each layer of vgg feature.
            Defalut: {'relu1_1': 1.0, 'relu2_1': 1.0, 'relu3_1': 1.0}
        use_input_norm (bool): If True, normalize the input image.
            Default: True.
    """

    def __init__(self,
                 use_weights=False,
                 loss_weight=1.0,
                 vgg_type='vgg19',
                 layer_weights={
                     'relu1_1': 1.0,
                     'relu2_1': 1.0,
                     'relu3_1': 1.0
                 },
                 use_input_norm=True):
        super(TextureLoss, self).__init__()
        self.use_weights = use_weights
        self.loss_weight = loss_weight

        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm)

    def gram_matrix(self, features):
        n, c, h, w = features.size()
        feat_reshaped = features.view(n, c, -1)

        # Use torch.bmm for batch multiplication of matrices
        gram = torch.bmm(feat_reshaped, feat_reshaped.transpose(1, 2))

        return gram

    def forward(self, x, maps, weights=0):
        """
        Args:
            x (Tensor): The input for the loss module.
            maps (Tensor): The maps generated by swap module.
            weights (bool): The weights generated by swap module. The weights
                are used for scale the maps.

        Returns:
            Tensor: Texture Loss value.
        """
        input_size = x.shape[-1]
        x_features = self.vgg(x)

        losses = 0.0
        if self.use_weights:
            if not isinstance(weights, dict):
                weights = F.pad(weights, (1, 1, 1, 1), mode='replicate')
        for k in x_features.keys():
            if self.use_weights:
                # adjust the scale according to the name of layer
                if k == 'relu3_1':
                    idx = 0
                    div_num = 256
                elif k == 'relu2_1':
                    idx = 1
                    div_num = 512
                elif k == 'relu1_1':
                    idx = 2
                    div_num = 1024
                else:
                    raise NotImplementedError

                if isinstance(weights, dict):
                    weights_scaled = F.pad(
                        weights[k], (1, 1, 1, 1), mode='replicate')
                else:
                    weights_scaled = F.interpolate(weights, None, 2**idx,
                                                   'bicubic', True)

                # compute coefficients
                # TODO: the input range of tensorflow and pytorch are different,
                # check the values of a and b
                coeff = weights_scaled * (-20.) + .65
                coeff = torch.sigmoid(coeff)

                # weighting features and swapped maps
                maps[k] = maps[k] * coeff
                x_features[k] = x_features[k] * coeff

            # TODO: think about why 4 and **2
            losses += torch.norm(
                self.gram_matrix(x_features[k]) -
                self.gram_matrix(maps[k])) / 4. / (
                    (input_size * input_size * div_num)**2)

        losses = losses / 3.

        return losses * self.loss_weight


class MapLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layers_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4 feature
            layer (before relu5_4) will be extracted with weight 1.0 in
            calculting losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        norm_img (bool): If True, the image will be normed to [0, 1]. Note that
            this is different from the `use_input_norm` which norm the input in
            in forward function of vgg according to the statistics of dataset.
            Importantly, the input image must be in range [-1, 1].
        pretrained (str): Path for pretrained weights. Default:
            'torchvision://vgg19'
    """

    def __init__(self, vgg_type='vgg19', map_weight=1.0, criterion='l1'):
        super(MapLoss, self).__init__()
        self.map_weight = map_weight
        self.vgg = VGGFeatureExtractor(
            layer_name_list=['relu3_1', 'relu2_1', 'relu1_1'],
            vgg_type=vgg_type)

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.MSELoss()
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(
                f'{criterion} criterion has not been supported in'
                ' this version.')

    def forward(self, swapped_features, gt):
        # extract vgg features
        gt_features = self.vgg(gt.detach())

        # calculate loss loss
        map_loss = 0
        for k in gt_features.keys():
            if self.criterion_type == 'fro':
                map_loss += torch.norm(
                    swapped_features[k] - gt_features[k], p='fro')
            else:
                map_loss += self.criterion(swapped_features[k], gt_features[k])
        map_loss *= self.map_weight

        return map_loss
