# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import csv
from mmpose.registry import MODELS
from mmengine import MessageHub
@MODELS.register_module()
class OPCLoss(nn.Module):
    def __init__(self,
                 use_target_weight: bool = False,
                 skip_empty_channel: bool = False,
                 sigma0: float = 3.0,
                 lamb: float = 0.1,
                 l2_lambda: float = 0.3,  # 这里的 l2_lambda 作为初始最大值
                 max_val: int = 2,
                 loss_weight: float = 1.):
        super().__init__()
        self.use_target_weight = use_target_weight
        self.skip_empty_channel = skip_empty_channel
        self.loss_weight = loss_weight
        self.sigma0 = sigma0
        self.max_val = max_val
        self.lamb = lamb
        self.l2_lambda_max = l2_lambda  # 最大值 0.5
        self.l2_lambda = 0.0  # 初始化为 0

    def _flat_softmax(self, Di: Tensor) -> Tensor:
        """Use Softmax to normalize the featmaps in depthwise."""
        return F.softmax(Di, dim=2)

    def forward(self,
                output: Tensor,
                target: Tensor,
                cood: Tensor,
                target_weights: Optional[Tensor] = None,
                mask: Optional[Tensor] = None) -> Tensor:

        message_hub = MessageHub.get_current_instance()
        epoch_num = message_hub.get_info('epoch')  # 获取当前的 epoch 轮次

        # 线性增长 l2_lambda
        if epoch_num < 150:
            self.l2_lambda = self.l2_lambda_max * (epoch_num / 150)
        else:
            self.l2_lambda = self.l2_lambda_max  # 保持最大值 0.5

        # 生成高斯分布
        d_norm = self.generate_gaussian(cood)

        # 计算 L2 正则项
        l2_reg_term = torch.mul(output, d_norm)
        L2 = torch.norm(l2_reg_term, dim=-1)
        L2 = torch.norm(L2, dim=-1)

        # 计算 MSE loss
        _loss = F.mse_loss(output, target, reduction='none')
        MSE = torch.sum(_loss, dim=-1)
        MSE = torch.sum(MSE, dim=-1)

        # 计算总 loss
        loss = (MSE + self.l2_lambda * L2).mean()

        return loss * self.loss_weight

    # def generate_gaussian(self, gt_coords, confidence):
    #     # Use confidence as sigma directly (no detaching)
    #     sigma = confidence.unsqueeze(2).unsqueeze(3)  # Expanding dimensions for broadcasting
    #     sigma=sigma+2
    #     # Create meshgrid for x and y coordinates
    #     x = torch.arange(0, 64, 1, dtype=torch.float32).cuda()
    #     y = torch.arange(0, 64, 1, dtype=torch.float32).cuda()
    #     x, y = torch.meshgrid(x, y)
    #
    #     # Expand meshgrid for broadcasting
    #     x = x.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, 64, 64]
    #     y = y.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, 64, 64]
    #
    #     # Expand gt_coords to match dimensions for broadcasting
    #     gt_coords_x = gt_coords[..., 0].unsqueeze(-1).unsqueeze(-1)  # Shape: [64, 98, 1, 1]
    #     gt_coords_y = gt_coords[..., 1].unsqueeze(-1).unsqueeze(-1)  # Shape: [64, 98, 1, 1]
    #
    #     # Compute distances
    #     distance_x = torch.abs(x - gt_coords_x)
    #     distance_y = torch.abs(y - gt_coords_y)
    #     distance = torch.sqrt(distance_x ** 2 + distance_y ** 2)
    #
    #     # Compute d_max based on sigma (confidence), without detaching
    #     d_max = torch.exp(self.max_val / sigma).cuda()
    #
    #     # Compute the Gaussian distribution
    #     d = torch.where(distance > self.max_val, d_max, torch.exp(distance  / (sigma ** 2)))
    #
    #     # Normalize the distribution
    #     d_norm = d / d_max
    #     return d_norm
    def generate_gaussian(self, gt_coords):

    # 创建网格坐标x和y
        x = torch.arange(0, 64, 1, dtype=torch.float32).cuda()
        y = torch.arange(0, 64, 1, dtype=torch.float32).cuda()
        x, y = torch.meshgrid(x, y)
    # 将网格扩展到正确的维度以进行广播
        x = x.unsqueeze(0).unsqueeze(0)  # [1, 1, 512, 512]
        y = y.unsqueeze(0).unsqueeze(0)  # [1, 1, 512, 512]
    # gt_coords扩展到相应维度
        gt_coords_x = gt_coords[..., 0].unsqueeze(-1).unsqueeze(-1)  # [64, 98, 1, 1]
        gt_coords_y = gt_coords[..., 1].unsqueeze(-1).unsqueeze(-1)  # [64, 98, 1, 1]
    # 计算距离
        distance_x = torch.abs(x - gt_coords_x)
        distance_y = torch.abs(y - gt_coords_y)
        distance = torch.sqrt(distance_x ** 2 + distance_y ** 2)
        d_max = torch.exp(torch.tensor(self.max_val / self.sigma0, dtype=torch.float32)).cuda()
        d = torch.where(distance > self.max_val, d_max, torch.exp(distance ** 2 / (2 * self.sigma0 ** 2)))
        d_norm = d / d_max
        return d_norm
    # def generate_gaussian(self, gt_coords):
    #     # 创建网格坐标x和y
    #     x = torch.arange(0, 64, 1, dtype=torch.float32).cuda()
    #     y = torch.arange(0, 64, 1, dtype=torch.float32).cuda()
    #     x, y = torch.meshgrid(x, y)
    #     # 将网格扩展到正确的维度以进行广播
    #     x = x.unsqueeze(0).unsqueeze(0)  # [1, 1, 512, 512]
    #     y = y.unsqueeze(0).unsqueeze(0)  # [1, 1, 512, 512]
    #     # gt_coords扩展到相应维度
    #     gt_coords_x = gt_coords[..., 0].unsqueeze(-1).unsqueeze(-1)  # [64, 98, 1, 1]
    #     gt_coords_y = gt_coords[..., 1].unsqueeze(-1).unsqueeze(-1)  # [64, 98, 1, 1]
    #     # 计算距离
    #     distance_x = torch.abs(x - gt_coords_x)
    #     distance_y = torch.abs(y - gt_coords_y)
    #     distance = torch.sqrt(distance_x ** 2 + distance_y ** 2)
    #
    #     d_max = torch.exp(torch.tensor(self.max_val / self.sigma0, dtype=torch.float32)).cuda()
    #     d = torch.where(distance > self.max_val, d_max, torch.exp(distance ** 2 / (2 * self.sigma0 ** 2)))
    #     d_norm = d / d_max
    #     return d_norm
    def _get_mask(self, target: Tensor, target_weights: Optional[Tensor],
                  mask: Optional[Tensor]) -> Optional[Tensor]:
        """Generate the heatmap mask w.r.t. the given mask, target weight and
        `skip_empty_channel` setting.

        Returns:
            Tensor: The mask in shape (B, K, *) or ``None`` if no mask is
            needed.
        """
        # Given spatial mask
        if mask is not None:
            # check mask has matching type with target
            assert (mask.ndim == target.ndim and all(
                d_m == d_t or d_m == 1
                for d_m, d_t in zip(mask.shape, target.shape))), (
                    f'mask and target have mismatched shapes {mask.shape} v.s.'
                    f'{target.shape}')

        # Mask by target weights (keypoint-wise mask)
        if target_weights is not None:
            # check target weight has matching shape with target
            assert (target_weights.ndim in (2, 4) and target_weights.shape
                    == target.shape[:target_weights.ndim]), (
                        'target_weights and target have mismatched shapes '
                        f'{target_weights.shape} v.s. {target.shape}')

            ndim_pad = target.ndim - target_weights.ndim
            _mask = target_weights.view(target_weights.shape +
                                        (1, ) * ndim_pad)

            if mask is None:
                mask = _mask
            else:
                mask = mask * _mask

        # Mask by ``skip_empty_channel``
        if self.skip_empty_channel:
            _mask = (target != 0).flatten(2).any(dim=2)
            ndim_pad = target.ndim - _mask.ndim
            _mask = _mask.view(_mask.shape + (1, ) * ndim_pad)

            if mask is None:
                mask = _mask
            else:
                mask = mask * _mask

        return mask
@MODELS.register_module()
class KeypointMSELoss(nn.Module):
    """MSE loss for heatmaps.

    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
            Defaults to ``False``
        skip_empty_channel (bool): If ``True``, heatmap channels with no
            non-zero value (which means no visible ground-truth keypoint
            in the image) will not be used to calculate the loss. Defaults to
            ``False``
        loss_weight (float): Weight of the loss. Defaults to 1.0
    """

    def __init__(self,
                 use_target_weight: bool = False,
                 skip_empty_channel: bool = False,
                 loss_weight: float = 1.):
        super().__init__()
        self.use_target_weight = use_target_weight
        self.skip_empty_channel = skip_empty_channel
        self.loss_weight = loss_weight

    def forward(self,
                output: Tensor,
                target: Tensor,
                target_weights: Optional[Tensor] = None,
                mask: Optional[Tensor] = None) -> Tensor:
        """Forward function of loss.

        Note:
            - batch_size: B
            - num_keypoints: K
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            output (Tensor): The output heatmaps with shape [B, K, H, W]
            target (Tensor): The target heatmaps with shape [B, K, H, W]
            target_weights (Tensor, optional): The target weights of differet
                keypoints, with shape [B, K] (keypoint-wise) or
                [B, K, H, W] (pixel-wise).
            mask (Tensor, optional): The masks of valid heatmap pixels in
                shape [B, K, H, W] or [B, 1, H, W]. If ``None``, no mask will
                be applied. Defaults to ``None``

        Returns:
            Tensor: The calculated loss.
        """

        _mask = self._get_mask(target, target_weights, mask)
        if _mask is None:
            loss = F.mse_loss(output, target)
        else:
            _loss = F.mse_loss(output, target, reduction='none')
            loss = (_loss * _mask).mean()

        return loss * self.loss_weight

    def _get_mask(self, target: Tensor, target_weights: Optional[Tensor],
                  mask: Optional[Tensor]) -> Optional[Tensor]:
        """Generate the heatmap mask w.r.t. the given mask, target weight and
        `skip_empty_channel` setting.

        Returns:
            Tensor: The mask in shape (B, K, *) or ``None`` if no mask is
            needed.
        """
        # Given spatial mask
        if mask is not None:
            # check mask has matching type with target
            assert (mask.ndim == target.ndim and all(
                d_m == d_t or d_m == 1
                for d_m, d_t in zip(mask.shape, target.shape))), (
                    f'mask and target have mismatched shapes {mask.shape} v.s.'
                    f'{target.shape}')

        # Mask by target weights (keypoint-wise mask)
        if target_weights is not None:
            # check target weight has matching shape with target
            assert (target_weights.ndim in (2, 4) and target_weights.shape
                    == target.shape[:target_weights.ndim]), (
                        'target_weights and target have mismatched shapes '
                        f'{target_weights.shape} v.s. {target.shape}')

            ndim_pad = target.ndim - target_weights.ndim
            _mask = target_weights.view(target_weights.shape +
                                        (1, ) * ndim_pad)

            if mask is None:
                mask = _mask
            else:
                mask = mask * _mask

        # Mask by ``skip_empty_channel``
        if self.skip_empty_channel:
            _mask = (target != 0).flatten(2).any(dim=2)
            ndim_pad = target.ndim - _mask.ndim
            _mask = _mask.view(_mask.shape + (1, ) * ndim_pad)

            if mask is None:
                mask = _mask
            else:
                mask = mask * _mask

        return mask


@MODELS.register_module()
class CombinedTargetMSELoss(nn.Module):
    """MSE loss for combined target.

    CombinedTarget: The combination of classification target
    (response map) and regression target (offset map).
    Paper ref: Huang et al. The Devil is in the Details: Delving into
    Unbiased Data Processing for Human Pose Estimation (CVPR 2020).

    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
            Defaults to ``False``
        loss_weight (float): Weight of the loss. Defaults to 1.0
    """

    def __init__(self,
                 use_target_weight: bool = False,
                 loss_weight: float = 1.):
        super().__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def forward(self, output: Tensor, target: Tensor,
                target_weights: Tensor) -> Tensor:
        """Forward function of loss.

        Note:
            - batch_size: B
            - num_channels: C
            - heatmaps height: H
            - heatmaps weight: W
            - num_keypoints: K
            Here, C = 3 * K

        Args:
            output (Tensor): The output feature maps with shape [B, C, H, W].
            target (Tensor): The target feature maps with shape [B, C, H, W].
            target_weights (Tensor): The target weights of differet keypoints,
                with shape [B, K].

        Returns:
            Tensor: The calculated loss.
        """
        batch_size = output.size(0)
        num_channels = output.size(1)
        heatmaps_pred = output.reshape(
            (batch_size, num_channels, -1)).split(1, 1)
        heatmaps_gt = target.reshape(
            (batch_size, num_channels, -1)).split(1, 1)
        loss = 0.
        num_joints = num_channels // 3
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx * 3].squeeze()
            heatmap_gt = heatmaps_gt[idx * 3].squeeze()
            offset_x_pred = heatmaps_pred[idx * 3 + 1].squeeze()
            offset_x_gt = heatmaps_gt[idx * 3 + 1].squeeze()
            offset_y_pred = heatmaps_pred[idx * 3 + 2].squeeze()
            offset_y_gt = heatmaps_gt[idx * 3 + 2].squeeze()
            if self.use_target_weight:
                target_weight = target_weights[:, idx, None]
                heatmap_pred = heatmap_pred * target_weight
                heatmap_gt = heatmap_gt * target_weight
            # classification loss
            loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)
            # regression loss
            loss += 0.5 * self.criterion(heatmap_gt * offset_x_pred,
                                         heatmap_gt * offset_x_gt)
            loss += 0.5 * self.criterion(heatmap_gt * offset_y_pred,
                                         heatmap_gt * offset_y_gt)
        return loss / num_joints * self.loss_weight


@MODELS.register_module()
class KeypointOHKMMSELoss(nn.Module):
    """MSE loss with online hard keypoint mining.

    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
            Defaults to ``False``
        topk (int): Only top k joint losses are kept. Defaults to 8
        loss_weight (float): Weight of the loss. Defaults to 1.0
    """

    def __init__(self,
                 use_target_weight: bool = False,
                 topk: int = 8,
                 loss_weight: float = 1.):
        super().__init__()
        assert topk > 0
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = use_target_weight
        self.topk = topk
        self.loss_weight = loss_weight

    def _ohkm(self, losses: Tensor) -> Tensor:
        """Online hard keypoint mining.

        Note:
            - batch_size: B
            - num_keypoints: K

        Args:
            loss (Tensor): The losses with shape [B, K]

        Returns:
            Tensor: The calculated loss.
        """
        ohkm_loss = 0.
        B = losses.shape[0]
        for i in range(B):
            sub_loss = losses[i]
            _, topk_idx = torch.topk(
                sub_loss, k=self.topk, dim=0, sorted=False)
            tmp_loss = torch.gather(sub_loss, 0, topk_idx)
            ohkm_loss += torch.sum(tmp_loss) / self.topk
        ohkm_loss /= B
        return ohkm_loss

    def forward(self, output: Tensor, target: Tensor,
                target_weights: Tensor) -> Tensor:
        """Forward function of loss.

        Note:
            - batch_size: B
            - num_keypoints: K
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            output (Tensor): The output heatmaps with shape [B, K, H, W].
            target (Tensor): The target heatmaps with shape [B, K, H, W].
            target_weights (Tensor): The target weights of differet keypoints,
                with shape [B, K].

        Returns:
            Tensor: The calculated loss.
        """
        num_keypoints = output.size(1)
        if num_keypoints < self.topk:
            raise ValueError(f'topk ({self.topk}) should not be '
                             f'larger than num_keypoints ({num_keypoints}).')

        losses = []
        for idx in range(num_keypoints):
            if self.use_target_weight:
                target_weight = target_weights[:, idx, None, None]
                losses.append(
                    self.criterion(output[:, idx] * target_weight,
                                   target[:, idx] * target_weight))
            else:
                losses.append(self.criterion(output[:, idx], target[:, idx]))

        losses = [loss.mean(dim=(1, 2)).unsqueeze(dim=1) for loss in losses]
        losses = torch.cat(losses, dim=1)

        return self._ohkm(losses) * self.loss_weight


@MODELS.register_module()
class AdaptiveWingLoss(nn.Module):
    """Adaptive wing loss. paper ref: 'Adaptive Wing Loss for Robust Face
    Alignment via Heatmap Regression' Wang et al. ICCV'2019.

    Args:
        alpha (float), omega (float), epsilon (float), theta (float)
            are hyper-parameters.
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self,
                 alpha=2.1,
                 omega=14,
                 epsilon=1,
                 theta=0.5,
                 use_target_weight=False,
                 loss_weight=1.):
        super().__init__()
        self.alpha = float(alpha)
        self.omega = float(omega)
        self.epsilon = float(epsilon)
        self.theta = float(theta)
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def criterion(self, pred, target):
        """Criterion of wingloss.

        Note:
            batch_size: N
            num_keypoints: K

        Args:
            pred (torch.Tensor[NxKxHxW]): Predicted heatmaps.
            target (torch.Tensor[NxKxHxW]): Target heatmaps.
        """
        H, W = pred.shape[2:4]
        delta = (target - pred).abs()

        A = self.omega * (
            1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - target))
        ) * (self.alpha - target) * (torch.pow(
            self.theta / self.epsilon,
            self.alpha - target - 1)) * (1 / self.epsilon)
        C = self.theta * A - self.omega * torch.log(
            1 + torch.pow(self.theta / self.epsilon, self.alpha - target))

        losses = torch.where(
            delta < self.theta,
            self.omega *
            torch.log(1 +
                      torch.pow(delta / self.epsilon, self.alpha - target)),
            A * delta - C)

        return torch.mean(losses)

    def forward(self,
                output: Tensor,
                target: Tensor,
                target_weights: Optional[Tensor] = None):
        """Forward function.

        Note:
            batch_size: N
            num_keypoints: K

        Args:
            output (torch.Tensor[N, K, H, W]): Output heatmaps.
            target (torch.Tensor[N, K, H, W]): Target heatmaps.
            target_weight (torch.Tensor[N, K]):
                Weights across different joint types.
        """
        if self.use_target_weight:
            assert (target_weights.ndim in (2, 4) and target_weights.shape
                    == target.shape[:target_weights.ndim]), (
                        'target_weights and target have mismatched shapes '
                        f'{target_weights.shape} v.s. {target.shape}')

            ndim_pad = target.ndim - target_weights.ndim
            target_weights = target_weights.view(target_weights.shape +
                                                 (1, ) * ndim_pad)
            loss = self.criterion(output * target_weights,
                                  target * target_weights)
        else:
            loss = self.criterion(output, target)

        return loss * self.loss_weight


@MODELS.register_module()
class FocalHeatmapLoss(KeypointMSELoss):
    """A class for calculating the modified focal loss for heatmap prediction.

    This loss function is exactly the same as the one used in CornerNet. It
    runs faster and costs a little bit more memory.

    `CornerNet: Detecting Objects as Paired Keypoints
    arXiv: <https://arxiv.org/abs/1808.01244>`_.

    Arguments:
        alpha (int): The alpha parameter in the focal loss equation.
        beta (int): The beta parameter in the focal loss equation.
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
            Defaults to ``False``
        skip_empty_channel (bool): If ``True``, heatmap channels with no
            non-zero value (which means no visible ground-truth keypoint
            in the image) will not be used to calculate the loss. Defaults to
            ``False``
        loss_weight (float): Weight of the loss. Defaults to 1.0
    """

    def __init__(self,
                 alpha: int = 2,
                 beta: int = 4,
                 use_target_weight: bool = False,
                 skip_empty_channel: bool = False,
                 loss_weight: float = 1.0):
        super(FocalHeatmapLoss, self).__init__(use_target_weight,
                                               skip_empty_channel, loss_weight)
        self.alpha = alpha
        self.beta = beta

    def forward(self,
                output: Tensor,
                target: Tensor,
                target_weights: Optional[Tensor] = None,
                mask: Optional[Tensor] = None) -> Tensor:
        """Calculate the modified focal loss for heatmap prediction.

        Note:
            - batch_size: B
            - num_keypoints: K
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            output (Tensor): The output heatmaps with shape [B, K, H, W]
            target (Tensor): The target heatmaps with shape [B, K, H, W]
            target_weights (Tensor, optional): The target weights of differet
                keypoints, with shape [B, K] (keypoint-wise) or
                [B, K, H, W] (pixel-wise).
            mask (Tensor, optional): The masks of valid heatmap pixels in
                shape [B, K, H, W] or [B, 1, H, W]. If ``None``, no mask will
                be applied. Defaults to ``None``

        Returns:
            Tensor: The calculated loss.
        """
        _mask = self._get_mask(target, target_weights, mask)

        pos_inds = target.eq(1).float()
        neg_inds = target.lt(1).float()

        if _mask is not None:
            pos_inds = pos_inds * _mask
            neg_inds = neg_inds * _mask

        neg_weights = torch.pow(1 - target, self.beta)

        pos_loss = torch.log(output) * torch.pow(1 - output,
                                                 self.alpha) * pos_inds
        neg_loss = torch.log(1 - output) * torch.pow(
            output, self.alpha) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        if num_pos == 0:
            loss = -neg_loss.sum()
        else:
            loss = -(pos_loss.sum() + neg_loss.sum()) / num_pos
        return loss * self.loss_weight


@MODELS.register_module()
class MLECCLoss(nn.Module):
    """Maximum Likelihood Estimation loss for Coordinate Classification.

    This loss function is designed to work with coordinate classification
    problems where the likelihood of each target coordinate is maximized.

    Args:
        reduction (str): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. Default: 'mean'.
        mode (str): Specifies the mode of calculating loss:
            'linear' | 'square' | 'log'. Default: 'log'.
        use_target_weight (bool): If True, uses weighted loss. Different
            joint types may have different target weights. Defaults to False.
        loss_weight (float): Weight of the loss. Defaults to 1.0.

    Raises:
        AssertionError: If the `reduction` or `mode` arguments are not in the
                        expected choices.
        NotImplementedError: If the selected mode is not implemented.
    """

    def __init__(self,
                 reduction: str = 'mean',
                 mode: str = 'log',
                 use_target_weight: bool = False,
                 loss_weight: float = 1.0):
        super().__init__()
        assert reduction in ('mean', 'sum', 'none'), \
            f"`reduction` should be either 'mean', 'sum', or 'none', " \
            f'but got {reduction}'
        assert mode in ('linear', 'square', 'log'), \
            f"`mode` should be either 'linear', 'square', or 'log', " \
            f'but got {mode}'

        self.reduction = reduction
        self.mode = mode
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def forward(self, outputs, targets, target_weight=None):
        """Forward pass for the MLECCLoss.

        Args:
            outputs (torch.Tensor): The predicted outputs.
            targets (torch.Tensor): The ground truth targets.
            target_weight (torch.Tensor, optional): Optional tensor of weights
                for each target.

        Returns:
            torch.Tensor: Calculated loss based on the specified mode and
                reduction.
        """

        assert len(outputs) == len(targets), \
            'Outputs and targets must have the same length'

        prob = 1.0
        for o, t in zip(outputs, targets):
            prob *= (o * t).sum(dim=-1)

        if self.mode == 'linear':
            loss = 1.0 - prob
        elif self.mode == 'square':
            loss = 1.0 - prob.pow(2)
        elif self.mode == 'log':
            loss = -torch.log(prob + 1e-4)

        loss[torch.isnan(loss)] = 0.0

        if self.use_target_weight:
            assert target_weight is not None
            for i in range(loss.ndim - target_weight.ndim):
                target_weight = target_weight.unsqueeze(-1)
            loss = loss * target_weight

        if self.reduction == 'sum':
            loss = loss.flatten(1).sum(dim=1)
        elif self.reduction == 'mean':
            loss = loss.flatten(1).mean(dim=1)

        return loss * self.loss_weight
