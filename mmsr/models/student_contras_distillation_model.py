import importlib
import logging
import os.path as osp
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F

import mmsr.models.networks as networks
from mmsr.utils import ProgressBar

from .base_model import BaseModel

loss_module = importlib.import_module('mmsr.models.losses')

logger = logging.getLogger('base')


def grid_positions(h, w, device, matrix=False):
    lines = torch.arange(0, h, device=device).view(-1, 1).float().repeat(1, w)
    columns = torch.arange(
        0, w, device=device).view(1, -1).float().repeat(h, 1)
    if matrix:
        return torch.stack([lines, columns], dim=0)
    else:
        return torch.cat([lines.view(1, -1), columns.view(1, -1)], dim=0)


def upscale_positions(pos, scaling_steps=0):
    for _ in range(scaling_steps):
        pos = pos * 2
    return pos


def downscale_positions(pos, scaling_steps=0):
    for _ in range(scaling_steps):
        pos = pos / 2
    return pos


def warp(pos1, max_h, max_w, transformed_coordinates):
    device = pos1.device
    ids = torch.arange(0, pos1.size(1), device=device)

    transformed_coordinates = transformed_coordinates[::4, ::4, :2]
    # dim 0: x, dim 1: y
    pos2 = transformed_coordinates.permute(2, 0, 1).reshape(2, -1)
    transformed_x = pos2[0, :]
    transformed_y = pos2[1, :]

    # eliminate the outlier pixels
    valid_ids_x = torch.min(transformed_x > 10, transformed_x < (max_w - 10))
    valid_ids_y = torch.min(transformed_y > 10, transformed_y < (max_h - 10))

    valid_ids = torch.min(valid_ids_x, valid_ids_y)

    ids = ids[valid_ids]
    pos1 = pos1[:, valid_ids]
    pos2 = pos2[:, valid_ids]

    pos2 = pos2[[1, 0], :]

    return pos1, pos2, ids


class StudentContrasDistillationModel(BaseModel):

    def __init__(self, opt):
        super(StudentContrasDistillationModel, self).__init__(opt)
        # define student network
        self.net_student = networks.define_net_student(opt)
        self.net_student = self.model_to_device(self.net_student)
        self.print_network(self.net_student)

        # define teacher network (fixed during training)
        self.net_teacher = networks.define_net_teacher(opt)
        self.net_teacher = self.model_to_device(self.net_teacher)
        self.print_network(self.net_teacher)

        # load pretrained teacher models
        load_path = self.opt['path'].get('pretrain_model_student', None)
        if load_path is not None:
            self.load_network(self.net_student, load_path,
                              self.opt['path']['strict_load'])

        # load pretrained teacher models
        load_path = self.opt['path'].get('pretrain_model_teacher', None)
        if load_path is not None:
            self.load_network(self.net_teacher, load_path,
                              self.opt['path']['strict_load'])

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_student.train()
        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

        self.log_dict = OrderedDict()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_student.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger.warning(f'Params {k} will not be optimized.')
        self.optimizer_g = torch.optim.Adam(optim_params, lr=train_opt['lr_g'])
        self.optimizers.append(self.optimizer_g)

        # hyper-parameters for loss
        self.margin = self.opt['train']['margin']
        self.safe_radius = self.opt['train']['safe_radius']
        self.scaling_steps = self.opt['train']['scaling_steps']
        self.distill_weight = self.opt['train']['distill_weight']
        self.temperature = self.opt['train']['temperature']

    def feed_data(self, data):
        self.img_in_lq = data['img_in_up'].to(self.device)
        self.img_in_gt = data['img_in'].to(self.device)
        self.img_ref_gt = data['img_ref'].to(self.device)
        self.transformed_coordinates = data['transformed_coordinate'].to(
            self.device)

    def loss_function(self):
        loss = torch.tensor(
            np.array([0], dtype=np.float32), device=self.device)
        pos_dist = 0.
        neg_dist = 0.
        distill_loss_all = 0.
        has_grad = False

        n_valid_samples = 0
        batch_size = self.output['dense_features1'].size(0)
        for idx_in_batch in range(batch_size):

            # Network output
            # shape: [c, h1, w1]
            dense_features1 = self.output['dense_features1'][idx_in_batch]
            c, h1, w1 = dense_features1.size()

            # shape: [c, h2, w2]
            dense_features2 = self.output['dense_features2'][idx_in_batch]
            _, h2, w2 = dense_features2.size()

            # shape: [c, h1 * w1]
            all_descriptors1 = F.normalize(dense_features1.view(c, -1), dim=0)
            descriptors1 = all_descriptors1

            # Warp the positions from image 1 to image 2\
            # shape: [2, h1 * w1], coordinate in [h1, w1] dim,
            # dim 0: y, dim 1: x
            # positions in feature map
            fmap_pos1 = grid_positions(h1, w1, self.device)
            # shape: [2, h1 * w1], coordinate in image level (4 * h1, 4 * w1)
            pos1 = upscale_positions(
                fmap_pos1, scaling_steps=self.scaling_steps)
            pos1, pos2, ids = warp(pos1, 4 * h1, 4 * w1,
                                   self.transformed_coordinates[idx_in_batch])

            # shape: [2, num_ids]
            fmap_pos1 = fmap_pos1[:, ids]
            # shape: [c, num_ids]
            descriptors1 = descriptors1[:, ids]

            # Skip the pair if not enough GT correspondences are available
            if ids.size(0) < 128:
                continue

            # Descriptors at the corresponding positions
            fmap_pos2 = torch.round(
                downscale_positions(pos2,
                                    scaling_steps=self.scaling_steps)).long()
            descriptors2 = F.normalize(
                dense_features2[:, fmap_pos2[0, :], fmap_pos2[1, :]], dim=0)

            positive_distance = 2 - 2 * (descriptors1.t().unsqueeze(
                1) @ descriptors2.t().unsqueeze(2)).squeeze()

            position_distance = torch.max(
                torch.abs(
                    fmap_pos2.unsqueeze(2).float() - fmap_pos2.unsqueeze(1)),
                dim=0)[0]
            is_out_of_safe_radius = position_distance > self.safe_radius
            distance_matrix = 2 - 2 * (descriptors1.t() @ descriptors2)
            negative_distance2 = torch.min(
                distance_matrix + (1 - is_out_of_safe_radius.float()) * 10.,
                dim=1)[0]

            all_fmap_pos1 = grid_positions(h1, w1, self.device)
            position_distance = torch.max(
                torch.abs(
                    fmap_pos1.unsqueeze(2).float() -
                    all_fmap_pos1.unsqueeze(1)),
                dim=0)[0]
            is_out_of_safe_radius = position_distance > self.safe_radius
            distance_matrix = 2 - 2 * (descriptors2.t() @ all_descriptors1)
            negative_distance1 = torch.min(
                distance_matrix + (1 - is_out_of_safe_radius.float()) * 10.,
                dim=1)[0]

            diff = positive_distance - torch.min(negative_distance1,
                                                 negative_distance2)

            # distillation loss
            # student model correlation
            student_distance = torch.matmul(
                descriptors1.transpose(0, 1), descriptors2)
            student_distance = student_distance / self.temperature
            student_distance = F.log_softmax(student_distance, dim=1)

            # teacher model correlation
            teacher_dense_features1 = self.teacher_feat['dense_features1'][
                idx_in_batch]
            c, h1, w1 = dense_features1.size()
            teacher_descriptors1 = F.normalize(
                teacher_dense_features1.view(c, -1), dim=0)
            teacher_descriptors1 = teacher_descriptors1[:, ids]

            teacher_dense_features2 = self.teacher_feat['dense_features2'][
                idx_in_batch]
            teacher_descriptors2 = F.normalize(
                teacher_dense_features2[:, fmap_pos2[0, :], fmap_pos2[1, :]],
                dim=0)
            teacher_distance = torch.matmul(
                teacher_descriptors1.transpose(0, 1), teacher_descriptors2)
            teacher_distance = teacher_distance / self.temperature
            teacher_distance = F.softmax(teacher_distance, dim=1)

            distill_loss = F.kl_div(
                student_distance, teacher_distance, reduction='batchmean')
            distill_loss_all += distill_loss

            loss = loss + torch.mean(
                F.relu(self.margin +
                       diff)) + self.distill_weight * distill_loss

            pos_dist = pos_dist + torch.mean(positive_distance)
            neg_dist = neg_dist + torch.mean(
                torch.min(negative_distance1, negative_distance2))

            has_grad = True
            n_valid_samples += 1

        if not has_grad:
            raise NotImplementedError

        loss = loss / n_valid_samples
        pos_dist = pos_dist / n_valid_samples
        neg_dist = neg_dist / n_valid_samples
        distill_loss_all = distill_loss_all / n_valid_samples

        return loss, pos_dist, neg_dist, distill_loss_all

    def optimize_parameters(self, step):
        self.optimizer_g.zero_grad()
        self.output = self.net_student(self.img_in_lq, self.img_ref_gt)
        self.teacher_feat = self.net_teacher(self.img_in_gt, self.img_ref_gt)

        loss, pos_dist, neg_dist, distill_loss_all = self.loss_function()

        self.log_dict['loss'] = loss.item()
        self.log_dict['pos_dist'] = pos_dist.item()
        self.log_dict['neg_dist'] = neg_dist.item()
        self.log_dict['distill_loss'] = distill_loss_all.item()

        loss.backward()
        self.optimizer_g.step()

    def test(self):
        self.net_student.eval()
        with torch.no_grad():
            self.output = self.net_student(self.img_in_lq, self.img_ref_gt)
            self.teacher_feat = self.net_teacher(self.img_in_gt,
                                                 self.img_ref_gt)
        self.net_student.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        logger.info('Only support single GPU validation.')
        self.nondist_val(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img):
        pbar = ProgressBar(len(dataloader))
        loss_val_all = 0.
        pos_dist_val_all = 0.
        neg_dist_val_all = 0.
        distill_loss_val_all = 0.
        dataset_name = dataloader.dataset.opt['name']
        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['name'][0]))[0]

            self.feed_data(val_data)
            self.test()

            loss_val, pos_dist_val, neg_dist_val, distill_loss_val = self.loss_function(  # noqa: E501
            )

            # tentative for out of GPU memory
            del self.img_in_lq
            del self.img_in_gt
            del self.img_ref_gt
            del self.transformed_coordinates
            del self.output
            torch.cuda.empty_cache()

            # calculate PSNR
            pbar.update(f'Test {img_name}')
            loss_val_all += loss_val.item()
            pos_dist_val_all += pos_dist_val.item()
            neg_dist_val_all += neg_dist_val.item()
            distill_loss_val_all += distill_loss_val.item()

        loss_val_all = loss_val_all / (idx + 1)
        pos_dist_val_all = pos_dist_val_all / (idx + 1)
        neg_dist_val_all = neg_dist_val_all / (idx + 1)
        distill_loss_val_all = distill_loss_val_all / (idx + 1)

        # log
        logger.info(
            f'# Validation {dataset_name} # loss_val: {loss_val_all:.4e} '
            f'# positive_distance: {pos_dist_val_all:.4e} '
            f'# negative_distance: {neg_dist_val_all:.4e} '
            f'# distill_loss: {distill_loss_val_all:.4e}.')
        if tb_logger:
            tb_logger.add_scalar('loss_val', loss_val_all, current_iter)

    def save(self, epoch, current_iter):
        self.save_network(self.net_student, 'net_student', current_iter)
        self.save_training_state(epoch, current_iter)
