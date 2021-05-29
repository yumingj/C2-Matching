import importlib
import logging
import os.path as osp
from collections import OrderedDict

import mmcv
import torch

import mmsr.models.networks as networks
import mmsr.utils.metrics as metrics
from mmsr.utils import ProgressBar, tensor2img

from .base_model import BaseModel

loss_module = importlib.import_module('mmsr.models.losses')

logger = logging.getLogger('base')


class SRModel(BaseModel):
    """Single image SR model.
    """

    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        # define network
        self.net_g = networks.define_net_g(opt)
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_model_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path']['strict_load'])

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        # define loss
        if train_opt['pixel_weight'] > 0:
            cri_pix_cls = getattr(loss_module, train_opt['pixel_criterion'])
            reduction = train_opt['reduction']
            self.cri_pix = cri_pix_cls(
                loss_weight=train_opt['pixel_weight'],
                reduction=reduction).to(self.device)
        else:
            logger.info('Remove pixel loss.')
            self.cri_pix = None

        if train_opt.get('perceptual_opt', None):
            percep_type = train_opt['perceptual_opt'].pop('type')
            cri_perceptual_cls = getattr(loss_module, percep_type)
            self.cri_perceptual = cri_perceptual_cls(
                **train_opt['perceptual_opt']).to(self.device)
        else:
            logger.info('Remove perceptual loss.')
            self.cri_perceptual = None
        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

        self.log_dict = OrderedDict()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        weight_decay_g = train_opt.get('weight_decay_g', 0)
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger.warning(f'Params {k} will not be optimized.')
        self.optimizer_g = torch.optim.Adam(
            optim_params,
            lr=train_opt['lr_g'],
            weight_decay=weight_decay_g,
            betas=train_opt['beta_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, step):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_total = 0
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            self.log_dict['l_pix'] = l_pix.item()
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                self.log_dict['l_percep'] = l_percep.item()
            if l_style is not None:
                l_total += l_style
                self.log_dict['l_style'] = l_style.item()

        l_total.backward()
        self.optimizer_g.step()

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            self.output = self.net_g(self.lq)
        self.net_g.train()

    def test_x8(self):
        # TODO
        # from https://github.com/thstkdgus35/EDSR-PyTorch
        self.net_g.eval()

        def _transform(v, op):
            # if self.precision != 'single': v = v.float()
            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            # if self.precision == 'half': ret = ret.half()

            return ret

        lr_list = [self.lq]
        for tf in 'v', 'h', 't':
            lr_list.extend([_transform(t, tf) for t in lr_list])
        with torch.no_grad():
            sr_list = [self.net_g(aug) for aug in lr_list]
        for i in range(len(sr_list)):
            if i > 3:
                sr_list[i] = _transform(sr_list[i], 't')
            if i % 4 > 1:
                sr_list[i] = _transform(sr_list[i], 'h')
            if (i % 4) % 2 == 1:
                sr_list[i] = _transform(sr_list[i], 'v')

        output_cat = torch.cat(sr_list, dim=0)
        self.output = output_cat.mean(dim=0, keepdim=True)
        self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        logger.info('Only support single GPU validation.')
        self.nondist_val(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img):
        pbar = ProgressBar(len(dataloader))
        avg_psnr = 0.
        dataset_name = dataloader.dataset.opt['name']
        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img, gt_img = tensor2img([visuals['rlt'], visuals['gt']])

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'],
                                             img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    save_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f"{img_name}_{self.opt['name']}.png")
                    if self.opt['suffix']:
                        save_img_path = save_img_path.replace(
                            '.png', f'_{self.opt["suffix"]}.png')
                mmcv.imwrite(sr_img, save_img_path)

            # tentative for out of GPU memory
            del self.lq
            del self.output
            del self.gt
            torch.cuda.empty_cache()

            # calculate PSNR
            avg_psnr += metrics.psnr(
                sr_img, gt_img, crop_border=self.opt['crop_border'])
            pbar.update(f'Test {img_name}')

        avg_psnr = avg_psnr / (idx + 1)

        # log
        logger.info(f'# Validation {dataset_name} # PSNR: {avg_psnr:.4e}.')
        if tb_logger:
            tb_logger.add_scalar('psnr', avg_psnr, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['rlt'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
