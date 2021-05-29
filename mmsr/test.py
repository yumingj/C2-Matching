import argparse
import logging
import os.path as osp

from mmcv.runner import get_time_str, init_dist

from mmsr.data import create_dataloader, create_dataset
from mmsr.models import create_model
from mmsr.utils import get_root_logger, make_exp_dirs
from mmsr.utils.options import dict2str, dict_to_nonedict, parse


def main():
    # options
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-opt', type=str, required=True, help='Path to option YMAL file.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = parse(args.opt, is_train=False)

    # distributed testing settings
    if args.launcher == 'none':  # disabled distributed training
        opt['dist'] = False
        print('Disabled distributed testing.', flush=True)
    else:
        opt['dist'] = True
        if args.launcher == 'slurm' and 'dist_params' in opt:
            init_dist(args.launcher, **opt['dist_params'])
        else:
            init_dist(args.launcher)

    opt = dict_to_nonedict(opt)

    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'],
                        f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name='base', log_level=logging.INFO, log_file=log_file)
    logger.info(dict2str(opt))

    # create test dataset and dataloader
    test_loaders = []
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(test_set, dataset_opt)
        logger.info(
            f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)

    # create model
    model = create_model(opt)

    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info(f'Testing {test_set_name}...')
        model.validation(
            test_loader,
            current_iter=opt['name'],
            tb_logger=None,
            save_img=opt['save_img'])


# TODO: need to be sorted out.
# need_avg_print = False
# for test_loader in test_loaders:
#     test_set_name = test_loader.dataset.opt['name']
#     logger.info(f'\nTesting {test_set_name}...')

#     dataset_dir = osp.join(opt['path']['results_root'], test_set_name)

#     test_results = OrderedDict()
#     test_results['psnr'] = []
#     test_results['ssim'] = []
#     test_results['psnr_y'] = []
#     test_results['ssim_y'] = []

#     for data in test_loader:
#         img_name = osp.splitext(osp.basename(data['lq_path'][0]))[0]

#         model.feed_data(data)
#         model.test()
#         visuals = model.get_current_visuals()
#         sr_img = tensor2img(visuals['rlt'])

#         # save images
#         suffix = opt['suffix']
#         if suffix:
#             save_img_path = osp.join(dataset_dir,
#                                      f'{img_name}_{suffix}.png')
#         else:
#             save_img_path = osp.join(dataset_dir, f'{img_name}.png')
#         mmcv.imwrite(sr_img, save_img_path)

#         if 'gt' in visuals:
#             need_avg_print = True
#             gt_img = tensor2img(visuals['gt'])

#             # calculate PSNR and SSIM
#             psnr = metrics.psnr(
#                 sr_img, gt_img, crop_border=opt['crop_border'])
#             ssim = metrics.ssim(
#                 sr_img, gt_img, crop_border=opt['crop_border'])
#             test_results['psnr'].append(psnr)
#             test_results['ssim'].append(ssim)

#             if gt_img.shape[2] == 3:
#                 sr_img_y = bgr2ycbcr(sr_img / 255., only_y=True)
#                 gt_img_y = bgr2ycbcr(gt_img / 255., only_y=True)

#                 psnr_y = metrics.psnr(
#                     sr_img_y * 255,
#                     gt_img_y * 255,
#                     crop_border=opt['crop_border'])
#                 ssim_y = metrics.ssim(
#                     sr_img_y * 255,
#                     gt_img_y * 255,
#                     crop_border=opt['crop_border'])
#                 test_results['psnr_y'].append(psnr_y)
#                 test_results['ssim_y'].append(ssim_y)
#                 logger.info(f'{img_name:20s} - PSNR: {psnr:.6f} dB; '
#                             f'SSIM: {ssim:.6f}; PSNR_Y: {psnr_y:.6f} dB; '
#                             f'SSIM_Y: {ssim_y:.6f}.')
#             else:
#                 logger.info(f'{img_name:20s} - PSNR: {psnr:.6f} dB; '
#                             f'SSIM: {ssim:.6f}.')
#         else:
#             logger.info(img_name)

#     # Average PSNR/SSIM results
#     if need_avg_print:
#         ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
#         ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
#         logger.info(
#             f'----Average PSNR/SSIM results for {test_set_name}----\n\t'
#             f'PSNR: {ave_psnr:.6f} dB; SSIM: {ave_ssim:.6f}\n')
#         if test_results['psnr_y'] and test_results['ssim_y']:
#             ave_psnr_y = sum(test_results['psnr_y']) / len(
#                 test_results['psnr_y'])
#             ave_ssim_y = sum(test_results['ssim_y']) / len(
#                 test_results['ssim_y'])
#             logger.info(
#                 '----Y channel, average PSNR/SSIM----\n\t'
#                 f'PSNR_Y: {ave_psnr_y:.6f} dB; SSIM_Y: {ave_ssim_y:.6f}\n')

if __name__ == '__main__':
    main()
