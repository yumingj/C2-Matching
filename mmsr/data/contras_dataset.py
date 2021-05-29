import cv2
import mmcv
import numpy as np
import torch.utils.data as data
from PIL import Image

from mmsr.data.transforms import augment, mod_crop, totensor
from mmsr.data.util import (paired_paths_from_ann_file,
                            paired_paths_from_folder, paired_paths_from_lmdb)
from mmsr.utils import FileClient


def image_pair_generation(img,
                          random_perturb_range=(0, 32),
                          cropping_window_size=160):

    if img is not None:
        shape1 = img.shape
        h = shape1[0]
        w = shape1[1]
    else:
        h = 160
        w = 160

    # ===== in image-1
    cropS = cropping_window_size
    x_topleft = np.random.randint(random_perturb_range[1],
                                  max(w, w - cropS - random_perturb_range[1]))
    y_topleft = np.random.randint(random_perturb_range[1],
                                  max(h, h - cropS - random_perturb_range[1]))

    x_topright = x_topleft + cropS
    y_topright = y_topleft

    x_bottomleft = x_topleft
    y_bottomleft = y_topleft + cropS

    x_bottomright = x_topleft + cropS
    y_bottomright = y_topleft + cropS

    tl = (x_topleft, y_topleft)
    tr = (x_topright, y_topright)
    br = (x_bottomright, y_bottomright)
    bl = (x_bottomleft, y_bottomleft)

    rect1 = np.array([tl, tr, br, bl], dtype=np.float32)

    # ===== in image-2
    x2_topleft = x_topleft + np.random.randint(
        random_perturb_range[0], random_perturb_range[1]) * np.random.choice(
            [-1.0, 1.0])
    y2_topleft = y_topleft + np.random.randint(
        random_perturb_range[0], random_perturb_range[1]) * np.random.choice(
            [-1.0, 1.0])

    x2_topright = x_topright + np.random.randint(
        random_perturb_range[0], random_perturb_range[1]) * np.random.choice(
            [-1.0, 1.0])
    y2_topright = y_topright + np.random.randint(
        random_perturb_range[0], random_perturb_range[1]) * np.random.choice(
            [-1.0, 1.0])

    x2_bottomleft = x_bottomleft + np.random.randint(
        random_perturb_range[0], random_perturb_range[1]) * np.random.choice(
            [-1.0, 1.0])
    y2_bottomleft = y_bottomleft + np.random.randint(
        random_perturb_range[0], random_perturb_range[1]) * np.random.choice(
            [-1.0, 1.0])

    x2_bottomright = x_bottomright + np.random.randint(
        random_perturb_range[0], random_perturb_range[1]) * np.random.choice(
            [-1.0, 1.0])
    y2_bottomright = y_bottomright + np.random.randint(
        random_perturb_range[0], random_perturb_range[1]) * np.random.choice(
            [-1.0, 1.0])

    tl2 = (x2_topleft, y2_topleft)
    tr2 = (x2_topright, y2_topright)
    br2 = (x2_bottomright, y2_bottomright)
    bl2 = (x2_bottomleft, y2_bottomleft)

    rect2 = np.array([tl2, tr2, br2, bl2], dtype=np.float32)

    # ===== homography
    H = cv2.getPerspectiveTransform(src=rect1, dst=rect2)
    H_inverse = np.linalg.inv(H)

    if img is not None:
        img_warped = cv2.warpPerspective(src=img, M=H_inverse, dsize=(w, h))
        return img_warped, H, H_inverse
    else:
        return H_inverse


class ContrasDataset(data.Dataset):
    """Dataset for the training of Contrastive Correspondence Network.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'ann_file': Use annotation file to generate paths.
        If opt['io_backend'] != lmdb and opt['ann_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The left.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
        dataroot_in (str): Data root path for input image.
        dataroot_ref (str): Data root path for ref image.
        ann_file (str): Path for annotation file.
        io_backend (dict): IO backend type and other kwarg.
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension. Default: '{}'.
        gt_size (int): Cropped patched size for gt patches.
        use_flip (bool): Use horizontal and vertical flips.
        use_rot (bool): Use rotation (use transposing h and w for
            implementation).

        scale (bool): Scale, which will be added automatically.
    """

    def __init__(self, opt):
        super(ContrasDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.in_folder, self.ref_folder = opt['dataroot_in'], opt[
            'dataroot_ref']
        if 'filename_tmpl' in opt:  # only used for folder mode
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.in_folder, self.ref_folder]
            self.io_backend_opt['client_keys'] = ['in', 'ref']
            self.paths = paired_paths_from_lmdb(
                [self.in_folder, self.ref_folder], ['in', 'ref'])
        elif 'ann_file' in self.opt:
            self.paths = paired_paths_from_ann_file(
                [self.in_folder, self.ref_folder], ['in', 'ref'],
                self.opt['ann_file'])
        else:
            self.paths = paired_paths_from_folder(
                [self.in_folder, self.ref_folder], ['in', 'ref'],
                self.filename_tmpl)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load in and ref images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1] float32.
        img_path = self.paths[index]['in_path']
        img_bytes = self.file_client.get(img_path, 'in')
        img_in = mmcv.imfrombytes(img_bytes).astype(np.float32) / 255.

        gt_h, gt_w = self.opt['gt_size'], self.opt['gt_size']
        # in case that some images may not have the same shape as gt_size
        img_in = mmcv.imresize(img_in, (gt_w, gt_h), interpolation='bicubic')

        # augmentation: flip, rotation
        img_in = augment([img_in], self.opt['use_flip'], self.opt['use_rot'])

        # image pair generation
        img_in_transformed, H, H_inverse = image_pair_generation(
            img_in, (0, 10), 160)

        grid_x, grid_y = np.meshgrid(np.arange(gt_w), np.arange(gt_h))
        grid_z = np.ones(grid_x.shape)

        coordinate = np.stack((grid_x, grid_y, grid_z), axis=0).reshape(
            (3, -1))
        transformed_coordinate = np.dot(H_inverse, coordinate)
        transformed_coordinate /= transformed_coordinate[2, :]
        # the transformed coordinates of the original image
        transformed_coordinate = transformed_coordinate.transpose(
            1, 0).reshape(gt_h, gt_w, 3)

        # downsample using PIL bicubic kernel
        lq_h, lq_w = gt_h // scale, gt_w // scale

        img_in_pil = img_in * 255
        img_in_pil = Image.fromarray(
            cv2.cvtColor(img_in_pil.astype(np.uint8), cv2.COLOR_BGR2RGB))

        img_in_transformed_pil = img_in_transformed * 255
        img_in_transformed_pil = Image.fromarray(
            cv2.cvtColor(
                img_in_transformed_pil.astype(np.uint8), cv2.COLOR_BGR2RGB))

        img_in_lq = img_in_pil.resize((lq_w, lq_h), Image.BICUBIC)
        img_in_transformed_lq = img_in_transformed_pil.resize((lq_w, lq_h),
                                                              Image.BICUBIC)

        # bicubic upsample LR
        img_in_up = img_in_lq.resize((gt_w, gt_h), Image.BICUBIC)
        img_in_transformed_up = img_in_transformed_lq.resize((gt_w, gt_h),
                                                             Image.BICUBIC)

        img_in_lq = cv2.cvtColor(np.array(img_in_lq), cv2.COLOR_RGB2BGR)
        img_in_lq = img_in_lq.astype(np.float32) / 255.
        img_in_up = cv2.cvtColor(np.array(img_in_up), cv2.COLOR_RGB2BGR)
        img_in_up = img_in_up.astype(np.float32) / 255.
        img_in_transformed_lq = cv2.cvtColor(
            np.array(img_in_transformed_lq), cv2.COLOR_RGB2BGR)
        img_in_transformed_lq = img_in_transformed_lq.astype(np.float32) / 255.
        img_in_transformed_up = cv2.cvtColor(
            np.array(img_in_transformed_up), cv2.COLOR_RGB2BGR)
        img_in_transformed_up = img_in_transformed_up.astype(np.float32) / 255.

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_in, img_in_transformed, img_in_up, img_in_transformed_up = totensor(
            [img_in, img_in_transformed, img_in_up, img_in_transformed_up],
            bgr2rgb=True,
            float32=True)

        return {
            'img_in': img_in,
            'img_in_up': img_in_up,
            'img_ref': img_in_transformed,
            'img_ref_up': img_in_transformed_up,
            'transformed_coordinate': transformed_coordinate
        }

    def __len__(self):
        return len(self.paths)


class ContrasValDataset(data.Dataset):
    """Dataset for the validation of Contrastive Correspondence Network.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'ann_file': Use annotation file to generate paths.
        If opt['io_backend'] != lmdb and opt['ann_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The left.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
        dataroot_in (str): Data root path for input image.
        dataroot_ref (str): Data root path for ref image.
        ann_file (str): Path for annotation file.
        io_backend (dict): IO backend type and other kwarg.
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension. Default: '{}'.
        gt_size (int): Cropped patched size for gt patches.
        use_flip (bool): Use horizontal and vertical flips.
        use_rot (bool): Use rotation (use transposing h and w for
            implementation).

        scale (bool): Scale, which will be added automatically.
    """

    def __init__(self, opt):
        super(ContrasValDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.in_folder, self.ref_folder = opt['dataroot_in'], opt[
            'dataroot_ref']
        if 'filename_tmpl' in opt:  # only used for folder mode
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.in_folder, self.ref_folder]
            self.io_backend_opt['client_keys'] = ['in', 'ref']
            self.paths = paired_paths_from_lmdb(
                [self.in_folder, self.ref_folder], ['in', 'ref'])
        elif 'ann_file' in self.opt:
            self.paths = paired_paths_from_ann_file(
                [self.in_folder, self.ref_folder], ['in', 'ref'],
                self.opt['ann_file'])
        else:
            self.paths = paired_paths_from_folder(
                [self.in_folder, self.ref_folder], ['in', 'ref'],
                self.filename_tmpl)

        # generation of transformation pools
        np.random.seed(0)
        self.transform_matrices = []
        for i in range(len(self.paths)):
            H_inverse = image_pair_generation(
                None, random_perturb_range=(0, 10), cropping_window_size=160)
            self.transform_matrices.append(H_inverse)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load in and ref images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1] float32.
        img_path = self.paths[index]['in_path']
        img_bytes = self.file_client.get(img_path, 'in')
        img_in = mmcv.imfrombytes(img_bytes).astype(np.float32) / 255.
        img_in = mod_crop(img_in, scale)
        gt_h, gt_w, _ = img_in.shape

        H_inverse = self.transform_matrices[index]
        img_in_transformed = cv2.warpPerspective(
            src=img_in, M=H_inverse, dsize=(gt_w, gt_h))

        grid_x, grid_y = np.meshgrid(np.arange(gt_w), np.arange(gt_h))
        grid_z = np.ones(grid_x.shape)

        coordinate = np.stack((grid_x, grid_y, grid_z), axis=0).reshape(
            (3, -1))
        transformed_coordinate = np.dot(H_inverse, coordinate)
        transformed_coordinate /= transformed_coordinate[2, :]
        # the transformed coordinates of the original image
        transformed_coordinate = transformed_coordinate.transpose(
            1, 0).reshape(gt_h, gt_w, 3)

        # downsample using PIL bicubic kernel
        lq_h, lq_w = gt_h // scale, gt_w // scale

        img_in_pil = img_in * 255
        img_in_pil = Image.fromarray(
            cv2.cvtColor(img_in_pil.astype(np.uint8), cv2.COLOR_BGR2RGB))

        img_in_transformed_pil = img_in_transformed * 255
        img_in_transformed_pil = Image.fromarray(
            cv2.cvtColor(
                img_in_transformed_pil.astype(np.uint8), cv2.COLOR_BGR2RGB))

        img_in_lq = img_in_pil.resize((lq_w, lq_h), Image.BICUBIC)
        img_in_transformed_lq = img_in_transformed_pil.resize((lq_w, lq_h),
                                                              Image.BICUBIC)

        # bicubic upsample lr
        img_in_up = img_in_lq.resize((gt_w, gt_h), Image.BICUBIC)
        img_in_transformed_up = img_in_transformed_lq.resize((gt_w, gt_h),
                                                             Image.BICUBIC)

        img_in_lq = cv2.cvtColor(np.array(img_in_lq), cv2.COLOR_RGB2BGR)
        img_in_lq = img_in_lq.astype(np.float32) / 255.
        img_in_up = cv2.cvtColor(np.array(img_in_up), cv2.COLOR_RGB2BGR)
        img_in_up = img_in_up.astype(np.float32) / 255.
        img_in_transformed_lq = cv2.cvtColor(
            np.array(img_in_transformed_lq), cv2.COLOR_RGB2BGR)
        img_in_transformed_lq = img_in_transformed_lq.astype(np.float32) / 255.
        img_in_transformed_up = cv2.cvtColor(
            np.array(img_in_transformed_up), cv2.COLOR_RGB2BGR)
        img_in_transformed_up = img_in_transformed_up.astype(np.float32) / 255.

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_in, img_in_transformed, img_in_up, img_in_transformed_up = totensor(
            [img_in, img_in_transformed, img_in_up, img_in_transformed_up],
            bgr2rgb=True,
            float32=True)

        return {
            'img_in': img_in,
            'img_in_up': img_in_up,
            'img_ref': img_in_transformed,
            'img_ref_up': img_in_transformed_up,
            'transformed_coordinate': transformed_coordinate,
            'name': img_path
        }

    def __len__(self):
        return len(self.paths)
