import math
import os.path as osp

import cv2
import mmcv
import numpy as np
import torch

from mmsr.data.transforms import totensor


def read_img_seq(path):
    """Read a sequence of images from a given folder path.

    Args:
        path (list[str] | str): List of image paths or image folder path.

    Returns:
        Tensor: size (T, C, H, W), RGB, [0, 1]
    """
    if isinstance(path, list):
        img_path_l = path
    else:
        img_path_l = sorted([osp.join(path, v) for v in mmcv.scandir(path)])
    img_l = [mmcv.imread(v).astype(np.float32) / 255. for v in img_path_l]
    img_l = totensor(img_l)
    imgs = torch.stack(img_l, dim=0)
    return imgs


def index_generation(crt_idx, max_n, num_frames, padding='reflection'):
    """Generate an index list for reading num_frames frames from a sequence
    of images.

    Args:
        crt_idx (int): Current center index.
        max_n (int): Max number of the sequence of images (from 1).
        num_frames (int): Reading num_frames frames.
        padding (str): padding mode, one of
            'replicate' | 'reflection' | 'reflection_circle' | 'circle'
            Example: crt_idx = 0, num_frames = 5
            replicate: [0, 0, 0, 1, 2]
            reflection: [2, 1, 0, 1, 2]
            reflection_circle: [4, 3, 0, 1, 2]
            circle: [3, 4, 0, 1, 2]

    Returns:
        list [int]: a list of indexes
    """
    assert num_frames % 2 == 1, 'num_frames should be an odd number.'
    assert padding in [
        'replicate', 'reflection', 'reflection_circle', 'circle'
    ], 'Wrong padding mode.'

    max_n = max_n - 1  # start from 0
    n_pad = num_frames // 2
    return_l = []

    for i in range(crt_idx - n_pad, crt_idx + n_pad + 1):
        if i < 0:
            if padding == 'replicate':
                add_idx = 0
            elif padding == 'reflection':
                add_idx = -i
            elif padding == 'reflection_circle':
                add_idx = crt_idx + n_pad - i
            else:
                add_idx = num_frames + i
        elif i > max_n:
            if padding == 'replicate':
                add_idx = max_n
            elif padding == 'reflection':
                add_idx = max_n * 2 - i
            elif padding == 'reflection_circle':
                add_idx = (crt_idx - n_pad) - (i - max_n)
            else:
                add_idx = i - num_frames
        else:
            add_idx = i
        return_l.append(add_idx)
    return return_l


def paired_paths_from_lmdb(folders, keys):
    """Generate paired paths from lmdb files.

    Contents of lmdb. Taking the `lq.lmdb` for example, the file structure is:

    lq.lmdb
    ├── data.mdb
    ├── lock.mdb
    ├── meta_info.txt

    The data.mdb and lock.mdb are standard lmdb files and you can refer to
    https://lmdb.readthedocs.io/en/release/ for more details.

    The meta_info.txt is a specified txt file to record the meta information
    of our datasets. It will be automatically created when preparing
    datasets by our provided dataset tools.
    Each line in the txt file records
    1)image name (with extension),
    2)image shape,
    3)compression level, separated by a white space.
    Example: `baboon.png (120,125,3) 1`

    We use the image name without extension as the lmdb key.
    Note that we use the same key for the corresponding lq and gt images.

    Args:
        folders (list): A list of folder path. The order of list should be:
            [input_folder, ref_folder] or [lq_folder, gt_folder].
        keys (list): A list of keys identifying folders. The order should be in
            consistent with folders. Here are examples: ['input', 'ref'] or
            ['lq', 'gt']. Note that this key is different from lmdb keys.

    Returns:
        list: Returned path list.
    """
    paths = []
    input_folder = folders[0]  # e.g., input, lq
    ref_folder = folders[1]  # e.g., ref, gt
    input_key = keys[0]
    ref_key = keys[1]

    if not (input_folder.endswith('.lmdb') and ref_folder.endswith('.lmdb')):
        raise ValueError(
            f'{input_key} folder and {ref_key} folder should both in lmdb '
            f'format. But received {input_key}: {input_folder}; '
            f'{ref_key}: {ref_folder}')
    # ensure that the two meta_info files are the same
    input_lmdb_keys = []
    with open(osp.join(input_folder, 'meta_info.txt')) as fin:
        for line in fin:
            lmdb_key = line.split(' ')[0].split('.')[0]
            input_lmdb_keys.append(lmdb_key)
    ref_lmdb_keys = []
    with open(osp.join(ref_folder, 'meta_info.txt')) as fin:
        for line in fin:
            lmdb_key = line.split(' ')[0].split('.')[0]
            ref_lmdb_keys.append(lmdb_key)

    if set(input_lmdb_keys) != set(ref_lmdb_keys):
        raise ValueError(
            f'Keys in {input_key}_folder and {ref_key}_folder are different.')
    else:
        for lmdb_key in input_lmdb_keys:
            paths.append(
                dict([(f'{input_key}_path', lmdb_key),
                      (f'{ref_key}_path', lmdb_key)]))
        return paths


def paired_paths_from_ann_file(folders, keys, ann_file):
    """Generate paired paths from an anno file.

    Annotation file is a txt file listing all paths of pairs.
    Each line contains the relative lq and gt paths (or input and ref paths),
    separated by a white space.

    Example of an annotation file:
    ```
    lq/0001_x4.png gt/0001.png
    lq/0002_x4.png gt/0002.png
    ```

    Args:
        folders (list): A list of folder root path. The order of list should be:
            [input_folder, ref_folder] or [lq_folder, gt_folder]
        keys (list): A list of keys identifying folders. The order should be in
            consistent with folders. Here are examples: ['input', 'ref'] or
            ['lq', 'gt']
        ann_file (str): Path for annotation file.

    Returns:
        list: Returned path list.
    """
    paths = []
    input_folder = folders[0]  # e.g., input, lq
    ref_folder = folders[1]  # e.g., ref, gt
    input_key = keys[0]
    ref_key = keys[1]

    with open(ann_file, 'r') as fin:
        for line in fin:
            input_path, ref_path = line.strip().split(' ')
            input_path = osp.join(input_folder, input_path)
            ref_path = osp.join(ref_folder, ref_path)
            paths.append(
                dict([(f'{input_key}_path', input_path),
                      (f'{ref_key}_path', ref_path)]))
    return paths


def paired_paths_from_folder(folders, keys, filename_tmpl):
    """Generate paired paths from folders.

    Args:
        folders (list): A list of folder path. The order of list should be:
            [input_folder, ref_folder] or [lq_folder, gt_folder].
        keys (list): a list of keys identifying folders. The order should be in
            consistent with folders. Here are examples: ['input', 'ref'] or
            ['lq', 'gt'].
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension. Usually the filename_tmpl is
            for files in input folder.

    Returns:
        list: Returned path list.
    """
    paths = []
    input_folder = folders[0]
    ref_folder = folders[1]
    input_key = keys[0]
    ref_key = keys[1]

    input_paths = list(mmcv.scandir(input_folder))
    ref_paths = list(mmcv.scandir(ref_folder))
    assert len(input_paths) == len(ref_paths), (
        f'{input_key} and {ref_key} datasets have different number of images: '
        f'{len(input_paths)}, {len(ref_paths)}.')
    for ref_path in ref_paths:
        basename, ext = osp.splitext(osp.basename(ref_path))
        input_path_base = f'{filename_tmpl.format(basename)}{ext}'
        input_path = osp.join(input_folder, input_path_base)
        assert input_path_base in input_paths, (f'{input_path_base} is not in '
                                                f'{input_key}_paths.')
        ref_path = osp.join(ref_folder, ref_path)
        paths.append(
            dict([(f'{input_key}_path', input_path),
                  (f'{ref_key}_path', ref_path)]))
    return paths


# TODO
def channel_convert(in_c, tar_type, img_list):
    """conversion among BGR, gray and y"""
    if in_c == 3 and tar_type == 'gray':  # BGR to gray
        gray_list = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in img_list]
        return [np.expand_dims(img, axis=2) for img in gray_list]
    elif in_c == 3 and tar_type == 'y':  # BGR to y
        y_list = [bgr2ycbcr(img, only_y=True) for img in img_list]
        return [np.expand_dims(img, axis=2) for img in y_list]
    elif in_c == 1 and tar_type == 'RGB':  # gray/y to BGR
        return [cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) for img in img_list]
    else:
        return img_list


# TODO
def rgb2ycbcr(img, only_y=True):
    """same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    """
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img,
                        [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                         [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


# TODO
def bgr2ycbcr(img, only_y=True):
    """bgr version of rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    """
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img,
                        [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                         [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


# TODO
def ycbcr2rgb(img):
    """same as matlab ycbcr2rgb
    Input:
        uint8, [0, 255]
        float, [0, 1]
    """
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    rlt = np.matmul(img, [[0.00456621, 0.00456621, 0.00456621],
                          [0, -0.00153632, 0.00791071],
                          [0.00625893, -0.00318811, 0]]) * 255.0 + [
                              -222.921, 135.576, -276.836
                          ]  # noqa:E126
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


# TODO
#####
# Functions
#####


# matlab 'imresize' function, now only support 'bicubic'
def cubic(x):
    absx = torch.abs(x)
    absx2 = absx**2
    absx3 = absx**3
    return (1.5 * absx3 - 2.5 * absx2 + 1) * (
        (absx <= 1).type_as(absx)) + (-0.5 * absx3 + 2.5 * absx2 - 4 * absx +
                                      2) * (((absx > 1) *
                                             (absx <= 2)).type_as(absx))


def calculate_weights_indices(in_length, out_length, scale, kernel,
                              kernel_width, antialiasing):
    if (scale < 1) and (antialiasing):
        # Use a modified kernel to simultaneously interpolate and antialias-
        # larger kernel width
        kernel_width = kernel_width / scale

    # Output-space coordinates
    x = torch.linspace(1, out_length, out_length)

    # Input-space coordinates. Calculate the inverse mapping such that 0.5
    # in output space maps to 0.5 in input space, and 0.5+scale in output
    # space maps to 1.5 in input space.
    u = x / scale + 0.5 * (1 - 1 / scale)

    # What is the left-most pixel that can be involved in the computation?
    left = torch.floor(u - kernel_width / 2)

    # What is the maximum number of pixels that can be involved in the
    # computation?  Note: it's OK to use an extra pixel here; if the
    # corresponding weights are all zero, it will be eliminated at the end
    # of this function.
    P = math.ceil(kernel_width) + 2

    # The indices of the input pixels involved in computing the k-th output
    # pixel are in row k of the indices matrix.
    indices = left.view(out_length, 1).expand(out_length, P) + torch.linspace(
        0, P - 1, P).view(1, P).expand(out_length, P)

    # The weights used to compute the k-th output pixel are in row k of the
    # weights matrix.
    distance_to_center = u.view(out_length, 1).expand(out_length, P) - indices
    # apply cubic kernel
    if (scale < 1) and (antialiasing):
        weights = scale * cubic(distance_to_center * scale)
    else:
        weights = cubic(distance_to_center)
    # Normalize the weights matrix so that each row sums to 1.
    weights_sum = torch.sum(weights, 1).view(out_length, 1)
    weights = weights / weights_sum.expand(out_length, P)

    # If a column in weights is all zero, get rid of it. only consider the
    # first and last column.
    weights_zero_tmp = torch.sum((weights == 0), 0)
    if not math.isclose(weights_zero_tmp[0], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 1, P - 2)
        weights = weights.narrow(1, 1, P - 2)
    if not math.isclose(weights_zero_tmp[-1], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 0, P - 2)
        weights = weights.narrow(1, 0, P - 2)
    weights = weights.contiguous()
    indices = indices.contiguous()
    sym_len_s = -indices.min() + 1
    sym_len_e = indices.max() - in_length
    indices = indices + sym_len_s - 1
    return weights, indices, int(sym_len_s), int(sym_len_e)


def imresize(img, scale, antialiasing=True):
    # Now the scale should be the same for H and W
    # input: img: CHW RGB [0,1]
    # output: CHW RGB [0,1] w/o round

    in_C, in_H, in_W = img.size()
    _, out_H, out_W = in_C, math.ceil(in_H * scale), math.ceil(in_W * scale)
    kernel_width = 4
    kernel = 'cubic'

    # Return the desired dimension order for performing the resize.  The
    # strategy is to perform the resize first along the dimension with the
    # smallest scale factor.
    # Now we do not support this.

    # get weights and indices
    weights_H, indices_H, sym_len_Hs, sym_len_He = calculate_weights_indices(
        in_H, out_H, scale, kernel, kernel_width, antialiasing)
    weights_W, indices_W, sym_len_Ws, sym_len_We = calculate_weights_indices(
        in_W, out_W, scale, kernel, kernel_width, antialiasing)
    # process H dimension
    # symmetric copying
    img_aug = torch.FloatTensor(in_C, in_H + sym_len_Hs + sym_len_He, in_W)
    img_aug.narrow(1, sym_len_Hs, in_H).copy_(img)

    sym_patch = img[:, :sym_len_Hs, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, 0, sym_len_Hs).copy_(sym_patch_inv)

    sym_patch = img[:, -sym_len_He:, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, sym_len_Hs + in_H, sym_len_He).copy_(sym_patch_inv)

    out_1 = torch.FloatTensor(in_C, out_H, in_W)
    kernel_width = weights_H.size(1)
    for i in range(out_H):
        idx = int(indices_H[i][0])
        out_1[0, i, :] = img_aug[0, idx:idx + kernel_width, :].transpose(
            0, 1).mv(weights_H[i])
        out_1[1, i, :] = img_aug[1, idx:idx + kernel_width, :].transpose(
            0, 1).mv(weights_H[i])
        out_1[2, i, :] = img_aug[2, idx:idx + kernel_width, :].transpose(
            0, 1).mv(weights_H[i])

    # process W dimension
    # symmetric copying
    out_1_aug = torch.FloatTensor(in_C, out_H, in_W + sym_len_Ws + sym_len_We)
    out_1_aug.narrow(2, sym_len_Ws, in_W).copy_(out_1)

    sym_patch = out_1[:, :, :sym_len_Ws]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, 0, sym_len_Ws).copy_(sym_patch_inv)

    sym_patch = out_1[:, :, -sym_len_We:]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, sym_len_Ws + in_W, sym_len_We).copy_(sym_patch_inv)

    out_2 = torch.FloatTensor(in_C, out_H, out_W)
    kernel_width = weights_W.size(1)
    for i in range(out_W):
        idx = int(indices_W[i][0])
        out_2[0, :, i] = out_1_aug[0, :,
                                   idx:idx + kernel_width].mv(weights_W[i])
        out_2[1, :, i] = out_1_aug[1, :,
                                   idx:idx + kernel_width].mv(weights_W[i])
        out_2[2, :, i] = out_1_aug[2, :,
                                   idx:idx + kernel_width].mv(weights_W[i])

    return out_2


def imresize_np(img, scale, antialiasing=True):
    # Now the scale should be the same for H and W
    # input: img: Numpy, HWC BGR [0,1]
    # output: HWC BGR [0,1] w/o round
    img = torch.from_numpy(img)

    in_H, in_W, in_C = img.size()
    _, out_H, out_W = in_C, math.ceil(in_H * scale), math.ceil(in_W * scale)
    kernel_width = 4
    kernel = 'cubic'

    # Return the desired dimension order for performing the resize.  The
    # strategy is to perform the resize first along the dimension with the
    # smallest scale factor.
    # Now we do not support this.

    # get weights and indices
    weights_H, indices_H, sym_len_Hs, sym_len_He = calculate_weights_indices(
        in_H, out_H, scale, kernel, kernel_width, antialiasing)
    weights_W, indices_W, sym_len_Ws, sym_len_We = calculate_weights_indices(
        in_W, out_W, scale, kernel, kernel_width, antialiasing)
    # process H dimension
    # symmetric copying
    img_aug = torch.FloatTensor(in_H + sym_len_Hs + sym_len_He, in_W, in_C)
    img_aug.narrow(0, sym_len_Hs, in_H).copy_(img)

    sym_patch = img[:sym_len_Hs, :, :]
    inv_idx = torch.arange(sym_patch.size(0) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(0, inv_idx)
    img_aug.narrow(0, 0, sym_len_Hs).copy_(sym_patch_inv)

    sym_patch = img[-sym_len_He:, :, :]
    inv_idx = torch.arange(sym_patch.size(0) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(0, inv_idx)
    img_aug.narrow(0, sym_len_Hs + in_H, sym_len_He).copy_(sym_patch_inv)

    out_1 = torch.FloatTensor(out_H, in_W, in_C)
    kernel_width = weights_H.size(1)
    for i in range(out_H):
        idx = int(indices_H[i][0])
        out_1[i, :, 0] = img_aug[idx:idx + kernel_width, :,
                                 0].transpose(0, 1).mv(weights_H[i])
        out_1[i, :, 1] = img_aug[idx:idx + kernel_width, :,
                                 1].transpose(0, 1).mv(weights_H[i])
        out_1[i, :, 2] = img_aug[idx:idx + kernel_width, :,
                                 2].transpose(0, 1).mv(weights_H[i])

    # process W dimension
    # symmetric copying
    out_1_aug = torch.FloatTensor(out_H, in_W + sym_len_Ws + sym_len_We, in_C)
    out_1_aug.narrow(1, sym_len_Ws, in_W).copy_(out_1)

    sym_patch = out_1[:, :sym_len_Ws, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    out_1_aug.narrow(1, 0, sym_len_Ws).copy_(sym_patch_inv)

    sym_patch = out_1[:, -sym_len_We:, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    out_1_aug.narrow(1, sym_len_Ws + in_W, sym_len_We).copy_(sym_patch_inv)

    out_2 = torch.FloatTensor(out_H, out_W, in_C)
    kernel_width = weights_W.size(1)
    for i in range(out_W):
        idx = int(indices_W[i][0])
        out_2[:, i, 0] = out_1_aug[:, idx:idx + kernel_width,
                                   0].mv(weights_W[i])
        out_2[:, i, 1] = out_1_aug[:, idx:idx + kernel_width,
                                   1].mv(weights_W[i])
        out_2[:, i, 2] = out_1_aug[:, idx:idx + kernel_width,
                                   2].mv(weights_W[i])

    return out_2.numpy()
