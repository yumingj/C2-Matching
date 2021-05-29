import os.path as osp
import sys

import cv2
import lmdb
import mmcv
import numpy as np
from PIL import Image

from mmsr.utils import ProgressBar


def create_lmdb_for_cufed():
    """Create lmdb files for DIV2K dataset.

    Usage:
        Before run this script, please run `extract_subimages.py`.
        Typically, there are four folders to be processed for DIV2K dataset.
            DIV2K_train_HR_sub
            DIV2K_train_LR_bicubic/X2_sub
            DIV2K_train_LR_bicubic/X3_sub
            DIV2K_train_LR_bicubic/X4_sub
        Remember to modify opt configurations according to your settings.
    """

    # for input image
    folder_path = '/home/ymjiang/dataset/CUFED/input'
    lmdb_paths = [
        '/home/ymjiang/dataset/CUFED/CUFED_input.lmdb',
        '/home/ymjiang/dataset/CUFED/CUFED_input_lq.lmdb',
        '/home/ymjiang/dataset/CUFED/CUFED_input_up.lmdb'
    ]
    img_path_list, keys = prepare_keys_cufed(folder_path)
    make_lmdb_cufed(folder_path, lmdb_paths, img_path_list, keys, is_ref=False)

    # for ref image
    folder_path = '/home/ymjiang/dataset/CUFED/ref'
    lmdb_paths = [
        '/home/ymjiang/dataset/CUFED/CUFED_ref.lmdb',
        '/home/ymjiang/dataset/CUFED/CUFED_ref_lq.lmdb',
        '/home/ymjiang/dataset/CUFED/CUFED_ref_up.lmdb'
    ]
    img_path_list, keys = prepare_keys_cufed(folder_path)
    make_lmdb_cufed(folder_path, lmdb_paths, img_path_list, keys, is_ref=True)


def prepare_keys_cufed(folder_path):
    """Prepare image path list and keys for DIV2K dataset.

    Args:
        folder_path (str): Folder path.

    Returns:
        list[str]: Image path list.
        list[str]: Key list.
    """
    print('Reading image path list ...')
    img_path_list = sorted(
        list(mmcv.scandir(folder_path, suffix='png', recursive=False)))
    keys = [img_path.split('.png')[0] for img_path in sorted(img_path_list)]

    return img_path_list, keys


def generate_lq_and_ref(img):
    gt_w = 160
    gt_h = 160

    lq_w = 40
    lq_h = 40
    # downsample the image
    img = Image.fromarray(
        cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB))
    img_lq = img.resize((lq_w, lq_h), Image.BICUBIC)
    img_up = img_lq.resize((gt_w, gt_h), Image.BICUBIC)

    img_lq = cv2.cvtColor(np.array(img_lq), cv2.COLOR_RGB2BGR)
    img_up = cv2.cvtColor(np.array(img_up), cv2.COLOR_RGB2BGR)

    return img_lq, img_up


def make_lmdb_cufed(data_path,
                    lmdb_paths,
                    img_path_list,
                    keys,
                    is_ref=False,
                    batch=5000,
                    compress_level=1,
                    multiprocessing_read=False,
                    n_thread=40):
    """Make lmdb.

    Contents of lmdb. The file structure is:
    example.lmdb
    ├── data.mdb
    ├── lock.mdb
    ├── meta_info.txt

    The data.mdb and lock.mdb are standard lmdb files and you can refer to
    https://lmdb.readthedocs.io/en/release/ for more details.

    The meta_info.txt is a specified txt file to record the meta information
    of our datasets. It will be automatically created when preparing
    datasets by our provided dataset tools.
    Each line in the txt file records 1)image name (with extension),
    2)image shape, and 3)compression level, separated by a white space.

    For example, the meta information could be:
    `000_00000000.png (720,1280,3) 1`, which means:
    1) image name (with extension): 000_00000000.png;
    2) image shape: (720,1280,3);
    3) compression level: 1

    We use the image name without extension as the lmdb key.

    If `multiprocessing_read` is True, it will read all the images to memory
    using multiprocessing. Thus, your server needs to have enough memory.

    Args:
        data_path (str): Data path for reading images.
        lmdb_path (str): Lmdb save path.
        img_path_list (str): Image path list.
        keys (str): Used for lmdb keys.
        batch (int): After processing batch images, lmdb commits.
            Default: 5000.
        compress_level (int): Compress level when encoding images. Default: 1.
        multiprocessing_read (bool): Whether use multiprocessing to read all
            the images to memory. Default: False.
        n_thread (int): For multiprocessing.
    """
    assert len(img_path_list) == len(keys), (
        'img_path_list and keys should have the same length, '
        f'but got {len(img_path_list)} and {len(keys)}')
    for lmdb_path in lmdb_paths:
        print(f'Create lmdb for {data_path}, save to {lmdb_path}...')
    print(f'Totoal images: {len(img_path_list)}')

    for lmdb_path in lmdb_paths:
        if not lmdb_path.endswith('.lmdb'):
            raise ValueError("lmdb_path must end with '.lmdb'.")
        if osp.exists(lmdb_path):
            print(f'Folder {lmdb_path} already exists. Exit.')
            sys.exit(1)

    # create lmdb environment
    # obtain data size for one image
    img = mmcv.imread(osp.join(data_path, img_path_list[0]), flag='unchanged')

    # resize the image
    if is_ref:
        img = Image.fromarray(
            cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB))
        img = img.resize((160, 160), Image.BICUBIC)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    _, img_byte = cv2.imencode('.png', img,
                               [cv2.IMWRITE_PNG_COMPRESSION, compress_level])

    data_size_per_img = img_byte.nbytes
    print('Data size per image is: ', data_size_per_img)
    data_size = data_size_per_img * len(img_path_list)
    env_img = lmdb.open(lmdb_paths[0], map_size=data_size * 10)

    img_lq, img_up = generate_lq_and_ref(img)
    _, img_lq_byte = cv2.imencode(
        '.png', img_lq, [cv2.IMWRITE_PNG_COMPRESSION, compress_level])
    data_size_per_img_lq = img_lq_byte.nbytes
    print('Data size per image_lq is: ', data_size_per_img_lq)
    data_lq_size = data_size_per_img_lq * len(img_path_list)
    env_lq = lmdb.open(lmdb_paths[1], map_size=data_lq_size * 10)

    _, img_up_byte = cv2.imencode(
        '.png', img_up, [cv2.IMWRITE_PNG_COMPRESSION, compress_level])
    data_size_per_img_up = img_up_byte.nbytes
    print('Data size per image_up is: ', data_size_per_img_up)
    data_up_size = data_size_per_img_up * len(img_path_list)
    env_up = lmdb.open(lmdb_paths[2], map_size=data_up_size * 10)

    # write data to lmdb
    pbar = ProgressBar(len(img_path_list))
    txn_img = env_img.begin(write=True)
    txt_img_file = open(osp.join(lmdb_paths[0], 'meta_info.txt'), 'w')
    txn_lq = env_lq.begin(write=True)
    txt_lq_file = open(osp.join(lmdb_paths[1], 'meta_info.txt'), 'w')
    txn_up = env_up.begin(write=True)
    txt_up_file = open(osp.join(lmdb_paths[2], 'meta_info.txt'), 'w')
    for idx, (path, key) in enumerate(zip(img_path_list, keys)):
        pbar.update(f'Write {key}')
        key_byte = key.encode('ascii')

        _, img_byte, img_lq_byte, img_up_byte = read_cufed_img_worker(
            osp.join(data_path, path), key, compress_level, is_ref)

        txn_img.put(key_byte, img_byte)
        # write meta information
        txt_img_file.write(f'{key}.png (160, 160, 3) {compress_level}\n')

        txn_lq.put(key_byte, img_lq_byte)
        # write meta information
        txt_lq_file.write(f'{key}.png (40, 40, 3) {compress_level}\n')

        txn_up.put(key_byte, img_up_byte)
        # write meta information
        txt_up_file.write(f'{key}.png (160, 160, 3) {compress_level}\n')
        if idx % batch == 0:
            txn_img.commit()
            txn_img = env_img.begin(write=True)

            txn_lq.commit()
            txn_lq = env_lq.begin(write=True)

            txn_up.commit()
            txn_up = env_up.begin(write=True)
    txn_img.commit()
    txn_lq.commit()
    txn_up.commit()

    env_img.close()
    env_lq.close()
    env_up.close()

    txt_img_file.close()
    txt_lq_file.close()
    txt_up_file.close()
    print('\nFinish writing lmdb.')


def read_cufed_img_worker(path, key, compress_level, is_ref):
    img = mmcv.imread(path, flag='unchanged')
    # resize the image
    if is_ref:
        img = Image.fromarray(
            cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB))
        img = img.resize((160, 160), Image.BICUBIC)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    _, img_byte = cv2.imencode('.png', img,
                               [cv2.IMWRITE_PNG_COMPRESSION, compress_level])
    img_lq, img_up = generate_lq_and_ref(img)
    _, img_lq_byte = cv2.imencode(
        '.png', img_lq, [cv2.IMWRITE_PNG_COMPRESSION, compress_level])
    _, img_up_byte = cv2.imencode(
        '.png', img_up, [cv2.IMWRITE_PNG_COMPRESSION, compress_level])
    return (key, img_byte, img_lq_byte, img_up_byte)


def read_img_worker(path, key, compress_level):
    """Read image worker

    Args:
        path (str): Image path.
        key (str): Image key.
        compress_level (int): Compress level when encoding images.

    Returns:
        str: Image key.
        byte: Image byte.
        tuple[int]: Image shape.
    """
    img = mmcv.imread(path, flag='unchanged')
    if img.ndim == 2:
        h, w = img.shape
        c = 1
    else:
        h, w, c = img.shape
    _, img_byte = cv2.imencode('.png', img,
                               [cv2.IMWRITE_PNG_COMPRESSION, compress_level])
    return (key, img_byte, (h, w, c))


if __name__ == '__main__':
    create_lmdb_for_cufed()
    # create_lmdb_for_reds()
    # create_lmdb_for_vimeo90k()
