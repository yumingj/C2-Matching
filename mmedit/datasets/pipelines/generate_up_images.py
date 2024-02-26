import copy
import cv2
import numpy as np
from PIL import Image

from ..registry import PIPELINES


@PIPELINES.register_module()
class GenerateUpImages:
    """Generate upsampled images.
    """

    def __init__(self, redownsample=False):
        self.redownsample = redownsample

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """

        up_images = []
        if self.redownsample:
            gt_images = copy.deepcopy(results['gt'])
            for gt_image in gt_images:
                gt_h, gt_w, _ = gt_image.shape
                lq_h, lq_w = gt_h // 4, gt_w // 4
                gt_image = gt_image * 255
                gt_image = Image.fromarray(gt_image.astype(np.uint8))

                lq_image = gt_image.resize((lq_w, lq_h), Image.BICUBIC)

                up_image = lq_image.resize((gt_w, gt_h), Image.BICUBIC)
                up_image = np.array(up_image).astype(np.float32) / 255.
                up_images.append(up_image)
        else:
            lq_images = copy.deepcopy(results['lq'])
            for lq_image in lq_images:
                lq_h, lq_w, _ = lq_image.shape
                gt_h, gt_w = lq_h * 4, lq_w * 4

                lq_image = lq_image * 255
                lq_image = Image.fromarray(lq_image.astype(np.uint8))

                up_image = lq_image.resize((gt_w, gt_h), Image.BICUBIC)
                up_image = np.array(up_image).astype(np.float32) / 255.
                up_images.append(up_image)

        results['up'] = up_images

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(redownsample='{self.redownsample}')"
        return repr_str


@PIPELINES.register_module()
class GenerateKeyframeUp:
    """Generate upsampled images.
    """

    def __init__(self, redownsample=False):
        self.redownsample = redownsample

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """

        if self.redownsample:
            keyframe_images = copy.deepcopy(results['keyframe'])
            gt_h, gt_w, _ = keyframe_images.shape
            lq_h, lq_w = gt_h // 4, gt_w // 4
            keyframe_images = keyframe_images * 255
            keyframe_images = Image.fromarray(keyframe_images.astype(np.uint8))

            lq_image = keyframe_images.resize((lq_w, lq_h), Image.BICUBIC)

            up_image = lq_image.resize((gt_w, gt_h), Image.BICUBIC)
            up_image = np.array(up_image).astype(np.float32) / 255.
        else:
            raise NotImplementedError

        results['keyframe_up'] = up_image

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(redownsample='{self.redownsample}')"
        return repr_str


@PIPELINES.register_module()
class GenerateKeyframeDown:
    """Generate upsampled images.
    """

    def __init__(self, redownsample=False):
        self.redownsample = redownsample

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """

        if self.redownsample:
            keyframe_images = copy.deepcopy(results['keyframe'])
            gt_h, gt_w, _ = keyframe_images.shape
            lq_h, lq_w = gt_h // 4, gt_w // 4
            keyframe_images = keyframe_images * 255
            keyframe_images = Image.fromarray(keyframe_images.astype(np.uint8))

            lq_image = keyframe_images.resize((lq_w, lq_h), Image.BICUBIC)

            lq_image = np.array(lq_image).astype(np.float32) / 255.
        else:
            raise NotImplementedError

        results['keyframe_down'] = lq_image

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(redownsample='{self.redownsample}')"
        return repr_str
