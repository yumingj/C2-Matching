import copy
import numpy as np

from ..registry import PIPELINES


@PIPELINES.register_module()
class GetKeyframe:
    """Generate Keyframe
    """

    def __init__(self, randomize):
        self.randomize = randomize

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """

        # get the index of keyframe
        num_frames = len(results['gt_path'])
        if self.randomize:
            index_keyframe = np.random.random_integers(0, num_frames - 1)
        else:
            index_keyframe = num_frames // 2

        results['index_keyframe'] = index_keyframe
        results['keyframe'] = copy.deepcopy(results['gt'][index_keyframe])

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + f"(randomize='{self.randomize}')"
        return repr_str
