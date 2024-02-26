# C2-Matching-Video (CVPR2021, TPAMI 2022)

![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg?style=plastic)
![pytorch 1.4.0](https://img.shields.io/badge/pytorch-1.4.0-green.svg?style=plastic)

This repository contains the implementation of the following paper:
> **Reference-based Image and Video Super-Resolution via C2-Matching**<br>
> Yuming Jiang, Kelvin C.K. Chan, Xintao Wang, Chen Change Loy, Ziwei Liu<br>
> IEEE Transactions on Pattern Analysis and Machine Intelligence (**TPAMI**), 2022<br>

[[Paper](https://arxiv.org/abs/2212.09581)]
[[Project Page](https://yumingj.github.io/projects/C2_matching)]

## Dependencies and Installation

1. Clone Repo

   ```bash
   git clone git@github.com:yumingj/C2-Matching.git
   ```

1. Create Conda Environment

   ```bash
   conda create --name c2_matching_video python=3.7
   conda activate c2_matching_video
   ```

1. Install Dependencies

   ```bash
   conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
   pip install -U openmim
   mim install mmcv-full==1.3.5
   pip install -r requirements.txt
   pip install -v -e .

   ```


## Get Started

### Pretrained Models
Downloading the pretrained models from this [link](https://drive.google.com/drive/folders/1qGyaJ61OkH5dCgcXZgx34q4j-HP-fzgs?usp=sharing) and put them under `experiments/pretrained_models folder`.

### Dataset

For REDS dataset, please refer to this [link](https://seungjunnah.github.io/Datasets/reds.html). The reference images are taken from the first frame of REDS 120fps. For your convience, we put the references of REDS4 dataset in this [link](https://drive.google.com/file/d/1Fc6TDBsUQ0pVVo72sInr08IjFGRAFA1f/view?usp=sharing).

For Vid4 dataset, please refer to this [link](https://drive.google.com/drive/folders/1An6hF1oYkeWxfOBxxKm073mvgIFrBNDA).

### Test

We provide quick test code with the pretrained model.


```bash
python tools/test.py configs/restorers/refvsr/test_reds4.py pretrained_models/c2_matching_video.pth --save-path work_dirs/reds4_results\

python tools/test.py configs/restorers/refvsr/test_vid4.py pretrained_models/c2_matching_video.pth --save-path work_dirs/vid4_results
```




## Citation

   If you find our repo useful for your research, please consider citing our paper:

   ```bibtex
   @article{jiang2022reference,
      title={Reference-based Image and Video Super-Resolution via $ C\^{}$\{$2$\}$ $-Matching},
      author={Jiang, Yuming and Chan, Kelvin CK and Wang, Xintao and Loy, Chen Change and Liu, Ziwei},
      journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
      year={2022},
      publisher={IEEE}
   }
   ```


## License and Acknowledgement

This project is open sourced under MIT license. The code framework is mainly modified from [BasicSR](https://github.com/xinntao/BasicSR) and [MMSR](https://github.com/open-mmlab/mmediting) (Now reorganized as MMEditing). Please refer to the original repo for more usage and documents.


## Contact

If you have any question, please feel free to contact us via `yuming002@e.ntu.edu.sg`.
