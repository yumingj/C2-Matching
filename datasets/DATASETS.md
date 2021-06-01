## Prepare Test Set

### Prepare CUFED5 dataset

1. Download the CUFED5 dataset through this [link](https://drive.google.com/file/d/1Fa1mopExA9YGG1RxrCZZn7QFTYXLx6ph/view).

2. Put the CUFED5 dataset into `datasets` folder. The folder structure should follow the format:
```
- CUFED5
|--- 000_0.png
|--- 000_1.png
|--- ...
|--- 125_4.png
|--- 125_5.png
```

3. Replace the following code into the dataset part of the corresponding configuration files.

```
datasets:
  test_1:
    name: CUFED5
    type: RefCUFEDDataset
    dataroot_in: ./datasets/CUFED5
    dataroot_ref: ./datasets/CUFED5
    io_backend:
      type: disk

    bicubic_model: PIL

    ann_file: ../datasets/CUFED5_pairs.txt
```

### Prepare WR-SR dataset

1. Download the WR-SR dataset through this [link](https://drive.google.com/drive/folders/16UKRu-7jgCYcndOlGYBmo5Pp0_Mq71hP?usp=sharing).

2. Put the WR-SR dataset into `datasets` folder. The folder structure should follow the format:
```
- WR-SR
|-- input
|--- 001.png
|--- 002.png
|--- ...
|-- ref
|--- 001_ref.png
|--- 002_ref.png
|--- ...
```

3. Replace the following code into the dataset part of the corresponding configuration files.

```
datasets:
  test_1:
    name: WR-SR
    type: RefCUFEDDataset
    dataroot_in: ./datasets/WR-SR/input
    dataroot_ref: ./datasets/WR-SR/ref
    io_backend:
      type: disk

    bicubic_model: PIL

    ann_file: ./datasets/WR-SR_pairs.txt
```

## Prepare Train Set

### Prepare CUFED dataset

1. Download the CUFED dataset through this [link](https://drive.google.com/drive/folders/1hGHy36XcmSZ1LtARWmGL5OK1IUdWJi3I).

2. Put the CUFED dataset into `datasets` folder. The folder structure should follow the format:
```
- CUFED
|-- input
|--- ****.png
|-- ref
|--- ****.png
```
