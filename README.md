# M2Det_obj365
M2Det for objects365 dataset.
The code based on [original M2Det](https://github.com/qijiezhao/M2Det)

# Contents

* [Data Preparation and Installation](#Preparation)

* [Demo(preparing...)](#Demo)

* [Evaluation(preparing...)](#Evaluation)

* [Training](#Training)

## Preparation
**the supported version is pytorch-0.4.1**

- Prepare python environment using [Anaconda3](https://www.anaconda.com/download/).
- Install deeplearning framework, i.e., pytorch, torchvision and other libs.

```Shell
conda install pytorch torchvision -c pytorch
pip install opencv-python,tqdm
```
- Clone this repository.
```Shell
git clone https://github.com/ryota717/M2Det_obj365.git
```
- Compile the nms and coco tools:

```Shell
sh make.sh
```

- Prepare objects365 dataset and put them on /home/data, as shown in below.

```
/home/
      ┣ data/
            ┣ objects365/
                ┣ annotations/
                    ┣ instances_train.json
                    ┣ instances_val.json
                ┣ images/
                    ┣ train/
                        ┣ 〇〇〇〇.jpg
                        ┣ △△△△.jpg
                    ┣ val/
                        ┣ 〇〇〇〇.jpg
                        ┣ △△△△.jpg
```


## Demo(preparing...)


## Evaluation(preparing...)


## Training

As simple as [demo](#Demo) and [evaluation](#Evaluation), Just use the train script:
```Shell
  CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -c=configs/m2det800_resnext.py --ngpu 4 -t True
```
All training configs and model configs are written well in configs/*.py.

### Citation:

Please cite the following paper if you feel M2Det useful to your research

```
@inproceedings{M2Det2019aaai,
  author    = {Qijie Zhao and
               Tao Sheng and
               Yongtao Wang and
               Zhi Tang and
               Ying Chen and
               Ling Cai and
               Haibing Lin},
  title     = {M2Det: A Single-Shot Object Detector based on Multi-Level Feature Pyramid Network},
  booktitle   = {The Thirty-Third AAAI Conference on Artificial Intelligence,AAAI},
  year      = {2019},
}
```
