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


## Demo

**We provide a M2Det512_vgg pretrained model for demonstration(visualization):**

First, download the pretrained m2det512_vgg.pth([baidu cloud](https://pan.baidu.com/s/1LDkpsQfpaGq_LECQItxRFQ),[google drive](https://drive.google.com/file/d/1NM1UDdZnwHwiNDxhcP-nndaWj24m-90L/view?usp=sharing)) file. Then, move the file to weights/.

```Shell
  python demo.py -c=configs/m2det512_vgg.py -m=weights/m2det512_vgg.pth --show
```
You can see the image with drawed boxes as:


<div align=center><img src="imgs/COCO_train2014_000000000659_m2det.jpg" width="450" hegiht="163" align=center />

<div align=left>

## Evaluation

1, **We provide evaluation script for M2Det:**
```Shell
  python test.py -c=configs/m2det512_vgg.py -m=weights/m2det512_vgg.pth
```
Then, the evaluated result is shown as:

<div align=center><img src="imgs/vis/eval_result.png" width="450" hegiht="163" align=center />

<div align=left>

 Even higher than our paper's original result! :)

**2, You can run the test set with M2Det and submit to get a score:**
```Shell
  python test.py -c=configs/m2det512_vgg.py -m=weights/m2det512_vgg.pth --test
```
and submit the result file to [CODALAB webpage](https://competitions.codalab.org/competitions/5181#participate).

## Training

As simple as [demo](#Demo) and [evaluation](#Evaluation), Just use the train script:
```Shell
  CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -c=configs/m2det512_vgg.py --ngpu 4 -t True
```
All training configs and model configs are written well in configs/*.py.

## Multi-scale Evaluation
To be added.

## Pre-trained Files
Now, we only provide m2det512_vgg.pth([baidu cloud](https://pan.baidu.com/s/1LDkpsQfpaGq_LECQItxRFQ),[google drive](https://drive.google.com/file/d/1NM1UDdZnwHwiNDxhcP-nndaWj24m-90L/view?usp=sharing)) due to we have other tasks recently, we decide to release other models in the future.

## Others

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


## Contact
For any question, please file an issue or contact
```
Qijie Zhao: zhaoqijie@pku.edu.cn
```
