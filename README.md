# Dual-awareness Attention for Few-shot Object Detection
<!-- ![alt text](http://github.com/Tung-I/DAnA_FSOD/blob/main/attention_visualization.jpg?raw=true) -->


<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#introduction">Introduction</a>
    </li>
    <li>
      <a href="#getting_started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#data_preparation">Data Preparation</a></li>
        <li><a href="#pretrained_weights">Pretrained Weights</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#train">Train</a></li>
    <li><a href="#inference">Inference</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>

<!-- INTRODUCTION -->
## Introduction

While recent progress has significantly boosted few-shot classification (FSC) performance, few-shot object detection (FSOD) remains challenging for modern learning systems.
Therefore, we propose DAnA (Dual-awareness Attention) mechanism which is adaptable to various existing object detection networks and enhances FSOD performance by paying adaptable attention to support images conditioned on given query information. The proposed method achieves SOTA results on COCO benchmark, outperforming the strongest baseline by 47% on performance.\
paper link: https://arxiv.org/abs/2102.12152

<br />
<p align="center">
  <a href="https://github.com/Tung-I/Dual-awareness-Attention-for-Few-shot-Object-Detection
">
    <img src="images/prediction.jpg" alt="prediction" width="1024" height="660">
  </a>
</p>

<!-- GETTING STARTED -->
## Getting Started
### Prerequisites
* Python 3.6
* Cuda 10.0 or 10.1
* Pytorch 1.2.0 or higher

### Data Preparation
1. First, clone the repository and create a data folder:
```
cd Dual-awareness-Attention-for-Few-shot-Object-Detection && mkdir data
```
2. Download the COCO dataset. Please follow the instruction in [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models).
Create the symlinks to datasets.
```
$ cd data

For VOC 2007
$ ln -s /your/path/to/VOC2007/VOCdevkit VOCdevkit2007

For COCO
$ ln -s /your/path/to/VOC2012/coco coco
```

3. Prepare support images for inference
```
$ ln -s /your/path/to/supports supports
```

4. Create the folder for saving model weights
```
$ mkdir models
```

### Pretrained Weights
1.Backbone Networks\
Please download the pretrained backbone models (e.g., res50, vgg16) and put them into data/pretrained_model. 
```
$ mkdir data/pretrained_model && cd data/pretrained_model
$ ln -s /your/path/to/res50.pth res50.pth
```
**NOTE**. We would suggest to use Caffe pretrained models to reproduce our results.
**If you want to use pytorch pre-trained models, please remember to transpose images from BGR to RGB, and also use the same data transformer (minus mean and normalize) as used in pretrained model.**

2. Model Weights
The pretrained weights of DAnA can be download [here](https://drive.google.com/file/d/1JaYF-Ep-C6b5X01_e9tFRzFgRXMJQYQ7/view?usp=sharing).\
To use the pretrained weights, please create a folder called "models" and put the unzipped folder in it.\
The architecture should be
```
models/DAnA_COCO_ft30/...
```

### Compilation
Prepare the environment.
```
$ conda env create -f env.yml
$ source activate [NAME_OF_THE_ENV]
```
Compile COCO API.
```
$ cd lib
$ git clone https://github.com/pdollar/coco.git 
$ cd coco/PythonAPI
$ make && make install
put pycocotools under data/
$ mv cocoapi/PythonAPI/pycocotools .
```
Compile the cuda dependencies using following commands.
```
$ cd lib
$ python setup.py build develop
```
If you are confronted with error during the compilation, you might miss to export the CUDA paths to your environment.

## Train

```
$ mkdir models
$ python train.py --dataset pascal_voc --net dana --lr 0.001 --bs 8 --epochs 16 --save_dir models/dana_bs8_lr1e3
re-training
$ python train.py --dataset pascal_voc --net dana --lr 0.001 --bs 8 --epochs 16 --save_dir models/dana_bs8_lr1e3 --r --load_dir models/dana_bs8_lr1e3 --checkepoch 12
```

## Inference
```
$ python inference.py --net dana --bs 1 --load_dir models/dana_bs8_lr1e3 --checkepoch 16 --way 1 --shot 3 --sup_dir data/sup_im 
```
## Attention Visualization
<br />
<p align="center">
  <a href="https://github.com/Tung-I/Dual-awareness-Attention-for-Few-shot-Object-Detection
">
    <img src="images/attention_visualization.jpg" alt="attention_visualization" width="1024" height="280">
  </a>
</p>

## Acknowledgements
The project is mainly build on [faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch/tree/pytorch-1.0).
