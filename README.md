# Dual-awareness Attention for Few-shot Object Detection
<!-- ![alt text](http://github.com/Tung-I/DAnA_FSOD/blob/main/attention_visualization.jpg?raw=true) -->

<br />
<p align="center">
  <a href="https://github.com/Tung-I/Dual-awareness-Attention-for-Few-shot-Object-Detection
">
    <img src="images/attention_visualization.jpg" alt="attention_visualization" width="1024" height="280">
  </a>
</p>

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
        <li><a href="#pretrained_model">Pretrained Model</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#train">Train</a></li>
    <li><a href="#inference">Inference</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>

<!-- INTRODUCTION -->
## Introduction

While recent progress has significantly boosted few-shot classification (FSC) performance, few-shot object detection (FSOD) remains challenging for modern learning systems.
Therefore, we propose DAnA (Dual-awareness Attention) mechanism which is adaptable to various existing object detection networks and enhances FSOD performance by paying attention to specific semantics conditioned on the query.
Under the few-shot settting, the proposed method achieves SOTA performance on COCO benchmark.

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
2. Download the COCO dataset. Please follow the instruction in [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models) to prepare either VOC or COCO dataset.
3. Create symlinks for the dataset
```
cd $DAnA_FSOD/data
ln -s /[your_path_to_coco]/coco coco
```

### Pretrained Model
Multiple pretrained models can ve used in our experiments (e.g., res50, vgg16).

Please download them and put them into the data/pretrained_model/.

**NOTE**. We compare the pretrained models from Pytorch and Caffe, and surprisingly find Caffe pretrained models have slightly better performance than Pytorch pretrained. We would suggest to use Caffe pretrained models from the above link to reproduce our results.

**If you want to use pytorch pre-trained models, please remember to transpose images from BGR to RGB, and also use the same data transformer (minus mean and normalize) as used in pretrained model.**

### Installation
## Train
## Inference
## Contact
## Acknowledgements
