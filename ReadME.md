# DLDL-v2-Torch

This repository is a Torch implementation of ["Age Estimation Using Expectation of Label Distribution Learning", Bin-Bin Gao, Hong-Yu Zhou, Jianxin Wu, Xin Geng](http://lamda.nju.edu.cn/gaobb/Pub_files/IJCAI2018_DLDLv2.pdf). The paper is accepted at the 27th International Joint Conference on Artificial Intelligence [(IJCAI 2018)](https://www.ijcai-18.org/).

You can train Deep ConvNets from a pre-trained model on your datasets with limited resources. This repo is created by [Bin-Bin Gao](http://lamda.nju.edu.cn/gaobb).

![Framework](https://csgaobb.github.io/Projects/DLDL-v2_files/DLDL-v2-Frame.png)
![Framework](https://csgaobb.github.io/Projects/DLDL-v2_files/DLDL-v2-Age.png)

# Download
ThinAgeNet models [Baidu Drive link](https://pan.baidu.com/s/1XFUZzEMhdyU68ATYFU0-IA)

TinyAgeNet models [Baidu Drive link](https://pan.baidu.com/s/1gBmM9FTOmbVGd8B3xeuAwA)

Align&Cropped ChaLearn Face Imgaes [Baidu Drive link](https://pan.baidu.com/s/1_HZ6P1DV9kVT4NlW3x2zFA)

Train&Test list [Baidu Drive link](https://pan.baidu.com/s/1RResT1cSTFhfrKO7-K3OeA)


# Image Demo
<video id="video" controls="" preload="none" poster="https://www.youtube.com/embed/ZtnygeUyYAs/0.jpg">
<source id="mp4" src="https://www.youtube.com/embed/ZtnygeUyYAs" type="video/mp4">
</video>

<video id="video" controls="" preload="none" poster="https://www.youtube.com/embed/ZtnygeUyYAs/0.jpg">
<source id="mp4" src="https://www.youtube.com/embed/H845rGgLgag" type="video/mp4">
</video>

<img src="./images/Thin-Trump.png" width="560"/>
<img src="./images/ThinAgeNet-ChaLearn16-Oscar2017.gif" width="560"/>
<img src="./images/ThinAgeNet-ChaLearn16-eschool.gif" width="560"/>


# Installation

## step 0: Install torch and cudnnv5.1

## step 1: Copy private file to torch
copy ./private/*.lua to the path of torch nn package (torch/extra/nn/)

copy ./private/*.c to the path of torch nn package (torch/extra/nn/lib/THNN/generic)

## step 2: Update some existing files
add the following lines to torch/nn/init.lua
```
require('nn.KLDivCriterion')
require('nn.ExpOut')
```
add the following lines to torch/extra/nn/lib/THNN/generic/THNN.h
```
TH_API void THNN_(KLDivCriterion_updateOutput)(
          THNNState *state,            // library's state
          THTensor *input,             // input tensor
          THTensor *target,            // target tensor
          THTensor *output,            // [OUT] a one-element tensor containing the loss
          bool sizeAverage);           // if true, the loss will be normalized **by total number of elements**
TH_API void THNN_(KLDivCriterion_updateGradInput)(
          THNNState *state,            // library's state
          THTensor *input,             // input tensor
          THTensor *target,            // target tensor
          THTensor *gradInput,         // [OUT] gradient w.r.t. input
          bool sizeAverage);           // if true, the loss will be normalized **by total number of elements**
```

add the following lines to torch/extra/nn/lib/THNN/init.c
```
#include "generic/KLDivCriterion.c"
#include "THGenerateFloatTypes.h"
```

## step 3: Rebuild nn package
```
luarocks install rocks/nn-scm-1.rockspec
```

## step 4: Training DLDL-v2
```
CUDA_VISIBLE_DEVICES=14,15 th main_agenet.lua -dataset chalearn15 -nGPU 2 -batchSize 128 -dataAug true  -nEpochs 60 -loss ldkl -LR 0.001 -netType hp-agenet-msceleb1m -CR 0.5 -labelStep 1

CUDA_VISIBLE_DEVICES=14,15 th main_mtagenet.lua -dataset chalearn15 -nGPU 2 -batchSize 128 -dataAug true -nEpochs 60 -loss ldklexpl1 -LR 0.001 -netType hp-mtagenet-msceleb1m  -CR 0.5 -labelStep 1 -lambda 1
```
## step 4: Evaluation
```
CUDA_VISIBLE_DEVICES=1 th evaluation.lua -dataset chalearn15 -loss ldkl -netType hp-agenet-msceleb1m -CR 0.5 -dataAug true  -labelStep 1

CUDA_VISIBLE_DEVICES=1 th evaluation.lua -dataset chalearn15 -loss ldklexpl1 -netType hp-mtagenet-msceleb1m -CR 0.5 -dataAug true -labelStep 1 -lambda 1
```


# Additional Information
If you find DLDL-v2 helpful, please cite it as
```
@inproceedings{gaoDLDLv2,
           title={Age Estimation Using Expectation of Label Distribution Learning},
           author={Gao, Bin-Bin and Zhou, Hong-Yu and Wu, Jianxin and Geng, Xin},
           booktitle={Proceedings of the 27th International Joint Conference on Artificial Intelligence (IJCAI 2018)},
           pages={xx--xx},
           year={2018}
            }

```

ATTN1: This packages are free for academic usage. You can run them at your own risk. For other purposes, please contact Prof. Jianxin Wu (wujx2001@gmail.com).

