# 사물 이미지 분류 경진대회

총 10개의 class 로 이루어진 데이터를 분류
<br>[Competition Link](https://dacon.io/competitions/official/235874/overview/description)
* 주최/주관: Dacon<br>
**Private 6th (6/235, 3%)**<br>
**최종 코드 검증 2nd (2/235, 1%)**
***

## Structure
Train/Test data folder and sample submission file must be placed under **dataset** folder.
```
repo
  |——dataset
        |——train
                |——airplane
                  |——0000.jpg
                |——....
        |——test
                |——0000.jpg
                |——....
        |——sample_submission.csv
  |——models
        |——model
        |——runners
  |——data
  |——utils
```
***

## Development Environment
* Ubuntu 18.04.5
* i9-10900X
* RTX 3090 1EA
* CUDA 11.3
***

## Install Dependencies (GPU)

[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-3812/)

```shell
sh install_dependency.sh
```
***

## Solution

### Train
* Transfer Learning with timm
* Resnet34d backbone
* Trained for 300 epochs
* 5 StratifiedKFold train
* Mixup on every 10 steps / Save plot, model arguments, best model spec, inference, model on every fold.
* Rotate(30), RamdomRotate90, Resize, HorizontalFlip, VerticalFlip, Normalize

```shell
python main.py
```
***

### Inference
5 fold ensemble (soft-voting) with Test Time Augmentation
* HorizontalFlip, VerticalFlip

```shell
python kfold_inference.py
```
