# Convolution-Centering-Semantic-Analysis

[![Paper](https://img.shields.io/badge/Paper-arXiv.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

This repository contains the official implementation of the paper:

> **Convolution-Centering Semantic Analysis: Beyond Normalization in Deep Networks**
> Author One, Author Two  
> *Conference / Journal Name*, Year.  
> [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX)

## 📄 Abstract

Despite the widespread using of Normalization Layers for stabilizing the deep networks training, the semantic role of the centering operation remains insufficiently explored. Modern research is largely focused on the optimization aspects e.g. of the Batch Normalization Layer. We postulate that centering actively suppresses activation components that are linearly dependent on the batch mean, which mostly correspond to domain-specific rather than class-specific features. To test this hypothesis, we analyze the interaction between the mean vector and the layer weights, and track the dynamics of cosine similarities within and between classes across a wide range of tasks and architectures. Specifically, we examine: a YOLO model on the COCO dataset, the MatchboxNet model for Keyword Spotting, and ResNet models for face recognition and image classification. The results empirically demonstrate that depth-wise centering hierarchically filters common patterns, enhances class-specific features, and improves class compactness in the activation space. Our findings reveal the nature of suppressing semantically common components allows us to use the center vector for semantic coloring of weights and opens new research directions for the interpretation analysis of representations in deep networks.

## Reproducing the results

### Installation

<details>
<summary>DAMO-YOLO</summary>

Step1. Install DAMO-YOLO.
```shell
git clone https://github.com/vshokorov/Convolution-Centering-Semantic-Analysis.git
cd Convolution-Centering-Semantic-Analysis/YOLO
conda create -n DAMO-YOLO python=3.7 -y
conda activate DAMO-YOLO
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
export PYTHONPATH=$PWD:$PYTHONPATH
```
Step2. Install [pycocotools](https://github.com/cocodataset/cocoapi).

```shell
pip install cython;
pip install git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI # for Linux
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI # for Windows
```

Step3. Download a pretrained torch from [the benchmark table](https://github.com/tinyvision/damo-yolo#model-zoo) for Tiny model: damoyolo_tinynasL20_T_420.pth
</details>

<details>

<summary>Audio</summary>

Step1. Install environment.
```shell
git clone https://github.com/vshokorov/Convolution-Centering-Semantic-Analysis.git
cd Convolution-Centering-Semantic-Analysis/sound
chmod +x install.sh
./install.sh
# This will create new "sound" conda environment with all dependencies
```
Step2. Prepare dataset.

```shell
# requires 7 GB of free space
unzip sound/sound_dataset_demo/google_speech_recognition_v1.zip -d sound/sound_dataset_demo/google_speech_recognition_v1
unzip sound/sound_dataset_demo/google_speech_recognition_v2_1.zip -d sound/sound_dataset_demo/google_speech_recognition_v2
unzip sound/sound_dataset_demo/google_speech_recognition_v2_2.zip -d sound/sound_dataset_demo/google_speech_recognition_v2
```

</details>

<details>

<summary>FaceRec</summary>

Step1. Install environment.

**Follow the installation instruction for YOLO, then**
```shell
cd ../facerec_and_cifar
pip install -r facerec_requirements.txt
```
Step2. Prepare dataset.

```shell
unzip datasets
```

</details>

<details>

<summary>Cifar100</summary>

Step1. Install environment.

**Follow the installation instruction for YOLO, then**
```shell
cd ../facerec_and_cifar
```

</details>


### Running Experiments
To reproduce any results check **plots_general.ipynb** for further instuctions


### Citation
If you use this code in your research, please cite our paper:

```text
@inproceedings{author2024title,
  ....
}
```
