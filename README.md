# Latent HyperNet
This repository implements the method proposed in our work "Latent HyperNet: Exploring the Layers of Convolutional Neural Networks".

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## What are HyperNets?
In deep convolutional networks, discriminative and complementary data representations can emerge in different places (i.e., layers) of the network and this information is interrupted in deeper architectures.
This occurs due to successive convolution and pooling operations, which can overlap discriminative features.
The figure below illustrates this behavior, where the attention maps (important regions in the image to predict the class label) 
present strong activations in early layers of the network. The idea behind HyperNets is to combine low-level information (shallow layers) with refined information (deep layers), focusing on achieving a better data representation.
![Figure1](/Figures/a.png)

## Requirements
- [Scikit-learn](http://scikit-learn.org/stable/)
- [Keras](https://github.com/fchollet/keras) (Recommended version 2.1.2)
- [Tensorflow](https://www.tensorflow.org/) (Recommended version 1.3.0 or 1.9)
- [Python 3](https://www.python.org/)

## Quick Start
[main.py](main.py) provides an example of our Latent HyperNet (LHN). In this example, we combine layers from a pre-trained ResNet20, which obtain an accuracy of 89.68.
We highlight that the ResNet20 accuracy is not considering data augmentation.

In this example, for fast training, we consider only 10% of the training data to learn PLS. Better results can be obtained by using more data.

## Parameters
Our method takes two parameters:
1. Layers to be combined (see line 169 in [main.py](main.py))
2. Number of components of Partial Least Squares (see line 170 in [main.py](main.py))
## Additional parameters (not recommended)
1. One-against-all scheme (oaa) (see line 200 in [main.py](main.py)). If oaa=True, LHN employs k-binary PLS, where k is the number of categories. Otherwise, LHN  learns a single multi-class PLS.
2. PLS as Network (see line 201 in [main.py](main.py)). This option inserts the PLS models inside the network, enabling inference on GPU. See [PLSGPU](https://github.com/arturjordao/PLSGPU) for more details.

### Results
The [figures](/Figures/b.png) below show the improvements (difference between the accuracy of the convolutional networks using
our LHN and the one without using LHN) achieved by our LHN. In addition, the figures show the comparison between our LHN and an existing [HyperNet](https://zpascal.net/cvpr2016/Kong_HyperNet_Towards_Accurate_CVPR_2016_paper.pdf)
.
![Figure2](/Figures/b.png)

Please cite our paper in your publications if it helps your research.
```bash
@inproceedings{Jordao:2018b:IJCNN,
title = {{Latent HyperNet: Exploring the Layers of Convolutional Neural Networks}},
author = {Artur Jordão and Ricardo Barbosa Kloss and William Robson Schwartz},
year = {2018},
booktitle = {IEEE International Joint Conference on Neural Networks (IJCNN)},
pages = {1-7},
}
```
