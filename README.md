## Summary:
This is an experimental study. The main goals of this study are implementing different CNN achitectures and achiving reasonable results in order to solve a specific scene classification problem by using AlexNet, VGG16, ResNet50 achitectures. MLRSNet, an aerial imaging dataset, is used to define the problem. All implementations are built with Keras and Tensorflow in Python.

## Problem Statement & Dataset:

MLRSNet is a dataset that contains 109.161 satellite images in 46 categories like residential areas, industrial areas, briges, freeways, parks, stadiums and ports. More information is [here](https://github.com/cugbrs/MLRSNet).

The main reason for usage of this dataset is to gain useful information for various applications like aerial vehicle observation, satellite imagery in uncertian geographical conditions.

## Results:

To solve the problem stated above, three different CNN architectures are implemented. The architectures are not state-of-the-art due to computation source limits. The trained models are available on [Google Drive](https://drive.google.com/drive/folders/1e5d7TV9LtmO1q9-6WXRWVvytPUQbOqBC?usp=sharing).

The results are shown below: 

|                  | Training Accuracy (Top-1) | Validation Accuracy (Top-1) | Test Accuracy (Top-1) | User Set Accuracy (Top-1) |
|------------------|---------------------------|-----------------------------|-----------------------|---------------------------|
| MLRSNet_AlexNet  | 0.9668                    | 0.9884                      | 0.9863                | 0.4743                    |
| MLRSNet_VGG16    | 0.9621                    | 0.9831                      | 0.9852                | 0.5128                    |
| MLRSNet_ResNet50 | 0.8981                    | 0.9081                      | 0.9108                | 0.4871                    |

## Conclusion:

Each CNN architecture that used for this problem archieved "good" results on validation set. A user-collected set is used to experiment this kind of generalization of the models and since the models are suffering from data mismatch problem, the generalization ability was not sufficient. The results show the better accuracy results can be obtained by using application-specific real life data. 

## References:

1. Xiaoman Qi, Panpan Zhu, Yuebin Wang, Liqiang Zhang, Junhuan Peng, Mengfan Wu, Jialong Chen, Xudong Zhao, Ning Zang, P.Takis Mathiopoulos (2021). _MLRSNet: A Multi-label High Spatial Resolution Remote Sensing Dataset for Semantic Scene Understanding_

2. Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton (2012). _ImageNet Classification with Deep Convolutional Neural Networks_

3. Karen Simonyan, Andrew Zisserman (2015). _Very Deep Convolutional Networks for Large-Scale Image Recognition_ 

4. Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (2015). _Deep Residual Learning for Image Recognition_ 