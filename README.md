# ConvolutionalDecisionTree

## MNIST
After 20 epochs training
| Model | Accuracy |
| - | -: |
| CDTree | 99.34% |
| gcForest* | 99.26% |
| DNDF* | 99.30% |
| LeNet | 98.44% |
| SDT* | 94.45% |

For * models, results are from the correspoding papers.

gcForest: Zhou, Z. H., and J. Feng. "Deep forest: Towards an alternative to deep neural networks. arXiv 2017." arXiv preprint arXiv:1702.08835.

DNDF: Kontschieder, Peter, et al. "Deep neural decision forests." Proceedings of the IEEE international conference on computer vision. 2015.

SDT: Frosst, Nicholas, and Geoffrey Hinton. "Distilling a neural network into a soft decision tree." arXiv preprint arXiv:1711.09784 (2017).

![MNIST](https://github.com/zhangyoujia1995/ConvolutionalDecisionTree/blob/master/image/MNIST.png)

## CIFAR10
| Model | Accuracy |
| - | -: |
| ResNet* | 93.57% |
| Network in Network* | 91.20% |
| MaxOut* | 90.65% |
| CDTree | 90.13% |
| gcForest* | 69.00% |

ResNet: He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

Network in Network: Lin, Min, Qiang Chen, and Shuicheng Yan. "Network in network." arXiv preprint arXiv:1312.4400 (2013).

MaxOut: Goodfellow, Ian J., et al. "Maxout networks." arXiv preprint arXiv:1302.4389 (2013).
