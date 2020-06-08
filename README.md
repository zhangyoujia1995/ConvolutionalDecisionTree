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
| CDTree | 90.23% |
| gcForest* | 69.00% |
