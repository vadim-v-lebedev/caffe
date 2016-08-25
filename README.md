This Caffe fork contains code for our paper:
[Fast ConvNets Using Group-wise Brain damage](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Lebedev_Fast_ConvNets_Using_CVPR_2016_paper.pdf)

Currently implemented features:

Sparse convolution on cpu, forward propagation only.
- [sparse_conv_layer.cpp](https://github.com/vadim-v-lebedev/caffe/blob/master/src/caffe/layers/sparse_conv_layer.cpp)
- [sparse_conv_layer.hpp](https://github.com/vadim-v-lebedev/caffe/blob/master/include/caffe/layers/sparse_conv_layer.hpp)

Without backpropagation, straightforward training of sparse convolution layer is not possible. Hovewer regularization_type: "fix_zeros", provides an alternative way to train sparse weigths with regular implementation of the convolution.

[Notebook example](https://github.com/vadim-v-lebedev/caffe/blob/master/examples/sparse/sparse_convolution_demonstration.ipynb)
