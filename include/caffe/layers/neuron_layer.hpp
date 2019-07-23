#ifndef CAFFE_NEURON_LAYER_HPP_
#define CAFFE_NEURON_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief An interface for layers that take one blob as input (@f$ x @f$)
 *        and produce one equally-sized blob as output (@f$ y @f$), where
 *        each element of the output depends only on the corresponding input
 *        element.
 * 将一个blob作为输入的图层接口（@ f $ x @ f $）
 * 并产生一个大小相同的blob作为输出（@ f $ y @ f $），其中
 * 输出的每个元素仅取决于相应的输入元件。
 */
//神经元层模板函数，主要是计算，并且输入和输出的元素数量一一对应
//参考连接：https://blog.csdn.net/samylee/article/details/75222745

template <typename Dtype>
class NeuronLayer : public Layer<Dtype> {
 public:
 //构造函数
  explicit NeuronLayer(const LayerParameter& param)
     : Layer<Dtype>(param) {}
  //更改大小函数
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  //更改底部blob数量
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  //更改顶部Blob数量
  virtual inline int ExactNumTopBlobs() const { return 1; }
};

}  // namespace caffe

#endif  // CAFFE_NEURON_LAYER_HPP_
