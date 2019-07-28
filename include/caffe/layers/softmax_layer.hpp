#ifndef CAFFE_SOFTMAX_LAYER_HPP_
#define CAFFE_SOFTMAX_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Computes the softmax function.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 * 计算softmax函数
 * https://xuke225.github.io/2018/02/02/DeepLearning/caffe/layer/Softmax/
 * https://blog.csdn.net/bitcarmanlee/article/details/82320853
 * https://blog.csdn.net/qingsong1001/article/details/82254412
 */
template <typename Dtype>
class SoftmaxLayer : public Layer<Dtype> {
 public:
  explicit SoftmaxLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Softmax"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
     const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    //输出vector数量
  int outer_num_;
  //内部遍历维度
  int inner_num_;
  //softmax的维度index
  int softmax_axis_;
  /// sum_multiplier is used to carry out sum using BLAS
  // sum_multiplier用于使用BLAS执行求和
  Blob<Dtype> sum_multiplier_;
  /// scale is an intermediate Blob to hold temporary results.
  //scale是一个临时结果的中间Blob。
  Blob<Dtype> scale_;
};

}  // namespace caffe

#endif  // CAFFE_SOFTMAX_LAYER_HPP_
