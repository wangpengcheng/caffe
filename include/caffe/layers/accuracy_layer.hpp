#ifndef CAFFE_ACCURACY_LAYER_HPP_
#define CAFFE_ACCURACY_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/**
 * @brief Computes the classification accuracy for a one-of-many
 *        classification task.
 * 从分类任务中计算分类的准确度
 * AccuracyLayer 只能计算准确率，不需要也不能后向传播。
 * 当只有一个top时，会记录一个总的分类准确率，当有两个top时，另一个top会记录每一类各自的准确率。
 */
template <typename Dtype>
class AccuracyLayer : public Layer<Dtype> {
 public:
  /**
   * @param param provides AccuracyParameter accuracy_param,
   *     with AccuracyLayer options:
   *   - top_k (\b optional, default 1).
   *     Sets the maximum rank @f$ k @f$ at which a prediction is considered
   *     correct.  For example, if @f$ k = 5 @f$, a prediction is counted
   *     correct if the correct label is among the top 5 predicted labels.
   */
  explicit AccuracyLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Accuracy"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }

  // If there are two top blobs, then the second blob will contain
  // accuracies per class.
  //如果这里有两个top blob，第二个blob会包含 每个类的准确率
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  /**
   * @param bottom input Blob vector (length 2)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the predictions @f$ x @f$, a Blob with values in
   *      @f$ [-\infty, +\infty] @f$ indicating the predicted score for each of
   *      the @f$ K = CHW @f$ classes. Each @f$ x_n @f$ is mapped to a predicted
   *      label @f$ \hat{l}_n @f$ given by its maximal index:
   *      @f$ \hat{l}_n = \arg\max\limits_k x_{nk} @f$
   *   -# @f$ (N \times 1 \times 1 \times 1) @f$
   *      the labels @f$ l @f$, an integer-valued Blob with values
   *      @f$ l_n \in [0, 1, 2, ..., K - 1] @f$
   *      indicating the correct class label among the @f$ K @f$ classes
   * @param top output Blob vector (length 1)
   *   -# @f$ (1 \times 1 \times 1 \times 1) @f$
   *      the computed accuracy: @f$
   *        \frac{1}{N} \sum\limits_{n=1}^N \delta\{ \hat{l}_n = l_n \}
   *      @f$, where @f$
   *      \delta\{\mathrm{condition}\} = \left\{
   *         \begin{array}{lr}
   *            1 & \mbox{if condition} \\
   *            0 & \mbox{otherwise}
   *         \end{array} \right.
   *      @f$
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);


  /// @brief Not implemented -- AccuracyLayer cannot be used as a loss.
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    for (int i = 0; i < propagate_down.size(); ++i) {
      if (propagate_down[i]) { NOT_IMPLEMENTED; }
    }
  }
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    //标签维度，外部计数，内部计数
  int label_axis_, outer_num_, inner_num_;
    //之将值最高的几个标签进行计算
  int top_k_;
    //https://blog.csdn.net/Scythe666/article/details/78790648
  /// Whether to ignore instances with a certain label.
  //是否开启距离label
  bool has_ignore_label_;
  /// The label indicating that an instance should be ignored.
  //忽略的距离上线
  int ignore_label_;
  /// Keeps counts of the number of samples per class.
  //对每个类别的准确率buffer
  Blob<Dtype> nums_buffer_;
};

}  // namespace caffe

#endif  // CAFFE_ACCURACY_LAYER_HPP_
