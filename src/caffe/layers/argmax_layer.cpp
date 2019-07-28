#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/argmax_layer.hpp"

namespace caffe {
  /* 
   * top_k 最大元素输出的k的数目
   * out_max_val（可选bool） 如果设置，则输出对的向量（max_ind，max_val），除非设置了轴，然后沿指定的轴输出max_val。
   * axis（可选int）。 如果设置，则沿指定轴最大化，否则最大化第一个/ num维的每个索引的展平尾随尺寸。
   * 
   * */
template <typename Dtype>
void ArgMaxLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const ArgMaxParameter& argmax_param = this->layer_param_.argmax_param();
  out_max_val_ = argmax_param.out_max_val();
  top_k_ = argmax_param.top_k();
  //是否按照维度输出
  has_axis_ = argmax_param.has_axis();
  CHECK_GE(top_k_, 1) << "top k must not be less than 1.";
  //如果按照维度向量输出
  if (has_axis_) {
    //我维度数量，axis是NCWH中的一个index即0-3
    axis_ = bottom[0]->CanonicalAxisIndex(argmax_param.axis());
    CHECK_GE(axis_, 0) << "axis must not be less than 0.";
    CHECK_LE(axis_, bottom[0]->num_axes()) <<
      "axis must be less than or equal to the number of axis.";
      //检查top_k_是否在输出维度之内
    CHECK_LE(top_k_, bottom[0]->shape(axis_))
      << "top_k must be less than or equal to the dimension of the axis.";
  } else {
    //否则按照通道数目N进行输出
    CHECK_LE(top_k_, bottom[0]->count(1))
      << "top_k must be less than or equal to"
        " the dimension of the flattened bottom blob per instance.";
  }
}
//重构维度
template <typename Dtype>
void ArgMaxLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //获取维度总数
  int num_top_axes = bottom[0]->num_axes();
  if ( num_top_axes < 3 ) num_top_axes = 3;
  //创建shap
  std::vector<int> shape(num_top_axes, 1);
  //这里的has_axis是为了为新的维度数据做兼容设置的
  if (has_axis_) {
    // Produces max_ind or max_val per axis
    shape = bottom[0]->shape();
    shape[axis_] = top_k_;
  } else {
    //图片数目n
    shape[0] = bottom[0]->shape(0);
    // Produces max_ind
    shape[2] = top_k_;
    if (out_max_val_) {
      // Produces max_ind and max_val
      shape[1] = 2;
    }
  }
  //更改最大值
  top[0]->Reshape(shape);
}
//前向计算
template <typename Dtype>
void ArgMaxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  //底部bottom
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  //定义维度和输出维度长度
  int dim, axis_dist;
  if (has_axis_) {
    //获取提取的维度，如axis=0,那么dim就是c*w*h
    dim = bottom[0]->shape(axis_);
    // Distance between values of axis in blob
    //每一个维度值的距离，当axis=0;axis_dist=n
    axis_dist = bottom[0]->count(axis_) / dim;
  } else {
    //计算1-num_axis的维度的乘积和
    dim = bottom[0]->count(1);
    //每个维度的步长
    axis_dist = 1;
  }
  //计算输出数量=维度总和/分块维度和
  int num = bottom[0]->count() / dim;
  //创建映射向量数量为维度
  std::vector<std::pair<Dtype, int> > bottom_data_vector(dim);
  //遍历blob，为bottom_data_vector初始化
  for (int i = 0; i < num; ++i) {//遍历每一个维度
    for (int j = 0; j < dim; ++j) {
      bottom_data_vector[j] = std::make_pair(
        bottom_data[(i / axis_dist * dim + j) * axis_dist + i % axis_dist], j);
    }
    //进行排序求top_k的最大值
    std::partial_sort(
        bottom_data_vector.begin(), bottom_data_vector.begin() + top_k_,
        bottom_data_vector.end(), std::greater<std::pair<Dtype, int> >());
    //遍历创建输出
    for (int j = 0; j < top_k_; ++j) {
      //是否输出最大值
      if (out_max_val_) {//是否输出没有我维度的最大值索引
        //是否使用axis
        if (has_axis_) {
          // Produces max_val per axis
          //计算每一个维度的最大值；注意这里的映射关系，是将bottom_data_vector映射到一个和bottom相同维度的稀疏矩阵上面当axis_dist=1时
          //top_data[i*top_k_+j]
          top_data[(i / axis_dist * top_k_ + j) * axis_dist + i % axis_dist]
            = bottom_data_vector[j].first;
        } else {
          // Produces max_ind and max_val
          //直接进行数据的输出，这里同时出书max_ind和max_val因此要*2；即最大值和他的索引
          top_data[2 * i * top_k_ + j] = bottom_data_vector[j].second;
          top_data[2 * i * top_k_ + top_k_ + j] = bottom_data_vector[j].first;
        }
      } else {
        // Produces max_ind per axis
        //计算每个维度的最大值缩影
        top_data[(i / axis_dist * top_k_ + j) * axis_dist + i % axis_dist]
          = bottom_data_vector[j].second;
      }
    }
  }
}

INSTANTIATE_CLASS(ArgMaxLayer);
REGISTER_LAYER_CLASS(ArgMax);
//这里主要就是一个简单的Max函数，不过这里复杂化了，可以求取任意维度和范围及数量的最大值
//caffe果然没有考虑性能和架构
}  // namespace caffe
