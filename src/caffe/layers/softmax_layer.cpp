#include <algorithm>
#include <vector>

#include "caffe/layers/softmax_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
//进行参数形状调整，主要是将出书矩阵和输入矩阵相互同步
template <typename Dtype>
void SoftmaxLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  top[0]->ReshapeLike(*bottom[0]);
  //选择最大化的标准维度
  vector<int> mult_dims(1, bottom[0]->shape(softmax_axis_));
  //执行的求和blob
  sum_multiplier_.Reshape(mult_dims);
  //求和数据
  Dtype* multiplier_data = sum_multiplier_.mutable_cpu_data();
  //分配内存
  caffe_set(sum_multiplier_.count(), Dtype(1), multiplier_data);
  //外部输出数量，一般是N
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  //内部数量，即剩下的循环数量
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  //设置缩放维度
  vector<int> scale_dims = bottom[0]->shape();
  //缩放维度中输出的标准维度为1，这里主要是为了方便中间变量的计算
  scale_dims[softmax_axis_] = 1;
  scale_.Reshape(scale_dims);
}
//前向计算函数
template <typename Dtype>
void SoftmaxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  //cpu data
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* scale_data = scale_.mutable_cpu_data();
  //获取标准维度的长度
  int channels = bottom[0]->shape(softmax_axis_);
  //标准维度的矩阵数目
  int dim = bottom[0]->count() / outer_num_;
  //将bottom_data拷贝到top_data中
  caffe_copy(bottom[0]->count(), bottom_data, top_data);
  // We need to subtract the max to avoid numerical issues, compute the exp,
  // and then normalize.
  for (int i = 0; i < outer_num_; ++i) {
    // initialize scale_data to the first plane
    caffe_copy(inner_num_, bottom_data + i * dim, scale_data);
    //求取范围内的最大值
    //计算最大值max，scale_data为保存最大值的变量
    for (int j = 0; j < channels; j++) {
      for (int k = 0; k < inner_num_; k++) {
        scale_data[k] = std::max(scale_data[k],
            bottom_data[i * dim + j * inner_num_ + k]);
      }
    }
    // subtraction
    //矩阵乘法；这里实质上是执行了一次矩阵减法，减去了最大值
    //每个输入Zi均减去最大值max
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, inner_num_,
        1, -1., sum_multiplier_.cpu_data(), scale_data, 1., top_data);
    // exponentiation
    //指数化
    //求e^(Zi-max)
    caffe_exp<Dtype>(dim, top_data, top_data);
    // sum after exp
    //进行指数求和结果保存在scale_data中
    //求和，计算e^(Zi-max),i=[0,n]之和
    caffe_cpu_gemv<Dtype>(CblasTrans, channels, inner_num_, 1.,
        top_data, sum_multiplier_.cpu_data(), 0., scale_data);
    // division
    //每一个指数e^(Zi-max)除以上一步求得的最大值
    for (int j = 0; j < channels; j++) {
      //分块矩阵除法；结果保存在top_data中
      caffe_div(inner_num_, top_data, scale_data, top_data);
      //更新top_data步长
      top_data += inner_num_;
    }
  }
}
//反向回调函数
template <typename Dtype>
void SoftmaxLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,//是否参与反向计算
    const vector<Blob<Dtype>*>& bottom
    ) {
  //top误差数据
  const Dtype* top_diff = top[0]->cpu_diff();
  //top_data
  const Dtype* top_data = top[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  //中间缩放计算数据
  Dtype* scale_data = scale_.mutable_cpu_data();
  //获取通道数目
  int channels = top[0]->shape(softmax_axis_);
  //计算比较范围和维度
  int dim = top[0]->count() / outer_num_;
  //进行内存拷贝
  caffe_copy(top[0]->count(), top_diff, bottom_diff);
  //开始遍历，寻找相关值
  for (int i = 0; i < outer_num_; ++i) {
    // compute dot(top_diff, top_data) and subtract them from the bottom diff
    //top_diff=
    for (int k = 0; k < inner_num_; ++k) {
      //计算向量的内积,按照inner_num作为维度进行计算
      //scale_data_i=bottom_diff_i*top_data_i;
      scale_data[k] = caffe_cpu_strided_dot<Dtype>(
        channels,
        bottom_diff + i * dim + k, 
        inner_num_,
        top_data + i * dim + k, 
        inner_num_
        );
    }
    // subtraction
    //进行矩阵的乘法:C = alpha*op( A )*op( B ) + beta*C
    //即：bottom_diff=-1*sum_multiplier_*scale_data+bottom_diff
    //一般的初始值bottom_diff为1
    //即：bottom_diff=bottom_diff_-bottom_diff_xtop_data_=bottom_diff(1-top_data);
    caffe_cpu_gemm<Dtype>(
      CblasNoTrans, 
      CblasNoTrans, 
      channels, //m
      inner_num_, //n
      1,//k;sum_mulitiplier是一列矩阵
      -1., //alpha
      sum_multiplier_.cpu_data(),//矩阵a 
      scale_data, //矩阵b
      1., //beta
      bottom_diff + i * dim //系数c
      );
  }
  // elementwise multiplication
  //计算乘积：
  //bottom_diff=top_data_x(top_diff-top_data*top_diff);
  caffe_mul(top[0]->count(), bottom_diff, top_data, bottom_diff);
}


#ifdef CPU_ONLY
STUB_GPU(SoftmaxLayer);
#endif

INSTANTIATE_CLASS(SoftmaxLayer);

}  // namespace caffe
