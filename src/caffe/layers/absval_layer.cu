#include <vector>

#include "caffe/layers/absval_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
//前向计算GPU
template <typename Dtype>
void AbsValLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
      //获取顶部Blob数据
  const int count = top[0]->count();
  Dtype* top_data = top[0]->mutable_gpu_data();
  //GPU计算绝对值
  caffe_gpu_abs(count, bottom[0]->gpu_data(), top_data);
}
//反向计算
template <typename Dtype>
void AbsValLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const int count = top[0]->count();
  const Dtype* top_diff = top[0]->gpu_diff();
  if (propagate_down[0]) {
    //获取底部数据指针
    const Dtype* bottom_data = bottom[0]->gpu_data();
    //获取误差
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    //将给予bottom 差值初始化
    caffe_gpu_sign(count, bottom_data, bottom_diff);
    //将关于top_diff乘以当前层bottom_diff中每个数据的符号,bottom的diff变为原来的绝对值
    //bottom_diff*=top_diff
    caffe_gpu_mul(count, bottom_diff, top_diff, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(AbsValLayer);


}  // namespace caffe
