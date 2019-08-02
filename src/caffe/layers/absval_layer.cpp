#include <vector>

#include "caffe/layers/absval_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "include/caffe/"

namespace caffe {
//初始化函数
template <typename Dtype>
void AbsValLayer<Dtype>::LayerSetUp(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top
      ) {
        //先将第一个设置成为相同
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  //查看是否相同
  CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
    "allow in-place computation.";
}
//cpu前向计算函数
template <typename Dtype>
void AbsValLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, 
    const vector<Blob<Dtype>*>& top
    ) {
      //获取数组长度
  const int count = top[0]->count();
  //获取数组初始指针
  Dtype* top_data = top[0]->mutable_cpu_data();
  //求取数据的绝对值
  caffe_abs(count, bottom[0]->cpu_data(), top_data);
}

template <typename Dtype>
void AbsValLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
      //获取数据的总个数
  const int count = top[0]->count();
  //获取CPU数据指针
  const Dtype* top_diff = top[0]->cpu_diff();
  //是否反向传播
  if (propagate_down[0]) {//是要反向传播
    //获取bottom 0的数据指针
    const Dtype* bottom_data = bottom[0]->cpu_data();
    //获取反向的数据指针
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    //由caffe_sign宏定义生成 将bottom_data里头的每个元素的正负值复制到bottom_diff，即diff=data
    caffe_cpu_sign(count, bottom_data, bottom_diff);
    //计算偏导数计算关于本层bottom的偏导
    //将关于top_diff乘以当前层bottom_diff中每个数据的符号,bottom的diff变为原来的绝对值
    //bottom_diff*=top_diff
    caffe_mul(count, bottom_diff, top_diff, bottom_diff);
  }
}
//如果表明只用cpu
#ifdef CPU_ONLY
//声明前后向函数，但是输出错误信息，不做任何工作
STUB_GPU(AbsValLayer);
#endif
//实例化模板类 char gInstantiationGuardAbsValLayer;template class classname<float>;template class classname<double>
INSTANTIATE_CLASS(AbsValLayer);
//注册函数
REGISTER_LAYER_CLASS(AbsVal);

}  // namespace caffe
