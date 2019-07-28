#include <vector>

#include "caffe/layers/accuracy_layer.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {

template <typename Dtype>
//前向计算的准确率
__global__ void AccuracyForwardGPU(
          
          const int nthreads,//线程数量
          const Dtype* bottom_data,//底部输入数据 
          const Dtype* label, //标签数据
          Dtype* acc,//准确率数组
          const int num,//数量 
          const int dim,//维度
          const int spatial_dim,//分块的维度
          const int num_labels,//标签数量 
          const int top_k,//每个映射准确率最高的k个
          const bool has_ignore_label_,//是否存在标签忽略 
          const int ignore_label_,//忽略标签值
          Dtype* counts//总计数组
        ) {
          //这里巧妙的利用cuda特性，将双重for循环展开了，值得学习
  CUDA_KERNEL_LOOP(index, nthreads) {
    //记录当前cuda维度x
    const int n = index / spatial_dim;
    //记录当前cuda维度y
    const int s = index % spatial_dim;
    //获取当前数据的标签值，相当于一次遍历label了
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);
    //查找当前对应的bottom的分为label的准确度
    const Dtype prob_of_true_class = bottom_data[n * dim
                                                 + label_value * spatial_dim
                                                 + s];
    //比它更好的数量
    int num_better_predictions = -1;  // true_class also counts as "better"
    //是否需要忽略该标签
    if (has_ignore_label_ && label_value == ignore_label_) {
      acc[index] = 0;
      counts[index] = 0;
    } else {
      //
      for (int k = 0; k < num_labels & num_better_predictions < top_k; k++) {
        //大于分类的标签数量
        num_better_predictions +=
          (bottom_data[n * dim + k * spatial_dim + s] >= prob_of_true_class);
      }
      //获取准确率
      acc[index] = (num_better_predictions < top_k);
      //获取计数
      counts[index] = 1;
    }
  }
}

template <typename Dtype>
//前向计算每一个类的准确率
__global__ void AccuracyForwardWithPerClassGPU(
          const int nthreads,//线程数目
          const Dtype* bottom_data,//底部数据 
          const Dtype* label,//标签数组
          Dtype* acc,//准确率数组 
          Dtype* counts,//计算统计数组
          const int num,//数量 
          const int dim,//维度
          const int spatial_dim,//分块维度，一般是w*h
          const int num_labels, //标签数量
          const int top_k,//最高的几个准确率
          const bool has_ignore_label_,//是否忽略标签
          const int ignore_label_//忽略标签编号
        ) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    //cuda x
    const int n = index / spatial_dim;
    //cuda y
    const int s = index % spatial_dim;
    //标签值
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);
    //获取对应的label的种类的准确度，注意这里的稀疏性
    const Dtype prob_of_true_class = bottom_data[n * dim
                                                 + label_value * spatial_dim
                                                 + s];
    if (has_ignore_label_ && label_value == ignore_label_) {
      // nothing to be done.
    } else {
      int num_better_predictions = -1;  // true_class also counts as "better"
      for (int k = 0; k < num_labels & num_better_predictions < top_k; k++) {
        num_better_predictions +=
          (bottom_data[n * dim + k * spatial_dim + s] >= prob_of_true_class);
      }
      //记录这个计算块的准确度
      acc[label_value*nthreads + index] += (num_better_predictions < top_k);
      //这个的计算量为1
      counts[label_value*nthreads + index] = 1;
    }
  }
}
//前向计算函数，注意这里主要还是使用数据的分裂和合并
template <typename Dtype>
void AccuracyLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
      //data
  const Dtype* bottom_data = bottom[0]->gpu_data();
  //label
  const Dtype* bottom_label = bottom[1]->gpu_data();
  //计算维度dim=N*C*H*W/N= C*H*W
  const int dim = bottom[0]->count() / outer_num_;
  //labels数量墨认是C
  const int num_labels = bottom[0]->shape(label_axis_);
  //计算线程数量N*C*H*W
  const int nthreads = outer_num_ * inner_num_;
  // Since this memory is not used for anything, we use it here to avoid having
  // to allocate new GPU memory to accumulate intermediate results.
  //因为这里的GPU数据没有被用到，因此在这里使用
  //gpu的diff数据来记录准确度
  Dtype* acc_data = bottom[0]->mutable_gpu_diff();
  //如果只有一个top直接计算总的准确度
  if (top.size() == 1) {
    // simple case - report only global accuracy.

    // Similarly, this memory is never used elsewhere, and thus we can use it
    // to avoid having to allocate additional GPU memory.
    //使用GPU的diff指针
    Dtype* counts = bottom[1]->mutable_gpu_diff();
    // NOLINT_NEXT_LINE(whitespace/operators)
    //前向计算获取accuracy
    AccuracyForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(
          nthreads, 
          bottom_data, 
          bottom_label,
          acc_data, 
          outer_num_, 
          dim, inner_num_,
          num_labels, 
          top_k_,
          has_ignore_label_, 
          ignore_label_, 
          counts);
          //定义准确度
    Dtype acc;
    //求取准确的总值，并计算准确度，注意这里的合并界限是w*h，因此存储的是每个图片的分类
    caffe_gpu_asum(nthreads, acc_data, &acc);
    Dtype valid_count;
    //计算总的计算次数
    caffe_gpu_asum(nthreads, counts, &valid_count);
    if (valid_count > 0) {
      //计算总的准确度
      top[0]->mutable_cpu_data()[0] = acc / valid_count;
    } else {
      top[0]->mutable_cpu_data()[0] = 0;
    }
  } else {
    // need to report per-class accuracy as well
    //需要统计每个类别的准确度
    // allocate space for more detailed "counts"
    //为counts分配内存空间
    nums_buffer_.ReshapeLike(*bottom[0]);
    //统计数组指针
    Dtype* counts = nums_buffer_.mutable_gpu_data();
    //为acc数组分配内存
    caffe_gpu_set(bottom[0]->count(), Dtype(0), acc_data);
    //为counts分配内存
    caffe_gpu_set(nums_buffer_.count(), Dtype(0), counts);

    // NOLINT_NEXT_LINE(whitespace/operators)
    //前向计算数据
    AccuracyForwardWithPerClassGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(
          nthreads, 
          bottom_data, 
          bottom_label,
        acc_data, 
        counts, 
        outer_num_, dim, 
        inner_num_, 
        num_labels, 
        top_k_,
        has_ignore_label_, 
        ignore_label_
      );

    // get the overall accuracy
    Dtype acc;
    //对数据进行求和
    caffe_gpu_asum(bottom[0]->count(), acc_data, &acc);
    Dtype valid_count;
    caffe_gpu_asum(nums_buffer_.count(), counts, &valid_count);
    if (valid_count > 0) {
      top[0]->mutable_cpu_data()[0] = acc / valid_count;
    } else {
      top[0]->mutable_cpu_data()[0] = 0;
    }

    // get per-class accuracy
    //计算每个类别的准确率
    Dtype* per_class_acc = top[1]->mutable_cpu_data();
    //对每个标签进行计算
    for (int l = 0; l < num_labels; l++) {
      //计算正确的数目
      caffe_gpu_asum(nthreads, acc_data + l*nthreads, per_class_acc+l);
      //计算总次数
      caffe_gpu_asum(nthreads, counts + l*nthreads, &valid_count);
      if (valid_count > 0) {
        //计算准确率
        per_class_acc[l] /= valid_count;
      } else {
        per_class_acc[l] = 0;
      }
    }
  }
  // Clear scratch memory to prevent interfering with backward (see #6202).
  //清除内存
  caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_gpu_diff());
}

//重写反向计算函数
template <typename Dtype>
void AccuracyLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {  NOT_IMPLEMENTED;  }
}

INSTANTIATE_LAYER_GPU_FUNCS(AccuracyLayer);
}  // namespace caffe
