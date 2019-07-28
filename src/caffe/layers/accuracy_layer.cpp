#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/accuracy_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

//参数初始化函数
template <typename Dtype>
void AccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  top_k_ = this->layer_param_.accuracy_param().top_k();

  has_ignore_label_ =
    this->layer_param_.accuracy_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.accuracy_param().ignore_label();
  }
}
//注意这里的reshape函数
template <typename Dtype>
void AccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    //bottom[0]是前一层的输入 bottom[1]是label的输入,两者的商就是分类的映射总数
  CHECK_LE(top_k_, bottom[0]->count() / bottom[1]->count())
      << "top_k must be less than or equal to the number of classes.";
      //计算标签维度
  label_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.accuracy_param().axis());
//上面的check表示的意思为： 
//先结合上图理解outer_num与inner_num的数量，以及bottom[1]的结构。 
//如果C表示的是预测的种类的数量（就是label有多少种），那么label的数量有多少? 
//就是N*H*W，一个输入是H*W大，对于H*W的每个元素都有一个label值，所以，一个输入就会有H*W个label， 
//N个输入（一批），就会有N*H*W个label数量。 
//很多网络最后一层都是全连接层，即一张h*w的图片进去，到最后就成为了一列单个的元素，N张图进去就是变成N列（计算上来说）， 
//所以一般输入给accuracy层的bottom[0]都是N*C*1*1. 
//对于分类来说，这个特别好理解，每张图片对应一个label，那么就会有N个label

  //如果默认的情况就是label_axis_ = 1，outer_num_ = N；inner_num_ = W*H
  //输出数目为0到指定维度的数目
  outer_num_ = bottom[0]->count(0, label_axis_);
  //内部数目为剩下的维度数据
  inner_num_ = bottom[0]->count(label_axis_ + 1);
  //确认输入维度与输出分类是否相同，注意这里忽略了c的数据
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if label axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
  top[0]->Reshape(top_shape);
  //如果有两个top
  if (top.size() > 1) {
    // Per-class accuracy is a vector; 1 axes.
    //对预分类的维度进行改变,这里主要是为了方便后面对维度进行映射
    vector<int> top_shape_per_class(1);
    //默认为1的情况下，top_shape_per_class=C
    top_shape_per_class[0] = bottom[0]->shape(label_axis_);
    //重新设置第二个blob维度
    top[1]->Reshape(top_shape_per_class);
    //设置计数缓冲维度
    nums_buffer_.Reshape(top_shape_per_class);
  }
}
//前向计算函数
template <typename Dtype>
void AccuracyLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top
    ) {
  //初始化准确率
  Dtype accuracy = 0;
  //获取数据
  const Dtype* bottom_data = bottom[0]->cpu_data();
  //获取label
  const Dtype* bottom_label = bottom[1]->cpu_data();
  //计算维度dim=N*C*H*W/N= C*H*W
  const int dim = bottom[0]->count() / outer_num_;
  //计算标签数量默认是C
  const int num_labels = bottom[0]->shape(label_axis_);
  //检查top blob数量
  if (top.size() > 1) {
    //开始为nums_buffer_分配内存
    caffe_set(nums_buffer_.count(), Dtype(0), nums_buffer_.mutable_cpu_data());
    //设置第二个top blob分配内存
    caffe_set(top[1]->count(), Dtype(0), top[1]->mutable_cpu_data());
  }
  int count = 0;
  //遍历输入维度，注意这里是循环遍历每一个映射维度的映射关系，得到标签的准确度
  for (int i = 0; i < outer_num_; ++i) {//遍历n
    for (int j = 0; j < inner_num_; ++j) {//遍历wh
      //获取临时标签值
      const int label_value =static_cast<int>(bottom_label[i * inner_num_ + j]);
      //检查是否将该标签忽略
      if (has_ignore_label_ && label_value == ignore_label_) {
        continue;
      }
      //检查标签值是否为0
      DCHECK_GE(label_value, 0);
      //检查是否大于标签总数
      DCHECK_LT(label_value, num_labels);
      //如果两个top就将记录data中的标签分类中，即属于label的数量++
       // 记录每个类别的图像总数
      if (top.size() > 1) ++nums_buffer_.mutable_cpu_data()[label_value];
      //获取输入数据对于正确标签的预测概率
      const Dtype prob_of_true_class = bottom_data[i * dim+ label_value * inner_num_+ j];
      //统计正确分类的数量

      int num_better_predictions = -1;  // true_class also counts as "better"
      // Top-k accuracy 
      // top_k为取前k个最高评分（的预测标签）;这里是直接进行比较，不再排序
      //对于顶部的top_k个数据进行正确分类的统计
      for (int k = 0; k < num_labels && num_better_predictions < top_k_; ++k) {
        //查看大于prob_of_true_class的有多少
        //注意这里是对所有标签中的属于label_value，进行统计分类i*dim 是层，k*inner_num_是k*w*h,j是最终确定的值
        num_better_predictions +=(bottom_data[i * dim + k * inner_num_ + j] >= prob_of_true_class);
      }
      // check if there are less than top_k_ predictions
      //确认是否小于top_k_，即并不是每一个都大于prob_of_true_class
      if (num_better_predictions < top_k_) {
        //增加准确率,即验证正确的数量
        ++accuracy;
        //输出该类图像数量++
        if (top.size() > 1) ++top[1]->mutable_cpu_data()[label_value];
      }
      //增加计数，记录总的数量
      ++count;
    }
  }

  // LOG(INFO) << "Accuracy: " << accuracy;
  //输出准确度，注意这里是统计的总的准确度
  top[0]->mutable_cpu_data()[0] = (count == 0) ? 0 : (accuracy / count);
  if (top.size() > 1) {
    //循环获取每个类的准确度
    for (int i = 0; i < top[1]->count(); ++i) {
      //注意；top[1]是正确分类正确的数量，nums_buffer_是可能为这类图像的总数

      top[1]->mutable_cpu_data()[i] =nums_buffer_.cpu_data()[i] == 0 ? 0: top[1]->cpu_data()[i] / nums_buffer_.cpu_data()[i];
    }
  }
  // Accuracy layer should not be used as a loss function.
}

#ifdef CPU_ONLY
STUB_GPU(AccuracyLayer);
#endif

INSTANTIATE_CLASS(AccuracyLayer);
REGISTER_LAYER_CLASS(Accuracy);
//注意：cpu模式一label为主进行映射并，计算准确率，GPU是输入data为主

}  // namespace caffe
