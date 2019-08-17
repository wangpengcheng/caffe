#include <algorithm>
#include <map>
#include <sstream>
#include <string>
#include <utility>

#include "caffe/common.hpp"
#include "caffe/util/insert_splits.hpp"

namespace caffe {

void InsertSplits(const NetParameter& param, NetParameter* param_split) {
  // Initialize by copying from the input NetParameter.
	//通过输入的参数来，进行初始化
  param_split->CopyFrom(param);
  //清楚原有数据
  param_split->clear_layer();
  map<string, pair<int, int> > blob_name_to_last_top_idx;//blob名称与坐标的映射，比如Blob data是第i个layer中的第j个，就是<data, pair<i, j> >;主要是这种方式表示映射连接
  map<pair<int, int>, pair<int, int> > bottom_idx_to_source_top_idx;//blob 前后逻辑位置的映射
  map<pair<int, int>, int> top_idx_to_bottom_count;//统计逻辑 top blob需要向下传播的数量
  map<pair<int, int>, float> top_idx_to_loss_weight;//不同逻辑的Blobloss权值
  map<pair<int, int>, int> top_idx_to_bottom_split_idx;//记录被下一层的x个层共享
  map<int, string> layer_idx_to_layer_name;//layer编号到名字的映射
  for (int i = 0; i < param.layer_size(); ++i) {//遍历没一层进行数据设置
    const LayerParameter& layer_param = param.layer(i);//获取这层的layer参数
    layer_idx_to_layer_name[i] = layer_param.name();//这个层的名字
    for (int j = 0; j < layer_param.bottom_size(); ++j) {//对层中的底部blob进行遍历
      const string& blob_name = layer_param.bottom(j);//获取bottom blob 名称
      if (blob_name_to_last_top_idx.find(blob_name) ==blob_name_to_last_top_idx.end()
      ) {//查看是否已经存在blob这个名字,不存在则输出错误信息
        LOG(FATAL) << "Unknown bottom blob '" << blob_name << "' (layer '"
                   << layer_param.name() << "', bottom index " << j << ")";
      }
      const pair<int, int>& bottom_idx = make_pair(i, j);//获取底部blob编号
      const pair<int, int>& top_idx = blob_name_to_last_top_idx[blob_name];//获取这个blob对应前一层的top blob坐标
      bottom_idx_to_source_top_idx[bottom_idx] = top_idx;//映射连接，这里就将blob前后的两个层之间连接起来了，中间的数据传递中间站还是只有一个
      ++top_idx_to_bottom_count[top_idx];//前一层top blob 向下的逻辑数量+1
    }
    for (int j = 0; j < layer_param.top_size(); ++j) {//对层中的顶部bottom进行遍历
      const string& blob_name = layer_param.top(j);//获取top_blob名称
      blob_name_to_last_top_idx[blob_name] = make_pair(i, j);//获取top blob 的逻辑坐标
    }
    // A use of a top blob as a loss should be handled similarly to the use of
    // a top blob as a bottom blob to another layer.
	//使用顶部blob作为损失应该与使用顶部blob作为底部blob到另一层的方式类似地进行处理。
  //这一层的top blob 是下一层的bottom blob；这里主要是看有没有loss_weight参数，也就是loss层
    const int last_loss =std::min(layer_param.loss_weight_size(), layer_param.top_size());//
    for (int j = 0; j < last_loss; ++j) {//遍历loss
      const string& blob_name = layer_param.top(j);//获取顶部blob名称
      const pair<int, int>& top_idx = blob_name_to_last_top_idx[blob_name];//获取top blob坐标
      top_idx_to_loss_weight[top_idx] = layer_param.loss_weight(j);//获取逻辑坐标i,j对应的权重
      if (top_idx_to_loss_weight[top_idx]) {//如过权重存在且不为0，这里主要是针对多个loss目标时的权重衡量
        ++top_idx_to_bottom_count[top_idx];//权重计数++
      }
    }
  }
  for (int i = 0; i < param.layer_size(); ++i) {//再次对所有的layer进行遍历
    LayerParameter* layer_param = param_split->add_layer();//添加新的参数层
    layer_param->CopyFrom(param.layer(i));//拷贝层参数
    // Replace any shared bottom blobs with split layer outputs.//将共享的bottom blob 用split进行替换
    for (int j = 0; j < layer_param->bottom_size(); ++j) {
      const pair<int, int>& top_idx =
          bottom_idx_to_source_top_idx[make_pair(i, j)];//获取前一层的top bottom 逻辑坐标
      const int split_count = top_idx_to_bottom_count[top_idx];//获取被共享的次数
      if (split_count > 1) {//存在被下一层的两个layer共享
        const string& layer_name = layer_idx_to_layer_name[top_idx.first];//获取前一层的layer名称
        const string& blob_name = layer_param->bottom(j);//获取被共享的blob层的名称
        layer_param->set_bottom(j, SplitBlobName(layer_name,
            blob_name, top_idx.second, top_idx_to_bottom_split_idx[top_idx]++));//设置当前层中blob的名字,设置这个blob的名称为blob_name_top_idx.second_split_(top_idx_to_bottom_split_idx[top_idx]++)
      }
    }
    // Create split layer for any top blobs used by other layer as bottom
    // blobs more than once.//对被多次引用的top blob创建拆分层
    for (int j = 0; j < layer_param->top_size(); ++j) {
      const pair<int, int>& top_idx = make_pair(i, j);//获取顶部逻辑坐标
      const int split_count = top_idx_to_bottom_count[top_idx];//查看他被使用的次数
      if (split_count > 1) {//使用次数大于1
        const string& layer_name = layer_idx_to_layer_name[i];//获取当前层的名字
        const string& blob_name = layer_param->top(j);//获取top blob的名字
        LayerParameter* split_layer_param = param_split->add_layer();//添加层参数，并返回层参数指针
        const float loss_weight = top_idx_to_loss_weight[top_idx];//获取top的loss值
        ConfigureSplitLayer(layer_name, blob_name, j, split_count,
            loss_weight, split_layer_param);
        if (loss_weight) {
          layer_param->clear_loss_weight();
          top_idx_to_bottom_split_idx[top_idx]++;//底部共享层计数+1
        }
      }
    }
  }
}
//重新设置layer层参数
void ConfigureSplitLayer(const string& layer_name, const string& blob_name,
    const int blob_idx, const int split_count, const float loss_weight,
    LayerParameter* split_layer_param) {
  split_layer_param->Clear();//清楚原有参数
  split_layer_param->add_bottom(blob_name);//添加blob名称
  split_layer_param->set_name(SplitLayerName(layer_name, blob_name, blob_idx));//更改名称
  split_layer_param->set_type("Split");//设置类型为Split
  for (int k = 0; k < split_count; ++k) {//遍历被共享的次数
    split_layer_param->add_top(
        SplitBlobName(layer_name, blob_name, blob_idx, k));//添加top blob 名称
    if (loss_weight) {//如果loss weight不为0
      if (k == 0) {
        split_layer_param->add_loss_weight(loss_weight);//添加权重
      } else {
        split_layer_param->add_loss_weight(0);//否则权重设置为0
      }
    }
  }
}
//一样的重命名函数
string SplitLayerName(const string& layer_name, const string& blob_name,
    const int blob_idx) {
  ostringstream split_layer_name;
  split_layer_name << blob_name << "_" << layer_name << "_" << blob_idx
      << "_split";
  return split_layer_name.str();
}
//根据分割参数创建新blob ;例如 conv2，pool1，0，1 ；返回名称就是pool1_conv2_0_split_1
string SplitBlobName(const string& layer_name, const string& blob_name,
    const int blob_idx, const int split_idx) {
  ostringstream split_blob_name;
  split_blob_name << blob_name << "_" << layer_name << "_" << blob_idx
      << "_split_" << split_idx;
  return split_blob_name.str();
}

}  // namespace caffe
