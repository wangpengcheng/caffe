#ifndef _CAFFE_UTIL_INSERT_SPLITS_HPP_
#define _CAFFE_UTIL_INSERT_SPLITS_HPP_

#include <string>

#include "caffe/proto/caffe.pb.h"

namespace caffe {

// Copy NetParameters with SplitLayers added to replace any shared bottom
// blobs with unique bottom blobs provided by the SplitLayer.
//添加了使用SplitLayers复制NetParameters以使用SplitLayer提供的唯一底部blob替换任何共享底部blob。
//主要是为了blob数据层之间的数据交换和共享
//插入参数
void InsertSplits(const NetParameter& param, NetParameter* param_split);
//匹配参数层
void ConfigureSplitLayer(const string& layer_name, const string& blob_name,
    const int blob_idx, const int split_count, const float loss_weight,
    LayerParameter* split_layer_param);
//更换layername
string SplitLayerName(const string& layer_name, const string& blob_name,
    const int blob_idx);
//更换blob名称
string SplitBlobName(const string& layer_name, const string& blob_name,
    const int blob_idx, const int split_idx);

}  // namespace caffe

#endif  // CAFFE_UTIL_INSERT_SPLITS_HPP_
