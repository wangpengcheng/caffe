#ifndef CAFFE_NET_HPP_
#define CAFFE_NET_HPP_

#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Connects Layer%s together into a directed acyclic graph (DAG)
 *        specified by a NetParameter.
 *
 * TODO(dox): more thorough description.
 */
template <typename Dtype>
class Net {
 public:
  explicit Net(const NetParameter& param);//基本构造函数
  explicit Net(const string& param_file, Phase phase,
      const int level = 0, const vector<string>* stages = NULL);//加状态的构造函数
  virtual ~Net() {}

  /// @brief Initialize a network with a NetParameter.
  void Init(const NetParameter& param);//初始化函数

  /**
   * @brief Run Forward and return the result.
   *
   */
  const vector<Blob<Dtype>*>& Forward(Dtype* loss = NULL);//前向计算函数
  /// @brief DEPRECATED; use Forward() instead.//旧函数已经被forward取代
  const vector<Blob<Dtype>*>& ForwardPrefilled(Dtype* loss = NULL) {
    LOG_EVERY_N(WARNING, 1000) << "DEPRECATED: ForwardPrefilled() "
        << "will be removed in a future version. Use Forward().";
    return Forward(loss);
  }

  /**
   * The From and To variants of Forward and Backward operate on the
   * (topological) ordering by which the net is specified. For general DAG
   * networks, note that (1) computing from one layer to another might entail
   * extra computation on unrelated branches, and (2) computation starting in
   * the middle may be incorrect if all of the layers of a fan-in are not
   * included.Forward和Backward的From和To变体对指定网络的（拓扑）排序进行操作。 对于一般的DAG网络，请注意（1）从一层到另一层的计算可能需要在不相关的分支上进行额外的计算，以及（2）如果不包括扇入的所有层，则从中间开始的计算可能是不正确的。
   * 
   */
  Dtype ForwardFromTo(int start, int end);
  Dtype ForwardFrom(int start);
  Dtype ForwardTo(int end);
  /// @brief DEPRECATED; set input blobs then use Forward() instead.
  const vector<Blob<Dtype>*>& Forward(const vector<Blob<Dtype>* > & bottom,
      Dtype* loss = NULL);

  /**
   * @brief Zeroes out the diffs of all net parameters.
   *        Should be run before Backward.
   */
  void ClearParamDiffs();//将所有参数置0，应该在开始run之前运行

  /**
   * The network backward should take no input and output, since it solely
   * computes the gradient w.r.t the parameters, and the data has already been
   * provided during the forward pass.
   */
  void Backward();//反向计算函数
  void BackwardFromTo(int start, int end);//指定反向计算的范围
  void BackwardFrom(int start);//
  void BackwardTo(int end);//

  /**
   * @brief Reshape all layers from bottom to top.
   *
   * This is useful to propagate changes to layer sizes without running
   * a forward pass, e.g. to compute output feature size.
   */
  void Reshape();
//反向计算函数
  Dtype ForwardBackward() {
    Dtype loss;
    Forward(&loss);//前向计算
    Backward();//反向计算
    return loss;
  }

  /// @brief Updates the network weights based on the diff values computed.
  void Update();//更新权重和误差值
  /**
   * @brief Shares weight data of owner blobs with shared blobs.
   *
   * Note: this is called by Net::Init, and thus should normally not be
   * called manually.
   */
  void ShareWeights();//权值共享

  /**
   * @brief For an already initialized net, implicitly copies (i.e., using no
   *        additional memory) the pre-trained layers from another Net.
   */
  void ShareTrainedLayersWith(const Net* other);//共享训练好的权值
  // For an already initialized net, CopyTrainedLayersFrom() copies the already
  // trained layers from another net parameter instance.
  /**
   * @brief For an already initialized net, copies the pre-trained layers from
   *        another Net.
   */
  void CopyTrainedLayersFrom(const NetParameter& param);//从已经训练好的权值中读取参数
  void CopyTrainedLayersFrom(const string& trained_filename);
  void CopyTrainedLayersFromBinaryProto(const string& trained_filename);
  void CopyTrainedLayersFromHDF5(const string& trained_filename);
  /// @brief Writes the net to a proto.
  void ToProto(NetParameter* param, bool write_diff = false) const;//转化为proto
  /// @brief Writes the net to an HDF5 file.
  void ToHDF5(const string& filename, bool write_diff = false) const;//转化为hdf5

  /// @brief returns the network name.
  inline const string& name() const { return name_; }//net name
  /// @brief returns the layer names
  inline const vector<string>& layer_names() const { return layer_names_; }//layer names
  /// @brief returns the blob names
  inline const vector<string>& blob_names() const { return blob_names_; }//blob names
  /// @brief returns the blobs
  inline const vector<shared_ptr<Blob<Dtype> > >& blobs() const {
    return blobs_; //返回blobs
  }
  /// @brief returns the layers
  inline const vector<shared_ptr<Layer<Dtype> > >& layers() const {
    return layers_;//返回layers
  }
  /// @brief returns the phase: TRAIN or TEST
  inline Phase phase() const { return phase_; }//返回shase函数
  /**
   * @brief returns the bottom vecs for each layer -- usually you won't
   *        need this unless you do per-layer checks such as gradients.
   */
  inline const vector<vector<Blob<Dtype>*> >& bottom_vecs() const {
    return bottom_vecs_;
  }
  /**
   * @brief returns the top vecs for each layer -- usually you won't
   *        need this unless you do per-layer checks such as gradients.
   */
  inline const vector<vector<Blob<Dtype>*> >& top_vecs() const {
    return top_vecs_;
  }
  /// @brief returns the ids of the top blobs of layer i，返回layer i的 top blobs
  inline const vector<int> & top_ids(int i) const {
    CHECK_GE(i, 0) << "Invalid layer id";
    CHECK_LT(i, top_id_vecs_.size()) << "Invalid layer id";
    return top_id_vecs_[i];
  }
  /// @brief returns the ids of the bottom blobs of layer i ;返回layer i的 bottom blobs
  inline const vector<int> & bottom_ids(int i) const {
    CHECK_GE(i, 0) << "Invalid layer id";
    CHECK_LT(i, bottom_id_vecs_.size()) << "Invalid layer id";
    return bottom_id_vecs_[i];
  }
  inline const vector<vector<bool> >& bottom_need_backward() const {
    return bottom_need_backward_;//返回是否需要前向计算
  }//blob 的loss权重
  inline const vector<Dtype>& blob_loss_weights() const {
    return blob_loss_weights_;
  }//是否需要前向计算
  inline const vector<bool>& layer_need_backward() const {
    return layer_need_backward_;
  }
  /// @brief returns the parameters,返回相关的参数
  inline const vector<shared_ptr<Blob<Dtype> > >& params() const {
    return params_;
  }
  inline const vector<Blob<Dtype>*>& learnable_params() const {
    return learnable_params_;//学习参数
  }
  /// @brief returns the learnable parameter learning rate multipliers
  //返回可学习参数学习速率乘数;即学习参数权重
  inline const vector<float>& params_lr() const { return params_lr_; }
  //是否存在学习参数权重
  inline const vector<bool>& has_params_lr() const { return has_params_lr_; }
  /// @brief returns the learnable parameter decay multipliers
  //返回可学习参数衰减乘数
  inline const vector<float>& params_weight_decay() const {
    return params_weight_decay_;
  }
  //是否存在学习参数衰减
  inline const vector<bool>& has_params_decay() const {
    return has_params_decay_;
  }
  //参数名称和index索引
  const map<string, int>& param_names_index() const {
    return param_names_index_;
  }
  //参数所有者
  inline const vector<int>& param_owners() const { return param_owners_; }
  //参数显示名称
  inline const vector<string>& param_display_names() const {
    return param_display_names_;
  }
  /// @brief Input and output blob numbers
  //输入输出blob的数目
  inline int num_inputs() const { return net_input_blobs_.size(); }
  inline int num_outputs() const { return net_output_blobs_.size(); }
  inline const vector<Blob<Dtype>*>& input_blobs() const {
    return net_input_blobs_;
  }
  inline const vector<Blob<Dtype>*>& output_blobs() const {
    return net_output_blobs_;
  }
  inline const vector<int>& input_blob_indices() const {
    return net_input_blob_indices_;
  }
  inline const vector<int>& output_blob_indices() const {
    return net_output_blob_indices_;
  }
  //是否含有blob
  bool has_blob(const string& blob_name) const;
  //通过名字查找Blob
  const shared_ptr<Blob<Dtype> > blob_by_name(const string& blob_name) const;
  bool has_layer(const string& layer_name) const;
  const shared_ptr<Layer<Dtype> > layer_by_name(const string& layer_name) const;
  //设置debug信息
  void set_debug_info(const bool value) { debug_info_ = value; }

  // Helpers for Init.
  /**
   * @brief Remove layers that the user specified should be excluded given the current
   *        phase, level, and stage.
   */
  //根据状态对网络进行筛选,根据参数移除指定的层
  static void FilterNet(const NetParameter& param,
      NetParameter* param_filtered);
  /// @brief return whether NetState state meets NetStateRule rule
  //返回NetStat状态是否符合NetStat规则规则
  static bool StateMeetsRule(const NetState& state, const NetStateRule& rule,
      const string& layer_name);

  // Invoked at specific points during an iteration
  //回调函数 在迭代期间在特定点调用，这个主要是多GPU训练时使用
  class Callback {
   protected:
    virtual void run(int layer) = 0;

    template <typename T>
    friend class Net;
  };
  const vector<Callback*>& before_forward() const { return before_forward_; }
  void add_before_forward(Callback* value) {
    before_forward_.push_back(value);
  }
  const vector<Callback*>& after_forward() const { return after_forward_; }
  void add_after_forward(Callback* value) {
    after_forward_.push_back(value);
  }
  const vector<Callback*>& before_backward() const { return before_backward_; }
  void add_before_backward(Callback* value) {
    before_backward_.push_back(value);
  }
  const vector<Callback*>& after_backward() const { return after_backward_; }
  void add_after_backward(Callback* value) {
    after_backward_.push_back(value);
  }

 protected:
  // Helpers for Init. Init辅助函数
  /// @brief Append a new top blob to the net.
  //为net添加一个新的top blob
  void AppendTop(const NetParameter& param, 
                  const int layer_id,
                 const int top_id, set<string>* available_blobs,
                 map<string, int>* blob_name_to_idx);
  /// @brief Append a new bottom blob to the net.
  int AppendBottom(const NetParameter& param, const int layer_id,
                   const int bottom_id, set<string>* available_blobs,
                   map<string, int>* blob_name_to_idx);
  /// @brief Append a new parameter blob to the net.
  //通过参数添加Blob
  void AppendParam(const NetParameter& param, const int layer_id,
                   const int param_id);

  /// @brief Helper for displaying debug info in Forward.
  void ForwardDebugInfo(const int layer_id);
  /// @brief Helper for displaying debug info in Backward.
  void BackwardDebugInfo(const int layer_id);
  /// @brief Helper for displaying debug info in Update.
  void UpdateDebugInfo(const int param_id);

  /// @brief The network name
  string name_;
  /// @brief The phase: TRAIN or TEST
  Phase phase_;
  /// @brief Individual layers in the net
  vector<shared_ptr<Layer<Dtype> > > layers_;
  vector<string> layer_names_;
  map<string, int> layer_names_index_;//layers 索引
  vector<bool> layer_need_backward_;//是否需要反向计算
  /// @brief the blobs storing intermediate results between the layer.
  vector<shared_ptr<Blob<Dtype> > > blobs_;//blob 
  vector<string> blob_names_;
  map<string, int> blob_names_index_;//blob names索引
  vector<bool> blob_need_backward_;//是否需要反向计算
  /// bottom_vecs stores the vectors containing the input for each layer.
  /// They don't actually host the blobs (blobs_ does), so we simply store
  /// pointers.
  vector<vector<Blob<Dtype>*> > bottom_vecs_;//存储每一个layer input  bottom blobs 指针
  vector<vector<int> > bottom_id_vecs_;//存储每一个bottom blobs id
  vector<vector<bool> > bottom_need_backward_;//是否需要反向计算
  /// top_vecs stores the vectors containing the output for each layer
  vector<vector<Blob<Dtype>*> > top_vecs_;//存储每一个layer output top blobs 指针
  vector<vector<int> > top_id_vecs_;//存储每一个layer output top blobs 指针
  /// Vector of weight in the loss (or objective) function of each net blob,
  /// indexed by blob_id.
  vector<Dtype> blob_loss_weights_;// layer 的loss函数值
  //这些主要都是为了输出信息使用
  vector<vector<int> > param_id_vecs_;//参数的 vector，大小由layer决定
  vector<int> param_owners_;//参数所有者
  vector<string> param_display_names_;//参数显示名称
  vector<pair<int, int> > param_layer_indices_;//
  map<string, int> param_names_index_;//参数名称索引
  /// blob indices for the input and the output of the net 输入和网络输出的blob索引
  vector<int> net_input_blob_indices_;
  vector<int> net_output_blob_indices_;
  vector<Blob<Dtype>*> net_input_blobs_;
  vector<Blob<Dtype>*> net_output_blobs_;
  /// The parameters in the network.
  vector<shared_ptr<Blob<Dtype> > > params_;//相关参数，这里主要用来计算层中定义的额外参数
  vector<Blob<Dtype>*> learnable_params_;//学习参数用Blob来进行存储
  /**
   * The mapping from params_ -> learnable_params_: we have
   * learnable_param_ids_.size() == params_.size(),
   * and learnable_params_[learnable_param_ids_[i]] == params_[i].get()
   * if and only if params_[i] is an "owner"; otherwise, params_[i] is a sharer
   * and learnable_params_[learnable_param_ids_[i]] gives its owner.
   * 
   * 来自params_ - > learnable_params_的映射：
   * 我们有learnable_param_ids_.size（）== params_.size（）
   * 和learnable_params_ [learnable_param_ids_ [i]] == params_ [i] .get（）
   * 当且仅当params_ [i]是“所有者”时; 否则，params_ [i]是一个共享者，
   * learnable_params_ [learnable_param_ids_ [i]]给它的主人。
   */
  vector<int> learnable_param_ids_; //学习参数索引
  /// the learning rate multipliers for learnable_params_
  //learnable_params_的学习率乘数
  vector<float> params_lr_;
  vector<bool> has_params_lr_;//是否需要这个主要是学习率设置的 ；https://www.cnblogs.com/JZ-Ser/p/7150950.html
  /// the weight decay multipliers for learnable_params_ ；learnable_params_的重量衰减乘数
  vector<float> params_weight_decay_;//权重参数
  vector<bool> has_params_decay_;//是否衰减
  /// The bytes of memory used by this net
  size_t memory_used_;//使用的内存量,这里主要是对输入数据的维度进行的统计
  /// Whether to compute and display debug info for the net.
  bool debug_info_;
  // Callbacks
  //回调函数；在进行前向计算后使用
  vector<Callback*> before_forward_;
  vector<Callback*> after_forward_;
  vector<Callback*> before_backward_;
  vector<Callback*> after_backward_;

DISABLE_COPY_AND_ASSIGN(Net);
};


}  // namespace caffe

#endif  // CAFFE_NET_HPP_
