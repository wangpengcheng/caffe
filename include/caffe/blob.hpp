#ifndef CAFFE_BLOB_HPP_
#define CAFFE_BLOB_HPP_

#include <algorithm>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/syncedmem.hpp"

// 设置最大的
const int kMaxBlobAxes = 32;

namespace caffe {

/**
 * @brief A wrapper around SyncedMemory holders serving as the basic
 *        computational unit through which Layer%s, Net%s, and Solver%s
 *        interact.
 *
 * TODO(dox): more thorough description.
 */
// 
template <typename Dtype>
class Blob {
 public:
  Blob()
       : data_(), diff_(), count_(0), capacity_(0) {}

  /// @brief Deprecated; use <code>Blob(const vector<int>& shape)</code>.
       //blob 的构造函数
  explicit Blob(const int num, const int channels, const int height,
      const int width);
  explicit Blob(const vector<int>& shape);

  /// @brief Deprecated; use <code>Reshape(const vector<int>& shape)</code>.
  void Reshape(const int num, const int channels, const int height,
      const int width);
  /**
   * @brief Change the dimensions of the blob, allocating new memory if
   *        necessary.
   *
   * This function can be called both to create an initial allocation
   * of memory, and to adjust the dimensions of a top blob during Layer::Reshape
   * or Layer::Forward. When changing the size of blob, memory will only be
   * reallocated if sufficient memory does not already exist, and excess memory
   * will never be freed.
   *
   * Note that reshaping an input blob and immediately calling Net::Backward is
   * an error; either Net::Forward or Net::Reshape need to be called to
   * propagate the new input shape to higher layers.
   */
  //函数主要是在Layer::Reshape被调用的时候，重新设置blob的大小。
  //如果还没有足够的内存，则只会重新分配内存，并且永远不会释放多余的内存。
  //请注意，重新整形输入blob并立即调用Net :: Backward是一个错误; 
  //需要调用Net :: Forward或Net :: Reshape将新输入形状传播到更高层。
  void Reshape(const vector<int>& shape);
  void Reshape(const BlobShape& shape);
  void ReshapeLike(const Blob& other);
 //将形状转换为string
  inline string shape_string() const {
    ostringstream stream;
    for (int i = 0; i < shape_.size(); ++i) {
      stream << shape_[i] << " ";
    }
    stream << "(" << count_ << ")";//输出数据总和
    return stream.str();
  }
  //返回形状
  inline const vector<int>& shape() const { return shape_; }
  /**
   * @brief Returns the dimension of the index-th axis (or the negative index-th
   *        axis from the end, if index is negative).
   *
   * @param index the axis index, which may be negative as it will be
   *        "canonicalized" using CanonicalAxisIndex.
   *        Dies on out of range index.
   */
  //索引轴索引，可能是负数，因为它将使用CanonicalAxisIndex进行“规范化”。 防止超出范围。
  inline int shape(int index) const {
    return shape_[CanonicalAxisIndex(index)];
  }
  //返回维度的大小
  inline int num_axes() const { return shape_.size(); }
  //返回数据的总计数
  inline int count() const { return count_; }

  /**
   * @brief Compute the volume of a slice; i.e., the product of dimensions
   *        among a range of axes.
   *
   * @param start_axis The first axis to include in the slice.
   *
   * @param end_axis The first axis to exclude from the slice.
   */
  //统计起始维度到终点维度的数据量，可以计算图片大小，shap(3)*shap(2); 总的通道数目shap(0)*shap(1)
  inline int count(int start_axis, int end_axis) const {
    //限定条件，检查输入的合法
    CHECK_LE(start_axis, end_axis);
    CHECK_GE(start_axis, 0);
    CHECK_GE(end_axis, 0);
    CHECK_LE(start_axis, num_axes());
    CHECK_LE(end_axis, num_axes());
    int count = 1;
    //统计数量
    for (int i = start_axis; i < end_axis; ++i) {
      count *= shape(i);
    }
    return count;
  }
  /**
   * @brief Compute the volume of a slice spanning from a particular first
   *        axis to the final axis.
   *
   * @param start_axis The first axis to include in the slice.
   */
  //起始维度到最后的数据总量
  inline int count(int start_axis) const {
    return count(start_axis, num_axes());
  }

  /**
   * @brief Returns the 'canonical' version of a (usually) user-specified axis,
   *        allowing for negative indexing (e.g., -1 for the last axis).
   *
   * @param axis_index the axis index.
   *        If 0 <= index < num_axes(), return index.
   *        If -num_axes <= index <= -1, return (num_axes() - (-index)),
   *        e.g., the last axis index (num_axes() - 1) if index == -1,
   *        the second to last if index == -2, etc.
   *        Dies on out of range index.
   */
  // 支持负数维度索引，负数表示从后往前，返回的是正确的维度索引（相当于将负数索引进行的转换）  
  inline int CanonicalAxisIndex(int axis_index) const {
    CHECK_GE(axis_index, -num_axes())//检查是否小于负数
        << "axis " << axis_index << " out of range for " << num_axes()
        << "-D Blob with shape " << shape_string();
    CHECK_LT(axis_index, num_axes())//检查是否大于正数
        << "axis " << axis_index << " out of range for " << num_axes()
        << "-D Blob with shape " << shape_string();
    if (axis_index < 0) {//返回逆序的坐标位置
      return axis_index + num_axes();
    }
    return axis_index;
  }

  /// @brief Deprecated legacy shape accessor num: use shape(0) instead.
  inline int num() const { return LegacyShape(0); }
  /// @brief Deprecated legacy shape accessor channels: use shape(1) instead.
  inline int channels() const { return LegacyShape(1); }
  /// @brief Deprecated legacy shape accessor height: use shape(2) instead.
  inline int height() const { return LegacyShape(2); }
  /// @brief Deprecated legacy shape accessor width: use shape(3) instead.
  inline int width() const { return LegacyShape(3); }
  //形状遗产，主要是为了兼容旧的接口
  inline int LegacyShape(int index) const {
    CHECK_LE(num_axes(), 4)
        << "Cannot use legacy accessors on Blobs with > 4 axes.";
    CHECK_LT(index, 4);
    CHECK_GE(index, -4);
    //超出范围直接返回
    if (index >= num_axes() || index < -num_axes()) {
      // Axis is out of range, but still in [0, 3] (or [-4, -1] for reverse
      // indexing) -- this special case simulates the one-padding used to fill
      // extraneous axes of legacy blobs.
      return 1;
    }
    return shape(index);
  }
  //获取数据的偏移
  inline int offset(const int n, const int c = 0, const int h = 0,
      const int w = 0) const {
    //检验数据是否输入合理
    CHECK_GE(n, 0);
    CHECK_LE(n, num());
    CHECK_GE(channels(), 0);
    CHECK_LE(c, channels());
    CHECK_GE(height(), 0);
    CHECK_LE(h, height());
    CHECK_GE(width(), 0);
    CHECK_LE(w, width());
    //返回数据下标
    return ((n * channels() + c) * height() + h) * width() + w;
  }
  //输入向量返回数据下标
  inline int offset(const vector<int>& indices) const {
    CHECK_LE(indices.size(), num_axes());
    int offset = 0;
    //提取维度相乘
    for (int i = 0; i < num_axes(); ++i) {
      offset *= shape(i);
      if (indices.size() > i) {
        CHECK_GE(indices[i], 0);
        CHECK_LT(indices[i], shape(i));
        offset += indices[i];
      }
    }
    return offset;
  }
  /**
   * @brief Copy from a source Blob.
   *
   * @param source the Blob to copy from
   * @param copy_diff if false, copy the data; if true, copy the diff
   * @param reshape if false, require this Blob to be pre-shaped to the shape
   *        of other (and die otherwise); if true, Reshape this Blob to other's
   *        shape if necessary
   */
  //从其他的实例中拷贝数据
  void CopyFrom(const Blob<Dtype>& source, bool copy_diff = false,
      bool reshape = false);

  //获取前向传播数据指针
  inline Dtype data_at(const int n, const int c, const int h,
      const int w) const {
    return cpu_data()[offset(n, c, h, w)];
  }

  //后向传播数据
  inline Dtype diff_at(const int n, const int c, const int h,
      const int w) const {
    return cpu_diff()[offset(n, c, h, w)];
  }
  inline Dtype data_at(const vector<int>& index) const {
    return cpu_data()[offset(index)];
  }

  inline Dtype diff_at(const vector<int>& index) const {
    return cpu_diff()[offset(index)];
  }
  //获取前向传播数据
  inline const shared_ptr<SyncedMemory>& data() const {
    CHECK(data_);
    return data_;
  }
  //获取后向传播数据
  inline const shared_ptr<SyncedMemory>& diff() const {
    CHECK(diff_);
    return diff_;
  }
  //------ 基本数据的存取类 start ------
  const Dtype* cpu_data() const;
  void set_cpu_data(Dtype* data);
  const int* gpu_shape() const;
  const Dtype* gpu_data() const;
  void set_gpu_data(Dtype* data);
  const Dtype* cpu_diff() const;
  const Dtype* gpu_diff() const;
  //------ 基本数据的存取类 end ------
  // 获取多的数据
  Dtype* mutable_cpu_data();
  Dtype* mutable_gpu_data();
  Dtype* mutable_cpu_diff();
  Dtype* mutable_gpu_diff();
  //更新数据
  void Update();//应该是为了计算
  //从BlobProto 中提取文件
  void FromProto(const BlobProto& proto, bool reshape = true);
  //将数据转换为 proto
  void ToProto(BlobProto* proto, bool write_diff = false) const;

  /// @brief Compute the sum of absolute values (L1 norm) of the data.
  //向前一维向量绝对值之和
  Dtype asum_data() const;
  /// @brief Compute the sum of absolute values (L1 norm) of the diff.
  Dtype asum_diff() const;
  //平方和
  /// @brief Compute the sum of squares (L2 norm squared) of the data.
  Dtype sumsq_data() const;
  /// @brief Compute the sum of squares (L2 norm squared) of the diff.
  Dtype sumsq_diff() const;

  /// @brief Scale the blob data by a constant factor.
  //按常数因子缩放blob数据。
  void scale_data(Dtype scale_factor);
  /// @brief Scale the blob diff by a constant factor.
  void scale_diff(Dtype scale_factor);

  /**
   * @brief Set the data_ shared_ptr to point to the SyncedMemory holding the
   *        data_ of Blob other -- useful in Layer%s which simply perform a copy
   *        in their Forward pass.
   *
   * This deallocates the SyncedMemory holding this Blob's data_, as
   * shared_ptr calls its destructor when reset with the "=" operator.
   * 将别的blob的data和响应的diff指针给这个Blob，实现数据的共享。 
   * 同时需要注意的是这个操作会引起这个Blob里面的SyncedMemory被释放， 
   * 因为shared_ptr指针被用=重置的时候回调用响应的析构器。 
   */
  void ShareData(const Blob& other);
  /**
   * @brief Set the diff_ shared_ptr to point to the SyncedMemory holding the
   *        diff_ of Blob other -- useful in Layer%s which simply perform a copy
   *        in their Forward pass.
   *
   * This deallocates the SyncedMemory holding this Blob's diff_, as
   * shared_ptr calls its destructor when reset with the "=" operator.
   */
  void ShareDiff(const Blob& other);

  bool ShapeEquals(const BlobProto& other);

 protected:
  shared_ptr<SyncedMemory> data_;//向前传播数据
  shared_ptr<SyncedMemory> diff_;//向后传播数据
  shared_ptr<SyncedMemory> shape_data_;//旧形状数据
  vector<int> shape_;//新形状数据，维度向量，一般是4个
  int count_;//所有维度相乘，即数据的总的个数
  int capacity_;//旧容量

  DISABLE_COPY_AND_ASSIGN(Blob);//禁止拷贝和拥有
};  // class Blob

}  // namespace caffe

#endif  // CAFFE_BLOB_HPP_
