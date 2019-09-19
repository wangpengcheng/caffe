#ifndef CAFFE_SGD_SOLVERS_HPP_
#define CAFFE_SGD_SOLVERS_HPP_

#include <string>
#include <vector>

#include "caffe/solver.hpp"
/**
 * 参考链接：
 * 公式解析：https://blog.csdn.net/crazyquhezheng/article/details/79322388
 * https://blog.csdn.net/q375010308/article/details/47021779
 * 代码解析：http://www.voidcn.com/article/p-aqbavnah-bcx.html
 * http://kaizhao.net/blog/posts/momentum-caffe-pytorch/
 * https://blog.csdn.net/suixinsuiyuan33/article/details/69229605
 * /

namespace caffe {

/**
 * @brief Optimizes the parameters of a Net using
 *        stochastic gradient descent (SGD) with momentum.
 */
template <typename Dtype>
class SGDSolver : public Solver<Dtype> {//继承了solver公有类的方法，包括初始构造函数
 public:
  explicit SGDSolver(const SolverParameter& param)
      : Solver<Dtype>(param) { PreSolve(); }//构造函数,注意这里继承了solver的参数，并添加了PreSolver方法
  explicit SGDSolver(const string& param_file)
      : Solver<Dtype>(param_file) { PreSolve(); }
  virtual inline const char* type() const { return "SGD"; }

  const vector<shared_ptr<Blob<Dtype> > >& history() { return history_; }

  virtual void ApplyUpdate();
  Dtype GetLearningRate();

 protected:
  void PreSolve();//预处理函数，主要进行参数的获取和内存变量的初始化
  virtual void Normalize(int param_id);//标准化
  virtual void Regularize(int param_id);//正则化
  virtual void ComputeUpdateValue(int param_id, Dtype rate);//根据学习率更新所有的计算
  virtual void ClipGradients();//梯度修正
  virtual void SnapshotSolverState(const string& model_filename);//这个是闪照的一系列动作，主要是闪照的存储。
  virtual void SnapshotSolverStateToBinaryProto(const string& model_filename);
  virtual void SnapshotSolverStateToHDF5(const string& model_filename);
  virtual void RestoreSolverStateFromHDF5(const string& state_file);
  virtual void RestoreSolverStateFromBinaryProto(const string& state_file);
  // history maintains the historical momentum data.
  // update maintains update related data and is not needed in snapshots.
  // temp maintains other information that might be needed in computation
  //   of gradients/updates and is not needed in snapshots //历史保留历史动量数据。
   // update维护更新相关数据，快照中不需要。
   // temp维护计算中可能需要的其他信息
   //渐变/更新，快照中不需要
  //history维护旧的动量数据。update维护更新的相关数据，而且在snapshots中是不需要的。temp维护其他信息，这些信息可能是在计算梯度或者更新时需要的，而且在snapshots中是不需要的。
  vector<shared_ptr<Blob<Dtype> > > history_, update_, temp_;//

  DISABLE_COPY_AND_ASSIGN(SGDSolver);//禁止类的拷贝
};
//Nesterov 的加速梯度法（Nesterov’s accelerated gradient）作为凸优化中最理想的方法，其收敛速度非常快。
//参考链接：https://zhuanlan.zhihu.com/p/22810533；这里相对SGD方法优化的Momentum方法，更进一步进行了迭代优化。
template <typename Dtype>
class NesterovSolver : public SGDSolver<Dtype> {
 public:
  explicit NesterovSolver(const SolverParameter& param)
      : SGDSolver<Dtype>(param) {}
  explicit NesterovSolver(const string& param_file)
      : SGDSolver<Dtype>(param_file) {}
  virtual inline const char* type() const { return "Nesterov"; }

 protected:
  virtual void ComputeUpdateValue(int param_id, Dtype rate);

  DISABLE_COPY_AND_ASSIGN(NesterovSolver);
};
//自适应梯度（adaptive gradient）是基于梯度的优化方法

template <typename Dtype>
class AdaGradSolver : public SGDSolver<Dtype> {
 public:
  explicit AdaGradSolver(const SolverParameter& param)
      : SGDSolver<Dtype>(param) { constructor_sanity_check(); }
  explicit AdaGradSolver(const string& param_file)
      : SGDSolver<Dtype>(param_file) { constructor_sanity_check(); }
  virtual inline const char* type() const { return "AdaGrad"; }

 protected:
  virtual void ComputeUpdateValue(int param_id, Dtype rate);
  void constructor_sanity_check() {
    CHECK_EQ(0, this->param_.momentum())
        << "Momentum cannot be used with AdaGrad.";
  }

  DISABLE_COPY_AND_ASSIGN(AdaGradSolver);
};
//RMSprop是Tieleman在一次 Coursera课程演讲中提出来的，也是一种基于梯度的优化方法

template <typename Dtype>
class RMSPropSolver : public SGDSolver<Dtype> {
 public:
  explicit RMSPropSolver(const SolverParameter& param)
      : SGDSolver<Dtype>(param) { constructor_sanity_check(); }
  explicit RMSPropSolver(const string& param_file)
      : SGDSolver<Dtype>(param_file) { constructor_sanity_check(); }
  virtual inline const char* type() const { return "RMSProp"; }

 protected:
  virtual void ComputeUpdateValue(int param_id, Dtype rate);
  void constructor_sanity_check() {
    CHECK_EQ(0, this->param_.momentum())
        << "Momentum cannot be used with RMSProp.";
    CHECK_GE(this->param_.rms_decay(), 0)
        << "rms_decay should lie between 0 and 1.";
    CHECK_LT(this->param_.rms_decay(), 1)
        << "rms_decay should lie between 0 and 1.";
  }

  DISABLE_COPY_AND_ASSIGN(RMSPropSolver);
};
//AdaDelta基本思想是用一阶的方法，近似模拟二阶牛顿法。
template <typename Dtype>
class AdaDeltaSolver : public SGDSolver<Dtype> {
 public:
  explicit AdaDeltaSolver(const SolverParameter& param)
      : SGDSolver<Dtype>(param) { AdaDeltaPreSolve(); }
  explicit AdaDeltaSolver(const string& param_file)
      : SGDSolver<Dtype>(param_file) { AdaDeltaPreSolve(); }
  virtual inline const char* type() const { return "AdaDelta"; }

 protected:
  void AdaDeltaPreSolve();
  virtual void ComputeUpdateValue(int param_id, Dtype rate);

  DISABLE_COPY_AND_ASSIGN(AdaDeltaSolver);
};

/**
 * @brief AdamSolver, an algorithm for first-order gradient-based optimization
 *        of stochastic objective functions, based on adaptive estimates of
 *        lower-order moments. Described in [1].
 *
 * [1] D. P. Kingma and J. L. Ba, "ADAM: A Method for Stochastic Optimization."
 *     arXiv preprint arXiv:1412.6980v8 (2014).
 */
// Adam是一种基于梯度的优化方法
template <typename Dtype>
class AdamSolver : public SGDSolver<Dtype> {
 public:
  explicit AdamSolver(const SolverParameter& param)
      : SGDSolver<Dtype>(param) { AdamPreSolve();}
  explicit AdamSolver(const string& param_file)
      : SGDSolver<Dtype>(param_file) { AdamPreSolve(); }
  virtual inline const char* type() const { return "Adam"; }

 protected:
  void AdamPreSolve();
  virtual void ComputeUpdateValue(int param_id, Dtype rate);

  DISABLE_COPY_AND_ASSIGN(AdamSolver);
};

}  // namespace caffe

#endif  // CAFFE_SGD_SOLVERS_HPP_
