#ifndef CAFFE_COMMON_HPP_
#define CAFFE_COMMON_HPP_

#include <boost/shared_ptr.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <climits>
#include <cmath>
#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <map>
#include <set>
#include <sstream> //字符流
#include <string>
#include <utility>  // pair 结构体链接两个函数
#include <vector>

//使用设备声明文件
#include "caffe/util/device_alternate.hpp"

// Convert macro to string
//将宏指令转换为string
#define STRINGIFY(m) #m
#define AS_STRING(m) STRINGIFY(m) // #m

// gflags 2.1 issue: namespace google was changed to gflags without warning.
// Luckily we will be able to use GFLAGS_GFLAGS_H_ to detect if it is version
// 2.1. If yes, we will add a temporary solution to redirect the namespace.
// TODO(Yangqing): Once gflags solves the problem in a more elegant way, let's
// remove the following hack.
//使用google flags
#ifndef GFLAGS_GFLAGS_H_ 
namespace gflags = google;
#endif  
// GFLAGS_GFLAGS_H_
//重载类的拷贝函数，将其变为私有； 禁止类的拷贝和和拷贝初始化有操作;c++11建议更改为=delete
// Disable the copy and assignment operator for a class.
#define DISABLE_COPY_AND_ASSIGN(classname) \
private:\
  classname(const classname&);\
  classname& operator=(const classname&)
//用double和float实例化一个类
// Instantiate a class with float and double specifications.
//预定义模板类，方便初始化
#define INSTANTIATE_CLASS(classname) \
  char gInstantiationGuard##classname; \
  template class classname<float>; \
  template class classname<double>
//声明网路向前计算函数
#define INSTANTIATE_LAYER_GPU_FORWARD(classname) \
  template void classname<float>::Forward_gpu( \
      const std::vector<Blob<float>*>& bottom, \
      const std::vector<Blob<float>*>& top); \
  template void classname<double>::Forward_gpu( \
      const std::vector<Blob<double>*>& bottom, \
      const std::vector<Blob<double>*>& top);
//申明网络向后计算
#define INSTANTIATE_LAYER_GPU_BACKWARD(classname) \
  template void classname<float>::Backward_gpu( \
      const std::vector<Blob<float>*>& top, \
      const std::vector<bool>& propagate_down, \
      const std::vector<Blob<float>*>& bottom); \
  template void classname<double>::Backward_gpu( \
      const std::vector<Blob<double>*>& top, \
      const std::vector<bool>& propagate_down, \
      const std::vector<Blob<double>*>& bottom)

//使用预定义，方便声明类的向前和向后计算，主要还是为了预定义，减少代码使用
#define INSTANTIATE_LAYER_GPU_FUNCS(classname) \
  INSTANTIATE_LAYER_GPU_FORWARD(classname); \
  INSTANTIATE_LAYER_GPU_BACKWARD(classname)

// A simple macro to mark codes that are not implemented, so that when the code
// is executed we will see a fatal log.
//简单的声明宏指令
#define NOT_IMPLEMENTED LOG(FATAL) << "Not Implemented Yet"

// See PR #1236
//声明Mat类
namespace cv { class Mat; }

namespace caffe {
//使用boost share_ptr 因为cuda不支持C++11
//ps cuda 10早就支持c++11和C++14了
// We will use the boost shared_ptr instead of the new C++11 one mainly
// because cuda does not work (at least now) well with C++11 features.
using boost::shared_ptr;

// Common functions and classes from std that caffe often uses.
using std::fstream;
using std::ios;
using std::isnan;
using std::isinf;
using std::iterator;
using std::make_pair;
using std::map;
using std::ostringstream;
using std::pair;
using std::set;
using std::string;
using std::stringstream;
using std::vector;

//全局初始化函数
// A global initialization function that you should call in your main function.
// Currently it initializes google flags and google logging.
//主要是为了初始化google flag

void GlobalInit(int* pargc, char*** pargv);

// A singleton class to hold common caffe stuff, such as the handler that
// caffe is going to use for cublas, curand, etc.
//用于保存常见caffe内容的单例类，例如处理程序caffe将用于cublas，curand等。
class Caffe {
 public:
  ~Caffe();

  // Thread local context for Caffe. Moved to common.cpp instead of
  // including boost/thread.hpp to avoid a boost/NVCC issues (#1009, #1010)
  // on OSX. Also fails on Linux with CUDA 7.0.18.
  //获取线程本地上下文，已经移动到common.cpp 代替boost/thread.hpp，防止与cuda的冲突
  //设置静态上下文
  static Caffe& Get(); //获得指向caffe的静态指针地址
  //选择模式：CPU,GPU
  enum Brew { CPU, GPU };

  // This random number generator facade hides boost and CUDA rng
  // implementation from one another (for cross-platform compatibility).

  //这个随机数生成器外观隐藏了boost和CUDA rng实现（用于跨平台兼容性）。
  class RNG {
   public:
    RNG();
    explicit RNG(unsigned int seed);
    //拷贝构造函数
    explicit RNG(const RNG&);
    //重载操作符
    RNG& operator=(const RNG&);
    //获取外观类
    void* generator();
   private:
    //定义类
    class Generator;
    //使用类的共享指针
    shared_ptr<Generator> generator_;
  };

  // Getters for boost rng, curand, and cublas handles
  //内联函数获取随机数据流
  inline static RNG& rng_stream() {
    //如果没有随机数据，则生成
    if (!Get().random_generator_) {
      Get().random_generator_.reset(new RNG());
    }
    //返回随机数据
    return *(Get().random_generator_);
  }
#ifndef CPU_ONLY
  //获取GPU句柄
  inline static cublasHandle_t cublas_handle() { return Get().cublas_handle_; }
  inline static curandGenerator_t curand_generator() {
    return Get().curand_generator_;
  }
#endif

  // Returns the mode: running on CPU or GPU.
  inline static Brew mode() { return Get().mode_; }
  // The setters for the variables
  // Sets the mode. It is recommended that you don't change the mode halfway
  // into the program since that may cause allocation of pinned memory being
  // freed in a non-pinned way, which may cause problems - I haven't verified
  // it personally but better to note it here in the header file.
  //模式不能中途动态更改
  inline static void set_mode(Brew mode) { Get().mode_ = mode; }
  // Sets the random seed of both boost and curand
  //设置随机数产生的速度
  static void set_random_seed(const unsigned int seed);
  // Sets the device. Since we have cublas and curand stuff, set device also
  // requires us to reset those values.
  //设置使用设备的编号
  static void SetDevice(const int device_id);
  // Prints the current GPU status.
  //查询GPU状态
  static void DeviceQuery();
  // Check if specified device is available
  static bool CheckDevice(const int device_id);
  // Search from start_id to the highest possible device ordinal,
  // return the ordinal of the first available device.
  //查找设备
  static int FindDevice(const int start_id = 0);
  // Parallel training
  //并行训练
  //基本的存取函数
  //solver
  inline static int solver_count() { return Get().solver_count_; }
  inline static void set_solver_count(int val) { Get().solver_count_ = val; }
  inline static int solver_rank() { return Get().solver_rank_; }
  inline static void set_solver_rank(int val) { Get().solver_rank_ = val; }
  //处理进程
  inline static bool multiprocess() { return Get().multiprocess_; }
  inline static void set_multiprocess(bool val) { Get().multiprocess_ = val; }
  inline static bool root_solver() { return Get().solver_rank_ == 0; }
//保护成员类
 protected:
  //GPU 模式额外变量
#ifndef CPU_ONLY
  cublasHandle_t cublas_handle_; //cublas句柄
  curandGenerator_t curand_generator_; //cuda 随机数发生器
#endif
  //随机数
  shared_ptr<RNG> random_generator_;//通用随机数发生器
  //模式
  Brew mode_;

  // Parallel training
  //并行训练参数
  int solver_count_;//求解线程数
  int solver_rank_;//求解等级
  bool multiprocess_;//是否多处理

 private:
  // The private constructor to avoid duplicate instantiation.
  //私有构造函数，以避免重复实例化。同时只能用new 在堆上初始化
  Caffe();

  DISABLE_COPY_AND_ASSIGN(Caffe);//禁止类的复制和拷贝构造
};

}  // namespace caffe

#endif  // CAFFE_COMMON_HPP_
