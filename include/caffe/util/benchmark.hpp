#ifndef CAFFE_UTIL_BENCHMARK_H_
#define CAFFE_UTIL_BENCHMARK_H_

#include <boost/date_time/posix_time/posix_time.hpp>

#include "caffe/util/device_alternate.hpp"

namespace caffe {
//时间计时接口类
class Timer {
 public:
  Timer();
  virtual ~Timer();
  virtual void Start();
  virtual void Stop();
  virtual float MilliSeconds();//毫秒
  virtual float MicroSeconds();//微秒
  virtual float Seconds();

  inline bool initted() { return initted_; }
  inline bool running() { return running_; }
  inline bool has_run_at_least_once() { return has_run_at_least_once_; }

 protected:
  void Init();//初始化计时器

  bool initted_;//是否初始化
  bool running_;//是否正在运行
  bool has_run_at_least_once_;//是否最后一次运行
#ifndef CPU_ONLY
  cudaEvent_t start_gpu_; //开始GPU
  cudaEvent_t stop_gpu_;//停止PU
#endif
  boost::posix_time::ptime start_cpu_; //boost开始cpu时钟
  boost::posix_time::ptime stop_cpu_;//boost开始cpu时钟
  float elapsed_milliseconds_; //毫秒
  float elapsed_microseconds_;//微秒
};
//cpu时钟
class CPUTimer : public Timer {
 public:
  explicit CPUTimer();
  virtual ~CPUTimer() {}
  virtual void Start();
  virtual void Stop();
  virtual float MilliSeconds();
  virtual float MicroSeconds();
};
//这个类主要是为了方便计时
}  // namespace caffe
//
#endif   // CAFFE_UTIL_BENCHMARK_H_
