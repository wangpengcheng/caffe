#ifndef CAFFE_UTIL_GPU_UTIL_H_
#define CAFFE_UTIL_GPU_UTIL_H_

namespace caffe {
//GPU的原子添加模板函数
template <typename Dtype>
inline __device__ Dtype caffe_gpu_atomic_add(const Dtype val, Dtype* address);

//特例化 float 和double的模板函数
template <>
inline __device__
float caffe_gpu_atomic_add(const float val, float* address) {
  return atomicAdd(address, val);
}

// double atomicAdd implementation taken from:
// http://docs.nvidia.com/cuda/cuda-c-programming-guide/#axzz3PVCpVsEG
template <>
inline __device__
//double添加函数
double caffe_gpu_atomic_add(const double val, double* address) {
  //初始化地址
  unsigned long long int* address_as_ull =  // NOLINT(runtime/int)
      // NOLINT_NEXT_LINE(runtime/int)
      reinterpret_cast<unsigned long long int*>(address);
  //获取旧地址
  unsigned long long int old = *address_as_ull;  // NOLINT(runtime/int)
  unsigned long long int assumed;  // NOLINT(runtime/int)
  do {
    assumed = old;

    old = atomicCAS(address_as_ull, assumed,
        __double_as_longlong(val + __longlong_as_double(assumed)));
  }
  //这里是线程之间的原子操作,当内存分配成功，即操作成功的时候跳出循环 
  while (assumed != old);
  //返回最终结果
  return __longlong_as_double(old);
}

}  // namespace caffe

#endif  // CAFFE_UTIL_GPU_UTIL_H_
