#ifndef CAFFE_UTIL_DEVICE_ALTERNATE_H_
#define CAFFE_UTIL_DEVICE_ALTERNATE_H_

//如果仅仅使用CPU
#ifdef CPU_ONLY  // CPU-only Caffe.

#include <vector>

// Stub out GPU calls as unavailable.
//定义无GPU输出状况
#define NO_GPU LOG(FATAL) << "Cannot use GPU in CPU-only Caffe: check mode."
//定义设置使用GPU向前和向后计算的时候重载函数输出错误警告信息,这个一般由具体的实现类进行重载
#define STUB_GPU(classname) \
template <typename Dtype> \
void classname<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, \
    const vector<Blob<Dtype>*>& top) { NO_GPU; } \
template <typename Dtype> \
void classname<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, \
    const vector<bool>& propagate_down, \
    const vector<Blob<Dtype>*>& bottom) { NO_GPU; } \
//更改重载类中向前计算函数声明
#define STUB_GPU_FORWARD(classname, funcname) \
template <typename Dtype> \
void classname<Dtype>::funcname##_##gpu(const vector<Blob<Dtype>*>& bottom, \
    const vector<Blob<Dtype>*>& top) { NO_GPU; } \
//定义向后计算函数声明
#define STUB_GPU_BACKWARD(classname, funcname) \
template <typename Dtype> \
void classname<Dtype>::funcname##_##gpu(const vector<Blob<Dtype>*>& top, \
    const vector<bool>& propagate_down, \
    const vector<Blob<Dtype>*>& bottom) { NO_GPU; } \

#else  // Normal GPU + CPU Caffe.
//否则使用标准的CPU+GPU模式
#include <cublas_v2.h>//引入cublas_v2
#include <cuda.h> //引入cuda.h
#include <cuda_runtime.h>//引入cuda运行时
#include <curand.h>//
#include <driver_types.h>  // cuda driver types
#ifdef USE_CUDNN  // cuDNN acceleration library.
#include "caffe/util/cudnn.hpp" //包含cudnn头文件
#endif
// CHECK_EQ(x,y)<<"x!=y"，EQ即equation，意为“等于”，函数判断是否x等于y，当x!=y时，函数打印出x!=y。

// CHECK_NE(x,y)<<"x=y"，NE即not equation，意为“不等于”，函数判断是否x不等于y，当x=y时，函数打印出x=y。

// CHECK_LE(x,y)<<"x<=y",LE即lower equation,意为小于等于，函数判断是否x小于等于y。当x>=y时，函数打印x>=y。

// CHECK_LT(x,y)<<"x<=y",LT即为lower to ，意为小于，函数判断是否x小于y，当x>y时，函数打印x>y。

// CHECK_GE(x,y)<<"x>=y",GE即为great equation，意为大于。判断意义根据上述可推导出。

// CHECK_GT(x,y)<<"x>y",同理如上。
// google flag
// #define CHECK_EQ(x,y) CHECK_OP(x,y,EQ,==)
// #define CHECK_NE(x,y) CHECK_OP(x,y,NE,!=)
// #define CHECK_LE(x,y) CHECK_OP(x,y,LE,<=)
// #define CHECK_LT(x,y) CHECK_OP(x,y,LT,<)
// #define CHECK_GE(x,y) CHECK_OP(x,y,GE,>=)
// #define CHECK_GT(x,y) CHECK_OP(x,y,GT,>)



//
// CUDA macros
//
//各种cuda函数句柄确认
// CUDA: various checks for different function calls.
#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)

#define CUBLAS_CHECK(condition) \
  do { \
    cublasStatus_t status = condition; \
    CHECK_EQ(status, CUBLAS_STATUS_SUCCESS) << " " \
      << caffe::cublasGetErrorString(status); \
  } while (0)

#define CURAND_CHECK(condition) \
  do { \
    curandStatus_t status = condition; \
    CHECK_EQ(status, CURAND_STATUS_SUCCESS) << " " \
      << caffe::curandGetErrorString(status); \
  } while (0)

//确定网格计算线程

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

// CUDA: check for error after kernel execution and exit loudly if there is one.
#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())

namespace caffe {

// CUDA: library error reporting.
//获取错误类型进行对象之间的转换
const char* cublasGetErrorString(cublasStatus_t error);
const char* curandGetErrorString(curandStatus_t error);

// CUDA: use 512 threads per block
//默认每个块中使用512个线程
const int CAFFE_CUDA_NUM_THREADS = 512;

// CUDA: number of blocks for threads.
//当前线程所在块编号
inline int CAFFE_GET_BLOCKS(const int N) {
  return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
}

}  // namespace caffe

#endif  // CPU_ONLY

#endif  // CAFFE_UTIL_DEVICE_ALTERNATE_H_
/*
wpc:
此文件定义了对于CPU和GPU基本的类信息转换和线程操作
包括错误信息的获取、所在计算单元的块位置
相关错误信息的输出实现，在common.hpp中
*/