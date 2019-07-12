#ifndef CAFFE_UTIL_NCCL_H_
#define CAFFE_UTIL_NCCL_H_
#ifdef USE_NCCL

#include <nccl.h>

#include "caffe/common.hpp"

//定义nccl check
#define NCCL_CHECK(condition) \
{ \
  ncclResult_t result = condition; \
  CHECK_EQ(result, ncclSuccess) << " " \
    << ncclGetErrorString(result); \
}

namespace caffe {

namespace nccl {

template <typename Dtype> class dataType;

//定义数据类型模板类 float和double的类型
template<> class dataType<float> {
 public:
	 //nccl类型
  static const ncclDataType_t type = ncclFloat;
};
template<> class dataType<double> {
 public:
  static const ncclDataType_t type = ncclDouble;
};

}  // namespace nccl

}  // namespace caffe

#endif  // end USE_NCCL

#endif  // CAFFE_UTIL_NCCL_H_
