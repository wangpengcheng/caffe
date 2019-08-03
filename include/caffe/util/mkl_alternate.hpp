#ifndef CAFFE_UTIL_MKL_ALTERNATE_H_
#define CAFFE_UTIL_MKL_ALTERNATE_H_

#ifdef USE_MKL

#include <mkl.h>

#else  // If use MKL, simply include the MKL header

#ifdef USE_ACCELERATE
#include <Accelerate/Accelerate.h>
#else
extern "C" {
#include <cblas.h>
}
#endif  // USE_ACCELERATE

#include <math.h>
//为了看懂match_function 不得不看懂 mkl
// 参考连接https://blog.csdn.net/fengbingchun/article/details/56280708

// Functions that caffe uses but are not present if MKL is not linked.

// A simple way to define the vsl unary functions. The operation should
// be in the form e.g. y[i] = sqrt(a[i])

//如果没有链接，则caffe使用但不存在的函数。

//定义vsl一元函数的简单方法。 操作应该
//表格如下 y [i] = sqrt（a [i]）

//预定义 类模板 
//连接函数名称，定义基本操作运算, 输入数量，参数a,结果y
#define DEFINE_VSL_UNARY_FUNC(name, operation) \
  template<typename Dtype> \
  void v##name(const int n, const Dtype* a, Dtype* y) { \
    CHECK_GT(n, 0); CHECK(a); CHECK(y); \
    for (int i = 0; i < n; ++i) { operation; } \
  } \
  inline void vs##name( \
    const int n, const float* a, float* y) { \
    v##name<float>(n, a, y); \
  } \
  inline void vd##name( \
      const int n, const double* a, double* y) { \
    v##name<double>(n, a, y); \
  }
//定义相关操作模板类

DEFINE_VSL_UNARY_FUNC(Sqr, y[i] = a[i] * a[i])//乘法
DEFINE_VSL_UNARY_FUNC(Sqrt, y[i] = sqrt(a[i]))//除法
DEFINE_VSL_UNARY_FUNC(Exp, y[i] = exp(a[i]))//指数
DEFINE_VSL_UNARY_FUNC(Ln, y[i] = log(a[i]))//对数
DEFINE_VSL_UNARY_FUNC(Abs, y[i] = fabs(a[i]))//绝对值

// A simple way to define the vsl unary functions with singular parameter b.
// The operation should be in the form e.g. y[i] = pow(a[i], b)
//使用奇异参数b定义vsl一元函数的简单方法。
//操作应采用以下形式： y [i] = pow（a [i]，b）

#define DEFINE_VSL_UNARY_FUNC_WITH_PARAM(name, operation) \
  template<typename Dtype> \
  void v##name(const int n, const Dtype* a, const Dtype b, Dtype* y) { \
    CHECK_GT(n, 0); CHECK(a); CHECK(y); \
    for (int i = 0; i < n; ++i) { operation; } \
  } \
  inline void vs##name( \
    const int n, const float* a, const float b, float* y) { \
    v##name<float>(n, a, b, y); \
  } \
  inline void vd##name( \
      const int n, const double* a, const float b, double* y) { \
    v##name<double>(n, a, b, y); \
  }

//a的点指数操作
DEFINE_VSL_UNARY_FUNC_WITH_PARAM(Powx, y[i] = pow(a[i], b))

// A simple way to define the vsl binary functions. The operation should
// be in the form e.g. y[i] = a[i] + b[i]
//简单一维和操作
#define DEFINE_VSL_BINARY_FUNC(name, operation) \
  template<typename Dtype> \
  void v##name(const int n, const Dtype* a, const Dtype* b, Dtype* y) { \
    CHECK_GT(n, 0); CHECK(a); CHECK(b); CHECK(y); \
    for (int i = 0; i < n; ++i) { operation; } \
  } \
  inline void vs##name( \
    const int n, const float* a, const float* b, float* y) { \
    v##name<float>(n, a, b, y); \
  } \
  inline void vd##name( \
      const int n, const double* a, const double* b, double* y) { \
    v##name<double>(n, a, b, y); \
  }
// 加减乘除
DEFINE_VSL_BINARY_FUNC(Add, y[i] = a[i] + b[i])
DEFINE_VSL_BINARY_FUNC(Sub, y[i] = a[i] - b[i])
DEFINE_VSL_BINARY_FUNC(Mul, y[i] = a[i] * b[i])
DEFINE_VSL_BINARY_FUNC(Div, y[i] = a[i] / b[i])

// In addition, MKL comes with an additional function axpby that is not present
// in standard blas. We will simply use a two-step (inefficient, of course) way
// to mimic that.
//此外，MKL附带了一个不存在的附加功能axpby
//在标准blas中。 我们将简单地使用两步（当然效率低下）模仿那个

inline void cblas_saxpby(const int N, const float alpha, const float* X,
                         const int incX, const float beta, float* Y,
                         const int incY) {
  cblas_sscal(N, beta, Y, incY);
  cblas_saxpy(N, alpha, X, incX, Y, incY);
}
inline void cblas_daxpby(const int N, const double alpha, const double* X,
                         const int incX, const double beta, double* Y,
                         const int incY) {
  cblas_dscal(N, beta, Y, incY);
  cblas_daxpy(N, alpha, X, incX, Y, incY);
}

#endif  // USE_MKL
#endif  // CAFFE_UTIL_MKL_ALTERNATE_H_
