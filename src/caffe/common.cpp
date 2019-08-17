#include <boost/thread.hpp>
#include <glog/logging.h>
#include <cmath>
#include <cstdio>
#include <ctime>

#include "caffe/common.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

// Make sure each thread can have different values.
//每个线程的caffe对象，关键控制值，然而caffe cpu单线程没什么用
static boost::thread_specific_ptr<Caffe> thread_instance_;
//Get函数的实现
Caffe& Caffe::Get() {
  //关键记录值不存在，就创建一个
  if (!thread_instance_.get()) {
    thread_instance_.reset(new Caffe());
  }
  return *(thread_instance_.get());
}

// random seeding
//打开cpu随机数产生器，获得随机数种子
int64_t cluster_seedgen(void) {
  int64_t s, seed, pid;
  FILE* f = fopen("/dev/urandom", "rb");//打开cpu随机数产生器
  if (f && fread(&seed, 1, sizeof(seed), f) == sizeof(seed)) {
    fclose(f);
    return seed;//获取随机数种子
  }
  //错误就产生提示信息
  LOG(INFO) << "System entropy source not available, "
              "using fallback algorithm to generate seed instead.";
  if (f)
    fclose(f);

  pid = getpid();
  s = time(NULL);//时间种子
  //根据线程编号,和当前时间秒数，产生随机种子
  seed = std::abs(((s * 181) * ((pid - 83) * 359)) % 104729);
  return seed;
}

//初始化google flag参数
void GlobalInit(int* pargc, char*** pargv) {
  // Google flags.
  ::gflags::ParseCommandLineFlags(pargc, pargv, true);
  // Google logging.
  ::google::InitGoogleLogging(*(pargv)[0]);
  // Provide a backtrace on segfault.
  ::google::InstallFailureSignalHandler();
}

#ifdef CPU_ONLY  // CPU-only Caffe.
//仅仅cpu模式下的类初始化
Caffe::Caffe()
    : random_generator_(), mode_(Caffe::CPU),
      solver_count_(1), solver_rank_(0), multiprocess_(false) { }
//设置空的析构函数
Caffe::~Caffe() { }

//设置随机数产生种子
void Caffe::set_random_seed(const unsigned int seed) {
  // RNG seed
  Get().random_generator_.reset(new RNG(seed));
}

void Caffe::SetDevice(const int device_id) {
  NO_GPU;
}

void Caffe::DeviceQuery() {
  NO_GPU;
}

bool Caffe::CheckDevice(const int device_id) {
  NO_GPU;
  return false;
}

int Caffe::FindDevice(const int start_id) {
  NO_GPU;
  return -1;
}
//定义 Generator  通用类
class Caffe::RNG::Generator {
 public:
  //初始化空构造函数
  Generator() : rng_(new caffe::rng_t(cluster_seedgen())) {}
  //含参数构造函数
  explicit Generator(unsigned int seed) : rng_(new caffe::rng_t(seed)) {}
  //获取随机数
  caffe::rng_t* rng() { return rng_.get(); }
 private:
  shared_ptr<caffe::rng_t> rng_;//随机数生成器
};
/* cpu 随机数产生类 start*/
//RNG 空构造函数
Caffe::RNG::RNG() : generator_(new Generator()) { }
//随机数种子的随机数
Caffe::RNG::RNG(unsigned int seed) : generator_(new Generator(seed)) { }
//重载拷贝赋值操作符
Caffe::RNG& Caffe::RNG::operator=(const RNG& other) {
  generator_ = other.generator_;
  return *this;
}
//强制类型转换，将boost::mt19937 转换为 void* 静态指针
void* Caffe::RNG::generator() {
  return static_cast<void*>(generator_->rng());
}

#else  // Normal GPU + CPU Caffe.
//标准的CPU和GPU构造函数
//空构造函数；多初始化了cublas，curand
Caffe::Caffe()
    : cublas_handle_(NULL), curand_generator_(NULL), random_generator_(),
    mode_(Caffe::CPU),
    solver_count_(1), solver_rank_(0), multiprocess_(false) {
  // Try to create a cublas handler, and report an error if failed (but we will
  // keep the program running as one might just want to run CPU code).
  //创建cublas句柄
  if (cublasCreate(&cublas_handle_) != CUBLAS_STATUS_SUCCESS) {
    LOG(ERROR) << "Cannot create Cublas handle. Cublas won't be available.";
  }
  // Try to create a curand handler.
  //创建curand句柄
  if (curandCreateGenerator(&curand_generator_, CURAND_RNG_PSEUDO_DEFAULT)
      != CURAND_STATUS_SUCCESS ||
      curandSetPseudoRandomGeneratorSeed(curand_generator_, cluster_seedgen())
      != CURAND_STATUS_SUCCESS) {
    LOG(ERROR) << "Cannot create Curand generator. Curand won't be available.";
  }
}
//定义析构函数
Caffe::~Caffe() {
  if (cublas_handle_) CUBLAS_CHECK(cublasDestroy(cublas_handle_));
  if (curand_generator_) {
    CURAND_CHECK(curandDestroyGenerator(curand_generator_));
  }
}
//设置随机数种子
void Caffe::set_random_seed(const unsigned int seed) {
  // Curand seed
  //设置静态Curand 随机数种子并将其设置为全局变量
  static bool g_curand_availability_logged = false;
  //如果不为空
  if (Get().curand_generator_) {
    //设置随机种子
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(curand_generator(),
        seed));
    //设置片偏移为0
    CURAND_CHECK(curandSetGeneratorOffset(curand_generator(), 0));
  } else {
    if (!g_curand_availability_logged) {
        LOG(ERROR) <<
            "Curand not available. Skipping setting the curand seed.";
        g_curand_availability_logged = true;
    }
  }
  // RNG seed
  //重新设置随机数生成器的随机种子
  Get().random_generator_.reset(new RNG(seed));
}
//指定设备编号
void Caffe::SetDevice(const int device_id) {
  int current_device;//当前设备编号
  //获取当前设备
  CUDA_CHECK(cudaGetDevice(&current_device));
  //如果就是当前设备，直接返回无需切换
  if (current_device == device_id) {
    return;
  }
  // The call to cudaSetDevice must come before any calls to Get, which
  // may perform initialization using the GPU.
  //对cudaSetDevice的调用必须在任何对Get的调用之前进行,可以使用GPU执行初始化。
  //设置当前设备id
  CUDA_CHECK(cudaSetDevice(device_id));
  //如果存在旧句柄,销毁原来句柄
  if (Get().cublas_handle_) CUBLAS_CHECK(cublasDestroy(Get().cublas_handle_));
  //如果存在旧随机数产生器，销毁旧随机数产生器
  if (Get().curand_generator_) {
    CURAND_CHECK(curandDestroyGenerator(Get().curand_generator_));
  }
  //创建新的cuda句柄
  CUBLAS_CHECK(cublasCreate(&Get().cublas_handle_));
  //创建新的随机数产生器
  CURAND_CHECK(curandCreateGenerator(&Get().curand_generator_,
      CURAND_RNG_PSEUDO_DEFAULT));
  //设置随机数种子为随机数
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(Get().curand_generator_,
      cluster_seedgen()));
}
//查询设备
void Caffe::DeviceQuery() {
  cudaDeviceProp prop;//cuda 基础结构体，主要显示设备结构属性
  int device;
  //获取设备信息
  if (cudaSuccess != cudaGetDevice(&device)) {
    printf("No cuda device present.\n");
    return;
  }
  //输出设备信息
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  LOG(INFO) << "Device id:                     " << device;
  LOG(INFO) << "Major revision number:         " << prop.major;
  LOG(INFO) << "Minor revision number:         " << prop.minor;
  LOG(INFO) << "Name:                          " << prop.name;
  LOG(INFO) << "Total global memory:           " << prop.totalGlobalMem;
  LOG(INFO) << "Total shared memory per block: " << prop.sharedMemPerBlock;
  LOG(INFO) << "Total registers per block:     " << prop.regsPerBlock;
  LOG(INFO) << "Warp size:                     " << prop.warpSize;
  LOG(INFO) << "Maximum memory pitch:          " << prop.memPitch;
  LOG(INFO) << "Maximum threads per block:     " << prop.maxThreadsPerBlock;
  //输出每个唯独最大数量，相乘就是最大线程数
  LOG(INFO) << "Maximum dimension of block:    "
      << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", "
      << prop.maxThreadsDim[2];
      //最大网格数量
  LOG(INFO) << "Maximum dimension of grid:     "
      << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", "
      << prop.maxGridSize[2];
  //时钟频率
  LOG(INFO) << "Clock rate:                    " << prop.clockRate;
  //总内存
  LOG(INFO) << "Total constant memory:         " << prop.totalConstMem;
  //纹理对齐
  LOG(INFO) << "Texture alignment:             " << prop.textureAlignment;
  //是否允许并发复制和执行
  LOG(INFO) << "Concurrent copy and execution: "
      << (prop.deviceOverlap ? "Yes" : "No");
  //多处理器数量
  LOG(INFO) << "Number of multiprocessors:     " << prop.multiProcessorCount;
  //允许核心超时
  LOG(INFO) << "Kernel execution timeout:      "
      << (prop.kernelExecTimeoutEnabled ? "Yes" : "No");
  return;
}
//确认设备
bool Caffe::CheckDevice(const int device_id) {
  // This function checks the availability of GPU #device_id.
  // It attempts to create a context on the device by calling cudaFree(0).
  // cudaSetDevice() alone is not sufficient to check the availability.
  // It lazily records device_id, however, does not initialize a
  // context. So it does not know if the host thread has the permission to use
  // the device or not.
  //此函数检查GPU #device_id的可用性。 
  //它尝试通过调用cudaFree（0）在设备上创建上下文。 
  //仅cudaSetDevice（）不足以检查可用性。 
  //它懒洋洋地记录了device_id，但是没有初始化上下文。 
  //因此，它不知道主机线程是否具有使用该设备的权限。
  //
  // In a shared environment where the devices are set to EXCLUSIVE_PROCESS
  // or EXCLUSIVE_THREAD mode, cudaSetDevice() returns cudaSuccess
  // even if the device is exclusively occupied by another process or thread.
  // Cuda operations that initialize the context are needed to check
  // the permission. cudaFree(0) is one of those with no side effect,
  // except the context initialization.
  //在设备设置为EXCLUSIVE_PROCESS的共享环境中
  //或EXCLUSIVE_THREAD模式，cudaSetDevice（）返回cudaSuccess
  //即使设备完全被另一个进程或线程占用。
  //需要检查初始化上下文的Cuda操作
  //许可 cudaFree（0）是没有副作用的人之一，
  //除了上下文初始化。
  //确认设置设备，并且是否内存
  bool r = ((cudaSuccess == cudaSetDevice(device_id)) &&
            (cudaSuccess == cudaFree(0)));
  // reset any error that may have occurred.
  //重置可能发生的任何错误。
  cudaGetLastError();
  //返回查询结果
  return r;
}
//查找设备
int Caffe::FindDevice(const int start_id) {
  // This function finds the first available device by checking devices with
  // ordinal from start_id to the highest available value. In the
  // EXCLUSIVE_PROCESS or EXCLUSIVE_THREAD mode, if it succeeds, it also
  // claims the device due to the initialization of the context.
  //此函数通过检查设备来查找第一个可用设备
  //从start_id到最高可用值的序数。 在里面
  // EXCLUSIVE_PROCESS或EXCLUSIVE_THREAD模式，如果成功，它也是
  //由于上下文的初始化而声明了设备。
  
  int count = 0;
  //查询设备总数
  CUDA_CHECK(cudaGetDeviceCount(&count));
  //查询设备是否可用，找到第一个可用设备就直接返回
  for (int i = start_id; i < count; i++) {
    if (CheckDevice(i)) return i;
  }
  return -1;
}
//发生器的定义
class Caffe::RNG::Generator {
 public:
  //空构造函数定义随机数种子
  Generator() : rng_(new caffe::rng_t(cluster_seedgen())) {}
  //自定义随机数种子
  explicit Generator(unsigned int seed) : rng_(new caffe::rng_t(seed)) {}
  //获取随机数生成器指针
  caffe::rng_t* rng() { return rng_.get(); }
 private:
  //定义共享指针，随机数生成器
  shared_ptr<caffe::rng_t> rng_;
};
//RNG构造函数
Caffe::RNG::RNG() : generator_(new Generator()) { }
//RNG随机种子构造函数
Caffe::RNG::RNG(unsigned int seed) : generator_(new Generator(seed)) { }
//拷贝赋值
Caffe::RNG& Caffe::RNG::operator=(const RNG& other) {
  generator_.reset(other.generator_.get());
  return *this;
}
//随机数生成器，转化为静态空指针
void* Caffe::RNG::generator() {
  return static_cast<void*>(generator_->rng());
}
//返回cublas错误信息，对支持版本只考虑了cuda7
const char* cublasGetErrorString(cublasStatus_t error) {
  switch (error) {
  case CUBLAS_STATUS_SUCCESS:
    return "CUBLAS_STATUS_SUCCESS";
  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "CUBLAS_STATUS_NOT_INITIALIZED";
  case CUBLAS_STATUS_ALLOC_FAILED:
    return "CUBLAS_STATUS_ALLOC_FAILED";
  case CUBLAS_STATUS_INVALID_VALUE:
    return "CUBLAS_STATUS_INVALID_VALUE";
  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "CUBLAS_STATUS_ARCH_MISMATCH";
  case CUBLAS_STATUS_MAPPING_ERROR:
    return "CUBLAS_STATUS_MAPPING_ERROR";
  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "CUBLAS_STATUS_EXECUTION_FAILED";
  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "CUBLAS_STATUS_INTERNAL_ERROR";
#if CUDA_VERSION >= 6000
  case CUBLAS_STATUS_NOT_SUPPORTED:
    return "CUBLAS_STATUS_NOT_SUPPORTED";
#endif
#if CUDA_VERSION >= 6050
  case CUBLAS_STATUS_LICENSE_ERROR:
    return "CUBLAS_STATUS_LICENSE_ERROR";
#endif
  }
  return "Unknown cublas status";
}
//curand 错误信息返回
const char* curandGetErrorString(curandStatus_t error) {
  switch (error) {
  case CURAND_STATUS_SUCCESS:
    return "CURAND_STATUS_SUCCESS";
  case CURAND_STATUS_VERSION_MISMATCH:
    return "CURAND_STATUS_VERSION_MISMATCH";
  case CURAND_STATUS_NOT_INITIALIZED:
    return "CURAND_STATUS_NOT_INITIALIZED";
  case CURAND_STATUS_ALLOCATION_FAILED:
    return "CURAND_STATUS_ALLOCATION_FAILED";
  case CURAND_STATUS_TYPE_ERROR:
    return "CURAND_STATUS_TYPE_ERROR";
  case CURAND_STATUS_OUT_OF_RANGE:
    return "CURAND_STATUS_OUT_OF_RANGE";
  case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
    return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
  case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
    return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
  case CURAND_STATUS_LAUNCH_FAILURE:
    return "CURAND_STATUS_LAUNCH_FAILURE";
  case CURAND_STATUS_PREEXISTING_FAILURE:
    return "CURAND_STATUS_PREEXISTING_FAILURE";
  case CURAND_STATUS_INITIALIZATION_FAILED:
    return "CURAND_STATUS_INITIALIZATION_FAILED";
  case CURAND_STATUS_ARCH_MISMATCH:
    return "CURAND_STATUS_ARCH_MISMATCH";
  case CURAND_STATUS_INTERNAL_ERROR:
    return "CURAND_STATUS_INTERNAL_ERROR";
  }
  return "Unknown curand status";
}

#endif  // CPU_ONLY

}  // namespace caffe
//这个common，cpp主要是caffe基础类的定义;主要封装了随机数生成器
//和多线程同步相关的基础变量
//接下来是caffe.pb.h