#include <algorithm>

#include "caffe/common.hpp"
#include "caffe/util/im2col.hpp"

namespace caffe {
//gpu版本的im2col
template <typename Dtype>
__global__ void im2col_gpu_kernel(
    const int n, //分的线程总数
    const Dtype* data_im,//im_data
    const int height, const int width,//宽高 
    const int kernel_h, const int kernel_w,//卷积盖度和宽度
    const int pad_h, const int pad_w,//扩充高宽
    const int stride_h, const int stride_w,//步长高宽
    const int dilation_h, const int dilation_w,//缩放高宽
    const int height_col, const int width_col,//输出col高宽
    Dtype* data_col //出输出数据col
  ) {
  CUDA_KERNEL_LOOP(index, n) //cuda网格流执行程序，n表示总线程数，index表示block的索引
  {
    //计算每个block中的h高度索引
    const int h_index = index / width_col;
    //计算每个block 中的col高度偏移
    const int h_col = h_index % height_col;
    //计算col中的w偏移
    const int w_col = index % width_col;
    //计算im的相对偏移索引，最终得到每个block中的块的索引
    const int c_im = h_index / height_col;
    //计算最终的输出大小
    const int c_col = c_im * kernel_h * kernel_w;
    //计算高度偏移
    const int h_offset = h_col * stride_h - pad_h;
    //计算宽度偏移
    const int w_offset = w_col * stride_w - pad_w;
    //定义输出数据指针
    Dtype* data_col_ptr = data_col;
    //设置当前block中的数据偏移
    data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;
    //im数据指针
    const Dtype* data_im_ptr = data_im;
    //当前block的im数据偏移和最终数据指向
    data_im_ptr += (c_im * height + h_offset) * width + w_offset;
    //遍历卷积核
    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        //获取h索引
        int h_im = h_offset + i * dilation_h;
        //获取w索引
        int w_im = w_offset + j * dilation_w;
        *data_col_ptr =
            (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ?
            data_im_ptr[i * dilation_h * width + j * dilation_w] : 0;
        //移动更新指针
        data_col_ptr += height_col * width_col;
      }
    }
  }
}

template <typename Dtype>
//将im2col gpu 版本
void im2col_gpu(
    const Dtype* data_im,//数据 
    const int channels,//通道
    const int height, const int width,//高宽 
    const int kernel_h, const int kernel_w,//卷积核高宽
    const int pad_h, const int pad_w,//扩充高宽
    const int stride_h, const int stride_w,//步长高宽
    const int dilation_h, const int dilation_w,//缩放高宽
    Dtype* data_col) {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  //我们将启动channel * height_col * width_col内核，每个内核负责复制单通道网格。
  //计算col高度
  int height_col = (height + 2 * pad_h -
      (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  //计算col宽度
  int width_col = (width + 2 * pad_w -
      (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  //计算kernel数量
  int num_kernels = channels * height_col * width_col;
  // NOLINT_NEXT_LINE(whitespace/operators)
  //使用im2col_gpu_kernel进行计算
  im2col_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, data_im, height, width, kernel_h, kernel_w, pad_h,
      pad_w, stride_h, stride_w, dilation_h, dilation_w, height_col,
      width_col, data_col);
  CUDA_POST_KERNEL_CHECK;
}
//特例化单双精度函数
// Explicit instantiation
template void im2col_gpu<float>(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, float* data_col);
template void im2col_gpu<double>(const double* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, double* data_col);

//im2col和col2im
template <typename Dtype, int num_axes>
__global__ void im2col_nd_gpu_kernel(
    const int n,//kernel数量 
    const Dtype* data_im,//im数据
    const int* im_shape,//imshape 
    const int* col_shape,//col shape
    const int* kernel_shape,//kernel维度数据 
    const int* pad,//扩充 
    const int* stride,//步长
    const int* dilation,//缩放 
    Dtype* data_col//输出数据
  ) {
  //初始化临时行长度
  int d_temp[num_axes];  // NOLINT(runtime/arrays)
  //初始化行内偏移
  int d_iter[num_axes];  // NOLINT(runtime/arrays)
  //host和device的共享数据：
  //缩放
  __shared__ int shared_dilation[num_axes];
  //kernel shape
  __shared__ int shared_kernel_shape[num_axes];
  //shape pad
  __shared__ int shared_pad[num_axes];
  //步长
  __shared__ int shared_stride[num_axes];
  //col形状
  __shared__ int shared_col_shape[num_axes + 1];
  //im形状
  __shared__ int shared_im_shape[num_axes + 1];
  //实现数据之间的同步
  if (threadIdx.x < num_axes) {
    shared_dilation[threadIdx.x] = dilation[threadIdx.x];
    shared_kernel_shape[threadIdx.x] = kernel_shape[threadIdx.x];
    shared_pad[threadIdx.x] = pad[threadIdx.x];
    shared_stride[threadIdx.x] = stride[threadIdx.x];
  }
  //实现数据共享和同步
  if (threadIdx.x < num_axes + 1) {
    shared_col_shape[threadIdx.x] = col_shape[threadIdx.x];
    shared_im_shape[threadIdx.x] = im_shape[threadIdx.x];
  }
  //同步线程数据
  __syncthreads();
  //
  int i;
  //设置kernel进行工作
  CUDA_KERNEL_LOOP(index, n) {
    // Initialize channel_in, computed in the loop below, with intermediate
    // computations used to compute the spatial indices.
    //初始化channel_in，在下面的循环中计算，使用中间计算来计算空间索引。
    int channel_in = index;
    //输出的channel数量
    int channel_out = 1;
    //遍历维度，计算维度长度和输入输出的索引
    for (i = num_axes - 1; i >= 0; --i) {
      d_temp[i] = channel_in % shared_col_shape[i + 1];
      channel_in /= shared_col_shape[i + 1];
      channel_out *= shared_kernel_shape[i];
    }
    //输出=输出*输入；更新输出的通道数量
    channel_out *= channel_in;
    //col数据的行索引初始化为1
    int data_col_inc = 1;
    for (i = 0; i < num_axes; ++i) {
      //计算维度
      channel_out *= shared_col_shape[i + 1];
      //计算偏移
      channel_out += d_temp[i];
      //temp更新temp的索引
      d_temp[i] = d_temp[i] * shared_stride[i] - shared_pad[i];
      //计算输入通道
      channel_in *= shared_im_shape[i + 1];
      //维度的输入偏移
      channel_in += d_temp[i];
      //输出数据索引
      data_col_inc *= shared_col_shape[i + 1];
      //重制行内偏移
      d_iter[i] = 0;
    }
    //计算当前block中data指针应该指向的位置
    Dtype* data_col_ptr = data_col + channel_out;
    //im指针应该指向的位置
    const Dtype* data_im_ptr = data_im + channel_in;
    //是否持续增加
    bool incremented;
    //开始循环
    do {
      //是否越界
      bool in_range = true;
      //遍历维度，计算im的行内偏移
      for (i = 0; i < num_axes; ++i) {
        const int d_iter_im = d_iter[i] * shared_dilation[i] + d_temp[i];
        //判断是否在范围内
        in_range &= d_iter_im >= 0 && d_iter_im < shared_im_shape[i + 1];
        if (!in_range) { break; }
      }
      //在范围内
      if (in_range) {
        //计算初始的伸缩后的数据偏移
        int data_im_offset = d_iter[0] * shared_dilation[0];
        //计算im的数据偏移
        for (i = 1; i < num_axes; ++i) {
          data_im_offset *= shared_im_shape[i + 1];
          data_im_offset += d_iter[i] * shared_dilation[i];
        }
        //计算最后的col数据指针位置
        *data_col_ptr = data_im_ptr[data_im_offset];
      } else {
        //范围之外数据置0
        *data_col_ptr = 0;
      }
      //将指针加上之前的data_col_inc偏移
      data_col_ptr += data_col_inc;
      //不再增加，这里要将每个维度都遍历完后才会停止
      incremented = false;
      //维度
      for (i = num_axes - 1; i >= 0; --i) {
        //最大线程数量
        const int d_max = shared_kernel_shape[i];
        if (d_iter[i] == d_max - 1) {
          //越界偏移为0
          d_iter[i] = 0;
        } else {  // d_iter[i] < d_max - 1
          //添加指针，更新维度
          ++d_iter[i];
          //继续增加
          incremented = true;
          break;
        }
      }  // for (int i = num_axes - 1; i >= 0; --i)
    } while (incremented);  // do
  }  // CUDA_KERNEL_LOOP(index, n)
}

//im2col的正真接口，将kernel核心数目作为参数，进行了封装
template <typename Dtype>
void im2col_nd_gpu(
    const Dtype* data_im, 
    const int num_spatial_axes,
    const int num_kernels, 
    const int* im_shape, 
    const int* col_shape,
    const int* kernel_shape, 
    const int* pad, const int* stride,
    const int* dilation, 
    Dtype* data_col
  ) {
  // num_axes should be smaller than block size
  DCHECK_LT(num_spatial_axes, CAFFE_CUDA_NUM_THREADS);
  switch (num_spatial_axes) {
  case 1:
    im2col_nd_gpu_kernel<Dtype, 1>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
        num_kernels, data_im, im_shape, col_shape,
        kernel_shape, pad, stride, dilation, data_col);
    break;
  case 2:
    im2col_nd_gpu_kernel<Dtype, 2>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
        num_kernels, data_im, im_shape, col_shape,
        kernel_shape, pad, stride, dilation, data_col);
    break;
  case 3:
    im2col_nd_gpu_kernel<Dtype, 3>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
        num_kernels, data_im, im_shape, col_shape,
        kernel_shape, pad, stride, dilation, data_col);
    break;
  case 4:
    im2col_nd_gpu_kernel<Dtype, 4>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
        num_kernels, data_im, im_shape, col_shape,
        kernel_shape, pad, stride, dilation, data_col);
    break;
  case 5:
    im2col_nd_gpu_kernel<Dtype, 5>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
        num_kernels, data_im, im_shape, col_shape,
        kernel_shape, pad, stride, dilation, data_col);
    break;
  case 6:
    im2col_nd_gpu_kernel<Dtype, 6>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
        num_kernels, data_im, im_shape, col_shape,
        kernel_shape, pad, stride, dilation, data_col);
    break;
  case 7:
    im2col_nd_gpu_kernel<Dtype, 7>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
        num_kernels, data_im, im_shape, col_shape,
        kernel_shape, pad, stride, dilation, data_col);
    break;
  case 8:
    im2col_nd_gpu_kernel<Dtype, 8>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
        num_kernels, data_im, im_shape, col_shape,
        kernel_shape, pad, stride, dilation, data_col);
    break;
  case 9:
    im2col_nd_gpu_kernel<Dtype, 9>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
        num_kernels, data_im, im_shape, col_shape,
        kernel_shape, pad, stride, dilation, data_col);
    break;
  case 10:
    im2col_nd_gpu_kernel<Dtype, 10>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
        num_kernels, data_im, im_shape, col_shape,
        kernel_shape, pad, stride, dilation, data_col);
    break;
  default:
    LOG(FATAL) << "im2col_nd_gpu does not support computation with "
               << num_spatial_axes << " spatial axes";
  }
  CUDA_POST_KERNEL_CHECK;
}

// Explicit instantiation
template void im2col_nd_gpu<float>(const float* data_im,
    const int num_spatial_axes, const int col_size,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, float* data_col);
template void im2col_nd_gpu<double>(const double* data_im,
    const int num_spatial_axes, const int col_size,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, double* data_col);


template <typename Dtype>
__global__ void col2im_gpu_kernel(const int n, const Dtype* data_col,
    const int height, const int width, const int channels,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int height_col, const int width_col,
    Dtype* data_im) {
  CUDA_KERNEL_LOOP(index, n) {
    //初始化真实值为0
    Dtype val = 0;
    const int w_im = index % width + pad_w;
    //计算bolck中对应的im的高度
    const int h_im = (index / width) % height + pad_h;
    const int c_im = index / (width * height);
    int kernel_extent_w = (kernel_w - 1) * dilation_w + 1;
    int kernel_extent_h = (kernel_h - 1) * dilation_h + 1;
    // compute the start and end of the output
    //计算线程内的输出起止节点
    const int w_col_start =
        (w_im < kernel_extent_w) ? 0 : (w_im - kernel_extent_w) / stride_w + 1;
    const int w_col_end = min(w_im / stride_w + 1, width_col);
    const int h_col_start =
        (h_im < kernel_extent_h) ? 0 : (h_im - kernel_extent_h) / stride_h + 1;
    const int h_col_end = min(h_im / stride_h + 1, height_col);
    // TODO: use LCM of stride and dilation to avoid unnecessary loops
    //循环遍历col查找对应的im的值
    for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
      for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {
        //获取行映射
        int h_k = (h_im - h_col * stride_h);
        //获取列映射
        int w_k = (w_im - w_col * stride_w);
        //正数倍缩放，即是缩放的值
        if (h_k % dilation_h == 0 && w_k % dilation_w == 0) {
          //还原真实索引h
          h_k /= dilation_h;
          //还原真实索引w
          w_k /= dilation_w;
          //计算真实的col索引值
          int data_col_index = (((c_im * kernel_h + h_k) * kernel_w + w_k) *
                                height_col + h_col) * width_col + w_col;
          //将数据index的数据进行叠加
          val += data_col[data_col_index];
        }
      }
    }
    data_im[index] = val;
  }
}

template <typename Dtype>
void col2im_gpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    Dtype* data_im) {
  int height_col = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) /
      stride_h + 1;
  int width_col = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) /
      stride_w + 1;
  //设置计算核心数目
  int num_kernels = channels * height * width;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  // NOLINT_NEXT_LINE(whitespace/operators)
  col2im_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, data_col, height, width, channels, kernel_h, kernel_w,
      pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
      height_col, width_col, data_im);
  CUDA_POST_KERNEL_CHECK;
}

// Explicit instantiation
template void col2im_gpu<float>(const float* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    float* data_im);
template void col2im_gpu<double>(const double* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    double* data_im);
//col2im与im2col操作的最后一步不一样，其它基本相同
template <typename Dtype, int num_axes>
__global__ void col2im_nd_gpu_kernel(const int n, const Dtype* data_col,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, Dtype* data_im) {
  int d_im[num_axes];  // NOLINT(runtime/arrays)
  int d_col_iter[num_axes];  // NOLINT(runtime/arrays)
  int d_col_start[num_axes];  // NOLINT(runtime/arrays)
  int d_col_end[num_axes];  // NOLINT(runtime/arrays)

  __shared__ int shared_dilation[num_axes];
  __shared__ int shared_kernel_shape[num_axes];
  __shared__ int shared_pad[num_axes];
  __shared__ int shared_stride[num_axes];
  __shared__ int shared_col_shape[num_axes + 1];
  __shared__ int shared_im_shape[num_axes + 1];

  if (threadIdx.x < num_axes) {
    shared_dilation[threadIdx.x] = dilation[threadIdx.x];
    shared_kernel_shape[threadIdx.x] = kernel_shape[threadIdx.x];
    shared_pad[threadIdx.x] = pad[threadIdx.x];
    shared_stride[threadIdx.x] = stride[threadIdx.x];
  }
  if (threadIdx.x < num_axes + 1) {
    shared_col_shape[threadIdx.x] = col_shape[threadIdx.x];
    shared_im_shape[threadIdx.x] = im_shape[threadIdx.x];
  }
  __syncthreads();

  CUDA_KERNEL_LOOP(index, n) {
    // Initialize channel_in, computed in the loop below, with intermediate
    // computations used to compute the spatial indices.
    int c_im = index;
    // Calculate d_im (image dimensions).
    for (int i = num_axes - 1; i >= 0; --i) {
      d_im[i] = c_im % shared_im_shape[i + 1] + shared_pad[i];
      c_im /= shared_im_shape[i + 1];
    }
    // Calculate col start/end indices.
    bool done = false;
    for (int i = 0; i < num_axes; ++i) {
      const int kernel_extent =
          shared_dilation[i] * (shared_kernel_shape[i] - 1) + 1;
      d_col_start[i] = d_col_iter[i] =
          (d_im[i] < kernel_extent) ? 0 :
          (d_im[i] - kernel_extent) / shared_stride[i] + 1;
      d_col_end[i] =
          min(d_im[i] / shared_stride[i] + 1, shared_col_shape[i + 1]);
      if (d_col_start[i] >= d_col_end[i]) {
        // Skip computation if the dimension is 0 at any spatial axis --
        // final val will be 0.
        data_im[index] = 0;
        done = true;
        break;  // for (int i = 0; i < num_axes; ++i)
      }
    }
    if (done) {
      continue;  // CUDA_KERNEL_LOOP(index, n)
    }
    // Loop over the col to compute the output val.
    Dtype val = 0;
    bool incremented = true;
    bool skip = false;
    do {
      // Compute the final offset.
      int final_offset = 0;
      int kernel_shape_prod = 1;
      int kernel_index;
      for (int i = num_axes - 1; i >= 0; --i) {
        kernel_index = d_im[i] - d_col_iter[i] * shared_stride[i];
        if (kernel_index % shared_dilation[i]) {
          skip = true;
          break;
        } else {
          kernel_index /= shared_dilation[i];
          final_offset += kernel_index * kernel_shape_prod;
          kernel_shape_prod *= shared_kernel_shape[i];
        }
      }
      if (!skip) {
        final_offset += kernel_shape_prod * c_im;
        for (int i = 0; i < num_axes; ++i) {
          final_offset *= shared_col_shape[i + 1];
          final_offset += d_col_iter[i];
        }
        val += data_col[final_offset];
      }
      skip = false;
      incremented = false;
      for (int i = num_axes - 1; i >= 0; --i) {
        const int d_max = d_col_end[i];
        if (d_col_iter[i] == d_max - 1) {
          d_col_iter[i] = d_col_start[i];
        } else {  // d_col_iter[i] < d_max - 1
          ++d_col_iter[i];
          incremented = true;
          break;  // for (int i = num_axes - 1; i >= 0; --i)
        }
      }  // for (int i = num_axes - 1; i >= 0; --i)
    }  while (incremented);
    data_im[index] = val;
  }  // CUDA_KERNEL_LOOP(index, n)
}

template <typename Dtype>
void col2im_nd_gpu(const Dtype* data_col, const int num_spatial_axes,
    const int im_size, const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, Dtype* data_im) {
  // num_axes should be smaller than block size
  DCHECK_LT(num_spatial_axes, CAFFE_CUDA_NUM_THREADS);
  switch (num_spatial_axes) {
  case 1:
    col2im_nd_gpu_kernel<Dtype, 1>  // NOLINT_NEXT_LINE(whitespace/operators)
          <<<CAFFE_GET_BLOCKS(im_size), CAFFE_CUDA_NUM_THREADS>>>(
          im_size, data_col, im_shape, col_shape,
          kernel_shape, pad, stride, dilation, data_im);
    break;
  case 2:
    col2im_nd_gpu_kernel<Dtype, 2>  // NOLINT_NEXT_LINE(whitespace/operators)
          <<<CAFFE_GET_BLOCKS(im_size), CAFFE_CUDA_NUM_THREADS>>>(
          im_size, data_col, im_shape, col_shape,
          kernel_shape, pad, stride, dilation, data_im);
    break;
  case 3:
    col2im_nd_gpu_kernel<Dtype, 3>  // NOLINT_NEXT_LINE(whitespace/operators)
          <<<CAFFE_GET_BLOCKS(im_size), CAFFE_CUDA_NUM_THREADS>>>(
          im_size, data_col, im_shape, col_shape,
          kernel_shape, pad, stride, dilation, data_im);
    break;
  case 4:
    col2im_nd_gpu_kernel<Dtype, 4>  // NOLINT_NEXT_LINE(whitespace/operators)
          <<<CAFFE_GET_BLOCKS(im_size), CAFFE_CUDA_NUM_THREADS>>>(
          im_size, data_col, im_shape, col_shape,
          kernel_shape, pad, stride, dilation, data_im);
    break;
  case 5:
    col2im_nd_gpu_kernel<Dtype, 5>  // NOLINT_NEXT_LINE(whitespace/operators)
          <<<CAFFE_GET_BLOCKS(im_size), CAFFE_CUDA_NUM_THREADS>>>(
          im_size, data_col, im_shape, col_shape,
          kernel_shape, pad, stride, dilation, data_im);
    break;
  case 6:
    col2im_nd_gpu_kernel<Dtype, 6>  // NOLINT_NEXT_LINE(whitespace/operators)
          <<<CAFFE_GET_BLOCKS(im_size), CAFFE_CUDA_NUM_THREADS>>>(
          im_size, data_col, im_shape, col_shape,
          kernel_shape, pad, stride, dilation, data_im);
    break;
  case 7:
    col2im_nd_gpu_kernel<Dtype, 7>  // NOLINT_NEXT_LINE(whitespace/operators)
          <<<CAFFE_GET_BLOCKS(im_size), CAFFE_CUDA_NUM_THREADS>>>(
          im_size, data_col, im_shape, col_shape,
          kernel_shape, pad, stride, dilation, data_im);
    break;
  case 8:
    col2im_nd_gpu_kernel<Dtype, 8>  // NOLINT_NEXT_LINE(whitespace/operators)
          <<<CAFFE_GET_BLOCKS(im_size), CAFFE_CUDA_NUM_THREADS>>>(
          im_size, data_col, im_shape, col_shape,
          kernel_shape, pad, stride, dilation, data_im);
    break;
  case 9:
    col2im_nd_gpu_kernel<Dtype, 9>  // NOLINT_NEXT_LINE(whitespace/operators)
          <<<CAFFE_GET_BLOCKS(im_size), CAFFE_CUDA_NUM_THREADS>>>(
          im_size, data_col, im_shape, col_shape,
          kernel_shape, pad, stride, dilation, data_im);
    break;
  case 10:
    col2im_nd_gpu_kernel<Dtype, 10>  // NOLINT_NEXT_LINE(whitespace/operators)
          <<<CAFFE_GET_BLOCKS(im_size), CAFFE_CUDA_NUM_THREADS>>>(
          im_size, data_col, im_shape, col_shape,
          kernel_shape, pad, stride, dilation, data_im);
    break;
  default:
    LOG(FATAL) << "col2im_nd_gpu does not support computation with "
               << num_spatial_axes << " spatial axes";
  }
  CUDA_POST_KERNEL_CHECK;
}

// Explicit instantiation
template void col2im_nd_gpu<float>(const float* data_col,
    const int num_spatial_axes, const int im_size,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, float* data_im);
template void col2im_nd_gpu<double>(const double* data_col,
    const int num_spatial_axes, const int im_size,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, double* data_im);

}  // namespace caffe
