#include <vector>

#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

// Function uses casting from int to unsigned to compare if value of
// parameter a is greater or equal to zero and lower than value of
// parameter b. The b parameter is of type signed and is always positive,
// therefore its value is always lower than 0x800... where casting
// negative value of a parameter converts it to value higher than 0x800...
// The casting allows to use one condition instead of two.
//
//若a大于等于零并且小于b，返回true，否则返回false
inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
  return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}
/*
im2col_cpu将c个通道的卷积层输入图像转化为c个通道的矩阵，
按照channel*kernel_h*kernel_w一列，将一个channel x kernel_h x kernel_w 大小的图像块变成一个列。
为了能让两个一维数组直接相乘，矩阵的行值为卷积核高*卷积核宽，
也就是说，矩阵的单列表征了卷积核操作一次处理的小窗口图像信息；而矩阵的列值为卷积层
输出单通道图像高*卷积层输出单通道图像宽，表示一共要处理多少个小窗口。
im2col_cpu接收13个参数，
分别为输入数据指针(data_im)，
卷积操作处理的一个卷积组的通道数(channels)，
输入图像的高(height)与宽(width)，
原始卷积核的高(kernel_h)与宽(kernel_w)，
输入图像高(pad_h)与宽(pad_w)方向的pad，
卷积操作高(stride_h)与宽(stride_w)方向的步长，
卷积方向上面的拓展高度(dilation_h)和宽度(dilation_w)
输出矩阵数据指针(data_col)
*/

template <typename Dtype>
//这个函数的主要功能是，将原始的图像数据转换为大小为channel*kernel_h*kernel_w的一个一维数据，
//这样就可以直接和同样展开的kernel数据进行直接一维运算。
void im2col_cpu(
    const Dtype* data_im,//输入的图片数据 
    const int channels,//通道数目
    const int height, const int width,//宽高 
    const int kernel_h, const int kernel_w,//卷积核的宽高
    const int pad_h, const int pad_w,//扩充的宽高
    const int stride_h, const int stride_w,//卷积步长的宽和高
    const int dilation_h, const int dilation_w,//卷积方向上的扩展倍数
    Dtype* data_col) //输出数据 
    {
      //计算输出高度=[(图像高度+2*填充高度-dilation高度*(kernel高度-1))/stride高度+1] * [(图像宽度+2*填充宽度-dilation宽度*(kernel宽度-1))/stride宽度+1]
      //注意这里的kernel_h - 1是为了保证奇偶数情况下的正确性
    const int output_h = (height + 2 * pad_h -(dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    //同理，计算宽度
    const int output_w = (width + 2 * pad_w -(dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    //计算通道大小，为了方便将几个通道上面的数据，放在一起进行操作，相当于将三维压缩到一维
    const int channel_size = height * width;
  //根据维度来遍历图像
  for (int channel = channels; channel--; data_im += channel_size) {//这里的channel一般是rgb通道数目
    //根据卷积核的大小来进行遍历卷积，对卷积核上面的每一个数据点计算对应图像数据的积
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {//按行遍历
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {//按列遍历
        //计算对应的输入行，计算对应的输入图像的y坐标，由pad_h和卷积内核的坐标所确定
        int input_row = -pad_h + kernel_row * dilation_h;
        //按照行来对图像进行读取计算
        for (int output_rows = output_h; output_rows; output_rows--) {
          //检查input_row是否越界，到图像的外面
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {//如果越界，即没有输入图像没有数据
            //按列读取数据，确定输入的坐标
            for (int output_cols = output_w; output_cols; output_cols--) {
              //将所有输出全部置0 相当于预先分配内存
              *(data_col++) = 0;
            }
          } else {//没有越界，存在输入数据，进行卷积运算
            //计算输入的列坐标即，x坐标
            int input_col = -pad_w + kernel_col * dilation_w;
            //按照x方向进行反向遍历
            for (int output_col = output_w; output_col; output_col--) {
              //检查是否越界
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {//没有越界
                //计算对应的原始输入的坐标，重新映射到data_col
                *(data_col++) = data_im[input_row * width + input_col];
              } else {
                //越界置0
                *(data_col++) = 0;
              }
              //y=y+stride_w,按照步长进行遍历
              input_col += stride_w;
            }
          }
          //x=x+stride_h,按照步长进行遍历
          input_row += stride_h;
        }
      }
    }
  }
}

// Explicit instantiation
//特例化单双精度的函数
template void im2col_cpu<float>(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    float* data_col);
template void im2col_cpu<double>(const double* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    double* data_col);

//n维通用im2col以及col2im的实现
//两个功能一起实现了就是编解码在一起
//n维卷积的实现与二维卷积的实现很类似
//d_offset 对应于im2col中的h_offset和w_offset是一个输入图像的channel 乘以kernel_size大小的图像块的偏移量(kernel_size下面的代码有定义)
//d_iter对应于im2col中内层for循环的h和w，是经过im2colnd处理过的col_buff中的偏移
//d_pad对应于im2col中内层for循环的h_pad和w_pad，是输入的原始图像中的偏移


template <typename Dtype>
inline void im2col_nd_core_cpu(
    const Dtype* data_input,//输入数据 
    const bool im2col,//是否为im2col还是col转化为img
    const int num_spatial_axes,//空间轴信息，总共的维度
    const int* im_shape, //im的维度数据
    const int* col_shape,//矩阵的维度数据
    const int* kernel_shape, //卷积内核的形状数据 
    const int* pad, //边界扩充
    const int* stride,//步长
    const int* dilation, //缩放
    Dtype* data_output)  //输出数据
    {
      //是否将图像转换为矩阵

    if (!im2col) {//不是，根据原始图像的大小进行数据内存的分配
      //获取图像数据的维度，即图片的数量
      int im_size = im_shape[0];
      //遍历维度
      for (int i = 0; i < num_spatial_axes; ++i) {
        //计算总的数据存储数组长度
        im_size *= im_shape[1 + i];
    }
    //分配输出的内存空间
    caffe_set(im_size, Dtype(0), data_output);
  }
  //初始化卷积核的尺寸
  int kernel_size = 1;
  //按照数据维度，获取卷积核的总数组大小
  for (int i = 0; i < num_spatial_axes; ++i) {
    kernel_size *= kernel_shape[i];
  }
  // channels_col = inputchannel(输入图像的channel)*kernel_size 
  //多张数据的总维数
  const int channels_col = col_shape[0];
  //设置每个维度上的偏移量，主要是为了进行一维寻址
  vector<int> d_offset(num_spatial_axes, 0);
  //col_buffer中的偏移量
  vector<int> d_iter(num_spatial_axes, 0);
  //根据维度进行遍历
  for (int c_col = 0; c_col < channels_col; ++c_col) {
    // Loop over spatial axes in reverse order to compute a per-axis offset.
    // 计算n维kernel上的offset,与im2col中对应的代码一样的道理  
    // 只不过这里是n维了，所以用d_offset来表示  
    // 注意，这里用逆序来进行计算得到每个轴的偏移 
    // 初始化偏移为当前维度数
    int offset = c_col;
    //逆序计算每个维度上的偏移
    for (int d_i = num_spatial_axes - 1; d_i >= 0; --d_i) {
      if (d_i < num_spatial_axes - 1) {
        //偏移数目=原始数据的宽高/对应的维度
        offset /= kernel_shape[d_i + 1];
      }
      
      d_offset[d_i] = offset % kernel_shape[d_i];
    }
    //即创建一个宽度为kernel_shape各个维度乘积的矩阵，其中offset是矩阵的行数
    //d_offset是矩阵的列数

    //循环增加
    for (bool incremented = true; incremented; ) {
      // Loop over spatial axes in forward order to compute the indices in the
      // image and column, and whether the index lies in the padding.
      //是经过im2colnd变换之后的col索引  
      int index_col = c_col;
      //经过im2col的im索引
      int index_im = c_col / kernel_size;
      //是否扩充
      bool is_padding = false;
      //正序迭代维度

      for (int d_i = 0; d_i < num_spatial_axes; ++d_i) {
        // d是col_buff上的偏移，与d_pad相对(d_pad是原始图像上的偏移)
        
        const int d = d_iter[d_i];
        // 在d_pad是经过pad之后的col_buff中的坐标经过转换成原图中的坐标
        const int d_im = d * stride[d_i] - pad[d_i]+d_offset[d_i] * dilation[d_i];
        // 判断经过im2colnd处理的图像上的像素是否位于输入的n维图像的上的pad的那个部分
        is_padding |= d_im < 0 || d_im >= im_shape[d_i + 1];
        //col索引 计算位于col_buff中的位置(就是经过im2colnd变换之后的)  
        index_col *= col_shape[d_i + 1];
        //索引添加偏移值
        index_col += d;
        //计算位于原始图像中的位置
        index_im *= im_shape[d_i + 1];
        //计算原始坐标
        index_im += d_im;
      }
      //如果是将im转换为矩阵
      if (im2col) {
        //是否为边界
        if (is_padding) {//是设置为0
          data_output[index_col] = 0;
        } else {
          //不是获取输入的数据
          data_output[index_col] = data_input[index_im];
        }
      } else if (!is_padding) {  // col2im
        //获取输出数据
        data_output[index_im] += data_input[index_col];
      }
      // Loop over spatial axes in reverse order to choose an index,
      // like counting.
      //更新位于col_buff上的偏移d(d_iter就是所有的d存进去的)
      //设置不再增加，结束for 增加循环
      incremented = false;
      //b按照维度进行遍历
      for (int d_i = num_spatial_axes - 1; d_i >= 0; --d_i) {
        //获取每一个维度的最大数据值
        const int d_max = col_shape[d_i + 1];
        //保证每一维的索引值，小于最大维度
        DCHECK_LT(d_iter[d_i], d_max);
        //如果是最后一个数据
        if (d_iter[d_i] == d_max - 1) {
          //数据置0
          d_iter[d_i] = 0;
        } else {  // d_iter[d_i] < d_max - 1
          //d_buffer维度索引值添加
          ++d_iter[d_i];
          //重新遍历增加
          incremented = true;
          break;
        }
      }//主要是重新设置d_iter的迭代步长，直到最后遍历完成循环结束
    }  // while(incremented) {
  }  // for (int c = 0; c < channels_col; ++c) {
}

// im2col_nd_cpu只是将kIm2Col=true然后调用im2col_nd_core_cpu
template <typename Dtype>
void im2col_nd_cpu(
    const Dtype* data_im, 
    const int num_spatial_axes,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, Dtype* data_col) {
    const bool kIm2Col = true;
  im2col_nd_core_cpu(data_im, kIm2Col, num_spatial_axes, im_shape, col_shape,
                  kernel_shape, pad, stride, dilation, data_col);
}
//特例化单双浮点函数
// Explicit instantiation
template void im2col_nd_cpu<float>(const float* data_im,
    const int num_spatial_axes,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, float* data_col);
template void im2col_nd_cpu<double>(const double* data_im,
    const int num_spatial_axes,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, double* data_col);

//将矩阵转换为im
template <typename Dtype>
void col2im_cpu(
    const Dtype* data_col,//列数据 
    const int channels,//通道数目
    const int height, const int width,//高度，宽度 
    const int kernel_h, const int kernel_w,//卷积核的高度、宽度
    const int pad_h, const int pad_w,//扩充高宽
    const int stride_h, const int stride_w,//步长高宽
    const int dilation_h, const int dilation_w,//缩放高宽
    Dtype* data_im //输出数据
    ) {
    //为输出分配内存空间
    caffe_set(height * width * channels, Dtype(0), data_im);
    //计算输出的im的高度与im2col计算相似
    const int output_h = (height + 2 * pad_h -(dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    //计算输出im宽度
    const int output_w = (width + 2 * pad_w -(dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    //计算通道中的数据大小
    const int channel_size = height * width;
    //通道遍历
    for (int channel = channels; channel--; data_im += channel_size) {
      //卷积核行遍历，++y
      for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
        //卷积核列遍历，++x
        for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
          //计算输入的y坐标
          int input_row = -pad_h + kernel_row * dilation_h;
          //反向遍历
          for (int output_rows = output_h; output_rows; output_rows--) {
            //检查是否越界
            if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
              //越界，更新行数
              data_col += output_w;
            } else {
              //没有越界，计算输入总列
              int input_col = -pad_w + kernel_col * dilation_w;
              //反向遍历输出列
              for (int output_col = output_w; output_col; output_col--) {
                //检查是否越界
                if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                  //将当前的指针指向数据，映射到计算结果上，这里注意和im2col中的区别，基本上是进行了一次反向映射
                  data_im[input_row * width + input_col] += *data_col;
                }
                //移动列数据指针
                data_col++;
                //输入列坐标+步长
                input_col += stride_w;
              }
            }
            //输入行坐标+步长
          input_row += stride_h;
        }
      }
    }
  }
}

//特例化单双精度函数
// Explicit instantiation
template void col2im_cpu<float>(const float* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    float* data_im);
template void col2im_cpu<double>(const double* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    double* data_im);

template <typename Dtype>
void col2im_nd_cpu(const Dtype* data_col, const int num_spatial_axes,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, Dtype* data_im) {
  const bool kIm2Col = false;
  im2col_nd_core_cpu(data_col, kIm2Col, num_spatial_axes, im_shape, col_shape,
                     kernel_shape, pad, stride, dilation, data_im);
}

// Explicit instantiation
template void col2im_nd_cpu<float>(const float* data_col,
    const int num_spatial_axes,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, float* data_im);
template void col2im_nd_cpu<double>(const double* data_col,
    const int num_spatial_axes,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, double* data_im);


}  // namespace caffe
