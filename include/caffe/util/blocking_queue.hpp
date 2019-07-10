#ifndef CAFFE_UTIL_BLOCKING_QUEUE_HPP_
#define CAFFE_UTIL_BLOCKING_QUEUE_HPP_

#include <queue>
#include <string>

namespace caffe {
//锁队列模板类，这里主要是等待完成的事物队列
template<typename T>
class BlockingQueue {
 public:
  explicit BlockingQueue();

  void push(const T& t);

  bool try_pop(T* t);

  // This logs a message if the threads needs to be blocked
  // useful for detecting e.g. when data feeding is too slow
  //当线程被封锁的时候，发送一条消息
  T pop(const string& log_on_wait = "");

  bool try_peek(T* t);

  // Return element without removing it
  //返回元素从拷贝
  T peek();

  size_t size() const;

 protected:
  /**
   Move synchronization fields out instead of including boost/thread.hpp
   to avoid a boost/NVCC issues (#1009, #1010) on OSX. Also fails on
   Linux CUDA 7.0.18.
   */
	 //定义异步类
  class sync;
	//队列
  std::queue<T> queue_;
  //异步指针
  shared_ptr<sync> sync_;

DISABLE_COPY_AND_ASSIGN(BlockingQueue);
};
//这个类主要是一个任务排队等待类
}  // namespace caffe

#endif
