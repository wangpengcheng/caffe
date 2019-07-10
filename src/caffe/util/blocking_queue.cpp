#include <boost/thread.hpp>
#include <string>

#include "caffe/layers/base_data_layer.hpp"
#include "caffe/parallel.hpp"
#include "caffe/util/blocking_queue.hpp"
//队列的实现函数
namespace caffe {

template<typename T>
//异步类的定义
class BlockingQueue<T>::sync {
 public:
	 //互斥锁
  mutable boost::mutex mutex_;
  //环境信号
  boost::condition_variable condition_;
};

template<typename T>
//基本构造函数
BlockingQueue<T>::BlockingQueue()
    : sync_(new sync()) {
}
//在队列中添加元素
template<typename T>
void BlockingQueue<T>::push(const T& t) {
  //同步加锁
	boost::mutex::scoped_lock lock(sync_->mutex_);
  //将元素添加到队伍中
	queue_.push(t);
	//解锁
	lock.unlock();
	//发射环境改变，让工作线程开始工作
  sync_->condition_.notify_one();
}
//取出队列中的作业
template<typename T>
bool BlockingQueue<T>::try_pop(T* t) {
  boost::mutex::scoped_lock lock(sync_->mutex_);

  if (queue_.empty()) {
    return false;
  }
	//将取出的值，传递给t
  *t = queue_.front();
  queue_.pop();
  return true;
}

template<typename T>
T BlockingQueue<T>::pop(const string& log_on_wait) {
  boost::mutex::scoped_lock lock(sync_->mutex_);
	
  while (queue_.empty()) {
    if (!log_on_wait.empty()) {
      LOG_EVERY_N(INFO, 1000)<< log_on_wait;
    }
    //等待新值
	sync_->condition_.wait(lock);
  }

  T t = queue_.front();
  queue_.pop();
  return t;
}

template<typename T>
bool BlockingQueue<T>::try_peek(T* t) {
  boost::mutex::scoped_lock lock(sync_->mutex_);

  if (queue_.empty()) {
    return false;
  }

  *t = queue_.front();
  return true;
}

template<typename T>
//返回前一个
T BlockingQueue<T>::peek() {
	//全局信号变量加锁
	boost::mutex::scoped_lock lock(sync_->mutex_);
	//队列为空则一直等待，直到存在新任务
  while (queue_.empty()) {
    sync_->condition_.wait(lock);
  }

  return queue_.front();
}

template<typename T>
size_t BlockingQueue<T>::size() const {
  boost::mutex::scoped_lock lock(sync_->mutex_);
  return queue_.size();
}

template class BlockingQueue<Batch<float>*>;
template class BlockingQueue<Batch<double>*>;

}  // namespace caffe
