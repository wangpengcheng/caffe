/**
 * @brief A layer factory that allows one to register layers.
 * During runtime, registered layers can be called by passing a LayerParameter
 * protobuffer to the CreateLayer function:
 *
 *     LayerRegistry<Dtype>::CreateLayer(param);
 *
 * There are two ways to register a layer. Assuming that we have a layer like:
 *
 *   template <typename Dtype>
 *   class MyAwesomeLayer : public Layer<Dtype> {
 *     // your implementations
 *   };
 *
 * and its type is its C++ class name, but without the "Layer" at the end
 * ("MyAwesomeLayer" -> "MyAwesome").
 *
 * If the layer is going to be created simply by its constructor, in your c++
 * file, add the following line:
 *
 *    REGISTER_LAYER_CLASS(MyAwesome);
 *
 * Or, if the layer is going to be created by another creator function, in the
 * format of:
 *
 *    template <typename Dtype>
 *    Layer<Dtype*> GetMyAwesomeLayer(const LayerParameter& param) {
 *      // your implementation
 *    }
 *
 * (for example, when your layer has multiple backends, see GetConvolutionLayer
 * for a use case), then you can register the creator function instead, like
 *
 * REGISTER_LAYER_CREATOR(MyAwesome, GetMyAwesomeLayer)
 *
 * Note that each layer type should only be registered once.
 */
//layer的工厂模式类
#ifndef CAFFE_LAYER_FACTORY_H_
#define CAFFE_LAYER_FACTORY_H_

#include <map>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class Layer;
//layer注册工厂类
template <typename Dtype>
class LayerRegistry {
 public:
  //定义函数指针，输入layerParameter输出layer的共享指针
  typedef shared_ptr<Layer<Dtype> > (*Creator)(const LayerParameter&);
  //map映射将字符串映射到Creator上
  typedef std::map<string, Creator> CreatorRegistry;
  //静态注册函数，主要是返回CreatorRegistry
  static CreatorRegistry& Registry() {
    static CreatorRegistry* g_registry_ = new CreatorRegistry();
    return *g_registry_;
  }

//添加注册函数
  // Adds a creator.
  static void AddCreator(const string& type, Creator creator) {
    CreatorRegistry& registry = Registry();
    //检查是否已经存在过
    CHECK_EQ(registry.count(type), 0)
        << "Layer type " << type << " already registered.";
    registry[type] = creator;
  }

  // Get a layer using a LayerParameter.
  //创建layer类的静态函数
  static shared_ptr<Layer<Dtype> > CreateLayer(const LayerParameter& param) {
    if (Caffe::root_solver()) {
      LOG(INFO) << "Creating layer " << param.name();
    }
    //获取layer类型
    const string& type = param.type();
    //获取注册句柄
    CreatorRegistry& registry = Registry();
    //查看是否错误
    CHECK_EQ(registry.count(type), 1) << "Unknown layer type: " << type
        << " (known types: " << LayerTypeListString() << ")";
    //返回layer
    return registry[type](param);
  }
//已经注册的layer列表
  static vector<string> LayerTypeList() {
    CreatorRegistry& registry = Registry();
    vector<string> layer_types;
    for (typename CreatorRegistry::iterator iter = registry.begin();
         iter != registry.end(); ++iter) {
      layer_types.push_back(iter->first);
    }
    return layer_types;
  }

 private:
  // Layer registry should never be instantiated - everything is done with its
  // static variables.
  LayerRegistry() {}
  //将所有的类型转换为字符串
  static string LayerTypeListString() {
    vector<string> layer_types = LayerTypeList();
    string layer_types_str;
    for (vector<string>::iterator iter = layer_types.begin();
         iter != layer_types.end(); ++iter) {
      if (iter != layer_types.begin()) {
        layer_types_str += ", ";
      }
      layer_types_str += *iter;
    }
    return layer_types_str;
  }
};
//模板注册器类
template <typename Dtype>
class LayerRegisterer {
 public:
  LayerRegisterer(const string& type,
                  shared_ptr<Layer<Dtype> > (*creator)(const LayerParameter&)) {
    // LOG(INFO) << "Registering layer type: " << type;
    LayerRegistry<Dtype>::AddCreator(type, creator);
  }
};
//使用define模板

#define REGISTER_LAYER_CREATOR(type, creator)                                  \
  static LayerRegisterer<float> g_creator_f_##type(#type, creator<float>);     \
  static LayerRegisterer<double> g_creator_d_##type(#type, creator<double>)    \
//创建相关类
#define REGISTER_LAYER_CLASS(type)                                             \
  template <typename Dtype>                                                    \
  shared_ptr<Layer<Dtype> > Creator_##type##Layer(const LayerParameter& param) \
  {                                                                            \
    return shared_ptr<Layer<Dtype> >(new type##Layer<Dtype>(param));           \
  }                                                                            \
  REGISTER_LAYER_CREATOR(type, Creator_##type##Layer)

}  // namespace caffe
//这个文件主要是使用工厂模式，可以自由的选择和添加类
#endif  // CAFFE_LAYER_FACTORY_H_
