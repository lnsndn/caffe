#ifndef CAFFE_TPP_LAYER_HPP_
#define CAFFE_TPP_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layers/spp_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class TPPLayer : public SPPLayer<Dtype> {
 public:
  explicit TPPLayer(const LayerParameter& param)
      : SPPLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "TPP"; }

 protected:  
  // calculates the kernel and stride dimensions for the pooling layer,
  // returns a correctly configured LayerParameter for a PoolingLayer
  virtual LayerParameter GetPoolingParam(const int pyramid_level,
      const int bottom_h, const int bottom_w, const TPPParameter& tpp_param);

  vector<int> pyramid_levels_;
};

}  // namespace caffe

#endif  // CAFFE_TPP_LAYER_HPP_
