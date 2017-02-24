#ifndef CAFFE_MULTILABEL_DATA_LAYER_HPP_
#define CAFFE_MULTILABEL_DATA_LAYER_HPP_

#include <vector>

#include "caffe/layers/data_layer.hpp"

namespace caffe {

template <typename Dtype>
class MultilabelDataLayer : public DataLayer<Dtype> {
 public:
  explicit MultilabelDataLayer(const LayerParameter& param);
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // DataLayer uses DataReader instead for sharing for parallelism
  virtual inline bool ShareInParallel() const { return false; }
  virtual inline const char* type() const { return "Data"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  virtual void load_batch(Batch<Dtype>* batch);
};

}  // namespace caffe

#endif  // CAFFE_MULTILABEL_DATA_LAYER_HPP_
