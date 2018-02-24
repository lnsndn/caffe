#ifndef CAFFE_BLOB_DATA_LAYER_HPP_
#define CAFFE_BLOB_DATA_LAYER_HPP_

#include <vector>

#include <boost/thread.hpp>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
//#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/blocking_queue.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

template <typename Dtype>
class BlobDataLayer : public Layer<Dtype>, public InternalThread {
 public:
  explicit BlobDataLayer(const LayerParameter& param);
  virtual ~BlobDataLayer();
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // Data layers should be shared by multiple solvers in parallel
  virtual inline bool ShareInParallel() const { return true; }
  virtual inline const char* type() const { return "BlobData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  // Data layers have no bottoms, so reshaping is trivial.
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

 protected:
  void Next();
  bool Skip();
  virtual void InternalThreadEntry();
  virtual void load_batch(vector<Blob<Dtype>*>* batch);

  vector<shared_ptr<vector<Blob<Dtype>*> > > prefetch_;
  BlockingQueue<vector<Blob<Dtype>*>*> prefetch_free_;
  BlockingQueue<vector<Blob<Dtype>*>*> prefetch_full_;
  vector<Blob<Dtype>*>* prefetch_current_;

  TransformationParameter transform_param_;
  shared_ptr<DataTransformer<Dtype> > data_transformer_;
  Blob<Dtype> transformed_data_;
  int n_tops_;

  shared_ptr<db::DB> db_;
  shared_ptr<db::Cursor> cursor_;
  uint64_t offset_;
};

}  // namespace caffe

#endif  // CAFFE_BLOB_DATA_LAYER_HPP_
