#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <algorithm>
#include <map>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/multilabel_data_layer.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

template <typename Dtype>
MultilabelDataLayer<Dtype>::MultilabelDataLayer(const LayerParameter& param)
    : DataLayer<Dtype>(param) {}

template <typename Dtype>
void MultilabelDataLayer<Dtype>::DataLayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.data_param().batch_size();
  // Read a data point, and use it to initialize the top blob.
  MultilabelDatum ml_datum;
  ml_datum.ParseFromString(this->cursor_->value());

  // Use data_transformer to infer the expected blob shape from datum.
  std::tuple<vector<int>, vector<int> > top_shapes =
      this->data_transformer_->InferBlobShape(ml_datum);
  vector<int> top_shape_data = std::get<0>(top_shapes);  
  this->transformed_data_.Reshape(top_shape_data);
  // Reshape top[0] and prefetch_data according to the batch_size.
  top_shape_data[0] = batch_size;
  top[0]->Reshape(top_shape_data);
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->data_.Reshape(top_shape_data);
  }
  LOG_IF(INFO, Caffe::root_solver())
      << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  if (this->output_labels_) {
    vector<int> top_shape_label = std::get<1>(top_shapes);
    top_shape_label[0] = batch_size;
    top[1]->Reshape(top_shape_label);
    for (int i = 0; i < this->prefetch_.size(); ++i) {
      this->prefetch_[i]->label_.Reshape(top_shape_label);
    }
  }
}

// This function is called on prefetch thread
template<typename Dtype>
void MultilabelDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  const int batch_size = this->layer_param_.data_param().batch_size();

  MultilabelDatum ml_datum;
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    while (this->Skip()) {
      this->Next();
    }
    ml_datum.ParseFromString(this->cursor_->value());
    read_time += timer.MicroSeconds();

    if (item_id == 0) {
      // Reshape according to the first datum of each batch
      // on single input batches allows for inputs of varying dimension.
      // Use data_transformer to infer the expected blob shape from datum.
      std::tuple<vector<int>, vector<int> > top_shapes =
          this->data_transformer_->InferBlobShape(ml_datum);
      vector<int> top_shape_data = std::get<0>(top_shapes);
      this->transformed_data_.Reshape(top_shape_data);
      // Reshape batch according to the batch_size.
      top_shape_data[0] = batch_size;
      batch->data_.Reshape(top_shape_data);

      //labels
      if (this->output_labels_) {
        vector<int> top_shape_labels = std::get<1>(top_shapes);
        top_shape_labels[0] = batch_size;
        batch->label_.Reshape(top_shape_labels);
      }
    }

    // Apply data transformations (mirror, scale, crop...)
    timer.Start();
    int data_offset = batch->data_.offset(item_id);
    Dtype* top_data = batch->data_.mutable_cpu_data();
    this->transformed_data_.set_cpu_data(top_data + data_offset);
    this->data_transformer_->Transform(ml_datum.data(), &(this->transformed_data_));
    // Copy label.
    if (this->output_labels_) {
      int label_offset = batch->label_.offset(item_id);
      int label_size = batch->label_.count(1);
      Dtype* top_label = batch->label_.mutable_cpu_data();
      const string& label_data = ml_datum.label().data();
      const bool has_uint8 = label_data.size() > 0;
      for (int i = 0; i < label_size; ++i) {
        if (has_uint8) {
          top_label[label_offset + i] =
              static_cast<Dtype>(static_cast<uint8_t>(label_data[i]));
        } else {
          top_label[label_offset + i] = ml_datum.label().float_data(i);
        }
      }
    }
    trans_time += timer.MicroSeconds();
    this->Next();
  }
  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(MultilabelDataLayer);
REGISTER_LAYER_CLASS(MultilabelData);

}  // namespace caffe
