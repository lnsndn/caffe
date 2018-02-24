#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include "caffe/layers/blob_data_layer.hpp"

namespace caffe {

template <typename Dtype>
BlobDataLayer<Dtype>::BlobDataLayer(const LayerParameter& param)
    : Layer<Dtype>(param),
      prefetch_(param.data_param().prefetch()),
      prefetch_free_(), prefetch_full_(), prefetch_current_(),
      transform_param_(param.transform_param()),
      transformed_data_(), n_tops_(-1),
      offset_() {
  // init prefetch
  for (int i = 0; i < prefetch_.size(); ++i) {
    prefetch_[i].reset(new vector<Blob<Dtype>*>());
    prefetch_free_.push(prefetch_[i].get());
  }
  // init DB backend
  db_.reset(db::GetDB(param.data_param().backend()));
  db_->Open(param.data_param().source(), db::READ);
  cursor_.reset(db_->NewCursor());
}

template <typename Dtype>
BlobDataLayer<Dtype>::~BlobDataLayer() {
  this->StopInternalThread();
}

template <typename Dtype>
void BlobDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  data_transformer_.reset(
      new DataTransformer<Dtype>(transform_param_, this->phase_));
  data_transformer_->InitRand();
  n_tops_ = top.size();

  // Before starting the prefetch thread, we make cpu_data and gpu_data
  // calls so that the prefetch thread does not accidentally make simultaneous
  // cudaMalloc calls when the main thread is running. In some GPUs this
  // seems to cause failures if we do not so.
  for (int i = 0; i < prefetch_.size(); ++i) {
    for (int j = 0; j < n_tops_; ++j) {
      prefetch_[i]->push_back(new Blob<Dtype>(top.at(j)->shape()));
      (*prefetch_[i])[j]->mutable_cpu_data();
    }
  }
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    for (int i = 0; i < prefetch_.size(); ++i) {
      for (int j = 0; j < n_tops_; ++j) {
        (*prefetch_[i])[j]->mutable_gpu_data();
      }
    }
  }
#endif
  DataLayerSetUp(bottom, top);
  DLOG(INFO) << "Initializing prefetch";
  this->data_transformer_->InitRand();
  StartInternalThread();
  DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
void BlobDataLayer<Dtype>::DataLayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.data_param().batch_size();
  // Read a data point, and use it to initialize the top blobs.
  BlobDatum datum;
  datum.ParseFromString(this->cursor_->value());

  // Use data_transformer to infer the expected blob shapes from datum.
  vector<vector<int> > top_shapes =
      this->data_transformer_->InferBlobShapes(datum);
  CHECK_GT(n_tops_, 0) << "Found 0 top blobs.";
  CHECK_LE(n_tops_, top_shapes.size())
      << "BlobDatum contains more blobs than there are top blobs for this layer";
  if (datum.label_name_size() > 0) {
    LOG(INFO) << "Using the following label blobs in this order:";
    for (int i = 0; i < n_tops_-1; ++i) {
      LOG(INFO) << "   " << datum.label_name(i);
    }
    if (datum.label_size() > (n_tops_-1)) {
      LOG(WARNING) << "Skipping following label blobs:";
      for (int i = top.size()-1; i < datum.label_name_size(); ++i) {
        LOG(INFO) << "   " << datum.label_name(i);
      }
    }
  }
  // Reshape tops according to the batch_size
  for (int i = 0; i < n_tops_; ++i) {
    top_shapes[i][0] = batch_size;
    top[i]->Reshape(top_shapes[i]);
    for (int j = 0; j < prefetch_.size(); ++j) {
      vector<int> ts = top_shapes[i];
      vector<Blob<Dtype>*>* vec_pointer = prefetch_[j].get();
      Blob<Dtype>* bl = (*vec_pointer)[i];
      bl->Reshape(ts);
    }
  }
  transformed_data_.Reshape(top_shapes[0]);
}

template <typename Dtype>
void BlobDataLayer<Dtype>::InternalThreadEntry() {
#ifndef CPU_ONLY
  cudaStream_t stream;
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  }
#endif

  try {
    while (!must_stop()) {
      vector<Blob<Dtype>*>* batch = prefetch_free_.pop();
      load_batch(batch);
#ifndef CPU_ONLY
      if (Caffe::mode() == Caffe::GPU) {
        for (int i = 0; i < batch->size(); ++i) {
          (*batch)[i]->data().get()->async_gpu_push(stream);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
      }
#endif
      prefetch_full_.push(batch);
    }
  } catch (boost::thread_interrupted&) {
    // Interrupted exception is expected on shutdown
  }
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaStreamDestroy(stream));
  }
#endif
}

template <typename Dtype>
bool BlobDataLayer<Dtype>::Skip() {
  int size = Caffe::solver_count();
  int rank = Caffe::solver_rank();
  bool keep = (offset_ % size) == rank ||
              // In test mode, only rank 0 runs, so avoid skipping
              this->layer_param_.phase() == TEST;
  return !keep;
}

template<typename Dtype>
void BlobDataLayer<Dtype>::Next() {
  cursor_->Next();
  if (!cursor_->valid()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Restarting data prefetching from start.";
    cursor_->SeekToFirst();
  }
  offset_++;
}

// This function is called on prefetch thread
template<typename Dtype>
void BlobDataLayer<Dtype>::load_batch(vector<Blob<Dtype>*>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK((*batch)[0]->count());
  CHECK(this->transformed_data_.count());
  const int batch_size = this->layer_param_.data_param().batch_size();

  BlobDatum datum;
  Blob<Dtype> datum_cur_blob;
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    while (this->Skip()) {
      this->Next();
    }
    datum.ParseFromString(this->cursor_->value());
    read_time += timer.MicroSeconds();

    if (item_id == 0) {
      // Reshape according to the first datum of each batch
      // on single input batches allows for inputs of varying dimension.
      // Use data_transformer to infer the expected blob shape from datum.
      vector<vector<int> > top_shapes = this->data_transformer_->InferBlobShapes(datum);
      this->transformed_data_.Reshape(top_shapes[0]);
      // Reshape batch according to the batch_size.
      for (int i = 0; i < n_tops_; ++i) {
        top_shapes[i][0] = batch_size;
        (*batch)[i]->Reshape(top_shapes[i]);
      }
    }
    // Apply data transformations (mirror, scale, crop...)
    timer.Start();
    int data_offset = (*batch)[0]->offset(item_id);
    Dtype* top_data = (*batch)[0]->mutable_cpu_data();
    this->transformed_data_.set_cpu_data(top_data + data_offset);
    /*datum_cur_blob.FromProto(datum.data());
    vector<int> shape = datum_cur_blob.shape();
    shape.insert(shape.begin(), 1);
    datum_cur_blob.Reshape(shape);*/
    this->data_transformer_->Transform(datum.data(), &(this->transformed_data_));
    // Copy label.
    for (int i = 0; i < n_tops_-1; ++i) {
      datum_cur_blob.FromProto(datum.label(i));
      vector<int> shape = datum_cur_blob.shape();
      shape.insert(shape.begin(), 1);
      datum_cur_blob.Reshape(shape);
      Blob<Dtype>* batch_label_blob = (*batch)[i+1];
      int batch_label_offset = batch_label_blob->offset(item_id);
      int batch_label_size = batch_label_blob->count(1);
      int datum_label_size = datum_cur_blob.count(1);
      CHECK_EQ(batch_label_size, datum_label_size);
      Dtype* batch_label_data = batch_label_blob->mutable_cpu_data();
      batch_label_data += batch_label_offset;
      const Dtype* datum_label_data = datum_cur_blob.cpu_data();
      caffe_copy(datum_label_size, datum_label_data, batch_label_data);
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

template <typename Dtype>
void BlobDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  if (prefetch_current_) {
    prefetch_free_.push(prefetch_current_);
  }
  prefetch_current_ = prefetch_full_.pop("Waiting for data");
  // Reshape to loaded data.
  for (int i = 0; i < top.size(); ++i) {
    top[i]->Reshape((*prefetch_current_)[i]->shape());
    top[i]->set_cpu_data((*prefetch_current_)[i]->mutable_cpu_data());
  }
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(BlobDataLayer, Forward);
#endif

INSTANTIATE_CLASS(BlobDataLayer);
REGISTER_LAYER_CLASS(BlobData);

}  // namespace caffe
