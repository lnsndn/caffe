#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/concat_layer.hpp"
#include "caffe/layers/flatten_layer.hpp"
#include "caffe/layers/pooling_layer.hpp"
#include "caffe/layers/split_layer.hpp"
#include "caffe/layers/spp_layer.hpp"
#include "caffe/layers/tpp_layer.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
LayerParameter TPPLayer<Dtype>::GetPoolingParam(const int pyramid_level,
      const int bottom_h, const int bottom_w, const TPPParameter& tpp_param) {
  LayerParameter pooling_param;
  //the number of bins is always equal to the pyramid level
  //(i.e. we can skip the num_bins paramter as used in SPPLayer and
  //proceed with calculating the kernel_width as height is given implicitely
  // as well)

  // find padding and kernel size so that the pooling is
  // performed across the entire image
  int kernel_w = ceil(bottom_w / static_cast<double>(pyramid_level));
  // remainder_w is the min number of pixels that need to be padded before
  // entire image height is pooled over with the chosen kernel dimension
  int remainder_w = kernel_w * pyramid_level - bottom_w;
  // pooling layer pads (2 * pad_h) pixels on the top and bottom of the
  // image.
  int pad_w = (remainder_w + 1) / 2;

  // we can skip the height calculations for kernel size and padding
  // as pooling is only performed along the horizontal axis of each blob
  pooling_param.mutable_pooling_param()->set_pad_h(0);
  pooling_param.mutable_pooling_param()->set_pad_w(pad_w);
  pooling_param.mutable_pooling_param()->set_kernel_h(bottom_h);
  pooling_param.mutable_pooling_param()->set_kernel_w(kernel_w);
  pooling_param.mutable_pooling_param()->set_stride_h(bottom_h);
  pooling_param.mutable_pooling_param()->set_stride_w(kernel_w);

  switch (tpp_param.pool()) {
  case TPPParameter_PoolMethod_MAX:
    pooling_param.mutable_pooling_param()->set_pool(
        PoolingParameter_PoolMethod_MAX);
    break;
  case TPPParameter_PoolMethod_AVE:
    pooling_param.mutable_pooling_param()->set_pool(
        PoolingParameter_PoolMethod_AVE);
    break;
  case TPPParameter_PoolMethod_STOCHASTIC:
    pooling_param.mutable_pooling_param()->set_pool(
        PoolingParameter_PoolMethod_STOCHASTIC);
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }

  return pooling_param;
}

template <typename Dtype>
void TPPLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  TPPParameter tpp_param = this->layer_param_.tpp_param();

  //copy pyramid_levels to local variable
  std::copy(tpp_param.pyramid_layer().begin(),
      tpp_param.pyramid_layer().end(),
      std::back_inserter(pyramid_levels_));

  this->num_ = bottom[0]->num();
  this->channels_ = bottom[0]->channels();
  this->bottom_h_ = bottom[0]->height();
  this->bottom_w_ = bottom[0]->width();
  this->reshaped_first_time_ = false;
  CHECK_GT(this->bottom_h_, 0) << "Input dimensions cannot be zero.";
  CHECK_GT(this->bottom_w_, 0) << "Input dimensions cannot be zero.";
  //pyramid_height_ is necessary for later Forward calls
  this->pyramid_height_ = pyramid_levels_.size();

  this->split_top_vec_.clear();
  this->pooling_bottom_vecs_.clear();
  this->pooling_layers_.clear();
  this->pooling_top_vecs_.clear();
  this->pooling_outputs_.clear();
  this->flatten_layers_.clear();
  this->flatten_top_vecs_.clear();
  this->flatten_outputs_.clear();
  this->concat_bottom_vec_.clear();

  if (this->pyramid_height_ == 1) {
    // pooling layer setup
    LayerParameter pooling_param = GetPoolingParam(pyramid_levels_[0],
        this->bottom_h_, this->bottom_w_, tpp_param);
    this->pooling_layers_.push_back(shared_ptr<PoolingLayer<Dtype> > (
        new PoolingLayer<Dtype>(pooling_param)));
    this->pooling_layers_[0]->SetUp(bottom, top);
    return;
  }
  // split layer output holders setup
  for (int i = 0; i < this->pyramid_height_; ++i) {
    this->split_top_vec_.push_back(new Blob<Dtype>());
  }

  // split layer setup
  LayerParameter split_param;
  this->split_layer_.reset(new SplitLayer<Dtype>(split_param));
  this->split_layer_->SetUp(bottom, this->split_top_vec_);

  for (int i = 0; i < this->pyramid_height_; ++i) {
    // pooling layer input holders setup
    this->pooling_bottom_vecs_.push_back(new vector<Blob<Dtype>*>);
    this->pooling_bottom_vecs_[i]->push_back(this->split_top_vec_[i]);

    // pooling layer output holders setup
    this->pooling_outputs_.push_back(new Blob<Dtype>());
    this->pooling_top_vecs_.push_back(new vector<Blob<Dtype>*>);
    this->pooling_top_vecs_[i]->push_back(this->pooling_outputs_[i]);

    // pooling layer setup
    LayerParameter pooling_param = GetPoolingParam(pyramid_levels_[i],
        this->bottom_h_, this->bottom_w_, tpp_param);

    this->pooling_layers_.push_back(shared_ptr<PoolingLayer<Dtype> > (
        new PoolingLayer<Dtype>(pooling_param)));
    this->pooling_layers_[i]->SetUp(*this->pooling_bottom_vecs_[i],
        *this->pooling_top_vecs_[i]);

    // flatten layer output holders setup
    this->flatten_outputs_.push_back(new Blob<Dtype>());
    this->flatten_top_vecs_.push_back(new vector<Blob<Dtype>*>);
    this->flatten_top_vecs_[i]->push_back(this->flatten_outputs_[i]);

    // flatten layer setup
    LayerParameter flatten_param;
    this->flatten_layers_.push_back(new FlattenLayer<Dtype>(flatten_param));
    this->flatten_layers_[i]->SetUp(*this->pooling_top_vecs_[i], *this->flatten_top_vecs_[i]);

    // concat layer input holders setup
    this->concat_bottom_vec_.push_back(this->flatten_outputs_[i]);
  }

  // concat layer setup
  LayerParameter concat_param;
  this->concat_layer_.reset(new ConcatLayer<Dtype>(concat_param));
  this->concat_layer_->SetUp(this->concat_bottom_vec_, top);

  //---------------------------
  //DEBUG STUFF!!!
  this->split_layer_->SetShared(false);
  for(int i = 0; i < this->pyramid_height_; ++i)
  {
      this->pooling_layers_[i]->SetShared(false);
      this->flatten_layers_[i]->SetShared(false);
  }
  this->concat_layer_->SetShared(false);
  //---------------------------
}

template <typename Dtype>
void TPPLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  // Do nothing if bottom shape is unchanged since last Reshape
  if (this->num_ == bottom[0]->num() && this->channels_ == bottom[0]->channels() &&
      this->bottom_h_ == bottom[0]->height() && this->bottom_w_ == bottom[0]->width() &&
      this->reshaped_first_time_) {
    return;
  }
  this->num_ = bottom[0]->num();
  this->channels_ = bottom[0]->channels();
  this->bottom_h_ = bottom[0]->height();
  this->bottom_w_ = bottom[0]->width();
  this->reshaped_first_time_ = true;
  TPPParameter tpp_param = this->layer_param_.tpp_param();
  if (this->pyramid_height_ == 1) {
    LayerParameter pooling_param = GetPoolingParam(pyramid_levels_[0],
        this->bottom_h_,this->bottom_w_, tpp_param);
    this->pooling_layers_[0].reset(new PoolingLayer<Dtype>(pooling_param));
    this->pooling_layers_[0]->SetUp(bottom, top);
    this->pooling_layers_[0]->Reshape(bottom, top);
    return;
  }
  this->split_layer_->Reshape(bottom, this->split_top_vec_);
  for (int i = 0; i < pyramid_levels_.size(); ++i) {
    LayerParameter pooling_param = GetPoolingParam(
        pyramid_levels_[i], this->bottom_h_, this->bottom_w_, tpp_param);

    this->pooling_layers_[i].reset(
        new PoolingLayer<Dtype>(pooling_param));
    this->pooling_layers_[i]->SetUp(
        *this->pooling_bottom_vecs_[i], *this->pooling_top_vecs_[i]);
    this->pooling_layers_[i]->Reshape(
        *this->pooling_bottom_vecs_[i], *this->pooling_top_vecs_[i]);
    this->flatten_layers_[i]->Reshape(
        *this->pooling_top_vecs_[i], *this->flatten_top_vecs_[i]);
  }
  this->concat_layer_->Reshape(this->concat_bottom_vec_, top);
}

INSTANTIATE_CLASS(TPPLayer);
REGISTER_LAYER_CLASS(TPP);

}  // namespace caffe
