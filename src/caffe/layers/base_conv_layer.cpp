#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/base_conv_layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Configure the kernel size, padding, stride, and inputs.
  ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  force_nd_im2col_ = conv_param.force_nd_im2col();
  channel_axis_ = bottom[0]->CanonicalAxisIndex(conv_param.axis());
  const int first_spatial_axis = channel_axis_ + 1;
  const int num_axes = bottom[0]->num_axes();
  num_spatial_axes_ = num_axes - first_spatial_axis;
  CHECK_GE(num_spatial_axes_, 0);
  vector<int> spatial_dim_blob_shape(1, std::max(num_spatial_axes_, 1));
  // Setup filter kernel dimensions (kernel_shape_).
  kernel_shape_.Reshape(spatial_dim_blob_shape);
  int* kernel_shape_data = kernel_shape_.mutable_cpu_data();
  if (conv_param.has_kernel_h() || conv_param.has_kernel_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "kernel_h & kernel_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.kernel_size_size())
        << "Either kernel_size or kernel_h/w should be specified; not both.";
    kernel_shape_data[0] = conv_param.kernel_h();
    kernel_shape_data[1] = conv_param.kernel_w();
  } else {
    const int num_kernel_dims = conv_param.kernel_size_size();
    CHECK(num_kernel_dims == 1 || num_kernel_dims == num_spatial_axes_)
        << "kernel_size must be specified once, or once per spatial dimension "
        << "(kernel_size specified " << num_kernel_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
      for (int i = 0; i < num_spatial_axes_; ++i) {
        kernel_shape_data[i] =
            conv_param.kernel_size((num_kernel_dims == 1) ? 0 : i);
      }
  }
  for (int i = 0; i < num_spatial_axes_; ++i) {
    CHECK_GT(kernel_shape_data[i], 0) << "Filter dimensions must be nonzero.";
  }
  // Setup stride dimensions (stride_).
  stride_.Reshape(spatial_dim_blob_shape);
  int* stride_data = stride_.mutable_cpu_data();
  if (conv_param.has_stride_h() || conv_param.has_stride_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "stride_h & stride_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.stride_size())
        << "Either stride or stride_h/w should be specified; not both.";
    stride_data[0] = conv_param.stride_h();
    stride_data[1] = conv_param.stride_w();
  } else {
    const int num_stride_dims = conv_param.stride_size();
    CHECK(num_stride_dims == 0 || num_stride_dims == 1 ||
          num_stride_dims == num_spatial_axes_)
        << "stride must be specified once, or once per spatial dimension "
        << "(stride specified " << num_stride_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
    const int kDefaultStride = 1;
    for (int i = 0; i < num_spatial_axes_; ++i) {
      stride_data[i] = (num_stride_dims == 0) ? kDefaultStride :
          conv_param.stride((num_stride_dims == 1) ? 0 : i);
      CHECK_GT(stride_data[i], 0) << "Stride dimensions must be nonzero.";
    }
  }
  // Setup pad dimensions (pad_).
  pad_.Reshape(spatial_dim_blob_shape);
  int* pad_data = pad_.mutable_cpu_data();
  if (conv_param.has_pad_h() || conv_param.has_pad_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "pad_h & pad_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.pad_size())
        << "Either pad or pad_h/w should be specified; not both.";
    pad_data[0] = conv_param.pad_h();
    pad_data[1] = conv_param.pad_w();
  } else {
    const int num_pad_dims = conv_param.pad_size();
    CHECK(num_pad_dims == 0 || num_pad_dims == 1 ||
          num_pad_dims == num_spatial_axes_)
        << "pad must be specified once, or once per spatial dimension "
        << "(pad specified " << num_pad_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
    const int kDefaultPad = 0;
    for (int i = 0; i < num_spatial_axes_; ++i) {
      pad_data[i] = (num_pad_dims == 0) ? kDefaultPad :
          conv_param.pad((num_pad_dims == 1) ? 0 : i);
    }
  }
  // Setup dilation dimensions (dilation_).
  dilation_.Reshape(spatial_dim_blob_shape);
  int* dilation_data = dilation_.mutable_cpu_data();
  const int num_dilation_dims = conv_param.dilation_size();
  CHECK(num_dilation_dims == 0 || num_dilation_dims == 1 ||
        num_dilation_dims == num_spatial_axes_)
      << "dilation must be specified once, or once per spatial dimension "
      << "(dilation specified " << num_dilation_dims << " times; "
      << num_spatial_axes_ << " spatial dims).";
  const int kDefaultDilation = 1;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    dilation_data[i] = (num_dilation_dims == 0) ? kDefaultDilation :
                       conv_param.dilation((num_dilation_dims == 1) ? 0 : i);
  }
  // Setup convolution flag
  const bool kDefaultPartialConv = false;
  if(this->layer_param_.convolution_param().has_partial_conv_lower()) {
    partial_conv_lower_ = this->layer_param_.convolution_param().partial_conv_lower();
    // std::cout << "\n\nsetting partial convolution to be activated!!!!: " << partial_conv_lower_ << "\n\n" << std::endl;
  }
  else {
    partial_conv_lower_ = kDefaultPartialConv;
  }

  // Special case: im2col is the identity for 1x1 convolution with stride 1
  // and no padding, so flag for skipping the buffer and transformation.
  is_1x1_ = true;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    is_1x1_ &=
        kernel_shape_data[i] == 1 && stride_data[i] == 1 && pad_data[i] == 0;
    if (!is_1x1_) { break; }
  }
  // Configure output channels and groups.
  channels_ = bottom[0]->shape(channel_axis_);
  num_output_ = this->layer_param_.convolution_param().num_output();
  CHECK_GT(num_output_, 0);
  group_ = this->layer_param_.convolution_param().group();
  CHECK_EQ(channels_ % group_, 0);
  CHECK_EQ(num_output_ % group_, 0)
      << "Number of output should be multiples of group.";
  if (reverse_dimensions()) {
    conv_out_channels_ = channels_;
    conv_in_channels_ = num_output_;
  } else {
    conv_out_channels_ = num_output_;
    conv_in_channels_ = channels_;
  }
  // Handle the parameters: weights and biases.
  // - blobs_[0] holds the filter weights
  // - blobs_[1] holds the biases (optional)
  vector<int> weight_shape(2);
  weight_shape[0] = conv_out_channels_;
  weight_shape[1] = conv_in_channels_ / group_;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    weight_shape.push_back(kernel_shape_data[i]);
  }
  bias_term_ = this->layer_param_.convolution_param().bias_term();
  vector<int> bias_shape(bias_term_, num_output_);
  if (this->blobs_.size() > 0) {
    CHECK_EQ(1 + bias_term_, this->blobs_.size())
        << "Incorrect number of weight blobs.";
    if (weight_shape != this->blobs_[0]->shape()) {
      Blob<Dtype> weight_shaped_blob(weight_shape);
      LOG(FATAL) << "Incorrect weight shape: expected shape "
          << weight_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[0]->shape_string();
    }
    if (bias_term_ && bias_shape != this->blobs_[1]->shape()) {
      Blob<Dtype> bias_shaped_blob(bias_shape);
      LOG(FATAL) << "Incorrect bias shape: expected shape "
          << bias_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[1]->shape_string();
    }
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize and fill the weights:
    // output channels x input channels per-group x kernel height x kernel width
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.convolution_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, initialize and fill the biases.
    if (bias_term_) {
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.convolution_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }
  kernel_dim_ = this->blobs_[0]->count(1);
  weight_offset_ = conv_out_channels_ * kernel_dim_ / group_;
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int first_spatial_axis = channel_axis_ + 1;
  CHECK_EQ(bottom[0]->num_axes(), first_spatial_axis + num_spatial_axes_)
      << "bottom num_axes may not change.";
  num_ = bottom[0]->count(0, channel_axis_);
  CHECK_EQ(bottom[0]->shape(channel_axis_), channels_)
      << "Input size incompatible with convolution kernel.";
  // TODO: generalize to handle inputs of different shapes.
  for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
    CHECK(bottom[0]->shape() == bottom[bottom_id]->shape())
        << "All inputs must have the same shape.";
  }
  // Shape the tops.
  bottom_shape_ = &bottom[0]->shape();
  compute_output_shape();
  vector<int> top_shape(bottom[0]->shape().begin(),
      bottom[0]->shape().begin() + channel_axis_);
  top_shape.push_back(num_output_);
  for (int i = 0; i < num_spatial_axes_; ++i) {
    top_shape.push_back(output_shape_[i]);
  }
  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(top_shape);
  }
  if (reverse_dimensions()) {
    conv_out_spatial_dim_ = bottom[0]->count(first_spatial_axis);
  } else {
    conv_out_spatial_dim_ = top[0]->count(first_spatial_axis);
  }
  col_offset_ = kernel_dim_ * conv_out_spatial_dim_;
  output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ / group_;
  // Setup input dimensions (conv_input_shape_).
  vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);
  conv_input_shape_.Reshape(bottom_dim_blob_shape);
  int* conv_input_shape_data = conv_input_shape_.mutable_cpu_data();
  for (int i = 0; i < num_spatial_axes_ + 1; ++i) {
    if (reverse_dimensions()) {
      conv_input_shape_data[i] = top[0]->shape(channel_axis_ + i);
    } else {
      conv_input_shape_data[i] = bottom[0]->shape(channel_axis_ + i);
    }
  }
  // for single channel convolution
  if(partial_conv_lower_) conv_input_shape_data[0] /= conv_in_channels_;  

  // The im2col result buffer will only hold one image at a time to avoid
  // overly large memory usage. In the special case of 1x1 convolution
  // it goes lazily unused to save memory.
  col_buffer_shape_.clear();
  //resize to perform convolution by channel if partial flag is set
  if(partial_conv_lower_) {
    col_buffer_shape_.push_back((kernel_dim_ * group_) / conv_in_channels_);
  }
  else{
    col_buffer_shape_.push_back(kernel_dim_ * group_);
  }

  for (int i = 0; i < num_spatial_axes_; ++i) {
    if (reverse_dimensions()) {
      col_buffer_shape_.push_back(input_shape(i + 1));
    } else {
      col_buffer_shape_.push_back(output_shape_[i]);
    }
  }
  col_buffer_.Reshape(col_buffer_shape_);

  bottom_dim_ = bottom[0]->count(channel_axis_);
  top_dim_ = top[0]->count(channel_axis_);
  num_kernels_im2col_ = conv_in_channels_ * conv_out_spatial_dim_;        //only used with gpu
  num_kernels_col2im_ = reverse_dimensions() ? top_dim_ : bottom_dim_;    //only used with gpu
  if(partial_conv_lower_) {
    num_kernels_im2col_ /= conv_in_channels_;
    num_kernels_col2im_ /= conv_in_channels_;    //only used with gpu
  }
  // Set up the all ones "bias multiplier" for adding biases by BLAS
  out_spatial_dim_ = top[0]->count(first_spatial_axis);
  if (bias_term_) {
    vector<int> bias_multiplier_shape(1, out_spatial_dim_);
    bias_multiplier_.Reshape(bias_multiplier_shape);
    caffe_set(bias_multiplier_.count(), Dtype(1),
        bias_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype> 
void BaseConvolutionLayer<Dtype>::get_col_from_row_major_matrix(const Dtype* matrix, Dtype* col_result, int len_row, int num_rows, int len_col, int col_number) {
  int index = 0;
  for (int row = 0; row < num_rows; ++row) {
    for (int col_pos = 0; col_pos < len_col; ++col_pos) {
      col_result[index++] = matrix[row * len_row + (col_pos + col_number * len_col)];
    }
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::add_col_to_row_major_matrix(Dtype* matrix, Dtype* add_col, int len_row, int num_rows, int len_col, int col_number) {
  int index = 0;
  for (int row = 0; row < num_rows; ++row) {
    for (int col_pos = 0; col_pos < len_col; ++col_pos) {
      matrix[row * len_row + (col_pos + col_number * len_col)] += add_col[index++];
    }
  }  
}

template <typename Dtype> 
void BaseConvolutionLayer<Dtype>::partial_forward_cpu_gemm(const Dtype* input, const Dtype* weights, Dtype* output) {
  const Dtype* col_buff;
  int input_channel_offset = reverse_dimensions() ? top_dim_ / conv_in_channels_ : bottom_dim_ / conv_in_channels_;
  int weights_per_col = kernel_dim_ / conv_in_channels_;
  Dtype gemm_beta = 0.;
  Dtype* channel_weights (new Dtype[conv_out_channels_ * weights_per_col]);

  // weight data is organized into <conv_in_channels> total columns, each of which is <conv_out_channels>
  // tall.  This 2d matrix is flattened out and stored in the weights matrix.
  // the weights channel matrix will iterate through 0 -
  int input_channels = conv_in_channels_;
  for (int channel_num = 0; channel_num < input_channels; ++channel_num) {
    get_col_from_row_major_matrix(weights, channel_weights, kernel_dim_, conv_out_channels_, weights_per_col, channel_num);
    
    if (!is_1x1_) {
      conv_in_channels_ = 1;    //this variable is used when im2col_cpu is called by conv_im2col_cpu for 2d convolution
      conv_im2col_cpu(input + channel_num * input_channel_offset, col_buffer_.mutable_cpu_data());
      conv_in_channels_ = input_channels;
      col_buff = col_buffer_.cpu_data();
    } 
    else {
      col_buff = input + channel_num * input_channel_offset;
    } 
    for (int g = 0; g < group_; ++g) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans,                         // CBLAS_TRANSPOSE TransA
                            CblasNoTrans,                         // CBLAS_TRANSPOSE TransB
                            conv_out_channels_ / group_,          // M  (# A and C rows)
                            conv_out_spatial_dim_,                // N  (# B and C columns)
                            kernel_dim_ / conv_in_channels_,      // K  (# A columns, # B rows)       
                            (Dtype)1.,                            // alpha
                            channel_weights + weight_offset_ * g, // A : m rows by k columns
                            col_buff + col_offset_ * g,           // B : k rows by n columns  
                            gemm_beta,                            // beta
                            output + output_offset_ * g);         // C : m rows by n columns = alpha * A * B + beta * C
    }
    if(channel_num == 0) gemm_beta = 1.;
  }    
  delete[] channel_weights;
}

template <typename Dtype> 
void BaseConvolutionLayer<Dtype>::full_forward_cpu_gemm(const Dtype* input, const Dtype* weights, Dtype* output, bool skip_im2col) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    if (!skip_im2col) {
      conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
    }
    col_buff = col_buffer_.cpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ / group_, 
                          conv_out_spatial_dim_, kernel_dim_, 
                          (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
                          (Dtype)0., output + output_offset_ * g);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_cpu_bias(Dtype* output,
    const Dtype* bias) {
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
      out_spatial_dim_, 1, (Dtype)1., bias, bias_multiplier_.cpu_data(),
      (Dtype)1., output);
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::partial_backward_cpu_gemm(const Dtype* output, const Dtype* weights, Dtype* input) {
  Dtype* col_buff = (is_1x1_) ? input : col_buffer_.mutable_cpu_data();
  int input_channel_offset = reverse_dimensions() ? top_dim_ / conv_in_channels_ : bottom_dim_ / conv_in_channels_;
  int weights_per_col = kernel_dim_ / conv_in_channels_;
  int input_channels = conv_in_channels_;
  Dtype* channel_weights (new Dtype[conv_out_channels_ * weights_per_col]);

  for (int channel_num = 0; channel_num < input_channels; ++channel_num) {
    get_col_from_row_major_matrix(weights, channel_weights, kernel_dim_, conv_out_channels_, weights_per_col, channel_num);
    for (int g = 0; g < group_; ++g) {
      caffe_cpu_gemm<Dtype>(CblasTrans, 
                            CblasNoTrans, 
                            kernel_dim_ / conv_in_channels_,  
                            conv_out_spatial_dim_,            
                            conv_out_channels_ / group_,               
                            (Dtype)1.,                       
                            channel_weights + weight_offset_ * g,  
                            output + output_offset_ * g,                          
                            (Dtype)0.,                       
                            col_buff + col_offset_ * g + (channel_num * input_channel_offset * is_1x1_));                       
    }
    if (!is_1x1_) {
      conv_in_channels_ = 1;
      conv_col2im_cpu(col_buff, input + channel_num * input_channel_offset);
      conv_in_channels_ = input_channels;
    }
  }
  delete[] channel_weights;
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::full_backward_cpu_gemm(const Dtype* output, const Dtype* weights, Dtype* input) {
  Dtype* col_buff = col_buffer_.mutable_cpu_data();
  if (is_1x1_) {
    col_buff = input;
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,              
                          conv_out_spatial_dim_, conv_out_channels_ / group_,     
                          (Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,                  
                          (Dtype)0., col_buff + col_offset_ * g);                
  }
  if (!is_1x1_) {
    conv_col2im_cpu(col_buff, input);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::partial_weight_cpu_gemm(const Dtype* input, const Dtype* output, Dtype* weights) {
  const Dtype* col_buff = input;
  
  int weights_per_col = kernel_dim_ / conv_in_channels_;
  int input_channel_offset = reverse_dimensions() ? top_dim_ / conv_in_channels_ : bottom_dim_ / conv_in_channels_;
  Dtype* channel_weights (new Dtype[conv_out_channels_ * weights_per_col]);
  int input_channels = conv_in_channels_;

  for (int channel_num = 0; channel_num < input_channels; ++channel_num) {
    if (!is_1x1_) {
      conv_in_channels_ = 1;    //this variable is used when im2col_cpu is called by conv_im2col_cpu
      conv_im2col_cpu(input + channel_num * input_channel_offset, col_buffer_.mutable_cpu_data());
      conv_in_channels_ = input_channels;
      col_buff = col_buffer_.cpu_data();
    } 
    else {
      col_buff = input + channel_num * input_channel_offset;
    } 
    for (int g = 0; g < group_; ++g) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans,                          // CBLAS_TRANSPOSE TransA
                            CblasTrans,                            // CBLAS_TRANSPOSE TransB
                            conv_out_channels_ / group_,           // M  (# A and C rows)
                            kernel_dim_ / conv_in_channels_,       // N  (# B and C columns)
                            conv_out_spatial_dim_,                 // K  (# A columns, # B rows)       
                            (Dtype)1.,                             // alpha
                            output + output_offset_ * g,           // A
                            col_buff + col_offset_ * g,            // B 
                            (Dtype)0.,                             // beta
                            channel_weights + weight_offset_ * g); // C
    }
    //transpose add column back to existing weights for update 
    add_col_to_row_major_matrix(weights, channel_weights, kernel_dim_, conv_out_channels_, weights_per_col , channel_num);  
  }
  delete[] channel_weights;
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::full_weight_cpu_gemm(const Dtype* input, const Dtype* output, Dtype* weights) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
    col_buff = col_buffer_.cpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
                          kernel_dim_, conv_out_spatial_dim_,
                          (Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
                          (Dtype)1., weights + weight_offset_ * g);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_cpu_bias(Dtype* bias,
    const Dtype* input) {
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1.,
      input, bias_multiplier_.cpu_data(), 1., bias);
}

#ifndef CPU_ONLY

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::partial_forward_gpu_gemm(const Dtype* input, const Dtype* weights, Dtype* output) {
  const Dtype* col_buff;
  int input_channel_offset = reverse_dimensions() ? top_dim_ / conv_in_channels_ : bottom_dim_ / conv_in_channels_;
  int weights_per_col = kernel_dim_ / conv_in_channels_;
  Dtype gemm_beta = 0.;
  int cuda_weight_col_mem_count = conv_out_channels_ * weights_per_col ;

  //transposed channel weights on gpu
  Dtype* channel_weights;
  CUDA_CHECK( cudaMalloc(&channel_weights, cuda_weight_col_mem_count * sizeof(Dtype)) ); 

  // weight data is organized into <conv_in_channels> total columns, each of which is <conv_out_channels>
  // tall.  This 2d matrix is flattened out and stored in the weights matrix.
  // the weights channel matrix will iterate through 0 -
  int input_channels = conv_in_channels_;
  for (int channel_num = 0; channel_num < input_channels; ++channel_num) {
    caffe_gpu_get_col(cuda_weight_col_mem_count, weights, channel_weights, kernel_dim_, conv_out_channels_, weights_per_col, channel_num);    
    if (!is_1x1_) {
      conv_in_channels_ = 1;    //this variable is used when im2col_cpu is called by conv_im2col_gpu for 2d convolution
      conv_im2col_gpu(input + channel_num * input_channel_offset, col_buffer_.mutable_gpu_data());
      conv_in_channels_ = input_channels;
      col_buff = col_buffer_.gpu_data();
    } 
    else {
      col_buff = input + channel_num * input_channel_offset;
    } 
    for (int g = 0; g < group_; ++g) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans,                         // CBLAS_TRANSPOSE TransA
                            CblasNoTrans,                         // CBLAS_TRANSPOSE TransB
                            conv_out_channels_ / group_,          // M  (# A and C rows)
                            conv_out_spatial_dim_,                // N  (# B and C columns)
                            kernel_dim_ / conv_in_channels_,      // K  (# A columns, # B rows)       
                            (Dtype)1.,                            // alpha
                            channel_weights + weight_offset_ * g, // A : m rows by k columns
                            col_buff + col_offset_ * g,           // B : k rows by n columns  
                            gemm_beta,                            // beta
                            output + output_offset_ * g);         // C : m rows by n columns = alpha * A * B + beta * C
      
    }
    if(channel_num == 0) gemm_beta = 1.;
  }    
  CUDA_CHECK( cudaFree(channel_weights) );  //release transposed column
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::full_forward_gpu_gemm(const Dtype* input, const Dtype* weights, Dtype* output, bool skip_im2col) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    if (!skip_im2col) {
      conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
    }
    col_buff = col_buffer_.gpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ / group_, 
                          conv_out_spatial_dim_, kernel_dim_,
                          (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
                          (Dtype)0., output + output_offset_ * g);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_gpu_bias(Dtype* output, const Dtype* bias) {
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
      out_spatial_dim_, 1, (Dtype)1., bias, bias_multiplier_.gpu_data(),
      (Dtype)1., output);
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::partial_backward_gpu_gemm(const Dtype* output, const Dtype* weights, Dtype* input) {
  Dtype* col_buff = (is_1x1_) ? input : col_buffer_.mutable_gpu_data();
  int input_channel_offset = reverse_dimensions() ? top_dim_ / conv_in_channels_ : bottom_dim_ / conv_in_channels_;
  int weights_per_col = kernel_dim_ / conv_in_channels_;
  int input_channels = conv_in_channels_;
  int cuda_weight_col_mem_count = conv_out_channels_ * weights_per_col ;

  //transposed channel weights on gpu
  Dtype* channel_weights;
  CUDA_CHECK( cudaMalloc(&channel_weights, cuda_weight_col_mem_count * sizeof(Dtype)) );

  for (int channel_num = 0; channel_num < input_channels; ++channel_num) {
    caffe_gpu_get_col(cuda_weight_col_mem_count, weights, channel_weights, kernel_dim_, conv_out_channels_, weights_per_col, channel_num);
    for (int g = 0; g < group_; ++g) {
      caffe_gpu_gemm<Dtype>(CblasTrans, 
                            CblasNoTrans, 
                            kernel_dim_ / conv_in_channels_,  
                            conv_out_spatial_dim_,            
                            conv_out_channels_ / group_,               
                            (Dtype)1.,                       
                            channel_weights + weight_offset_ * g,  
                            output + output_offset_ * g,                          
                            (Dtype)0.,                       
                            col_buff + col_offset_ * g + (channel_num * input_channel_offset * is_1x1_));                       
    }
    if (!is_1x1_) {
      conv_in_channels_ = 1;
      conv_col2im_gpu(col_buff, input + channel_num * input_channel_offset);
      conv_in_channels_ = input_channels;
    }
  }
  CUDA_CHECK( cudaFree(channel_weights) );  //release transposed column
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::full_backward_gpu_gemm(const Dtype* output, const Dtype* weights, Dtype* input) {
  Dtype* col_buff = col_buffer_.mutable_gpu_data();
  if (is_1x1_) {
    col_buff = input;
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
                          conv_out_spatial_dim_, conv_out_channels_ / group_,
                          (Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
                          (Dtype)0., col_buff + col_offset_ * g);
  }
  if (!is_1x1_) {
    conv_col2im_gpu(col_buff, input);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::partial_weight_gpu_gemm(const Dtype* input, const Dtype* output, Dtype* weights) {
  const Dtype* col_buff = input;
  int weights_per_col = kernel_dim_ / conv_in_channels_;
  int input_channel_offset = reverse_dimensions() ? top_dim_ / conv_in_channels_ : bottom_dim_ / conv_in_channels_;
  int input_channels = conv_in_channels_;
  int cuda_weight_col_mem_count = conv_out_channels_ * weights_per_col ;
  
  Dtype* channel_weights;
  CUDA_CHECK( cudaMalloc(&channel_weights, cuda_weight_col_mem_count * sizeof(Dtype)) );

  for (int channel_num = 0; channel_num < input_channels; ++channel_num) {
    if (!is_1x1_) {
      conv_in_channels_ = 1;    //this variable is used when im2col_cpu is called by conv_im2col_cpu
      conv_im2col_gpu(input + channel_num * input_channel_offset, col_buffer_.mutable_gpu_data());
      conv_in_channels_ = input_channels;
      col_buff = col_buffer_.gpu_data();
    } 
    else {
      col_buff = input + channel_num * input_channel_offset;
    } 
    for (int g = 0; g < group_; ++g) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans,                          // CBLAS_TRANSPOSE TransA
                            CblasTrans,                            // CBLAS_TRANSPOSE TransB
                            conv_out_channels_ / group_,           // M  (# A and C rows)
                            kernel_dim_ / conv_in_channels_,       // N  (# B and C columns)
                            conv_out_spatial_dim_,                 // K  (# A columns, # B rows)       
                            (Dtype)1.,                             // alpha
                            output + output_offset_ * g,           // A
                            col_buff + col_offset_ * g,            // B 
                            (Dtype)0.,                             // beta
                            channel_weights + weight_offset_ * g); // C
    }
    //transpose add column back to existing weights for update 
    caffe_gpu_add_col(cuda_weight_col_mem_count, weights, channel_weights, kernel_dim_, conv_out_channels_, weights_per_col, channel_num);
  }
  CUDA_CHECK( cudaFree(channel_weights) );  //release transposed column
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::full_weight_gpu_gemm(const Dtype* input, const Dtype* output, Dtype* weights) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
    col_buff = col_buffer_.gpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
        kernel_dim_, conv_out_spatial_dim_,
        (Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)1., weights + weight_offset_ * g);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_gpu_bias(Dtype* bias,
    const Dtype* input) {
  caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1.,
      input, bias_multiplier_.gpu_data(), 1., bias);
}

#endif  // !CPU_ONLY

INSTANTIATE_CLASS(BaseConvolutionLayer);

}  // namespace caffe
