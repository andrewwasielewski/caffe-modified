#ifndef CAFFE_BASE_CONVOLUTION_LAYER_HPP_
#define CAFFE_BASE_CONVOLUTION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/im2col.hpp"

namespace caffe {

/**
 * @brief Abstract base class that factors out the BLAS code common to
 *        ConvolutionLayer and DeconvolutionLayer.
 */
template <typename Dtype>
class BaseConvolutionLayer : public Layer<Dtype> {
 public:
  explicit BaseConvolutionLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline bool EqualNumBottomTopBlobs() const { return true; }

  virtual void ReleaseTemporaryBuffers() {
    col_buffer_.ReleaseMemory();
  }

  //TODO - it might not be efficient to release all the smaller buffers
  virtual void ReleaseAllBuffers() {
    col_buffer_.ReleaseMemory();
    bias_multiplier_.ReleaseMemory();
    kernel_shape_.ReleaseMemory();
    stride_.ReleaseMemory();
    pad_.ReleaseMemory();
    dilation_.ReleaseMemory();
    conv_input_shape_.ReleaseMemory();
  }

 protected:
  // Helper functions that abstract away the column buffer and gemm arguments.
  // The last argument in forward_cpu_gemm is so that we can skip the im2col if
  // we just called weight_cpu_gemm with the same input.
  void partial_forward_cpu_gemm(const Dtype* input, const Dtype* weights, Dtype* output);
  void full_forward_cpu_gemm(const Dtype* input, const Dtype* weights, Dtype* output, bool skip_im2col);
  inline void forward_cpu_gemm(const Dtype* input, const Dtype* weights, Dtype* output, bool skip_im2col = false) {
    if(partial_conv_lower_) partial_forward_cpu_gemm(input, weights, output);
    else full_forward_cpu_gemm(input, weights, output, skip_im2col);
  }

  void partial_backward_cpu_gemm(const Dtype* input, const Dtype* weights, Dtype* output);
  void full_backward_cpu_gemm(const Dtype* input, const Dtype* weights, Dtype* output);
  inline void backward_cpu_gemm(const Dtype* input, const Dtype* weights, Dtype* output) {
    if(partial_conv_lower_) partial_backward_cpu_gemm(input, weights, output);
    else full_backward_cpu_gemm(input, weights, output);
  }

  void partial_weight_cpu_gemm(const Dtype* input, const Dtype* output, Dtype* weights);
  void full_weight_cpu_gemm(const Dtype* input, const Dtype* output, Dtype* weights);
  inline void weight_cpu_gemm(const Dtype* input, const Dtype* output, Dtype* weights) {
    if(partial_conv_lower_) partial_weight_cpu_gemm(input, output, weights);
    else full_weight_cpu_gemm(input, output, weights);
  }

  void forward_cpu_bias(Dtype* output, const Dtype* bias);
  void backward_cpu_bias(Dtype* bias, const Dtype* input);


#ifndef CPU_ONLY
  void partial_forward_gpu_gemm(const Dtype* input, const Dtype* weights, Dtype* output);
  void full_forward_gpu_gemm(const Dtype* input, const Dtype* weights, Dtype* output, bool skip_im2col);
  inline void forward_gpu_gemm(const Dtype* input, const Dtype* weights, Dtype* output, bool skip_im2col = false) {
    if(partial_conv_lower_) partial_forward_gpu_gemm(input, weights, output);
    else full_forward_gpu_gemm(input, weights, output, skip_im2col);
  }
  
  void partial_backward_gpu_gemm(const Dtype* output, const Dtype* weights, Dtype* input);
  void full_backward_gpu_gemm(const Dtype* output, const Dtype* weights, Dtype* input);
  inline void backward_gpu_gemm(const Dtype* output, const Dtype* weights, Dtype* input) {
    if(partial_conv_lower_) partial_backward_gpu_gemm(output, weights, input);
    else full_backward_gpu_gemm(output, weights, input);
  }
  
  void partial_weight_gpu_gemm(const Dtype* input, const Dtype* output, Dtype* weights);
  void full_weight_gpu_gemm(const Dtype* input, const Dtype* output, Dtype* weights);
  inline void weight_gpu_gemm(const Dtype* input, const Dtype* output, Dtype* weights) {
    if(partial_conv_lower_) partial_weight_gpu_gemm(input, output, weights);
    else full_weight_gpu_gemm(input, output, weights);
  }

  void forward_gpu_bias(Dtype* output, const Dtype* bias);
  void backward_gpu_bias(Dtype* bias, const Dtype* input);
#endif

  /// @brief The spatial dimensions of the input.
  inline int input_shape(int i) {
    return (*bottom_shape_)[channel_axis_ + i];
  }
  // reverse_dimensions should return true iff we are implementing deconv, so
  // that conv helpers know which dimensions are which.
  virtual bool reverse_dimensions() = 0;
  // Compute height_out_ and width_out_ from other parameters.
  virtual void compute_output_shape() = 0;

  long get_buffer_size() { return col_buffer_.data()->size(); }

  /// @brief The spatial dimensions of a filter kernel.
  Blob<int> kernel_shape_;
  /// @brief The spatial dimensions of the stride.
  Blob<int> stride_;
  /// @brief The spatial dimensions of the padding.
  Blob<int> pad_;
  /// @brief The spatial dimensions of the dilation.
  Blob<int> dilation_;
  /// @brief The spatial dimensions of the convolution input.
  Blob<int> conv_input_shape_;
  /// @brief The spatial dimensions of the col_buffer.
  vector<int> col_buffer_shape_;
  /// @brief The spatial dimensions of the output.
  vector<int> output_shape_;
  const vector<int>* bottom_shape_;

  int num_spatial_axes_;
  int bottom_dim_;
  int top_dim_;

  int channel_axis_;
  int num_;
  int channels_;
  int group_;
  int out_spatial_dim_;
  int weight_offset_;
  int num_output_;
  bool partial_conv_lower_;
  bool bias_term_;
  bool is_1x1_;
  bool force_nd_im2col_;

 private:
  // wrap im2col/col2im so we don't have to remember the (long) argument lists
  inline void conv_im2col_cpu(const Dtype* data, Dtype* col_buff) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      im2col_cpu(data, conv_in_channels_,
          conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
          kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
          pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1],
          dilation_.cpu_data()[0], dilation_.cpu_data()[1], col_buff);
    } else {
      im2col_nd_cpu(data, num_spatial_axes_, conv_input_shape_.cpu_data(),
          col_buffer_shape_.data(), kernel_shape_.cpu_data(),
          pad_.cpu_data(), stride_.cpu_data(), dilation_.cpu_data(), col_buff);
    }
  }
  inline void conv_col2im_cpu(const Dtype* col_buff, Dtype* data) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      col2im_cpu(col_buff, conv_in_channels_,
          conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
          kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
          pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1],
          dilation_.cpu_data()[0], dilation_.cpu_data()[1], data);
    } else {
      col2im_nd_cpu(col_buff, num_spatial_axes_, conv_input_shape_.cpu_data(),
          col_buffer_shape_.data(), kernel_shape_.cpu_data(),
          pad_.cpu_data(), stride_.cpu_data(), dilation_.cpu_data(), data);
    }
  }
#ifndef CPU_ONLY
  inline void conv_im2col_gpu(const Dtype* data, Dtype* col_buff) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      im2col_gpu(data, conv_in_channels_,
          conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
          kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
          pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1],
          dilation_.cpu_data()[0], dilation_.cpu_data()[1], col_buff);
    } else {
      im2col_nd_gpu(data, num_spatial_axes_, num_kernels_im2col_,
          conv_input_shape_.gpu_data(), col_buffer_.gpu_shape(),
          kernel_shape_.gpu_data(), pad_.gpu_data(),
          stride_.gpu_data(), dilation_.gpu_data(), col_buff);
    }
  }
  inline void conv_col2im_gpu(const Dtype* col_buff, Dtype* data) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      col2im_gpu(col_buff, conv_in_channels_,
          conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
          kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
          pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1],
          dilation_.cpu_data()[0], dilation_.cpu_data()[1], data);
    } else {
      col2im_nd_gpu(col_buff, num_spatial_axes_, num_kernels_col2im_,
          conv_input_shape_.gpu_data(), col_buffer_.gpu_shape(),
          kernel_shape_.gpu_data(), pad_.gpu_data(), stride_.gpu_data(),
          dilation_.gpu_data(), data);
    }
  }
#endif

  void get_col_from_row_major_matrix(const Dtype* matrix, Dtype* col_result, int len_row, int num_rows, int len_col, int col_number);
  void add_col_to_row_major_matrix(Dtype* matrix, Dtype* add_col, int len_row, int num_rows, int len_col, int col_number);

  int num_kernels_im2col_;
  int num_kernels_col2im_;
  int conv_out_channels_;
  int conv_in_channels_;
  int conv_out_spatial_dim_;
  int kernel_dim_;
  int col_offset_;
  int output_offset_;

  Blob<Dtype> col_buffer_;
  Blob<Dtype> bias_multiplier_;
};

}  // namespace caffe

#endif  // CAFFE_BASE_CONVOLUTION_LAYER_HPP_
