int index = 0;
      int channel_weights_index;
      for (int row = 0; row < conv_out_channels_; ++row) {
        for (int col_pos = 0; col_pos < weights_per_col; ++col_pos) {
          channel_weights_index = row * kernel_dim_ + (col_pos + channel_num * weights_per_col);
          channel_weights[index] = weights[channel_weights_index];
          index++;
        }
      }