#include "layer.h"
#include <omp.h>
#include <chrono>
#include <map>
#include <cmath>
#include <cstring>
#include <immintrin.h>

class Timer {
private:
  std::chrono::time_point<std::chrono::high_resolution_clock> start;
public:
  Timer() { start = std::chrono::high_resolution_clock::now(); }
  double elapsed() {
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    return duration.count();
  }
};

static std::map<std::string, double> time_map;


/*
  * PixelNorm
  * @param [in & out] inout: [N, C]
  * Normalizes the input tensor along dim=1.
  * Equivalent to: input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)
  */

void PixelNorm(Tensor *inout) {
  Timer timer;
  size_t C = inout->shape[1];
  
  float mean_squares = 0.f;
  float* buf = inout->buf;
  // for (size_t i = 0; i < C; i++) {
  //   mean_squares += inout->buf[i] * inout->buf[i];
  // }

  __m256 a, x, s = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
  size_t i;
  for (i = 0; i < C - 7; i += 8) {
    a = _mm256_load_ps(buf + i);
    s = _mm256_fmadd_ps(a, a, s);
  }
  for (; i < C; ++i) {
    mean_squares += buf[i] * buf[i];
  }
  for (int i = 0; i < 8; ++i) {
    mean_squares += s[i];
  }

  mean_squares /= C;
  float norm_factor = rsqrtf(mean_squares + 1e-8f);

  // for (size_t i = 0; i < C; i++) {
  //   inout->buf[i] *= norm_factor;
  // }
  __m256 factor = _mm256_set1_ps(norm_factor);

  for (i = 0; i < C - 7; i += 8) {
    a = _mm256_load_ps(buf + i);
    x = _mm256_mul_ps(a, factor);
    _mm256_store_ps(buf + i, x);
  }
  for (; i < C; ++i) {
    buf[i] *= norm_factor;
  }

  time_map["PixelNorm"] += timer.elapsed();
}

/*
 * Upsample and Pad
 * input shape = (N, C, H, W)
 * output shape = (N, C, OH, OW)
 *   where OH = H * up + pad0 + pad1,
 *         OW = W * up + pad0 + pad1
 */
void UpsamplePad(Tensor *input, Tensor *output, int up, int pad0, int pad1) {
  Timer timer;
  size_t N = input->shape[0];
  size_t C = input->shape[1]; 
  size_t H = input->shape[2];
  size_t W = input->shape[3];

  size_t OH = up * H + pad0 + pad1;
  size_t OW = up * W + pad0 + pad1;

  memset(output->buf, 0, N * C * OH * OW * sizeof(float));
  for (size_t c = 0; c < C; ++c) {
      for (size_t h = 0; h < H; ++h) {
          for (size_t w = 0; w < W; ++w) {
              output->buf[c * OH * OW + (h * up + pad0) * OW + w * up + pad0] +=
                  input->buf[c * H * W + h * W + w];
          }
      }
  }
  time_map["UpsamplePad"] += timer.elapsed();
}

/*
 * Convolution
 * input shape = (N, C, H, W)
 * weight shape = (K, C, R, S)
 * bias shape = (K)
 * output shape = (N, K, OH, OW)
 *   where OH = (H + 2 * pad - dilation * (R - 1) - 1) / stride + 1,
 *         OW = (W + 2 * pad - dilation * (S - 1) - 1) / stride + 1
 */
void Conv2d(Tensor *input, Tensor *weight, Tensor *bias, Tensor *output,
            int stride, int pad, int dilation, bool has_bias) {
  Timer timer;
  int N = input->shape[0], C = input->shape[1], H = input->shape[2], W = input->shape[3];
  int K = weight->shape[0], R = weight->shape[2], S = weight->shape[3];
  int OH = output->shape[2], OW = output->shape[3];

  // Clear output buffer
  memset(output->buf, 0, N * K * OH * OW * sizeof(float));

  const int TILE_H = 32;  // Tile size for OH dimension
  const int TILE_W = 64;  // Tile size for OW dimension

  #pragma omp parallel for collapse(4)
  for (int n = 0; n < N; ++n) {
    for (int k = 0; k < K; ++k) {
      for (int oh_tile = 0; oh_tile < OH; oh_tile += TILE_H) {
        for (int ow_tile = 0; ow_tile < OW; ow_tile += TILE_W) {
          int c;
          for (c = 0; c + 3 < C; c += 4) {
            int oh_end = min(oh_tile + TILE_H, OH);
            int ow_end = min(ow_tile + TILE_W, OW);
            
            for (int oh = oh_tile; oh < oh_end; ++oh) {
              for (int ow = ow_tile; ow < ow_end; ++ow) {
                float o = has_bias ? bias->buf[k] : 0;
                int minr = max(0, (pad - oh * stride + dilation - 1) / dilation);
                int maxr = min(R, (H + pad - oh * stride + dilation - 1) / dilation);
                int mins = max(0, (pad - ow * stride + dilation - 1) / dilation);
                int maxs = min(S, (W + pad - ow * stride + dilation - 1) / dilation);

                for (int r = minr, h = oh * stride - pad + r * dilation; r < maxr; ++r, h += dilation) {
                  for (int s = mins, w = ow * stride - pad + s * dilation; s < maxs; ++s, w += dilation) {
                    o += input->buf[n * C * H * W + c * H * W + h * W + w] * weight->buf[k * C * R * S + c * R * S + r * S + s];
                    o += input->buf[n * C * H * W + (c + 1) * H * W + h * W + w] * weight->buf[k * C * R * S + (c + 1) * R * S + r * S + s];
                    o += input->buf[n * C * H * W + (c + 2) * H * W + h * W + w] * weight->buf[k * C * R * S + (c + 2) * R * S + r * S + s];
                    o += input->buf[n * C * H * W + (c + 3) * H * W + h * W + w] * weight->buf[k * C * R * S + (c + 3) * R * S + r * S + s];
                  }
                }
                output->buf[n * K * OH * OW + k * OH * OW + oh * OW + ow] += o;
              }
            }
          }
          for (; c < C; ++c) {
            int oh_end = min(oh_tile + TILE_H, OH);
            int ow_end = min(ow_tile + TILE_W, OW);
            
            for (int oh = oh_tile; oh < oh_end; ++oh) {
              for (int ow = ow_tile; ow < ow_end; ++ow) {
                float o = has_bias ? bias->buf[k] : 0;
                int minr = max(0, (pad - oh * stride + dilation - 1) / dilation);
                int maxr = min(R, (H + pad - oh * stride + dilation - 1) / dilation);
                int mins = max(0, (pad - ow * stride + dilation - 1) / dilation);
                int maxs = min(S, (W + pad - ow * stride + dilation - 1) / dilation);
                
                for (int r = minr, h = oh * stride - pad + r * dilation; r < maxr; ++r, h += dilation) {
                  for (int s = mins, w = ow * stride - pad + s * dilation; s < maxs; ++s, w += dilation) {
                    float i = input->buf[n * C * H * W + c * H * W + h * W + w];
                    float f = weight->buf[k * C * R * S + c * R * S + r * S + s];
                    o += i * f;
                  }
                }
                output->buf[n * K * OH * OW + k * OH * OW + oh * OW + ow] += o;
              }
            }
          }
        }
      }
    }
  }
  time_map["Conv2d"] += timer.elapsed();
}

/*
 * Transposed convolution
 * input shape = (N, C, H, W)
 * weight shape = (C, K, R, S)
 * output shape = (N, K, OH, OW)
 *   where OH = (H - 1) * stride - 2 * pad + R
 *         OW = (W - 1) * stride - 2 * pad + S
 */
void ConvTranspose2d(Tensor *input, Tensor *weight, Tensor *output, 
                     int stride, int pad) {
  Timer timer;
  int C = input->shape[1], H = input->shape[2], W = input->shape[3];
  int K = weight->shape[1], R = weight->shape[2], S = weight->shape[3];
  int OH = output->shape[2], OW = output->shape[3];

  memset(output->buf, 0, K * OH * OW * sizeof(float));

  const int TILE_H = 32;  
  const int TILE_W = 64;  
  
  #pragma omp parallel for collapse(3)  
  for (int k = 0; k < K; ++k) {
    for (int oh_tile = 0; oh_tile < OH; oh_tile += TILE_H) {
      for (int ow_tile = 0; ow_tile < OW; ow_tile += TILE_W) {
        int c = 0;
        for (c = 0; c + 3 < C; c += 4) {
          int oh_end = min(oh_tile + TILE_H, OH);
          int ow_end = min(ow_tile + TILE_W, OW);
          
          for (int oh = oh_tile; oh < oh_end; ++oh) {
            for (int ow = ow_tile; ow < ow_end; ++ow) {
              float o = 0.0f;
              int minr = max((oh + pad) % stride, oh + pad - H * stride + stride);
              int maxr = min(oh + pad + 1, R);
              int mins = max((ow + pad) % stride, ow + pad - W * stride + stride);
              int maxs = min(ow + pad + 1, S);

              for (int r = minr, h = (oh + pad - r) / stride; r < maxr; r += stride, h -= 1) {
                for (int s = mins, w = (ow + pad - s) / stride; s < maxs; s += stride, w -= 1) {
                  o += input->buf[c * H * W + h * W + w] * weight->buf[c * K * R * S + k * R * S + r * S + s];
                  o += input->buf[(c+1) * H * W + h * W + w] * weight->buf[(c+1) * K * R * S + k * R * S + r * S + s];
                  o += input->buf[(c+2) * H * W + h * W + w] * weight->buf[(c+2) * K * R * S + k * R * S + r * S + s];
                  o += input->buf[(c+3) * H * W + h * W + w] * weight->buf[(c+3) * K * R * S + k * R * S + r * S + s];
                }
              }
              output->buf[k * OH * OW + oh * OW + ow] += o;
            }
          }
        }
        for (; c < C; c++) {
          int oh_end = min(oh_tile + TILE_H, OH);
          int ow_end = min(ow_tile + TILE_W, OW);
          
          for (int oh = oh_tile; oh < oh_end; ++oh) {
            for (int ow = ow_tile; ow < ow_end; ++ow) {
              float o = 0.0f;
              int minr = max((oh + pad) % stride, oh + pad - H * stride + stride);
              int maxr = min(oh + pad + 1, R);
              int mins = max((ow + pad) % stride, ow + pad - W * stride + stride);
              int maxs = min(ow + pad + 1, S);

              for (int r = minr, h = (oh + pad - r) / stride; r < maxr; r += stride, h -= 1) {
                for (int s = mins, w = (ow + pad - s) / stride; s < maxs; s += stride, w -= 1) {
                  o += input->buf[c * H * W + h * W + w] * weight->buf[c * K * R * S + k * R * S + r * S + s];
                }
              }
              output->buf[k * OH * OW + oh * OW + ow] += o;
            }
          }
        }
      }
    }
  }
  time_map["ConvTranspose2d"] += timer.elapsed();
}

/* Transpose
 * input shape = (N, C, H, W)
 * output shape = (C, N, H, W)
 * Transposes the first two dimensions of the input tensor.
 */
void transpose(Tensor *input, Tensor *output) {
  Timer timer;
  size_t N = input->shape[0];
  size_t C = input->shape[1];
  size_t H = input->shape[2];
  size_t W = input->shape[3];

  for (size_t n = 0; n < N; n++) {
    for (size_t c = 0; c < C; c++) {
      for (size_t h = 0; h < H; h++) {
        // for (size_t w = 0; w < W; w++) {
        //   size_t input_idx  = ((n * C + c) * H + h) * W + w;         // NCHW
        //   size_t output_idx = ((c * N + n) * H + h) * W + w;         // CNHW
        //   output->buf[output_idx] = input->buf[input_idx];
        // }
        size_t base_in  = ((n * C + c) * H + h) * W;
        size_t base_out = ((c * N + n) * H + h) * W;
        size_t w = 0;
        for (; w + 8 <= W; w += 8) {
          __m256 vec = _mm256_loadu_ps(input->buf + base_in + w);
          _mm256_storeu_ps(output->buf + base_out + w, vec);
        }
        for (; w < W; ++w) {
          output->buf[base_out + w] = input->buf[base_in + w];
        }
      }
    }
  }
  time_map["transpose"] += timer.elapsed();
}

/* Linear
 * @param [in1]  in: [M, K]
 * @param [in2]   w: [N, K]
 * @param [in3]   b: [N]
 * @param [out] out: [M, N]
 */
void Linear(Tensor *in, Tensor *w, Tensor *b, Tensor *out, float lr_mul) {
  Timer timer;
  size_t M = out->shape[0];
  size_t N = out->shape[1];
  size_t K = w->shape[1];

  float scale = (1.0f / sqrtf(K)) * lr_mul;
  __m256 x, y, s;

  for (size_t m = 0; m < M; m++) {
    for (size_t n = 0; n < N; n ++) {
      // out->buf[m * N + n] = 0;
      // for (size_t k = 0; k < K; k++) {
      //   out->buf[m * N + n] += in->buf[m * K + k] * w->buf[n * K + k] * scale;
      // }
      size_t k;
      s = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
      for(k = 0; k + 7 < K; k += 8) {
        x = _mm256_load_ps((in->buf) + m * K + k);
        y = _mm256_load_ps((w->buf) + n * K + k);
        s = _mm256_fmadd_ps(x, y, s);
      }
      float sum = 0;
      for(int i = 0; i < 8; ++i) {
        sum += s[i];
      }
      for(; k < K; k++) {
        sum += in->buf[m * K + k] * w->buf[n * K + k];
      }
      out->buf[m * N + n] = sum * scale + b->buf[n] * lr_mul;
//      out->buf[m * N + n] += b->buf[n] * lr_mul;
    }
  }
  time_map["Linear"] += timer.elapsed();
}

/* LeakyReLU
 * @param [in & out] inout: [N]
 * 'N' is the number of elements in the tensor.
 */
void LeakyReLU(Tensor *inout) {
  Timer timer;
  size_t N = inout->num_elem();

  float negative_slope = 0.2f;
  float scale = sqrtf(2.0f);
  __m256 neg = _mm256_set1_ps(negative_slope * scale);
  __m256 pos = _mm256_set1_ps(scale);
  __m256 a, mask, rpos, rneg, result;

  size_t i;
  for (i = 0; i + 7 < N; i += 8) {
    // if (inout->buf[i] < 0) { inout->buf[i] *= negative_slope; }
    // inout->buf[i] *= scale;
    a = _mm256_loadu_ps(inout->buf + i);
    mask = _mm256_cmp_ps(a, _mm256_setzero_ps(), _CMP_LT_OS);;
    rneg = _mm256_mul_ps(a, neg);
    rpos = _mm256_mul_ps(a, pos);
    result = _mm256_blendv_ps(rpos, rneg, mask);
    _mm256_storeu_ps(inout->buf + i, result);
  }
  for (; i < N; ++i) {
    if (inout->buf[i] < 0) { inout->buf[i] *= negative_slope; }
    inout->buf[i] *= scale;
  }
  time_map["LeakyReLU"] += timer.elapsed();
}

void upfir2d(Tensor *input, Tensor *kernel, Tensor *output,
               Tensor *upsample_a, Tensor *conv_a,
               int up, int pad0, int pad1) {
  Timer timer;
  // Upsample and Pad -> Conv2d (FIR filter)
  UpsamplePad(input, upsample_a, up, pad0, pad1);

  int C = upsample_a->shape[1];
  int H = upsample_a->shape[2];
  int W = upsample_a->shape[3];
  upsample_a->reshape({C, 1, H, W});
  Conv2d(upsample_a, kernel, nullptr, output, 1, 0, 1, false);
  upsample_a->reshape({1, C, H, W});

  time_map["upfir2d"] += timer.elapsed();
}

void ModulatedConv2d(Tensor *input, Tensor *style, Tensor *modulate_weight, Tensor *modulate_bias, Tensor *conv_weight, Tensor *kernel, Tensor *output,
                     Tensor *style_a, Tensor *weight_a, Tensor *demod_a, Tensor *transpose_a, Tensor *conv_a, Tensor *upsample_a, Tensor *conv2_a,
                     bool demodulate, bool upsample, int padding, int up
) {
  Timer timer;
  size_t in_C = input->shape[1];
  size_t out_C = conv_weight->shape[0];
  size_t kernel_size = conv_weight->shape[2];

  Linear(style, modulate_weight, modulate_bias, style_a, 1.0f);

  float scale = 1 / sqrtf((float) (in_C * kernel_size * kernel_size));
  size_t K = kernel_size * kernel_size;
  __m256 a, s;

  for (size_t oc = 0; oc < out_C; oc++) {
    for (size_t ic = 0; ic < in_C; ic++) {
      __m256 style = _mm256_set1_ps(style_a->buf[ic] * scale);
      size_t K_idx = oc * in_C * K + ic * K, k;
      // for (size_t k = 0; k < kernel_size * kernel_size; k++) {
      //   size_t idx = oc * in_C * kernel_size * kernel_size + ic * kernel_size * kernel_size + k;
      //   weight_a->buf[idx] = conv_weight->buf[idx] * style_a->buf[ic] * scale;
      // }
      for (k = 0; k + 7 < K; k += 8) {
        a = _mm256_loadu_ps(conv_weight->buf + K_idx + k);
        s = _mm256_mul_ps(a, style);
        _mm256_storeu_ps(weight_a->buf + K_idx + k, s);
      }
      for (; k < K; k++) {
        size_t idx = K_idx + k;
        weight_a->buf[idx] = conv_weight->buf[idx] * style_a->buf[ic] * scale;
      }
    }
  }

  if (demodulate) {
    for (size_t oc = 0; oc < out_C; oc++) {
      float sum = 0.0f;
      for (size_t ic = 0; ic < in_C; ic++) {
        // for (size_t k = 0; k < kernel_size * kernel_size; k++) {
        //   size_t idx = oc * in_C * kernel_size * kernel_size + ic * kernel_size * kernel_size + k;
        //   sum += weight_a->buf[idx] * weight_a->buf[idx];
        // }
        size_t K_idx = oc * in_C * K + ic * K, k;
        s = _mm256_setzero_ps();
        for (k = 0; k + 7 < K; k += 8) {
          a = _mm256_loadu_ps(weight_a->buf + K_idx + k);
          s = _mm256_fmadd_ps(a, a, s);
        }
        for(int i = 0; i < 8; i++) {
          sum += s[i];
        }
        for (; k < K; k++) {
          sum += weight_a->buf[K_idx + k] * weight_a->buf[K_idx + k];
        }
      }
      demod_a->buf[oc] = 1.0f / sqrtf(sum + 1e-8f);
    }

    for (size_t oc = 0; oc < out_C; oc++) {
      __m256 demod = _mm256_set1_ps(demod_a->buf[oc]), a, s;
      for (size_t ic = 0; ic < in_C; ic++) {
        size_t K_idx = oc * in_C * K + ic * K, k;
        // for (size_t k = 0; k < kernel_size * kernel_size; k++) {
        //   size_t idx = oc * in_C * kernel_size * kernel_size + ic * kernel_size * kernel_size + k;
        //   weight_a->buf[idx] *= demod_a->buf[oc];
        // }
        for (k = 0; k + 7 < K; k += 8) {
          a = _mm256_loadu_ps(weight_a->buf + K_idx + k);
          s = _mm256_mul_ps(a, demod);
          _mm256_storeu_ps(weight_a->buf + K_idx + k, s);
        }
        for (; k < K; k++) {
          weight_a->buf[K_idx + k] *= demod_a->buf[oc];
        }
      }
    }
  }

  if (upsample) {
    transpose(weight_a, transpose_a);
    ConvTranspose2d(input, transpose_a, conv_a, 2, 0);
    upfir2d(conv_a, kernel, output, upsample_a, conv2_a, up, 1, 1);
  }
  else {
    Conv2d(input, weight_a, nullptr, output, 1, padding, 1, false);
  }
  time_map["ModulatedConv2d"] += timer.elapsed();
}

/* Add noise to the input tensor
 * @param [in & out] inout: [N, C, H, W]
 * @param [in] noise: [H, W]
 * Adds noise to the input tensor in-place.
 */

void addNoise(Tensor *inout, Tensor *noise) {
  Timer timer;
  size_t C = inout->shape[1];
  size_t H = inout->shape[2];
  size_t W = inout->shape[3];

  __m256 a, b, result;

  for (size_t c = 0; c < C; c++) {
    for (size_t h = 0; h < H; h++) {
      // for (size_t w = 0; w < W; w++) {
      //   size_t idx = (c * H + h) * W + w;
      //   inout->buf[idx] += noise->buf[h * W + w];
      // }
      size_t w;
      size_t base_idx = (c * H + h) * W;
      size_t noise_idx = h * W;

      for (w = 0; w + 7 < W; w += 8) {
        a = _mm256_load_ps(inout->buf + base_idx + w);
        b = _mm256_load_ps(noise->buf + noise_idx + w);
        result = _mm256_add_ps(a, b);
        _mm256_store_ps(inout->buf + base_idx + w , result);
      }
      for (; w < W; w++) {
        size_t idx = base_idx + w;
        inout->buf[idx] += noise->buf[noise_idx + w];
      }
    }
  }
  time_map["addNoise"] += timer.elapsed();
}

/* Add bias to the input tensor
 * @param [in & out] inout: [N, C, H, W]
 * @param [in] bias: [C]
 * Adds bias to the input tensor in-place.
 */

void addBias(Tensor *inout, Tensor *bias) {
  Timer timer;
  size_t C = inout->shape[1];
  size_t H = inout->shape[2];
  size_t W = inout->shape[3];

  for (size_t c = 0; c < C; c++) {
    __m256 bias_ = _mm256_set1_ps(bias->buf[c]), a, result;
    for (size_t h = 0; h < H; h++) {
      // for (size_t w = 0; w < W; w++) {
      //   size_t idx = (c * H + h) * W + w;
      //   inout->buf[idx] += bias->buf[c];
      // }
      size_t base_idx = (c * H + h) * W;
      size_t w;
      for (w = 0; w + 7 < W; w += 8) {
        a = _mm256_load_ps(inout->buf + base_idx + w);
        result = _mm256_add_ps(a, bias_);
        _mm256_store_ps(inout->buf + base_idx + w , result);
      }
      for (; w < W; w++) {
        size_t idx = base_idx + w;
        inout->buf[idx] += bias->buf[c];
      }
    }
  }
  time_map["addBias"] += timer.elapsed();
}

/*
 * Element-wise addition of two tensors
 * @param [in & out] inout: [N, C, H, W]
 * @param [in] addend: [N, C, H, W]
 * Adds the elements of addend to inout in-place.
 */
void elemAdd(Tensor *inout, Tensor *addend) {
  Timer timer;
  size_t N = inout->num_elem();

  size_t i;
  __m256 a, b, s;
  for (i = 0; i + 7 < N; i += 8) {
    //inout->buf[i] += addend->buf[i];
    a = _mm256_load_ps(inout->buf + i);
    b = _mm256_load_ps(addend->buf + i);
    s = _mm256_add_ps(a, b);
    _mm256_store_ps(inout->buf + i, s);
  }
  for (; i < N; ++i) {
    inout->buf[i] += addend->buf[i];
  }
  time_map["elemAdd"] += timer.elapsed();
}

void StyledConv(Tensor *input, Tensor *style, Tensor *modulate_weight, Tensor *modulate_bias, Tensor *conv_weight, Tensor *conv_bias, Tensor *kernel, Tensor *noise, Tensor *output,
                Tensor *style_a, Tensor *weight_a, Tensor *demod_a, Tensor *transpose_a, Tensor *conv_a, Tensor *upsample_a, Tensor *conv2_a,
                bool demodulate, bool upsample, int padding) {
  Timer timer;
  ModulatedConv2d(input, style, modulate_weight, modulate_bias, conv_weight, kernel, output,
                  style_a, weight_a, demod_a, transpose_a, conv_a, upsample_a, conv2_a,
                  demodulate, upsample, padding, 1);
  addNoise(output, noise);
  addBias(output, conv_bias);
  LeakyReLU(output);
  time_map["StyledConv"] += timer.elapsed();
}

void ToRGB(Tensor *input, Tensor *skip, Tensor *style, Tensor *modulate_weight, Tensor *modulate_bias, Tensor *conv_weight, Tensor *conv_bias, Tensor *kernel, Tensor *output,
           Tensor *style_a, Tensor *weight_a, Tensor *demod_a, Tensor *transpose_a, Tensor *conv_a, Tensor *upsample_a, Tensor *conv2_a, Tensor *skip_upsample_a, Tensor *skip_conv_a, Tensor *skip_a,
           bool demodulate, bool upsample, int padding) {
  Timer timer;
  ModulatedConv2d(input, style, modulate_weight, modulate_bias, conv_weight, kernel, output,
                  style_a, weight_a, demod_a, transpose_a, conv_a, upsample_a, conv2_a,
                  demodulate, upsample, padding, 2);
  addBias(output, conv_bias);

  if (skip != nullptr) {
    upfir2d(skip, kernel, skip_a, skip_upsample_a, skip_conv_a, 2, 2, 1);
    elemAdd(output, skip_a);
  }
  time_map["ToRGB"] += timer.elapsed();
}

void printTimeMap() {
  printf("\n-------------------------------------\n");
  for (const auto &entry : time_map) {
    printf("%s: %.6f seconds\n", entry.first.c_str(), entry.second);
  }
  time_map.clear();
}