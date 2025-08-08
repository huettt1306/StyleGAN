#include "layer.h"
#include <omp.h>
#include <chrono>
#include <map>
#include <cmath>
#include <cstring>

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
  for (size_t i = 0; i < C; i++) {
    mean_squares += inout->buf[i] * inout->buf[i];
  }
  mean_squares /= C;
  float norm_factor = rsqrtf(mean_squares + 1e-8f);

  for (size_t i = 0; i < C; i++) {
    inout->buf[i] *= norm_factor;
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

void mat_mul(float *a, float *b, float *c, size_t M, size_t N, size_t K) {
  Timer timer;
  #pragma omp parallel for collapse(2)
  for (size_t m = 0; m < M; m++) {
    for (size_t n = 0; n < N; n++) {
      float sum = 0.0f;
      for (size_t k = 0; k < K; k++) {
        sum += a[m * K + k] * b[k * N + n];
      }
      c[m * N + n] = sum;
    }
  }
  time_map["mat_mul"] += timer.elapsed();
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
static inline size_t round_up(size_t x, size_t a) { return (x + a - 1) / a * a; }

// Convert NCHW input into (N*OH*OW) x (C*R*S_padded) matrix
static void im2col_nchw(const Tensor *input,
                        float *A, size_t Kc,
                        int stride, int pad, int dilation,
                        int R, int S,
                        int N, int C, int H, int W, int OH, int OW) {
  Timer timer;
  // A shape: M x Kc where M = N*OH*OW
  #pragma omp parallel for collapse(3)
  for (int n = 0; n < N; ++n) {
    for (int oh = 0; oh < OH; ++oh) {
      for (int ow = 0; ow < OW; ++ow) {
        size_t m = ((size_t)n * OH + oh) * OW + ow;
        float *prow = A + m * Kc;
        memset(prow, 0, Kc * sizeof(float));

        for (int c = 0; c < C; ++c) {
          for (int r = 0; r < R; ++r) {
            int h = oh * stride - pad + r * dilation;
            if (h < 0 || h >= H) continue;

            for (int s = 0; s < S; ++s) {
              int w = ow * stride - pad + s * dilation;
              size_t col = (size_t)c * R * S + r * S + s;

              if (w < 0 || w >= W) continue;

              prow[col] = input->buf[((size_t)n * C + c) * H * W + (size_t)h * W + w];
            }
          }
        }
      }
    }
  }
  time_map["im2col_nchw"] += timer.elapsed();
}

// Convert NCHW input into (N*OH*OW) x (C*R*S_padded) matrix for transposed conv
static void im2col_tconv_nchw(const Tensor *input,
                              float *A, size_t Kc,
                              int stride, int pad,
                              int R, int S,
                              int N, int C, int H, int W, int OH, int OW) {
  Timer timer;
  // A shape: M x Kc where M = N*OH*OW
  #pragma omp parallel for collapse(3)
  for (int n = 0; n < N; ++n) {
    for (int oh = 0; oh < OH; ++oh) {
      for (int ow = 0; ow < OW; ++ow) {
        size_t m = ((size_t)n * OH + oh) * OW + ow;
        float *prow = A + m * Kc;
        memset(prow, 0, Kc * sizeof(float));

        for (int c = 0; c < C; ++c) {
          for (int r = 0; r < R; ++r) {
            int t_h = oh + pad - r;
            if (t_h % stride != 0) continue;
            int h = t_h / stride;
            if (h < 0 || h >= H) continue;

            for (int s = 0; s < S; ++s) {
              int t_w = ow + pad - s;
              if (t_w % stride != 0) continue;
              int w = t_w / stride;
              if (w < 0 || w >= W) continue;

              size_t col = (size_t)c * R * S + r * S + s;
              prow[col] = input->buf[((size_t)n * C + c) * H * W + (size_t)h * W + w];
            }
          }
        }
      }
    }
  }
  time_map["im2col_tconv_nchw"] += timer.elapsed();
}

void Conv2d(Tensor *input, Tensor *weight, Tensor *bias, Tensor *output,
            int stride, int pad, int dilation, bool has_bias) {
  Timer timer;
  int N = input->shape[0], C = input->shape[1], H = input->shape[2], W = input->shape[3];
  int K = weight->shape[0], R = weight->shape[2], S = weight->shape[3];
  int OH = output->shape[2], OW = output->shape[3];

  // Dimensions for GEMM: (M x Kc) * (Kc x Nout) = (M x Nout)
  size_t M = (size_t)N * OH * OW;           
  size_t K0 = (size_t)C * R * S;           
  size_t Kc = round_up(K0, 8);             
  size_t Nout = (size_t)K;                 

  // Allocate buffers
  float *A = new float[M * Kc];            
  float *B = new float[Kc * Nout];         
  float *Cmat = new float[M * Nout];       

  // Build im2col matrix A
  im2col_nchw(input, A, Kc, stride, pad, dilation, R, S, N, C, H, W, OH, OW);

  // Build B by transposing weights to (C*R*S) x K and padding rows
  memset(B, 0, Kc * Nout * sizeof(float));
  #pragma omp parallel for collapse(2)
  for (int k = 0; k < K; ++k) {
    for (int c = 0; c < C; ++c) {
      for (int r = 0; r < R; ++r) {
        for (int s_ = 0; s_ < S; ++s_) {
          size_t row = (size_t)c * R * S + r * S + s_;
          B[row * Nout + k] = weight->buf[(size_t)k * C * R * S + (size_t)c * R * S + r * S + s_];
        }
      }
    }
  }

  // GEMM: (M x Kc) * (Kc x K) -> (M x K)
  mat_mul(A, B, Cmat, M, Nout, Kc);

  // Write back to output tensor [N, K, OH, OW] and add bias if needed
  #pragma omp parallel for collapse(3)
  for (int n = 0; n < N; ++n) {
    for (int oh = 0; oh < OH; ++oh) {
      for (int ow = 0; ow < OW; ++ow) {
        size_t m = ((size_t)n * OH + oh) * OW + ow;
        for (int k = 0; k < K; ++k) {
          float v = Cmat[m * Nout + k];
          if (has_bias) v += bias->buf[k];
          output->buf[((size_t)n * K + k) * OH * OW + (size_t)oh * OW + ow] = v;
        }
      }
    }
  }

  delete[] A;
  delete[] B;
  delete[] Cmat;

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
  int N = input->shape[0];
  int C = input->shape[1], H = input->shape[2], W = input->shape[3];
  int K = weight->shape[1], R = weight->shape[2], S = weight->shape[3];
  int OH = output->shape[2], OW = output->shape[3];

  // GEMM dims: (M x Kc) * (Kc x K) = (M x K)
  size_t M   = (size_t)N * OH * OW;      
  size_t K0  = (size_t)C * R * S;        
  size_t Kc  = round_up(K0, 8);          
  size_t Nout = (size_t)K;               

  float *A    = new float[M * Kc];       
  float *B    = new float[Kc * Nout];    
  float *Cmat = new float[M * Nout];     

  // Build A from input using transposed-conv im2col
  im2col_tconv_nchw(input, A, Kc, stride, pad, R, S, N, C, H, W, OH, OW);

  // Build B by flattening weights (C, K, R, S) -> (C*R*S, K), with padding
  memset(B, 0, Kc * Nout * sizeof(float));
  #pragma omp parallel for collapse(2)
  for (int k = 0; k < K; ++k) {
    for (int c = 0; c < C; ++c) {
      for (int r = 0; r < R; ++r) {
        for (int s_ = 0; s_ < S; ++s_) {
          size_t row = (size_t)c * R * S + r * S + s_;
          B[row * Nout + k] =
              weight->buf[((size_t)c * K + k) * R * S + (size_t)r * S + s_];
        }
      }
    }
  }

  // GEMM
  mat_mul(A, B, Cmat, M, Nout, Kc);

  // Write back to output tensor [N, K, OH, OW]
  #pragma omp parallel for collapse(3)
  for (int n = 0; n < N; ++n) {
    for (int oh = 0; oh < OH; ++oh) {
      for (int ow = 0; ow < OW; ++ow) {
        size_t m = ((size_t)n * OH + oh) * OW + ow;
        for (int k = 0; k < K; ++k) {
          output->buf[((size_t)n * K + k) * OH * OW + (size_t)oh * OW + ow] =
              Cmat[m * Nout + k];
        }
      }
    }
  }

  delete[] A;
  delete[] B;
  delete[] Cmat;

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
        for (size_t w = 0; w < W; w++) {
          size_t input_idx  = ((n * C + c) * H + h) * W + w;         // NCHW
          size_t output_idx = ((c * N + n) * H + h) * W + w;         // CNHW
          output->buf[output_idx] = input->buf[input_idx];
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

  for (size_t m = 0; m < M; m++) {
    for (size_t n = 0; n < N; n++) {
      out->buf[m * N + n] = 0;
      for (size_t k = 0; k < K; k++) {
        out->buf[m * N + n] += in->buf[m * K + k] * w->buf[n * K + k] * scale;
      }
      out->buf[m * N + n] += b->buf[n] * lr_mul;
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

  for (size_t i = 0; i < N; i++) {
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

  for (size_t oc = 0; oc < out_C; oc++) {
    for (size_t ic = 0; ic < in_C; ic++) {
      for (size_t k = 0; k < kernel_size * kernel_size; k++) {
        size_t idx = oc * in_C * kernel_size * kernel_size + ic * kernel_size * kernel_size + k;
        weight_a->buf[idx] = conv_weight->buf[idx] * style_a->buf[ic] * scale;
      }
    }
  }

  if (demodulate) {
    for (size_t oc = 0; oc < out_C; oc++) {
      float sum = 0.0f;
      for (size_t ic = 0; ic < in_C; ic++) {
        for (size_t k = 0; k < kernel_size * kernel_size; k++) {
          size_t idx = oc * in_C * kernel_size * kernel_size + ic * kernel_size * kernel_size + k;
          sum += weight_a->buf[idx] * weight_a->buf[idx];
        }
      }
      demod_a->buf[oc] = 1.0f / sqrtf(sum + 1e-8f);
    }

    for (size_t oc = 0; oc < out_C; oc++) {
      for (size_t ic = 0; ic < in_C; ic++) {
        for (size_t k = 0; k < kernel_size * kernel_size; k++) {
          size_t idx = oc * in_C * kernel_size * kernel_size + ic * kernel_size * kernel_size + k;
          weight_a->buf[idx] *= demod_a->buf[oc];
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

  for (size_t c = 0; c < C; c++) {
    for (size_t h = 0; h < H; h++) {
      for (size_t w = 0; w < W; w++) {
        size_t idx = (c * H + h) * W + w;
        inout->buf[idx] += noise->buf[h * W + w];
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
    for (size_t h = 0; h < H; h++) {
      for (size_t w = 0; w < W; w++) {
        size_t idx = (c * H + h) * W + w;
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

  for (size_t i = 0; i < N; i++) {
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