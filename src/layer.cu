#include "layer.h"
#include <omp.h>
#include <chrono>
#include <map>
#include <cmath>
#include <cstring>
#include <immintrin.h>
#include <sched.h>

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
#define ITILESIZE (32)
#define JTILESIZE (256)
#define KTILESIZE (128)

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

  #pragma omp parallel for simd
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

  #pragma omp parallel for collapse(2)
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
 * Matrix multiplication: C = A * B
 * A: (M, K), B: (K, N), C: (M, N)
 */
void mat_mul(float *A, float *B, float *C, int M, int N, int K) {
  Timer timer;
      
  #pragma omp parallel num_threads(32)
  {
  int tid = omp_get_thread_num();
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(tid, &cpuset);
  sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);

  #pragma omp for simd
  for (int i = 0; i < M * N; i++) {
    C[i] = 0.0f;
  }

  #pragma omp for collapse(2)
  for (int i = 0; i < M; i += ITILESIZE) {
    for (int j = 0; j < N; j += JTILESIZE) {
      for (int k = 0; k < K; k += KTILESIZE) {
        for (int kk = k; kk < K && kk < k + KTILESIZE; kk+=8) {
          for (int ii = i;ii < M && ii < i + ITILESIZE; ii++) {
            __m256 a0 = _mm256_set1_ps(A[(ii+0)*K+(kk+0)]);
            __m256 a1 = _mm256_set1_ps(A[(ii+0)*K+(kk+1)]);
            __m256 a2 = _mm256_set1_ps(A[(ii+0)*K+(kk+2)]);
            __m256 a3 = _mm256_set1_ps(A[(ii+0)*K+(kk+3)]);
            __m256 a4 = _mm256_set1_ps(A[(ii+0)*K+(kk+4)]);
            __m256 a5 = _mm256_set1_ps(A[(ii+0)*K+(kk+5)]);
            __m256 a6 = _mm256_set1_ps(A[(ii+0)*K+(kk+6)]);
            __m256 a7 = _mm256_set1_ps(A[(ii+0)*K+(kk+7)]);

            for (int jj = j; jj < N && jj < j + JTILESIZE; jj+=16) {
              __m256 c0 = _mm256_load_ps(&C[(ii+0) * N + jj]);


              __m256 b0 = _mm256_load_ps(&B[(kk+0) * N + jj]);
              __m256 b1 = _mm256_load_ps(&B[(kk+1) * N + jj]);
              __m256 b2 = _mm256_load_ps(&B[(kk+2) * N + jj]);
              __m256 b3 = _mm256_load_ps(&B[(kk+3) * N + jj]);
              __m256 b4 = _mm256_load_ps(&B[(kk+4) * N + jj]);
              __m256 b5 = _mm256_load_ps(&B[(kk+5) * N + jj]);
              __m256 b6 = _mm256_load_ps(&B[(kk+6) * N + jj]);
              __m256 b7 = _mm256_load_ps(&B[(kk+7) * N + jj]);

              c0 = _mm256_fmadd_ps(a0, b0, c0);
              c0 = _mm256_fmadd_ps(a1, b1, c0);
              c0 = _mm256_fmadd_ps(a2, b2, c0);
              c0 = _mm256_fmadd_ps(a3, b3, c0);
              c0 = _mm256_fmadd_ps(a4, b4, c0);
              c0 = _mm256_fmadd_ps(a5, b5, c0);
              c0 = _mm256_fmadd_ps(a6, b6, c0);
              c0 = _mm256_fmadd_ps(a7, b7, c0);

              __m256 d0 = _mm256_load_ps(&C[(ii+0) * N + jj+8]);

              __m256 e0 = _mm256_load_ps(&B[(kk+0) * N + jj+8]);
              __m256 e1 = _mm256_load_ps(&B[(kk+1) * N + jj+8]);
              __m256 e2 = _mm256_load_ps(&B[(kk+2) * N + jj+8]);
              __m256 e3 = _mm256_load_ps(&B[(kk+3) * N + jj+8]);
              __m256 e4 = _mm256_load_ps(&B[(kk+4) * N + jj+8]);
              __m256 e5 = _mm256_load_ps(&B[(kk+5) * N + jj+8]);
              __m256 e6 = _mm256_load_ps(&B[(kk+6) * N + jj+8]);
              __m256 e7 = _mm256_load_ps(&B[(kk+7) * N + jj+8]);

              d0 = _mm256_fmadd_ps(a0, e0, d0);
              d0 = _mm256_fmadd_ps(a1, e1, d0);
              d0 = _mm256_fmadd_ps(a2, e2, d0);
              d0 = _mm256_fmadd_ps(a3, e3, d0);
              d0 = _mm256_fmadd_ps(a4, e4, d0);
              d0 = _mm256_fmadd_ps(a5, e5, d0);
              d0 = _mm256_fmadd_ps(a6, e6, d0);
              d0 = _mm256_fmadd_ps(a7, e7, d0);

              _mm256_store_ps(&C[(ii+0)*N+jj], c0);
              _mm256_store_ps(&C[(ii+0)*N+jj+8], d0);
            }
          }
        }
      }
    }
  }

  }

  time_map["matmul"] += timer.elapsed();
}

static inline size_t round_up(size_t x, size_t a) { return (x + a - 1) / a * a; }

/*
 * im2col
 * image shape = (N, C, H, W)
 * col shape = (round_up(C x R x S, 8), N x OH x OW)
 * If C x R x S is not divisible by 8, it will be padded to the next multiple of 8.
 */
void im2col (const Tensor *img, float *col, int N, int C, int H, int W, int OH, int OW,
             int R, int S, int stride, int pad, int dilation) {
  Timer timer;

  const int img_stride_c = H * W;
  const int num_patches = OH * OW;

  #pragma omp parallel for collapse(4)
  for (int n = 0; n < N; n++) {
    for (int c = 0; c < C; c++) {
      for (int r = 0; r < R; r++) {
        for (int s = 0; s < S; s++) {
          int k = c * R * S + r * S + s;
          int h_start = r * dilation - pad;
          int w_start = s * dilation - pad;

          for (int oh = 0, h = h_start; oh < OH; ++oh, h += stride) {
            for (int ow = 0, w = w_start; ow < OW; ++ow, w += stride) {
              float val = 0.0f;
              if (h >= 0 && h < H && w >= 0 && w < W) {
                val = img->buf[n * C * H * W + c * img_stride_c + h * W + w];
              }
              
              col[k * N * num_patches + n * num_patches + oh * OW + ow] = val;
            }
          }
        }
      }
    }
  }
  
  time_map["im2col"] += timer.elapsed();
}

/*
 * Add padding to weight matrix
 * A: weight matrix (K, CRS) = (M, N)
 * B: padded weight matrix (K, round_up(CRS, 8)) = (M_, N_)
*/
void addPadding(float *A, float *B, int K, int C, int R, int S) {
  Timer timer;

  int N_ = round_up(C * R * S, 8);
  int N = C * R * S;

  #pragma omp parallel for
  for (int i = 0; i < K; ++i) {
    float *dst = B + (size_t)i * N_;
    const float *src = A + (size_t)i * N;
    memcpy(dst, src, N * sizeof(float));
    if (N_ > N) {
      memset(dst + N, 0, (N_ - N) * sizeof(float));
    }
  }

  time_map["addPadding"] += timer.elapsed();
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

  int _M = K;
  int _N = N * OH * OW;
  int _K = round_up(C * R * S, 8);

  float *col = (float *)aligned_alloc(32, _N * _K * sizeof(float));
  float *wei = (float *)aligned_alloc(32, _M * _K * sizeof(float));

  addPadding(weight->buf, wei, K, C, R, S);
  im2col(input, col, N, C, H, W, OH, OW, R, S, stride, pad, dilation);
  mat_mul(wei, col, output->buf, _M, _N, _K);

  if (has_bias) {
    #pragma omp parallel for simd
    for (int oh = 0; oh < OH; ++oh) {
      for (int ow = 0; ow < OW; ++ow) {
        for (int k = 0; k < K; ++k) {
          output->buf[(size_t)k * OH * OW + (size_t)oh * OW + ow] += bias->buf[k];
        }
      }
    }
  }

  time_map["Conv2d"] += timer.elapsed();
}

/*
 * Add padding and transpose
 * A: weight matrix (CKRS)
 * B: padded and transposed weight matrix (KRS, C)
*/
void addPaddingTranspose(float *A, float *B, int K, int C, int R, int S) {
  Timer timer;
  int RS = R * S;

  #pragma omp parallel for
  for (int k = 0; k < K; ++k) {
    for (int r = 0; r < R; ++r) {
      for (int s = 0; s < S; ++s) {
        int kr = (k * R + r) * S + s;
        int baseB = kr * C;
        for (int c = 0; c < C; ++c) {
          // idx A: (c * K + k) * R*S + r*S + s
          int idxA = (c * K + k) * RS + r * S + s;
          B[baseB + c] = A[idxA];
        }
      }
    }
  }

    time_map["addPaddingTranspose"] += timer.elapsed();
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
  int N = input->shape[0], C = input->shape[1], H = input->shape[2], W = input->shape[3];
  int K = weight->shape[1], R = weight->shape[2], S = weight->shape[3];
  int OH = output->shape[2], OW = output->shape[3];

  int _M = K * R * S;
  int _N = H * W;
  int _K = C;

  float *wei = (float *)aligned_alloc(32, _M * _K * sizeof(float));
  float *temp = (float *)aligned_alloc(32, _M * _N * sizeof(float));

  addPaddingTranspose(weight->buf, wei, K, C, R, S);

  mat_mul(wei, input->buf, temp, _M, _N, _K);

  memset(output->buf, 0, N * K * OH * OW * sizeof(float));
  // KRS x HW -> K x OH x OW
  #pragma omp parallel for collapse(3)
  for (int k = 0; k < K; ++k) {
    for (int r = 0; r < R; ++r) {
      for (int s = 0; s < S; ++s) {
        int kr = k * R * S + r * S + s; 
        for (int h = 0; h < H; ++h) {
          for (int w = 0; w < W; ++w) {
            int oh = h * stride - pad + r;
            int ow = w * stride - pad + s;
            if (oh >= 0 && oh < OH && ow >= 0 && ow < OW) {
              output->buf[k * (OH * OW) + oh * OW + ow] +=
              temp[kr * (H * W) + h * W + w];
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

  #pragma omp parallel for simd collapse(2)
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

  #pragma omp parallel for 
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

  #pragma omp parallel for simd
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

  #pragma omp parallel for 
  for (size_t oc = 0; oc < out_C; oc++) {
    for (size_t ic = 0; ic < in_C; ic++) {
      for (size_t k = 0; k < kernel_size * kernel_size; k++) {
        size_t idx = oc * in_C * kernel_size * kernel_size + ic * kernel_size * kernel_size + k;
        weight_a->buf[idx] = conv_weight->buf[idx] * style_a->buf[ic] * scale;
      }
    }
  }

  if (demodulate) {
    #pragma omp parallel for
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

    #pragma omp parallel for
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

  #pragma omp parallel for simd collapse(2)
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

  #pragma omp parallel for simd collapse(2) 
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

  #pragma omp parallel for simd
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