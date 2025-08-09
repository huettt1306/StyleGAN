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
#define KTILESIZE (64)

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
  #pragma omp parallel for simd reduction(+:mean_squares)
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
          for (int kk = k; kk < K && kk < k + KTILESIZE; kk += 8) {
            for (int ii = i; ii < M && ii < i + ITILESIZE; ii++) {
              __m256 a0 = _mm256_set1_ps(A[ii*K + (kk+0)]);
              __m256 a1 = _mm256_set1_ps(A[ii*K + (kk+1)]);
              __m256 a2 = _mm256_set1_ps(A[ii*K + (kk+2)]);
              __m256 a3 = _mm256_set1_ps(A[ii*K + (kk+3)]);
              __m256 a4 = _mm256_set1_ps(A[ii*K + (kk+4)]);
              __m256 a5 = _mm256_set1_ps(A[ii*K + (kk+5)]);
              __m256 a6 = _mm256_set1_ps(A[ii*K + (kk+6)]);
              __m256 a7 = _mm256_set1_ps(A[ii*K + (kk+7)]);

              for (int jj = j; jj < N && jj < j + JTILESIZE; jj += 8) {
                __m256 c0 = _mm256_load_ps(&C[ii * N + jj]);

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

                _mm256_store_ps(&C[ii * N + jj], c0);
              }
            }
          }
        }
      }
    }
  }

  time_map["matmul"] += timer.elapsed();
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
  #pragma omp parallel for collapse(2)
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
  #pragma omp parallel for collapse(2)
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

  size_t M = (size_t)N * OH * OW;
  size_t K0 = (size_t)C * R * S;
  size_t Kc = round_up(K0, 8);
  size_t Nout = (size_t)K;
  size_t Nc = round_up(Nout, 8); 

  float *A, *B, *Cmat;
  A    = (float*) aligned_alloc(32, M * Kc * sizeof(float));
  B    = (float*) aligned_alloc(32, Kc * Nc * sizeof(float));
  Cmat = (float*) aligned_alloc(32, M * Nc * sizeof(float));

  im2col_nchw(input, A, Kc, stride, pad, dilation, R, S, N, C, H, W, OH, OW);

  #pragma omp parallel num_threads(32)
  {
    int tid = omp_get_thread_num();
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(tid, &cpuset);
    sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);

    #pragma omp for simd
    for (size_t i = 0; i < Kc * Nc; i++) {
      B[i] = 0.0f;
    }

    #pragma omp for collapse(2)
    for (int k = 0; k < K; ++k) {
      for (int c = 0; c < C; ++c) {
        for (int r = 0; r < R; ++r) {
          for (int s_ = 0; s_ < S; ++s_) {
            size_t row = (size_t)c * R * S + r * S + s_;
            B[row * Nc + k] =
              weight->buf[(size_t)k * C * R * S + (size_t)c * R * S + r * S + s_];
          }
        }
      }
    }
  }

  mat_mul(A, B, Cmat, (int)M, (int)Nc, (int)Kc);

  #pragma omp parallel for collapse(2)
  for (int n = 0; n < N; ++n) {
    for (int oh = 0; oh < OH; ++oh) {
      for (int ow = 0; ow < OW; ++ow) {
        size_t m = ((size_t)n * OH + oh) * OW + ow;
        for (int k = 0; k < K; ++k) {
          float v = Cmat[m * Nc + k];
          if (has_bias) v += bias->buf[k];
          output->buf[((size_t)n * K + k) * OH * OW + (size_t)oh * OW + ow] = v;
        }
      }
    }
  }

  // free(A); free(B); free(Cmat);
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

  size_t M    = (size_t)N * OH * OW;
  size_t K0   = (size_t)C * R * S;
  size_t Kc   = round_up(K0, 8);
  size_t Nout = (size_t)K;
  size_t Nc   = round_up(Nout, 8);

  float *A, *B, *Cmat;
  A    = (float*) aligned_alloc(32, M * Kc * sizeof(float));
  B    = (float*) aligned_alloc(32, Kc * Nc * sizeof(float));
  Cmat = (float*) aligned_alloc(32, M * Nc * sizeof(float));

  im2col_tconv_nchw(input, A, Kc, stride, pad, R, S, N, C, H, W, OH, OW);

  #pragma omp parallel num_threads(32)
  {
    int tid = omp_get_thread_num();
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(tid, &cpuset);
    sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);

    #pragma omp for simd
    for (size_t i = 0; i < Kc * Nc; i++) {
      B[i] = 0.0f;
    }

    #pragma omp for collapse(2)
    for (int k = 0; k < K; ++k) {
      for (int c = 0; c < C; ++c) {
        for (int r = 0; r < R; ++r) {
          for (int s_ = 0; s_ < S; ++s_) {
            size_t row = (size_t)c * R * S + r * S + s_;
            B[row * Nc + k] =
              weight->buf[((size_t)c * K + k) * R * S + (size_t)r * S + s_];
          }
        }
      }
    }
  }

  mat_mul(A, B, Cmat, (int)M, (int)Nc, (int)Kc);

  #pragma omp parallel for collapse(2)
  for (int n = 0; n < N; ++n) {
    for (int oh = 0; oh < OH; ++oh) {
      for (int ow = 0; ow < OW; ++ow) {
        size_t m = ((size_t)n * OH + oh) * OW + ow;
        for (int k = 0; k < K; ++k) {
          output->buf[((size_t)n * K + k) * OH * OW + (size_t)oh * OW + ow] =
              Cmat[m * Nc + k];
        }
      }
    }
  }

  free(A); free(B); free(Cmat);
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

  #pragma omp parallel for collapse(2)
  for (size_t n = 0; n < N; n++) {
    for (size_t c = 0; c < C; c++) {
      for (size_t h = 0; h < H; h++) {
        size_t base_in  = ((n * C + c) * H + h) * W;
        size_t base_out = ((c * N + n) * H + h) * W;
        size_t w = 0;
        for (; w + 8 <= W; w += 8) {
          __m256 vec = _mm256_load_ps(input->buf + base_in + w);
          _mm256_store_ps(output->buf + base_out + w, vec);
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

  #pragma omp parallel for 
  for (size_t m = 0; m < M; m++) {
    for (size_t n = 0; n < N; n ++) {
      size_t k;
      __m256 x, y, s = _mm256_setzero_ps();
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
  size_t K = kernel_size * kernel_size;
  

  #pragma omp parallel for
  for (size_t oc = 0; oc < out_C; oc++) {
    for (size_t ic = 0; ic < in_C; ic++) {
      __m256 style = _mm256_set1_ps(style_a->buf[ic] * scale), a, s;
      size_t K_idx = oc * in_C * K + ic * K, k;

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
    #pragma omp parallel for
    for (size_t oc = 0; oc < out_C; oc++) {
      float sum = 0.0f;
      __m256 a, s;
      for (size_t ic = 0; ic < in_C; ic++) {
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

    #pragma omp parallel for
    for (size_t oc = 0; oc < out_C; oc++) {
      __m256 demod = _mm256_set1_ps(demod_a->buf[oc]), a, s;
      for (size_t ic = 0; ic < in_C; ic++) {
        size_t K_idx = oc * in_C * K + ic * K, k;
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


  #pragma omp parallel for
  for (size_t c = 0; c < C; c++) {
    for (size_t h = 0; h < H; h++) {
      size_t w;
      size_t base_idx = (c * H + h) * W;
      size_t noise_idx = h * W;
      __m256 a, b, result;

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

  #pragma omp parallel for
  for (size_t c = 0; c < C; c++) {
    __m256 bias_ = _mm256_set1_ps(bias->buf[c]), a, result;
    for (size_t h = 0; h < H; h++) {
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