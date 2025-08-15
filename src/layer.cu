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

__global__ void reduction_square(float *A, float *sum_square, size_t C) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < C) {
		atomicAdd(sum_square, A[i] * A[i] / C);
	}
}

__global__ void normalize(float *A, float* norm_factor, size_t C) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < C) {
    A[i] *= rsqrtf(*norm_factor + 1e-8f);
  }
}

/*
  * PixelNorm
  * @param [in & out] inout: [N, C]
  * Normalizes the input tensor along dim=1.
  * Equivalent to: input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)
  */
void PixelNorm(Tensor *inout) {
  Timer timer;
  size_t C = inout->shape[1];
  // printf("PixelNorm %d\n", C);

  dim3 blockDim(256);
  dim3 gridDim((C + blockDim.x - 1) / blockDim.x);

  // Allocate and initialize mean_square in global memory
  float *d_mean_square;
  CHECK_CUDA(cudaMalloc(&d_mean_square, sizeof(float)));
  CHECK_CUDA(cudaMemset(d_mean_square, 0, sizeof(float)));

  // Launch kernel 
  reduction_square<<<gridDim, blockDim>>>(inout->buf, d_mean_square, C);
  CHECK_CUDA(cudaDeviceSynchronize());
  normalize<<<gridDim, blockDim>>>(inout->buf, d_mean_square, C);
  CHECK_CUDA(cudaDeviceSynchronize());

  // Free the allocated memory
  CHECK_CUDA(cudaFree(d_mean_square));

  time_map["PixelNorm"] += timer.elapsed();
}


// ---------Upsample and Pad---------

__global__ void upsample_pad_kernel(float *input, float *output, int C, int H, int W, int up, int pad0, int pad1) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int h = blockIdx.y * blockDim.y + threadIdx.y;
  int w = blockIdx.z * blockDim.z + threadIdx.z;

  if(c >= C || h >= H || w >= W) return;

  int OH = H * up + pad0 + pad1;
  int OW = W * up + pad0 + pad1;

  atomicAdd(&output[c * OH * OW + (h * up + pad0) * OW + (w * up + pad0)],
        input[c * H * W + h * W + w]);
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

  // printf("UpsamplePad %d %d %d %d %d %d\n", N, C, H, W, up, pad0, pad1);

  size_t OH = up * H + pad0 + pad1;
  size_t OW = up * W + pad0 + pad1;

  // memset(output->buf, 0, N * C * OH * OW * sizeof(float));
  CHECK_CUDA(cudaMemset(output->buf, 0, N * C * OH * OW * sizeof(float)));

  dim3 blockDim(1, 16, 16);
  dim3 gridDim((C + blockDim.x - 1) / blockDim.x,
               (H + blockDim.y - 1) / blockDim.y,  // H, không phải OH
               (W + blockDim.z - 1) / blockDim.z); // W, không phải OW
  upsample_pad_kernel<<<gridDim, blockDim>>>(input->buf, output->buf, C, H, W, up, pad0, pad1);
  CHECK_CUDA(cudaDeviceSynchronize());

  time_map["UpsamplePad"] += timer.elapsed();
}


__global__ void mat_mul_kernel(float *A_T, float4 *B, float4 *C, int M, int N, int K) {
  int j = blockDim.x * blockIdx.x + threadIdx.x;
  int i = blockDim.y * blockIdx.y + threadIdx.y;
  if (i >= M || j * 4 >= N) return;
  float4 sum = make_float4(0, 0, 0, 0);
  for (int k = 0; k < K; ++k) {
    float a = A_T[k * M + i];
    float4 b = B[k * (N / 4) + j];
    sum = make_float4(sum.x + a * b.x, sum.y + a * b.y,
    sum.z + a * b.z, sum.w + a * b.w);
  }
  C[i * (N / 4) + j] = sum;
}

#define BLOCK_SIZE 16
__global__ void mat_mul_kernel_(float *A_T, float4 *B, float4 *C, int M, int N, int K) {
    // Index computation - each thread handles 4 elements using float4
    int j4 = blockIdx.x * blockDim.x + threadIdx.x;  // j index in float4 units
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int gj = blockIdx.x, gi = blockIdx.y;
    int lj = threadIdx.x, li = threadIdx.y;
    
    // Check bounds in float4 units
    if (gi * BLOCK_SIZE >= M || gj * BLOCK_SIZE >= (N / 4)) return;
    
    // Shared memory allocation
    __shared__ float Alocal[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float4 Blocal[BLOCK_SIZE][BLOCK_SIZE];
    float4 c = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    
    // Iterate over the blocks
    for (int bk = 0; bk < K; bk += BLOCK_SIZE) {
        // Load from global memory to shared memory
        int Ai = gi * BLOCK_SIZE + li;
        int Aj = bk + lj;
        // Load from transposed A (A_T is K x M)
        Alocal[li][lj] = (Aj < K && Ai < M) ? A_T[Aj * M + Ai] : 0.0f;
        
        // Load float4 from B
        int Bi = bk + li;
        int Bj4 = gj * BLOCK_SIZE + lj;  // j index in float4 units
        if (Bi < K && Bj4 < (N / 4)) {
            Blocal[li][lj] = B[Bi * (N / 4) + Bj4];
        } else {
            Blocal[li][lj] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        __syncthreads();
        
        // Accumulate the results in blocks
        for (int lk = 0; lk < BLOCK_SIZE; ++lk) {
            float a_val = Alocal[li][lk];
            float4 b_val = Blocal[lk][lj];
            c.x += a_val * b_val.x;
            c.y += a_val * b_val.y;
            c.z += a_val * b_val.z;
            c.w += a_val * b_val.w;
        }
        __syncthreads();
    }
    
    // Write result to global memory
    if (i < M && j4 < (N / 4)) {
        C[i * (N / 4) + j4] = c;
    }
}

/*
 * Matrix multiplication: C = A * B
 * A: (M, K), B: (K, N), C: (M, N)
 */
void mat_mul(float *A_T, float *B, float *C, int M, int N, int K) {
  printf("Matrix Multiplication %d %d %d.", M, N, K);
  Timer timer;
  
  // memset(C, 0, M * N * sizeof(float));
  CHECK_CUDA(cudaMemset(C, 0, M * N * sizeof(float)));
  if(M < BLOCK_SIZE) {
    dim3 blockDim(256, 1);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);
    mat_mul_kernel<<<gridDim, blockDim>>>(A_T, (float4 *) B, (float4 *) C, M, N, K);
  } else {
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);
    mat_mul_kernel_<<<gridDim, blockDim>>>(A_T, (float4 *) B, (float4 *) C, M, N, K);
  }
  CHECK_CUDA(cudaDeviceSynchronize());
  printf(" Done in %f ms\n", timer.elapsed());
  time_map["matmul"] += timer.elapsed();
}

__host__ __device__ static inline size_t round_up(size_t x, size_t a) { return (x + a - 1) / a * a; }

// ---------Image to col---------
__global__ void im2col_kernel(float *input, float* output, 
                              int N, int C, int H, int W, 
                              int OH, int OW, int R, int S, 
                              int stride, int pad, int dilation) {
  int ow = blockIdx.x * blockDim.x + threadIdx.x;
  int oh = blockIdx.y * blockDim.y + threadIdx.y;
  // n, c calculate from z 
  int n = blockIdx.z / C;
  int c = blockIdx.z % C;

  if (n < N && c < C && oh < OH && ow < OW) {
    int h_start = oh * stride - pad;
    int w_start = ow * stride - pad;
    for (int r = 0; r < R; ++r) {
      for (int s = 0; s < S; ++s) {
        int h = h_start + r * dilation;
        int w = w_start + s * dilation;
        if (h >= 0 && h < H && w >= 0 && w < W) {
          output[((c * R + r) * S + s) * N * OH * OW + n * OH * OW + oh * OW + ow] =
              input[n * C * H * W + c * H * W + h * W + w];
        } else {
          output[((c * R + r) * S + s) * N * OH * OW + n * OH * OW + oh * OW + ow] = 0.0f;
        }
      }
    }
  }

}

/*
 * im2col
 * image shape = (N, C, H, W)
 * col shape = (round_up(C x R x S, 8), N x OH x OW)
 * If C x R x S is not divisible by 8, it will be padded to the next multiple of 8.
 */
void im2col (const Tensor *img, float *col, int N, int C, int H, int W, int OH, int OW,
             int R, int S, int stride, int pad, int dilation) {
  // printf("im2col %d %d %d %d %d %d %d %d %d %d %d %d %d\n", N, C, H, W, OH, OW, R, S, stride, pad, dilation);
  Timer timer;

  CHECK_CUDA(cudaMemset(col, 0, round_up(C * R * S, 8) * N * OH * OW * sizeof(float)));
  dim3 blockDim(16, 16);
  dim3 gridDim((OW + blockDim.x - 1) / blockDim.x, (OH + blockDim.y - 1) / blockDim.y, N * C);

  im2col_kernel<<<gridDim, blockDim>>>(img->buf, col, N, C, H, W, OH, OW, R, S, stride, pad, dilation);
  CHECK_CUDA(cudaDeviceSynchronize());

  time_map["im2col"] += timer.elapsed();
}

// --------Convolution Layer---------
__global__ void add_bias_kernel(float *output, float *bias, int K, int OH, int OW) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  int oh = blockIdx.y * blockDim.y + threadIdx.y;
  int ow = blockIdx.z * blockDim.z + threadIdx.z;

  if (k < K && oh < OH && ow < OW) {
    atomicAdd(&output[k * OH * OW + oh * OW + ow], bias[k]);
  }
}

__global__ void add_padding_kernel(float *A, float *B, int K, int C, int R, int S) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  
  int N = C * R * S;
  int N_ = round_up(N, 8);

  if (k < K && j < N_) {
    if (j < N) {
      B[j * K + k] = A[k * N + j];
    } else {
      B[j * K + k] = 0.0f;
    }
  }
}

void addPadding(float *A, float *B, int K, int C, int R, int S) {
  Timer timer;

  int N_ = round_up(C * R * S, 8);
  int N = C * R * S;

  dim3 blockDim(16, 16);
  dim3 gridDim((K + blockDim.x - 1) / blockDim.x, (N_ + blockDim.y - 1) / blockDim.y);
  add_padding_kernel<<<gridDim, blockDim>>>(A, B, K, C, R, S);
  CHECK_CUDA(cudaDeviceSynchronize());

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

  // printf("Conv2d %d %d %d %d %d %d %d %d %d %d %d %d %d\n", N, C, H, W, OH, OW, R, S, stride, pad, dilation);

  int _M = K;
  int _N = N * OH * OW;
  int _K = round_up(C * R * S, 8);

  // float *col = (float *)aligned_alloc(32, _N * _K * sizeof(float));
  // float *wei = (float *)aligned_alloc(32, _M * _K * sizeof(float));
  float *col, *wei;
  CHECK_CUDA(cudaMalloc((void**)&col, _N * _K * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void**)&wei, _M * _K * sizeof(float)));

  addPadding(weight->buf, wei, K, C, R, S);
  im2col(input, col, N, C, H, W, OH, OW, R, S, stride, pad, dilation);
  mat_mul(wei, col, output->buf, _M, _N, _K);

  if (has_bias) {
    dim3 blockDim(1, 16, 16);
    dim3 gridDim((K + blockDim.x - 1) / blockDim.x, (OH + blockDim.y - 1) / blockDim.y, (OW + blockDim.z - 1) / blockDim.z);
    add_bias_kernel<<<gridDim, blockDim>>>(output->buf, bias->buf, K, OH, OW);
    CHECK_CUDA(cudaDeviceSynchronize());
  }

  // Free memory - thêm dòng này
  CHECK_CUDA(cudaFree(col));
  CHECK_CUDA(cudaFree(wei));  // Thêm dòng này
  
  time_map["Conv2d"] += timer.elapsed();
}


// --------Transpose Convolution Layer---------

__global__ void addPaddingTranspose_kernel(float *A, float *B, int K, int C, int R, int S) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  int c = blockIdx.y * blockDim.y + threadIdx.y;

  if (k < K && c < C) {
    for (int r = 0; r < R; ++r) {
      for (int s = 0; s < S; ++s) {
        int krs = (k * R + r) * S + s;
        int idxA = (c * K + k) * R * S + r * S + s;
        int idxB = c * K * R * S + krs;
        B[idxB] = A[idxA];
      }
    }
  }
}
/*
 * Add padding and transpose
 * A: weight matrix (CKRS)
 * B: padded and transposed weight matrix (KRS, C)
*/
void addPaddingTranspose(float *A, float *B, int K, int C, int R, int S) {
  Timer timer;
  // printf("addPaddingTranspose %d %d %d %d %d %d\n", K, C, R, S);
  dim3 blockDim(16, 16);
  dim3 gridDim((K + blockDim.x - 1) / blockDim.x, 
               (C + blockDim.y - 1) / blockDim.y);
  // memset(B, 0, K * R * S * C * sizeof(float));
  addPaddingTranspose_kernel<<<gridDim, blockDim>>>(A, B, K, C, R, S);
  CHECK_CUDA(cudaDeviceSynchronize());

  time_map["addPaddingTranspose"] += timer.elapsed();
}


__global__ void reshape_kernel(float *input, float *output, 
                               int N, int K, int H, int W, 
                               int R, int S, int stride, int pad,
                               int OH, int OW) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  int h = blockIdx.y * blockDim.y + threadIdx.y;
  int w = blockIdx.z * blockDim.z + threadIdx.z;

  if (k < K && h < H && w < W) {
    for (int r = 0; r < R; r++) {
      for (int s = 0; s < S; s++) {
        int kr = k * R * S + r * S + s; 
        int oh = h * stride - pad + r;
        int ow = w * stride - pad + s;
        if (oh >= 0 && oh < OH && ow >= 0 && ow < OW) {
          atomicAdd(&output[k * (OH * OW) + oh * OW + ow],
          input[kr * (H * W) + h * W + w]);
        }
      }
    }
  }
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

  // printf("ConvTranspose2d %d %d %d %d %d %d %d %d %d %d %d %d %d\n", N, C, H, W, K, R, S, OH, OW, stride, pad);


  int _M = K * R * S;
  int _N = H * W;
  int _K = C;

  float *wei, *temp;
  CHECK_CUDA(cudaMalloc((void**)&wei, _M * _K * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void**)&temp, _M * _N * sizeof(float)));

  addPaddingTranspose(weight->buf, wei, K, C, R, S);

  mat_mul(wei, input->buf, temp, _M, _N, _K);

  // memset(output->buf, 0, N * K * OH * OW * sizeof(float));
  CHECK_CUDA(cudaMemset(output->buf, 0, N * K * OH * OW * sizeof(float)));
  dim3 blockDim(1, 16, 16);
  dim3 gridDim((K + blockDim.x - 1) / blockDim.x, 
               (H + blockDim.y - 1) / blockDim.y, 
               (W + blockDim.z - 1) / blockDim.z);
  reshape_kernel<<<gridDim, blockDim>>>(temp, output->buf, N, K, H, W, R, S, stride, pad, OH, OW);
  CHECK_CUDA(cudaDeviceSynchronize());
  // Free memory
  CHECK_CUDA(cudaFree(wei));
  CHECK_CUDA(cudaFree(temp));

  time_map["ConvTranspose2d"] += timer.elapsed();
}

// --------Transpose---------
__global__ void transpose_kernel(float *input, float *output, size_t N, size_t C, size_t H, size_t W) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int h = blockIdx.y * blockDim.y + threadIdx.y;
  int w = blockIdx.z * blockDim.z + threadIdx.z;

  if (c < C && w < W && h < H) {
    for (size_t n = 0; n < N; n++) { 
      size_t input_idx  = ((n * C + c) * H + h) * W + w;         // NCHW
      size_t output_idx = ((c * N + n) * H + h) * W + w;         // CNHW
      output[output_idx] = input[input_idx];
    }
  }
}

/* Transpose
 * input shape = (N, C, H, W)
 * output shape = (C, N, H, W)
 * Transposes the first two dimensions of the input tensor.
 */
void transpose(Tensor *input, Tensor *output) {
  // printf("Transpose\n");
  Timer timer;
  size_t N = input->shape[0];
  size_t C = input->shape[1];
  size_t H = input->shape[2];
  size_t W = input->shape[3];

  dim3 blockDim(1, 16, 16);
  dim3 gridDim((C + blockDim.x - 1) / blockDim.x, 
               (H + blockDim.y - 1) / blockDim.y, 
               (W + blockDim.z - 1) / blockDim.z);

  transpose_kernel<<<gridDim, blockDim>>>(input->buf, output->buf, N, C, H, W);
  CHECK_CUDA(cudaDeviceSynchronize());

  time_map["transpose"] += timer.elapsed();
}

// --------Linear Layer---------
__global__ void linear_kernel(float *in, float *w, float *b, float *out, 
                            size_t M, size_t N, size_t K, float scale, float lr_mul) {
  int m = blockIdx.x * blockDim.x + threadIdx.x;
  int n = blockIdx.y * blockDim.y + threadIdx.y;
  if (m < M && n < N) {
    float sum = 0.0f;
    for (size_t k = 0; k < K; k++) {
      sum += in[m * K + k] * w[n * K + k] * scale;
    }
    out[m * N + n] = sum + b[n] * lr_mul;
  }
}

/* Linear
 * @param [in1]  in: [M, K]
 * @param [in2]   w: [N, K]
 * @param [in3]   b: [N]
 * @param [out] out: [M, N]
 */
void Linear(Tensor *in, Tensor *w, Tensor *b, Tensor *out, float lr_mul) {
  // printf("Linear\n");
  Timer timer;
  size_t M = out->shape[0];
  size_t N = out->shape[1];
  size_t K = w->shape[1];

  float scale = (1.0f / sqrtf(K)) * lr_mul;

  dim3 blockDim(16, 16);
  dim3 gridDim((M + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);
  linear_kernel<<<gridDim, blockDim>>>(in->buf, w->buf, b->buf, out->buf, M, N, K, scale, lr_mul);
  CHECK_CUDA(cudaDeviceSynchronize());  

  time_map["Linear"] += timer.elapsed();
}

// ---------LeakyReLU---------
__global__ void leaky_relu_kernel(float *inout, float negative_slope, float scale, size_t N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    if (inout[i] < 0) {
      inout[i] *= negative_slope;
    }
    inout[i] *= scale;
  }
}

/* LeakyReLU
 * @param [in & out] inout: [N]
 * 'N' is the number of elements in the tensor.
 */
void LeakyReLU(Tensor *inout) {
  // printf("LeakyReLU\n");
  Timer timer;
  size_t N = inout->num_elem();

  float negative_slope = 0.2f;
  float scale = sqrtf(2.0f);

  dim3 blockDim(256);
  dim3 gridDim((N + blockDim.x - 1) / blockDim.x);
  leaky_relu_kernel<<<gridDim, blockDim>>>(inout->buf, negative_slope, scale, N);
  CHECK_CUDA(cudaDeviceSynchronize());

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

// ---------Modulated Convolution---------

__global__ void weight_modulation_kernel(float *conv_weight, float *style_a, float *weight_a, float scale, size_t out_C, size_t in_C, size_t kernel_size) {
  int oc = blockIdx.x * blockDim.x + threadIdx.x;
  if (oc >= out_C) return;

  for (size_t ic = 0; ic < in_C; ic++) {
    for (size_t k = 0; k < kernel_size * kernel_size; k++) {
      size_t idx = oc * in_C * kernel_size * kernel_size + ic * kernel_size * kernel_size + k;
      float style_value = style_a[ic];
      float conv_weight_value = conv_weight[idx];
      weight_a[idx] = conv_weight_value * style_value * scale;
    }
  }
}

__global__ void demodulation_kernel(float *weight_a, float *demod_a, size_t out_C, size_t in_C, size_t kernel_size) {
  int oc = blockIdx.x * blockDim.x + threadIdx.x;

  if (oc < out_C) {
    float sum = 0.0f;
    for (size_t ic = 0; ic < in_C; ic++) {
      for (size_t k = 0; k < kernel_size * kernel_size; k++) {
        size_t idx = oc * in_C * kernel_size * kernel_size + ic * kernel_size * kernel_size + k;
        sum += weight_a[idx] * weight_a[idx];
      }
    }
    demod_a[oc] = 1.0f / sqrtf(sum + 1e-8f);
  }
}

__global__ void apply_demodulation_kernel(float *weight_a, float *demod_a, size_t out_C, size_t in_C, size_t kernel_size) {
  int oc = blockIdx.x * blockDim.x + threadIdx.x;
  if (oc < out_C) {
    for (size_t ic = 0; ic < in_C; ic++) {
      for (size_t k = 0; k < kernel_size * kernel_size; k++) {
        size_t idx = oc * in_C * kernel_size * kernel_size + ic * kernel_size * kernel_size + k;
        weight_a[idx] *= demod_a[oc];
      }
    }
  }
}

void ModulatedConv2d(Tensor *input, Tensor *style, Tensor *modulate_weight, Tensor *modulate_bias, Tensor *conv_weight, Tensor *kernel, Tensor *output,
                     Tensor *style_a, Tensor *weight_a, Tensor *demod_a, Tensor *transpose_a, Tensor *conv_a, Tensor *upsample_a, Tensor *conv2_a,
                     bool demodulate, bool upsample, int padding, int up
) {
  Timer timer;
  size_t in_C = input->shape[1];
  size_t out_C = conv_weight->shape[0];
  size_t kernel_size = conv_weight->shape[2];
  // printf("ModulatedConv2d %d %d %d\n", in_C, out_C, kernel_size);


  Linear(style, modulate_weight, modulate_bias, style_a, 1.0f);

  float scale = 1 / sqrtf((float) (in_C * kernel_size * kernel_size));

  dim3 blockDim(256);
  dim3 gridDim((out_C + blockDim.x - 1) / blockDim.x);

  weight_modulation_kernel<<<gridDim, blockDim>>>(conv_weight->buf, style_a->buf, weight_a->buf, scale, out_C, in_C, kernel_size);
  CHECK_CUDA(cudaDeviceSynchronize());

  if (demodulate) {
    demodulation_kernel<<<(out_C + blockDim.x - 1) / blockDim.x, blockDim>>>(weight_a->buf, demod_a->buf, out_C, in_C, kernel_size);
    CHECK_CUDA(cudaDeviceSynchronize());
    apply_demodulation_kernel<<<(out_C + blockDim.x - 1) / blockDim.x, blockDim>>>(weight_a->buf, demod_a->buf, out_C, in_C, kernel_size);
    CHECK_CUDA(cudaDeviceSynchronize());    
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


// ---------Add Noise---------
__global__ void addNoiseKernel(float *inout, float *noise, size_t C, size_t H, size_t W) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int h = blockIdx.y * blockDim.y + threadIdx.y;
  int w = blockIdx.z * blockDim.z + threadIdx.z;

  if (c < C && h < H && w < W) {
    size_t idx = (c * H + h) * W + w;
    inout[idx] += noise[h * W + w];
  }
}

/* Add noise to the input tensor
 * @param [in & out] inout: [N, C, H, W]
 * @param [in] noise: [H, W]
 * Adds noise to the input tensor in-place.
 */

void addNoise(Tensor *inout, Tensor *noise) {
  // printf("addNoise\n");
  Timer timer;
  size_t C = inout->shape[1];
  size_t H = inout->shape[2];
  size_t W = inout->shape[3];

  dim3 blockDim(1, 16, 16);
  dim3 gridDim((C + blockDim.x - 1) / blockDim.x,
               (H + blockDim.y - 1) / blockDim.y,
               (W + blockDim.z - 1) / blockDim.z);
  addNoiseKernel<<<gridDim, blockDim>>>(inout->buf, noise->buf, C, H, W);
  CHECK_CUDA(cudaDeviceSynchronize());

  time_map["addNoise"] += timer.elapsed();
}

// ---------Add Bias---------
__global__ void addBiasKernel(float *inout, float *bias, size_t C, size_t H, size_t W) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int h = blockIdx.y * blockDim.y + threadIdx.y;
  int w = blockIdx.z * blockDim.z + threadIdx.z;

  if (c < C && h < H && w < W) {
    size_t idx = (c * H + h) * W + w;
    inout[idx] += bias[c];
  }
}

/* Add bias to the input tensor
 * @param [in & out] inout: [N, C, H, W]
 * @param [in] bias: [C]
 * Adds bias to the input tensor in-place.
 */

void addBias(Tensor *inout, Tensor *bias) {
  // printf("addBias\n");
  Timer timer;
  size_t C = inout->shape[1];
  size_t H = inout->shape[2];
  size_t W = inout->shape[3];

  dim3 blockDim(1, 16, 16);
  dim3 gridDim((C + blockDim.x - 1) / blockDim.x,
               (H + blockDim.y - 1) / blockDim.y,
               (W + blockDim.z - 1) / blockDim.z);
  addBiasKernel<<<gridDim, blockDim>>>(inout->buf, bias->buf, C, H, W);
  CHECK_CUDA(cudaDeviceSynchronize());

  time_map["addBias"] += timer.elapsed();
}

// -------------Element-wise addition---------
__global__ void elemAddKernel(float *inout, float *addend, size_t N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    inout[i] += addend[i];
  }
}

/*
 * Element-wise addition of two tensors
 * @param [in & out] inout: [N, C, H, W]
 * @param [in] addend: [N, C, H, W]
 * Adds the elements of addend to inout in-place.
 */
void elemAdd(Tensor *inout, Tensor *addend) {
  // printf("elemAdd\n");
  Timer timer;
  size_t N = inout->num_elem();

  dim3 blockDim(256);
  dim3 gridDim((N + blockDim.x - 1) / blockDim.x);
  elemAddKernel<<<gridDim, blockDim>>>(inout->buf, addend->buf, N);
  CHECK_CUDA(cudaDeviceSynchronize());

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