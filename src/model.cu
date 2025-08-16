#include <cstdio>
#include <mpi.h>
#include "layer.h"
#include "model.h"
#include <thread>


/* [Model Parameters]
 * _w: Weight parameter
 * _b: Bias parameter
 */
// Multi-layer perceptron (MLP) parameters
const int NGPUS = 16;
const int GPUS_PER_RANK = 4;

Parameter *mlp0_w[NGPUS], *mlp0_b[NGPUS];
Parameter *mlp1_w[NGPUS], *mlp1_b[NGPUS];
Parameter *mlp2_w[NGPUS], *mlp2_b[NGPUS];
Parameter *mlp3_w[NGPUS], *mlp3_b[NGPUS];
Parameter *mlp4_w[NGPUS], *mlp4_b[NGPUS];
Parameter *mlp5_w[NGPUS], *mlp5_b[NGPUS];
Parameter *mlp6_w[NGPUS], *mlp6_b[NGPUS];
Parameter *mlp7_w[NGPUS], *mlp7_b[NGPUS];
Parameter *constant_input[NGPUS];  // Constant input for the model
Parameter *kernel[NGPUS];  // Blur kernel

// conv1
Parameter *conv1_modulate_w[NGPUS], *conv1_modulate_b[NGPUS];
Parameter *conv1_w[NGPUS], *conv1_b[NGPUS];

// torgb1
Parameter *to_rgb_modulate_w[NGPUS], *to_rgb_modulate_b[NGPUS];
Parameter *to_rgb_w[NGPUS], *to_rgb_b[NGPUS];

// Parameters for 7 blocks
Parameter *block0_conv_up_modulate_w[NGPUS], *block0_conv_up_modulate_b[NGPUS], *block0_conv_up_w[NGPUS], *block0_conv_up_b[NGPUS];
Parameter *block0_conv_modulate_w[NGPUS], *block0_conv_modulate_b[NGPUS], *block0_conv_w[NGPUS], *block0_conv_b[NGPUS];
Parameter *block0_to_rgb_modulate_w[NGPUS], *block0_to_rgb_modulate_b[NGPUS], *block0_to_rgb_w[NGPUS], *block0_to_rgb_b[NGPUS];

Parameter *block1_conv_up_modulate_w[NGPUS], *block1_conv_up_modulate_b[NGPUS], *block1_conv_up_w[NGPUS], *block1_conv_up_b[NGPUS];
Parameter *block1_conv_modulate_w[NGPUS], *block1_conv_modulate_b[NGPUS], *block1_conv_w[NGPUS], *block1_conv_b[NGPUS];
Parameter *block1_to_rgb_modulate_w[NGPUS], *block1_to_rgb_modulate_b[NGPUS], *block1_to_rgb_w[NGPUS], *block1_to_rgb_b[NGPUS];

Parameter *block2_conv_up_modulate_w[NGPUS], *block2_conv_up_modulate_b[NGPUS], *block2_conv_up_w[NGPUS], *block2_conv_up_b[NGPUS];
Parameter *block2_conv_modulate_w[NGPUS], *block2_conv_modulate_b[NGPUS], *block2_conv_w[NGPUS], *block2_conv_b[NGPUS];
Parameter *block2_to_rgb_modulate_w[NGPUS], *block2_to_rgb_modulate_b[NGPUS], *block2_to_rgb_w[NGPUS], *block2_to_rgb_b[NGPUS];

Parameter *block3_conv_up_modulate_w[NGPUS], *block3_conv_up_modulate_b[NGPUS], *block3_conv_up_w[NGPUS], *block3_conv_up_b[NGPUS];
Parameter *block3_conv_modulate_w[NGPUS], *block3_conv_modulate_b[NGPUS], *block3_conv_w[NGPUS], *block3_conv_b[NGPUS];
Parameter *block3_to_rgb_modulate_w[NGPUS], *block3_to_rgb_modulate_b[NGPUS], *block3_to_rgb_w[NGPUS], *block3_to_rgb_b[NGPUS];

Parameter *block4_conv_up_modulate_w[NGPUS], *block4_conv_up_modulate_b[NGPUS], *block4_conv_up_w[NGPUS], *block4_conv_up_b[NGPUS];
Parameter *block4_conv_modulate_w[NGPUS], *block4_conv_modulate_b[NGPUS], *block4_conv_w[NGPUS], *block4_conv_b[NGPUS];
Parameter *block4_to_rgb_modulate_w[NGPUS], *block4_to_rgb_modulate_b[NGPUS], *block4_to_rgb_w[NGPUS], *block4_to_rgb_b[NGPUS];

Parameter *block5_conv_up_modulate_w[NGPUS], *block5_conv_up_modulate_b[NGPUS], *block5_conv_up_w[NGPUS], *block5_conv_up_b[NGPUS];
Parameter *block5_conv_modulate_w[NGPUS], *block5_conv_modulate_b[NGPUS], *block5_conv_w[NGPUS], *block5_conv_b[NGPUS];
Parameter *block5_to_rgb_modulate_w[NGPUS], *block5_to_rgb_modulate_b[NGPUS], *block5_to_rgb_w[NGPUS], *block5_to_rgb_b[NGPUS];

Parameter *block6_conv_up_modulate_w[NGPUS], *block6_conv_up_modulate_b[NGPUS], *block6_conv_up_w[NGPUS], *block6_conv_up_b[NGPUS];
Parameter *block6_conv_modulate_w[NGPUS], *block6_conv_modulate_b[NGPUS], *block6_conv_w[NGPUS], *block6_conv_b[NGPUS];
Parameter *block6_to_rgb_modulate_w[NGPUS], *block6_to_rgb_modulate_b[NGPUS], *block6_to_rgb_w[NGPUS], *block6_to_rgb_b[NGPUS];

// Noise parameters for each layer
Parameter *conv1_noise[NGPUS];
Parameter *block0_noise1[NGPUS], *block0_noise2[NGPUS];
Parameter *block1_noise1[NGPUS], *block1_noise2[NGPUS];
Parameter *block2_noise1[NGPUS], *block2_noise2[NGPUS];
Parameter *block3_noise1[NGPUS], *block3_noise2[NGPUS];
Parameter *block4_noise1[NGPUS], *block4_noise2[NGPUS];
Parameter *block5_noise1[NGPUS], *block5_noise2[NGPUS];
Parameter *block6_noise1[NGPUS], *block6_noise2[NGPUS];


void alloc_and_set_parameters(float *param, size_t param_size) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  for(int local_gpu = 0; local_gpu < GPUS_PER_RANK; local_gpu++) {
    cudaSetDevice(local_gpu);
    int gpu = local_gpu + rank * GPUS_PER_RANK;
    size_t pos = 0;

    mlp0_w[gpu] = new Parameter({512, 512}, param + pos); pos += 512 * 512;
    mlp0_b[gpu] = new Parameter({512}, param + pos); pos += 512;

    mlp1_w[gpu] = new Parameter({512, 512}, param + pos); pos += 512 * 512;
    mlp1_b[gpu] = new Parameter({512}, param + pos); pos += 512;

    mlp2_w[gpu] = new Parameter({512, 512}, param + pos); pos += 512 * 512;
    mlp2_b[gpu] = new Parameter({512}, param + pos); pos += 512;

    mlp3_w[gpu] = new Parameter({512, 512}, param + pos); pos += 512 * 512;
    mlp3_b[gpu] = new Parameter({512}, param + pos); pos += 512;

    mlp4_w[gpu] = new Parameter({512, 512}, param + pos); pos += 512 * 512;
    mlp4_b[gpu] = new Parameter({512}, param + pos); pos += 512;

    mlp5_w[gpu] = new Parameter({512, 512}, param + pos); pos += 512 * 512;
    mlp5_b[gpu] = new Parameter({512}, param + pos); pos += 512;

    mlp6_w[gpu] = new Parameter({512, 512}, param + pos); pos += 512 * 512;
    mlp6_b[gpu] = new Parameter({512}, param + pos); pos += 512;

    mlp7_w[gpu] = new Parameter({512, 512}, param + pos); pos += 512 * 512;
    mlp7_b[gpu] = new Parameter({512}, param + pos); pos += 512;

    constant_input[gpu] = new Parameter({1, 512, 4, 4}, param + pos); pos += 512 * 4 * 4;

    kernel[gpu] = new Parameter({1, 1, 4, 4}, param + pos); pos += 4 * 4;

    conv1_modulate_w[gpu] = new Parameter({512, 512}, param + pos); pos += 512 * 512;
    conv1_modulate_b[gpu] = new Parameter({512}, param + pos); pos += 512;
    conv1_w[gpu] = new Parameter({512, 512, 3, 3}, param + pos); pos += 512 * 512 * 3 * 3;
    conv1_b[gpu] = new Parameter({512}, param + pos); pos += 512;

    to_rgb_modulate_w[gpu] = new Parameter({512, 512}, param + pos); pos += 512 * 512;
    to_rgb_modulate_b[gpu] = new Parameter({512}, param + pos); pos += 512;
    to_rgb_w[gpu] = new Parameter({3, 512, 1, 1}, param + pos); pos += 3 * 512 * 1 * 1;
    to_rgb_b[gpu] = new Parameter({3}, param + pos); pos += 3;

    block0_conv_up_modulate_w[gpu] = new Parameter({512, 512}, param + pos); pos += 512 * 512;
    block0_conv_up_modulate_b[gpu] = new Parameter({512}, param + pos); pos += 512;
    block0_conv_up_w[gpu] = new Parameter({512, 512, 3, 3}, param + pos); pos += 512 * 512 * 3 * 3;
    block0_conv_up_b[gpu] = new Parameter({512}, param + pos); pos += 512;

    block0_conv_modulate_w[gpu] = new Parameter({512, 512}, param + pos); pos += 512 * 512;
    block0_conv_modulate_b[gpu] = new Parameter({512}, param + pos); pos += 512;
    block0_conv_w[gpu] = new Parameter({512, 512, 3, 3}, param + pos); pos += 512 * 512 * 3 * 3;
    block0_conv_b[gpu] = new Parameter({512}, param + pos); pos += 512;

    block0_to_rgb_modulate_w[gpu] = new Parameter({512, 512}, param + pos); pos += 512 * 512;
    block0_to_rgb_modulate_b[gpu] = new Parameter({512}, param + pos); pos += 512;
    block0_to_rgb_w[gpu] = new Parameter({3, 512, 1, 1}, param + pos); pos += 3 * 512 * 1 * 1;
    block0_to_rgb_b[gpu] = new Parameter({3}, param + pos); pos += 3;

    block1_conv_up_modulate_w[gpu] = new Parameter({512, 512}, param + pos); pos += 512 * 512;
    block1_conv_up_modulate_b[gpu] = new Parameter({512}, param + pos); pos += 512;
    block1_conv_up_w[gpu] = new Parameter({512, 512, 3, 3}, param + pos); pos += 512 * 512 * 3 * 3;
    block1_conv_up_b[gpu] = new Parameter({512}, param + pos); pos += 512;

    block1_conv_modulate_w[gpu] = new Parameter({512, 512}, param + pos); pos += 512 * 512;
    block1_conv_modulate_b[gpu] = new Parameter({512}, param + pos); pos += 512;
    block1_conv_w[gpu] = new Parameter({512, 512, 3, 3}, param + pos); pos += 512 * 512 * 3 * 3;
    block1_conv_b[gpu] = new Parameter({512}, param + pos); pos += 512;

    block1_to_rgb_modulate_w[gpu] = new Parameter({512, 512}, param + pos); pos += 512 * 512;
    block1_to_rgb_modulate_b[gpu] = new Parameter({512}, param + pos); pos += 512;
    block1_to_rgb_w[gpu] = new Parameter({3, 512, 1, 1}, param + pos); pos += 3 * 512 * 1 * 1;
    block1_to_rgb_b[gpu] = new Parameter({3}, param + pos); pos += 3;

    block2_conv_up_modulate_w[gpu] = new Parameter({512, 512}, param + pos); pos += 512 * 512;
    block2_conv_up_modulate_b[gpu] = new Parameter({512}, param + pos); pos += 512;
    block2_conv_up_w[gpu] = new Parameter({512, 512, 3, 3}, param + pos); pos += 512 * 512 * 3 * 3;
    block2_conv_up_b[gpu] = new Parameter({512}, param + pos); pos += 512;

    block2_conv_modulate_w[gpu] = new Parameter({512, 512}, param + pos); pos += 512 * 512;
    block2_conv_modulate_b[gpu] = new Parameter({512}, param + pos); pos += 512;
    block2_conv_w[gpu] = new Parameter({512, 512, 3, 3}, param + pos); pos += 512 * 512 * 3 * 3;
    block2_conv_b[gpu] = new Parameter({512}, param + pos); pos += 512;

    block2_to_rgb_modulate_w[gpu] = new Parameter({512, 512}, param + pos); pos += 512 * 512;
    block2_to_rgb_modulate_b[gpu] = new Parameter({512}, param + pos); pos += 512;
    block2_to_rgb_w[gpu] = new Parameter({3, 512, 1, 1}, param + pos); pos += 3 * 512 * 1 * 1;
    block2_to_rgb_b[gpu] = new Parameter({3}, param + pos); pos += 3;

    block3_conv_up_modulate_w[gpu] = new Parameter({512, 512}, param + pos); pos += 512 * 512;
    block3_conv_up_modulate_b[gpu] = new Parameter({512}, param + pos); pos += 512;
    block3_conv_up_w[gpu] = new Parameter({512, 512, 3, 3}, param + pos); pos += 512 * 512 * 3 * 3;
    block3_conv_up_b[gpu] = new Parameter({512}, param + pos); pos += 512;

    block3_conv_modulate_w[gpu] = new Parameter({512, 512}, param + pos); pos += 512 * 512;
    block3_conv_modulate_b[gpu] = new Parameter({512}, param + pos); pos += 512;
    block3_conv_w[gpu] = new Parameter({512, 512, 3, 3}, param + pos); pos += 512 * 512 * 3 * 3;
    block3_conv_b[gpu] = new Parameter({512}, param + pos); pos += 512;

    block3_to_rgb_modulate_w[gpu] = new Parameter({512, 512}, param + pos); pos += 512 * 512;
    block3_to_rgb_modulate_b[gpu] = new Parameter({512}, param + pos); pos += 512;
    block3_to_rgb_w[gpu] = new Parameter({3, 512, 1, 1}, param + pos); pos += 3 * 512 * 1 * 1;
    block3_to_rgb_b[gpu] = new Parameter({3}, param + pos); pos += 3;

    block4_conv_up_modulate_w[gpu] = new Parameter({512, 512}, param + pos); pos += 512 * 512;
    block4_conv_up_modulate_b[gpu] = new Parameter({512}, param + pos); pos += 512;
    block4_conv_up_w[gpu] = new Parameter({256, 512, 3, 3}, param + pos); pos += 256 * 512 * 3 * 3;
    block4_conv_up_b[gpu] = new Parameter({256}, param + pos); pos += 256;

    block4_conv_modulate_w[gpu] = new Parameter({256, 512}, param + pos); pos += 256 * 512;
    block4_conv_modulate_b[gpu] = new Parameter({256}, param + pos); pos += 256;
    block4_conv_w[gpu] = new Parameter({256, 256, 3, 3}, param + pos); pos += 256 * 256 * 3 * 3;
    block4_conv_b[gpu] = new Parameter({256}, param + pos); pos += 256;

    block4_to_rgb_modulate_w[gpu] = new Parameter({256, 512}, param + pos); pos += 256 * 512;
    block4_to_rgb_modulate_b[gpu] = new Parameter({256}, param + pos); pos += 256;
    block4_to_rgb_w[gpu] = new Parameter({3, 256, 1, 1}, param + pos); pos += 3 * 256;
    block4_to_rgb_b[gpu] = new Parameter({3}, param + pos); pos += 3;

    block5_conv_up_modulate_w[gpu] = new Parameter({256, 512}, param + pos); pos += 256 * 512;
    block5_conv_up_modulate_b[gpu] = new Parameter({256}, param + pos); pos += 256;
    block5_conv_up_w[gpu] = new Parameter({128, 256, 3, 3}, param + pos); pos += 128 * 256 * 3 * 3;
    block5_conv_up_b[gpu] = new Parameter({128}, param + pos); pos += 128;

    block5_conv_modulate_w[gpu] = new Parameter({128, 512}, param + pos); pos += 128 * 512;
    block5_conv_modulate_b[gpu] = new Parameter({128}, param + pos); pos += 128;
    block5_conv_w[gpu] = new Parameter({128, 128, 3, 3}, param + pos); pos += 128 * 128 * 3 * 3;
    block5_conv_b[gpu] = new Parameter({128}, param + pos); pos += 128;

    block5_to_rgb_modulate_w[gpu] = new Parameter({128, 512}, param + pos); pos += 128 * 512;
    block5_to_rgb_modulate_b[gpu] = new Parameter({128}, param + pos); pos += 128;
    block5_to_rgb_w[gpu] = new Parameter({3, 128, 1, 1}, param + pos); pos += 3 * 128;
    block5_to_rgb_b[gpu] = new Parameter({3}, param + pos); pos += 3;

    block6_conv_up_modulate_w[gpu] = new Parameter({128, 512}, param + pos); pos += 128 * 512;
    block6_conv_up_modulate_b[gpu] = new Parameter({128}, param + pos); pos += 128;
    block6_conv_up_w[gpu] = new Parameter({64, 128, 3, 3}, param + pos); pos += 64 * 128 * 3 * 3;
    block6_conv_up_b[gpu] = new Parameter({64}, param + pos); pos += 64;

    block6_conv_modulate_w[gpu] = new Parameter({64, 512}, param + pos); pos += 64 * 512;
    block6_conv_modulate_b[gpu] = new Parameter({64}, param + pos); pos += 64;
    block6_conv_w[gpu] = new Parameter({64, 64, 3, 3}, param + pos); pos += 64 * 64 * 3 * 3;
    block6_conv_b[gpu] = new Parameter({64}, param + pos); pos += 64;

    block6_to_rgb_modulate_w[gpu] = new Parameter({64, 512}, param + pos); pos += 64 * 512;
    block6_to_rgb_modulate_b[gpu] = new Parameter({64}, param + pos); pos += 64;
    block6_to_rgb_w[gpu] = new Parameter({3, 64, 1, 1}, param + pos); pos += 3 * 64;
    block6_to_rgb_b[gpu] = new Parameter({3}, param + pos); pos += 3;

    conv1_noise[gpu] = new Parameter({4, 4}, param + pos); pos += 4 * 4;
    block0_noise1[gpu] = new Parameter({8, 8}, param + pos); pos += 8 * 8;
    block0_noise2[gpu] = new Parameter({8, 8}, param + pos); pos += 8 * 8;
    block1_noise1[gpu] = new Parameter({16, 16}, param + pos); pos += 16 * 16;
    block1_noise2[gpu] = new Parameter({16, 16}, param + pos); pos += 16 * 16;
    block2_noise1[gpu] = new Parameter({32, 32}, param + pos); pos += 32 * 32;
    block2_noise2[gpu] = new Parameter({32, 32}, param + pos); pos += 32 * 32;
    block3_noise1[gpu] = new Parameter({64, 64}, param + pos); pos += 64 * 64;
    block3_noise2[gpu] = new Parameter({64, 64}, param + pos); pos += 64 * 64;
    block4_noise1[gpu] = new Parameter({128, 128}, param + pos); pos += 128 * 128;
    block4_noise2[gpu] = new Parameter({128, 128}, param + pos); pos += 128 * 128;
    block5_noise1[gpu] = new Parameter({256, 256}, param + pos); pos += 256 * 256;
    block5_noise2[gpu] = new Parameter({256, 256}, param + pos); pos += 256 * 256;
    block6_noise1[gpu] = new Parameter({512, 512}, param + pos); pos += 512 * 512;
    block6_noise2[gpu] = new Parameter({512, 512}, param + pos); pos += 512 * 512;

    if (pos != param_size) {
      fprintf(stderr, "Parameter size mismatched: %zu != %zu\n", 
              pos, param_size);
      exit(EXIT_FAILURE);
    }
  }
}

void free_parameters() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  for(int local_gpu = 0; local_gpu < GPUS_PER_RANK; local_gpu++) {
    cudaSetDevice(local_gpu);
    int gpu = local_gpu + rank * GPUS_PER_RANK;
    delete mlp0_w[gpu];
    delete mlp0_b[gpu];
    delete mlp1_w[gpu];
    delete mlp1_b[gpu];
    delete mlp2_w[gpu];
    delete mlp2_b[gpu];
    delete mlp3_w[gpu];
    delete mlp3_b[gpu];
    delete mlp4_w[gpu];
    delete mlp4_b[gpu];
    delete mlp5_w[gpu];
    delete mlp5_b[gpu];
    delete mlp6_w[gpu];
    delete mlp6_b[gpu];
    delete mlp7_w[gpu];
    delete mlp7_b[gpu];

    delete constant_input[gpu];
    delete kernel[gpu];
    delete conv1_modulate_w[gpu];
    delete conv1_modulate_b[gpu];
    delete conv1_w[gpu];
    delete conv1_b[gpu];
    delete to_rgb_modulate_w[gpu];
    delete to_rgb_modulate_b[gpu];
    delete to_rgb_w[gpu];
    delete to_rgb_b[gpu];

    delete block0_conv_up_modulate_w[gpu]; delete block0_conv_up_modulate_b[gpu]; delete block0_conv_up_w[gpu]; delete block0_conv_up_b[gpu];
    delete block0_conv_modulate_w[gpu]; delete block0_conv_modulate_b[gpu]; delete block0_conv_w[gpu]; delete block0_conv_b[gpu];
    delete block0_to_rgb_modulate_w[gpu]; delete block0_to_rgb_modulate_b[gpu]; delete block0_to_rgb_w[gpu]; delete block0_to_rgb_b[gpu];

    delete block1_conv_up_modulate_w[gpu]; delete block1_conv_up_modulate_b[gpu]; delete block1_conv_up_w[gpu]; delete block1_conv_up_b[gpu];
    delete block1_conv_modulate_w[gpu]; delete block1_conv_modulate_b[gpu]; delete block1_conv_w[gpu]; delete block1_conv_b[gpu];
    delete block1_to_rgb_modulate_w[gpu]; delete block1_to_rgb_modulate_b[gpu]; delete block1_to_rgb_w[gpu]; delete block1_to_rgb_b[gpu];

    delete block2_conv_up_modulate_w[gpu]; delete block2_conv_up_modulate_b[gpu]; delete block2_conv_up_w[gpu]; delete block2_conv_up_b[gpu];
    delete block2_conv_modulate_w[gpu]; delete block2_conv_modulate_b[gpu]; delete block2_conv_w[gpu]; delete block2_conv_b[gpu];
    delete block2_to_rgb_modulate_w[gpu]; delete block2_to_rgb_modulate_b[gpu]; delete block2_to_rgb_w[gpu]; delete block2_to_rgb_b[gpu];

    delete block3_conv_up_modulate_w[gpu]; delete block3_conv_up_modulate_b[gpu]; delete block3_conv_up_w[gpu]; delete block3_conv_up_b[gpu];
    delete block3_conv_modulate_w[gpu]; delete block3_conv_modulate_b[gpu]; delete block3_conv_w[gpu]; delete block3_conv_b[gpu];
    delete block3_to_rgb_modulate_w[gpu]; delete block3_to_rgb_modulate_b[gpu]; delete block3_to_rgb_w[gpu]; delete block3_to_rgb_b[gpu];

    delete block4_conv_up_modulate_w[gpu]; delete block4_conv_up_modulate_b[gpu]; delete block4_conv_up_w[gpu]; delete block4_conv_up_b[gpu];
    delete block4_conv_modulate_w[gpu]; delete block4_conv_modulate_b[gpu]; delete block4_conv_w[gpu]; delete block4_conv_b[gpu];
    delete block4_to_rgb_modulate_w[gpu]; delete block4_to_rgb_modulate_b[gpu]; delete block4_to_rgb_w[gpu]; delete block4_to_rgb_b[gpu];

    delete block5_conv_up_modulate_w[gpu]; delete block5_conv_up_modulate_b[gpu]; delete block5_conv_up_w[gpu]; delete block5_conv_up_b[gpu];
    delete block5_conv_modulate_w[gpu]; delete block5_conv_modulate_b[gpu]; delete block5_conv_w[gpu]; delete block5_conv_b[gpu];
    delete block5_to_rgb_modulate_w[gpu]; delete block5_to_rgb_modulate_b[gpu]; delete block5_to_rgb_w[gpu]; delete block5_to_rgb_b[gpu];

    delete block6_conv_up_modulate_w[gpu]; delete block6_conv_up_modulate_b[gpu]; delete block6_conv_up_w[gpu]; delete block6_conv_up_b[gpu];
    delete block6_conv_modulate_w[gpu]; delete block6_conv_modulate_b[gpu]; delete block6_conv_w[gpu]; delete block6_conv_b[gpu];
    delete block6_to_rgb_modulate_w[gpu]; delete block6_to_rgb_modulate_b[gpu]; delete block6_to_rgb_w[gpu]; delete block6_to_rgb_b[gpu];

    delete conv1_noise[gpu];
    delete block0_noise1[gpu]; delete block0_noise2[gpu];
    delete block1_noise1[gpu]; delete block1_noise2[gpu];
    delete block2_noise1[gpu]; delete block2_noise2[gpu];
    delete block3_noise1[gpu]; delete block3_noise2[gpu];
    delete block4_noise1[gpu]; delete block4_noise2[gpu];
    delete block5_noise1[gpu]; delete block5_noise2[gpu];
    delete block6_noise1[gpu]; delete block6_noise2[gpu];
  }
}

/* [Model Activations] 
 * _a: Activation buffer
 */
Activation *mlp0_a[NGPUS], *mlp1_a[NGPUS], *mlp2_a[NGPUS], *mlp3_a[NGPUS], *mlp4_a[NGPUS], *mlp5_a[NGPUS], *mlp6_a[NGPUS], *mlp7_a[NGPUS];
Activation *constant_input_a[NGPUS];

// conv1 activations
Activation *conv1_style_a[NGPUS], *conv1_weight_a[NGPUS], *conv1_demod_a[NGPUS];
Activation *conv1_output_a[NGPUS];

// ToRGB activations
Activation *to_rgb_style_a[NGPUS], *to_rgb_weight_a[NGPUS];
Activation *to_rgb_output_a[NGPUS];

// Activations for 7 blocks
Activation *block0_conv_up_style_a[NGPUS], *block0_conv_up_weight_a[NGPUS], *block0_conv_up_demod_a[NGPUS], *block0_conv_up_transpose_a[NGPUS];
Activation *block0_conv_up_conv_a[NGPUS], *block0_conv_up_upsample_a[NGPUS], *block0_conv_up_conv2_a[NGPUS], *block0_conv_up_output_a[NGPUS];
Activation *block0_conv_style_a[NGPUS], *block0_conv_weight_a[NGPUS], *block0_conv_demod_a[NGPUS];
Activation *block0_conv_output_a[NGPUS];
Activation *block0_to_rgb_style_a[NGPUS], *block0_to_rgb_weight_a[NGPUS];
Activation *block0_to_rgb_output_a[NGPUS];
Activation *block0_skip_a[NGPUS];
Activation *block0_to_rgb_skip_upsample_a[NGPUS], *block0_to_rgb_skip_conv_a[NGPUS];

Activation *block1_conv_up_style_a[NGPUS], *block1_conv_up_weight_a[NGPUS], *block1_conv_up_demod_a[NGPUS], *block1_conv_up_transpose_a[NGPUS];
Activation *block1_conv_up_conv_a[NGPUS], *block1_conv_up_upsample_a[NGPUS], *block1_conv_up_conv2_a[NGPUS], *block1_conv_up_output_a[NGPUS];
Activation *block1_conv_style_a[NGPUS], *block1_conv_weight_a[NGPUS], *block1_conv_demod_a[NGPUS];
Activation *block1_conv_output_a[NGPUS];
Activation *block1_to_rgb_style_a[NGPUS], *block1_to_rgb_weight_a[NGPUS];
Activation *block1_to_rgb_output_a[NGPUS];
Activation *block1_skip_a[NGPUS];
Activation *block1_to_rgb_skip_upsample_a[NGPUS], *block1_to_rgb_skip_conv_a[NGPUS];

Activation *block2_conv_up_style_a[NGPUS], *block2_conv_up_weight_a[NGPUS], *block2_conv_up_demod_a[NGPUS], *block2_conv_up_transpose_a[NGPUS];
Activation *block2_conv_up_conv_a[NGPUS], *block2_conv_up_upsample_a[NGPUS], *block2_conv_up_conv2_a[NGPUS], *block2_conv_up_output_a[NGPUS];
Activation *block2_conv_style_a[NGPUS], *block2_conv_weight_a[NGPUS], *block2_conv_demod_a[NGPUS];
Activation *block2_conv_output_a[NGPUS];
Activation *block2_to_rgb_style_a[NGPUS], *block2_to_rgb_weight_a[NGPUS];
Activation *block2_to_rgb_output_a[NGPUS];
Activation *block2_skip_a[NGPUS];
Activation *block2_to_rgb_skip_upsample_a[NGPUS], *block2_to_rgb_skip_conv_a[NGPUS];

Activation *block3_conv_up_style_a[NGPUS], *block3_conv_up_weight_a[NGPUS], *block3_conv_up_demod_a[NGPUS], *block3_conv_up_transpose_a[NGPUS];
Activation *block3_conv_up_conv_a[NGPUS], *block3_conv_up_upsample_a[NGPUS], *block3_conv_up_conv2_a[NGPUS], *block3_conv_up_output_a[NGPUS];
Activation *block3_conv_style_a[NGPUS], *block3_conv_weight_a[NGPUS], *block3_conv_demod_a[NGPUS];
Activation *block3_conv_output_a[NGPUS];
Activation *block3_to_rgb_style_a[NGPUS], *block3_to_rgb_weight_a[NGPUS];
Activation *block3_to_rgb_output_a[NGPUS];
Activation *block3_skip_a[NGPUS];
Activation *block3_to_rgb_skip_upsample_a[NGPUS], *block3_to_rgb_skip_conv_a[NGPUS];

Activation *block4_conv_up_style_a[NGPUS], *block4_conv_up_weight_a[NGPUS], *block4_conv_up_demod_a[NGPUS], *block4_conv_up_transpose_a[NGPUS];
Activation *block4_conv_up_conv_a[NGPUS], *block4_conv_up_upsample_a[NGPUS], *block4_conv_up_conv2_a[NGPUS], *block4_conv_up_output_a[NGPUS];
Activation *block4_conv_style_a[NGPUS], *block4_conv_weight_a[NGPUS], *block4_conv_demod_a[NGPUS];
Activation *block4_conv_output_a[NGPUS];
Activation *block4_to_rgb_style_a[NGPUS], *block4_to_rgb_weight_a[NGPUS];
Activation *block4_to_rgb_output_a[NGPUS];
Activation *block4_skip_a[NGPUS];
Activation *block4_to_rgb_skip_upsample_a[NGPUS], *block4_to_rgb_skip_conv_a[NGPUS];

Activation *block5_conv_up_style_a[NGPUS], *block5_conv_up_weight_a[NGPUS], *block5_conv_up_demod_a[NGPUS], *block5_conv_up_transpose_a[NGPUS];
Activation *block5_conv_up_conv_a[NGPUS], *block5_conv_up_upsample_a[NGPUS], *block5_conv_up_conv2_a[NGPUS], *block5_conv_up_output_a[NGPUS];
Activation *block5_conv_style_a[NGPUS], *block5_conv_weight_a[NGPUS], *block5_conv_demod_a[NGPUS];
Activation *block5_conv_output_a[NGPUS];
Activation *block5_to_rgb_style_a[NGPUS], *block5_to_rgb_weight_a[NGPUS];
Activation *block5_to_rgb_output_a[NGPUS];
Activation *block5_skip_a[NGPUS];
Activation *block5_to_rgb_skip_upsample_a[NGPUS], *block5_to_rgb_skip_conv_a[NGPUS];

Activation *block6_conv_up_style_a[NGPUS], *block6_conv_up_weight_a[NGPUS], *block6_conv_up_demod_a[NGPUS], *block6_conv_up_transpose_a[NGPUS];
Activation *block6_conv_up_conv_a[NGPUS], *block6_conv_up_upsample_a[NGPUS], *block6_conv_up_conv2_a[NGPUS], *block6_conv_up_output_a[NGPUS];
Activation *block6_conv_style_a[NGPUS], *block6_conv_weight_a[NGPUS], *block6_conv_demod_a[NGPUS];
Activation *block6_conv_output_a[NGPUS];
Activation *block6_to_rgb_style_a[NGPUS], *block6_to_rgb_weight_a[NGPUS];
Activation *block6_to_rgb_output_a[NGPUS];
Activation *block6_skip_a[NGPUS];
Activation *block6_to_rgb_skip_upsample_a[NGPUS], *block6_to_rgb_skip_conv_a[NGPUS];

void alloc_activations() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  for(int local_gpu = 0; local_gpu < GPUS_PER_RANK; local_gpu++) {
    int gpu = local_gpu + rank * GPUS_PER_RANK;
    cudaSetDevice(local_gpu);

    mlp0_a[gpu] = new Activation({1, 512});
    mlp1_a[gpu] = new Activation({1, 512});
    mlp2_a[gpu] = new Activation({1, 512});
    mlp3_a[gpu] = new Activation({1, 512});
    mlp4_a[gpu] = new Activation({1, 512});
    mlp5_a[gpu] = new Activation({1, 512});
    mlp6_a[gpu] = new Activation({1, 512});
    mlp7_a[gpu] = new Activation({1, 512});

    constant_input_a[gpu] = new Activation({1, 512, 4, 4});

    // ModulatedConv2d activations for conv1
    conv1_style_a[gpu] = new Activation({1, 512});
    conv1_weight_a[gpu] = new Activation({512, 512, 3, 3});
    conv1_demod_a[gpu] = new Activation({1, 512});
    conv1_output_a[gpu] = new Activation({1, 512, 4, 4});

    // ToRGB activations
    to_rgb_style_a[gpu] = new Activation({1, 512});
    to_rgb_weight_a[gpu] = new Activation({3, 512, 1, 1});
    to_rgb_output_a[gpu] = new Activation({1, 3, 4, 4});

    // Block 0: 8x8, 512 channels
    block0_conv_up_style_a[gpu] = new Activation({1, 512});
    block0_conv_up_weight_a[gpu] = new Activation({512, 512, 3, 3});
    block0_conv_up_demod_a[gpu] = new Activation({1, 512});
    block0_conv_up_transpose_a[gpu] = new Activation({512, 512, 3, 3});
    block0_conv_up_conv_a[gpu] = new Activation({1, 512, 9, 9});
    block0_conv_up_upsample_a[gpu] = new Activation({1, 512, 11, 11});
    block0_conv_up_conv2_a[gpu] = new Activation({1, 512, 8, 8});
    block0_conv_up_output_a[gpu] = new Activation({1, 512, 8, 8});

    block0_conv_style_a[gpu] = new Activation({1, 512});
    block0_conv_weight_a[gpu] = new Activation({512, 512, 3, 3});
    block0_conv_demod_a[gpu] = new Activation({1, 512});
    block0_conv_output_a[gpu] = new Activation({1, 512, 8, 8});

    block0_to_rgb_style_a[gpu] = new Activation({1, 512});
    block0_to_rgb_weight_a[gpu] = new Activation({3, 512, 1, 1});
    block0_to_rgb_output_a[gpu] = new Activation({1, 3, 8, 8});
    block0_skip_a[gpu] = new Activation({1, 3, 8, 8});
    block0_to_rgb_skip_upsample_a[gpu] = new Activation({1, 3, 11, 11});
    block0_to_rgb_skip_conv_a[gpu] = new Activation({1, 3, 8, 8});

    // Block 1: 16x16, 512 channels
    block1_conv_up_style_a[gpu] = new Activation({1, 512});
    block1_conv_up_weight_a[gpu] = new Activation({512, 512, 3, 3});
    block1_conv_up_demod_a[gpu] = new Activation({1, 512});
    block1_conv_up_transpose_a[gpu] = new Activation({512, 512, 3, 3});
    block1_conv_up_conv_a[gpu] = new Activation({1, 512, 17, 17});
    block1_conv_up_upsample_a[gpu] = new Activation({1, 512, 19, 19});
    block1_conv_up_conv2_a[gpu] = new Activation({1, 512, 16, 16});
    block1_conv_up_output_a[gpu] = new Activation({1, 512, 16, 16});

    block1_conv_style_a[gpu] = new Activation({1, 512});
    block1_conv_weight_a[gpu] = new Activation({512, 512, 3, 3});
    block1_conv_demod_a[gpu] = new Activation({1, 512});
    block1_conv_output_a[gpu] = new Activation({1, 512, 16, 16});

    block1_to_rgb_style_a[gpu] = new Activation({1, 512});
    block1_to_rgb_weight_a[gpu] = new Activation({3, 512, 1, 1});
    block1_to_rgb_output_a[gpu] = new Activation({1, 3, 16, 16});
    block1_skip_a[gpu] = new Activation({1, 3, 16, 16});
    block1_to_rgb_skip_upsample_a[gpu] = new Activation({1, 3, 19, 19});
    block1_to_rgb_skip_conv_a[gpu] = new Activation({1, 3, 16, 16});

    // Block 2: 32x32, 512 channels
    block2_conv_up_style_a[gpu] = new Activation({1, 512});
    block2_conv_up_weight_a[gpu] = new Activation({512, 512, 3, 3});
    block2_conv_up_demod_a[gpu] = new Activation({1, 512});
    block2_conv_up_transpose_a[gpu] = new Activation({512, 512, 3, 3});
    block2_conv_up_conv_a[gpu] = new Activation({1, 512, 33, 33});
    block2_conv_up_upsample_a[gpu] = new Activation({1, 512, 35, 35});
    block2_conv_up_conv2_a[gpu] = new Activation({1, 512, 32, 32});
    block2_conv_up_output_a[gpu] = new Activation({1, 512, 32, 32});
    
    block2_conv_style_a[gpu] = new Activation({1, 512});
    block2_conv_weight_a[gpu] = new Activation({512, 512, 3, 3});
    block2_conv_demod_a[gpu] = new Activation({1, 512});
    block2_conv_output_a[gpu] = new Activation({1, 512, 32, 32});
    
    block2_to_rgb_style_a[gpu] = new Activation({1, 512});
    block2_to_rgb_weight_a[gpu] = new Activation({3, 512, 1, 1});
    block2_to_rgb_output_a[gpu] = new Activation({1, 3, 32, 32});
    block2_skip_a[gpu] = new Activation({1, 3, 32, 32});
    block2_to_rgb_skip_upsample_a[gpu] = new Activation({1, 3, 35, 35});
    block2_to_rgb_skip_conv_a[gpu] = new Activation({1, 3, 32, 32});

    // Block 3: 64x64, 512 channels
    block3_conv_up_style_a[gpu] = new Activation({1, 512});
    block3_conv_up_weight_a[gpu] = new Activation({512, 512, 3, 3});
    block3_conv_up_demod_a[gpu] = new Activation({1, 512});
    block3_conv_up_transpose_a[gpu] = new Activation({512, 512, 3, 3});
    block3_conv_up_conv_a[gpu] = new Activation({1, 512, 65, 65});
    block3_conv_up_upsample_a[gpu] = new Activation({1, 512, 67, 67});
    block3_conv_up_conv2_a[gpu] = new Activation({1, 512, 64, 64});
    block3_conv_up_output_a[gpu] = new Activation({1, 512, 64, 64});

    block3_conv_style_a[gpu] = new Activation({1, 512});
    block3_conv_weight_a[gpu] = new Activation({512, 512, 3, 3});
    block3_conv_demod_a[gpu] = new Activation({1, 512});
    block3_conv_output_a[gpu] = new Activation({1, 512, 64, 64});

    block3_to_rgb_style_a[gpu] = new Activation({1, 512});
    block3_to_rgb_weight_a[gpu] = new Activation({3, 512, 1, 1});
    block3_to_rgb_output_a[gpu] = new Activation({1, 3, 64, 64});
    block3_skip_a[gpu] = new Activation({1, 3, 64, 64});
    block3_to_rgb_skip_upsample_a[gpu] = new Activation({1, 3, 67, 67});
    block3_to_rgb_skip_conv_a[gpu] = new Activation({1, 3, 64, 64});

    // Block 4: 128x128, 256 channels  
    block4_conv_up_style_a[gpu] = new Activation({1, 512});
    block4_conv_up_weight_a[gpu] = new Activation({256, 512, 3, 3});
    block4_conv_up_demod_a[gpu] = new Activation({1, 256});
    block4_conv_up_transpose_a[gpu] = new Activation({512, 256, 3, 3});
    block4_conv_up_conv_a[gpu] = new Activation({1, 256, 129, 129});
    block4_conv_up_upsample_a[gpu] = new Activation({1, 256, 131, 131});
    block4_conv_up_conv2_a[gpu] = new Activation({1, 256, 128, 128});
    block4_conv_up_output_a[gpu] = new Activation({1, 256, 128, 128});
    
    block4_conv_style_a[gpu] = new Activation({1, 256});
    block4_conv_weight_a[gpu] = new Activation({256, 256, 3, 3});
    block4_conv_demod_a[gpu] = new Activation({1, 256});
    block4_conv_output_a[gpu] = new Activation({1, 256, 128, 128});
    
    block4_to_rgb_style_a[gpu] = new Activation({1, 256});
    block4_to_rgb_weight_a[gpu] = new Activation({3, 256, 1, 1});
    block4_to_rgb_output_a[gpu] = new Activation({1, 3, 128, 128});
    block4_skip_a[gpu] = new Activation({1, 3, 128, 128});
    block4_to_rgb_skip_upsample_a[gpu] = new Activation({1, 3, 131, 131});
    block4_to_rgb_skip_conv_a[gpu] = new Activation({1, 3, 128, 128});

    // Block 5: 256x256, 128 channels
    block5_conv_up_style_a[gpu] = new Activation({1, 256});
    block5_conv_up_weight_a[gpu] = new Activation({128, 256, 3, 3});
    block5_conv_up_demod_a[gpu] = new Activation({1, 128});
    block5_conv_up_transpose_a[gpu] = new Activation({256, 128, 3, 3});
    block5_conv_up_conv_a[gpu] = new Activation({1, 128, 257, 257});
    block5_conv_up_upsample_a[gpu] = new Activation({1, 128, 259, 259});
    block5_conv_up_conv2_a[gpu] = new Activation({1, 128, 256, 256});
    block5_conv_up_output_a[gpu] = new Activation({1, 128, 256, 256});
    
    block5_conv_style_a[gpu] = new Activation({1, 128});
    block5_conv_weight_a[gpu] = new Activation({128, 128, 3, 3});
    block5_conv_demod_a[gpu] = new Activation({1, 128});
    block5_conv_output_a[gpu] = new Activation({1, 128, 256, 256});
    
    block5_to_rgb_style_a[gpu] = new Activation({1, 128});
    block5_to_rgb_weight_a[gpu] = new Activation({3, 128, 1, 1});
    block5_to_rgb_output_a[gpu] = new Activation({1, 3, 256, 256});
    block5_skip_a[gpu] = new Activation({1, 3, 256, 256});
    block5_to_rgb_skip_upsample_a[gpu] = new Activation({1, 3, 259, 259});
    block5_to_rgb_skip_conv_a[gpu] = new Activation({1, 3, 256, 256});

    // Block 6: 512x512, 64 channels
    block6_conv_up_style_a[gpu] = new Activation({1, 128});
    block6_conv_up_weight_a[gpu] = new Activation({64, 128, 3, 3});
    block6_conv_up_demod_a[gpu] = new Activation({1, 64});
    block6_conv_up_transpose_a[gpu] = new Activation({128, 64, 3, 3});
    block6_conv_up_conv_a[gpu] = new Activation({1, 64, 513, 513});
    block6_conv_up_upsample_a[gpu] = new Activation({1, 64, 515, 515});
    block6_conv_up_conv2_a[gpu] = new Activation({1, 64, 512, 512});
    block6_conv_up_output_a[gpu] = new Activation({1, 64, 512, 512});
    
    block6_conv_style_a[gpu] = new Activation({1, 64});
    block6_conv_weight_a[gpu] = new Activation({64, 64, 3, 3});
    block6_conv_demod_a[gpu] = new Activation({1, 64});
    block6_conv_output_a[gpu] = new Activation({1, 64, 512, 512});
    
    block6_to_rgb_style_a[gpu] = new Activation({1, 64});
    block6_to_rgb_weight_a[gpu] = new Activation({3, 64, 1, 1});
    block6_to_rgb_output_a[gpu] = new Activation({1, 3, 512, 512});
    block6_skip_a[gpu] = new Activation({1, 3, 512, 512});
    block6_to_rgb_skip_upsample_a[gpu] = new Activation({1, 3, 515, 515});
    block6_to_rgb_skip_conv_a[gpu] = new Activation({1, 3, 512, 512});
  }
}

void free_activations() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  for(int local_gpu = 0; local_gpu < GPUS_PER_RANK; local_gpu++) {
    cudaSetDevice(local_gpu);
    int gpu = local_gpu + rank * GPUS_PER_RANK;
    delete mlp0_a[gpu];
    delete mlp1_a[gpu];
    delete mlp2_a[gpu];
    delete mlp3_a[gpu];
    delete mlp4_a[gpu];
    delete mlp5_a[gpu];
    delete mlp6_a[gpu];
    delete mlp7_a[gpu];

    delete constant_input_a[gpu];

    delete conv1_style_a[gpu];
    delete conv1_weight_a[gpu];
    delete conv1_demod_a[gpu];
    delete conv1_output_a[gpu];

    delete to_rgb_style_a[gpu];
    delete to_rgb_weight_a[gpu];
    delete to_rgb_output_a[gpu];

    // Free block activations - All blocks
    delete block0_conv_up_style_a[gpu]; delete block0_conv_up_weight_a[gpu]; delete block0_conv_up_demod_a[gpu]; delete block0_conv_up_transpose_a[gpu];
    delete block0_conv_up_conv_a[gpu]; delete block0_conv_up_upsample_a[gpu]; delete block0_conv_up_conv2_a[gpu]; delete block0_conv_up_output_a[gpu];
    delete block0_conv_style_a[gpu]; delete block0_conv_weight_a[gpu]; delete block0_conv_demod_a[gpu];
    delete block0_conv_output_a[gpu];
    delete block0_to_rgb_style_a[gpu]; delete block0_to_rgb_weight_a[gpu];
    delete block0_to_rgb_output_a[gpu];
    delete block0_skip_a[gpu];
    delete block0_to_rgb_skip_upsample_a[gpu]; delete block0_to_rgb_skip_conv_a[gpu];

    delete block1_conv_up_style_a[gpu]; delete block1_conv_up_weight_a[gpu]; delete block1_conv_up_demod_a[gpu]; delete block1_conv_up_transpose_a[gpu];
    delete block1_conv_up_conv_a[gpu]; delete block1_conv_up_upsample_a[gpu]; delete block1_conv_up_conv2_a[gpu]; delete block1_conv_up_output_a[gpu];
    delete block1_conv_style_a[gpu]; delete block1_conv_weight_a[gpu]; delete block1_conv_demod_a[gpu];
    delete block1_conv_output_a[gpu];
    delete block1_to_rgb_style_a[gpu]; delete block1_to_rgb_weight_a[gpu];
    delete block1_to_rgb_output_a[gpu];
    delete block1_skip_a[gpu];
    delete block1_to_rgb_skip_upsample_a[gpu]; delete block1_to_rgb_skip_conv_a[gpu];

    delete block2_conv_up_style_a[gpu]; delete block2_conv_up_weight_a[gpu]; delete block2_conv_up_demod_a[gpu]; delete block2_conv_up_transpose_a[gpu];
    delete block2_conv_up_conv_a[gpu]; delete block2_conv_up_upsample_a[gpu]; delete block2_conv_up_conv2_a[gpu]; delete block2_conv_up_output_a[gpu];
    delete block2_conv_style_a[gpu]; delete block2_conv_weight_a[gpu]; delete block2_conv_demod_a[gpu]; 
    delete block2_conv_output_a[gpu];
    delete block2_to_rgb_style_a[gpu]; delete block2_to_rgb_weight_a[gpu];
    delete block2_to_rgb_output_a[gpu];
    delete block2_skip_a[gpu];
    delete block2_to_rgb_skip_upsample_a[gpu]; delete block2_to_rgb_skip_conv_a[gpu];

    delete block3_conv_up_style_a[gpu]; delete block3_conv_up_weight_a[gpu]; delete block3_conv_up_demod_a[gpu]; delete block3_conv_up_transpose_a[gpu];
    delete block3_conv_up_conv_a[gpu]; delete block3_conv_up_upsample_a[gpu]; delete block3_conv_up_conv2_a[gpu]; delete block3_conv_up_output_a[gpu];
    delete block3_conv_style_a[gpu]; delete block3_conv_weight_a[gpu]; delete block3_conv_demod_a[gpu];
    delete block3_conv_output_a[gpu];
    delete block3_to_rgb_style_a[gpu]; delete block3_to_rgb_weight_a[gpu];
    delete block3_to_rgb_output_a[gpu];
    delete block3_skip_a[gpu];
    delete block3_to_rgb_skip_upsample_a[gpu]; delete block3_to_rgb_skip_conv_a[gpu];

    delete block4_conv_up_style_a[gpu]; delete block4_conv_up_weight_a[gpu]; delete block4_conv_up_demod_a[gpu]; delete block4_conv_up_transpose_a[gpu];
    delete block4_conv_up_conv_a[gpu]; delete block4_conv_up_upsample_a[gpu]; delete block4_conv_up_conv2_a[gpu]; delete block4_conv_up_output_a[gpu];
    delete block4_conv_style_a[gpu]; delete block4_conv_weight_a[gpu]; delete block4_conv_demod_a[gpu]; 
    delete block4_conv_output_a[gpu];
    delete block4_to_rgb_style_a[gpu]; delete block4_to_rgb_weight_a[gpu];
    delete block4_to_rgb_output_a[gpu];
    delete block4_skip_a[gpu];
    delete block4_to_rgb_skip_upsample_a[gpu]; delete block4_to_rgb_skip_conv_a[gpu];

    delete block5_conv_up_style_a[gpu]; delete block5_conv_up_weight_a[gpu]; delete block5_conv_up_demod_a[gpu]; delete block5_conv_up_transpose_a[gpu];
    delete block5_conv_up_conv_a[gpu]; delete block5_conv_up_upsample_a[gpu]; delete block5_conv_up_conv2_a[gpu]; delete block5_conv_up_output_a[gpu];
    delete block5_conv_style_a[gpu]; delete block5_conv_weight_a[gpu]; delete block5_conv_demod_a[gpu]; 
    delete block5_conv_output_a[gpu];
    delete block5_to_rgb_style_a[gpu]; delete block5_to_rgb_weight_a[gpu];
    delete block5_to_rgb_output_a[gpu];
    delete block5_skip_a[gpu];
    delete block5_to_rgb_skip_upsample_a[gpu]; delete block5_to_rgb_skip_conv_a[gpu];

    delete block6_conv_up_style_a[gpu]; delete block6_conv_up_weight_a[gpu]; delete block6_conv_up_demod_a[gpu]; delete block6_conv_up_transpose_a[gpu];
    delete block6_conv_up_conv_a[gpu]; delete block6_conv_up_upsample_a[gpu]; delete block6_conv_up_conv2_a[gpu]; delete block6_conv_up_output_a[gpu];
    delete block6_conv_style_a[gpu]; delete block6_conv_weight_a[gpu]; delete block6_conv_demod_a[gpu]; 
    delete block6_conv_output_a[gpu];
    delete block6_to_rgb_style_a[gpu]; delete block6_to_rgb_weight_a[gpu];
    delete block6_to_rgb_output_a[gpu];
    delete block6_skip_a[gpu];
    delete block6_to_rgb_skip_upsample_a[gpu]; delete block6_to_rgb_skip_conv_a[gpu];
  }
}

#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)


/* [Model Computation] */
void generate(float *inputs, float *outputs, size_t n_samples) {  
  
  int mpi_rank, mpi_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  // Distribute samples across all GPUs in all ranks
  int total_gpus = mpi_size * GPUS_PER_RANK;
  int base_samples_per_gpu = n_samples / total_gpus;
  int extra_samples = n_samples % total_gpus;
  
  // Calculate samples for this rank
  int rank_start_gpu = mpi_rank * GPUS_PER_RANK;
  int samples_for_rank = 0;
  int rank_start_sample = 0;
  
  // Calculate total samples and starting position for this rank
  for (int gpu_id = 0; gpu_id < total_gpus; gpu_id++) {
    int samples_for_gpu = base_samples_per_gpu + (gpu_id < extra_samples ? 1 : 0);
    if (gpu_id < rank_start_gpu) {
      rank_start_sample += samples_for_gpu;
    } else if (gpu_id < rank_start_gpu + GPUS_PER_RANK) {
      samples_for_rank += samples_for_gpu;
    }
  }
  
  float* local_inputs = new float[samples_for_rank * 512];
  float* local_outputs = new float[samples_for_rank * 3 * 512 * 512];

  // Setup MPI scatter/gather counts
  std::vector<int> sendcounts(mpi_size), senddispls(mpi_size);
  std::vector<int> recvcounts(mpi_size), recvdispls(mpi_size);

  if(mpi_rank == 0) {
    for (int rank = 0; rank < mpi_size; rank++) {
      int rank_samples = 0;
      int rank_start = 0;
      
      for (int gpu_id = 0; gpu_id < total_gpus; gpu_id++) {
        int samples_for_gpu = base_samples_per_gpu + (gpu_id < extra_samples ? 1 : 0);
        if (gpu_id < rank * GPUS_PER_RANK) {
          rank_start += samples_for_gpu;
        } else if (gpu_id < (rank + 1) * GPUS_PER_RANK) {
          rank_samples += samples_for_gpu;
        }
      }
      
      sendcounts[rank] = rank_samples * 512;
      senddispls[rank] = rank_start * 512;
      recvcounts[rank] = rank_samples * 3 * 512 * 512;
      recvdispls[rank] = rank_start * 3 * 512 * 512;
    } 
  }

  // Scatter inputs to all ranks
  MPI_Scatterv(inputs, sendcounts.data(), senddispls.data(), MPI_FLOAT, 
               local_inputs, samples_for_rank * 512, MPI_FLOAT, 0, MPI_COMM_WORLD);

  // Process samples on GPUs within this rank
  std::vector<cudaStream_t> streams(GPUS_PER_RANK);
  std::vector<float*> gpu_inputs(GPUS_PER_RANK);
  std::vector<float*> gpu_outputs(GPUS_PER_RANK);
  std::vector<int> gpu_sample_counts(GPUS_PER_RANK);
  std::vector<int> gpu_sample_offsets(GPUS_PER_RANK);
  
  // Initialize streams and allocate GPU memory
  int sample_offset = 0;
  for (int local_gpu = 0; local_gpu < GPUS_PER_RANK; local_gpu++) {
    int global_gpu_id = rank_start_gpu + local_gpu;
    gpu_sample_counts[local_gpu] = base_samples_per_gpu + (global_gpu_id < extra_samples ? 1 : 0);
    gpu_sample_offsets[local_gpu] = sample_offset;
    sample_offset += gpu_sample_counts[local_gpu];
    
    CHECK_CUDA(cudaSetDevice(local_gpu));
    CHECK_CUDA(cudaStreamCreate(&streams[local_gpu]));
    
    // Allocate GPU memory
    CHECK_CUDA(cudaMalloc(&gpu_inputs[local_gpu], gpu_sample_counts[local_gpu] * 512 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gpu_outputs[local_gpu], gpu_sample_counts[local_gpu] * 3 * 512 * 512 * sizeof(float)));
    
    // Copy input data to GPU asynchronously
    CHECK_CUDA(cudaMemcpyAsync(gpu_inputs[local_gpu], 
                              local_inputs + gpu_sample_offsets[local_gpu] * 512,
                              gpu_sample_counts[local_gpu] * 512 * sizeof(float),
                              cudaMemcpyHostToDevice, streams[local_gpu]));
  }
  
  // FIX: Launch kernels on all GPUs in parallel using threads
  std::vector<std::thread> gpu_threads;
  
  // Launch processing on each GPU in parallel
  for (int local_gpu = 0; local_gpu < GPUS_PER_RANK; local_gpu++) {
    gpu_threads.emplace_back([&, local_gpu]() {
      int global_gpu_id = rank_start_gpu + local_gpu;
      CHECK_CUDA(cudaSetDevice(local_gpu));
      CHECK_CUDA(cudaStreamSynchronize(streams[local_gpu])); // Wait for input copy
      
      for (int n = 0; n < gpu_sample_counts[local_gpu]; n++) {
        /* Load a style from the inputs */
        Tensor *input = new Tensor({1, 512}, gpu_inputs[local_gpu] + n * 512);

        /* Get latent from style */
        PixelNorm(input);

        Linear(input, mlp0_w[global_gpu_id], mlp0_b[global_gpu_id], mlp0_a[global_gpu_id], 0.01f);
        LeakyReLU(mlp0_a[global_gpu_id]);

        Linear(mlp0_a[global_gpu_id], mlp1_w[global_gpu_id], mlp1_b[global_gpu_id], mlp1_a[global_gpu_id], 0.01f);
        LeakyReLU(mlp1_a[global_gpu_id]);

        Linear(mlp1_a[global_gpu_id], mlp2_w[global_gpu_id], mlp2_b[global_gpu_id], mlp2_a[global_gpu_id], 0.01f);
        LeakyReLU(mlp2_a[global_gpu_id]);

        Linear(mlp2_a[global_gpu_id], mlp3_w[global_gpu_id], mlp3_b[global_gpu_id], mlp3_a[global_gpu_id], 0.01f);
        LeakyReLU(mlp3_a[global_gpu_id]);

        Linear(mlp3_a[global_gpu_id], mlp4_w[global_gpu_id], mlp4_b[global_gpu_id], mlp4_a[global_gpu_id], 0.01f);
        LeakyReLU(mlp4_a[global_gpu_id]);

        Linear(mlp4_a[global_gpu_id], mlp5_w[global_gpu_id], mlp5_b[global_gpu_id], mlp5_a[global_gpu_id], 0.01f);
        LeakyReLU(mlp5_a[global_gpu_id]);

        Linear(mlp5_a[global_gpu_id], mlp6_w[global_gpu_id], mlp6_b[global_gpu_id], mlp6_a[global_gpu_id], 0.01f);
        LeakyReLU(mlp6_a[global_gpu_id]);

        Linear(mlp6_a[global_gpu_id], mlp7_w[global_gpu_id], mlp7_b[global_gpu_id], mlp7_a[global_gpu_id], 0.01f);
        LeakyReLU(mlp7_a[global_gpu_id]); // mlp7_a is now the latent vector

        // Constant input - use device-to-device copy
        CHECK_CUDA(cudaMemcpy(constant_input_a[global_gpu_id]->buf, 
                             constant_input[global_gpu_id]->buf, 
                             1 * 512 * 4 * 4 * sizeof(float),
                             cudaMemcpyDeviceToDevice));

        StyledConv(constant_input_a[global_gpu_id], mlp7_a[global_gpu_id], conv1_modulate_w[global_gpu_id], conv1_modulate_b[global_gpu_id], conv1_w[global_gpu_id], conv1_b[global_gpu_id], kernel[global_gpu_id], conv1_noise[global_gpu_id], conv1_output_a[global_gpu_id], 
                    conv1_style_a[global_gpu_id], conv1_weight_a[global_gpu_id], conv1_demod_a[global_gpu_id], nullptr, nullptr, nullptr, nullptr,
                    true, false, 1);

        ToRGB(conv1_output_a[global_gpu_id], nullptr, mlp7_a[global_gpu_id], to_rgb_modulate_w[global_gpu_id], to_rgb_modulate_b[global_gpu_id], to_rgb_w[global_gpu_id], to_rgb_b[global_gpu_id], kernel[global_gpu_id], to_rgb_output_a[global_gpu_id], 
                to_rgb_style_a[global_gpu_id], to_rgb_weight_a[global_gpu_id], nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                false, false, 0); 

        // Block 0
        StyledConv(conv1_output_a[global_gpu_id], mlp7_a[global_gpu_id], block0_conv_up_modulate_w[global_gpu_id], block0_conv_up_modulate_b[global_gpu_id], block0_conv_up_w[global_gpu_id], block0_conv_up_b[global_gpu_id], kernel[global_gpu_id], block0_noise1[global_gpu_id], block0_conv_up_output_a[global_gpu_id],
                    block0_conv_up_style_a[global_gpu_id], block0_conv_up_weight_a[global_gpu_id], block0_conv_up_demod_a[global_gpu_id], block0_conv_up_transpose_a[global_gpu_id], block0_conv_up_conv_a[global_gpu_id], block0_conv_up_upsample_a[global_gpu_id], block0_conv_up_conv2_a[global_gpu_id],
                    true, true, 0);

        StyledConv(block0_conv_up_output_a[global_gpu_id], mlp7_a[global_gpu_id], block0_conv_modulate_w[global_gpu_id], block0_conv_modulate_b[global_gpu_id], block0_conv_w[global_gpu_id], block0_conv_b[global_gpu_id], kernel[global_gpu_id], block0_noise2[global_gpu_id], block0_conv_output_a[global_gpu_id],
                    block0_conv_style_a[global_gpu_id], block0_conv_weight_a[global_gpu_id], block0_conv_demod_a[global_gpu_id], nullptr, nullptr, nullptr, nullptr,
                    true, false, 1);  

        ToRGB(block0_conv_output_a[global_gpu_id], to_rgb_output_a[global_gpu_id], mlp7_a[global_gpu_id], block0_to_rgb_modulate_w[global_gpu_id], block0_to_rgb_modulate_b[global_gpu_id], block0_to_rgb_w[global_gpu_id], block0_to_rgb_b[global_gpu_id], kernel[global_gpu_id], block0_to_rgb_output_a[global_gpu_id],
                block0_to_rgb_style_a[global_gpu_id], block0_to_rgb_weight_a[global_gpu_id], nullptr, nullptr, nullptr, nullptr, nullptr, block0_to_rgb_skip_upsample_a[global_gpu_id], block0_to_rgb_skip_conv_a[global_gpu_id], block0_skip_a[global_gpu_id],
                false, false, 0);

        // Block 1
        StyledConv(block0_conv_output_a[global_gpu_id], mlp7_a[global_gpu_id], block1_conv_up_modulate_w[global_gpu_id], block1_conv_up_modulate_b[global_gpu_id], block1_conv_up_w[global_gpu_id], block1_conv_up_b[global_gpu_id], kernel[global_gpu_id], block1_noise1[global_gpu_id], block1_conv_up_output_a[global_gpu_id],
                    block1_conv_up_style_a[global_gpu_id], block1_conv_up_weight_a[global_gpu_id], block1_conv_up_demod_a[global_gpu_id], block1_conv_up_transpose_a[global_gpu_id], block1_conv_up_conv_a[global_gpu_id], block1_conv_up_upsample_a[global_gpu_id], block1_conv_up_conv2_a[global_gpu_id],
                    true, true, 0);
        StyledConv(block1_conv_up_output_a[global_gpu_id], mlp7_a[global_gpu_id], block1_conv_modulate_w[global_gpu_id], block1_conv_modulate_b[global_gpu_id], block1_conv_w[global_gpu_id], block1_conv_b[global_gpu_id], kernel[global_gpu_id], block1_noise2[global_gpu_id], block1_conv_output_a[global_gpu_id],
                    block1_conv_style_a[global_gpu_id], block1_conv_weight_a[global_gpu_id], block1_conv_demod_a[global_gpu_id], nullptr, nullptr, nullptr, nullptr,
                    true, false, 1);  
        ToRGB(block1_conv_output_a[global_gpu_id], block0_to_rgb_output_a[global_gpu_id], mlp7_a[global_gpu_id], block1_to_rgb_modulate_w[global_gpu_id], block1_to_rgb_modulate_b[global_gpu_id], block1_to_rgb_w[global_gpu_id], block1_to_rgb_b[global_gpu_id], kernel[global_gpu_id], block1_to_rgb_output_a[global_gpu_id],
                block1_to_rgb_style_a[global_gpu_id], block1_to_rgb_weight_a[global_gpu_id], nullptr, nullptr, nullptr, nullptr, nullptr, block1_to_rgb_skip_upsample_a[global_gpu_id], block1_to_rgb_skip_conv_a[global_gpu_id], block1_skip_a[global_gpu_id],
                false, false, 0);

        // Block 2
        StyledConv(block1_conv_output_a[global_gpu_id], mlp7_a[global_gpu_id], block2_conv_up_modulate_w[global_gpu_id], block2_conv_up_modulate_b[global_gpu_id], block2_conv_up_w[global_gpu_id], block2_conv_up_b[global_gpu_id], kernel[global_gpu_id], block2_noise1[global_gpu_id], block2_conv_up_output_a[global_gpu_id],
                    block2_conv_up_style_a[global_gpu_id], block2_conv_up_weight_a[global_gpu_id], block2_conv_up_demod_a[global_gpu_id], block2_conv_up_transpose_a[global_gpu_id], block2_conv_up_conv_a[global_gpu_id], block2_conv_up_upsample_a[global_gpu_id], block2_conv_up_conv2_a[global_gpu_id],
                    true, true, 0);
        StyledConv(block2_conv_up_output_a[global_gpu_id], mlp7_a[global_gpu_id], block2_conv_modulate_w[global_gpu_id], block2_conv_modulate_b[global_gpu_id], block2_conv_w[global_gpu_id], block2_conv_b[global_gpu_id], kernel[global_gpu_id], block2_noise2[global_gpu_id], block2_conv_output_a[global_gpu_id],
                    block2_conv_style_a[global_gpu_id], block2_conv_weight_a[global_gpu_id], block2_conv_demod_a[global_gpu_id], nullptr, nullptr, nullptr, nullptr,
                    true, false, 1);  
        ToRGB(block2_conv_output_a[global_gpu_id], block1_to_rgb_output_a[global_gpu_id], mlp7_a[global_gpu_id], block2_to_rgb_modulate_w[global_gpu_id], block2_to_rgb_modulate_b[global_gpu_id], block2_to_rgb_w[global_gpu_id], block2_to_rgb_b[global_gpu_id], kernel[global_gpu_id], block2_to_rgb_output_a[global_gpu_id],
                block2_to_rgb_style_a[global_gpu_id], block2_to_rgb_weight_a[global_gpu_id], nullptr, nullptr, nullptr, nullptr, nullptr, block2_to_rgb_skip_upsample_a[global_gpu_id], block2_to_rgb_skip_conv_a[global_gpu_id], block2_skip_a[global_gpu_id],
                false, false, 0);

        // Block 3
        StyledConv(block2_conv_output_a[global_gpu_id], mlp7_a[global_gpu_id], block3_conv_up_modulate_w[global_gpu_id], block3_conv_up_modulate_b[global_gpu_id], block3_conv_up_w[global_gpu_id], block3_conv_up_b[global_gpu_id], kernel[global_gpu_id], block3_noise1[global_gpu_id], block3_conv_up_output_a[global_gpu_id],
                    block3_conv_up_style_a[global_gpu_id], block3_conv_up_weight_a[global_gpu_id], block3_conv_up_demod_a[global_gpu_id], block3_conv_up_transpose_a[global_gpu_id], block3_conv_up_conv_a[global_gpu_id], block3_conv_up_upsample_a[global_gpu_id], block3_conv_up_conv2_a[global_gpu_id],
                    true, true, 0);
        StyledConv(block3_conv_up_output_a[global_gpu_id], mlp7_a[global_gpu_id], block3_conv_modulate_w[global_gpu_id], block3_conv_modulate_b[global_gpu_id], block3_conv_w[global_gpu_id], block3_conv_b[global_gpu_id], kernel[global_gpu_id], block3_noise2[global_gpu_id], block3_conv_output_a[global_gpu_id],
                    block3_conv_style_a[global_gpu_id], block3_conv_weight_a[global_gpu_id], block3_conv_demod_a[global_gpu_id], nullptr, nullptr, nullptr, nullptr,
                    true, false, 1);  
        ToRGB(block3_conv_output_a[global_gpu_id], block2_to_rgb_output_a[global_gpu_id], mlp7_a[global_gpu_id], block3_to_rgb_modulate_w[global_gpu_id], block3_to_rgb_modulate_b[global_gpu_id], block3_to_rgb_w[global_gpu_id], block3_to_rgb_b[global_gpu_id], kernel[global_gpu_id], block3_to_rgb_output_a[global_gpu_id],
                block3_to_rgb_style_a[global_gpu_id], block3_to_rgb_weight_a[global_gpu_id], nullptr, nullptr, nullptr, nullptr, nullptr, block3_to_rgb_skip_upsample_a[global_gpu_id], block3_to_rgb_skip_conv_a[global_gpu_id], block3_skip_a[global_gpu_id],
                false, false, 0);

        // Block 4
        StyledConv(block3_conv_output_a[global_gpu_id], mlp7_a[global_gpu_id], block4_conv_up_modulate_w[global_gpu_id], block4_conv_up_modulate_b[global_gpu_id], block4_conv_up_w[global_gpu_id], block4_conv_up_b[global_gpu_id], kernel[global_gpu_id], block4_noise1[global_gpu_id], block4_conv_up_output_a[global_gpu_id],
                    block4_conv_up_style_a[global_gpu_id], block4_conv_up_weight_a[global_gpu_id], block4_conv_up_demod_a[global_gpu_id], block4_conv_up_transpose_a[global_gpu_id], block4_conv_up_conv_a[global_gpu_id], block4_conv_up_upsample_a[global_gpu_id], block4_conv_up_conv2_a[global_gpu_id],
                    true, true, 0);
        StyledConv(block4_conv_up_output_a[global_gpu_id], mlp7_a[global_gpu_id], block4_conv_modulate_w[global_gpu_id], block4_conv_modulate_b[global_gpu_id], block4_conv_w[global_gpu_id], block4_conv_b[global_gpu_id], kernel[global_gpu_id], block4_noise2[global_gpu_id], block4_conv_output_a[global_gpu_id],
                    block4_conv_style_a[global_gpu_id], block4_conv_weight_a[global_gpu_id], block4_conv_demod_a[global_gpu_id], nullptr, nullptr, nullptr, nullptr,
                    true, false, 1);  
        ToRGB(block4_conv_output_a[global_gpu_id], block3_to_rgb_output_a[global_gpu_id], mlp7_a[global_gpu_id], block4_to_rgb_modulate_w[global_gpu_id], block4_to_rgb_modulate_b[global_gpu_id], block4_to_rgb_w[global_gpu_id], block4_to_rgb_b[global_gpu_id], kernel[global_gpu_id], block4_to_rgb_output_a[global_gpu_id],
                block4_to_rgb_style_a[global_gpu_id], block4_to_rgb_weight_a[global_gpu_id], nullptr, nullptr, nullptr, nullptr, nullptr, block4_to_rgb_skip_upsample_a[global_gpu_id], block4_to_rgb_skip_conv_a[global_gpu_id], block4_skip_a[global_gpu_id],
                false, false, 0);

        // Block 5
        StyledConv(block4_conv_output_a[global_gpu_id], mlp7_a[global_gpu_id], block5_conv_up_modulate_w[global_gpu_id], block5_conv_up_modulate_b[global_gpu_id], block5_conv_up_w[global_gpu_id], block5_conv_up_b[global_gpu_id], kernel[global_gpu_id], block5_noise1[global_gpu_id], block5_conv_up_output_a[global_gpu_id],
                    block5_conv_up_style_a[global_gpu_id], block5_conv_up_weight_a[global_gpu_id], block5_conv_up_demod_a[global_gpu_id], block5_conv_up_transpose_a[global_gpu_id], block5_conv_up_conv_a[global_gpu_id], block5_conv_up_upsample_a[global_gpu_id], block5_conv_up_conv2_a[global_gpu_id],
                    true, true, 0);
        StyledConv(block5_conv_up_output_a[global_gpu_id], mlp7_a[global_gpu_id], block5_conv_modulate_w[global_gpu_id], block5_conv_modulate_b[global_gpu_id], block5_conv_w[global_gpu_id], block5_conv_b[global_gpu_id], kernel[global_gpu_id], block5_noise2[global_gpu_id], block5_conv_output_a[global_gpu_id],
                    block5_conv_style_a[global_gpu_id], block5_conv_weight_a[global_gpu_id], block5_conv_demod_a[global_gpu_id], nullptr, nullptr, nullptr, nullptr,
                    true, false, 1);  
        ToRGB(block5_conv_output_a[global_gpu_id], block4_to_rgb_output_a[global_gpu_id], mlp7_a[global_gpu_id], block5_to_rgb_modulate_w[global_gpu_id], block5_to_rgb_modulate_b[global_gpu_id], block5_to_rgb_w[global_gpu_id], block5_to_rgb_b[global_gpu_id], kernel[global_gpu_id], block5_to_rgb_output_a[global_gpu_id],
                block5_to_rgb_style_a[global_gpu_id], block5_to_rgb_weight_a[global_gpu_id], nullptr, nullptr, nullptr, nullptr, nullptr, block5_to_rgb_skip_upsample_a[global_gpu_id], block5_to_rgb_skip_conv_a[global_gpu_id], block5_skip_a[global_gpu_id],
                false, false, 0);

        // Block 6
        StyledConv(block5_conv_output_a[global_gpu_id], mlp7_a[global_gpu_id], block6_conv_up_modulate_w[global_gpu_id], block6_conv_up_modulate_b[global_gpu_id], block6_conv_up_w[global_gpu_id], block6_conv_up_b[global_gpu_id], kernel[global_gpu_id], block6_noise1[global_gpu_id], block6_conv_up_output_a[global_gpu_id],
                    block6_conv_up_style_a[global_gpu_id], block6_conv_up_weight_a[global_gpu_id], block6_conv_up_demod_a[global_gpu_id], block6_conv_up_transpose_a[global_gpu_id], block6_conv_up_conv_a[global_gpu_id], block6_conv_up_upsample_a[global_gpu_id], block6_conv_up_conv2_a[global_gpu_id],
                    true, true, 0);
        StyledConv(block6_conv_up_output_a[global_gpu_id], mlp7_a[global_gpu_id], block6_conv_modulate_w[global_gpu_id], block6_conv_modulate_b[global_gpu_id], block6_conv_w[global_gpu_id], block6_conv_b[global_gpu_id], kernel[global_gpu_id], block6_noise2[global_gpu_id], block6_conv_output_a[global_gpu_id],
                    block6_conv_style_a[global_gpu_id], block6_conv_weight_a[global_gpu_id], block6_conv_demod_a[global_gpu_id], nullptr, nullptr, nullptr, nullptr,
                    true, false, 1);  
        ToRGB(block6_conv_output_a[global_gpu_id], block5_to_rgb_output_a[global_gpu_id], mlp7_a[global_gpu_id], block6_to_rgb_modulate_w[global_gpu_id], block6_to_rgb_modulate_b[global_gpu_id], block6_to_rgb_w[global_gpu_id], block6_to_rgb_b[global_gpu_id], kernel[global_gpu_id], block6_to_rgb_output_a[global_gpu_id],
                block6_to_rgb_style_a[global_gpu_id], block6_to_rgb_weight_a[global_gpu_id], nullptr, nullptr, nullptr, nullptr, nullptr, block6_to_rgb_skip_upsample_a[global_gpu_id], block6_to_rgb_skip_conv_a[global_gpu_id], block6_skip_a[global_gpu_id],
                false, false, 0);

        /* Copy the result to GPU output buffer */
        CHECK_CUDA(cudaMemcpy(gpu_outputs[local_gpu] + n * 3 * 512 * 512, 
                             block6_to_rgb_output_a[global_gpu_id]->buf, 
                             3 * 512 * 512 * sizeof(float),
                             cudaMemcpyDeviceToDevice));
        
        delete input;
      }
      
      // Copy results back to host asynchronously
      CHECK_CUDA(cudaMemcpyAsync(local_outputs + gpu_sample_offsets[local_gpu] * 3 * 512 * 512,
                                gpu_outputs[local_gpu],
                                gpu_sample_counts[local_gpu] * 3 * 512 * 512 * sizeof(float),
                                cudaMemcpyDeviceToHost, streams[local_gpu]));
    });
  }
  
  // Wait for all GPU threads to complete
  for (auto& thread : gpu_threads) {
    thread.join();
  }
  
  // Synchronize all streams
  for (int local_gpu = 0; local_gpu < GPUS_PER_RANK; local_gpu++) {
    CHECK_CUDA(cudaSetDevice(local_gpu));
    CHECK_CUDA(cudaStreamSynchronize(streams[local_gpu]));
  }

  // Gather results from all ranks
  MPI_Gatherv(local_outputs, samples_for_rank * 3 * 512 * 512, MPI_FLOAT,
              outputs, recvcounts.data(), recvdispls.data(), MPI_FLOAT,
              0, MPI_COMM_WORLD);

  // Clean up GPU memory and streams
  for (int local_gpu = 0; local_gpu < GPUS_PER_RANK; local_gpu++) {
    CHECK_CUDA(cudaSetDevice(local_gpu));
    CHECK_CUDA(cudaFree(gpu_inputs[local_gpu]));
    CHECK_CUDA(cudaFree(gpu_outputs[local_gpu]));
    CHECK_CUDA(cudaStreamDestroy(streams[local_gpu]));
  }
  
  // Clean up host memory
  delete[] local_inputs;
  delete[] local_outputs;
}