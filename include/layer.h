#pragma once

#include "tensor.h"


/* Layers (Operations) */
void PixelNorm(Tensor *inout);

void UpsamplePad(Tensor *input, Tensor *output, int up, int pad0, int pad1);

void Conv2d(Tensor *input, Tensor *weight, Tensor *bias, Tensor *output,
            int stride, int pad, int dilation, bool has_bias);

void ConvTranspose2d(Tensor *input, Tensor *weight, Tensor *output, 
                     int stride, int pad);

void transpose(Tensor *input, Tensor *output);

void Linear(Tensor *in, Tensor *w, Tensor *b, Tensor *out, float lr_mul);

void LeakyReLU(Tensor *inout);

void upfir2d(Tensor *input, Tensor *kernel, Tensor *output,
               Tensor *upsample_a, Tensor *conv_a,
               int up, int pad0, int pad1);

void ModulatedConv2d(Tensor *input, Tensor *style, Tensor *modulate_weight, Tensor *modulate_bias, Tensor *conv_weight, Tensor *kernel, Tensor *output,
                     Tensor *style_a, Tensor *weight_a, Tensor *demod_a, Tensor *transpose_a, Tensor *conv_a, Tensor *upsample_a, Tensor *conv2_a,
                     bool demodulate, bool upsample, int padding, int up);

void addNoise(Tensor *inout, Tensor *noise);

void addBias(Tensor *inout, Tensor *bias);

void elemAdd(Tensor *inout, Tensor *addend);

void StyledConv(Tensor *input, Tensor *style, Tensor *modulate_weight, Tensor *modulate_bias, Tensor *conv_weight, Tensor *conv_bias, Tensor *kernel, Tensor *noise, Tensor *output,
                Tensor *style_a, Tensor *weight_a, Tensor *demod_a, Tensor *transpose_a, Tensor *conv_a, Tensor *upsample_a, Tensor *conv2_a,
                bool demodulate, bool upsample, int padding);

void ToRGB(Tensor *input, Tensor *skip, Tensor *style, Tensor *modulate_weight, Tensor *modulate_bias, Tensor *conv_weight, Tensor *conv_bias, Tensor *kernel, Tensor *output,
           Tensor *style_a, Tensor *weight_a, Tensor *demod_a, Tensor *transpose_a, Tensor *conv_a, Tensor *upsample_a, Tensor *conv2_a, Tensor *skip_upsample_a, Tensor *skip_conv_a, Tensor *skip_a,
           bool demodulate, bool upsample, int padding);

void printTimeMap();