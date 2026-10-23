// Copyright 2026 Arm Limited and/or its affiliates.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include "model_io.h"

// Input: already resized 224x224 RGB888, HWC. Output: signed INT8 NHWC storage.
// The generated lookup table includes ImageNet normalization and calibration
// quantization, so preprocessing performs no floating-point arithmetic.
inline void rps_preprocess_rgb888(const uint8_t* rgb, int8_t* output) {
  for (unsigned i = 0; i < kInputElements; ++i) {
    output[i] = kRGBInputLUT[(i % 3) * 256 + rgb[i]];
  }
}
