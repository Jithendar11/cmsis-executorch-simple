// Copyright 2026 Arm Limited and/or its affiliates.
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <executorch/runtime/core/data_loader.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/hierarchical_allocator.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/executor/memory_manager.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/runtime.h>
#include "model_pte.h"
#include "model_reference.h"
#ifdef MODEL_INPUT_RGB888
#include "rps_preprocess.h"
#endif

using namespace executorch::runtime;
namespace {
constexpr size_t kMethodBytes = kMethodPoolBytes;
constexpr size_t kTempBytes = kTempPoolBytes;
alignas(16) uint8_t method_pool[kMethodBytes];
alignas(16) uint8_t temp_pool[kTempBytes];

class BufferLoader final : public DataLoader {
 public:
  Result<FreeableBuffer> load(size_t offset, size_t bytes, const SegmentInfo&) const override {
    if (offset > model_pte_size || bytes > model_pte_size - offset) return Error::InvalidArgument;
    return FreeableBuffer(model_pte + offset, bytes, nullptr);
  }
  Result<size_t> size() const override { return model_pte_size; }
};
int run_tests() {
  runtime_init();
  printf("ExecuTorch Cortex-M85 CPU / CMSIS-NN: %lu byte model\n", model_pte_size);
  printf("Tensor interface: INT8 -> INT8\n");
  BufferLoader loader;
  auto program = Program::load(&loader);
  if (!program.ok()) { printf("Program::load error %u\n", unsigned(program.error())); return 1; }
  auto meta = program->method_meta("forward");
  if (!meta.ok()) return 1;
  MemoryAllocator method_allocator(kMethodBytes, method_pool);
  MemoryAllocator temp_allocator(kTempBytes, temp_pool);
  const size_t count = meta->num_memory_planned_buffers();
  Span<uint8_t> spans[8];
  if (count > 8) { printf("Too many planned buffers\n"); return 1; }
  size_t planned_bytes = 0;
  for (size_t i = 0; i < count; ++i) {
    auto required = meta->memory_planned_buffer_size(i);
    if (!required.ok()) return 1;
    const auto required_bytes = required.get();
    if (required_bytes < 0 || required_bytes > static_cast<int64_t>(kMethodBytes - planned_bytes)) {
      printf("Planned memory exceeds pool\n");
      return 1;
    }
    const size_t bytes = static_cast<size_t>(required_bytes);
    auto* buffer = static_cast<uint8_t*>(method_allocator.allocate(bytes, 16));
    if (!buffer) { printf("Planned memory exceeds pool\n"); return 1; }
    spans[i] = {buffer, bytes};
    planned_bytes += bytes;
  }
  HierarchicalAllocator planned({spans, count});
  MemoryManager memory(&method_allocator, &planned, &temp_allocator);
  auto method = program->load_method("forward", &memory);
  if (!method.ok()) { printf("load_method error %u\n", unsigned(method.error())); return 1; }
  printf("Planned tensor bytes: %u; reserved method/temp: %u/%u\n",
         unsigned(planned_bytes), unsigned(kMethodBytes), unsigned(kTempBytes));
  if (method->inputs_size() != 1 || method->outputs_size() != 1) return 1;
  for (unsigned test = 0; test < kTestCases; ++test) {
    auto input = method->get_input(0);
    if (!input.isTensor()) return 1;
    auto tensor = input.toTensor();
    if (tensor.scalar_type() != executorch::aten::ScalarType::Char || tensor.numel() != kInputElements) return 1;
#ifdef MODEL_INPUT_RGB888
    rps_preprocess_rgb888(kRGBInputs + test * kInputElements, tensor.mutable_data_ptr<int8_t>());
#else
    memcpy(tensor.mutable_data_ptr<int8_t>(), kInputs + test * kInputElements, kInputElements * sizeof(int8_t));
#endif
    auto err = method->execute();
    if (err != Error::Ok) { printf("execute error %u\n", unsigned(err)); return 1; }
    auto output = method->get_output(0);
    if (!output.isTensor()) return 1;
    auto result = output.toTensor();
    if (result.scalar_type() != executorch::aten::ScalarType::Char || result.numel() != kOutputElements) return 1;
    const int8_t* values = result.const_data_ptr<int8_t>();
    int max_error = 0;
    unsigned predicted = 0;
    for (unsigned i = 0; i < kOutputElements; ++i) {
      const int difference = int(values[i]) - int(kExpected[test * kOutputElements + i]);
      const int error = difference < 0 ? -difference : difference;
      if (error > kOutputTolerance) {
        printf("Case %u output %u mismatch: %d expected %d\n", test, i, int(values[i]), int(kExpected[test * kOutputElements + i]));
        return 1;
      }
      if (error > max_error) max_error = error;
      if (values[i] > values[predicted]) predicted = i;
    }
    printf("Case %u: PASS; max error %d INT8 codes (tolerance %d)\n", test, max_error, kOutputTolerance);
    printf("Prediction: %s (index %u)\n", kClassNames[predicted], predicted);
  }
  return 0;
}
} // namespace

extern "C" int app_main(void) {
  const int result = run_tests();
  printf("Test_result: %s\n", result == 0 ? "PASS" : "FAIL");
  fflush(stdout);
  putchar(4);
  fflush(stdout);
  return result;
}
