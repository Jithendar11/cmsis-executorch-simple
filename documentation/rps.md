# SqueezeNet rock-paper-scissors on the Cortex-M85 CPU

The `SSE-320-RPS` target exports the supplied SqueezeNet 1.1 checkpoint for
CMSIS-NN CPU inference with signed INT8 inputs and outputs. The classifier
has four outputs in the order `PAPER`, `ROCK`, `SCISSORS`, `UNKNOWN`, matching
the supplied application's `algorithm/ML/model_config.h`.

## Local configuration

`model/rps_config.json` contains the local checkpoint and calibration-image
paths supplied for this workspace. It is ignored by Git. For a new checkout,
copy `model/rps_config.example.json` to `model/rps_config.json` and update its
paths. Relative paths resolve from the configuration file's directory.
Install the optional `requirements-rps.txt` dependencies if torchvision or
Pillow is missing; both are already installed in this workspace.

The checkpoint is loaded as a state dictionary with `weights_only=True` and
must match torchvision's SqueezeNet 1.1 with a four-class head exactly.
No pretrained weights are downloaded.

## Export, build and run

From the repository root, model-only export is:

```powershell
.\.venv\Scripts\python.exe model\export_cpu.py --model rps --cpu cortex-m85
```

For host validation without writing outputs, append `--check-only`.
The `.pte` is written to `model/rps/model.pte`; generated headers are in
`ai_layer/rps_model/`, and the lowered graph and quantization report are in
`model/rps/`. The report records the checkpoint SHA-256 and source images.

In CMSIS Solution, select **SSE-320-RPS / Debug**, then **Build** and **Run**.
The build exports the model automatically and selects its operators from the
installed ExecuTorch 1.4.1 pack. If the operator list changes, build a second
time as requested. The runner compares all four INT8 outputs against host
references, prints the predicted class, and reports `Test_result: PASS` only
when every image passes the numerical check. These checks compare inference
implementations; they do not measure dataset accuracy.

The checkpoint and images are external assets. If their contents change
without changing the configuration paths, use a clean rebuild to re-export.

## Image interface

The logical tensor shape is **[1, 3, 224, 224]**, with channels-last storage
(**NHWC**, dimension order [0,2,3,1]). Its physical bytes are RGBRGB…, matching
a **[1, 224, 224, 3]** RGB image buffer. The output shape is **[1, 4] INT8**.
Do not transpose the input into contiguous NCHW storage.

Image loading converts to RGB, resizes directly to 224x224 using bilinear
interpolation, divides pixels by 255, and applies the supplied ImageNet mean
`[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`.
The result is quantized with the calibrated input scale and zero point.

For firmware, `src/rps_preprocess.h` provides `rps_preprocess_rgb888` for an
already resized 224x224 RGB888 image. It uses a generated 3x256 lookup table
in `model_io.h` to perform normalization and quantization without floating-point
arithmetic. The host export checks its bytes exactly against the normalized
and quantized image tensors. The RPS runner embeds the supplied RGB images and
calls this helper before inference. The helper does not resize camera frames.

Argmax operates directly on the four INT8 logits; softmax is unnecessary for
class selection. Confidence probabilities, if needed, require an additional
postprocessing step and must not be inferred from the raw INT8 scores.

For fixed 224x224 inputs, the three max-pools receive odd spatial sizes
111, 55 and 27, so floor and ceil modes give the same output. The loader
checks exact FP32 output equality on the supplied images before switching to
the floor-mode configuration supported by CMSIS-NN. Changing resolution needs
fresh validation.

## Validation and memory

Host checks produced an **831,552-byte** serialized program with INT8 I/O
and **1,175,744 planned tensor bytes**. All three supplied examples classify
as PAPER, ROCK and SCISSORS in both FP32 and INT8. The integer boundary
conversion preserves the original quantized reference outputs exactly.

`model/runtime_layout.py` inserts default-contiguous copies around portable
concatenation and before `view_copy`, then restores channels-last storage for
CMSIS-NN consumers. The pack's portable cat copies blocks in logical-axis
order, while its view kernel explicitly requires default contiguous storage.
Export validates all nine serialized portable cat/view calls against these
requirements. The previous export passed a channels-last tensor directly to
the final view and failed on target with `InvalidArgument` (18). The correction
adds layout-copy overhead; its new memory plan is reflected in the generated
method pool. A fresh FVP run is required to validate execution and numerics.

Only three calibration examples were supplied, with no UNKNOWN example.
They are sufficient for an initial deployment smoke test. A representative
calibration set and independent validation set, covering all four classes,
are required before making accuracy claims.

The generated method pool reserves planned tensor bytes plus 64 KiB for
method metadata; temporary memory reserves a further 16 KiB. The FVP board
uses its existing DDR-backed memory map. These figures exclude stack, heap,
other globals and reference images. The three embedded RGB888 images consume
451,584 bytes of read-only data. Physical-board memory placement and measured
latency still need validation.

The preceding RPS version compiled and loaded on the FVP but returned error
18 during execution. Build and numerical verification of the corrected layout
version remain pending because the debug launch request was declined.
