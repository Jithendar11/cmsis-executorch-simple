# Compile PyTorch models for the Cortex-M85 CPU

This guide covers model compilation with the ExecuTorch **Cortex-M CPU
backend**, using the versions and export flow verified in this repository.
The result is a `.pte` program with **signed INT8 input and output tensors**.

The workflow applies to models whose graphs can be exported and whose
operators, shapes, layouts and quantization schemes are supported by the
Cortex-M backend and the selected runtime. A new model needs those checks;
changing a checkpoint filename alone is insufficient.

## 1. Understand the compilation pipeline

```text
PyTorch model definition + trained weights
                   |
         Evaluate in FP32 on the host
                   |
          torch.export.export
                   |
      CortexMQuantizer + PT2E prepare
                   |
       Calibrate with representative data
                   |
             PT2E convert
                   |
    Expose calibrated INT8 input/output tensors
                   |
       Re-export with INT8 example inputs
                   |
              Edge dialect
                   |
   CortexMPassManager: Cortex-M85 / MVE
                   |
       Adapt portable operator layouts
                   |
     Memory planning and serialization
                   |
                model.pte
```

The Cortex-M passes replace eligible operations with CMSIS-NN-backed
`cortex_m::` operators. Supported portable operations can remain in the graph.
There must be no Ethos-U delegate in a CPU-only export.

The `.pte` contains the execution plan, tensor metadata and model data. It is
consumed by the matching ExecuTorch runtime; it is not a standalone executable
or an ELF file of Cortex-M85 machine instructions.

The CPU compilation path uses `CortexMQuantizer` and `CortexMPassManager`.
It does not invoke Vela or use `EthosUQuantizer`, `EthosUCompileSpec`, or
`EthosUPartitioner`.

## 2. Set up the verified Python environment

The following versions were used for the successful RPS compilation:

| Dependency  | Version | Purpose                                                       |
| ----------- | ------- | ------------------------------------------------------------- |
| Python      | 3.13    | Host interpreter used for this verification                   |
| PyTorch     | 2.13.0  | Export and model execution                                    |
| ExecuTorch  | 1.4.1   | Cortex-M lowering and `.pte` generation                     |
| TorchAO     | 0.18.0  | PT2E quantization                                             |
| torchvision | 0.28.0  | SqueezeNet model definition; optional for other architectures |
| Pillow      | 12.3.0  | Image loading; optional for non-image inputs                  |

Use the exporter with the matching runtime version. This repository selects
`PyTorch::ExecuTorch@1.4.1`. When changing versions, validate the Python
packages and runtime together, then regenerate the model.

From PowerShell, enter the repository root:

```powershell
Set-Location 'C:\Users\Jithendar\Downloads\arm_sdk\et_1.4\CMSIS-Executorch'
```

For a fresh environment, run the repository setup script:

```powershell
.\setup_venv.bat
```

The repository setup installs its shared CPU/NPU export dependencies. Some
Arm Python modules are shared, so retain those dependencies even though the
CPU export does not invoke the NPU compiler.

For SqueezeNet and image loading, install the optional dependencies if absent:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-rps.txt
```

Verify package versions:

```powershell
.\.venv\Scripts\python.exe -c "from importlib.metadata import version; print({p: version(p) for p in ['torch', 'executorch', 'torchao']})"
```

Verify that the Cortex-M backend imports:

```powershell
.\.venv\Scripts\python.exe -c "from executorch.backends.cortex_m.quantizer.quantizer import CortexMQuantizer; from executorch.backends.cortex_m.target_config import CortexMTargetConfig; t = CortexMTargetConfig.from_target_string('cortex-m85'); print(t.target_string, t.backend.name)"
```

For the verified target, the last command prints `cortex-m85 MVE`.
These commands use the virtual environment directly; activation is optional.

**API compatibility:** this guide follows ExecuTorch **1.4.1**. Its
`CortexMPassManager(exported_program, target_config=...).transform()` API
differs from the newer API shown in the PyTorch Cortex-M blog. Do not mix
code examples from different versions.

## 3. Collect the model information before exporting

For each new model, establish:

1. The exact Python architecture and constructor arguments.
2. The trained checkpoint and its state-dictionary format.
3. Input count, dtype, shape, batch size and memory layout.
4. Training-time preprocessing: channel order, resizing, normalization,
   feature extraction or other transformations.
5. Output count, shapes and meaning, including class-index order.
6. Representative calibration samples and a separate validation set.
7. The target CPU/ISA and available model-data and tensor-memory budgets.

A state dictionary does not normally contain the model's `forward()` method.
Construct the matching architecture, then load its weights strictly:

```python
model = MyModel(num_classes=4)  # Use the arguments required by your architecture.
state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
model.load_state_dict(state, strict=True)
model.eval()
```

Import your model class and use its actual constructor arguments.
If the checkpoint contains a dictionary such as `{"state_dict": ...}`, pass
that nested state dictionary to `load_state_dict`.

Run representative samples through the FP32 model before quantization. Keep
their outputs as the baseline for accuracy and numerical comparisons.

## 4. Prepare shapes and layouts correctly

For the four-dimensional image CNNs verified here, PyTorch uses logical NCHW
shapes while CMSIS-NN convolution uses channels-last physical storage:

```python
model = model.eval().to(memory_format=torch.channels_last)
x = x.contiguous(memory_format=torch.channels_last)
example_inputs = (x,)
```

For an RGB image classifier:

| Property                               | Example                         |
| -------------------------------------- | ------------------------------- |
| Logical PyTorch shape                  | `[1, 3, 224, 224]`            |
| Physical dimension order               | `[0, 2, 3, 1]`                |
| Physical byte layout                   | NHWC, or RGBRGB… for one image |
| External input dtype after compilation | `torch.int8`                  |

Changing the physical layout does not change the logical meaning of axis 1:
it remains the channel axis. Do not pass a logical `[1,224,224,3]` tensor to a
model defined for `[1,3,224,224]` without adapting its definition.

Use the same shape and preprocessing for export, calibration and validation.
Start with fixed shapes. Other ranks, multiple inputs or dynamic shapes need
their own export, layout and runtime-support checks.

## 5. Calibrate and convert to INT8

First export the trained FP32 model:

```python
exported = torch.export.export(model, example_inputs, strict=True)
graph_module = exported.module()
```

Then prepare, calibrate and convert it:

```python
from executorch.backends.cortex_m.quantizer.quantizer import CortexMQuantizer
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e

prepared = prepare_pt2e(graph_module, CortexMQuantizer())

with torch.no_grad():
    for sample_inputs in calibration_inputs:
        prepared(*sample_inputs)

quantized = convert_pt2e(prepared)
```

Here each `sample_inputs` is a tuple matching the model's input signature.
Calibration samples are preprocessed floating-point tensors at this stage.

Calibration estimates activation ranges and quantization parameters. It does
not train the network. Use real samples covering deployment conditions and
all relevant classes, including background/unknown cases when applicable.
Random tensors are suitable for exercising an export pipeline, not for
establishing a trained model's accuracy.

Compare the converted model against FP32 on held-out samples. Investigate
accuracy regressions before proceeding to CPU lowering.

## 6. Make the external input and output INT8

PT2E conversion can leave a float input followed by quantization, and an INT8
result followed by output dequantization. Internal INT8 computation therefore
does not automatically imply INT8 model inputs and outputs.

For the single-input, single-output graphs supported by this example,
`make_int8_io()` in `model/export_cpu.py`:

1. Finds the calibrated per-tensor input quantizer.
2. Saves its scale, zero point, range and dtype.
3. Replaces that quantizer's output with the external INT8 input.
4. Finds the final output dequantizer and saves its parameters.
5. Exposes the INT8 tensor before output dequantization.
6. Recompiles the graph, then re-exports using INT8 example inputs.

Usage:

```python
import sys
from pathlib import Path

# For a standalone exporter run from this repository's root directory:
sys.path.insert(0, str(Path("model").resolve()))
from export_cpu import make_int8_io

(input_scale, input_zp), (output_scale, output_zp) = make_int8_io(quantized)

int8_x = torch.ops.quantized_decomposed.quantize_per_tensor.default(
    x, input_scale, input_zp, -128, 127, torch.int8
).contiguous(memory_format=torch.channels_last)

int8_exported = torch.export.export(quantized, (int8_x,), strict=True)
```

The input conversion is:

```text
q = clamp(round(real_input / input_scale) + input_zero_point, -128, 127)
```

`real_input` means the model's preprocessed input, not necessarily raw pixels.
Use the quantization operation above for reference generation so its rounding
matches the exported model.

To interpret output codes in real units:

```text
real_output = (output_code - output_zero_point) * output_scale
```

Do not remove internal Q/DQ operations indiscriminately. They carry
quantization information used during lowering. The helper removes only the
verified boundary pair and rejects signatures it does not recognize.

For multiple inputs/outputs, structured outputs, or different quantization
schemes, implement boundary conversion for each tensor and retain its
individual parameters. The current helper deliberately supports one
per-tensor-quantized INT8 input and one output.

INT8 refers to inputs, outputs, weights and supported activation tensors.
Biases, accumulators, shape values and requantization parameters can use
wider integer types. Quantization metadata can contain floating-point scales.

## 7. Lower to the Cortex-M85 CPU

The verified 1.4.1 sequence is:

```python
from executorch.exir import EdgeCompileConfig, to_edge
from executorch.backends.cortex_m.target_config import CortexMTargetConfig
from executorch.backends.cortex_m.passes.cortex_m_pass_manager import CortexMPassManager

edge = to_edge(
    int8_exported,
    compile_config=EdgeCompileConfig(
        _check_ir_validity=False,
        preserve_ops=[torch.ops.aten.linear.default],
        _core_aten_ops_exception_list=[torch.ops.aten.max_pool2d.default],
    ),
)

target = CortexMTargetConfig.from_target_string("cortex-m85")
edge._edge_programs["forward"] = CortexMPassManager(
    edge.exported_program(), target_config=target
).transform()
```

The compile configuration above follows this repository's verified 1.4.1
flow; `_check_ir_validity=False` is not evidence that every graph is valid.
Follow it with explicit operator, dtype, layout and numerical validation.
The `_edge_programs` assignment is version-specific and must be revisited
when upgrading ExecuTorch.

Inspect the lowered graph. Eligible convolutions, linear layers and pooling
operations should appear as appropriate `cortex_m::` operations. A model
without a linear layer should not be required to contain a linear operator.
Check the operations your architecture actually contains.

## 8. Adapt portable operator layouts

An exported graph may mix CMSIS-NN kernels with portable kernels whose layout
requirements differ. Successful export does not by itself prove runtime
layout compatibility.

For the 1.4.1 pack used here:

- CMSIS-NN image convolution and pooling use channels-last tensors.
- Portable `cat` copies blocks in logical-axis order; the verified export
  provides default-contiguous input/output storage around it.
- Portable `view_copy` requires default-contiguous input storage.

Apply the repository pass after Cortex-M lowering:

```python
from runtime_layout import PortableLayoutPass

edge = edge.transform([PortableLayoutPass()])
```

This pass inserts layout copies around concatenation and before reshaping,
and restores channels-last storage for four-dimensional concatenation outputs
that feed the CNN's CMSIS-NN operations. It preserves logical values while
adding copy work and potentially increasing planned memory.

This is a checked adaptation for the image-CNN path in this repository.
For other model families or consumers with different layout requirements,
review and adapt the pass rather than applying it blindly.

For SqueezeNet at fixed 224x224 input, the loader also changes max-pooling
from ceil mode to floor mode after checking exact FP32 output equality on
the supplied images. Pool inputs have odd spatial sizes 111, 55 and 27, so
both modes have the same geometry in this case. Do not apply that change
universally to another architecture or resolution.

## 9. Serialize the program and validate its metadata

```python
from pathlib import Path
from executorch.exir._serialize._program import deserialize_pte_binary
from executorch.exir.schema import ScalarType
from runtime_layout import validate_portable_layouts

program = edge.to_executorch()
binary = program.buffer
plan = deserialize_pte_binary(binary).program.execution_plan[0]

for index in plan.inputs + plan.outputs:
    assert plan.values[index].val.scalar_type == ScalarType.CHAR

assert plan.values[plan.inputs[0]].val.dim_order == [0, 2, 3, 1]
validate_portable_layouts(plan)
assert not plan.delegates

output_dir = Path("model/my_model_m85")
output_dir.mkdir(parents=True, exist_ok=True)
(output_dir / "model.pte").write_bytes(binary)
(output_dir / "graph.txt").write_text(
    str(edge.exported_program().graph_module.graph)
)

print("Model bytes:", len(binary))
print("Planned tensor bytes:", sum(plan.non_const_buffer_sizes[1:]))
print("Operators:", sorted({op.name for op in plan.operators}))
```

`ScalarType.CHAR` is the serialized signed INT8 type. Inspect the serialized
operator table, including namespaces such as `dim_order_ops`. Searching only
for convolution operators or selected namespaces can miss required kernels.

Confirm that every operator is available in the selected runtime. This
repository's `scripts/gen_components.py` reads the complete table and maps
operators against the installed pack. A valid `.pte` can still require an
operator missing from a particular runtime package.

Planned tensor bytes are not total peak RAM. They exclude runtime metadata,
stack, other workspaces and other program data. Recheck memory after layout
or graph changes; avoid sizing a model using its weight file alone.

## 10. Compile the supplied models using the existing commands

Run all commands below from the repository root.

### TinyCNN pipeline check

Validate compilation without writing generated files:

```powershell
.\.venv\Scripts\python.exe model\export_cpu.py --model tinycnn --cpu cortex-m85 --check-only
```

Generate the model:

```powershell
.\.venv\Scripts\python.exe model\export_cpu.py --model tinycnn --cpu cortex-m85
```

Primary outputs:

```text
model/cpu/model.pte
model/cpu/graph.txt
model/cpu/validation.json
```

TinyCNN uses deterministic untrained weights and synthetic samples. Use it
to check the pipeline, not to assess a trained model's task accuracy.

### SqueezeNet rock-paper-scissors

Configure `model/rps_config.json`. On another machine, start from
`model/rps_config.example.json`:

```json
{
  "checkpoint": "rps_alif_model.pth",
  "images": [
    "calibration/Paper.png",
    "calibration/Rock.png",
    "calibration/Scissors.png"
  ],
  "labels": ["PAPER", "ROCK", "SCISSORS", "UNKNOWN"]
}
```

Relative asset paths resolve from the configuration file's directory.
The current workspace's local configuration already points to the supplied
checkpoint and images. The architecture is SqueezeNet 1.1 with four outputs.

Check the compilation:

```powershell
.\.venv\Scripts\python.exe model\export_cpu.py --model rps --cpu cortex-m85 --check-only
```

Generate the `.pte`:

```powershell
.\.venv\Scripts\python.exe model\export_cpu.py --model rps --cpu cortex-m85
```

To choose a different configuration file:

```powershell
.\.venv\Scripts\python.exe model\export_cpu.py --model rps --cpu cortex-m85 --rps-config model\my_rps_config.json
```

Primary outputs:

```text
model/rps/model.pte
model/rps/graph.txt
model/rps/validation.json
```

The report includes input/output quantization parameters, expected INT8
outputs, FP32 reference outputs, planned memory, checkpoint SHA-256, source
images, normalization and ordered labels.

The current RPS input is logical `[1,3,224,224]`, physically NHWC; its output
is `[1,4]` INT8. Compilation uses RGB images, direct bilinear resize to
224x224, division by 255, then ImageNet normalization:

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

Use the normalization from training for another model. These values are
specific to this RPS configuration.

## 11. Extend compilation to another model

The existing CLI accepts **`tinycnn` and `rps` only**. It is not an arbitrary
checkpoint importer. For another architecture, use the stages above to create
a separate exporter, or add a model-specific loader and CLI option.

Use `model/rps_model.py` as a loader example and `model/export_cpu.py` as the
compilation sequence. The changes for another model belong in model loading,
preprocessing, calibration, signatures and model-specific export checks:

| Export concern     | What to provide or adapt                                        |
| ------------------ | --------------------------------------------------------------- |
| Model construction | Exact architecture and trained checkpoint                       |
| Inputs             | Correct number, logical shapes, dtypes and physical layouts     |
| Data loading       | Representative samples processed exactly as during training     |
| Calibration        | Representative deployment ranges, distinct from validation      |
| Outputs            | Structure, shapes, class labels or other interpretation         |
| INT8 boundaries    | Adapt boundary conversion if not one per-tensor input/output    |
| Expected lowering  | Require only operations that exist in the new architecture      |
| Layout conversion  | Adapt portable-kernel layout handling to the new graph          |
| Artifacts          | Separate `.pte`, graph and metadata output directory          |
| Validation         | FP32 baseline, quantized baseline, and compiled-runtime results |

A minimal model-loader interface can look like this:

```python
def load_model_and_data():
    model = build_exact_architecture()
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()

    # Each item is a tuple matching model.forward's signature.
    calibration_inputs = load_preprocessed_calibration_samples()
    validation_inputs = load_preprocessed_validation_samples()
    example_inputs = validation_inputs[0]
    return model, example_inputs, calibration_inputs, validation_inputs
```

The loader functions above are placeholders to implement for the new model.
For a four-dimensional CNN, make its model parameters and input tensors
channels-last before the first export. Then follow sections 5 through 9.

Do not retain TinyCNN's 16x16 shape, ten-class assumptions, synthetic
calibration data or required-linear-operator check for an unrelated network.
Likewise, do not retain RPS's four labels, RGB preprocessing or image-only
assumptions for another task.

If conversion leaves unsupported operations, determine whether a compatible
portable kernel exists and whether its dtype/layout is appropriate. Otherwise
change the model, implement the missing support, or use a compatible backend
version. An unresolved operator is not a successful CPU deployment.

## 12. Validate the compiled model

Use three comparisons:

1. **FP32 versus calibrated INT8:** assess task accuracy and quantization
   error on an independent validation set.
2. **Float-boundary quantized versus INT8-boundary quantized:** converting
   the model boundary must preserve the expected integer output codes.
3. **Host quantized reference versus compiled runtime:** compare the same
   input bytes and all output values using a documented tolerance.

Also verify:

- All external input/output dtypes and shapes match the intended interface.
- The CPU target and ISA are correct.
- Expected CMSIS-NN operators appear and no NPU delegation remains.
- Every serialized operator has a corresponding runtime kernel.
- Layouts satisfy both optimized and portable kernel requirements.
- Planned tensor memory and model-data size fit the target budget.
- Re-export is reproducible with the same checkpoint, samples and versions.

The verified RPS result for this repository was:

| Item                                             | Result                               |
| ------------------------------------------------ | ------------------------------------ |
| CPU target                                       | Cortex-M85 / MVE                     |
| Model                                            | Trained SqueezeNet 1.1, four outputs |
| External tensors                                 | Signed INT8 input and output         |
| Serialized program                               | 831,552 bytes                        |
| Planned tensor memory                            | 1,175,744 bytes                      |
| Portable cat/view layout checks                  | Nine operations checked              |
| Supplied image predictions                       | PAPER, ROCK, SCISSORS                |
| M85 FVP output-code error versus host references | Zero for all three examples          |

Those three images establish the demonstrated compilation and numerical
checks. They do not establish general classification accuracy or coverage of
UNKNOWN examples. Use a larger calibration and validation set before drawing
accuracy conclusions.

## 13. Common compilation problems

| Symptom                                   | Check or correction                                                                                     |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| Cortex-M modules fail to import           | Use the pinned environment and run its import check before exporting.                                   |
| API arguments differ from online examples | Use the API for installed ExecuTorch 1.4.1; do not mix versions.                                        |
| Convolution remains floating point        | Check channels-last inputs/parameters and the quantizer's shape/operator constraints.                   |
| Serialized inputs/outputs remain FLOAT    | Convert the calibrated boundary and re-export with INT8 examples; verify serialized dtypes.             |
| Portable reshape rejects a tensor         | Provide default-contiguous layout before `view_copy`.                                                 |
| Concatenation values are wrong            | Check physical layout against the actual portable kernel's copy implementation.                         |
| Required operator is missing              | Read the complete serialized operator table, including layout namespaces, and check runtime coverage.   |
| Model/exporter version mismatch           | Align the exporter and selected runtime, then regenerate the `.pte`.                                  |
| Large numerical or accuracy regression    | Verify preprocessing, calibration data, label order, quantization parameters and graph transformations. |
| Model exceeds memory limits               | Check activation shapes and layout-copy buffers as well as weights; recompute the final memory plan.    |

Deprecation warnings for the verified export APIs do not by themselves mean
compilation failed. Require the expected artifacts and explicit validation
results; investigate exceptions, missing operators and failed checks.

## Repository references

- [CPU exporter](../model/export_cpu.py)
- [RPS checkpoint and image loader](../model/rps_model.py)
- [Portable-kernel layout adaptation and validation](../model/runtime_layout.py)
- [Operator-table and runtime-component mapping](../scripts/gen_components.py)
- [Core Python requirements](../requirements.txt)
- [ExecuTorch and TorchAO pins](../requirements-executorch.txt)
- [Optional RPS requirements](../requirements-rps.txt)
- [Example RPS asset configuration](../model/rps_config.example.json)

## 14. Files required for CPU model compilation

The following checklist applies to the **current repository exporter**, run
directly with Python to produce `model.pte`.

### Required Python source files

| File | Why it is required |
|---|---|
| `model/export_cpu.py` | Entry point: quantization, INT8 boundaries, Cortex-M85 lowering, serialization and export checks. |
| `model/runtime_layout.py` | Layout adaptation and validation for portable concatenation and reshape operations. |
| `model/export_model.py` | Supplies `_strip_guards_fn` and `_write_header`, which the CPU exporter imports. Its Ethos-U export function is not called by CPU compilation. |
| `model/model.py` | Defines TinyCNN and its inputs. Keep it even for the current RPS command because the exporter and shared helper module import it. |
| `model/rps_model.py` | Required when using `--model rps`; constructs SqueezeNet 1.1, loads its weights and prepares calibration/validation images. |

For a different architecture, also provide its Python model definition and
loader, and connect that loader to the export sequence described above.

### Required model assets

| Asset | Requirement |
|---|---|
| Trained checkpoint, such as `.pth` or `.pt` | Required for a trained model; its state dictionary must match the Python architecture. TinyCNN's pipeline demonstration creates its own untrained weights. |
| Representative calibration data | Required to determine useful activation ranges for INT8 quantization. For RPS, configure image paths in `model/rps_config.json`. |
| Validation samples | Required for numerical/accuracy checks; use an independent set when assessing accuracy. |
| Input/output specification | Shapes, preprocessing, normalization, layout and output interpretation must be known. The loader and configuration encode these for RPS. |
| `model/rps_config.json` | Required for the RPS command; contains the checkpoint path, image paths and ordered labels. Create it from `model/rps_config.example.json` on a new machine. |

The checkpoint and images may remain outside the repository. The local RPS
configuration can use absolute paths or paths relative to its own directory.

### Environment setup files

An existing compatible Python environment is required for every export.
To recreate it using this repository's setup workflow, retain:

- `setup_venv.py` and the platform wrapper: `setup_venv.bat` on Windows or
  `setup_venv.sh` on Linux/macOS.
- `requirements.txt` and `requirements-executorch.txt` for pinned dependencies.
- `requirements-arm-tosa.txt`, read by the shared repository setup script.
- `requirements-rps.txt` when installing the optional SqueezeNet/image dependencies.

Once those packages are installed, the setup scripts are not invoked by the
standalone model-export command. Recreate the virtual environment on another
machine rather than copying `.venv` between machines.

### Optional files for additional checks or build automation

| File or resource | When it is needed |
|---|---|
| `scripts/gen_components.py` | To check serialized operators against CMSIS runtime components and generate their selection. Standalone `.pte` export does not invoke it. |
| Installed `PyTorch::ExecuTorch@1.4.1` pack, including its `.pdsc` | For that runtime-component coverage check. Python export itself uses the installed ExecuTorch wheel. |
| `scripts/run_export.py` and `scripts/run_export.cmake` | When invoking export through the repository's automated build step instead of running `model/export_cpu.py` directly. |

### Generated outputs, not required input files

For RPS, compilation creates `model/rps/model.pte`, `graph.txt` and
`validation.json`. For TinyCNN, the corresponding directory is `model/cpu/`.
The exporter also creates auxiliary headers under `ai_layer/rps_model/` or
`ai_layer/cpu_model/`; these are generated artifacts and are not prerequisites
for model compilation.

With the required source files, assets and environment present, the complete
RPS model-compilation command remains:

```powershell
.\.venv\Scripts\python.exe model\export_cpu.py --model rps --cpu cortex-m85 --rps-config model\rps_config.json
```
