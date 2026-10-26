#!/usr/bin/env python3
# Copyright 2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Export TinyCNN or SqueezeNet RPS with INT8 I/O for CPU-only Cortex-M inference."""
import argparse
import json
from importlib.metadata import version
from pathlib import Path

import torch
from executorch.backends.cortex_m.passes.cortex_m_pass_manager import CortexMPassManager
from executorch.backends.cortex_m.quantizer.quantizer import CortexMQuantizer
from executorch.backends.cortex_m.target_config import CortexMTargetConfig
from executorch.exir import EdgeCompileConfig, to_edge
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

from export_model import _strip_guards_fn, _write_header
from model import get_model
from runtime_layout import PortableLayoutPass, validate_portable_layouts
from network_summary import collect_network_stats, render_network_summary

ROOT = Path(__file__).resolve().parent.parent


def c_array(name, values, ctype="int8_t"):
    return f"static const {ctype} {name}[] = {{" + ", ".join(str(int(v)) for v in values) + "};\n"


def make_int8_io(module):
    """Expose the calibrated quantized tensors at the model boundary.

    Keep internal Q/DQ nodes for Cortex-M lowering. Only remove the input
    quantize and final output dequantize, using their calibrated parameters.
    Re-exporting with an INT8 example regenerates tensor dtype metadata.
    """
    graph = module.graph
    placeholders = [n for n in graph.nodes if n.op == "placeholder"]
    if len(placeholders) != 1:
        raise RuntimeError("This example requires exactly one input")
    input_node = placeholders[0]
    users = list(input_node.users)
    if len(users) != 1 or users[0].target != torch.ops.quantized_decomposed.quantize_per_tensor.default:
        raise RuntimeError("Expected one calibrated per-tensor quantizer at the input")
    quantize = users[0]
    _, scale, zero_point, qmin, qmax, dtype = quantize.args
    if dtype != torch.int8 or (qmin, qmax) != (-128, 127):
        raise RuntimeError("Expected signed INT8 input quantization")
    output = next(n for n in graph.nodes if n.op == "output")
    result = output.args[0]
    if not isinstance(result, (tuple, list)) or len(result) != 1:
        raise RuntimeError("This example requires exactly one output")
    dequantize = result[0]
    if dequantize.target != torch.ops.quantized_decomposed.dequantize_per_tensor.default:
        raise RuntimeError("Expected calibrated per-tensor dequantization at the output")
    _, out_scale, out_zero_point, out_min, out_max, out_dtype = dequantize.args
    if out_dtype != torch.int8 or (out_min, out_max) != (-128, 127):
        raise RuntimeError("Expected signed INT8 output quantization")
    output.args = ((dequantize.args[0],),)
    graph.erase_node(dequantize)
    quantize.replace_all_uses_with(input_node)
    graph.erase_node(quantize)
    graph.lint()
    module.recompile()
    return (float(scale), int(zero_point)), (float(out_scale), int(out_zero_point))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", choices=("cortex-m55", "cortex-m85"), default="cortex-m85")
    parser.add_argument("--model", choices=("tinycnn", "rps"), default="tinycnn")
    parser.add_argument("--rps-config", type=Path, default=ROOT / "model" / "rps_config.json")
    parser.add_argument("--check-only", action="store_true", help="Validate the INT8 program in memory without writing generated files")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if version("executorch") != "1.4.1":
        raise RuntimeError("CPU export requires executorch==1.4.1 to match the runtime pack")
    torch.manual_seed(42)
    if args.model == "rps":
        from rps_model import load_rps
        model, inputs, model_info, rgb_images = load_rps(args.rps_config)
        shape = tuple(inputs[0].shape)
        calibration = inputs
    else:
        model = get_model().to(memory_format=torch.channels_last)
        shape = (1, 3, 16, 16)
        ramp = ((torch.arange(768) % 32).float() / 32 - 0.5).reshape(shape).contiguous(memory_format=torch.channels_last)
        inputs = [ramp, torch.zeros_like(ramp), (torch.rand(shape) - 0.5).contiguous(memory_format=torch.channels_last)]
        calibration = inputs + [(torch.rand(shape) - 0.5).contiguous(memory_format=torch.channels_last) for _ in range(16)]
        model_info = dict(name="TinyCNN", labels=[str(i) for i in range(10)])
    exported = torch.export.export(model, (inputs[0],), strict=True)
    _strip_guards_fn(exported.graph_module)
    gm = exported.module()
    _strip_guards_fn(gm)
    prepared = prepare_pt2e(gm, CortexMQuantizer())
    with torch.no_grad():
        for x in calibration:
            prepared(x)
    quantized = convert_pt2e(prepared)
    with torch.no_grad():
        original_outputs = [quantized(x).detach().clone() for x in inputs]
        float_outputs = [model(x).flatten().tolist() for x in inputs]
    (input_scale, input_zero_point), (output_scale, output_zero_point) = make_int8_io(quantized)
    int8_inputs = [torch.ops.quantized_decomposed.quantize_per_tensor.default(
        x, input_scale, input_zero_point, -128, 127, torch.int8
    ).contiguous(memory_format=torch.channels_last) for x in inputs]
    rgb_lut = None
    if args.model == "rps":
        normalized_codes = (torch.arange(256).float().reshape(1, 256) / 255.0 - torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1)) / torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1)
        rgb_lut = torch.ops.quantized_decomposed.quantize_per_tensor.default(
            normalized_codes, input_scale, input_zero_point, -128, 127, torch.int8
        )
        for rgb, tensor in zip(rgb_images, int8_inputs):
            codes = rgb_lut[torch.arange(3).reshape(1, 1, 3), rgb.long()]
            if not torch.equal(codes.flatten(), tensor.permute(0, 2, 3, 1).contiguous().flatten()):
                raise RuntimeError("RGB INT8 preprocessing LUT differs from ImageNet normalization and quantization")
    with torch.no_grad():
        int8_outputs = [quantized(x) for x in int8_inputs]
    for original, actual in zip(original_outputs, int8_outputs):
        expected_int8 = torch.ops.quantized_decomposed.quantize_per_tensor.default(
            original, output_scale, output_zero_point, -128, 127, torch.int8
        )
        if actual.dtype != torch.int8 or not torch.equal(expected_int8, actual):
            raise RuntimeError("Changing the I/O boundary changed the quantized reference results")
    expected = [y.flatten().tolist() for y in int8_outputs]
    tolerance = 2  # Integer output codes; allows up to two quantization steps.
    ep = torch.export.export(quantized, (int8_inputs[0],), strict=True)
    _strip_guards_fn(ep.graph_module)
    # ExecuTorch 1.4.1 uses the pass-manager API, predating the blog's API.
    edge = to_edge(ep, compile_config=EdgeCompileConfig(
        _check_ir_validity=False,
        preserve_ops=[torch.ops.aten.linear.default],
        _core_aten_ops_exception_list=[torch.ops.aten.max_pool2d.default],
    ))
    target = CortexMTargetConfig.from_target_string(args.cpu)
    edge._edge_programs["forward"] = CortexMPassManager(
        edge.exported_program(), target_config=target
    ).transform()
    edge = edge.transform([PortableLayoutPass()])
    graph = str(edge.exported_program().graph_module.graph)
    required_ops = ("quantized_conv2d", "quantized_max_pool2d")
    if args.model == "tinycnn":
        required_ops += ("quantized_linear",)
    for op in required_ops:
        if f"cortex_m.{op}" not in graph:
            raise RuntimeError(f"Expected CMSIS-NN operator missing: {op}\n{graph}")
    if "executorch_call_delegate" in graph:
        raise RuntimeError("CPU-only export contains a delegate")
    for node in edge.exported_program().graph.nodes:
        if node.op == "call_function" and any(op in str(node.target) for op in ("cortex_m.quantize_per_tensor", "cortex_m.dequantize_per_tensor")):
            raise RuntimeError("Unexpected remaining quantize/dequantize operator in the INT8 network")
    program = edge.to_executorch()
    from executorch.exir._serialize._program import deserialize_pte_binary
    from executorch.exir.schema import ScalarType
    serialized = deserialize_pte_binary(program.buffer).program
    plan = serialized.execution_plan[0]
    for value in plan.inputs + plan.outputs:
        if plan.values[value].val.scalar_type != ScalarType.CHAR:
            raise RuntimeError("Serialized model input/output is not INT8")
    if plan.values[plan.inputs[0]].val.dim_order != [0, 2, 3, 1]:
        raise RuntimeError("Expected NHWC physical storage for the NCHW-shaped input")
    layout_checks = validate_portable_layouts(plan)
    print(f"[cpu export] verified runtime layouts for {layout_checks} portable cat/view operators")
    print(f"[cpu export] verified serialized INT8 input/output; {len(inputs)} host reference cases unchanged")
    print(f"[cpu export] input tensor: dtype=int8 shape={list(shape)} storage=NHWC scale={input_scale:.10g} zero_point={input_zero_point}")
    print(f"[cpu export] output tensor: dtype=int8 shape={list(int8_outputs[0].shape)} scale={output_scale:.10g} zero_point={output_zero_point}")
    if rgb_lut is not None:
        print("[cpu export] integer RGB preprocessing LUT matches all supplied images")
    planned_bytes = sum(plan.non_const_buffer_sizes[1:])
    print(f"[cpu export] {model_info['name']}: planned tensors {planned_bytes} bytes; input {shape}; output {list(int8_outputs[0].shape)}")
    for i, (integer, fp32) in enumerate(zip(int8_outputs, float_outputs)):
        predicted = int(integer.argmax())
        float_class = max(range(len(fp32)), key=lambda j: fp32[j])
        print(f"[cpu export] case {i}: INT8 {model_info['labels'][predicted]}; FP32 {model_info['labels'][float_class]}")
    out = ROOT / "model" / ("rps" if args.model == "rps" else "cpu")
    headers = ROOT / "ai_layer" / ("rps_model" if args.model == "rps" else "cpu_model")
    pte = out / "model.pte"
    io_header = "// Generated by model/export_cpu.py; do not edit.\n#pragma once\n#include <cstdint>\n"
    io_header += f"constexpr unsigned kInputElements = {inputs[0].numel()};\nconstexpr unsigned kOutputElements = {int8_outputs[0].numel()};\n"
    io_header += f"constexpr unsigned kMethodPoolBytes = {((planned_bytes + 65536 + 15) // 16) * 16};\n"
    io_header += "constexpr unsigned kTempPoolBytes = 16384;\n"
    io_header += "static const char kModelName[] = " + json.dumps(model_info["name"]) + ";\n"
    io_header += "static const char* const kClassNames[] = {" + ", ".join(json.dumps(s) for s in model_info["labels"]) + "};\n"
    io_header += f"constexpr float kInputScale = {input_scale:.9e}f;\nconstexpr int kInputZeroPoint = {input_zero_point};\n"
    io_header += f"constexpr float kOutputScale = {output_scale:.9e}f;\nconstexpr int kOutputZeroPoint = {output_zero_point};\n"
    if rgb_lut is not None:
        io_header += "#define MODEL_INPUT_RGB888 1\n"
        io_header += c_array("kRGBInputLUT", rgb_lut.flatten().tolist())
    header = '// Generated by model/export_cpu.py; do not edit.\n#pragma once\n#include "model_io.h"\n'
    header += f"constexpr unsigned kTestCases = {len(inputs)};\n"
    header += f"constexpr int kOutputTolerance = {tolerance};\n"
    # Runtime input uses channels-last storage; retain logical NCHW model shapes.
    if rgb_lut is not None:
        header += c_array("kRGBInputs", [v for x in rgb_images for v in x.flatten().tolist()], "uint8_t")
    else:
        header += c_array("kInputs", [v for x in int8_inputs for v in x.permute(0, 2, 3, 1).contiguous().flatten().tolist()])
    header += c_array("kExpected", [v for y in expected for v in y])
    report = dict(cpu=args.cpu, isa=target.backend.name, executorch=version("executorch"),
                  model=model_info, planned_tensor_bytes=planned_bytes,
                  calibration_samples=len(calibration),
                  pte_bytes=len(program.buffer), output_scale=output_scale,
                  input=dict(dtype="int8", shape=list(shape), storage="NHWC", scale=input_scale, zero_point=input_zero_point),
                  output=dict(dtype="int8", shape=list(int8_outputs[0].shape), scale=output_scale, zero_point=output_zero_point),
                  tolerance=tolerance, expected=expected,
                  float_reference=float_outputs,
                  max_quantization_error=max(abs((a-output_zero_point)*output_scale-b) for y,z in zip(expected,float_outputs) for a,b in zip(y,z)))
    stats = collect_network_stats(serialized, model)
    report["network_summary"] = stats
    summary_path = ROOT / "ai_layer" / "NETWORK_SUMMARY.md"
    summary_markdown = render_network_summary(report, stats, pte.relative_to(ROOT).as_posix())
    report_json = json.dumps(report, indent=2) + "\n"
    if args.check_only:
        print(f"[cpu export] check-only: {len(program.buffer)} byte program; headers, reports and network summary prepared; no files written")
        return
    out.mkdir(parents=True, exist_ok=True)
    headers.mkdir(parents=True, exist_ok=True)
    pte.write_bytes(program.buffer)
    _write_header(pte, headers / "model_pte.h", "model_pte")
    (headers / "model_io.h").write_text(io_header)
    (headers / "model_reference.h").write_text(header)
    (out / "graph.txt").write_text(graph)
    (out / "validation.json").write_text(report_json)
    summary_path.write_text(summary_markdown, encoding="utf-8")
    print(f"[cpu export] network summary: {summary_path}")
    print(f"[cpu export] planned model RAM: {planned_bytes:,} bytes; model storage payload: {len(program.buffer):,} bytes (firmware overhead excluded)")
    print(f"[cpu export] {args.cpu} {target.backend.name}: {pte.stat().st_size} bytes; {len(inputs)} reference cases")
    print(f"[cpu export] INT8 input/output; output tolerance {tolerance} codes; FP32/INT8 max error {report['max_quantization_error']:.7f}")


if __name__ == "__main__":
    main()
