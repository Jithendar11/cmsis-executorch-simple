# Copyright 2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Adapt channels-last Cortex-M graphs to the pack's portable kernel layouts."""
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass


class PortableLayoutPass(ExportPass):
    def call_operator(self, op, args, kwargs, meta):
        clone = exir_ops.edge.dim_order_ops._clone_dim_order.default
        if op == exir_ops.edge.aten.cat.default:
            tensors = args[0]
            rank = tensors[0].to_tensor().dim()
            # The portable cat implementation copies blocks in logical-axis
            # order; its implementation is only correct for default storage.
            contiguous = [
                super(PortableLayoutPass, self).call_operator(
                    clone, (tensor,), {"dim_order": list(range(rank))}, meta
                ) for tensor in tensors
            ]
            result = super().call_operator(op, (contiguous, *args[1:]), kwargs, meta)
            # Fire modules feed CMSIS-NN convolutions, which require NHWC.
            if rank == 4:
                return super().call_operator(clone, (result,), {"dim_order": [0, 2, 3, 1]}, meta)
            return result
        if op == exir_ops.edge.aten.view_copy.default:
            rank = args[0].to_tensor().dim()
            contiguous = super().call_operator(
                clone, (args[0],), {"dim_order": list(range(rank))}, meta
            )
            return super().call_operator(op, (contiguous, *args[1:]), kwargs, meta)
        return super().call_operator(op, args, kwargs, meta)


def validate_portable_layouts(plan):
    """Reject serialized arguments that violate the pack's portable kernels."""
    checked = 0
    for chain in plan.chains:
        for instruction in chain.instructions:
            call = instruction.instr_args
            if not hasattr(call, "op_index"):
                continue
            name = plan.operators[call.op_index].name
            if name == "aten::cat":
                ids = plan.values[call.args[0]].val.items + [call.args[-1]]
            elif name == "aten::view_copy":
                ids = [call.args[0], call.args[-1]]
            else:
                continue
            for index in ids:
                tensor = plan.values[index].val
                if tensor.dim_order != list(range(len(tensor.sizes))):
                    raise RuntimeError(f"{name} needs default contiguous storage; got {tensor.dim_order}")
            checked += 1
    return checked
