#!/usr/bin/env python3
# Copyright 2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Generate the AI layer's component list from an exported .pte.

The PyTorch::ExecuTorch pack exposes every operator as a selectable CMSIS
component, so an image links only the kernels its model actually uses. This
script reads the operators referenced by a .pte, looks up which components the
pack offers (straight from its .pdsc), and writes an `ai_layer.clayer.yml`
selecting the runtime plus exactly those operator components.

  python scripts/gen_components.py --pte model/model.pte \
      --output ai_layer/ai_layer.clayer.yml

The pack is read from the CMSIS pack root, where `cpackget` / `cbuild --packs`
install it; --pack-path overrides that with a pack directory of your own.
"""

import argparse
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# quantized_decomposed:: op base name -> "Quantized <Csub>" component suffix.
QUANTIZED_DECOMPOSED_TO_COMPONENT = {
    "quantize_per_tensor": "quantize",
    "quantize_per_channel": "quantize",
    "dequantize_per_tensor": "dequantize",
    "dequantize_per_channel": "dequantize",
    "choose_qparams": "choose_qparams",
    "embedding_byte": "embedding",
}


def pack_root() -> Path:
    """The CMSIS pack root: $CMSIS_PACK_ROOT, else cpackget's own default."""
    env = os.environ.get("CMSIS_PACK_ROOT")
    if env:
        return Path(env)
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local"
        return Path(base) / "Arm" / "Packs"
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "arm" / "packs"


def version_key(version: str) -> tuple:
    """Sort key placing a release above its own pre-releases (1.4.0 > 1.4.0-rc2)."""
    core, _, pre = version.partition("-")
    numbers = tuple(int(p) if p.isdigit() else 0 for p in core.split("."))
    return (numbers, pre == "", pre)


def find_pack(pack_name: str, root: Path, version: str | None) -> Path:
    """Locate an installed pack: <root>/<vendor>/<name>/<version>/."""
    vendor, _, name = pack_name.partition("::")
    base = root / vendor / name
    if version:
        found = base / version
        if not found.is_dir():
            sys.exit(f"[gen_components] {pack_name}@{version} is not installed in {root}")
        return found
    installed = sorted(
        (d.name for d in base.iterdir() if d.is_dir()) if base.is_dir() else (),
        key=version_key,
    )
    if not installed:
        sys.exit(
            f"[gen_components] no {pack_name} installed in {root}.\n"
            f"  Install it with: cpackget add {pack_name}"
        )
    return base / installed[-1]


def pack_component_suffixes(pdsc: Path) -> dict[str, set[str]]:
    """Return the operator Csub suffixes the pack offers, keyed by family
    ('Portable' / 'Quantized' / 'Cortex-M'), parsed from the pack .pdsc."""
    families: dict[str, set[str]] = {"Portable": set(), "Quantized": set(), "Cortex-M": set()}
    for csub in re.findall(r'Csub="([^"]+)"', pdsc.read_text()):
        family, _, name = csub.partition(" ")
        if family == "Cortex-M":
            families["Cortex-M"].add(name)
        elif family in ("Portable", "Quantized"):
            families[family].add(name)
    return families


def extract_ops(pte: Path) -> set[str]:
    """Read every execution plan's operator table, including layout operators.

    Scanning selected namespaces in the bytes silently omitted dim_order_ops
    kernels and could mistake constants for operator names. Use the serialized
    schema, so unsupported namespaces also reach the unmapped-operator check.
    """
    from executorch.exir._serialize._program import deserialize_pte_binary

    program = deserialize_pte_binary(pte.read_bytes()).program
    return {
        op.name + (f".{op.overload}" if op.overload else "")
        for plan in program.execution_plan
        for op in plan.operators
    }


def map_ops(ops: set[str], avail: dict[str, set[str]]) -> tuple[list[str], list[str], list[str], list[str]]:
    portable, quantized, cortex_m, unmapped = set(), set(), set(), []
    for op in sorted(ops):
        ns, _, rest = op.partition("::")
        base = rest.split(".")[0]
        if ns == "cortex_m" and base in avail["Cortex-M"]:
            cortex_m.add(base)
        elif ns == "dim_order_ops":
            # The CMSIS component drops the operator's leading underscore:
            # dim_order_ops::_clone_dim_order.out -> Portable clone_dim_order.
            comp = base.lstrip("_")
            if comp in avail["Portable"]:
                portable.add(comp)
            else:
                unmapped.append(op)
        elif ns == "quantized_decomposed":
            comp = QUANTIZED_DECOMPOSED_TO_COMPONENT.get(base, base)
            if comp in avail["Quantized"]:
                quantized.add(comp)
            else:
                unmapped.append(op)
        elif ns == "aten":
            cand = base[:-5] if base.endswith("_copy") and base[:-5] in avail["Portable"] else base
            if cand in avail["Portable"]:
                portable.add(cand)
            else:
                unmapped.append(op)
        else:
            unmapped.append(op)
    return sorted(portable), sorted(quantized), sorted(cortex_m), unmapped


def render(portable, quantized, cortex_m, pack_name, description, backend="ethosu", pack_version=None) -> str:
    lines = [
        "# Generated by scripts/gen_components.py -- regenerate when the model's",
        "# operator set changes. This is the model's CMSIS component selection.",
        "layer:",
        "  type: AI",
        f"  description: {description}",
        "",
        "  packs:",
        # An unversioned layer can retain an older version from the pack lock
        # even when the solution requests a newer one. Pin the pack we read.
        f"    - pack: {pack_name}@{pack_version}" if pack_version else f"    - pack: {pack_name}",
        "",
        "  define:",
        "    - ET_LOG_ENABLED: 0",
        "",
        "  components:",
        "    # ExecuTorch core runtime (Runtime ships the weak bare-metal PAL).",
        "    - component: Machine Learning:ExecuTorch:Runtime",
        "    - component: Machine Learning:ExecuTorch:Kernel Utils",
        "    - component: Machine Learning:ExecuTorch:Kernel Registration",
    ]
    if backend == "ethosu":
        lines += ["    - component: Machine Learning:ExecuTorch:Backend EthosU"]
    if cortex_m:
        lines.append("    - component: CMSIS:NN Lib")
        lines.append("    # Cortex-M operators, including CMSIS-NN compute kernels.")
        lines += [f"    - component: Machine Learning:ExecuTorch Operators:Cortex-M {c}" for c in cortex_m]
    if quantized:
        lines.append("    # Quantized reference operators.")
        lines += [f"    - component: Machine Learning:ExecuTorch Operators:Quantized {c}" for c in quantized]
    if portable:
        lines.append("    # Portable operators executed on the CPU.")
        lines += [f"    - component: Machine Learning:ExecuTorch Operators:Portable {c}" for c in portable]
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pte", type=Path, required=True)
    p.add_argument("--pack-name", default="PyTorch::ExecuTorch")
    p.add_argument("--pack-version", help="installed version to read (default: the newest installed)")
    p.add_argument("--pack-root", type=Path, default=None, help="default: $CMSIS_PACK_ROOT")
    p.add_argument("--pack-path", type=Path, help="read this pack directory instead of the installed one")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--description", default="AI layer for the TinyCNN Ethos-U85 example")
    p.add_argument("--backend", choices=("ethosu", "cpu"), default="ethosu")
    args = p.parse_args()

    pack = args.pack_path or find_pack(
        args.pack_name, args.pack_root or pack_root(), args.pack_version
    )
    pdsc = next(pack.glob("*.pdsc"))
    release = ET.parse(pdsc).find("releases/release")
    if release is None or not release.get("version"):
        sys.exit(f"[gen_components] pack version missing from {pdsc}")
    pack_version = release.get("version")
    avail = pack_component_suffixes(pdsc)
    ops = extract_ops(args.pte)
    portable, quantized, cortex_m, unmapped = map_ops(ops, avail)
    if unmapped:
        sys.exit(f"[gen_components] operators missing from pack: {unmapped}")
    if args.backend == "cpu" and b"EthosUBackend" in args.pte.read_bytes():
        sys.exit("[gen_components] CPU model unexpectedly contains an Ethos-U delegate")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        render(portable, quantized, cortex_m, args.pack_name, args.description, args.backend, pack_version),
        newline="\n",
    )
    print(f"[gen_components] {args.output} (from {pdsc})")
    print(f"  cortex-m: {cortex_m}")
    print(f"  quantized: {quantized}")
    print(f"  portable: {portable}")
    if unmapped:
        print(f"  [warn] unmapped ops (not in pack): {unmapped}")


if __name__ == "__main__":
    main()
