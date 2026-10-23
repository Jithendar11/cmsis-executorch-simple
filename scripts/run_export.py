#!/usr/bin/env python3
# Copyright 2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# Build-step launcher for the model conversion. Invoked by the cproject
# `executes:` node through scripts/run_export.cmake, which picks the venv
# interpreter for the host OS and runs this file with it.
#
# Self-locates the solution directory: the working directory of an `executes:`
# step is not controlled by CMSIS-Toolbox (it is typically tmp/), so nothing
# here may assume the caller's CWD.
"""Export the model, then refresh the AI layer's component selection."""

from __future__ import annotations

import argparse
import filecmp
import re
import shutil
import subprocess
import sys
from pathlib import Path

SOL = Path(__file__).resolve().parent.parent
PACK = "PyTorch::ExecuTorch"


def run(*args: str | Path) -> None:
    cmd = [str(a) for a in args]
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def resolved_pack_version(sol: Path, required: str | None = None) -> str | None:
    """The pack version CMSIS-Toolbox settled on, from its own record of the
    resolution. Without it gen_components.py falls back to the newest version
    installed, which is only the same thing when one is installed."""
    # The lock can retain older selections alongside a newly pinned version.
    # Match the CPU project's exact version rather than taking the first entry.
    # Entries without selected-by-pack are installed, not selected.
    selected = rf"resolved-pack: {re.escape(PACK)}@(\S+)\n\s+selected-by-pack:"
    versions = set()
    for record in sol.glob("*.cbuild-pack.yml"):
        versions.update(re.findall(selected, record.read_text()))
    if required:
        if versions and required not in versions:
            raise RuntimeError(f"{PACK}@{required} is required; resolved versions: {sorted(versions)}")
        return required
    if len(versions) > 1:
        raise RuntimeError(f"Ambiguous {PACK} selection: {sorted(versions)}. Align the project and AI-layer pack pins, then rebuild.")
    return next(iter(versions), None)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("ethosu", "cpu"), default="ethosu")
    parser.add_argument("--model", choices=("tinycnn", "rps"), default="tinycnn")
    args = parser.parse_args()
    # sys.executable is the venv interpreter: run_export.cmake invoked us with it.
    python = sys.executable
    cpu = args.backend == "cpu"
    version = resolved_pack_version(SOL, required="1.4.1" if cpu else None)

    if args.backend == "cpu":
        run(python, SOL / "model" / "export_cpu.py", "--cpu", "cortex-m85", "--model", args.model)
    else:
        run(
            python,
            SOL / "model" / "export_model.py",
            "--mlops",
            SOL / "cmsis-executorch-simple.cbuild-mlops.yml",
            "--output-dir",
            SOL / "model",
            "--header",
            SOL / "ai_layer" / "model" / "model_pte.h",
        )

    # Regenerate the AI layer's component selection from the fresh .pte. The
    # clayer is resolved at *configure* time: touching it mid-build makes
    # CMSIS-Toolbox reconfigure underneath the running ninja, so only move the
    # regenerated file into place when its content actually changed -- and then
    # fail with a re-run hint, since the new selection cannot take effect in
    # this build anyway.
    rps = cpu and args.model == "rps"
    clayer = SOL / "ai_layer" / ("rps.clayer.yml" if rps else "cpu.clayer.yml" if cpu else "ai_layer.clayer.yml")
    new = clayer.with_suffix(clayer.suffix + ".new")

    pte = SOL / "model" / ("rps" if rps else "cpu") / "model.pte" if cpu else SOL / "model" / "model.pte"
    description = "SqueezeNet RPS Cortex-M85 CPU" if rps else "TinyCNN Cortex-M85 CPU" if cpu else "AI layer for the TinyCNN Ethos-U85 example"
    run(
        python,
        SOL / "scripts" / "gen_components.py",
        "--pte",
        pte,
        *(("--pack-version", version) if version else ()),
        "--output",
        new,
        "--backend",
        args.backend,
        "--description",
        description,
    )

    if clayer.is_file() and filecmp.cmp(new, clayer, shallow=False):
        new.unlink()
        return 0

    shutil.move(str(new), str(clayer))
    print(
        "[run_export] The model's operator set changed: "
        f"{clayer.name} was regenerated.",
        file=sys.stderr,
    )
    print(
        "[run_export] Re-run the build to compile and link the updated "
        "component selection.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
