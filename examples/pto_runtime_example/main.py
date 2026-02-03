#!/usr/bin/env python3
"""
PTO Runtime Simulation Example (Phase 8: PTO-Only Mode)

This demonstrates the PTO runtime running on the a2a3sim simulation platform,
using the formula:

    f = (a + b + 1) * (a + b + 2)

With a=2.0, b=3.0, expected result is 42.0 for all elements.

This example validates that the PTO runtime:
1. Builds correctly via RuntimeBuilder
2. Loads orchestration and builds task graph via pto_submit_task()
3. Schedules and executes tasks on simulated AICore workers
4. Produces correct results

Usage:
    python main.py
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path so we can import bindings
example_root = Path(__file__).parent
runtime_root = Path(__file__).parent.parent.parent
runtime_dir = runtime_root / "python"
sys.path.insert(0, str(runtime_dir))
sys.path.insert(0, str(example_root))

try:
    from runtime_builder import RuntimeBuilder
    from bindings import bind_host_binary, register_kernel, set_device, launch_runtime
    from elf_parser import extract_text_section
    from kernels.kernel_config import KERNELS, ORCHESTRATION
except ImportError as e:
    print(f"Error: Cannot import module: {e}")
    print("Make sure you are running this from the correct directory")
    sys.exit(1)


def main():
    device_id = 0
    print("\n=== PTO Runtime Simulation (Phase 8: PTO-Only Mode) ===")

    # Build PTO runtime
    print("\n=== Building PTO Runtime (Simulation) ===")
    builder = RuntimeBuilder(platform="a2a3sim")
    pto_compiler = builder.get_pto_compiler()
    print(f"Available runtimes: {builder.list_runtimes()}")
    try:
        host_binary, aicpu_binary, aicore_binary = builder.build("pto_runtime")
    except Exception as e:
        print(f"Error: Failed to build runtime libraries: {e}")
        return -1

    # Load runtime library and get Runtime class
    print("\n=== Loading PTO Runtime Library ===")
    Runtime = bind_host_binary(host_binary)
    print(f"Loaded runtime ({len(host_binary)} bytes)")

    # Set device
    print(f"\n=== Setting Device {device_id} ===")
    set_device(device_id)

    # Compile orchestration shared library
    print("\n=== Compiling Orchestration Function ===")

    orch_so_binary = pto_compiler.compile_orchestration(
        ORCHESTRATION["source"],
        extra_include_dirs=[
            str(runtime_root / "src" / "runtime" / "pto_runtime" / "runtime"),  # for runtime.h
        ] + pto_compiler.get_platform_include_dirs()
    )
    print(f"Compiled orchestration: {len(orch_so_binary)} bytes")

    # Compile and register simulation kernels
    print("\n=== Compiling and Registering Simulation Kernels ===")

    pto_isa_root = "/data/wcwxy/workspace/pypto/pto-isa"

    for kernel in KERNELS:
        print(f"Compiling {kernel['source']}...")
        kernel_o = pto_compiler.compile_incore(
            kernel["source"],
            core_type=kernel.get("core_type", "aiv"),
            pto_isa_root=pto_isa_root
        )

        kernel_bin = extract_text_section(kernel_o)
        register_kernel(kernel["func_id"], kernel_bin)

    print("All kernels compiled and registered successfully")

    # Prepare input tensors
    print("\n=== Preparing Input Tensors ===")
    ROWS = 128
    COLS = 128
    SIZE = ROWS * COLS  # 16384 elements

    host_a = np.full(SIZE, 2.0, dtype=np.float32)
    host_b = np.full(SIZE, 3.0, dtype=np.float32)
    host_f = np.zeros(SIZE, dtype=np.float32)

    expected = 42.0  # (2+3+1)*(2+3+2) = 42

    print(f"Created tensors: {SIZE} elements each")
    print(f"  host_a: all {host_a[0]}")
    print(f"  host_b: all {host_b[0]}")
    print(f"  host_f: zeros (output)")
    print(f"Expected result: {expected}")

    func_args = [
        host_a.ctypes.data,
        host_b.ctypes.data,
        host_f.ctypes.data,
        host_a.nbytes,
        host_b.nbytes,
        host_f.nbytes,
        SIZE,
    ]

    # Create and initialize runtime
    print("\n=== Creating and Initializing PTO Runtime ===")
    runtime = Runtime()
    runtime.initialize(orch_so_binary, ORCHESTRATION["function_name"], func_args)

    # Execute runtime (simulation: uses threads)
    print("\n=== Executing PTO Runtime (Simulation) ===")
    launch_runtime(runtime,
                   aicpu_thread_num=3,
                   block_dim=3,
                   device_id=device_id,
                   aicpu_binary=aicpu_binary,
                   aicore_binary=aicore_binary)

    # Finalize and copy results back to host
    print("\n=== Finalizing and Copying Results ===")
    runtime.finalize()

    # Validate results
    print("\n=== Validating Results ===")
    print(f"First 10 elements of result (host_f):")
    for i in range(10):
        print(f"  f[{i}] = {host_f[i]}")

    all_correct = np.allclose(host_f, expected, rtol=1e-5)
    error_count = np.sum(~np.isclose(host_f, expected, rtol=1e-5))

    if all_correct:
        print(f"\nSUCCESS: All {SIZE} elements are correct ({expected})")
    else:
        print(f"\nFAILED: {error_count} elements are incorrect (expected {expected})")

    return 0 if all_correct else -1


if __name__ == '__main__':
    sys.exit(main())
