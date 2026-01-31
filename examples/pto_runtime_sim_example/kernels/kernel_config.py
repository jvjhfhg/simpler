"""
Kernel and Orchestration Configuration (PTO Runtime - Phase 8)

Defines the kernels and orchestration function used by the PTO runtime example.
PTO mode is the default and only mode.
"""

from pathlib import Path

_KERNELS_ROOT = Path(__file__).parent

# PTO Orchestration (default)
ORCHESTRATION = {
    "source": str(_KERNELS_ROOT / "orchestration" / "pto_example_orch.cpp"),
    "function_name": "build_pto_example_graph",
}

# Kernel configs (simulation kernels, compiled with g++)
KERNELS = [
    {"func_id": 0, "source": str(_KERNELS_ROOT / "aiv" / "kernel_add.cpp"),        "core_type": "aiv"},
    {"func_id": 1, "source": str(_KERNELS_ROOT / "aiv" / "kernel_add_scalar.cpp"), "core_type": "aiv"},
    {"func_id": 2, "source": str(_KERNELS_ROOT / "aiv" / "kernel_mul.cpp"),        "core_type": "aiv"},
]
