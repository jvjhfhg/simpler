"""
Kernel and Orchestration Configuration (PTO Runtime Simulation)

Defines the kernels and orchestration function used by the PTO runtime example.
Reuses the same kernels and orchestration as host_build_graph_sim_example.
"""

from pathlib import Path

_KERNELS_ROOT = Path(__file__).parent

# Orchestration configs
ORCHESTRATIONS = {
    "legacy": {
        "source": str(_KERNELS_ROOT / "orchestration" / "example_orch.cpp"),
        "function_name": "build_example_graph",
    },
    "pto": {
        "source": str(_KERNELS_ROOT / "orchestration" / "pto_example_orch.cpp"),
        "function_name": "build_pto_example_graph",
    },
    "inplace": {
        "source": str(_KERNELS_ROOT / "orchestration" / "pto_inplace_test.cpp"),
        "function_name": "build_inplace_test_graph",
    },
    "multiconsumer": {
        "source": str(_KERNELS_ROOT / "orchestration" / "pto_multiconsumer_test.cpp"),
        "function_name": "build_multiconsumer_test_graph",
    },
}

# Default to legacy for backward compatibility
ORCHESTRATION = ORCHESTRATIONS["legacy"]

# Kernel configs (simulation kernels, compiled with g++)
KERNELS = [
    {"func_id": 0, "source": str(_KERNELS_ROOT / "aiv" / "kernel_add.cpp"),        "core_type": "aiv"},
    {"func_id": 1, "source": str(_KERNELS_ROOT / "aiv" / "kernel_add_scalar.cpp"), "core_type": "aiv"},
    {"func_id": 2, "source": str(_KERNELS_ROOT / "aiv" / "kernel_mul.cpp"),        "core_type": "aiv"},
]
