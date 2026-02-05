# Orchestration Build Graph Runtime Example

This example demonstrates how to build and execute task dependency graphs using the orchestration runtime on the simulation platform (a2a3sim).

## Overview

The example implements a comprehensive test suite combining three test scenarios:

### Test 1: Diamond Pattern
Formula: `f = (a + b + 1) * (a + b + 2)`

```
    a, b
     |
     c = a + b
    / \
   d   e     (d = c+1, e = c+2)
    \ /
     f = d * e
```

**Expected Result:** With `a=2.0` and `b=3.0`, result is `f = (2+3+1)*(2+3+2) = 42.0`

**Tests:** Basic DAG structure, intermediate buffers, parallel branches

### Test 2: In-Place Updates (SSA Versioning)
Formula: `g = (((a + 1) + 1) + 1) + 1 = a + 4`

```
a -> x_v0 -> x_v1 -> x_v2 -> g
```

**Expected Result:** With `a=2.0`, result is `g = 2 + 4 = 6.0`

**Tests:** SSA-style versioning via `pto_version_inc()`, dependency tracking across versions

### Test 3: Multi-Consumer (Fan-out Pattern)
Formula: `h = (a+1) + (a+2) + (a+3) = 3a + 6`

```
        a (3 consumers)
      / | \
    p   q   r     (p=a+1, q=a+2, r=a+3)
      \ | /
        s         (s=p+q)
        |
        h         (h=s+r)
```

**Expected Result:** With `a=2.0`, result is `h = 3*2 + 6 = 12.0`

**Tests:** Multiple consumers of single buffer, scope-based lifecycle, complex DAG

## Platform Support

This example currently supports only the simulation platform:

| Platform | Description | Flag | Requirements |
|----------|-------------|------|--------------|
| **a2a3sim** | Thread-based simulation | `-p a2a3sim` | gcc/g++ only |

### Key Characteristics

| Aspect | Simulation (a2a3sim) |
|--------|----------------------|
| Kernel compilation | g++ compiler |
| Execution | Host threads |
| Kernel format | Plain C++ loops |
| Device required | No |
| Runtime mode | Orchestration build graph |

## Dependencies

### Simulation Platform (a2a3sim)
- Python 3
- NumPy
- gcc/g++ compiler

## Quick Start

### Run on Simulation Platform

```bash
# From repository root
# Add `-v` for verbose output
python examples/scripts/run_example.py \
  -k examples/orch_build_graph_example/kernels \
  -g examples/orch_build_graph_example/golden.py \
  -p a2a3sim \
  -r orch_build_graph [-v]
```

### Run on Ascend NPU

```bash
# From repository root
# Replace `$device_id` with real ascend device id
# Add `-v` for verbose output
python examples/scripts/run_example.py \
  -k examples/orch_build_graph_example/kernels \
  -g examples/orch_build_graph_example/golden.py \
  -p a2a3 \
  -r orch_build_graph \
  -d $device_id [-v]
```

## Directory Structure

```
orch_build_graph_example/
├── README.md                    # This file
├── main.py                      # Standalone test script
├── golden.py                    # Input generation and expected output
└── kernels/
    ├── kernel_config.py         # Kernel configuration
    ├── aiv/                      # AIV kernel implementations
    │   ├── kernel_add.cpp        # Element-wise tensor addition
    │   ├── kernel_add_scalar.cpp # Add scalar to tensor elements
    │   └── kernel_mul.cpp        # Element-wise tensor multiplication
    └── orchestration/
        └── orch_example_orch.cpp # Orchestration graph building function
```

## Files

### [golden.py](golden.py)

Defines input tensors and expected output computation:

```python
__outputs__ = ["f"]           # Output tensor names
TENSOR_ORDER = ["a", "b", "f"]  # Order passed to orchestration function

def generate_inputs(params: dict) -> dict:
    # Returns: {"a": ..., "b": ..., "f": ...}

def compute_golden(tensors: dict, params: dict) -> None:
    # Computes expected output: f = (a + b + 1) * (a + b + 2)
```

### [kernels/kernel_config.py](kernels/kernel_config.py)

Defines kernel sources and orchestration function:

```python
KERNELS = [
    {"func_id": 0, "core_type": "aiv", "source": ".../kernel_add.cpp"},
    {"func_id": 1, "core_type": "aiv", "source": ".../kernel_add_scalar.cpp"},
    {"func_id": 2, "core_type": "aiv", "source": ".../kernel_mul.cpp"},
]

ORCHESTRATION = {
    "source": ".../orch_example_orch.cpp",
    "function_name": "build_orch_example_graph"
}
```

### [kernels/orchestration/orch_example_orch.cpp](kernels/orchestration/orch_example_orch.cpp)

The orchestration function that builds the complete task dependency graph combining all three test scenarios. Key features:

- **Scope-based buffer lifetime:** Uses `pto_scope_begin()` and `pto_scope_end()` to manage intermediate buffer lifecycles
- **Implicit memory allocation:** Output buffers are allocated automatically during `pto_submit_task()`
- **SSA-style versioning:** Uses `pto_version_inc()` for in-place updates with proper dependency tracking
- **Multi-consumer support:** Demonstrates buffer sharing across multiple tasks

## Expected Output

```
=== Building Runtime: orch_build_graph (platform: a2a3sim) ===
...
=== Compiling and Registering Kernels ===
Compiling kernel: kernels/aiv/kernel_add.cpp (func_id=0)
...
=== Generating Input Tensors ===
Inputs: ['a', 'b']
Outputs: ['f']
...
=== Launching Runtime ===
=== Orchestration Comprehensive Test Suite ===
Testing: Diamond pattern, In-place updates, Multi-consumer
SIZE: 16384 elements

--- Test 1: Diamond Pattern ---
Formula: f = (a + b + 1) * (a + b + 2)
Task 0: c = a + b
Task 1: d = c + 1
Task 2: e = c + 2
Task 3: f = d * e (expected: 42.0)

--- Test 2: In-Place Updates (SSA Versioning) ---
Formula: g = (((a + 1) + 1) + 1) + 1 = a + 4
Task 4: x_v0 = a + 1
Task 5: x_v1 = x_v0 + 1 (in-place)
Task 6: x_v2 = x_v1 + 1 (in-place)
Task 7: g = x_v2 + 1 (expected: 6.0)

--- Test 3: Multi-Consumer (Fan-out Pattern) ---
Formula: h = (a+1) + (a+2) + (a+3) = 3a + 6
Task 8: p = a + 1 (consumer 1 of 'a')
Task 9: q = a + 2 (consumer 2 of 'a')
Task 10: r = a + 3 (consumer 3 of 'a')
Task 11: s = p + q
Task 12: h = s + r (expected: 12.0)

=== Task Graph Summary ===
Total tasks: 13
...
=== Comparing Results ===
Comparing f: shape=(16384,), dtype=float32
  First 10 actual:   [42. 42. 42. 42. 42. 42. 42. 42. 42. 42.]
  First 10 expected: [42. 42. 42. 42. 42. 42. 42. 42. 42. 42.]
  f: PASS (16384/16384 elements matched)

============================================================
TEST PASSED
============================================================
```

## Environment Setup

### For Simulation Platform (a2a3sim)

No special environment setup required. Just ensure gcc/g++ is in PATH.

## Kernels

The same kernel source files work for both host and orchestration runtimes:

- [kernel_add.cpp](kernels/aiv/kernel_add.cpp) - Element-wise tensor addition
- [kernel_add_scalar.cpp](kernels/aiv/kernel_add_scalar.cpp) - Add scalar to each tensor element
- [kernel_mul.cpp](kernels/aiv/kernel_mul.cpp) - Element-wise tensor multiplication

On a2a3sim, kernels are compiled as plain C++ with g++.

## Orchestration Runtime Features

The orchestration runtime provides several key features demonstrated in this example:

### 1. Task Submission API
```cpp
int pto_submit_task(
    int func_id,              // Kernel function ID
    PTOWorkerType worker_type, // VECTOR, CUBE, etc.
    PTOParam* params,         // Input/output parameters
    int param_count           // Number of parameters
)
```

### 2. Scope-Based Memory Management
```cpp
runtime->pto_scope_begin();  // Start scope
// ... submit tasks ...
runtime->pto_scope_end();    // End scope - frees intermediate buffers
```

Intermediate buffers are automatically freed when:
- All consumer tasks complete
- The scope ends

### 3. SSA-Style Versioning
```cpp
PTOBufferHandle* new_version = runtime->pto_version_inc(old_buffer);
```

Enables in-place updates while maintaining proper dependencies:
- Creates a new version of a buffer
- Tracks dependencies between versions
- Ensures correct execution order

### 4. Parameter Types

**INPUT:** Existing buffer (read-only)
```cpp
PTOParam p = {};
p.type = PTOParamType::INPUT;
p.buffer = &existing_buffer;
p.tensor = make_tensor_bbox(buffer->addr, size);
```

**OUTPUT:** New buffer (allocated automatically)
```cpp
PTOParam p = {};
p.type = PTOParamType::OUTPUT;
p.buffer = &output_buffer_handle;
p.tensor = make_tensor_bbox(0, size);  // addr=0, filled during submit
```

**SCALAR:** Immediate value
```cpp
PTOParam p = {};
p.type = PTOParamType::SCALAR;
p.scalar_value = value;
```

## Simulation Architecture

The simulation platform (a2a3sim) emulates the AICPU/AICore execution model:

- **Kernel loading:** Kernel `.text` sections are mmap'd into executable memory
- **Thread execution:** Host threads emulate AICPU scheduling and AICore computation
- **Memory:** All allocations use host memory (malloc/free)
- **Same API:** Uses identical C API as the real hardware platform
- **Task scheduling:** Automatically handles dependencies and parallel execution

## Comparison: Host vs Orchestration Runtimes

| Feature | Host Build Graph | Orchestration Build Graph |
|---------|------------------|---------------------------|
| Graph construction | Host code (C++ in main.py) | Orchestration function (orch_example_orch.cpp) |
| Execution mode | Host directly calls runtime | Orchestration function submitted to runtime |
| Memory management | Explicit allocation/free | Scope-based with automatic allocation |
| Use case | Simple, direct control | Complex DAGs, automatic optimization |

## Troubleshooting

### Compilation Errors (a2a3sim)

- Ensure gcc/g++ is installed and available in PATH
- Check kernel source syntax for C++ errors
- Use `-v` flag for verbose compilation logs

### Runtime Errors

**Task submission failed:**
- Verify kernel `func_id` matches registered kernels
- Check parameter count and types
- Ensure buffer handles are properly initialized

**Memory errors:**
- Verify buffer sizes match tensor descriptors
- Check that input buffers are allocated before use
- Ensure `pto_scope_begin()` is called before task submission

### Result Validation Failed

- Use `-v` flag to see detailed execution logs
- Check that kernel implementations match expected operations
- Verify input data generation in [golden.py](golden.py)

## See Also

- [Host Build Graph Example](../host_build_graph_example/README.md) - Alternative runtime mode with explicit graph construction
- [Test Framework Documentation](../scripts/README.md) - Details on the test framework
- [Main Project README](../../README.md) - Overall project documentation