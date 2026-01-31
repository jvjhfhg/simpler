# Implementing a New Runtime

This guide explains how to implement a new runtime based on the existing `host_build_graph` framework.

## Architecture Overview

The runtime system consists of three components that run on different processors:

```
┌─────────────────────────────────────────────────────────────┐
│                         HOST                                 │
│  - Loads orchestration .so dynamically                      │
│  - Calls orchestration function to build task graph         │
│  - Manages device memory via HostApi                        │
│  - Copies results back after execution                      │
└─────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────┐
│                         AICPU                                │
│  - Schedules tasks based on dependencies                    │
│  - Dispatches tasks to AICore workers                       │
│  - Tracks task completion via handshake buffers             │
└─────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────┐
│                        AICORE                                │
│  - Polls handshake buffer for assigned tasks                │
│  - Executes kernel via function pointer                     │
│  - Signals completion back to AICPU                         │
└─────────────────────────────────────────────────────────────┘
```

## Directory Structure

Create your runtime under `src/runtime/<your_runtime_name>/`:

```
src/runtime/my_runtime/
├── build_config.py           # Required: RuntimeBuilder discovers this
├── runtime/
│   ├── runtime.h             # Core data structures
│   └── runtime.cpp           # Runtime implementation
├── host/
│   └── runtimemaker.cpp      # Host-side init/finalize
├── aicpu/
│   └── runtimeexecutor.cpp   # AICPU scheduler
└── aicore/
    └── aicore_executor.cpp   # AICore worker
```

## Step 1: Create `build_config.py`

This file is required for `RuntimeBuilder` to discover and build your runtime.

```python
# src/runtime/my_runtime/build_config.py

BUILD_CONFIG = {
    "aicore": {
        "include_dirs": ["runtime"],      # Relative paths from this file
        "source_dirs": ["aicore", "runtime"]
    },
    "aicpu": {
        "include_dirs": ["runtime"],
        "source_dirs": ["aicpu", "runtime"]
    },
    "host": {
        "include_dirs": ["runtime"],
        "source_dirs": ["host", "runtime"]
    }
}
```

## Step 2: Define Runtime Data Structures (`runtime/runtime.h`)

This header defines the core data structures shared across all three components.

### Required Structures

#### Handshake Buffer (AICPU ↔ AICore communication)

```cpp
// Must be cache-line aligned (64 bytes) for coherency
struct alignas(64) Handshake {
    volatile int aicpu_ready;    // AICPU signals readiness
    volatile int aicore_done;    // AICore signals initialization complete
    volatile void* task;         // Pointer to current task (or nullptr if idle)
    volatile int task_status;    // 0=complete/idle, 1=in-progress
    volatile int control;        // 0=run, 1=quit
    int core_type;               // 0=AIC, 1=AIV
    char padding[64 - 24];       // Pad to 64 bytes
};
```

#### Task Structure

```cpp
struct Task {
    int task_id;
    int func_id;                              // Maps to registered kernel
    uint64_t args[RUNTIME_MAX_ARGS];          // Kernel arguments
    uint64_t functionBinAddr;                 // Device address of kernel binary
    int core_type;                            // 0=AIC, 1=AIV

    // For DAG scheduling (optional - depends on your scheduling model)
    std::atomic<int> fanin;                   // Incomplete dependencies
    int fanout[RUNTIME_MAX_FANOUT];           // Successor task IDs
    int fanout_count;
};
```

#### Host API (Device Memory Operations)

```cpp
struct HostApi {
    void* (*DeviceMalloc)(size_t size);
    void (*DeviceFree)(void* ptr);
    void (*CopyToDevice)(void* dst, const void* src, size_t size);
    void (*CopyFromDevice)(void* dst, const void* src, size_t size);
};
```

#### TensorPair (For Copy-Back)

```cpp
struct TensorPair {
    void* hostPtr;
    void* devPtr;
    size_t size;
};
```

### Runtime Class

```cpp
class Runtime {
public:
    // Configuration
    int scheCpuNum;          // Number of AICPU threads
    int block_dim;           // Number of AICore workers

    // Communication
    Handshake handshakes[RUNTIME_MAX_WORKER];

    // Tasks
    Task tasks[RUNTIME_MAX_TASKS];
    int task_count;

    // Memory management
    HostApi host_api;
    TensorPair tensor_pairs[RUNTIME_MAX_TENSOR_PAIRS];
    int tensor_pair_count;

    // API methods
    int add_task(uint64_t* args, int arg_count, int func_id, int core_type);
    void add_successor(int from, int to);  // For DAG scheduling
    void RecordTensorPair(void* host, void* dev, size_t size);

    // Introspection
    int get_task_count() const;
    void print_runtime() const;
};
```

## Step 3: Implement Host Component (`host/runtimemaker.cpp`)

The host component must export two C functions:

### InitRuntime

Called by `runtime.initialize()` from Python.

```cpp
extern "C" int InitRuntime(
    Runtime* runtime,
    const char* orch_so_binary,      // Orchestration .so as binary data
    size_t orch_so_binary_len,
    const char* orch_func_name,      // Function name to call
    uint64_t* func_args,             // Arguments for orchestration
    int func_arg_count
) {
    // 1. Write orchestration binary to memory-backed file
    int fd = memfd_create("orch_so", MFD_CLOEXEC);
    write(fd, orch_so_binary, orch_so_binary_len);

    // 2. Load as shared library
    char path[64];
    snprintf(path, sizeof(path), "/proc/self/fd/%d", fd);
    void* handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);

    // 3. Resolve orchestration function
    typedef int (*OrchFunc)(Runtime*, uint64_t*, int);
    OrchFunc orch_func = (OrchFunc)dlsym(handle, orch_func_name);

    // 4. Call orchestration to build task graph
    int result = orch_func(runtime, func_args, func_arg_count);

    return result;
}
```

### FinalizeRuntime

Called by `runtime.finalize()` from Python.

```cpp
extern "C" int FinalizeRuntime(Runtime* runtime) {
    // Copy recorded tensors back to host
    for (int i = 0; i < runtime->tensor_pair_count; i++) {
        TensorPair& tp = runtime->tensor_pairs[i];
        runtime->host_api.CopyFromDevice(tp.hostPtr, tp.devPtr, tp.size);
        runtime->host_api.DeviceFree(tp.devPtr);
    }
    runtime->tensor_pair_count = 0;
    return 0;
}
```

## Step 4: Implement AICPU Scheduler (`aicpu/runtimeexecutor.cpp`)

The AICPU component must export a `Run()` function as its entry point.

### Key Responsibilities

1. **Initialize**: Read runtime config, set up per-thread core assignments
2. **Handshake**: Signal readiness to AICore workers, wait for acknowledgment
3. **Schedule**: Dispatch tasks to idle cores, track completions
4. **Shutdown**: Send quit signal to all AICore workers

### Simplified Single-Threaded Example

```cpp
class AicpuExecutor {
public:
    void Run() {
        Runtime* runtime = GetRuntimePtr();  // Platform-specific

        // Handshake with AICore
        for (int i = 0; i < runtime->block_dim; i++) {
            runtime->handshakes[i].aicpu_ready = 1;
        }
        for (int i = 0; i < runtime->block_dim; i++) {
            while (runtime->handshakes[i].aicore_done == 0) {}
        }

        // Main scheduling loop
        int next_task = 0;
        int completed = 0;

        while (completed < runtime->task_count) {
            for (int core = 0; core < runtime->block_dim; core++) {
                Handshake& hs = runtime->handshakes[core];

                // Check for task completion
                if (hs.task != nullptr && hs.task_status == 0) {
                    completed++;
                    hs.task = nullptr;
                    // Update successor fanin counts if using DAG
                }

                // Dispatch next task if core is idle
                if (hs.task == nullptr && next_task < runtime->task_count) {
                    Task& task = runtime->tasks[next_task++];
                    hs.task = &task;
                    hs.task_status = 1;
                }
            }
        }

        // Shutdown AICore workers
        for (int i = 0; i < runtime->block_dim; i++) {
            runtime->handshakes[i].control = 1;
        }
    }
};

extern "C" void Run() {
    AicpuExecutor executor;
    executor.Run();
}
```

### DAG Scheduling (like `host_build_graph`)

For dependency-based scheduling, track `fanin` counts:

```cpp
// On task completion
Task* completed_task = (Task*)hs.task;
for (int i = 0; i < completed_task->fanout_count; i++) {
    Task& successor = runtime->tasks[completed_task->fanout[i]];
    if (successor.fanin.fetch_sub(1) == 1) {
        // Successor is now ready, add to ready queue
        ready_queue.push(&successor);
    }
}
```

## Step 5: Implement AICore Worker (`aicore/aicore_executor.cpp`)

The AICore component executes kernels assigned by AICPU.

```cpp
extern "C" __aicore__ void AicoreExecute(
    __gm__ char* runtime_ptr,
    int blockIdx,
    int coreType
) {
    __gm__ Runtime* runtime = reinterpret_cast<__gm__ Runtime*>(runtime_ptr);
    __gm__ Handshake* hs = &runtime->handshakes[blockIdx];

    // Wait for AICPU ready signal
    while (hs->aicpu_ready == 0) {
        dcci();  // Cache flush for coherency
    }
    hs->aicore_done = blockIdx + 1;

    // Main execution loop
    while (true) {
        dcci();

        // Check for quit signal
        if (hs->control == 1) break;

        // Check for assigned task
        if (hs->task != nullptr && hs->task_status == 1) {
            __gm__ Task* task = (__gm__ Task*)hs->task;

            // Execute kernel via function pointer
            typedef void (*KernelFunc)(__gm__ int64_t*);
            KernelFunc kernel = (KernelFunc)task->functionBinAddr;
            kernel(task->args);

            // Signal completion
            hs->task_status = 0;
        }
    }
}
```

## Step 6: Create an Example

Create an example under `examples/<your_runtime_name>_example/`:

```
examples/my_runtime_example/
├── main.py
├── kernels/
│   ├── kernel_config.py
│   ├── orchestration/
│   │   └── my_orch.cpp
│   └── aiv/
│       └── my_kernel.cpp
```

### Kernel Configuration (`kernel_config.py`)

```python
from pathlib import Path

_KERNELS_ROOT = Path(__file__).parent

ORCHESTRATION = {
    "source": str(_KERNELS_ROOT / "orchestration" / "my_orch.cpp"),
    "function_name": "BuildMyGraph",
}

KERNELS = [
    {"func_id": 0, "source": str(_KERNELS_ROOT / "aiv" / "my_kernel.cpp"), "core_type": "aiv"},
]
```

### Orchestration Function (`orchestration/my_orch.cpp`)

```cpp
#include "runtime.h"

extern "C" int BuildMyGraph(Runtime* runtime, uint64_t* args, int arg_count) {
    // Extract host pointers from args
    void* host_input = reinterpret_cast<void*>(args[0]);
    void* host_output = reinterpret_cast<void*>(args[1]);
    size_t size = static_cast<size_t>(args[2]);

    // Allocate device memory
    void* dev_input = runtime->host_api.DeviceMalloc(size);
    void* dev_output = runtime->host_api.DeviceMalloc(size);

    // Copy input to device
    runtime->host_api.CopyToDevice(dev_input, host_input, size);

    // Record output for copy-back
    runtime->RecordTensorPair(host_output, dev_output, size);

    // Build task
    uint64_t task_args[3] = {
        reinterpret_cast<uint64_t>(dev_input),
        reinterpret_cast<uint64_t>(dev_output),
        size
    };
    runtime->add_task(task_args, 3, /*func_id=*/0, /*core_type=*/1);

    return 0;
}
```

### Main Script (`main.py`)

```python
import numpy as np
from pathlib import Path
import sys

# Setup paths
runtime_dir = Path(__file__).parent.parent.parent / "python"
sys.path.insert(0, str(runtime_dir))

from runtime_builder import RuntimeBuilder
from runtime_bindings import load_runtime, register_kernel, set_device, launch_runtime
from pto_compiler import PTOCompiler
from elf_parser import extract_text_section
from kernels.kernel_config import KERNELS, ORCHESTRATION

def main():
    # Build runtime
    builder = RuntimeBuilder()
    host_binary, aicpu_binary, aicore_binary = builder.build("my_runtime")

    # Load runtime
    Runtime = load_runtime(host_binary)
    set_device(0)

    # Compile orchestration
    compiler = PTOCompiler()
    orch_binary = compiler.compile_orchestration(ORCHESTRATION["source"])

    # Compile and register kernels
    for kernel in KERNELS:
        kernel_o = compiler.compile_incore(kernel["source"], core_type=kernel["core_type"])
        kernel_bin = extract_text_section(kernel_o)
        register_kernel(kernel["func_id"], kernel_bin)

    # Prepare data
    SIZE = 1024
    input_data = np.ones(SIZE, dtype=np.float32)
    output_data = np.zeros(SIZE, dtype=np.float32)

    func_args = [
        input_data.ctypes.data,
        output_data.ctypes.data,
        input_data.nbytes,
    ]

    # Execute
    runtime = Runtime()
    runtime.initialize(orch_binary, ORCHESTRATION["function_name"], func_args)
    launch_runtime(runtime, aicpu_thread_num=1, block_dim=1, device_id=0,
                   aicpu_binary=aicpu_binary, aicore_binary=aicore_binary)
    runtime.finalize()

    print(f"Result: {output_data[:10]}")

if __name__ == "__main__":
    main()
```

## Design Considerations

### Scheduling Models

| Model | Complexity | Use Case |
|-------|-----------|----------|
| **Linear** | Simple | Sequential kernels, no parallelism |
| **DAG** | Medium | Task parallelism with dependencies |
| **Pipeline** | Complex | Streaming data through stages |
| **Static** | Simple | Pre-computed schedule, no runtime decisions |

### What to Reuse vs. Customize

| Component | Reusable? | Notes |
|-----------|-----------|-------|
| `RuntimeBuilder` | ✅ Yes | Discovers any runtime with `build_config.py` |
| `PTOCompiler` | ✅ Yes | Compiles kernels and orchestration generically |
| `runtime_bindings.py` | ✅ Yes | Generic ctypes wrapper |
| `elf_parser.py` | ✅ Yes | Extracts `.text` from any ELF |
| Handshake protocol | ⚠️ Mostly | Core polling mechanism is device-specific |
| `Runtime` struct | ❌ No | Define your own task/scheduling model |
| Orchestration functions | ❌ No | Must use your `Runtime` API |

### Tips

1. **Start simple**: Copy `host_build_graph`, strip out DAG scheduling, implement linear execution first
2. **Keep the Handshake struct**: The AICPU↔AICore polling protocol is tied to hardware
3. **Keep unified kernel signature**: `void kernel(__gm__ int64_t* args)` enables kernel reuse
4. **Test incrementally**: Verify host→AICPU→AICore communication before adding complex scheduling