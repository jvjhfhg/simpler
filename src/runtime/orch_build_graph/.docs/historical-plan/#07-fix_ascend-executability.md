# Orchestration Build Graph Runtime - To Fix Executability On Ascend NPU Device

Execute with `python examples/scripts/run_example.py -k examples/orch_build_graph_example/kernels -g examples/orch_build_graph_example/golden.py -p a2a3 -r orch_build_graph -d $device_id` will cause an Ascend internal error:

```bash
Error: rtStreamSynchronize (AICPU) failed: 507018

TEST FAILED: launch_runtime failed: 507018
```

`$device_id` available from 0 to 15 on my device. You can use card 9 which the specific command should be:

```bash
python examples/scripts/run_example.py -k examples/orch_build_graph_example/kernels -g examples/orch_build_graph_example/golden.py -p a2a3 -r orch_build_graph -d 9
```

When it gets error, it may become unstoppable itself. You need a timeout or manually stop the program when the error message is displayed in terminal.

This doc is aiming for reviewing the present implementation and to fix the potential logic mistakes.

Since part of the program is executing on a heterogeneous device, you need to debug focus on prints. You need the following command to enable the full ascend logs:

```bash
export ASCEND_WORK_PATH=~/ascend-log
export ASCEND_GLOBAL_LOG_LEVEL=0
```

Then logs can be found under `~/ascend-log`.

Use the following macros to print on ascend device:
- `DEV_INFO`: Informational messages
- `DEV_DEBUG`: Debug messages
- `DEV_WARN`: Warning messages
- `DEV_ERROR`: Error messages

## Issue

The orchestration function should run on aicpu. You can launch one more aicpu thread (so 4 for now) and let aicpu thread 3 (`thread_idx == 3`) run the orchestration so that they are in the same memory space. While the orchestration is running on aicpu, it should no longer call `host_api` related functions (like `host_api.device_malloc`), but the original malloc.

You may need to do `dlopen` but make sure you are authenticated to modify in the directory (like in `/tmp` you need to take care).

FOCUS ON ORCHESTRATION DEVICE MODE, WE DON'T NEED HOST MODE FOR NOW.

---

## Analysis (Based on RT2 Commit ac71e31)

### Current Implementation Overview

The `orch_build_graph` runtime currently executes orchestration on the **host side**:

1. **Host-side orchestration** ([runtime_maker.cpp](../../host/runtime_maker.cpp)):
   - Loads orchestration SO via `dlopen` on host
   - Calls orchestration function directly on host
   - Uses `host_api.device_malloc`, `host_api.copy_to_device` for memory operations

2. **AICPU execution** ([aicpu_executor.cpp](../../aicpu/aicpu_executor.cpp)):
   - Runs `PtoScheduler` with 3 AICPU threads (configurable via `sche_cpu_num`)
   - Threads 0-2 handle task scheduling and dispatch to AICore workers
   - No orchestration thread exists

3. **Memory model**:
   - Host allocates device memory via `host_api`
   - Task graph is built on host, then executed on device
   - `HeapRing` and `TaskRing` manage device memory allocation

### Root Cause of Error 507018

The error `rtStreamSynchronize (AICPU) failed: 507018` likely occurs because:

1. **Memory space mismatch**: Orchestration runs on host but tries to access device memory directly
2. **Host API calls from device context**: If any code path attempts to call `host_api` functions from AICPU, it will fail
3. **Pointer translation issues**: Host pointers passed to orchestration may not be valid device addresses

### RT2 Device Orchestration Design

The RT2 plan (commit ac71e31) proposes running orchestration on AICPU thread 3:

```
┌─────────────────────────────────────────────────────────────────┐
│                      Device Execution                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ AICPU Thread 3 (Orchestrator)                           │   │
│  │  - dlopen SO from device memory                         │   │
│  │  - call orch_func(runtime, device_args)                 │   │
│  │  - build task graph                                     │   │
│  │  - signal ready                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ AICPU Thread 0-2 (Schedulers)                           │   │
│  │  - wait for orchestrator ready                          │   │
│  │  - dispatch tasks to AICore                             │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

Key design elements:
- **4 AICPU threads**: 3 schedulers + 1 orchestrator
- **Device-side dlopen**: Load orchestration SO from device memory
- **No host_api calls**: Use device-native malloc on AICPU
- **Metadata-driven parameter conversion**: Host converts pointers before passing to device

---

## Fix Plan

### Phase 1: Parameter Conversion Infrastructure

**Goal**: Convert host pointers to device pointers before orchestration runs on device.

#### 1.1 Add Parameter Type Metadata

Add to [runtime.h](../../runtime/runtime.h):
```cpp
enum class ArgType : int {
    SCALAR = 0,      // Direct value, no conversion
    INPUT_PTR = 1,   // Input: device_malloc + copy_to_device
    OUTPUT_PTR = 2,  // Output: device_malloc + record for copy-back
    INOUT_PTR = 3,   // Both: copy_to_device + copy-back
};
```

#### 1.2 Modify init_runtime_impl Signature

Update [runtime_maker.cpp](../../host/runtime_maker.cpp):
```cpp
int init_runtime_impl(Runtime *runtime,
                     const uint8_t* orch_so_binary,
                     size_t orch_so_size,
                     const char* orch_func_name,
                     uint64_t* func_args,
                     int func_args_count,
                     int* arg_types,           // NEW: parameter types
                     uint64_t* arg_sizes,      // NEW: sizes for pointer args
                     int orchestration_mode);  // NEW: 0=host, 1=device
```

#### 1.3 Implement Generic Parameter Conversion

In `init_runtime_impl` when `orchestration_mode == 1`:
```cpp
for (int i = 0; i < func_args_count; i++) {
    switch (arg_types[i]) {
        case ARG_INPUT_PTR:
            // Allocate device memory, copy data
            dev_ptr = runtime->host_api.device_malloc(arg_sizes[i]);
            runtime->host_api.copy_to_device(dev_ptr, host_ptr, arg_sizes[i]);
            device_args[i] = (uint64_t)dev_ptr;
            break;
        case ARG_OUTPUT_PTR:
            // Allocate device memory, record for copy-back
            dev_ptr = runtime->host_api.device_malloc(arg_sizes[i]);
            device_args[i] = (uint64_t)dev_ptr;
            runtime->record_tensor_pair(host_ptr, dev_ptr, arg_sizes[i]);
            break;
        // ... similar for INOUT_PTR and SCALAR
    }
}
```

### Phase 2: Device-Side Orchestration

**Goal**: Run orchestration function on AICPU thread 3.

#### 2.1 Copy Orchestration SO to Device

In `init_runtime_impl`:
```cpp
// Copy orchestration SO binary to device memory
void* dev_so = runtime->host_api.device_malloc(orch_so_size);
runtime->host_api.copy_to_device(dev_so, orch_so_binary, orch_so_size);
runtime->set_device_orch_so(dev_so, orch_so_size);
```

#### 2.2 Add Orchestrator Thread to AICPU Executor

Modify [aicpu_executor.cpp](../../aicpu/aicpu_executor.cpp):

```cpp
// Increase thread count
constexpr int MAX_AICPU_THREADS = 4;  // Was 3

// Add orchestrator role detection
int PtoScheduler::run(Runtime* runtime) {
    int thread_idx = thread_idx_++;

    if (thread_idx == 3) {
        // Orchestrator thread
        return run_orchestrator(runtime);
    }

    // Scheduler threads (0-2)
    return run_scheduler(runtime, thread_idx);
}

int PtoScheduler::run_orchestrator(Runtime* runtime) {
    DEV_INFO("Orchestrator thread starting");

    // 1. Load SO from device memory via dlopen
    //    Note: May need to write to a device-accessible path
    void* handle = dlopen_from_memory(runtime->get_device_orch_so(),
                                      runtime->get_device_orch_so_size());

    // 2. Get orchestration function
    OrchestrationFunc orch_func = dlsym(handle, runtime->get_orch_func_name());

    // 3. Call orchestration (builds task graph)
    //    Uses device_args (already converted pointers)
    int rc = orch_func(runtime, runtime->get_orch_args(),
                       runtime->get_orch_args_count());

    // 4. Signal schedulers that graph is ready
    orchestration_done_.store(true, std::memory_order_release);

    dlclose(handle);
    return rc;
}
```

#### 2.3 Replace host_api Calls in Orchestration

The orchestration function must NOT call `host_api` methods. Instead:

| Host API Call | Device Replacement |
|---------------|-------------------|
| `host_api.device_malloc(size)` | `malloc(size)` (AICPU heap) |
| `host_api.copy_to_device(...)` | Not needed (already on device) |
| `host_api.device_free(ptr)` | `free(ptr)` |

This requires either:
- **Option A**: Compile separate orchestration SO for device (no host_api calls)
- **Option B**: Use function pointers that resolve differently on host vs device

### Phase 3: Synchronization

**Goal**: Coordinate orchestrator and scheduler threads.

#### 3.1 Add Synchronization Primitives

```cpp
struct PtoScheduler {
    // ... existing fields ...

    // Orchestration synchronization
    std::atomic<bool> orchestration_done_{false};
    std::atomic<bool> orchestration_failed_{false};
};
```

#### 3.2 Scheduler Wait Logic

```cpp
int PtoScheduler::run_scheduler(Runtime* runtime, int thread_idx) {
    // Wait for orchestration to complete
    while (!orchestration_done_.load(std::memory_order_acquire)) {
        if (orchestration_failed_.load(std::memory_order_acquire)) {
            DEV_ERROR("Orchestration failed, aborting scheduler");
            return -1;
        }
        // Spin or yield
    }

    // Proceed with existing scheduling logic
    return schedule_and_dispatch(...);
}
```

### Phase 4: dlopen on Device

**Goal**: Handle SO loading in AICPU environment.

#### 4.1 Device-Accessible Path

AICPU may have restrictions on `/tmp`. Options:
- Use a device-specific temp directory
- Use `memfd_create` if available
- Write to a known writable location

```cpp
void* dlopen_from_memory(const void* so_data, size_t so_size) {
    // Option 1: memfd_create (Linux 3.17+)
    int fd = memfd_create("orch_so", MFD_CLOEXEC);
    write(fd, so_data, so_size);

    char fd_path[64];
    snprintf(fd_path, sizeof(fd_path), "/proc/self/fd/%d", fd);
    void* handle = dlopen(fd_path, RTLD_NOW | RTLD_LOCAL);

    close(fd);
    return handle;
}
```

#### 4.2 Fallback: Pre-loaded Function Table

If dlopen is problematic on device, use a static function table:
```cpp
// Compile orchestration functions into AICPU binary
extern "C" int BuildGraph_matmul(Runtime*, uint64_t*, int);
extern "C" int BuildGraph_attention(Runtime*, uint64_t*, int);

static OrchestrationFunc g_orch_funcs[] = {
    {"BuildGraph_matmul", BuildGraph_matmul},
    {"BuildGraph_attention", BuildGraph_attention},
};
```

---

## File Modification Checklist

| File | Changes |
|------|---------|
| `runtime/runtime.h` | Add `ArgType` enum, `set_device_orch_so()`, `set_orch_args()` methods |
| `runtime/runtime.cpp` | Implement new methods |
| `host/runtime_maker.cpp` | Add parameter conversion, copy SO to device |
| `aicpu/aicpu_executor.cpp` | Add orchestrator thread (thread 3), synchronization |
| Platform API | Update `init_runtime` signature with `arg_types`, `arg_sizes`, `orchestration_mode` |
| Python layer | Pass parameter metadata to `init_runtime` |

---

## Testing Strategy

1. **Unit test parameter conversion**: Verify host→device pointer translation
2. **Test with simple orchestration**: Single-task graph to validate dlopen on device
3. **Test synchronization**: Ensure schedulers wait for orchestration completion
4. **Full integration test**: Run the original failing command with fixes applied

```bash
# Enable verbose logging
export ASCEND_WORK_PATH=~/ascend-log
export ASCEND_GLOBAL_LOG_LEVEL=0

# Run test
timeout 60 python examples/scripts/run_example.py \
    -k examples/orch_build_graph_example/kernels \
    -g examples/orch_build_graph_example/golden.py \
    -p a2a3 -r orch_build_graph -d 9
```

---

## Conclusion (Fixes Applied)

The error 507018 (`rtStreamSynchronize (AICPU) failed`) was resolved by addressing three root causes:

### 1. DepListPool Pointer Invalidation After Host-to-Device Copy

**Problem**: The `DepListPool` struct contains a `base` pointer to the `dep_list_entries_` array. When the `Runtime` struct is copied from host to device, the embedded array gets a new device address, but the stored `base` pointer still points to the old host address.

**Fix**: Added `reinit_dep_list_pool_base()` method to [runtime.h](../../runtime/runtime.h):
```cpp
void reinit_dep_list_pool_base() {
    dep_list_pool_.base = dep_list_entries_;
}
```

Called in [aicpu_executor.cpp](../../aicpu/aicpu_executor.cpp) `init()`:
```cpp
runtime->reinit_dep_list_pool_base();
```

### 2. Per-Thread Handshaking Race Condition

**Problem**: The original implementation performed AICore handshaking per-thread, which could cause race conditions where multiple scheduler threads attempted to handshake with overlapping core ranges.

**Fix**: Implemented centralized handshaking in `PtoScheduler::init()` (matching the `host_build_graph` pattern):
```cpp
// Handshake with ALL cores during init
for (int i = 0; i < cores_total_num_; i++) {
    all_hanks[i].aicpu_ready = 1;
}
for (int i = 0; i < cores_total_num_; i++) {
    while (hank->aicore_done == 0) { /* Busy wait */ }
}
```

Removed per-thread handshaking from `run_scheduler()`.

### 3. Dual-Mode Orchestration Function

**Problem**: The orchestration function was calling `host_api` methods (`device_malloc`, `copy_to_device`) which only work on the host side.

**Fix**: Made orchestration function in [orch_example_orch.cpp](../../../../examples/orch_build_graph_example/kernels/orchestration/orch_example_orch.cpp) check the orchestration mode:
```cpp
int orch_mode = runtime->get_orchestration_mode();
if (orch_mode == 0) {
    // HOST MODE: use host_api for memory allocation
    runtime->pto_init();
    dev_a_ptr = runtime->host_api.device_malloc(size_a);
    runtime->host_api.copy_to_device(dev_a_ptr, host_a, size_a);
    // ...
} else {
    // DEVICE MODE: args are already device pointers
    dev_a_ptr = reinterpret_cast<void*>(args[0]);
    // pto_init() already called by host
}
```

### 4. Input-Only Tensor Copy-Back Skip

**Fix**: Updated [runtime_maker.cpp](../../host/runtime_maker.cpp) `validate_runtime_impl()` to skip copy-back for input-only tensors:
```cpp
if (pair.host_ptr == nullptr) {
    continue;  // Input-only, no copy-back needed
}
```

### Test Results

Both simulation and Ascend device tests now pass:

```bash
# Simulation
python examples/scripts/run_example.py -k examples/orch_build_graph_example/kernels \
    -g examples/orch_build_graph_example/golden.py -p a2a3sim -r orch_build_graph
# Result: TEST PASSED

# Ascend Device (card 9)
python examples/scripts/run_example.py -k examples/orch_build_graph_example/kernels \
    -g examples/orch_build_graph_example/golden.py -p a2a3 -r orch_build_graph -d 9
# Result: TEST PASSED
# f: PASS (16384/16384 elements matched, expected=42.0)
```

### Summary of Modified Files

| File | Changes |
|------|---------|
| `runtime/runtime.h` | Added `reinit_dep_list_pool_base()` method |
| `aicpu/aicpu_executor.cpp` | Centralized handshaking in `init()`, removed per-thread handshaking, call `reinit_dep_list_pool_base()` |
| `host/runtime_maker.cpp` | Skip copy-back for input-only tensors (host_ptr == nullptr) |
| `orch_example_orch.cpp` | Dual-mode support checking `get_orchestration_mode()` |
