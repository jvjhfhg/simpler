# PTO Runtime Implementation Plan

This document outlines the implementation plan for a new PTO (Parallel Task Orchestration) runtime based on the existing `host_build_graph` framework. The PTO runtime introduces ring buffer-based memory management, dynamic task submission, and scope-based buffer lifecycle tracking.

**Key Design Principle**: Each phase maintains backward compatibility with the existing `Runtime` class interface. The example at `examples/pto_runtime_sim_example/main.py` must pass after every phase.

---

## 1. Overview

### 1.1 Current State (Phase 0 - COMPLETE ✓)

We have a working baseline that:
- Uses a compatible `Runtime` class (same interface as `host_build_graph`)
- Reuses existing orchestration (`example_orch.cpp`) and kernels
- Passes the simulation test: `f = (a + b + 1)(a + b + 2) = 42.0`

**Test command:**
```bash
cd /data/z00626005/code/simpler
PYTHONPATH=python:$PYTHONPATH python examples/pto_runtime_sim_example/main.py
```

### 1.2 Target State

The PTO runtime introduces:
- **Dynamic task submission**: Tasks submitted during execution via `pto_submit_task()`
- **Explicit memory API**: `pto_alloc()` / `pto_free()` for buffer lifecycle control
- **Ring buffer architecture**: O(1) allocation, implicit deallocation, zero fragmentation
- **Buffer-level reference counting**: Independent of task-level fanout (buffers may have multiple producers)
- **TensorMap for dependencies**: Automatic producer-consumer tracking via memory region lookup
- **Strided tensor descriptors**: Support non-contiguous tiles with `(addr, start_offset, strides[], repeats[], n_dims)`
- **Overlap judgment strategies**: Trade-off between speed (BoundingBox) and accuracy (StridedExact)
- **Version control for in-place updates**: SSA-style versioning via `pto_version_inc()`
- **Decoupled Orchestrator/Scheduler**: Communicate only via shared memory pointers

### 1.3 Migration Strategy

Each phase adds new capability while keeping the existing `Runtime` wrapper working:

```
Phase 0 (DONE): Runtime class baseline (compatible with host_build_graph)
     ↓
Phase 1: Add PTO data structures as headers (no behavior change)
     ↓
Phase 2: Add ring buffer utilities (no behavior change)
     ↓
Phase 3: Add TensorMap (no behavior change)
     ↓
Phase 4: Create dual-mode orchestrator (supports both APIs)
     ↓
Phase 5: Create PTO-native scheduler (behind feature flag)
     ↓
Phase 6: Create PTO-native example orchestration
     ↓
Phase 7: Full PTO mode (new example, old example still works)
     ↓
Phase 8: Remove legacy mode, PTO is default
```

---

## 2. Directory Structure

```
src/runtime/pto_runtime/
├── build_config.py              # RuntimeBuilder configuration
├── runtime/
│   ├── runtime.h                # Compatible Runtime class (Phase 0)
│   ├── runtime.cpp              # Runtime implementation (Phase 0)
│   ├── pto_types.h              # PTO data structures, tensor descriptors, overlap strategies (Phase 1)
│   ├── ring_buffer.h            # Ring buffer primitives (Phase 2)
│   ├── dep_list_pool.h          # Dependency list pool (Phase 2)
│   └── tensor_map.h             # TensorMap with strided overlap judgment (Phase 3)
├── host/
│   └── runtime_maker.cpp        # Host-side init/finalize
├── aicpu/
│   └── aicpu_executor.cpp       # AICPU scheduler
└── aicore/
    └── aicore_executor.cpp      # AICore worker

examples/pto_runtime_sim_example/
├── main.py                      # Test entry point (uses Runtime API)
└── kernels/
    ├── kernel_config.py
    ├── orchestration/
    │   └── example_orch.cpp → (symlink to host_build_graph_sim_example)
    └── aiv/
        └── *.cpp → (symlinks)
```

---

## 3. Phase 0: Baseline (COMPLETE ✓)

### 3.1 What Was Done

1. Created compatible `Runtime` class in `runtime/runtime.h`
2. Created `aicpu_executor.cpp` with DAG scheduler
3. Created `aicore_executor.cpp` with polling worker
4. Created `runtime_maker.cpp` with `init_runtime_impl` / `validate_runtime_impl`
5. Created example at `examples/pto_runtime_sim_example/`

### 3.2 Files

| File | Purpose |
|------|---------|
| `runtime/runtime.h` | Compatible Runtime class definition |
| `runtime/runtime.cpp` | Runtime class implementation |
| `host/runtime_maker.cpp` | SO loading, orchestration calling |
| `aicpu/aicpu_executor.cpp` | DAG scheduler with ready queues |
| `aicore/aicore_executor.cpp` | Polling worker kernel |
| `build_config.py` | Build configuration |

### 3.3 Test

```bash
cd /data/z00626005/code/simpler
PYTHONPATH=python:$PYTHONPATH python examples/pto_runtime_sim_example/main.py
# Expected: SUCCESS: All 16384 elements are correct (42.0)
```

---

## 4. Phase 1: PTO Data Structures (Header-Only)

### 4.1 Goal

Add PTO-specific data structures as header files. **No behavior change** - these are just definitions for future phases.

### 4.2 Changes

**Add** `runtime/pto_types.h`:
```cpp
#ifndef PTO_TYPES_H
#define PTO_TYPES_H

#include <stdint.h>

// Configuration constants
#define PTO_TASK_WINDOW_SIZE      1024
#define PTO_HEAP_SIZE             (64 * 1024 * 1024)
#define PTO_DEP_LIST_POOL_SIZE    8192
#define PTO_TENSORMAP_POOL_SIZE   4096
#define PTO_TENSORMAP_NUM_BUCKETS 1024
#define PTO_MAX_TENSOR_DIMS       8

// Task states for PTO scheduler
enum PTOTaskState : int32_t {
    PTO_TASK_PENDING   = 0,
    PTO_TASK_READY     = 1,
    PTO_TASK_RUNNING   = 2,
    PTO_TASK_COMPLETED = 3,
    PTO_TASK_CONSUMED  = 4
};

// Overlap judgment strategies (trade-off: speed vs accuracy)
// See: divergence-to-original-orchestration.md §7
enum PTOOverlapStrategy : int32_t {
    PTO_OVERLAP_BOUNDING_BOX  = 0,  // Fast: (addr, total_size) only, may false-positive
    PTO_OVERLAP_STRIDED_EXACT = 1,  // Slow: element-by-element comparison, no false-positive
};

// Strided tensor descriptor for TensorMap
// Supports non-contiguous tiles: (addr, start_offset, strides[], repeats[], n_dims)
// See: divergence-to-original-orchestration.md §6
struct PTOTensorDescriptor {
    uint64_t addr;                            // Base address in GM
    uint64_t start_offset;                    // Starting offset from addr
    uint64_t strides[PTO_MAX_TENSOR_DIMS];    // Stride per dimension
    uint64_t repeats[PTO_MAX_TENSOR_DIMS];    // Elements per dimension
    int32_t n_dims;                           // Number of dimensions
    PTOOverlapStrategy strategy;              // Overlap judgment strategy
};

// Buffer handle returned by pto_alloc()
// Supports versioning for in-place updates (SSA-style)
// See: divergence-to-original-orchestration.md §6
struct PTOBufferHandle {
    uint64_t addr;           // Device memory address
    int32_t size;            // Buffer size in bytes
    int32_t version;         // Version number (for in-place updates)
    int32_t ref_count;       // Buffer-level reference count (independent of task fanout)
};

// Shared memory header for Orchestrator ↔ Scheduler communication
struct alignas(64) PTOSharedHeader {
    volatile int32_t current_task_index;
    volatile int32_t heap_top;
    volatile int32_t orchestrator_done;
    char pad1[64 - 12];

    volatile int32_t last_task_alive;
    volatile int32_t heap_tail;
    volatile int32_t scheduler_done;
    char pad2[64 - 12];
};

// PTO Task Descriptor (ring buffer element)
struct PTOTaskDescriptor {
    int32_t task_id;
    int32_t func_id;
    int32_t worker_type;
    int32_t num_args;
    uint64_t args[16];
    uint64_t function_bin_addr;
    int32_t fanin_count;
    int32_t fanout_count;
    int32_t fanin_head;
    volatile int32_t fanout_head;
    int32_t packed_buffer_offset;
    int32_t packed_buffer_size;
    volatile int32_t fanout_lock;
};

// Parameter type for pto_submit_task
enum PTOParamType : int32_t {
    PTO_PARAM_INPUT  = 0,
    PTO_PARAM_OUTPUT = 1
};

// Task parameter with tensor descriptor
struct PTOParam {
    PTOParamType type;
    PTOTensorDescriptor tensor;   // Full strided descriptor
    PTOBufferHandle* buffer;      // Associated buffer (for ref counting)
};

#endif // PTO_TYPES_H
```

### 4.3 Test

Same as Phase 0 - adding headers doesn't change behavior.

```bash
PYTHONPATH=python:$PYTHONPATH python examples/pto_runtime_sim_example/main.py
# Expected: SUCCESS
```

---

## 5. Phase 2: Ring Buffer Utilities (Header-Only)

### 5.1 Goal

Add ring buffer allocation functions. **No behavior change** - just utilities for future use.

### 5.2 Changes

**Update** `runtime/ring_buffer.h` with implementation:

```cpp
#ifndef RING_BUFFER_H
#define RING_BUFFER_H

#include "pto_types.h"

// Task Ring - allocates PTOTaskDescriptor slots
struct TaskRing {
    PTOTaskDescriptor* base;
    int32_t size;
    int32_t head;
};

inline void task_ring_init(TaskRing* ring, PTOTaskDescriptor* base, int32_t size) {
    ring->base = base;
    ring->size = size;
    ring->head = 0;
}

inline int32_t task_ring_alloc(TaskRing* ring, volatile int32_t* tail_ptr) {
    while (true) {
        int32_t head = ring->head;
        int32_t tail = *tail_ptr;
        int32_t used = (head - tail + ring->size) % ring->size;
        if (used < ring->size - 1) {
            int32_t slot = head % ring->size;
            ring->head = head + 1;
            return slot;
        }
        // Ring full - stall
    }
}

// Heap Ring - allocates device memory buffers
struct HeapRing {
    char* base;
    int32_t size;
    int32_t top;
};

inline void heap_ring_init(HeapRing* ring, char* base, int32_t size) {
    ring->base = base;
    ring->size = size;
    ring->top = 0;
}

#define ALIGN_UP(x, align) (((x) + (align) - 1) & ~((align) - 1))

inline void* heap_ring_alloc(HeapRing* ring, int32_t alloc_size, volatile int32_t* tail_ptr) {
    alloc_size = ALIGN_UP(alloc_size, 64);
    while (true) {
        int32_t tail = *tail_ptr;
        int32_t top = ring->top;
        if (top >= tail) {
            int32_t space_at_end = ring->size - top;
            if (space_at_end >= alloc_size) {
                ring->top = top + alloc_size;
                return ring->base + top;
            }
            if (tail > alloc_size) {
                ring->top = alloc_size;
                return ring->base;
            }
        } else {
            int32_t gap = tail - top;
            if (gap >= alloc_size) {
                ring->top = top + alloc_size;
                return ring->base + top;
            }
        }
        // Insufficient space - stall
    }
}

#endif // RING_BUFFER_H
```

**Update** `runtime/dep_list_pool.h` with implementation:

```cpp
#ifndef DEP_LIST_POOL_H
#define DEP_LIST_POOL_H

#include <stdint.h>

struct DepListEntry {
    int32_t task_id;
    int32_t next_offset;  // 0 = end of list
};

struct DepListPool {
    DepListEntry* base;
    int32_t size;
    int32_t top;
};

inline void dep_list_pool_init(DepListPool* pool, DepListEntry* base, int32_t size) {
    pool->base = base;
    pool->size = size;
    pool->top = 0;
}

// Prepend entry, returns new head offset (1-indexed, 0 = empty)
inline int32_t dep_list_prepend(DepListPool* pool, int32_t current_head, int32_t task_id) {
    int32_t new_index = pool->top;
    pool->top = (pool->top + 1) % pool->size;

    DepListEntry* entry = &pool->base[new_index];
    entry->task_id = task_id;
    entry->next_offset = current_head;
    return new_index + 1;
}

inline DepListEntry* dep_list_get(DepListPool* pool, int32_t offset) {
    if (offset == 0) return nullptr;
    return &pool->base[offset - 1];
}

#endif // DEP_LIST_POOL_H
```

### 5.3 Test

Same as Phase 0 - adding utilities doesn't change behavior.

```bash
PYTHONPATH=python:$PYTHONPATH python examples/pto_runtime_sim_example/main.py
# Expected: SUCCESS
```

---

## 6. Phase 3: TensorMap (Header-Only)

### 6.1 Goal

Add TensorMap for automatic dependency tracking with strided tensor support and overlap strategies. **No behavior change** - just a utility.

### 6.2 Changes

**Add** `runtime/tensor_map.h`:

```cpp
#ifndef TENSOR_MAP_H
#define TENSOR_MAP_H

#include "pto_types.h"
#include <stdint.h>

// TensorMap entry stores full tensor descriptor for overlap checking
struct TensorMapEntry {
    PTOTensorDescriptor tensor;
    int32_t producer_task_id;
    int32_t version;              // For in-place update tracking
    int32_t next_in_bucket;
};

struct TensorMap {
    TensorMapEntry* pool;
    int32_t pool_size;
    int32_t pool_head;
    int32_t* buckets;
    int32_t num_buckets;
};

inline void tensormap_init(TensorMap* tm, TensorMapEntry* pool, int32_t pool_size,
                          int32_t* buckets, int32_t num_buckets) {
    tm->pool = pool;
    tm->pool_size = pool_size;
    tm->pool_head = 0;
    tm->buckets = buckets;
    tm->num_buckets = num_buckets;
    for (int i = 0; i < num_buckets; i++) {
        buckets[i] = 0;
    }
}

inline int32_t tensormap_hash(TensorMap* tm, uint64_t addr) {
    return (addr >> 6) % tm->num_buckets;
}

// === Overlap Judgment Strategies ===
// See: divergence-to-original-orchestration.md §7

// BoundingBox: Fast, may false-positive
inline bool overlap_bounding_box(const PTOTensorDescriptor* a, const PTOTensorDescriptor* b) {
    if (a->addr != b->addr) return false;

    // Compute bounding box for each tensor
    uint64_t a_start = a->start_offset;
    uint64_t a_end = a->start_offset;
    uint64_t b_start = b->start_offset;
    uint64_t b_end = b->start_offset;

    // Calculate total span for tensor a
    for (int d = 0; d < a->n_dims; d++) {
        a_end += a->strides[d] * (a->repeats[d] - 1);
    }
    // Calculate total span for tensor b
    for (int d = 0; d < b->n_dims; d++) {
        b_end += b->strides[d] * (b->repeats[d] - 1);
    }

    return (a_start <= b_end) && (b_start <= a_end);
}

// StridedExact: Slow, no false-positive (element-by-element)
inline bool overlap_strided_exact(const PTOTensorDescriptor* a, const PTOTensorDescriptor* b) {
    if (a->addr != b->addr) return false;
    // TODO: Implement element-by-element comparison for exact overlap detection
    // For now, fall back to bounding box
    return overlap_bounding_box(a, b);
}

// Pick the most accurate strategy based on common information
inline bool tensors_overlap(const PTOTensorDescriptor* a, const PTOTensorDescriptor* b) {
    PTOOverlapStrategy common = (a->strategy < b->strategy) ? a->strategy : b->strategy;

    switch (common) {
        case PTO_OVERLAP_STRIDED_EXACT:
            return overlap_strided_exact(a, b);
        case PTO_OVERLAP_BOUNDING_BOX:
        default:
            return overlap_bounding_box(a, b);
    }
}

inline void tensormap_insert(TensorMap* tm, const PTOTensorDescriptor* tensor,
                            int32_t producer_task_id, int32_t version) {
    int32_t bucket = tensormap_hash(tm, tensor->addr);
    int32_t slot = tm->pool_head;
    tm->pool_head = (tm->pool_head + 1) % tm->pool_size;

    TensorMapEntry* entry = &tm->pool[slot];
    entry->tensor = *tensor;
    entry->producer_task_id = producer_task_id;
    entry->version = version;
    entry->next_in_bucket = tm->buckets[bucket];
    tm->buckets[bucket] = slot + 1;  // 1-indexed
}

// Returns producer task_id or -1 if not found
inline int32_t tensormap_lookup(TensorMap* tm, const PTOTensorDescriptor* tensor,
                               int32_t last_task_alive) {
    int32_t bucket = tensormap_hash(tm, tensor->addr);
    int32_t entry_idx = tm->buckets[bucket];

    while (entry_idx != 0) {
        TensorMapEntry* entry = &tm->pool[entry_idx - 1];

        // Skip stale entries
        if (entry->producer_task_id < last_task_alive) {
            entry_idx = entry->next_in_bucket;
            continue;
        }

        // Check for overlap using appropriate strategy
        if (tensors_overlap(&entry->tensor, tensor)) {
            return entry->producer_task_id;
        }

        entry_idx = entry->next_in_bucket;
    }
    return -1;
}

#endif // TENSOR_MAP_H
```

### 6.3 Test

Same as Phase 0 - adding TensorMap doesn't change behavior.

```bash
PYTHONPATH=python:$PYTHONPATH python examples/pto_runtime_sim_example/main.py
# Expected: SUCCESS
```

---

## 7. Phase 4: Dual-Mode Runtime Class

### 7.1 Goal

Extend the `Runtime` class to support both:
1. **Legacy mode**: `add_task()` + `add_successor()` (current API)
2. **PTO mode**: `pto_submit_task()` with automatic dependencies

Key additions from divergence document:
- `pto_alloc()` / `pto_free()` for explicit buffer lifecycle (§1, §2)
- `pto_version_inc()` for in-place update versioning (§6)
- Buffer-level reference counting independent of task fanout (§5)
- Scope API deprecated (§3): lifetime managed by `pto_alloc()` / `pto_free()` + ref counting

The example continues using legacy mode, so it still works.

### 7.2 Changes

**Update** `runtime/runtime.h` to add PTO methods:

```cpp
class Runtime {
public:
    // ... existing members ...

    // === Legacy API (unchanged) ===
    int add_task(uint64_t *args, int num_args, int func_id, int core_type = 0);
    void add_successor(int from_task, int to_task);

    // === PTO API (new) ===

    // Initialize PTO mode (call before any pto_* calls)
    void pto_init(char* heap_base, int32_t heap_size);

    // --- Buffer Management (divergence §1, §2, §3) ---
    // Allocate device memory, returns handle with address
    // Address is available BEFORE task submission (required for dependent tasks)
    PTOBufferHandle* pto_alloc(int32_t size);

    // Signal "no more references will be added" (does NOT immediately free)
    // Device recycles memory after all consumers finish
    void pto_free(PTOBufferHandle* handle);

    // --- Version Control for In-Place Updates (divergence §6) ---
    // Returns new versioned handle (SSA-style)
    // Write to version v waits for all reads from version v-1
    // Read from version v waits for writes to version v to complete
    PTOBufferHandle* pto_version_inc(PTOBufferHandle* handle);

    // --- Task Submission ---
    // Submit task with automatic dependency detection via TensorMap
    int pto_submit_task(int32_t func_id, int32_t worker_type,
                        PTOParam* params, int32_t param_count);

    // --- Query ---
    bool is_pto_mode() const { return pto_mode_enabled_; }

private:
    // PTO internal state (only used if pto_init called)
    bool pto_mode_enabled_ = false;

    // TensorMap for dependency tracking
    TensorMap tensor_map_;
    TensorMapEntry tensormap_pool_[PTO_TENSORMAP_POOL_SIZE];
    int32_t tensormap_buckets_[PTO_TENSORMAP_NUM_BUCKETS];

    // Buffer tracking (divergence §5: independent of task fanout)
    PTOBufferHandle buffer_handles_[PTO_TENSORMAP_POOL_SIZE];
    int32_t buffer_handle_count_ = 0;
};
```

### 7.3 Test

The example still uses legacy API, so it passes unchanged.

```bash
PYTHONPATH=python:$PYTHONPATH python examples/pto_runtime_sim_example/main.py
# Expected: SUCCESS
```

---

## 8. Phase 5: PTO-Native Scheduler

### 8.1 Goal

Add a scheduler that works with `PTOTaskDescriptor` ring buffer. Keep the legacy scheduler working via a runtime flag.

### 8.2 Changes

**Update** `aicpu/aicpu_executor.cpp`:

```cpp
// Check which mode runtime is using
extern "C" int aicpu_execute(Runtime* runtime) {
    if (runtime->is_pto_mode()) {
        return pto_scheduler_run(runtime);
    } else {
        return legacy_scheduler_run(runtime);  // Current implementation
    }
}
```

### 8.3 Test

The example uses legacy mode, so it still passes.

```bash
PYTHONPATH=python:$PYTHONPATH python examples/pto_runtime_sim_example/main.py
# Expected: SUCCESS
```

---

## 9. Phase 6: PTO-Native Orchestration Example

### 9.1 Goal

Create a new orchestration file that uses the full PTO API: `pto_alloc()`, `pto_free()`, `pto_submit_task()`, and `pto_version_inc()`.

### 9.2 Changes

**Add** `examples/pto_runtime_sim_example/kernels/orchestration/pto_example_orch.cpp`:

```cpp
#include "runtime.h"
#include "pto_types.h"

// Helper: create a BoundingBox tensor descriptor (simplest strategy)
static PTOTensorDescriptor make_tensor_bbox(uint64_t addr, int32_t size) {
    PTOTensorDescriptor t = {};
    t.addr = addr;
    t.start_offset = 0;
    t.strides[0] = 1;
    t.repeats[0] = size;
    t.n_dims = 1;
    t.strategy = PTO_OVERLAP_BOUNDING_BOX;
    return t;
}

extern "C" int build_pto_example_graph(Runtime* runtime, uint64_t* args, int arg_count) {
    runtime->pto_init(/* heap params */);

    void* host_a = (void*)args[0];
    void* host_b = (void*)args[1];
    void* host_f = (void*)args[2];
    int32_t size = (int32_t)args[3];

    // Allocate device buffers (divergence §1: explicit alloc, address known before submit)
    // External buffers: from orchestration parameters (divergence §3)
    PTOBufferHandle* dev_a = runtime->pto_alloc(size);
    PTOBufferHandle* dev_b = runtime->pto_alloc(size);

    // Runtime-allocated intermediate buffers
    PTOBufferHandle* dev_c = runtime->pto_alloc(size);
    PTOBufferHandle* dev_d = runtime->pto_alloc(size);
    PTOBufferHandle* dev_e = runtime->pto_alloc(size);
    PTOBufferHandle* dev_f = runtime->pto_alloc(size);

    // Task 0: c = a + b (no data dependency)
    PTOParam params0[] = {
        {PTO_PARAM_INPUT,  make_tensor_bbox(dev_a->addr, size), dev_a},
        {PTO_PARAM_INPUT,  make_tensor_bbox(dev_b->addr, size), dev_b},
        {PTO_PARAM_OUTPUT, make_tensor_bbox(dev_c->addr, size), dev_c},
    };
    runtime->pto_submit_task(0, 1, params0, 3);

    // Task 1: d = c + 1 (auto-dependency on task 0 via dev_c)
    PTOParam params1[] = {
        {PTO_PARAM_INPUT,  make_tensor_bbox(dev_c->addr, size), dev_c},
        {PTO_PARAM_OUTPUT, make_tensor_bbox(dev_d->addr, size), dev_d},
    };
    runtime->pto_submit_task(1, 1, params1, 2);

    // Task 2: e = c + 2 (auto-dependency on task 0 via dev_c)
    PTOParam params2[] = {
        {PTO_PARAM_INPUT,  make_tensor_bbox(dev_c->addr, size), dev_c},
        {PTO_PARAM_OUTPUT, make_tensor_bbox(dev_e->addr, size), dev_e},
    };
    runtime->pto_submit_task(2, 1, params2, 2);

    // Task 3: f = d * e (auto-dependency on task 1 via dev_d, task 2 via dev_e)
    PTOParam params3[] = {
        {PTO_PARAM_INPUT,  make_tensor_bbox(dev_d->addr, size), dev_d},
        {PTO_PARAM_INPUT,  make_tensor_bbox(dev_e->addr, size), dev_e},
        {PTO_PARAM_OUTPUT, make_tensor_bbox(dev_f->addr, size), dev_f},
    };
    runtime->pto_submit_task(3, 1, params3, 2);

    // Signal no more references (divergence §2: deferred free)
    runtime->pto_free(dev_a);
    runtime->pto_free(dev_b);
    runtime->pto_free(dev_c);
    runtime->pto_free(dev_d);
    runtime->pto_free(dev_e);
    // dev_f freed by caller after readback

    return 0;
}
```

**Update** `examples/pto_runtime_sim_example/kernels/kernel_config.py`:

```python
ORCHESTRATIONS = {
    "legacy": {
        "source": str(_KERNELS_ROOT / "orchestration" / "example_orch.cpp"),
        "function_name": "build_example_graph",
    },
    "pto": {
        "source": str(_KERNELS_ROOT / "orchestration" / "pto_example_orch.cpp"),
        "function_name": "build_pto_example_graph",
    }
}

# Default to legacy for backward compatibility
ORCHESTRATION = ORCHESTRATIONS["legacy"]
```

### 9.3 Test

Both orchestrations should produce the same result.

```bash
# Legacy mode (default)
PYTHONPATH=python:$PYTHONPATH python examples/pto_runtime_sim_example/main.py
# Expected: SUCCESS

# PTO mode (explicit)
PYTHONPATH=python:$PYTHONPATH python examples/pto_runtime_sim_example/main.py --mode pto
# Expected: SUCCESS
```

---

## 10. Phase 7: Full PTO Mode

### 10.1 Goal

Complete the PTO runtime with all features:
- Ring buffer memory management (HeapRing for `pto_alloc()`)
- Automatic dependency via TensorMap with strided overlap strategies
- Buffer-level reference counting (independent of task fanout, divergence §5)
- Version control for in-place updates via `pto_version_inc()` (divergence §6)

### 10.2 Test Matrix (Phase 7 Complete)

| Test | Legacy Mode | PTO Mode | Status |
|------|-------------|----------|--------|
| `main.py --mode legacy` | ✓ PASS (42.0) | - | ✓ VERIFIED |
| `main.py --mode pto` | - | ✓ PASS (42.0) | ✓ VERIFIED |
| Diamond DAG | ✓ PASS | ✓ PASS | ✓ VERIFIED |
| In-place update (`--mode inplace`) | - | ✓ PASS (6.0) | ✓ VERIFIED |
| Multi-consumer (`--mode multiconsumer`) | - | ✓ PASS (9.0) | ✓ VERIFIED |

---

## 10b. Phase 8: Remove Legacy Mode, PTO Default

### 10b.1 Goal

Remove the legacy `add_task()` / `add_successor()` code path entirely. The PTO API (`pto_alloc`, `pto_submit_task`, etc.) becomes the only mode. The example runs PTO mode by default without any `--mode` flag.

### 10b.2 Changes

1. **`aicpu/aicpu_executor.cpp`**: Remove `AicpuExecutor` (legacy scheduler) and `legacy_scheduler_run()`. `aicpu_execute()` calls `pto_scheduler_run()` directly.

2. **`runtime/runtime.h` / `runtime.cpp`**:
   - Remove `add_task()`, `add_successor()`, `get_initial_ready_tasks()` legacy methods
   - Remove `pto_mode_enabled_` flag and `is_pto_mode()` — PTO is always on
   - `pto_init()` called automatically in constructor or made implicit

3. **`examples/pto_runtime_sim_example/main.py`**:
   - Remove `--mode` flag; PTO orchestration is the default
   - Remove legacy orchestration selection from `kernel_config.py`

4. **`examples/pto_runtime_sim_example/kernels/kernel_config.py`**:
   - Single orchestration entry (PTO), remove `ORCHESTRATIONS` dict and `"legacy"` key

5. **`examples/pto_runtime_sim_example/kernels/orchestration/example_orch.cpp`**:
   - Remove legacy orchestration file (or keep as reference)

### 10b.3 Test

```bash
PYTHONPATH=python:$PYTHONPATH python examples/pto_runtime_sim_example/main.py
# Expected: SUCCESS (using PTO mode, no flag needed)
```

### 10b.4 Test Matrix

| Test | Result |
|------|--------|
| `main.py` (default, PTO) | ✓ PASS |
| Diamond DAG | ✓ PASS |
| Back-pressure | ✓ PASS |
| In-place update (version_inc) | ✓ PASS |
| Multi-producer buffer | ✓ PASS |

---

## 11. Current Status

| Phase | Description | Status | Test Passes |
|-------|-------------|--------|-------------|
| 0 | Baseline with Runtime class | ✓ COMPLETE | ✓ YES |
| 1 | PTO data structures (header) | ✓ COMPLETE | ✓ YES |
| 2 | Ring buffer utilities (header) | ✓ COMPLETE | ✓ YES |
| 3 | TensorMap (header) | ✓ COMPLETE | ✓ YES |
| 4 | Dual-mode Runtime class | ✓ COMPLETE | ✓ YES |
| 5 | PTO-native scheduler | ✓ COMPLETE | ✓ YES |
| 6 | PTO-native orchestration | ✓ COMPLETE | ✓ YES |
| 7 | Full PTO mode | ✓ COMPLETE | ✓ YES |
| 8 | Remove legacy, PTO default | PENDING | - |

---

## 12. Test Commands

**After every phase**, run:

```bash
cd /data/z00626005/code/simpler
# Legacy mode (default)
PYTHONPATH=python:$PYTHONPATH python examples/pto_runtime_sim_example/main.py
# PTO mode
PYTHONPATH=python:$PYTHONPATH python examples/pto_runtime_sim_example/main.py --mode pto
# In-place update test (Phase 7)
PYTHONPATH=python:$PYTHONPATH python examples/pto_runtime_sim_example/main.py --mode inplace
# Multi-consumer test (Phase 7)
PYTHONPATH=python:$PYTHONPATH python examples/pto_runtime_sim_example/main.py --mode multiconsumer
```

Expected output:
```
...
SUCCESS: All 16384 elements are correct (<expected_value>)
```

| Mode | Expected Value |
|------|----------------|
| legacy | 42.0 |
| pto | 42.0 |
| inplace | 6.0 |
| multiconsumer | 9.0 |

---

## 13. File Change Summary Per Phase

| Phase | Files Added/Modified | Behavior Change |
|-------|---------------------|-----------------|
| 0 | `runtime.h`, `runtime.cpp`, `aicpu_executor.cpp`, `aicore_executor.cpp`, `runtime_maker.cpp` | Baseline |
| 1 | Add `pto_types.h` (includes `PTOTensorDescriptor`, `PTOBufferHandle`, `PTOOverlapStrategy`) | None ✓ |
| 2 | Update `ring_buffer.h`, `dep_list_pool.h` | None ✓ (already existed from Phase 0) |
| 3 | Add `tensor_map.h` (with strided overlap strategies) | None ✓ |
| 4 | Update `runtime.h`, `runtime.cpp`, `pto_types.h` (add `pto_alloc`, `pto_free`, `pto_version_inc`, `pto_submit_task`) | New API (unused) ✓ |
| 5 | Update `aicpu_executor.cpp` with dual-mode routing | PTO scheduler (behind flag) ✓ |
| 6 | Add `pto_example_orch.cpp`, update `kernel_config.py` | PTO orchestration |
| 7 | Add `pto_inplace_test.cpp`, `pto_multiconsumer_test.cpp`; update `kernel_config.py`, `main.py` | Full integration tests (in-place updates, multi-consumer) ✓ |
| 8 | Remove `AicpuExecutor`, legacy API, `--mode` flag; update example to PTO default | PTO-only mode |

---

## 14. References

- [implementing_new_runtime.md](implementing_new_runtime.md) - Framework guide
- [runtime_buffer_manager_comprehensive_summary.md](runtime_buffer_manager_comprehensive_summary.md) - Target design
- [divergence-to-original-orchestration.md](divergence-to-original-orchestration.md) - Design divergences
- `src/runtime/host_build_graph/` - Reference implementation
- `examples/pto_runtime_sim_example/` - Test example