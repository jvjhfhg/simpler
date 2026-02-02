# PTO Runtime Correction Plan

This document outlines the corrections needed to the implementation from Plan #01, based on the fixes specified in `divergence-to-original-orchestration.md`.

---

## 1. Summary of Incorrect Assumptions in Plan #01

The following items in Plan #01 were based on incorrect divergence points:

| Original Divergence | What Was Implemented | Correction Needed |
|---------------------|---------------------|-------------------|
| §1 Explicit `pto_alloc()` | Added `pto_alloc()` API | **Remove** - allocation is implicit during task submit |
| §2 Explicit `pto_free()` | Added `pto_free()` API | **Remove** - no explicit free needed |
| §3 Deprecate scope-based lifecycle | Removed scope APIs | **Restore** - scope is the primary lifecycle mechanism |
| §4 Two buffer sources | Runtime-allocated + External | **Simplify** - only external (params) and implicit (output) |
| §5 Buffer-level ref counting | Independent buffer ref count | **Remove** - buffer lifetime = task lifetime (task fanout + scope) |

**Key clarification:** TensorMap is **kept** for automatic dependency detection. What's removed is the idea of buffer-level reference counting *independent* of task fanout. Buffer lifetime is simply tied to its producer task's lifetime.

---

## 2. Correct Design Principles

### 2.1 Memory Allocation is Implicit

**Wrong (Plan #01):**
```cpp
// Orchestration explicitly allocates before submit
PTOBufferHandle* dev_c = runtime->pto_alloc(size);
pto_submit_task(..., PTO_OUTPUT(dev_c->addr, size), ...);
```

**Correct:**
```cpp
// Runtime allocates during pto_submit_task()
void* output_ptr = nullptr;  // Will be filled by runtime
pto_submit_task(..., PTO_OUTPUT(&output_ptr, size), ...);
// output_ptr now contains the allocated address
```

**Rationale:** Memory allocation should NOT be sequential. The runtime schedules allocation dynamically with tasks, enabling back-pressure when heap is full.

### 2.2 Scope Controls Buffer Lifetime

**Wrong (Plan #01):**
```cpp
// Explicit alloc/free pairs
PTOBufferHandle* buf = pto_alloc(size);
// ... use buf ...
pto_free(buf);  // Signal no more references
```

**Correct:**
```cpp
pto_scope_begin(rt);
    pto_submit_task(..., PTO_OUTPUT(&buf, size), ...);
    // buf is allocated implicitly, lifetime managed by scope
pto_scope_end(rt);  // Decrements fanout for buffers in scope
```

**Rationale:** Scope-based lifecycle with fanout initialized to `scope_depth` provides automatic memory reclamation without explicit free calls.

### 2.3 INOUT is NOT a Producer

**Wrong (Plan #01):**
```cpp
// Treated INOUT same as OUTPUT for producer tracking
if (param.type == PTO_PARAM_OUTPUT || param.type == PTO_PARAM_INOUT) {
    tensormap_insert(...);  // Register as producer - WRONG for INOUT
}
```

**Correct:**
```cpp
// INOUT creates a dependency but does NOT produce a new buffer
if (param.type == PTO_PARAM_INPUT) {
    // Lookup producer via TensorMap, add to fanin
    int32_t producer = tensormap_lookup(...);
    add_fanin(task_id, producer);
} else if (param.type == PTO_PARAM_INOUT) {
    // Lookup producer via TensorMap (like INPUT)
    int32_t producer = tensormap_lookup(...);
    add_fanin(task_id, producer);
    // Does NOT register as new producer - just updates version for next reader
} else if (param.type == PTO_PARAM_OUTPUT) {
    // Allocate new buffer from HeapRing
    // Register in TensorMap as producer
    tensormap_insert(..., task_id, ...);
}
```

**Rationale:** INOUT modifies an existing buffer in-place. It depends on the previous writer but doesn't create a new buffer address. TensorMap is still used for dependency detection.

### 2.4 Buffer Lifetime = Task Lifetime (No Separate Buffer Ref Count)

**Wrong (divergence §5 / Plan #01):**
```cpp
// Separate buffer-level reference counting
struct PTOBufferHandle {
    int32_t ref_count;  // Independent of task fanout - WRONG
};
```

**Correct:**
```cpp
// Buffer lifetime is tied to producer task lifetime
// When task transitions to CONSUMED, its output buffers are freed
// No separate buffer ref count needed

struct TaskDescriptor {
    int32_t fanout_count;    // Consumers + scope references
    int32_t fanout_refcount; // Decremented by consumers + scope_end
    // When fanout_refcount == fanout_count && state == COMPLETED
    //   → task becomes CONSUMED → output buffers freed
};
```

**Rationale:** A buffer's lifetime is exactly its producer task's lifetime. When all consumers finish and all scopes exit, the task becomes CONSUMED and its packed output buffer is implicitly freed when `last_task_alive` advances.

### 2.5 Single Allocation Per Task (Packed Outputs)

**Correct (already in comprehensive summary, verify implementation):**
```cpp
// Task with multiple outputs gets ONE allocation
// Runtime calculates total size, allocates once
void* packed_buffer = heap_ring_alloc(total_output_size);

// Individual outputs are offsets within packed buffer
output_A = packed_buffer + offset_A;
output_B = packed_buffer + offset_B;
```

**Rationale:** Reduces allocation overhead, ensures outputs are contiguous for DMA efficiency.

### 2.6 Version Control for In-Place Updates

**Correct (keep from Plan #01, but clarify semantics):**
```cpp
// Version increment creates a new logical version, NOT a new buffer
PTOVersion v1 = pto_version_inc(buffer);
// v1 refers to same address as buffer, but different version number
// Write to v1 waits for all reads of previous version to complete
```

**Rationale:** SSA-style versioning enables in-place updates with correct dependency ordering.

### 2.7 Output Buffer Size Limits Scheduling

**New addition:**
```cpp
// During pto_submit_task:
int32_t total_output_size = calculate_output_size(params);
void* buffer = heap_ring_alloc(total_output_size, &heap_tail);
// ^ This may STALL if heap is full (back-pressure)

// Scheduler advances heap_tail when tasks complete
// Orchestrator can only proceed when space is available
```

**Rationale:** Large output buffers naturally limit how many tasks can be in-flight, providing automatic back-pressure.

---

## 3. Implementation Phases

### Phase C1: Remove Explicit Alloc/Free APIs

**Files to modify:**
- `runtime/runtime.h` - Remove `pto_alloc()`, `pto_free()` declarations
- `runtime/runtime.cpp` - Remove implementations
- `runtime/pto_types.h` - Remove `PTOBufferHandle` if only used for explicit alloc

**Changes:**
1. Delete `pto_alloc()` method and implementation
2. Delete `pto_free()` method and implementation
3. Remove `buffer_handles_[]` and `buffer_handle_count_` members
4. Update any code that uses these APIs

**Test:** Example should fail to compile if it uses explicit alloc/free

### Phase C2: Restore Scope-Based Lifecycle

**Files to modify:**
- `runtime/runtime.h` - Add `pto_scope_begin()`, `pto_scope_end()`
- `runtime/runtime.cpp` - Implement scope stack management
- `runtime/pto_types.h` - Add scope-related types if needed

**Implementation:**
```cpp
class Runtime {
    // Scope stack (Orchestrator-private)
    int32_t scope_stack_[PTO_MAX_SCOPE_DEPTH];
    int32_t scope_stack_top_ = 0;

public:
    void pto_scope_begin() {
        scope_stack_[scope_stack_top_++] = current_task_index_;
    }

    void pto_scope_end() {
        int32_t begin_pos = scope_stack_[--scope_stack_top_];
        int32_t end_pos = current_task_index_;

        // Decrement fanout for all tasks in [begin_pos, end_pos)
        for (int32_t i = begin_pos; i < end_pos; i++) {
            fanout_refcount_[i]++;
            check_consumed(i);
        }
    }
};
```

**Test:** Example should work with scope-based buffer management

### Phase C3: Fix INOUT Semantics and Remove Buffer Ref Count

**Files to modify:**
- `runtime/runtime.cpp` - Fix `pto_submit_task()` INOUT handling
- `runtime/pto_types.h` - Remove `PTOBufferHandle.ref_count` if present

**Key clarification:** TensorMap is **kept** for automatic dependency detection. We're fixing:
1. INOUT should not register as producer in TensorMap
2. Remove any separate buffer-level reference counting (buffer lifetime = task lifetime)

**Changes:**
```cpp
// In pto_submit_task:
for (int i = 0; i < param_count; i++) {
    if (params[i].type == PTO_PARAM_INPUT) {
        // Lookup producer via TensorMap, add to fanin
        int32_t producer = tensormap_lookup(&tensor_map_, &params[i].tensor, last_task_alive);
        if (producer >= 0) {
            add_fanin(task_id, producer);
            add_fanout(producer, task_id);  // Producer's fanout grows
        }
    } else if (params[i].type == PTO_PARAM_INOUT) {
        // INOUT: Lookup producer (like INPUT)
        int32_t producer = tensormap_lookup(&tensor_map_, &params[i].tensor, last_task_alive);
        if (producer >= 0) {
            add_fanin(task_id, producer);
            add_fanout(producer, task_id);
        }
        // DO NOT call tensormap_insert() - INOUT is not a new producer
    } else if (params[i].type == PTO_PARAM_OUTPUT) {
        // OUTPUT: Allocate from HeapRing and register as producer
        // ... allocation code ...
        tensormap_insert(&tensor_map_, &params[i].tensor, task_id, version);
    }
}
```

**Test:**
1. INOUT operations should create correct dependencies without registering as new producer
2. No separate buffer ref count - buffer freed when producer task becomes CONSUMED

### Phase C4: Update Orchestration Examples

**Files to modify:**
- `examples/pto_runtime_sim_example/kernels/orchestration/pto_example_orch.cpp`

**Changes:**
```cpp
// Before (Plan #01 style - WRONG):
PTOBufferHandle* dev_c = runtime->pto_alloc(size);
PTOParam params[] = {
    {PTO_PARAM_OUTPUT, make_tensor(dev_c->addr, size), dev_c},
};
runtime->pto_submit_task(...);
runtime->pto_free(dev_c);

// After (Correct style):
extern "C" int build_pto_example_graph(Runtime* runtime, uint64_t* args, int arg_count) {
    void* host_a = (void*)args[0];
    void* host_b = (void*)args[1];
    void* host_f = (void*)args[2];
    int32_t size = (int32_t)args[3];

    // External buffers from parameters (passed in by caller)
    void* dev_a = (void*)args[4];  // Pre-allocated by host
    void* dev_b = (void*)args[5];

    runtime->pto_scope_begin();

    // Output buffers allocated implicitly by runtime
    void* dev_c = nullptr;
    void* dev_d = nullptr;
    void* dev_e = nullptr;
    void* dev_f = nullptr;

    // Task 0: c = a + b
    PTOParam params0[] = {
        {PTO_PARAM_INPUT,  make_tensor(dev_a, size), nullptr},
        {PTO_PARAM_INPUT,  make_tensor(dev_b, size), nullptr},
        {PTO_PARAM_OUTPUT, make_tensor(&dev_c, size), nullptr},  // &dev_c: pointer-to-pointer
    };
    runtime->pto_submit_task(0, 1, params0, 3);
    // dev_c now contains allocated address

    // Task 1: d = c + 1
    PTOParam params1[] = {
        {PTO_PARAM_INPUT,  make_tensor(dev_c, size), nullptr},
        {PTO_PARAM_OUTPUT, make_tensor(&dev_d, size), nullptr},
    };
    runtime->pto_submit_task(1, 1, params1, 2);

    // Task 2: e = c + 2
    PTOParam params2[] = {
        {PTO_PARAM_INPUT,  make_tensor(dev_c, size), nullptr},
        {PTO_PARAM_OUTPUT, make_tensor(&dev_e, size), nullptr},
    };
    runtime->pto_submit_task(2, 1, params2, 2);

    // Task 3: f = d * e
    PTOParam params3[] = {
        {PTO_PARAM_INPUT,  make_tensor(dev_d, size), nullptr},
        {PTO_PARAM_INPUT,  make_tensor(dev_e, size), nullptr},
        {PTO_PARAM_OUTPUT, make_tensor(&dev_f, size), nullptr},
    };
    runtime->pto_submit_task(3, 1, params3, 3);

    runtime->pto_scope_end();  // Decrements fanout for all buffers

    return 0;
}
```

**Test:** Example should pass with scope-based lifecycle

### Phase C5: Add Back-Pressure for Output Size

**Files to modify:**
- `runtime/runtime.cpp` - Verify heap_ring_alloc stalls when full
- `aicpu/aicpu_executor.cpp` - Verify scheduler advances heap_tail

**Verification:**
1. `heap_ring_alloc()` should spin/stall when insufficient space
2. Scheduler should advance `heap_tail` when tasks complete
3. Large output buffers should naturally limit in-flight tasks

**Test:** Submit tasks with large outputs, verify back-pressure works

---

## 4. API Changes Summary

### Removed APIs (from Plan #01):
```cpp
// DELETE THESE:
PTOBufferHandle* pto_alloc(int32_t size);
void pto_free(PTOBufferHandle* handle);

// Also remove from PTOBufferHandle if present:
int32_t ref_count;  // No separate buffer ref count
```

### Restored APIs:
```cpp
// ADD THESE BACK:
void pto_scope_begin();
void pto_scope_end();
```

### Kept APIs (unchanged):
```cpp
// TensorMap for automatic dependency detection - KEPT
tensormap_insert(...);  // Register OUTPUT as producer
tensormap_lookup(...);  // Find producer for INPUT/INOUT

// Task submission - KEPT
int pto_submit_task(int32_t func_id, int32_t worker_type,
                    PTOParam* params, int32_t param_count);

// Version control for in-place updates - KEPT
PTOBufferHandle* pto_version_inc(PTOBufferHandle* handle);
```

### Modified APIs:
```cpp
// PTOParam for OUTPUT now uses pointer-to-pointer for address
struct PTOParam {
    PTOParamType type;
    PTOTensorDescriptor tensor;
    // For OUTPUT: tensor.addr is void** (pointer to receive allocated address)
    // For INPUT/INOUT: tensor.addr is void* (actual buffer address)
};

// INOUT handling changed:
// - Still uses TensorMap lookup to find producer (creates dependency)
// - NO LONGER calls tensormap_insert (not a new producer)
```

---

## 5. Test Matrix

| Test | Before Correction | After Correction |
|------|-------------------|------------------|
| Explicit alloc/free | Compiles, runs | Compile error (removed) |
| Scope-based lifecycle | Not available | Works correctly |
| Diamond DAG | Works | Works |
| INOUT as producer | Registers in TensorMap (wrong) | Only creates dependency (correct) |
| Buffer ref count | Separate from task | Tied to task lifetime |
| TensorMap dependency detection | Works | Works (unchanged) |
| Large output back-pressure | May OOM | Stalls correctly |

---

## 6. Implementation Order

1. **Phase C1** - Remove explicit alloc/free (breaking change to Plan #01 API)
2. **Phase C2** - Restore scope APIs (enable correct lifecycle)
3. **Phase C3** - Fix INOUT semantics + remove buffer ref count (TensorMap kept for dependency detection)
4. **Phase C4** - Update examples (use correct API)
5. **Phase C5** - Verify back-pressure (performance validation)

Each phase should pass the example test before proceeding.

---

## 7. Key Clarifications

### What's KEPT:
- **TensorMap** - for automatic dependency detection via buffer overlap
- **Task fanin/fanout** - for task dependency tracking and buffer lifecycle
- **Scope-based lifecycle** - fanout initialized to scope_depth

### What's REMOVED:
- **Explicit `pto_alloc()`/`pto_free()`** - allocation is implicit during task submit
- **Separate buffer ref count** - buffer lifetime = producer task lifetime
- **INOUT as producer** - INOUT creates dependency but doesn't register new producer

### Core Principle:
Buffer lifetime is tied to its producer task's lifetime. When a task becomes CONSUMED (all consumers done + all scopes exited), its packed output buffer is freed. No separate buffer-level reference counting is needed.

---

## 8. References

- [divergence-to-original-orchestration.md](../divergence-to-original-orchestration.md) - Source of corrections
- [runtime_buffer_manager_comprehensive_summary.md](../runtime_buffer_manager_comprehensive_summary.md) - Target design
- [#01-pto_runtime_implementation_plan.md](#01-pto_runtime_implementation_plan.md) - Original (incorrect) plan