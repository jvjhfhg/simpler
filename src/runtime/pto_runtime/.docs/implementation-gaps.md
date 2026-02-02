# PTO Runtime V2 Implementation Gaps

This document identifies gaps between the design in `key-point-summary.md` and the current implementation in `src/runtime/pto_runtime/`.

---

## Summary

The current implementation covers the basic framework but has significant gaps in the ring buffer-based memory management and lifecycle tracking systems described in the design.

| Category | Status |
|----------|--------|
| Core Scheduling | Mostly Complete |
| Ring Buffer Architecture | Not Integrated |
| Lifecycle Management | Partially Implemented |
| Back-pressure Flow Control | Not Implemented |

---

## 1. Task Ring Buffer

**Status**: NOT IMPLEMENTED

| Design (key-point-summary.md) | Current Implementation |
|-------------------------------|------------------------|
| Ring buffer for `PTOTaskDescriptor` slots | Fixed array `Task tasks[RUNTIME_MAX_TASKS]` in `runtime.h:133` |
| O(1) bump pointer allocation | Simple `next_task_id++` counter |
| Blocks when `tasks_in_flight >= TASK_WINDOW_SIZE` | No back-pressure |
| Implicit deallocation via tail advance | No reclamation mechanism |

**Gap**: The `TaskRing` structure exists in `ring_buffer.h` but is **not integrated** into the Runtime class.

**Impact**: Cannot limit in-flight tasks; no automatic task slot reclamation.

---

## 2. GM Heap (HeapRing)

**Status**: NOT IMPLEMENTED

| Design | Current Implementation |
|--------|------------------------|
| Ring buffer for GM allocation | Uses `host_api.device_malloc()` in `runtime.cpp:270` |
| Blocks when insufficient contiguous space | No back-pressure |
| Scheduler advances `heap_tail` when tasks complete | No heap lifecycle tracking |
| Packed output buffers per task | Individual allocations per OUTPUT param |

**Gap**: The `HeapRing` structure exists in `ring_buffer.h` but is **not integrated**. Memory allocation does not provide back-pressure.

**Impact**:
- Cannot limit memory consumption
- No automatic memory reclamation
- Potential OOM instead of graceful stall

---

## 3. Task State Machine

**Status**: PARTIALLY IMPLEMENTED

| State | Design | Current Implementation |
|-------|--------|------------------------|
| PENDING | Task submitted, fanin not satisfied | Implicit via `fanin > 0` |
| READY | All deps satisfied, in ready queue | Implicit via `fanin == 0` |
| RUNNING | Executing on worker | Via `Handshake.task_status = 1` |
| COMPLETED | Execution finished | Via `Handshake.task_status = 0` |
| CONSUMED | All consumers done, buffers freed | **NOT IMPLEMENTED** |

**Gap**: No explicit state tracking. Missing `CONSUMED` state transition and associated buffer reclamation.

**Impact**: Cannot determine when task output buffers are safe to free.

---

## 4. Scope Management

**Status**: PARTIALLY IMPLEMENTED

| Design | Current Implementation |
|--------|------------------------|
| `fanout_count` initialized to `scope_depth` | Not implemented |
| `scope_end()` decrements fanout for tasks in range | Only logs message, no actual decrement (`runtime.cpp:237`) |
| Buffer freed when `fanout_count == 0` | Not implemented |

**Gap**: Scope stack exists but **fanout decrement logic is stubbed out** with comment: "For simulation, tasks are cleaned up at the end of execution."

**Location**: `runtime.cpp:224-239`

```cpp
void Runtime::pto_scope_end() {
    // ...
    // Note: In full implementation, this would decrement fanout_refcount
    // for all tasks in [begin_pos, end_pos) and check for CONSUMED transition.
    // For simulation, tasks are cleaned up at the end of execution.
}
```

**Impact**: Scope-based buffer lifecycle not enforced; all cleanup deferred to end.

---

## 5. Fanout Reference Counting

**Status**: NOT IMPLEMENTED

| Design | Current Implementation |
|--------|------------------------|
| `fanout_refcount` tracks consumer references | Only `fanin` is tracked |
| Decrement when consumer completes | Not implemented |
| Decrement when scope exits | Not implemented |
| Buffer freed when reaches zero | Not implemented |

**Gap**: The Task structure in `runtime.h:104-116` has `fanout[]` array but no `fanout_refcount` counter.

**Required Changes**:
```cpp
typedef struct {
    // ... existing fields ...
    int fanout_refcount;  // ADD: initialized to scope_depth + consumer_count
} Task;
```

**Impact**: Cannot track when all consumers have finished using a buffer.

---

## 6. DepListPool

**Status**: NOT USED

| Design | Current Implementation |
|--------|------------------------|
| Ring buffer for linked list nodes | Defined in `dep_list_pool.h` |
| Used for fanin/fanout dependency lists | **Not integrated** |
| O(1) prepend operation | N/A |

**Gap**: Header exists but completely unused. Current implementation uses fixed-size arrays:
- `fanout[RUNTIME_MAX_FANOUT]` limits fanout to 512

**Impact**: Fixed fanout limit; wastes memory for tasks with few dependents.

---

## 7. PTOSharedHeader

**Status**: NOT IMPLEMENTED

| Design | Current Implementation |
|--------|------------------------|
| `current_task_index` (Orchestrator writes) | Not present |
| `heap_top` / `heap_tail` pointers | Not present |
| `last_task_alive` for TensorMap staleness | Hardcoded to `0` in `runtime.cpp:302` |
| Cache-line aligned for false sharing prevention | N/A |

**Gap**: The `PTOSharedHeader` is defined in `pto_runtime.h` but not used for Orchestrator-Scheduler communication.

**Impact**: No decoupled communication channel; TensorMap never invalidates stale entries.

---

## 8. TensorMap Staleness

**Status**: PARTIALLY IMPLEMENTED

| Design | Current Implementation |
|--------|------------------------|
| Entries valid only if `producer_task_id >= last_task_alive` | `last_task_alive` always passed as `0` |
| Stale entries automatically ignored | Works but never filters |

**Location**: `runtime.cpp:302`
```cpp
int32_t producer = tensormap_lookup(&tensor_map_, &params[i].tensor, 0);  // Always 0
```

**Gap**: Lazy invalidation mechanism exists but is disabled.

**Impact**: TensorMap pool may fill with stale entries over long-running orchestrations.

---

## 9. Worker Types

**Status**: PARTIALLY IMPLEMENTED

| Worker Type | Design | Current Implementation |
|-------------|--------|------------------------|
| AICore_CUBE | Matrix ops | Implemented as `CoreType::AIC` |
| AICore_VECTOR | Vector ops | Implemented as `CoreType::AIV` |
| AI_CPU (AICPU) | Scalar ops, control flow | AICPU is scheduler only, **not a worker** |
| Accelerators | DMA, fixed-function | **Not implemented** |

**Gap**: AICPU and Accelerator worker types are not available for task dispatch.

**Impact**: Cannot offload scalar operations or DMA transfers as tasks.

---

## 10. Back-Pressure Flow Control

**Status**: NOT IMPLEMENTED

| Design | Current Implementation |
|--------|------------------------|
| Task ring blocks when full | No limit on task submissions |
| Heap ring blocks when insufficient space | Uses malloc, no limit |
| Ready queue blocks when full | Ready queue can grow unbounded |
| Output buffer size limits scheduling | No memory-based scheduling constraints |

**Impact**:
- No graceful degradation under memory pressure
- Potential OOM crashes instead of stalls
- No automatic pacing of task submission

---

## 11. Packed Output Buffers

**Status**: NOT IMPLEMENTED

| Design | Current Implementation |
|--------|------------------------|
| Single allocation per task for all outputs | Individual allocation per OUTPUT param |
| `packed_buffer_offset` + `packed_buffer_size` in task | Not present in Task struct |
| Contiguous outputs for DMA efficiency | Scattered allocations |

**Gap**: Each OUTPUT parameter calls `device_malloc()` separately.

**Impact**:
- More allocation overhead
- Potential memory fragmentation
- Less DMA-friendly layout

---

## Correction Plan (#02) Status

| Phase | Description | Status |
|-------|-------------|--------|
| C1 | Remove explicit `pto_alloc()`/`pto_free()` | Done |
| C2 | Restore scope-based lifecycle | Partial (scope exists, fanout stubbed) |
| C3 | Fix INOUT semantics (not a producer) | Done (`runtime.cpp:306-313`) |
| C4 | Update orchestration examples | Unknown |
| C5 | Verify back-pressure for output size | Not implemented |

---

## Priority Recommendations

### High Priority
1. **Integrate HeapRing** - Required for memory back-pressure and automatic reclamation
2. **Implement fanout_refcount** - Required for CONSUMED state and buffer lifecycle

### Medium Priority
3. **Complete scope_end() logic** - Fanout decrement for scope-based lifecycle
4. **Integrate TaskRing** - Task window management and back-pressure
5. **Wire up PTOSharedHeader** - Orchestrator-Scheduler decoupling

### Low Priority
6. **Integrate DepListPool** - Replace fixed-size fanout arrays
7. **Add AICPU/Accelerator workers** - Extended worker types
8. **Implement packed output buffers** - Memory layout optimization

---

## File References

| File | Contains |
|------|----------|
| `runtime/runtime.h` | Runtime class, Task struct |
| `runtime/runtime.cpp` | PTO API implementation |
| `runtime/ring_buffer.h` | TaskRing, HeapRing (unused) |
| `runtime/dep_list_pool.h` | DepListPool (unused) |
| `runtime/tensor_map.h` | TensorMap (used) |
| `runtime/pto_runtime.h` | PTOSharedHeader, PTOTaskDescriptor |
| `runtime/pto_types.h` | PTOWorkerType, PTOParam, PTOBufferHandle |
| `aicpu/aicpu_executor.cpp` | PtoScheduler, ready queues |
| `aicore/aicore_executor.cpp` | AICore worker execution |

---

## References

- [key-point-summary.md](key-point-summary.md) - Design summary
- [historical-plan/#01-pto_runtime_implementation_plan.md](historical-plan/#01-pto_runtime_implementation_plan.md) - Original plan
- [historical-plan/#02-pto_runtime_correction_plan.md](historical-plan/#02-pto_runtime_correction_plan.md) - Correction plan