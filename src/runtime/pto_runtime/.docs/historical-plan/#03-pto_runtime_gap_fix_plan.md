# PTO Runtime V2 Gap Fix Plan

This document describes a phased implementation plan to close the 11 gaps identified in `implementation-gaps.md` between the design in `runtime_buffer_manager_comprehensive_summary.md` and the current implementation.

---

## Dependency Graph

Before phasing, the dependency order between gaps:

```
Gap 5 (fanout_refcount)  ──────────────────────┐
                                                 ├─► Gap 4 (scope_end logic)
Gap 3 (TaskState machine) ──────────────────────┘
                                                 │
Gap 1 (TaskRing) ───────────────────────────────►│
                                                 ├─► Gap 7 (PTOSharedHeader)
Gap 2 (HeapRing) ───────────────────────────────►│
                                                 │
Gap 7 (PTOSharedHeader) ────────────────────────►├─► Gap 8 (TensorMap staleness)
                                                 │
Gap 7 (PTOSharedHeader) ────────────────────────►├─► Gap 10 (Back-pressure)
                                                 │
Gap 5 (fanout_refcount) ────────────────────────►├─► Gap 11 (Packed output buffers)
Gap 2 (HeapRing) ───────────────────────────────►│
                                                 │
Gap 6 (DepListPool) ── standalone, replaces fixed arrays
Gap 9 (Worker Types) ── standalone, extends scheduler
```

---

## Current Progress

| Phase | Status | Date |
|-------|--------|------|
| Phase 1: Task State Machine & Fanout Reference Counting | ✅ Completed | 2026-02-02 |
| Phase 2: Scope End Logic | ⏳ Pending | - |
| Phase 3: TaskRing and HeapRing Integration | ⏳ Pending | - |
| Phase 4: PTOSharedHeader and TensorMap Staleness | ⏳ Pending | - |
| Phase 5: Back-Pressure Flow Control | ⏳ Pending | - |
| Phase 6: DepListPool Integration | ⏳ Pending | - |
| Phase 7: Extended Worker Types | ⏳ Pending | - |

---

## Phase 1: Task State Machine and Fanout Reference Counting ✅ COMPLETED

**Status:** ✅ Implemented and tested (2026-02-02)
- All 52 tests passed
- State transitions: PENDING → READY → RUNNING → COMPLETED → CONSUMED working correctly
- Fanout reference counting properly tracks consumer completions
- Commit: `57d38c4`

**Gaps addressed:** Gap 3 (Task State Machine), Gap 5 (Fanout Reference Counting)

**Rationale:** These are the most fundamental missing primitives. Nearly every subsequent phase depends on explicit task state tracking and `fanout_refcount`. They can be added without changing the ring buffer architecture.

### Before

- Task state is implicit: `fanin > 0` = pending, `fanin == 0` = ready, handshake `task_status` tracks running/completed. No CONSUMED state.
- `Task` struct (`runtime.h:104-116`) has `fanout_count` and `fanout[]` but no `fanout_refcount`.
- Scheduler (`aicpu_executor.cpp`) only decrements `fanin` on completion. Never touches fanout refcounts.

### After

- Each task has an explicit `TaskState` field: `PENDING → READY → RUNNING → COMPLETED → CONSUMED`.
- Each task has `fanout_refcount` initialized to 0, incremented by `scope_end()` and consumer completions.
- When `fanout_refcount == fanout_count` AND `state == COMPLETED`, the task transitions to CONSUMED.

### Files and Changes

**`runtime/runtime.h`**
- Add `#include "pto_runtime.h"` (for `TaskState` enum class at `pto_runtime.h:90-96`).
- Add to `Task` struct:
  ```cpp
  TaskState state;              // Explicit task state (default: PENDING)
  int fanout_refcount;          // Completed consumers + scope_end count
  int fanin_producers[RUNTIME_MAX_ARGS];  // Temporary: reverse dep list (replaced by DepListPool in Phase 6)
  int fanin_producer_count;               // Temporary: count of producers
  ```
- Add to `Runtime` class:
  ```cpp
  int32_t last_task_alive_;     // Oldest non-CONSUMED task
  ```

**`runtime/runtime.cpp`**
- In `add_task()`: initialize `task->state = TaskState::PENDING`, `task->fanout_refcount = 0`, `task->fanin_producer_count = 0`.
- In `add_successor(from, to)`: record `to->fanin_producers[to->fanin_producer_count++] = from`.
- In `pto_submit_task()`: after wiring deps, if `fanin == 0`, set `task->state = TaskState::READY`.
- Add helper `check_consumed(int task_id)`: if `fanout_refcount == fanout_count && state == COMPLETED`, transition to CONSUMED.
- Initialize `last_task_alive_ = 0` in constructor.

**`aicpu/aicpu_executor.cpp`**
- On task completion: set `task->state = TaskState::COMPLETED`.
- On dispatch to core: set `task->state = TaskState::RUNNING`.
- After decrementing downstream fanin: if fanin reaches 0, set `dep->state = TaskState::READY`.
- After waking consumers: for each producer in `task->fanin_producers[]`, increment `producer->fanout_refcount++` and check CONSUMED.

### Testability

- Print state at each transition. Verify PENDING → READY → RUNNING → COMPLETED flow.
- CONSUMED fires only after Phase 2 (scope_end logic). Print count of CONSUMED tasks at end (should be 0 until Phase 2).

---

## Phase 2: Scope End Logic with Fanout Decrement

**Gaps addressed:** Gap 4 (Scope Management)

**Rationale:** With `fanout_refcount` and `TaskState` from Phase 1, implement real `scope_end()` logic.

### Before

- `pto_scope_end()` (`runtime.cpp:224-239`) pops scope stack and logs a message. No fanout decrement.
- `fanout_count` is set only by `add_successor()`. Does NOT include scope_depth.

### After

- During `pto_submit_task()`, `fanout_count += scope_stack_top_` (add current scope depth).
- `pto_scope_end()` iterates `[begin_pos, end_pos)` and increments `fanout_refcount` for each task, checking CONSUMED transition.
- Tasks whose scopes have all exited and all consumers have finished → CONSUMED.

### Files and Changes

**`runtime/runtime.cpp`**
- In `pto_submit_task()`, after `add_task()` and dependency wiring:
  ```cpp
  tasks[task_id].fanout_count += scope_stack_top_;
  ```
- Replace stub in `pto_scope_end()`:
  ```cpp
  void Runtime::pto_scope_end() {
      int32_t begin_pos = scope_stack_[--scope_stack_top_];
      int32_t end_pos = next_task_id;
      for (int32_t i = begin_pos; i < end_pos; i++) {
          tasks[i].fanout_refcount++;
          if (tasks[i].fanout_refcount == tasks[i].fanout_count &&
              tasks[i].state == TaskState::COMPLETED) {
              tasks[i].state = TaskState::CONSUMED;
              // Future: advance last_task_alive_, reclaim heap
          }
      }
  }
  ```

### Testability

- Run diamond DAG with wrapping scope. After `scope_end()`, tasks that completed should transition to CONSUMED.
- Count CONSUMED tasks at end of execution. For a 4-task diamond DAG with 1 scope, all 4 should eventually reach CONSUMED.
- Without `scope_end()`, tasks remain COMPLETED (not CONSUMED).

---

## Phase 3: TaskRing and HeapRing Integration

**Gaps addressed:** Gap 1 (Task Ring Buffer), Gap 2 (GM Heap / HeapRing), Gap 11 (Packed Output Buffers)

**Rationale:** With lifecycle tracking working (state machine + scope_end), replace fixed array and malloc-based allocation with ring buffers. Packed output buffers are naturally part of HeapRing integration.

### Before

- `Task tasks[RUNTIME_MAX_TASKS]` fixed array with `next_task_id++` (`runtime.h:133`).
- Output buffers allocated individually via `host_api.device_malloc()` per OUTPUT param (`runtime.cpp:270`).
- `ring_buffer.h` defines `TaskRing` and `HeapRing` with complete implementations but never instantiated.

### After

- `Runtime` holds a `TaskRing` and `HeapRing`.
- Task allocation via `task_ring_alloc()` which returns slot index, stalls if full.
- Output allocation calculates total size, calls `heap_ring_alloc()` once, assigns sub-offsets within packed buffer.
- `packed_buffer_offset` and `packed_buffer_size` stored per task.

### Files and Changes

**`runtime/runtime.h`**
- Add `#include "ring_buffer.h"`.
- Add to Runtime class:
  ```cpp
  TaskRing task_ring_;
  PTOTaskDescriptor task_descriptors_[PTO_TASK_WINDOW_SIZE];
  HeapRing heap_ring_;
  ```
- Add to Task struct:
  ```cpp
  int32_t packed_buffer_offset;  // Offset in HeapRing
  int32_t packed_buffer_size;    // Total packed output size
  ```
- **Compatibility:** Keep old `Task tasks[]` temporarily as shadow copy for scheduler. Copy fields from `PTOTaskDescriptor` after allocation. Remove shadow in future cleanup.

**`runtime/runtime.cpp`**
- In `pto_init()`:
  ```cpp
  task_ring_init(&task_ring_, task_descriptors_, PTO_TASK_WINDOW_SIZE);
  char* heap_base = (char*)host_api.device_malloc(PTO_HEAP_SIZE);
  heap_ring_init(&heap_ring_, heap_base, PTO_HEAP_SIZE);
  ```
- In `pto_submit_task()` OUTPUT allocation: replace individual `device_malloc()` with:
  ```cpp
  // Calculate total output size
  int32_t total_output_size = 0;
  for (int32_t i = 0; i < param_count; i++) {
      if (params[i].type == PTOParamType::OUTPUT && params[i].buffer->addr == 0) {
          total_output_size += ALIGN_UP(params[i].buffer->size, PTO_ALIGNMENT);
      }
  }
  // Single allocation from HeapRing
  void* packed_base = nullptr;
  if (total_output_size > 0) {
      int32_t dummy_tail = 0;  // Real back-pressure in Phase 5
      packed_base = heap_ring_alloc(&heap_ring_, total_output_size, &dummy_tail);
  }
  // Assign sub-offsets
  int32_t offset = 0;
  for (int32_t i = 0; i < param_count; i++) {
      if (params[i].type == PTOParamType::OUTPUT && params[i].buffer->addr == 0) {
          params[i].buffer->addr = (uint64_t)((char*)packed_base + offset);
          offset += ALIGN_UP(params[i].buffer->size, PTO_ALIGNMENT);
      }
  }
  // Record on task
  tasks[task_id].packed_buffer_offset = heap_ring_offset(&heap_ring_, packed_base);
  tasks[task_id].packed_buffer_size = total_output_size;
  ```

### Testability

- Verify output addresses are contiguous (packed) per task.
- Print heap ring top before/after submission. Verify it advances by total output size.
- Test with artificially small `PTO_HEAP_SIZE` to exercise wrap-around.

---

## Phase 4: PTOSharedHeader and TensorMap Staleness

**Gaps addressed:** Gap 7 (PTOSharedHeader), Gap 8 (TensorMap Staleness)

**Rationale:** With TaskRing and HeapRing in place, wire up PTOSharedHeader for Orchestrator ↔ Scheduler communication. This enables TensorMap staleness filtering.

### Before

- `PTOSharedHeader` defined (`pto_runtime.h:115-127`) but never instantiated.
- `tensormap_lookup()` always receives `last_task_alive = 0` (`runtime.cpp:302`), staleness filtering disabled.

### After

- `Runtime` holds `PTOSharedHeader` instance.
- Orchestrator writes `current_task_index`, `heap_top` after each submission.
- Scheduler writes `last_task_alive`, `heap_tail` after each consumption.
- `tensormap_lookup()` passes `shared_header_.last_task_alive` instead of `0`.

### Files and Changes

**`runtime/runtime.h`**
- Add to Runtime class:
  ```cpp
  PTOSharedHeader shared_header_;
  PTOSharedHeader* get_shared_header() { return &shared_header_; }
  ```

**`runtime/runtime.cpp`**
- In `pto_init()`: zero-initialize `shared_header_`.
- In `pto_submit_task()`, after task creation:
  ```cpp
  shared_header_.current_task_index = next_task_id;
  shared_header_.heap_top = heap_ring_.top;
  ```
- In `tensormap_lookup()` calls (`runtime.cpp:302, 308`): replace `0` with `shared_header_.last_task_alive`.

**`aicpu/aicpu_executor.cpp`**
- After each CONSUMED transition:
  ```cpp
  while (last_task_alive < task_count &&
         runtime.get_task(last_task_alive)->state == TaskState::CONSUMED) {
      last_task_alive++;
  }
  shared_header->last_task_alive = last_task_alive;
  ```
- Need access to `shared_header_` via `runtime.get_shared_header()`.

### Testability

- After all tasks complete + scope_end: verify `last_task_alive` advances to `task_count`.
- Print in `tensormap_lookup` when stale entry is skipped. Verify entries are filtered for consumed tasks.
- Create test with tasks near `PTO_TENSORMAP_POOL_SIZE`. Verify pool doesn't overflow.

---

## Phase 5: Back-Pressure Flow Control

**Gaps addressed:** Gap 10 (Back-Pressure Flow Control)

**Rationale:** With PTOSharedHeader wired up, connect ring buffer allocation functions to real tail pointers for back-pressure.

### Before

- `task_ring_alloc()` and `heap_ring_alloc()` exist with stalling loops but never called with real tails.
- No limit on task submissions or memory consumption.

### After

- `task_ring_alloc()` reads `shared_header_.last_task_alive` as tail. Stalls when `in_flight >= PTO_TASK_WINDOW_SIZE - 1`.
- `heap_ring_alloc()` reads `shared_header_.heap_tail` as tail. Stalls when insufficient space.
- Scheduler advances `heap_tail` based on packed buffer end of oldest CONSUMED task.
- Graceful stalling under pressure instead of OOM.

### Files and Changes

**`runtime/runtime.cpp`**
- Replace `next_task_id++` with `task_ring_alloc(&task_ring_, &shared_header_.last_task_alive)`.
- Replace dummy tail in HeapRing allocation with `&shared_header_.heap_tail`.

**`aicpu/aicpu_executor.cpp`**
- After advancing `last_task_alive`, compute `heap_tail`:
  ```cpp
  if (last_task_alive > 0) {
      Task* last_consumed = runtime.get_task(last_task_alive - 1);
      shared_header->heap_tail = last_consumed->packed_buffer_offset
                                + last_consumed->packed_buffer_size;
  }
  ```

### Testability

- Set `PTO_TASK_WINDOW_SIZE` to 8, submit 20 tasks. Verify Orchestrator stalls at 7 and resumes as Scheduler consumes.
- Set `PTO_HEAP_SIZE` small, submit large-output tasks. Verify stalling, not OOM.
- Measure throughput with/without back-pressure for regression.

---

## Phase 6: DepListPool Integration

**Gaps addressed:** Gap 6 (DepListPool)

**Rationale:** Replace fixed-size `fanout[512]` and temporary `fanin_producers[]` with dynamic linked-list DepListPool. Removes arbitrary fanout limits, reduces per-task memory.

### Before

- `Task.fanout[RUNTIME_MAX_FANOUT]` = fixed 512-entry array (~2KB per task).
- `Task.fanin_producers[RUNTIME_MAX_ARGS]` = temporary reverse dep list (Phase 1).

### After

- `Task` uses `fanout_head` and `fanin_head` as offsets into shared `DepListPool`.
- Fanout lists grow dynamically with O(1) prepend.
- Memory reclaimed implicitly when task ring wraps.
- Per-task: ~8 bytes (two offsets) instead of ~2KB.

### Files and Changes

**`runtime/runtime.h`**
- Add to Runtime class:
  ```cpp
  DepListPool dep_list_pool_;
  DepListEntry dep_list_entries_[PTO_DEP_LIST_POOL_SIZE];
  ```
- Replace fixed arrays in Task struct:
  ```cpp
  // Remove: int fanout[RUNTIME_MAX_FANOUT];
  // Remove: int fanin_producers[RUNTIME_MAX_ARGS]; int fanin_producer_count;
  // Add:
  int32_t fanin_head;          // Offset into DepListPool (0 = empty)
  int32_t fanout_head;         // Offset into DepListPool (0 = empty)
  int32_t fanin_count;         // Number of producers (immutable after submission)
  volatile int32_t fanout_lock;  // Spinlock for concurrent fanout modification
  ```

**`runtime/runtime.cpp`**
- In `pto_init()`: `dep_list_pool_init(&dep_list_pool_, dep_list_entries_, PTO_DEP_LIST_POOL_SIZE)`.
- In `add_successor()`:
  ```cpp
  spinlock_acquire(&from->fanout_lock);
  from->fanout_head = dep_list_prepend(&dep_list_pool_, from->fanout_head, to_task);
  from->fanout_count++;
  spinlock_release(&from->fanout_lock);
  to->fanin_head = dep_list_prepend(&dep_list_pool_, to->fanin_head, from_task);
  to->fanin_count++;
  ```

**`aicpu/aicpu_executor.cpp`**
- Replace fanout array iteration with `dep_list_foreach()`:
  ```cpp
  dep_list_foreach(&dep_list_pool, task->fanout_head,
      [&](int32_t consumer_id, void*) {
          Task* consumer = runtime.get_task(consumer_id);
          if (consumer->fanin.fetch_sub(1) == 1) {
              enqueue_ready_task(consumer_id, consumer->core_type);
          }
      }, nullptr);
  ```
- Similarly traverse fanin list to increment producers' `fanout_refcount`.

### Testability

- Run diamond DAG. Verify identical dependency resolution.
- Test task with >512 consumers (impossible with fixed array, works with DepListPool).
- Print DepListPool utilization (`pool.top`) at end.

---

## Phase 7: Extended Worker Types

**Gaps addressed:** Gap 9 (Worker Types)

**Rationale:** Independent extension adding AICPU and Accelerator as dispatchable worker types.

### Before

- `PTOWorkerType`: CUBE (0), VECTOR (1) only. `PTO_NUM_WORKER_TYPES = 2`.
- AICPU only acts as scheduler.

### After

- `PTOWorkerType`: CUBE (0), VECTOR (1), AI_CPU (2), ACCELERATOR (3). `PTO_NUM_WORKER_TYPES = 4`.
- Ready queues expand to 4 types.
- AICPU worker dispatch: execute directly on scheduler thread or worker thread pool.
- Accelerator dispatch: stub with TODO.

### Files and Changes

**`runtime/pto_types.h`**
- Extend:
  ```cpp
  enum class PTOWorkerType : int32_t {
      CUBE        = 0,
      VECTOR      = 1,
      AI_CPU      = 2,
      ACCELERATOR = 3,
  };
  constexpr int32_t PTO_NUM_WORKER_TYPES = 4;
  ```

**`aicpu/aicpu_executor.cpp`**
- Ready queue arrays auto-expand via `PTO_NUM_WORKER_TYPES`.
- Add dispatch for AI_CPU (execute on scheduler thread or delegate to worker pool).
- Add stub for ACCELERATOR.

### Testability

- Submit AI_CPU task, verify execution.
- Verify CUBE/VECTOR tasks unaffected.
- Submit ACCELERATOR task, verify "not supported" warning.

---

## Summary Table

| Phase | Gaps Fixed | Priority | Dependencies | Key Files Modified | Status |
|-------|-----------|----------|-------------|-------------------|--------|
| 1 | #3 TaskState, #5 fanout_refcount | High | None | `runtime.h`, `runtime.cpp`, `aicpu_executor.cpp` | ✅ Done |
| 2 | #4 scope_end() | Medium | Phase 1 | `runtime.cpp` | ⏳ |
| 3 | #1 TaskRing, #2 HeapRing, #11 Packed outputs | High | Phase 1 | `runtime.h`, `runtime.cpp` | ⏳ |
| 4 | #7 PTOSharedHeader, #8 TensorMap staleness | Medium | Phase 3 | `runtime.h`, `runtime.cpp`, `aicpu_executor.cpp` | ⏳ |
| 5 | #10 Back-pressure | Medium | Phase 3+4 | `runtime.cpp`, `aicpu_executor.cpp` | ⏳ |
| 6 | #6 DepListPool | Low | Phase 1 | `runtime.h`, `runtime.cpp`, `aicpu_executor.cpp` | ⏳ |
| 7 | #9 Worker types | Low | None | `pto_types.h`, `aicpu_executor.cpp` | ⏳ |

---

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Changing Task struct breaks AICore executor | AICore only reads `function_bin_addr` and `args` — new fields are safe additions |
| Ring buffer wrap-around bugs | Test with artificially small ring sizes; implementations already handle wrap-around |
| Spinlock `fanout_lock` deadlocks | Only one lock held at a time (per-task, no nesting). Contention ~1/1024 probability |
| Back-pressure stalling livelock | Scheduler always progresses (completions before dispatch). Orchestrator only waits on Scheduler, not reverse |
| Packed buffer alignment waste | Bounded by `PTO_ALIGNMENT × num_outputs_per_task`, negligible |

---

## References

- [implementation-gaps.md](../implementation-gaps.md) — Gap analysis
- [runtime_buffer_manager_comprehensive_summary.md](../runtime_buffer_manager_comprehensive_summary.md) — Design reference
- [#01-pto_runtime_implementation_plan.md](#01-pto_runtime_implementation_plan.md) — Original plan
- [#02-pto_runtime_correction_plan.md](#02-pto_runtime_correction_plan.md) — Correction plan