# PTO Runtime Buffer Management — Comprehensive Summary

> **Legend**: Sections marked with 🆕 are new compared to the concise summary. Content marked with ✨ indicates enhanced details within existing sections.

## 1. What is PTO Runtime?

PTO Runtime executes two kinds of functions:

- **Orchestration Functions** — Turing-complete control flow (loops, conditionals, recursion, ✨nesting) that allocates buffers, submits tasks, and builds a dynamic dependency graph. Runs on Host CPU or Device AICPU.
- **InCore Functions** — Computational kernels (GEMM, vector ops, DMA copies) that run to completion on hardware units (Cube, Vector, AICPU, accelerators).

The runtime has three roles:

| Role | Runs on | Does what |
|------|---------|-----------|
| **Orchestrator** | Host CPU or Device AICPU | Executes orchestration function, submits tasks, builds dependency graph ✨via TensorMap |
| **Scheduler** | Device AICPU | Maintains ready queues, dispatches tasks, tracks dependencies, manages buffer lifecycle |
| **Workers** | Cube/Vector/AICPU/Accelerators | Execute InCore functions, signal completion |

## 2. Execution Flow

```
Host: Allocate device buffers → Copy inputs H2D → Launch orchestration
  ↓
Orchestrator: Allocate intermediates → Submit async tasks → Build dependency graph
  ↓
Scheduler: Resolve dependencies → Dispatch to workers → Track buffer lifecycle
  ↓
Workers: Execute kernels → Signal completion
  ↓
Host: Copy outputs D2H → Free device buffers
```

## 3. Runtime API (4 Functions)

| API | Caller | Purpose |
|-----|--------|---------|
| `pto_scope_begin()` | Orchestrator | Push current task position to scope stack |
| `pto_submit_task(kernel, params[])` | Orchestrator | Submit task; handles I/O registration, dependency tracking, ✨buffer allocation from HeapRing |
| `pto_scope_end()` | Orchestrator | Decrement fanout for all buffers in scope range `[begin, end)` |
| `pto_task_complete(task_id)` | Worker | Signal completion, update fanin/fanout refcounts |

## 4. Task Dependencies

Each task has:

- **Fanin** (dependencies) — producer tasks that must complete before this task runs. Set once at submission, immutable.
- **Fanout** (dependents) — consumer tasks ✨+ scope references that depend on this task's output. Grows dynamically as new consumers are submitted.

Task state transitions: `PENDING → READY → RUNNING → COMPLETED → CONSUMED`

- `PENDING → READY`: when `fanin_refcount == fanin_count` (all producers done)
- `COMPLETED → CONSUMED`: when `fanout_refcount == fanout_count` (all consumers + scopes released) → output buffers freed

## 5. Buffer Lifecycle (Scope + Reference Counting)

A buffer's `fanout_count` is initialized to `scope_depth` (number of enclosing scopes), then incremented for each consumer task.

Decrements come from two sources:
1. **`scope_end()`** — decrements fanout for all buffers in `[scope_begin_pos, scope_end_pos)`
2. **Consumer task completion** — each consumer decrements the buffer's fanout

When `fanout_count` reaches 0, the buffer is freed. This ensures buffers outlive both their lexical scope and all consumer tasks.

## 6. Ring Buffer Architecture (Zero-Overhead Memory)

All dynamic data uses ring buffers — no malloc/free needed.

| Ring Buffer | Head (Orchestrator writes) | Tail (Scheduler advances) | Purpose |
|-------------|---------------------------|--------------------------|---------|
| **Task Ring** | `current_task_index` | `last_task_alive` | Task slot allocation |
| **Heap Ring** | `heap_top` | `heap_tail` | Output buffer allocation (GM Heap) |
| **DepListPool** | `top` | Implicit (wraps with task ring) | Fanin/fanout linked list nodes |
| **TensorMap** | `pool_head` | Lazy invalidation | Producer lookup (region → task_id) |

Key properties:
- **O(1) allocation** — bump pointer
- **Implicit deallocation** — tail advances when `last_task_alive` moves forward
- **Zero fragmentation** — contiguous FIFO allocation
- **Back-pressure** — Orchestrator stalls when rings are full, waiting for Scheduler to consume

Wrap-around rule: never split a buffer across the ring boundary; skip remaining space and allocate from the beginning.

## 7. TensorMap (Producer Lookup with Overlap Detection) ✨Enhanced

A hash table backed by a ring buffer pool with **lazy invalidation**:

- ✨**Hash function**: Hash by `base_ptr` ONLY (not offset) to enable overlap detection — all sub-regions of same tensor must be in same bucket
- **Insert**: always at bucket chain head → chains naturally sorted newest-to-oldest by task_id
- **Lookup**: walk chain; if entry's `producer_task_id < last_task_alive`, it's stale — truncate entire chain tail
- ✨**Overlap detection**: Check if regions share same base_ptr + tile_index AND byte ranges `[offset, offset+size)` intersect
- **Reuse**: when pool wraps and overwrites a slot, old entry must be unlinked from bucket chain first (`in_bucket` flag)
- **Periodic cleanup**: every 64 retired tasks, explicitly remove stale entries from bucket chains

Sizing: ~136 KB total (4096 entries × 32B + 1024 buckets × 4B + 1024 task heads × 4B)

✨Complexity: O(valid_entries_only) due to chain truncation on first stale entry

## 8. Data Ownership & Shared Memory

```
┌─────────────────────────┐     Shared Memory      ┌─────────────────────────┐
│  Orchestrator Private   │◄──────────────────────►│   Scheduler Private     │
│                         │  • Header (flow ctrl)   │                         │
│  • TensorMap            │  • TaskDescriptor[]     │  • task_state[]         │
│  • scope_stack          │  • DepListPool          │  • fanin_refcount[]     │
│  • local ring pointers  │                         │  • fanout_refcount[]    │
│                         │                         │  • ready_queues[]       │
└─────────────────────────┘                         └─────────────────────────┘
```

Flow control via shared memory header:
- Orchestrator writes: `current_task_index`, `heap_top`
- Scheduler writes: `last_task_alive`, `heap_tail`
- Each reads the other's pointers for back-pressure

The Orchestrator can run on Host CPU (via PCIe shared memory) or Device AICPU (on-chip shared memory) — identical protocol either way.

## 9. Concurrency: Per-Task Fanout Lock

The only concurrent access requiring synchronization is the **fanout_list + fanout_count** of a task:

- **Orchestrator** prepends new consumers (writes `fanout_head`, increments `fanout_count`)
- **Scheduler** reads `fanout_count` to check the `CONSUMED` condition

Without atomicity, the Scheduler can see a stale `fanout_count`, prematurely mark a task as CONSUMED, and release its buffer → **use-after-free**.

**Solution**: per-task spinlock on `fanout_lock`. Contention is negligible (~1/1024 probability, ~50 cycles overhead).

Alternatives: lock-free CAS prepend, or a design-level sequencing constraint (all consumers submitted before producer completes).

**Why fanin doesn't need lock**: fanin_list and fanin_count are set once at submission and never modified — no concurrent access.

## 10. Output Buffer Allocation Mechanism 🆕

**Critical design**: Runtime (not orchestration function) allocates output buffers from HeapRing during `pto_submit_task()`.

```
Orchestration Function                    Runtime (pto_submit_task)
──────────────────────                    ─────────────────────────

void* P = NULL;     ──────────────────►  1. Receives &P (pointer-to-pointer)
                                         2. Calculate total output size
pto_submit_task(rt, "gemm", {            3. Allocate packed buffer from HeapRing
    PTO_INPUT(A, ...),                      (may stall if full)
    PTO_OUTPUT(&P, size)  ◄──────────    4. Write allocated address back to *P
});                                      5. Register in TensorMap for dependency
                                            tracking
// After submit returns:
// P now contains allocated address

pto_submit_task(rt, "add", {
    PTO_INPUT(P, ...),  ◄── Uses runtime-allocated address
    ...
});
```

**Two allocation modes**:
- **Mode A**: Pre-allocated large buffer (current bgemm), uses tile offsets
- **Mode B**: Runtime dynamic allocation (document-described), uses pointer-to-pointer for OUTPUT params

## 11. Heap Ring Allocation with Flow Control 🆕

Allocation from heap ring (O(1), stalls if insufficient space):

```c
// Never split buffer across ring boundary - skip to beginning instead
void* heap_ring_alloc(HeapRing* ring, int32_t size, volatile int32_t* tail_ptr) {
    size = ALIGN_UP(size, 64);  // Align for DMA efficiency

    while (true) {
        int32_t tail = *tail_ptr;  // Read from shared memory
        int32_t top = ring->top;

        if (top >= tail) {
            // Case 1: [....tail====top......]
            int32_t space_at_end = ring->size - top;
            if (space_at_end >= size) {
                void* ptr = (char*)ring->base + top;
                ring->top = top + size;
                return ptr;
            }
            // Try wrap to beginning
            if (tail > size) {
                ring->top = size;
                return ring->base;
            }
            continue;  // Stall
        } else {
            // Case 2: [====top....tail=====]
            int32_t gap = tail - top;
            if (gap >= size) {
                void* ptr = (char*)ring->base + top;
                ring->top = top + size;
                return ptr;
            }
            continue;  // Stall
        }
    }
}
```

**No explicit free operation** — tail is advanced by Scheduler when `last_task_alive` moves forward.

## 12. Dependency List Pool (Dynamic Fanout Growth) 🆕

Fanout lists grow dynamically as new consumers are submitted. Implementation uses linked list with pool allocation:

```c
typedef struct DepListEntry {
    int32_t task_id;              // The consumer task ID
    int32_t next_offset;          // Offset to next entry (0 = end)
} DepListEntry;

// Prepend operation (O(1))
int32_t dep_list_prepend(DepListPool* pool, int32_t current_head, int32_t task_id) {
    int32_t new_offset = pool->top++;  // Allocate from pool
    DepListEntry* new_entry = &pool->base[new_offset];
    new_entry->task_id = task_id;
    new_entry->next_offset = current_head;  // Link to old head
    return new_offset;  // New head
}
```

Memory reclaimed implicitly when task ring wraps — old lists become garbage and are overwritten.

## 13. Task Submission Flow (pto_submit_task) 🆕

```
1. Sync TensorMap validity threshold (read last_task_alive)
   ↓
2. Allocate task slot from Task Ring (may stall if full)
   ↓
3. First pass: Process INPUT params
   - Lookup producers via TensorMap
   - Build fanin_list (dependencies)
   - Update producer's fanout_count/list (with spinlock)
   ↓
4. Collect OUTPUT sizes → calculate total packed buffer size
   ↓
5. Allocate packed buffer from Heap Ring (may stall if full)
   ↓
6. Second pass: Register OUTPUTs in TensorMap
   - Compute addresses within packed buffer
   - Write back allocated addresses to caller (&P)
   - Insert in TensorMap: region → task_id
   ↓
7. Initialize fanin_count, fanout_count = scope_depth
   ↓
8. Check if task is immediately ready (fanin_refcount == fanin_count)
   → If yes: push to ready_queue[worker_type]
```

## 14. Scope Management Simplification 🆕

**Key design**: `fanout_count` starts from `scope_depth`, making `scope_end()` very simple:

```c
void pto_scope_end(PTORuntime* rt) {
    // Pop scope stack to get begin position
    int32_t scope_begin_pos = rt->scope_stack[rt->scope_stack_top--];
    int32_t scope_end_pos = rt->current_task_index;

    // Simple: increment fanout_refcount for ALL tasks in [begin, end)
    // No need to filter - every task in range has a reference from this scope
    for (int32_t i = scope_begin_pos; i < scope_end_pos; i++) {
        sched->fanout_refcount[i]++;

        // Check if task transitions to CONSUMED
        if (sched->fanout_refcount[i] == task->fanout_count &&
            sched->task_state[i] == TASK_COMPLETED) {
            sched->task_state[i] = TASK_CONSUMED;
            scheduler_on_task_consumed(rt, i);
        }
    }
}
```

**Why this works**: Every buffer allocated at depth D receives exactly D decrements from enclosing scopes (one per scope), plus decrements from consumers.

## 15. Task Completion and Dependency Updates 🆕

```c
void pto_task_complete(PTORuntime* rt, int32_t task_id) {
    // Mark task as completed
    sched->task_state[task_id] = TASK_COMPLETED;

    // Update fanin_refcount of all consumers (make them ready)
    for (each consumer in task->fanout_list) {
        sched->fanin_refcount[consumer]++;
        if (sched->fanin_refcount[consumer] == fanin_count) {
            sched->task_state[consumer] = TASK_READY;
            ready_queue_push(consumer);
        }
    }

    // Update fanout_refcount of all producers (for buffer lifecycle)
    for (each producer in task->fanin_list) {
        sched->fanout_refcount[producer]++;
        if (sched->fanout_refcount[producer] == fanout_count &&
            sched->task_state[producer] == TASK_COMPLETED) {
            sched->task_state[producer] = TASK_CONSUMED;
            scheduler_on_task_consumed(rt, producer);
        }
    }
}
```

## 16. Memory Reclamation via LastTaskAlive 🆕

```c
// Advance last_task_alive until we find a non-CONSUMED task
static void advance_last_task_alive(PTORuntime* rt) {
    while (sched->last_task_alive < task_count &&
           sched->task_state[sched->last_task_alive] == TASK_CONSUMED) {
        sched->last_task_alive++;
    }

    // Update heap_tail based on last CONSUMED task
    if (sched->last_task_alive > 0) {
        TaskDescriptor* last_consumed = &tasks[sched->last_task_alive - 1];
        int32_t new_tail = (char*)last_consumed->packed_buffer_end -
                          (char*)heap_ring.base;
        sched->heap_tail = new_tail;
        sm->heap_tail = new_tail;  // Write to shared memory
    }

    sm->last_task_alive = sched->last_task_alive;  // Flow control
}
```

**Key insight**: All memory is freed implicitly by advancing `last_task_alive` — no per-buffer free calls needed.

## 17. Per-Worker-Type Ready Queues 🆕

```
ready_queues[WORKER_CUBE]    ───► [T1, T5, T9, ...]
ready_queues[WORKER_VECTOR]  ───► [T2, T7, T12, ...]
ready_queues[WORKER_AI_CPU]  ───► [T3, T8, ...]
ready_queues[WORKER_ACCEL]   ───► [T4, T6, ...]
```

Benefits:
- Avoid single global queue lock contention
- Natural load balancing (same-type workers share queue)
- Support heterogeneous compute units

## 18. Multi-Threaded Implementation (ascend_a2a3_sim) 🆕

For the `ascend_a2a3_sim` platform, Orchestrator, Scheduler, and Workers run in independent threads:

```
┌─────────────────────┐
│  ORCHESTRATOR       │  pthread
│  Execute user func  │  Writes: current_task_index, heap_top
│  Submit tasks       │  Reads: last_task_alive, heap_tail
└──────────┬──────────┘  Sets: orchestrator_done
           │
           ▼ (via shared memory)
┌─────────────────────┐
│  SCHEDULER          │  pthread
│  Poll new tasks     │  Writes: last_task_alive, heap_tail
│  Dispatch to queues │  Reads: current_task_index, heap_top
│  Process completions│  Signals: all_done
└──────────┬──────────┘
           │
           ▼ (ready_queues with mutex + cond_var)
┌───────────────────────────────┐
│  WORKERS (multiple pthreads)  │
│  Wait on ready_queue          │
│  Execute kernels              │
│  Push to completion_queue     │
└───────────────────────────────┘
```

Synchronization mechanisms:
- **Shared memory pointers**: volatile + atomic for flow control
- **fanout_head/count**: per-task spinlock (correctness-critical)
- **Ready queues**: mutex + cond_var per worker type
- **Completion queue**: MPSC mutex (workers push, scheduler pops)
- **All-done signaling**: mutex + cond_var (scheduler → main)

## 19. Sizing Guidelines

```
TASK_WINDOW_SIZE       = 1024            (power of 2)
HEAP_SIZE              = workload-dependent (e.g. 64MB)
DEP_LIST_POOL_SIZE     = TASK_WINDOW_SIZE × AVG_FANOUT × 2 ≈ 8K entries (32KB)
TENSORMAP_POOL_SIZE    = TASK_WINDOW_SIZE × AVG_OUTPUTS × 2 ≈ 4K entries (128KB)
TENSORMAP_NUM_BUCKETS  = TASK_WINDOW_SIZE = 1024
```

## 20. Key Design Decisions ✨Enhanced (6→10 items)

1. **All dynamic data uses ring buffers** — no malloc/free, no fragmentation
2. **Implicit memory reclamation** — `last_task_alive` advancement frees everything
3. **TensorMap lazy invalidation + chain truncation** — stale entries ignored on lookup, O(valid_only) complexity
4. **Per-task spinlock for fanout** — required for correctness (count + list consistency), negligible overhead
5. **Shared memory decoupling** — Orchestrator and Scheduler communicate only via ring pointers
6. **Scope-based buffer lifecycle** — fanout initialized to scope_depth, scopes decrement on exit
7. 🆕 **Packed output buffers** — multiple outputs in single allocation, reduces alloc/free count by Nx
8. 🆕 **Overlap detection in TensorMap** — hash by base_ptr only, check byte range intersection for dependencies
9. 🆕 **Per-worker-type ready queues** — avoid global queue lock, natural load balancing
10. 🆕 **Zero-overhead allocation** — O(1) bump pointer, implicit free when tail advances

## 21. Performance Characteristics 🆕

| Aspect | Traditional Allocator | Ring Buffer Design |
|--------|----------------------|-------------------|
| **Allocation** | 100-500 cycles | ~5 cycles |
| **Deallocation** | 100-500 cycles | 0 cycles (implicit) |
| **Fragmentation** | High | Zero |
| **Flow control** | Manual rate limiting | Automatic back-pressure |
| **Memory overhead** | Per-buffer metadata | Fixed ring size |
| **TensorMap lookup** | O(total_chain_length) | O(valid_entries_only) |
| **Lock contention** | Global locks | Per-task locks, rare contention |

At 10M tasks/sec:
- Traditional: 2-10B cycles/sec overhead
- Ring buffer: ~50M cycles/sec overhead → **20-200x faster**

## 22. Complete Memory Layout 🆕

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SHARED MEMORY                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  • Header: current_task_index, heap_top, last_task_alive, heap_tail         │
│  • TaskDescriptor[TASK_WINDOW_SIZE] (ring buffer)                           │
│  • DepListPool (ring buffer for fanin/fanout linked lists)                  │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                       GLOBAL MEMORY HEAP                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  HeapRing: [RETIRED | ACTIVE BUFFERS | FREE SPACE] (ring buffer)            │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────┐         ┌──────────────────────────┐
│  ORCHESTRATOR PRIVATE    │         │  SCHEDULER PRIVATE       │
├──────────────────────────┤         ├──────────────────────────┤
│  • TensorMap (ring pool) │         │  • task_state[]          │
│  • scope_stack           │         │  • fanin_refcount[]      │
│  • Local ring pointers   │         │  • fanout_refcount[]     │
│  • tensormap_last_cleanup│         │  • ready_queues[]        │
└──────────────────────────┘         └──────────────────────────┘
```

## 23. Design Principles Summary 🆕

| Principle | Description | Benefit |
|-----------|-------------|---------|
| **FIFO lifecycle** | Tasks created and retired in order → use ring buffers | Implicit reclamation, O(1) operations |
| **Minimize shared state** | Only share what must be communicated | Reduced cache thrashing, less locking |
| **Lazy processing** | Defer work until necessary (lazy invalidation) | Lower overhead, automatic cleanup |
| **Batch operations** | Pack multiple outputs into one buffer | Reduce alloc/free count by Nx |
| **Fine-grained locks** | Per-task spinlock, not global lock | Minimal contention, ~0.1% probability |
| **Natural flow control** | Ring full = stall, no explicit rate limiting | Automatic back-pressure, no OOM |
| **Overlap detection** | Hash by base_ptr, check byte ranges | Correct dependencies for sub-tensors |