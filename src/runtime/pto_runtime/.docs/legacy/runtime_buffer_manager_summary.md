# PTO Runtime Buffer Management — Summary

## 1. What is PTO Runtime?

PTO Runtime executes two kinds of functions:

- **Orchestration Functions** — Turing-complete control flow (loops, conditionals, recursion) that allocates buffers, submits tasks, and builds a dependency graph. Runs on Host CPU or Device AICPU.
- **InCore Functions** — Computational kernels (GEMM, vector ops, DMA copies) that run to completion on hardware units (Cube, Vector, AICPU, accelerators).

The runtime has three roles:

| Role | Runs on | Does what |
|------|---------|-----------|
| **Orchestrator** | Host CPU or Device AICPU | Executes orchestration function, submits tasks, builds dependency graph |
| **Scheduler** | Device AICPU | Maintains ready queues, dispatches tasks, manages buffer lifecycle |
| **Workers** | Cube/Vector/AICPU/Accelerators | Execute InCore functions, signal completion |

## 2. Execution Flow

```
Host: Allocate device buffers → Copy inputs H2D → Launch orchestration
  ↓
Orchestrator: Allocate intermediates → Submit async tasks → Build dependency graph
  ↓
Scheduler: Resolve dependencies → Dispatch to workers → Track buffer lifecycle
  ↓
Host: Copy outputs D2H → Free device buffers
```

## 3. Runtime API (4 Functions)

| API | Caller | Purpose |
|-----|--------|---------|
| `pto_scope_begin()` | Orchestrator | Push current task position to scope stack |
| `pto_submit_task(kernel, params[])` | Orchestrator | Submit task; handles I/O registration, dependency tracking, buffer allocation |
| `pto_scope_end()` | Orchestrator | Decrement fanout for all buffers in scope range `[begin, end)` |
| `pto_task_complete(task_id)` | Worker | Signal completion, update fanin/fanout counters |

## 4. Task Dependencies

Each task has:

- **Fanin** (dependencies) — producer tasks that must complete before this task runs. Set once at submission, immutable.
- **Fanout** (dependents) — consumer tasks that depend on this task's output. Grows dynamically as new consumers are submitted.

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

## 7. TensorMap (Producer Lookup)

A hash table backed by a ring buffer pool with **lazy invalidation**:

- **Insert**: always at bucket chain head → chains are naturally sorted newest-to-oldest
- **Lookup**: walk chain; if an entry's `producer_task_id < last_task_alive`, it's stale — truncate the entire chain tail (all subsequent entries are older, thus also stale)
- **Reuse**: when pool wraps and overwrites a slot, the old entry must be removed from its bucket chain first (tracked via `in_bucket` flag)
- **Periodic cleanup**: every 64 retired tasks, explicitly remove stale entries from bucket chains

Sizing: ~136 KB total (4096 entries × 32B + 1024 buckets × 4B + 1024 task heads × 4B)

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

## 10. Sizing Guidelines

```
TASK_WINDOW_SIZE       = 1024            (power of 2)
HEAP_SIZE              = workload-dependent (e.g. 64MB)
DEP_LIST_POOL_SIZE     = TASK_WINDOW_SIZE × AVG_FANOUT × 2 ≈ 8K entries (32KB)
TENSORMAP_POOL_SIZE    = TASK_WINDOW_SIZE × AVG_OUTPUTS × 2 ≈ 4K entries (128KB)
TENSORMAP_NUM_BUCKETS  = TASK_WINDOW_SIZE = 1024
```

## 11. Key Design Decisions

1. **All dynamic data uses ring buffers** — no malloc/free, no fragmentation
2. **Implicit memory reclamation** — `last_task_alive` advancement frees everything
3. **TensorMap lazy invalidation** — stale entries ignored on lookup, cleaned lazily
4. **Per-task spinlock for fanout** — required for correctness, negligible overhead
5. **Shared memory decoupling** — Orchestrator and Scheduler communicate only via ring pointers
6. **Scope-based buffer lifecycle** — fanout initialized to scope_depth, scopes decrement on exit