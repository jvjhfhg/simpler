# PTO Runtime Buffer Management - Concise Guide

## Overview

The PTO Runtime is a task execution framework for Ascend AI accelerators that supports **dynamic task submission** with Turing-complete control flow.

### Core Components

1. **Orchestration Functions** - Turing-complete programs with loops, conditionals, recursion
2. **InCore Functions** - Computational kernels executed on hardware (AICore CUBE/VECTOR, AICPU, Accelerators)
3. **Runtime APIs** - 4 simple APIs for task submission and lifecycle management

### Key Features

- **Dynamic Task Graphs** - Tasks submitted at runtime, not predefined
- **Zero-Copy Buffers** - Ring buffer allocation (~5 cycles vs 100-500 for malloc/free)
- **Automatic Dependencies** - TensorMap resolves producer-consumer relationships
- **Scope-Based Lifecycle** - Buffers freed when no longer needed

---

## Runtime Architecture

```
┌─────────────────────────────────────────────────────────────┐
│   ORCHESTRATOR (Host CPU or Device AICPU)                   │
│   • Execute orchestration function                          │
│   • Submit tasks via pto_submit_task()                      │
│   • Build dependency graph via TensorMap                    │
│   • Manage buffer scopes                                    │
└────────────────────┬────────────────────────────────────────┘
                     │ Shared Memory (Task Descriptors)
┌────────────────────▼────────────────────────────────────────┐
│   SCHEDULER (Device AICPU)                                  │
│   • Maintain ready queues (per worker type)                │
│   • Resolve dependencies (fanin tracking)                   │
│   • Dispatch tasks to workers                               │
│   • Track buffer lifecycle (fanout tracking)                │
└────────────────────┬────────────────────────────────────────┘
                     │ Task Dispatch
┌────────────────────▼────────────────────────────────────────┐
│   WORKERS                                                   │
│   • AICore CUBE    - Matrix ops, GEMM                       │
│   • AICore VECTOR  - Element-wise, reduction                │
│   • AI_CPU         - Scalar ops, control flow               │
│   • Accelerators   - DMA, fixed-function HW                 │
└─────────────────────────────────────────────────────────────┘
```

---

## The 4 Core APIs

### Called by Orchestrator

1. **`pto_scope_begin()`** - Mark beginning of scope
2. **`pto_submit_task(kernel, params[])`** - Submit task with inputs/outputs
3. **`pto_scope_end()`** - End scope, release scope references

### Called by Worker

4. **`pto_task_complete(task_id)`** - Signal task completion

---

## Key Data Structures

### Task Descriptor (Orchestrator-Owned, Read-Only for Scheduler)

```c
typedef struct TaskDescriptor {
    int32_t task_id;
    int32_t kernel_id;            // Which InCore function
    int32_t worker_type;          // CUBE, VECTOR, AI_CPU, ACCELERATOR

    // Dependency graph (static after submission)
    int32_t* fanin_list;          // Producer task IDs
    int32_t fanin_count;          // Total dependencies
    int32_t* fanout_list;         // Consumer task IDs
    int32_t fanout_count;         // Consumers + scope_depth

    // Packed output buffer (all outputs in one allocation)
    void* packed_buffer_base;
    void* packed_buffer_end;
    int32_t output_offsets[MAX_OUTPUTS];

    int32_t scope_depth;          // Scope when created
} TaskDescriptor;
```

### Scheduler State (Scheduler-Owned, Dynamic)

```c
typedef struct SchedulerState {
    // Per-task runtime state
    TaskState* task_state;         // PENDING/READY/RUNNING/COMPLETED/CONSUMED
    int32_t* fanin_refcount;       // Completed dependencies (starts at 0)
    int32_t* fanout_refcount;      // Released references (starts at 0)

    // Ready queues (one per worker type)
    ReadyQueue ready_queues[NUM_WORKER_TYPES];

    // Ring pointers
    int32_t last_task_alive;       // Task ring tail (for reclamation)
    int32_t heap_tail;             // Heap ring tail (for reclamation)
} SchedulerState;
```

### TensorMap - Producer Lookup

**Purpose**: Map buffer addresses → producer task IDs for automatic dependency resolution

**Design**: Hash table with ring buffer entry pool + lazy invalidation

```c
typedef struct TensorMap {
    int32_t* buckets;              // Hash table (power of 2 size)
    int32_t num_buckets;

    TensorMapEntry* entry_pool;    // Ring buffer of entries
    int32_t pool_size;
    int32_t pool_head;             // Next allocation position

    int32_t last_task_alive;       // Validity threshold
} TensorMap;

typedef struct TensorMapEntry {
    TensorRegion region;           // {buffer_ptr, tile_index, offset, size}
    int32_t producer_task_id;
    int32_t next_in_bucket;        // Collision chain
    bool in_bucket;                // Must unlink before overwrite
} TensorMapEntry;
```

**Key Operations**:
- **Insert**: O(1) - Hash and prepend to bucket chain
- **Lookup**: O(valid_entries) - Walk chain, truncate on first stale entry
- **Lazy Invalidation**: Entries auto-invalidate when `task_id < last_task_alive`

**Optimization**: Chain truncation on stale entry
- Chains sorted by task_id (newest first from HEAD insert)
- First stale entry → rest are guaranteed stale → truncate immediately
- Lookup cost = O(valid entries only)

---

## Ring Buffer Allocation

### Task Ring

Allocates **task slots** from a ring buffer:

```c
typedef struct TaskRing {
    int32_t head;                  // Next allocation (Orchestrator writes)
    int32_t tail;                  // Oldest live task (Scheduler writes)
    int32_t capacity;
    int32_t current_index;
} TaskRing;

// Allocate task slot (may stall if full)
int32_t task_ring_alloc(TaskRing* ring, atomic_int32_t* tail_ptr) {
    while (true) {
        int32_t head = ring->head;
        int32_t tail = atomic_load(tail_ptr);
        int32_t available = ring->capacity - (head - tail);

        if (available > 0) {
            int32_t slot = head % ring->capacity;
            ring->head = head + 1;
            return slot;
        }
        // Ring full - wait for scheduler to advance tail
    }
}
```

### Heap Ring

Allocates **output buffers** from a heap ring:

```c
typedef struct HeapRing {
    uint64_t head;                 // Next allocation (Orchestrator)
    uint64_t tail;                 // Oldest live buffer (Scheduler)
    uint64_t capacity;
    uint8_t* base_ptr;
} HeapRing;

// Allocate buffer (may stall if full)
uint64_t heap_ring_alloc(HeapRing* ring, uint32_t size,
                         atomic_uint64_t* tail_ptr) {
    size = ALIGN_UP(size, 64);  // Cache line alignment

    while (true) {
        uint64_t head = ring->head;
        uint64_t tail = atomic_load(tail_ptr);
        uint64_t available = ring->capacity - (head - tail);

        if (available >= size) {
            uint64_t addr = (uint64_t)ring->base_ptr + (head % ring->capacity);
            ring->head = head + size;
            return addr;
        }
        // Ring full - wait
    }
}
```

**Back-Pressure**: When rings are full, Orchestrator automatically stalls, preventing OOM.

---

## Dependency Resolution

### Fanin (Task Dependencies)

- **fanin_count** (Orchestrator): Total number of producer tasks (immutable)
- **fanin_refcount** (Scheduler): Starts at 0, increments as producers complete
- **Condition**: Task is READY when `fanin_refcount == fanin_count`

### Fanout (Buffer Lifecycle)

- **fanout_count** (Orchestrator): `scope_depth + num_consumers` (can grow as consumers added)
- **fanout_refcount** (Scheduler): Starts at 0, increments as consumers complete + scopes end
- **Condition**: Buffer freed when `fanout_refcount == fanout_count` AND task is COMPLETED

```
Task State Transitions:

PENDING ──► READY ──► RUNNING ──► COMPLETED ──► CONSUMED
   │          │          │            │             │
fanin      fanin      dispatch    execution    fanout_refcount
refcount   ==fanin    to worker    done        ==fanout_count
<fanin     _count                              (release buffers)
_count
```

---

## Scope-Based Lifecycle

### Key Insight

Initialize `fanout_count = scope_depth` at task creation. When scope ends, increment `fanout_refcount` for all tasks in scope range. This ensures buffers live at least until scope exits.

### Example

```c
void orchestration() {
    pto_scope_begin();              // depth = 1

    void* P = alloc();              // P allocated
    pto_submit_task(gemm, out=P);   // Task A, fanout_count = 1 (scope)
    pto_submit_task(add, in=P);     // Task B, A.fanout_count++ → 2

    pto_scope_end();                // A.fanout_refcount++ → 1

    // Later: Task B completes
    // → A.fanout_refcount++ → 2 == fanout_count → P freed
}
```

### Nested Scopes

```c
pto_scope_begin();              // depth = 1
  pto_scope_begin();            // depth = 2
    void* P = alloc();          // fanout_count = 2
    submit(gemm, out=P);
    submit(add, in=P);          // fanout_count = 3 (2 scopes + 1 consumer)
  pto_scope_end();              // fanout_refcount++ → 1
pto_scope_end();                // fanout_refcount++ → 2

// After add completes: fanout_refcount = 3 → P freed
```

---

## API Implementation

### 1. pto_scope_begin()

```c
void pto_scope_begin(PTORuntime* rt) {
    int32_t current_pos = rt->task_ring.current_index;
    rt->scope_stack[++rt->scope_stack_top] = current_pos;
}
```

### 2. pto_submit_task()

```c
int32_t pto_submit_task(PTORuntime* rt, int32_t kernel_id,
                        int32_t worker_type, TaskParam* params,
                        int32_t num_params) {

    // STEP 1: Sync TensorMap validity
    tensormap_sync_validity(&rt->tensor_map, rt->sm_header);

    // STEP 2: Allocate task slot (may stall)
    int32_t task_id = task_ring_alloc(&rt->task_ring,
                                      &rt->sm_header->last_task_alive);

    // STEP 3: Initialize task
    TaskDescriptor* task = &rt->task_descriptors[task_id];
    task->fanout_count = rt->scope_stack_top + 1;  // scope_depth
    task->fanin_count = 0;

    // STEP 4: Process inputs - lookup producers, build fanin
    int32_t fanin_temp[32];
    int32_t local_fanin_count = 0;
    int32_t total_output_size = 0;

    for (int i = 0; i < num_params; i++) {
        if (params[i].type == PARAM_INPUT) {
            TensorRegion region = {params[i].buffer, ...};
            int32_t producer_id = tensormap_lookup(&rt->tensor_map, &region);

            if (producer_id >= 0) {
                fanin_temp[local_fanin_count++] = producer_id;
                rt->task_descriptors[producer_id].fanout_count++;
            }
        }
        else if (params[i].type == PARAM_OUTPUT) {
            total_output_size += ALIGN_UP(params[i].size, 64);
        }
    }

    task->fanin_count = local_fanin_count;

    // STEP 5: Allocate packed output buffer (may stall)
    if (total_output_size > 0) {
        task->packed_buffer_base = heap_ring_alloc(&rt->heap_ring,
                                                    total_output_size,
                                                    &rt->sm_header->heap_tail);
    }

    // STEP 6: Register outputs in TensorMap
    uint32_t offset = 0;
    for (int i = 0; i < num_params; i++) {
        if (params[i].type == PARAM_OUTPUT) {
            void* output_addr = task->packed_buffer_base + offset;
            TensorRegion region = {output_addr, ...};
            tensormap_insert(&rt->tensor_map, &region, task_id);
            offset += ALIGN_UP(params[i].size, 64);
        }
    }

    // STEP 7: Check if immediately ready
    if (local_fanin_count == 0) {
        rt->sched_state.task_state[task_id] = TASK_READY;
        ready_queue_push(&rt->sched_state.ready_queues[worker_type], task_id);
    } else {
        rt->sched_state.task_state[task_id] = TASK_PENDING;
    }

    return task_id;
}
```

### 3. pto_scope_end()

```c
void pto_scope_end(PTORuntime* rt) {
    // Pop scope stack
    int32_t scope_begin_pos = rt->scope_stack[rt->scope_stack_top--];
    int32_t scope_end_pos = rt->task_count;

    SchedulerState* sched = &rt->sched_state;

    // Simple: increment fanout_refcount for ALL tasks in [begin, end)
    for (int32_t task_id = scope_begin_pos; task_id < scope_end_pos; task_id++) {
        sched->fanout_refcount[task_id]++;

        // Check if CONSUMED
        TaskDescriptor* task = &rt->task_descriptors[task_id];
        if (sched->fanout_refcount[task_id] == task->fanout_count &&
            sched->task_state[task_id] == TASK_COMPLETED) {
            sched->task_state[task_id] = TASK_CONSUMED;
            advance_last_task_alive(rt);
        }
    }
}
```

### 4. pto_task_complete()

```c
void pto_task_complete(PTORuntime* rt, int32_t task_id) {
    TaskDescriptor* task = &rt->task_descriptors[task_id];
    SchedulerState* sched = &rt->sched_state;

    sched->task_state[task_id] = TASK_COMPLETED;

    // Update fanin_refcount of consumers (make them ready)
    for (int i = 0; i < task->fanout_list_size; i++) {
        int32_t consumer_id = task->fanout_list[i];
        TaskDescriptor* consumer = &rt->task_descriptors[consumer_id];

        sched->fanin_refcount[consumer_id]++;

        if (sched->fanin_refcount[consumer_id] == consumer->fanin_count &&
            sched->task_state[consumer_id] == TASK_PENDING) {
            sched->task_state[consumer_id] = TASK_READY;
            ready_queue_push(&sched->ready_queues[consumer->worker_type],
                           consumer_id);
        }
    }

    // Update fanout_refcount of producers (for buffer lifecycle)
    for (int i = 0; i < task->fanin_list_size; i++) {
        int32_t producer_id = task->fanin_list[i];
        TaskDescriptor* producer = &rt->task_descriptors[producer_id];

        sched->fanout_refcount[producer_id]++;

        if (sched->fanout_refcount[producer_id] == producer->fanout_count &&
            sched->task_state[producer_id] == TASK_COMPLETED) {
            sched->task_state[producer_id] = TASK_CONSUMED;
            advance_last_task_alive(rt);
        }
    }
}
```

---

## Memory Reclamation

### Automatic via Ring Pointers

When `last_task_alive` advances (task transitions to CONSUMED):
- **Task Ring**: Old slots available for reuse
- **Heap Ring**: `heap_tail` follows `last_task_alive`, implicitly freeing buffers
- **TensorMap**: Old entries become invalid (lazy invalidation)

```c
void advance_last_task_alive(PTORuntime* rt) {
    SchedulerState* sched = &rt->sched_state;

    // Advance until we find a non-CONSUMED task
    while (sched->last_task_alive < rt->task_count &&
           sched->task_state[sched->last_task_alive] == TASK_CONSUMED) {
        sched->last_task_alive++;
    }

    // Update heap_tail to follow last_task_alive
    if (sched->last_task_alive > 0) {
        TaskDescriptor* oldest_alive =
            &rt->task_descriptors[sched->last_task_alive];
        rt->heap_ring.tail = (uint64_t)oldest_alive->packed_buffer_base;
    }
}
```

**No explicit free() needed!** Buffer reclamation happens implicitly via pointer advancement.

---

## Design Optimizations

### 1. Ring Buffer Everywhere

**Problem**: malloc/free too slow (100-500 cycles)
**Solution**: Ring buffer allocation (~5 cycles)

- Task Ring: Fixed-size task slots
- Heap Ring: Variable-size buffers
- DepList Pool: Fanin/fanout arrays
- TensorMap: Entry pool

### 2. Packed Output Buffers

**Problem**: Multiple outputs → multiple allocations
**Solution**: Pack all outputs into single contiguous buffer

```
Before:  [Out0] [Out1] [Out2]  → 3 allocs, 3 frees
After:   [Out0|Out1|Out2]      → 1 alloc, 1 free
```

### 3. TensorMap Lazy Invalidation

**Problem**: Explicit removal expensive
**Solution**: Let entries become stale, ignore in lookup

- Entry valid if `producer_task_id >= last_task_alive`
- Ring wrap automatically overwrites stale entries
- Chain truncation removes stale tail during lookup

### 4. Data Ownership Separation

**Problem**: Lock contention between Orchestrator and Scheduler
**Solution**: Separate data by writer

- **Orchestrator owns**: TaskDescriptor, TensorMap, fanin_count, fanout_count
- **Scheduler owns**: task_state, fanin_refcount, fanout_refcount, ready_queues
- **Synchronization**: Only fanout_count updates need per-task spinlock

### 5. Scope-Based Fanout Initialization

**Problem**: Complex scope lifetime tracking
**Solution**: `fanout_count = scope_depth` at creation

- Each enclosing scope holds a reference
- `pto_scope_end()` just increments fanout_refcount for range
- No need to track which scope owns which buffer

### 6. Per-Worker-Type Ready Queues

**Problem**: Single global queue → lock contention
**Solution**: Separate queue per worker type

- `ready_queues[CUBE]`, `ready_queues[VECTOR]`, etc.
- Natural load balancing
- Minimal contention

---

## Performance Characteristics

| Operation | Complexity | Typical Cycles |
|-----------|------------|----------------|
| Task ring alloc | O(1) | ~5 cycles |
| Heap ring alloc | O(1) | ~5 cycles |
| TensorMap insert | O(1) | ~10 cycles |
| TensorMap lookup | O(valid_entries) | ~20 cycles |
| Task submission | O(num_params) | ~100 cycles |
| Task completion | O(fanout_size) | ~50 cycles |
| Scope end | O(scope_task_count) | ~10 cycles/task |

**Memory Footprint** (for 1024 task window):
- Task descriptors: ~128 KB
- TensorMap: ~136 KB
- Scheduler state: ~64 KB
- Total: ~328 KB

---

## Example: Matrix Multiply Orchestration

```c
void matmul_orchestration(Runtime* rt, uint64_t* args) {
    float* A = (float*)args[0];
    float* B = (float*)args[1];
    float* C = (float*)args[2];
    int M = args[3], K = args[4], N = args[5];

    const int TILE_M = 128, TILE_K = 128, TILE_N = 128;

    for (int i = 0; i < M; i += TILE_M) {
        for (int j = 0; j < N; j += TILE_N) {
            pto_scope_begin();

            for (int k = 0; k < K; k += TILE_K) {
                TaskParam params[3];
                params[0] = {PARAM_INPUT, &A[i*K + k], TILE_M*TILE_K*4};
                params[1] = {PARAM_INPUT, &B[k*N + j], TILE_K*TILE_N*4};
                params[2] = {PARAM_OUTPUT, NULL, TILE_M*TILE_N*4};

                pto_submit_task(rt, KERNEL_GEMM, WORKER_CUBE, params, 3);
            }

            pto_scope_end();
            // All partial results freed after reduction completes
        }
    }
}
```

---

## Summary

**4 Simple APIs** enable powerful orchestration:
1. `pto_scope_begin()` - Mark scope start
2. `pto_submit_task()` - Submit task with automatic dependency resolution
3. `pto_scope_end()` - Release scope references
4. `pto_task_complete()` - Signal completion, propagate to consumers

**Key Benefits**:
- **Fast**: Ring buffer allocation (~5 cycles)
- **Simple**: No manual dependency specification or buffer management
- **Automatic**: Dependencies via TensorMap, lifecycle via fanin/fanout
- **Bounded**: Back-pressure prevents OOM
- **Lock-Free**: Mostly lock-free via data ownership separation

**Memory is implicit**: Buffers allocated inside `pto_submit_task()`, freed implicitly via ring pointer advancement when no longer needed.