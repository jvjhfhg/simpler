# PTO Orchestrator Framework Implementation Plan

This document outlines the implementation plan for the PTO Runtime Orchestrator Framework based on the design specifications in `runtime_buffer_manager_methods.md`.

---

## 1. Overview

### 1.1 Goals

Implement a dynamic task submission runtime with:
- **4 Simple APIs**: `pto_scope_begin()`, `pto_submit_task()`, `pto_scope_end()`, `pto_task_complete()`
- **Zero-copy ring buffers**: ~5 cycles allocation vs 100-500 for malloc/free
- **Automatic dependency resolution**: TensorMap tracks producer-consumer relationships
- **Scope-based buffer lifecycle**: Buffers freed when no longer needed
- **Orchestrator-Scheduler decoupling**: Communicate via shared memory

### 1.2 Architecture Summary

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
│   WORKERS (AICore CUBE/VECTOR, AICPU, Accelerators)        │
│   • Execute InCore functions                                │
│   • Signal completion via pto_task_complete()               │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Implementation Phases

### Phase 1: Core Data Structures
### Phase 2: Ring Buffer Infrastructure
### Phase 3: TensorMap Implementation
### Phase 4: Orchestrator APIs
### Phase 5: Scheduler Implementation
### Phase 6: Integration with Existing Runtime
### Phase 7: Testing & Validation

---

## 3. Phase 1: Core Data Structures

**Location**: `src/runtime/pto_orchestrator/runtime/`

### 3.1 Shared Memory Header

```c
// File: pto_runtime_types.h

typedef struct SharedMemoryHeader {
    // Flow control (Orchestrator → Scheduler)
    volatile int32_t current_task_index;  // Task ring head
    volatile int32_t heap_top;            // Heap allocation pointer
    volatile int32_t orchestrator_done;   // Completion flag

    // Flow control (Scheduler → Orchestrator, for back-pressure)
    volatile int32_t last_task_alive;     // Task ring tail
    volatile int32_t heap_tail;           // Heap free pointer

    // Layout info (set once at init)
    int32_t task_window_size;
    int32_t heap_size;
    int32_t dep_list_pool_size;

    // Offsets into shared memory
    int32_t task_descriptors_offset;
    int32_t dep_list_pool_offset;
} SharedMemoryHeader;
```

### 3.2 Task Descriptor

```c
typedef struct TaskDescriptor {
    int32_t task_id;
    int32_t kernel_id;
    int32_t worker_type;          // CUBE, VECTOR, AI_CPU, ACCELERATOR

    // Dependency lists (offsets into dep_list_pool)
    int32_t fanin_head;           // Linked list head
    int32_t fanin_count;
    int32_t fanout_head;
    int32_t fanout_count;

    // Per-task lock for fanout updates
    volatile int32_t fanout_lock;

    // Scope tracking
    int32_t scope_depth;

    // Packed output buffer
    void* packed_buffer_base;
    void* packed_buffer_end;
    int32_t output_offsets[MAX_OUTPUTS];
    int32_t num_outputs;
} TaskDescriptor;
```

### 3.3 Task State Enum

```c
typedef enum {
    TASK_PENDING   = 0,  // Waiting for dependencies
    TASK_READY     = 1,  // Ready to dispatch
    TASK_RUNNING   = 2,  // Currently executing
    TASK_COMPLETED = 3,  // Execution done, output may be in use
    TASK_CONSUMED  = 4   // Output consumed, buffers can be freed
} TaskState;
```

### 3.4 Worker Types

```c
typedef enum {
    WORKER_CUBE = 0,
    WORKER_VECTOR = 1,
    WORKER_AI_CPU = 2,
    WORKER_ACCELERATOR = 3,
    NUM_WORKER_TYPES = 4
} WorkerType;
```

### 3.5 Task Parameters

```c
typedef enum {
    PARAM_INPUT = 0,
    PARAM_OUTPUT = 1,
    PARAM_INOUT = 2
} ParamType;

typedef struct TaskParam {
    ParamType type;
    void* buffer;         // For INPUT: source buffer; for OUTPUT: NULL (allocated)
    int32_t tile_index;   // Tile within buffer
    int32_t offset;       // Byte offset within tile
    int32_t size;         // Size in bytes
} TaskParam;
```

**Deliverables**:
- [ ] `pto_runtime_types.h` - All type definitions
- [ ] `pto_runtime_config.h` - Configuration constants

---

## 4. Phase 2: Ring Buffer Infrastructure

**Location**: `src/runtime/pto_orchestrator/runtime/`

### 4.1 Task Ring

```c
// File: task_ring.h

typedef struct TaskRing {
    int32_t head;           // Next allocation (Orchestrator writes)
    int32_t capacity;       // TASK_WINDOW_SIZE
} TaskRing;

// Allocate task slot (may stall if full)
int32_t task_ring_alloc(TaskRing* ring, volatile int32_t* tail_ptr);
```

**Implementation Notes**:
- Stall (spin-wait) when `head - tail >= capacity`
- Return slot index: `head % capacity`
- Advance head after allocation

### 4.2 Heap Ring

```c
// File: heap_ring.h

typedef struct HeapRing {
    uint8_t* base;          // GM Heap base pointer
    uint64_t head;          // Allocation pointer
    uint64_t capacity;      // Total heap size
} HeapRing;

// Allocate buffer (may stall if full)
void* heap_ring_alloc(HeapRing* ring, uint32_t size, volatile uint64_t* tail_ptr);
```

**Implementation Notes**:
- Align allocations to 64 bytes (cache line)
- Never split buffer across ring boundary - skip to beginning
- Stall when insufficient space

### 4.3 Dependency List Pool

```c
// File: dep_list_pool.h

typedef struct DepListEntry {
    int32_t task_id;
    int32_t next_offset;    // 0 = end of list
} DepListEntry;

typedef struct DepListPool {
    DepListEntry* base;
    int32_t capacity;
    int32_t top;            // Next allocation position
} DepListPool;

// Prepend to linked list (O(1))
int32_t dep_list_prepend(DepListPool* pool, int32_t current_head, int32_t task_id);
```

**Deliverables**:
- [ ] `task_ring.h` / `task_ring.c`
- [ ] `heap_ring.h` / `heap_ring.c`
- [ ] `dep_list_pool.h` / `dep_list_pool.c`

---

## 5. Phase 3: TensorMap Implementation

**Location**: `src/runtime/pto_orchestrator/runtime/`

### 5.1 Data Structures

```c
// File: tensor_map.h

typedef struct TensorRegion {
    void* base_ptr;
    int32_t tile_index;
    int32_t offset;
    int32_t size;
} TensorRegion;

typedef struct TensorMapEntry {
    TensorRegion region;
    int32_t producer_task_id;
    int32_t next_in_bucket;   // Hash chain
    int32_t next_in_task;     // Per-task list
    bool in_bucket;           // For safe reuse
} TensorMapEntry;

typedef struct TensorMap {
    int32_t* buckets;              // Hash table (power of 2)
    int32_t num_buckets;

    TensorMapEntry* entry_pool;    // Ring buffer
    int32_t pool_size;
    int32_t pool_head;

    int32_t* task_entry_head;      // Per-task entry list
    int32_t last_task_alive;       // Validity threshold
} TensorMap;
```

### 5.2 Operations

```c
// Initialize TensorMap
void tensormap_init(TensorMap* tm, int32_t num_buckets, int32_t pool_size);

// Sync validity threshold from shared memory
void tensormap_sync_validity(TensorMap* tm, SharedMemoryHeader* sm);

// Insert producer-output mapping
void tensormap_insert(TensorMap* tm, TensorRegion* region, int32_t producer_task_id);

// Lookup producer for input region
// Returns producer_task_id or -1 if not found
int32_t tensormap_lookup(TensorMap* tm, TensorRegion* region);
```

### 5.3 Key Optimizations

1. **Lazy Invalidation**: Entry valid only if `producer_task_id >= last_task_alive`
2. **Chain Truncation**: On lookup, truncate chain at first stale entry (all subsequent are stale)
3. **HEAD Insert**: Chains naturally sorted newest-to-oldest
4. **Periodic Cleanup**: Every 64 retired tasks, explicitly remove stale entries

**Deliverables**:
- [ ] `tensor_map.h` / `tensor_map.c`
- [ ] Unit tests for TensorMap operations

---

## 6. Phase 4: Orchestrator APIs

**Location**: `src/runtime/pto_orchestrator/orchestrator/`

### 6.1 Orchestrator State

```c
// File: orchestrator.h

typedef struct OrchestratorState {
    // Shared memory access
    SharedMemoryHeader* sm_header;
    TaskDescriptor* task_descriptors;
    DepListPool dep_list_pool;

    // Local ring state
    TaskRing task_ring;
    HeapRing heap_ring;

    // Private data
    TensorMap tensor_map;
    int32_t* scope_stack;
    int32_t scope_stack_top;
    int32_t scope_stack_capacity;
} OrchestratorState;
```

### 6.2 API: pto_scope_begin()

```c
void pto_scope_begin(OrchestratorState* orch) {
    int32_t current_pos = orch->task_ring.head;
    orch->scope_stack[++orch->scope_stack_top] = current_pos;
}
```

**Complexity**: O(1)

### 6.3 API: pto_submit_task()

```c
int32_t pto_submit_task(
    OrchestratorState* orch,
    int32_t kernel_id,
    int32_t worker_type,
    TaskParam* params,
    int32_t num_params
);
```

**Implementation Steps**:

1. **Sync TensorMap validity** from shared memory
2. **Allocate task slot** from task ring (may stall)
3. **Initialize TaskDescriptor**:
   - Set `fanout_count = scope_depth` (scope references)
   - Set `fanin_count = 0`
4. **Process INPUT params**:
   - Lookup producer in TensorMap
   - Add to fanin_list
   - Increment producer's fanout_count (with lock)
5. **Calculate total output size** and allocate packed buffer from heap ring
6. **Process OUTPUT params**:
   - Assign offsets within packed buffer
   - Register in TensorMap
7. **Check if immediately ready** (`fanin_count == 0`)
   - If ready, add to ready queue

**Complexity**: O(num_params)

### 6.4 API: pto_scope_end()

```c
void pto_scope_end(OrchestratorState* orch) {
    int32_t scope_begin_pos = orch->scope_stack[orch->scope_stack_top--];
    int32_t scope_end_pos = orch->task_ring.head;

    // Increment fanout_refcount for all tasks in [begin, end)
    for (int32_t task_id = scope_begin_pos; task_id < scope_end_pos; task_id++) {
        // Atomically increment fanout_refcount
        // Check if CONSUMED (fanout_refcount == fanout_count && state == COMPLETED)
    }
}
```

**Complexity**: O(scope_task_count)

### 6.5 Fanout Lock for Concurrent Access

The Orchestrator updates `fanout_count` when new consumers are added, while the Scheduler reads it to check CONSUMED condition. Use per-task spinlock:

```c
void acquire_fanout_lock(TaskDescriptor* task) {
    while (__sync_lock_test_and_set(&task->fanout_lock, 1)) {
        // spin
    }
}

void release_fanout_lock(TaskDescriptor* task) {
    __sync_lock_release(&task->fanout_lock);
}
```

**Deliverables**:
- [ ] `orchestrator.h` / `orchestrator.c`
- [ ] `pto_api.h` - Public API declarations

---

## 7. Phase 5: Scheduler Implementation

**Location**: `src/runtime/pto_orchestrator/scheduler/`

### 7.1 Scheduler State

```c
// File: scheduler.h

typedef struct ReadyQueue {
    int32_t* task_ids;
    int32_t head;
    int32_t tail;
    int32_t capacity;
} ReadyQueue;

typedef struct SchedulerState {
    // Shared memory access (read-only for task descriptors)
    SharedMemoryHeader* sm_header;
    TaskDescriptor* task_descriptors;
    DepListPool* dep_list_pool;

    // Local tracking
    int32_t last_task_alive;
    uint64_t heap_tail;

    // Per-task state (private)
    TaskState* task_state;
    int32_t* fanin_refcount;
    int32_t* fanout_refcount;

    // Ready queues (one per worker type)
    ReadyQueue ready_queues[NUM_WORKER_TYPES];
} SchedulerState;
```

### 7.2 API: pto_task_complete()

```c
void pto_task_complete(SchedulerState* sched, int32_t task_id) {
    TaskDescriptor* task = &sched->task_descriptors[task_id];

    sched->task_state[task_id] = TASK_COMPLETED;

    // Update consumers' fanin_refcount (make them ready)
    iterate_fanout_list(task, consumer_id) {
        sched->fanin_refcount[consumer_id]++;

        TaskDescriptor* consumer = &sched->task_descriptors[consumer_id];
        if (sched->fanin_refcount[consumer_id] == consumer->fanin_count &&
            sched->task_state[consumer_id] == TASK_PENDING) {
            sched->task_state[consumer_id] = TASK_READY;
            ready_queue_push(&sched->ready_queues[consumer->worker_type], consumer_id);
        }
    }

    // Update producers' fanout_refcount (for buffer lifecycle)
    iterate_fanin_list(task, producer_id) {
        sched->fanout_refcount[producer_id]++;

        TaskDescriptor* producer = &sched->task_descriptors[producer_id];
        if (sched->fanout_refcount[producer_id] == producer->fanout_count &&
            sched->task_state[producer_id] == TASK_COMPLETED) {
            sched->task_state[producer_id] = TASK_CONSUMED;
            advance_last_task_alive(sched);
        }
    }
}
```

### 7.3 Memory Reclamation

```c
void advance_last_task_alive(SchedulerState* sched) {
    // Advance until non-CONSUMED task found
    while (sched->last_task_alive < current_task_index &&
           sched->task_state[sched->last_task_alive % TASK_WINDOW_SIZE] == TASK_CONSUMED) {
        sched->last_task_alive++;
    }

    // Update heap_tail to follow last_task_alive
    if (sched->last_task_alive > 0) {
        TaskDescriptor* oldest = &sched->task_descriptors[
            (sched->last_task_alive - 1) % TASK_WINDOW_SIZE];
        sched->heap_tail = (uint64_t)oldest->packed_buffer_end;
    }

    // Write to shared memory for Orchestrator back-pressure
    sched->sm_header->last_task_alive = sched->last_task_alive;
    sched->sm_header->heap_tail = sched->heap_tail;
}
```

### 7.4 Scheduler Main Loop

```c
void scheduler_run(SchedulerState* sched) {
    while (!sched->sm_header->orchestrator_done || has_active_tasks(sched)) {
        // Poll for new tasks from Orchestrator
        process_new_tasks(sched);

        // Dispatch ready tasks to workers
        for (int wt = 0; wt < NUM_WORKER_TYPES; wt++) {
            while (!ready_queue_empty(&sched->ready_queues[wt])) {
                int32_t task_id = ready_queue_pop(&sched->ready_queues[wt]);
                dispatch_to_worker(sched, task_id, wt);
            }
        }

        // Process completed tasks (from worker callbacks)
        process_completed_tasks(sched);
    }
}
```

**Deliverables**:
- [ ] `scheduler.h` / `scheduler.c`
- [ ] `ready_queue.h` / `ready_queue.c`

---

## 8. Phase 6: Integration with Existing Runtime

### 8.1 Directory Structure

```
src/runtime/pto_orchestrator/
├── build_config.py
├── runtime/
│   ├── pto_runtime_types.h
│   ├── pto_runtime_config.h
│   ├── task_ring.h / task_ring.c
│   ├── heap_ring.h / heap_ring.c
│   ├── dep_list_pool.h / dep_list_pool.c
│   └── tensor_map.h / tensor_map.c
├── orchestrator/
│   ├── orchestrator.h / orchestrator.c
│   └── pto_api.h / pto_api.c
├── scheduler/
│   ├── scheduler.h / scheduler.c
│   └── ready_queue.h / ready_queue.c
├── host/
│   └── runtimemaker.cpp          # Reuse pattern from host_build_graph
├── aicpu/
│   └── runtimeexecutor.cpp       # Scheduler entry point
└── aicore/
    └── aicore_executor.cpp       # Worker execution (reuse from host_build_graph)
```

### 8.2 Build Configuration

```python
# build_config.py
BUILD_CONFIG = {
    "aicore": {
        "include_dirs": ["runtime", "scheduler"],
        "source_dirs": ["aicore", "runtime"]
    },
    "aicpu": {
        "include_dirs": ["runtime", "scheduler"],
        "source_dirs": ["aicpu", "runtime", "scheduler"]
    },
    "host": {
        "include_dirs": ["runtime", "orchestrator"],
        "source_dirs": ["host", "runtime", "orchestrator"]
    }
}
```

### 8.3 Integration Points

| Component | Reuse from `host_build_graph` | New Implementation |
|-----------|------------------------------|-------------------|
| Host loader | `runtimemaker.cpp` pattern | Orchestrator init |
| AICPU entry | `runtimeexecutor.cpp` pattern | Scheduler main loop |
| AICore worker | `aicore_executor.cpp` | Minimal changes |
| Handshake | Reuse existing | - |
| Python bindings | `runtime_bindings.py` | Add orchestrator APIs |

---

## 9. Phase 7: Testing & Validation

### 9.1 Unit Tests

| Component | Test Cases |
|-----------|-----------|
| Task Ring | Allocation, wrap-around, back-pressure stall |
| Heap Ring | Allocation, alignment, wrap-around, stall |
| TensorMap | Insert, lookup, lazy invalidation, chain truncation |
| Dep List Pool | Prepend, traversal, wrap-around |
| Orchestrator APIs | Scope nesting, task submission, dependency building |
| Scheduler | Ready queue, task dispatch, completion handling |

### 9.2 Integration Tests

1. **Simple Linear Chain**: A → B → C
2. **Diamond DAG**: A → B, A → C, B → D, C → D
3. **Nested Scopes**: Verify buffer lifecycle
4. **Back-Pressure**: Fill task ring, verify stall and recovery
5. **Large Graph**: 1000+ tasks with complex dependencies

### 9.3 Performance Benchmarks

| Metric | Target |
|--------|--------|
| Task ring alloc | ~5 cycles |
| Heap ring alloc | ~5 cycles |
| TensorMap insert | ~10 cycles |
| TensorMap lookup | ~20 cycles |
| Task submission | ~100 cycles |
| Task completion | ~50 cycles |

### 9.4 Example Orchestration

```c
// examples/pto_orchestrator_example/kernels/orchestration/matmul_orch.cpp

extern "C" int MatmulOrchestration(PTORuntime* rt, uint64_t* args) {
    float* A = (float*)args[0];
    float* B = (float*)args[1];
    float* C = (float*)args[2];
    int M = args[3], K = args[4], N = args[5];

    const int TILE = 128;

    for (int i = 0; i < M; i += TILE) {
        for (int j = 0; j < N; j += TILE) {
            pto_scope_begin(rt);

            for (int k = 0; k < K; k += TILE) {
                TaskParam params[3] = {
                    {PARAM_INPUT, &A[i*K + k], 0, 0, TILE*TILE*4},
                    {PARAM_INPUT, &B[k*N + j], 0, 0, TILE*TILE*4},
                    {PARAM_OUTPUT, NULL, 0, 0, TILE*TILE*4}
                };
                pto_submit_task(rt, KERNEL_GEMM, WORKER_CUBE, params, 3);
            }

            pto_scope_end(rt);
        }
    }

    return 0;
}
```

---

## 10. Configuration Constants

```c
// pto_runtime_config.h

#define TASK_WINDOW_SIZE        1024    // Power of 2
#define HEAP_SIZE               (64 * 1024 * 1024)  // 64 MB
#define DEP_LIST_POOL_SIZE      8192    // TASK_WINDOW_SIZE * AVG_FANOUT * 2
#define TENSORMAP_POOL_SIZE     4096    // TASK_WINDOW_SIZE * AVG_OUTPUTS * 2
#define TENSORMAP_NUM_BUCKETS   1024    // Power of 2
#define MAX_OUTPUTS             8
#define MAX_SCOPE_DEPTH         64
#define CACHE_LINE_SIZE         64
```

---

## 11. Memory Footprint Estimate

| Component | Size |
|-----------|------|
| Task Descriptors | ~128 KB (1024 × 128 bytes) |
| TensorMap Pool | ~128 KB (4096 × 32 bytes) |
| TensorMap Buckets | ~4 KB (1024 × 4 bytes) |
| Dep List Pool | ~32 KB (8192 × 4 bytes) |
| Scheduler State | ~64 KB (refcounts, queues) |
| **Total** | **~356 KB** |

---

## 12. Timeline & Milestones

| Phase | Duration | Milestone |
|-------|----------|-----------|
| Phase 1: Data Structures | 1 week | Type definitions complete |
| Phase 2: Ring Buffers | 1 week | Ring buffer tests passing |
| Phase 3: TensorMap | 1 week | TensorMap tests passing |
| Phase 4: Orchestrator APIs | 2 weeks | API implementation complete |
| Phase 5: Scheduler | 2 weeks | Scheduler implementation complete |
| Phase 6: Integration | 1 week | End-to-end example working |
| Phase 7: Testing | 2 weeks | All tests passing, benchmarks met |

**Total Estimated Duration**: 10 weeks

---

## 13. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Fanout lock contention | Per-task lock, contention ~1/1024 |
| TensorMap chain length | Periodic cleanup, chain truncation |
| Ring buffer sizing | Conservative sizing with monitoring |
| Orchestrator-Scheduler latency | Batch updates, minimize shared memory writes |
| Back-pressure deadlock | Careful sizing of task/heap rings |

---

## 14. Success Criteria

1. All 4 APIs implemented and tested
2. Ring buffer allocation < 10 cycles average
3. TensorMap lookup < 30 cycles average
4. End-to-end matmul example working
5. No memory leaks under sustained workload
6. Back-pressure correctly prevents OOM