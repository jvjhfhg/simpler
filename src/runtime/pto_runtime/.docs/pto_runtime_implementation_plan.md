# PTO Runtime Implementation Plan

This document outlines the implementation plan for a new PTO (Parallel Task Orchestration) runtime based on the existing `host_build_graph` framework. The PTO runtime introduces ring buffer-based memory management, dynamic task submission, and scope-based buffer lifecycle tracking.

## 1. Overview

### 1.1 Current State (`host_build_graph`)

The existing runtime uses:
- **Static task graph**: All tasks and dependencies built upfront in orchestration, then executed
- **Fixed-size arrays**: `Task tasks[RUNTIME_MAX_TASKS]` with compile-time bounds
- **Simple scheduling**: AICPU dispatches from ready queues, tracks fanin/fanout
- **Manual memory**: Host allocates device memory, orchestration records tensor pairs for copy-back

### 1.2 Target State (`pto_runtime`)

The PTO runtime introduces:
- **Dynamic task submission**: Tasks submitted during execution via `pto_submit_task()`
- **Ring buffer architecture**: O(1) allocation, implicit deallocation, zero fragmentation
- **Scope-based lifecycle**: Buffers freed when all consumers + enclosing scopes complete
- **TensorMap for dependencies**: Automatic producer-consumer tracking via memory region lookup
- **Decoupled Orchestrator/Scheduler**: Communicate only via shared memory pointers

### 1.3 Key Differences

| Aspect | `host_build_graph` | `pto_runtime` |
|--------|-------------------|---------------|
| Task creation | Upfront in orchestration | Dynamic via `pto_submit_task()` |
| Memory allocation | Host `DeviceMalloc` | Ring buffer (`HeapRing`) |
| Dependency tracking | Manual `add_successor()` | Automatic via `TensorMap` |
| Buffer lifecycle | Manual `DeviceFree` | Scope + refcount based |
| Scheduling | Post-build execution | Concurrent with submission |

---

## 2. Directory Structure

```
src/runtime/pto_runtime/
├── build_config.py              # RuntimeBuilder configuration
├── runtime/
│   ├── pto_runtime.h            # Core data structures
│   ├── pto_runtime.cpp          # Shared implementation
│   ├── ring_buffer.h            # Ring buffer primitives
│   ├── tensor_map.h             # TensorMap for dependency tracking
│   └── dep_list_pool.h          # Dependency list pool
├── host/
│   └── pto_runtime_maker.cpp    # Host-side init/finalize
├── orchestrator/
│   └── pto_orchestrator.cpp     # Orchestrator (can run on Host or AICPU)
├── scheduler/
│   └── pto_scheduler.cpp        # AICPU scheduler
└── aicore/
    └── pto_aicore_executor.cpp  # AICore worker
```

---

## 3. Implementation Phases

### Phase 1: Core Data Structures
### Phase 2: Ring Buffer Infrastructure
### Phase 3: TensorMap Implementation
### Phase 4: Orchestrator Component
### Phase 5: Scheduler Component
### Phase 6: AICore Worker (Reuse)
### Phase 7: Integration & Testing

---

## 4. Phase 1: Core Data Structures

### 4.1 File: `runtime/pto_runtime.h`

#### 4.1.1 Configuration Constants

```cpp
// Ring buffer sizes (must be power of 2 for efficient modulo)
#define PTO_TASK_WINDOW_SIZE      1024
#define PTO_HEAP_SIZE             (64 * 1024 * 1024)  // 64MB
#define PTO_DEP_LIST_POOL_SIZE    8192
#define PTO_TENSORMAP_POOL_SIZE   4096
#define PTO_TENSORMAP_NUM_BUCKETS 1024

// Task limits
#define PTO_MAX_ARGS              16
#define PTO_MAX_WORKER            72
#define PTO_MAX_SCOPE_DEPTH       32

// Worker types
#define PTO_WORKER_CUBE           0
#define PTO_WORKER_VECTOR         1
#define PTO_WORKER_AICPU          2
#define PTO_NUM_WORKER_TYPES      3
```

#### 4.1.2 Task States

```cpp
enum TaskState : int32_t {
    TASK_PENDING   = 0,  // Waiting for dependencies
    TASK_READY     = 1,  // All dependencies met, in ready queue
    TASK_RUNNING   = 2,  // Executing on worker
    TASK_COMPLETED = 3,  // Execution done, may have live consumers
    TASK_CONSUMED  = 4   // All consumers done, buffer can be freed
};
```

#### 4.1.3 Shared Memory Header

```cpp
// Flow control between Orchestrator and Scheduler
struct alignas(64) PTOSharedHeader {
    // Written by Orchestrator, read by Scheduler
    volatile int32_t current_task_index;  // Next task slot to allocate
    volatile int32_t heap_top;            // Current heap allocation pointer
    volatile int32_t orchestrator_done;   // 1 when orchestration complete

    char pad1[64 - 12];

    // Written by Scheduler, read by Orchestrator
    volatile int32_t last_task_alive;     // Oldest non-CONSUMED task
    volatile int32_t heap_tail;           // Oldest live buffer start
    volatile int32_t scheduler_done;      // 1 when all tasks complete

    char pad2[64 - 12];
};
```

#### 4.1.4 Task Descriptor

```cpp
struct PTOTaskDescriptor {
    int32_t task_id;
    int32_t func_id;
    int32_t worker_type;              // PTO_WORKER_CUBE, VECTOR, or AICPU
    int32_t num_args;

    uint64_t args[PTO_MAX_ARGS];
    uint64_t function_bin_addr;

    // Dependency counts (set once at submission)
    int32_t fanin_count;              // Number of producer tasks
    int32_t fanout_count;             // Number of consumers + scope_depth

    // Linked list heads (offsets into DepListPool)
    int32_t fanin_head;               // List of producer task IDs
    int32_t fanout_head;              // List of consumer task IDs

    // Output buffer info (for heap reclamation)
    int32_t packed_buffer_offset;     // Offset in HeapRing
    int32_t packed_buffer_size;       // Total size of packed outputs

    // Spinlock for fanout modification
    volatile int32_t fanout_lock;
};
```

#### 4.1.5 Scheduler-Private State

```cpp
struct PTOSchedulerState {
    // Per-task state (indexed by task_id % PTO_TASK_WINDOW_SIZE)
    TaskState task_state[PTO_TASK_WINDOW_SIZE];
    int32_t fanin_refcount[PTO_TASK_WINDOW_SIZE];   // Completed producers
    int32_t fanout_refcount[PTO_TASK_WINDOW_SIZE];  // Completed consumers + scopes

    // Ready queues per worker type
    int32_t ready_queue[PTO_NUM_WORKER_TYPES][PTO_TASK_WINDOW_SIZE];
    int32_t ready_head[PTO_NUM_WORKER_TYPES];
    int32_t ready_tail[PTO_NUM_WORKER_TYPES];
};
```

#### 4.1.6 Handshake (Reuse from `host_build_graph`)

```cpp
// Reuse existing Handshake struct - no changes needed
struct Handshake {
    volatile uint32_t aicpu_ready;
    volatile uint32_t aicore_done;
    volatile uint64_t task;
    volatile int32_t task_status;
    volatile int32_t control;
    volatile int32_t core_type;
} __attribute__((aligned(64)));
```

---

## 5. Phase 2: Ring Buffer Infrastructure

### 5.1 File: `runtime/ring_buffer.h`

#### 5.1.1 Task Ring

```cpp
struct TaskRing {
    PTOTaskDescriptor* base;
    int32_t size;        // PTO_TASK_WINDOW_SIZE
    int32_t head;        // Next slot to allocate (Orchestrator)
    // tail is last_task_alive in shared header (Scheduler)
};

// Allocate task slot (may stall if ring full)
inline int32_t task_ring_alloc(TaskRing* ring, volatile int32_t* tail_ptr) {
    while (true) {
        int32_t head = ring->head;
        int32_t tail = *tail_ptr;

        // Check if ring has space (leave 1 slot empty to distinguish full/empty)
        int32_t used = (head - tail + ring->size) % ring->size;
        if (used < ring->size - 1) {
            ring->head = (head + 1) % ring->size;
            return head;
        }
        // Ring full, spin-wait for Scheduler to advance tail
    }
}
```

#### 5.1.2 Heap Ring

```cpp
struct HeapRing {
    char* base;
    int32_t size;        // PTO_HEAP_SIZE
    int32_t top;         // Next allocation offset (Orchestrator)
    // tail is heap_tail in shared header (Scheduler)
};

// Allocate buffer from heap (may stall if insufficient space)
// Never splits allocation across ring boundary
inline void* heap_ring_alloc(HeapRing* ring, int32_t alloc_size,
                             volatile int32_t* tail_ptr) {
    alloc_size = ALIGN_UP(alloc_size, 64);  // Cache-line align

    while (true) {
        int32_t tail = *tail_ptr;
        int32_t top = ring->top;

        if (top >= tail) {
            // [....tail====top......]
            int32_t space_at_end = ring->size - top;
            if (space_at_end >= alloc_size) {
                ring->top = top + alloc_size;
                return ring->base + top;
            }
            // Try wrap to beginning
            if (tail > alloc_size) {
                ring->top = alloc_size;
                return ring->base;
            }
        } else {
            // [====top....tail=====]
            int32_t gap = tail - top;
            if (gap >= alloc_size) {
                ring->top = top + alloc_size;
                return ring->base + top;
            }
        }
        // Insufficient space, spin-wait for Scheduler to advance tail
    }
}
```

#### 5.1.3 Dependency List Pool

```cpp
struct DepListEntry {
    int32_t task_id;
    int32_t next_offset;  // 0 = end of list
};

struct DepListPool {
    DepListEntry* base;
    int32_t size;         // PTO_DEP_LIST_POOL_SIZE
    int32_t top;          // Next slot to allocate
};

// Prepend entry to linked list (O(1))
inline int32_t dep_list_prepend(DepListPool* pool, int32_t current_head,
                                int32_t task_id) {
    int32_t new_offset = pool->top++;
    if (pool->top >= pool->size) {
        pool->top = 0;  // Wrap around
    }

    DepListEntry* entry = &pool->base[new_offset];
    entry->task_id = task_id;
    entry->next_offset = current_head;
    return new_offset + 1;  // +1 so 0 means empty list
}
```

---

## 6. Phase 3: TensorMap Implementation

### 6.1 File: `runtime/tensor_map.h`

#### 6.1.1 TensorMap Entry

```cpp
struct TensorMapEntry {
    uint64_t base_ptr;           // Hash key (tensor base address)
    int32_t offset;              // Byte offset within tensor
    int32_t size;                // Region size in bytes
    int32_t producer_task_id;    // Task that produces this region
    int32_t next_in_bucket;      // Chain for hash collisions
    int32_t next_in_task;        // Chain for same-task outputs
    uint8_t in_bucket;           // Flag: entry is in a bucket chain
    uint8_t padding[3];
};
```

#### 6.1.2 TensorMap Structure

```cpp
struct TensorMap {
    TensorMapEntry* pool;        // Ring buffer of entries
    int32_t pool_size;           // PTO_TENSORMAP_POOL_SIZE
    int32_t pool_head;           // Next slot to allocate

    int32_t* buckets;            // Hash table buckets
    int32_t num_buckets;         // PTO_TENSORMAP_NUM_BUCKETS

    int32_t* task_heads;         // Per-task output list heads
    int32_t last_cleanup_task;   // For periodic cleanup
};
```

#### 6.1.3 Key Operations

```cpp
// Hash by base_ptr only (enables overlap detection)
inline int32_t tensormap_hash(TensorMap* tm, uint64_t base_ptr) {
    return (base_ptr >> 6) % tm->num_buckets;  // Shift for cache-line alignment
}

// Check if two regions overlap
inline bool regions_overlap(int32_t off1, int32_t size1,
                           int32_t off2, int32_t size2) {
    return (off1 < off2 + size2) && (off2 < off1 + size1);
}

// Insert output region (called during pto_submit_task)
void tensormap_insert(TensorMap* tm, uint64_t base_ptr, int32_t offset,
                     int32_t size, int32_t producer_task_id);

// Lookup producer for input region (returns -1 if none found)
// Truncates chain on first stale entry (task_id < last_task_alive)
int32_t tensormap_lookup(TensorMap* tm, uint64_t base_ptr, int32_t offset,
                        int32_t size, int32_t last_task_alive);

// Periodic cleanup (every N retired tasks)
void tensormap_cleanup(TensorMap* tm, int32_t last_task_alive);
```

---

## 7. Phase 4: Orchestrator Component

### 7.1 File: `orchestrator/pto_orchestrator.cpp`

#### 7.1.1 Orchestrator State

```cpp
struct PTOOrchestrator {
    // Shared memory pointers
    PTOSharedHeader* shared_header;
    PTOTaskDescriptor* task_ring_base;
    DepListPool dep_list_pool;

    // Private state
    TensorMap tensor_map;
    int32_t scope_stack[PTO_MAX_SCOPE_DEPTH];
    int32_t scope_stack_top;

    // Ring buffer state
    TaskRing task_ring;
    HeapRing heap_ring;
};
```

#### 7.1.2 Runtime API Implementation

```cpp
// Push scope marker
void pto_scope_begin(PTOOrchestrator* orch) {
    orch->scope_stack[++orch->scope_stack_top] = orch->task_ring.head;
}

// Pop scope, decrement fanout for all tasks in range
void pto_scope_end(PTOOrchestrator* orch) {
    int32_t begin_pos = orch->scope_stack[orch->scope_stack_top--];
    int32_t end_pos = orch->task_ring.head;

    // Signal Scheduler to decrement fanout_refcount for [begin, end)
    // Implementation: write scope_end message to shared memory queue
    // or inline the decrement if Orchestrator runs on same memory domain
}

// Submit task with automatic dependency detection
int32_t pto_submit_task(PTOOrchestrator* orch,
                        int32_t func_id,
                        int32_t worker_type,
                        PTOParam* params,
                        int32_t param_count);
```

#### 7.1.3 Task Submission Flow

```cpp
int32_t pto_submit_task(PTOOrchestrator* orch, int32_t func_id,
                        int32_t worker_type, PTOParam* params,
                        int32_t param_count) {
    // 1. Sync TensorMap validity (read last_task_alive)
    int32_t last_alive = orch->shared_header->last_task_alive;

    // 2. Allocate task slot
    int32_t slot = task_ring_alloc(&orch->task_ring,
                                   &orch->shared_header->last_task_alive);
    PTOTaskDescriptor* task = &orch->task_ring_base[slot];
    task->task_id = slot;  // Or monotonic counter if needed
    task->func_id = func_id;
    task->worker_type = worker_type;

    // 3. First pass: process INPUT params, build fanin list
    int32_t fanin_head = 0;
    int32_t fanin_count = 0;

    for (int i = 0; i < param_count; i++) {
        if (params[i].type == PTO_INPUT) {
            int32_t producer = tensormap_lookup(&orch->tensor_map,
                                                params[i].base_ptr,
                                                params[i].offset,
                                                params[i].size,
                                                last_alive);
            if (producer >= 0) {
                // Add to fanin list
                fanin_head = dep_list_prepend(&orch->dep_list_pool,
                                             fanin_head, producer);
                fanin_count++;

                // Update producer's fanout (with spinlock)
                PTOTaskDescriptor* prod_task = &orch->task_ring_base[producer];
                spinlock_acquire(&prod_task->fanout_lock);
                prod_task->fanout_head = dep_list_prepend(&orch->dep_list_pool,
                                                          prod_task->fanout_head,
                                                          slot);
                prod_task->fanout_count++;
                spinlock_release(&prod_task->fanout_lock);
            }
        }
    }

    // 4. Calculate total output size
    int32_t total_output_size = 0;
    for (int i = 0; i < param_count; i++) {
        if (params[i].type == PTO_OUTPUT) {
            total_output_size += ALIGN_UP(params[i].size, 64);
        }
    }

    // 5. Allocate packed output buffer from HeapRing
    void* packed_buffer = NULL;
    if (total_output_size > 0) {
        packed_buffer = heap_ring_alloc(&orch->heap_ring, total_output_size,
                                        &orch->shared_header->heap_tail);
        task->packed_buffer_offset = (char*)packed_buffer - orch->heap_ring.base;
        task->packed_buffer_size = total_output_size;
    }

    // 6. Second pass: register OUTPUTs, write back addresses
    int32_t output_offset = 0;
    int32_t arg_idx = 0;

    for (int i = 0; i < param_count; i++) {
        if (params[i].type == PTO_OUTPUT) {
            // Compute address within packed buffer
            void* output_addr = (char*)packed_buffer + output_offset;

            // Write back to caller's pointer
            if (params[i].ptr_to_ptr != NULL) {
                *params[i].ptr_to_ptr = output_addr;
            }

            // Register in TensorMap
            tensormap_insert(&orch->tensor_map,
                            (uint64_t)output_addr,  // base_ptr
                            0,                       // offset (start of region)
                            params[i].size,
                            slot);

            // Add to task args
            task->args[arg_idx++] = (uint64_t)output_addr;
            output_offset += ALIGN_UP(params[i].size, 64);
        } else {
            // INPUT: add address directly to args
            task->args[arg_idx++] = params[i].base_ptr + params[i].offset;
        }
    }
    task->num_args = arg_idx;

    // 7. Initialize dependency counts
    task->fanin_head = fanin_head;
    task->fanin_count = fanin_count;
    task->fanout_head = 0;
    task->fanout_count = orch->scope_stack_top + 1;  // scope_depth
    task->fanout_lock = 0;

    // 8. Update shared header (makes task visible to Scheduler)
    __sync_synchronize();  // Memory barrier
    orch->shared_header->current_task_index = orch->task_ring.head;

    return slot;
}
```

---

## 8. Phase 5: Scheduler Component

### 8.1 File: `scheduler/pto_scheduler.cpp`

#### 8.1.1 Scheduler State

```cpp
struct PTOScheduler {
    // Shared memory
    PTOSharedHeader* shared_header;
    PTOTaskDescriptor* task_ring_base;
    DepListPool* dep_list_pool;

    // Private state
    PTOSchedulerState state;

    // Handshake buffers (reuse from host_build_graph)
    Handshake* workers;
    int32_t worker_count;
};
```

#### 8.1.2 Main Scheduler Loop

```cpp
void pto_scheduler_run(PTOScheduler* sched) {
    int32_t local_task_index = 0;

    while (true) {
        // Check for new tasks from Orchestrator
        int32_t current = sched->shared_header->current_task_index;
        while (local_task_index < current) {
            PTOTaskDescriptor* task = &sched->task_ring_base[local_task_index];
            int32_t slot = local_task_index % PTO_TASK_WINDOW_SIZE;

            // Initialize scheduler state for this task
            sched->state.task_state[slot] = TASK_PENDING;
            sched->state.fanin_refcount[slot] = 0;
            sched->state.fanout_refcount[slot] = 0;

            // Check if immediately ready
            if (task->fanin_count == 0) {
                sched->state.task_state[slot] = TASK_READY;
                ready_queue_push(sched, task->worker_type, local_task_index);
            }

            local_task_index++;
        }

        // Process completions from workers
        for (int i = 0; i < sched->worker_count; i++) {
            Handshake* h = &sched->workers[i];

            if (h->task_status == 0 && h->task != 0) {
                PTOTaskDescriptor* completed = (PTOTaskDescriptor*)h->task;
                h->task = 0;

                pto_task_complete(sched, completed->task_id);
            }
        }

        // Dispatch ready tasks to idle workers
        for (int i = 0; i < sched->worker_count; i++) {
            Handshake* h = &sched->workers[i];

            if (h->task_status == 0 && h->task == 0) {
                int32_t worker_type = h->core_type;
                int32_t task_id = ready_queue_pop(sched, worker_type);

                if (task_id >= 0) {
                    PTOTaskDescriptor* task = &sched->task_ring_base[task_id];
                    int32_t slot = task_id % PTO_TASK_WINDOW_SIZE;

                    sched->state.task_state[slot] = TASK_RUNNING;
                    h->task = (uint64_t)task;
                    h->task_status = 1;
                }
            }
        }

        // Check termination
        if (sched->shared_header->orchestrator_done &&
            all_tasks_consumed(sched, local_task_index)) {
            break;
        }
    }

    sched->shared_header->scheduler_done = 1;
}
```

#### 8.1.3 Task Completion Handler

```cpp
void pto_task_complete(PTOScheduler* sched, int32_t task_id) {
    PTOTaskDescriptor* task = &sched->task_ring_base[task_id];
    int32_t slot = task_id % PTO_TASK_WINDOW_SIZE;

    // Mark completed
    sched->state.task_state[slot] = TASK_COMPLETED;

    // Update fanin_refcount of all consumers (may make them ready)
    int32_t fanout_offset = task->fanout_head;
    while (fanout_offset != 0) {
        DepListEntry* entry = &sched->dep_list_pool->base[fanout_offset - 1];
        int32_t consumer_id = entry->task_id;
        int32_t consumer_slot = consumer_id % PTO_TASK_WINDOW_SIZE;

        sched->state.fanin_refcount[consumer_slot]++;

        PTOTaskDescriptor* consumer = &sched->task_ring_base[consumer_id];
        if (sched->state.fanin_refcount[consumer_slot] == consumer->fanin_count) {
            sched->state.task_state[consumer_slot] = TASK_READY;
            ready_queue_push(sched, consumer->worker_type, consumer_id);
        }

        fanout_offset = entry->next_offset;
    }

    // Update fanout_refcount of all producers (for buffer lifecycle)
    int32_t fanin_offset = task->fanin_head;
    while (fanin_offset != 0) {
        DepListEntry* entry = &sched->dep_list_pool->base[fanin_offset - 1];
        int32_t producer_id = entry->task_id;
        int32_t producer_slot = producer_id % PTO_TASK_WINDOW_SIZE;

        sched->state.fanout_refcount[producer_slot]++;

        PTOTaskDescriptor* producer = &sched->task_ring_base[producer_id];
        if (sched->state.fanout_refcount[producer_slot] == producer->fanout_count &&
            sched->state.task_state[producer_slot] == TASK_COMPLETED) {
            sched->state.task_state[producer_slot] = TASK_CONSUMED;
            scheduler_on_task_consumed(sched, producer_id);
        }

        fanin_offset = entry->next_offset;
    }

    // Check if this task itself is now CONSUMED
    if (sched->state.fanout_refcount[slot] == task->fanout_count) {
        sched->state.task_state[slot] = TASK_CONSUMED;
        scheduler_on_task_consumed(sched, task_id);
    }
}
```

#### 8.1.4 Memory Reclamation

```cpp
void scheduler_on_task_consumed(PTOScheduler* sched, int32_t task_id) {
    // Advance last_task_alive as far as possible
    while (sched->state.task_state[sched->shared_header->last_task_alive %
                                   PTO_TASK_WINDOW_SIZE] == TASK_CONSUMED) {
        int32_t consumed_id = sched->shared_header->last_task_alive;
        PTOTaskDescriptor* task = &sched->task_ring_base[consumed_id];

        // Update heap_tail based on this task's buffer end
        int32_t buffer_end = task->packed_buffer_offset + task->packed_buffer_size;
        if (buffer_end > sched->shared_header->heap_tail) {
            sched->shared_header->heap_tail = buffer_end;
        }

        sched->shared_header->last_task_alive++;
    }
}
```

---

## 9. Phase 6: AICore Worker

The AICore worker can be **reused from `host_build_graph`** with minimal changes:

```cpp
// aicore/pto_aicore_executor.cpp
// Identical to host_build_graph/aicore/aicore_executor.cpp
// The unified kernel signature (void kernel(__gm__ int64_t* args))
// works with both runtime designs.
```

Key points:
- Handshake protocol unchanged
- Task structure field names may differ (`function_bin_addr` vs `functionBinAddr`)
- Kernel dispatch via function pointer remains the same

---

## 10. Phase 7: Host Component

### 10.1 File: `host/pto_runtime_maker.cpp`

```cpp
extern "C" int InitPTORuntime(
    PTOSharedHeader* shared_header,
    PTOTaskDescriptor* task_ring,
    char* heap_base,
    DepListEntry* dep_list_pool,
    Handshake* workers,
    const char* orch_so_binary,
    size_t orch_so_len,
    const char* orch_func_name,
    uint64_t* func_args,
    int func_arg_count
) {
    // 1. Initialize shared header
    shared_header->current_task_index = 0;
    shared_header->heap_top = 0;
    shared_header->orchestrator_done = 0;
    shared_header->last_task_alive = 0;
    shared_header->heap_tail = 0;
    shared_header->scheduler_done = 0;

    // 2. Load orchestration .so (same as host_build_graph)
    int fd = memfd_create("orch_so", MFD_CLOEXEC);
    write(fd, orch_so_binary, orch_so_len);
    char path[64];
    snprintf(path, sizeof(path), "/proc/self/fd/%d", fd);
    void* handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);

    // 3. Resolve and call orchestration function
    typedef int (*OrchFunc)(PTOOrchestrator*, uint64_t*, int);
    OrchFunc orch_func = (OrchFunc)dlsym(handle, orch_func_name);

    // 4. Setup orchestrator context
    PTOOrchestrator orch;
    orch.shared_header = shared_header;
    orch.task_ring_base = task_ring;
    orch.task_ring.base = task_ring;
    orch.task_ring.size = PTO_TASK_WINDOW_SIZE;
    orch.task_ring.head = 0;
    orch.heap_ring.base = heap_base;
    orch.heap_ring.size = PTO_HEAP_SIZE;
    orch.heap_ring.top = 0;
    orch.dep_list_pool.base = dep_list_pool;
    orch.dep_list_pool.size = PTO_DEP_LIST_POOL_SIZE;
    orch.dep_list_pool.top = 0;
    orch.scope_stack_top = -1;

    tensormap_init(&orch.tensor_map);

    // 5. Execute orchestration (builds task graph dynamically)
    int result = orch_func(&orch, func_args, func_arg_count);

    // 6. Signal orchestration complete
    shared_header->orchestrator_done = 1;

    close(fd);
    dlclose(handle);
    return result;
}

extern "C" int FinalizePTORuntime(PTOSharedHeader* shared_header) {
    // Wait for scheduler to complete
    while (!shared_header->scheduler_done) {
        // Spin or sleep
    }
    return 0;
}
```

---

## 11. Build Configuration

### 11.1 File: `build_config.py`

```python
BUILD_CONFIG = {
    "aicore": {
        "include_dirs": ["runtime"],
        "source_dirs": ["aicore", "runtime"]
    },
    "aicpu": {
        "include_dirs": ["runtime"],
        "source_dirs": ["scheduler", "runtime"]
    },
    "host": {
        "include_dirs": ["runtime"],
        "source_dirs": ["host", "orchestrator", "runtime"]
    }
}
```

---

## 12. Migration Considerations

### 12.1 What to Reuse from `host_build_graph`

| Component | Reusability | Notes |
|-----------|-------------|-------|
| `Handshake` struct | 100% | No changes needed |
| `HostApi` struct | 100% | Device memory abstraction |
| `TensorPair` struct | 100% | For copy-back |
| AICore worker | 95% | Minor field name changes |
| Handshake protocol | 100% | Same polling-based sync |
| Kernel signature | 100% | `void kernel(__gm__ int64_t* args)` |

### 12.2 What Must Be Rewritten

| Component | Reason |
|-----------|--------|
| Task struct | Add ring buffer support, different dependency model |
| Dependency tracking | TensorMap-based vs manual `add_successor()` |
| Memory allocation | Ring buffers vs host `DeviceMalloc` |
| Scheduling loop | Concurrent with orchestration vs post-build |

### 12.3 API Mapping

| `host_build_graph` | `pto_runtime` |
|-------------------|---------------|
| `runtime.add_task()` | `pto_submit_task()` |
| `runtime.add_successor()` | Automatic via TensorMap |
| `host_api.DeviceMalloc()` | `heap_ring_alloc()` (internal) |
| `host_api.DeviceFree()` | Implicit via `last_task_alive` |
| N/A | `pto_scope_begin()` / `pto_scope_end()` |

---

## 13. Testing Strategy

### 13.1 Unit Tests

1. **Ring buffer tests**: Allocation, wrap-around, back-pressure
2. **TensorMap tests**: Insert, lookup, overlap detection, lazy invalidation
3. **Dependency tests**: Fanin/fanout tracking, scope-based lifecycle

### 13.2 Integration Tests

1. **Linear pipeline**: A → B → C (no parallelism)
2. **Diamond DAG**: A → B, A → C, B → D, C → D
3. **Nested scopes**: Verify buffer lifetime across scope boundaries
4. **Back-pressure**: Submit tasks faster than execution, verify stall behavior

### 13.3 Performance Tests

1. **Allocation throughput**: Tasks/sec, bytes/sec
2. **Memory efficiency**: Peak vs average heap usage
3. **Scheduling latency**: Time from task submission to dispatch

---

## 14. Implementation Order

| Step | Component | Estimated Effort | Dependencies |
|------|-----------|------------------|--------------|
| 1 | `pto_runtime.h` (data structures) | 1 day | None |
| 2 | `ring_buffer.h` | 1 day | Step 1 |
| 3 | `dep_list_pool.h` | 0.5 day | Step 1 |
| 4 | `tensor_map.h` | 2 days | Step 1 |
| 5 | `pto_orchestrator.cpp` | 2 days | Steps 1-4 |
| 6 | `pto_scheduler.cpp` | 2 days | Steps 1-3 |
| 7 | `pto_aicore_executor.cpp` | 0.5 day | Step 1 |
| 8 | `pto_runtime_maker.cpp` | 1 day | Steps 5-7 |
| 9 | `build_config.py` | 0.5 day | None |
| 10 | Unit tests | 2 days | Steps 1-4 |
| 11 | Integration tests | 2 days | Steps 5-8 |
| 12 | Example application | 1 day | All |

---

## 15. Open Questions

1. **Orchestrator location**: Should run on Host (current plan) or Device AICPU?
   - Host: Simpler debugging, PCIe latency for shared memory
   - AICPU: Lower latency, more complex deployment

2. **Scope semantics**: Should `scope_end()` be synchronous (wait for all tasks in scope) or asynchronous (just decrement refcounts)?
   - Current plan: Asynchronous (matches document)

3. **Multiple outputs per task**: Pack into single buffer (current plan) or allocate separately?
   - Single packed buffer reduces allocation count but may waste space on alignment

4. **TensorMap cleanup frequency**: Every 64 retired tasks (per document) or adaptive?

---

## 16. References

- [implementing_new_runtime.md](implementing_new_runtime.md) - Framework guide
- [runtime_buffer_manager_comprehensive_summary.md](runtime_buffer_manager_comprehensive_summary.md) - Target design
- [divergence-to-original-orchestration.md](divergence-to-original-orchestration.md) - Design divergences
- `src/runtime/host_build_graph/` - Reference implementation