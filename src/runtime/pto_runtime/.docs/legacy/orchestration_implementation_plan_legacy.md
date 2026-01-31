# Orchestration Function Implementation Plan

## Overview

Add orchestration function support to PTO Runtime enabling dynamic task submission with Turing-complete control flow (loops, conditionals, recursion, function nesting).

**Constraint**: All modifications confined to `src/runtime/` directory (plugin architecture).

---

## Current System vs Target

**Current**: Static task graph - all tasks defined before execution starts
**Target**: Dynamic task submission - tasks created during execution with automatic dependency resolution

---

## Memory Architecture on Ascend

**Key Constraints:**
- Host and Device have **separate address spaces** (different physical memory)
- Host allocates device memory via `rtMalloc/rtFree` (wrapped in `MemoryAllocator`)
- AICPU can `malloc` on itself for local data structures
- **AICore has NO malloc capability** - only reads/writes pre-allocated device memory

**Strategy**:
- Allocate large device memory heap once (via MemoryAllocator) during initialization
- Manage as ring buffer for task output buffers (~5 cycles per allocation)
- TensorMap and other metadata in host or AICPU memory

---

## Core Design

### Six APIs (called by orchestration functions)

**Memory Management:**
```c
void* pto_alloc(Runtime* rt, size_t size);
void pto_free(Runtime* rt, void* buffer);
```

**Task & Scope Management:**
```c
void pto_scope_begin(Runtime* rt);
int32_t pto_submit_task(Runtime* rt, int32_t kernel_id, int32_t worker_type,
                        TaskParam* params, int32_t num_params);
void pto_scope_end(Runtime* rt);
void pto_task_complete(Runtime* rt, int32_t task_id);
```

### Key Data Structures

**TaskDescriptor** - Task metadata with dependency tracking
```cpp
struct TaskDescriptor {
    int32_t task_id, func_id, core_type;
    atomic<int32_t> fanin_remaining;  // Dependencies countdown
    int32_t fanout_count;             // Consumers + scope_depth
    uint64_t packed_buffer_base;      // Device memory for all outputs
    atomic<TaskState> state;          // PENDING/READY/RUNNING/COMPLETED/CONSUMED
};
```

**TensorMap** - Hash table mapping buffer addresses → producer task IDs
- Enables automatic dependency discovery
- Lazy invalidation (no explicit removal)
- O(1) insert, O(valid_entries) lookup

**HeapRing** - Device memory ring buffer
```cpp
struct HeapRing {
    void* base_ptr;      // Device memory (from rtMalloc)
    uint64_t head, tail; // Allocation pointers
};
```

**ScopeStack** - Tracks task ranges for lifetime management

**BufferTracker** - Tracks explicitly allocated buffers
```cpp
struct BufferInfo {
    void* base_ptr;
    size_t size;
    int32_t writer_count;          // Number of tasks writing to this buffer
    int32_t* writer_task_ids;      // Task IDs that write to this buffer
    bool free_requested;           // pto_free() called, no more writers will be added
};
```

---

## Implementation (src/runtime/ only)

### 1. Core Runtime (`src/runtime/runtime/`)

**New Files:**
- `runtime_api.h/cpp` - Six orchestration APIs
- `tensormap.h/cpp` - Producer lookup hash table
- `ring_buffer.h/cpp` - Ring allocation helpers
- `buffer_tracker.h/cpp` - Explicit buffer tracking

**Modified Files:**
- `runtime.h` - Add orchestration data structures to Runtime class
- `runtime.cpp` - Add `init_orchestration()` method

**Runtime Class Extensions:**
```cpp
class Runtime {
    // NEW: Orchestration support
    TaskDescriptor* task_descriptors;  // Device memory (shared)
    HeapRing heap_ring;                // Device memory for buffers
    TensorMap tensor_map;              // Host/AICPU memory
    ScopeStack scope_stack;            // Host/AICPU memory
    BufferTracker buffer_tracker;      // Host/AICPU memory (explicit buffers)
    SchedulerState* scheduler_state;   // Device memory (shared)
    MemoryAllocator* mem_allocator;    // From platform layer

    int init_orchestration(int32_t max_tasks, uint64_t heap_size);
};
```

### 2. AICPU Scheduler (`src/runtime/aicpu/`)

**Modified Files:**
- `aicpu_executor.cpp` - Replace static graph traversal with dynamic ready queues

**Changes:**
```cpp
// Current: Iterate predefined task graph
// New: Poll per-worker-type ready queues
while (not_done) {
    if (ready_queue_aic has tasks) dispatch_to_aicore();
    if (ready_queue_aiv has tasks) dispatch_to_aivector();
    if (ready_queue_cpu has tasks) execute_orchestration();
}
```

### 3. Host Runtime (`src/runtime/host/`)

**Optional**: Add initialization helper if needed for device memory setup

### 4. AICore (`src/runtime/aicore/`)

**No changes needed** - AICore kernels just execute on pre-allocated buffers

---

## Key Algorithms

### pto_alloc() - Explicit Buffer Allocation

```
1. Allocate from heap ring (may stall if full)
2. Create BufferInfo entry:
   - base_ptr = allocated address
   - writer_count = 0
   - free_requested = false
3. Register in BufferTracker
4. Return buffer address to orchestration function
```

### pto_free() - Signal End of References

```
1. Find BufferInfo by base_ptr in BufferTracker
2. Set free_requested = true
3. If writer_count == 0:
   - Buffer has no writers, can reclaim immediately
   - Check if all tasks in heap ring before this buffer are CONSUMED
   - Advance heap_tail if possible
4. Otherwise:
   - Reclamation happens when all writer tasks become CONSUMED
```

**Note**: `pto_free()` does NOT immediately reclaim memory. It signals "no more references will be added". Device recycles memory after all current usage finishes.

### pto_submit_task() - Dynamic Task Submission

```
1. Allocate task slot from task ring (may stall if full)
2. For each INPUT parameter:
   - Lookup producer via TensorMap(buffer_address or region)
   - Increment producer's fanout_count
   - Build fanin_list
3. For each OUTPUT parameter:
   - If buffer address provided (explicit allocation):
     * Find BufferInfo in BufferTracker
     * Add task_id to writer_task_ids[]
     * Increment writer_count
   - If no address (implicit allocation):
     * Allocate from heap ring
   - Register output region in TensorMap
4. If fanin_count == 0, mark READY and add to ready_queue
```

### pto_task_complete() - Dependency Propagation

```
1. Mark task COMPLETED
2. For each consumer in fanout_list:
   - Decrement fanin_remaining
   - If reaches 0, mark READY and add to ready_queue
3. For each producer in fanin_list:
   - Increment fanout_refcount
   - If == fanout_count, mark CONSUMED
4. If task writes to explicit buffer:
   - Find BufferInfo in BufferTracker
   - Check if ALL writers for this buffer are CONSUMED
   - If yes AND free_requested == true:
     * Buffer can be reclaimed
     * Advance heap_tail if this is the oldest live buffer
```

### pto_scope_end() - Scope Cleanup

```
1. Pop scope_stack to get task range [begin, end)
2. For each task in range:
   - Increment fanout_refcount (scope releases reference)
   - Check if can transition to CONSUMED
```

---

## Implementation Phases

### Phase 1: Core Data Structures (2 days)
- Add TaskDescriptor, HeapRing, TensorMap, ScopeStack, BufferTracker to runtime.h
- Implement ring_buffer.cpp (task_ring_alloc, heap_ring_alloc)
- Implement tensormap.cpp (insert, lookup, lazy invalidation)
- Implement buffer_tracker.cpp (register, find, update writer counts)
- Use MemoryAllocator from platform layer for device memory

### Phase 2: Orchestration APIs (2-3 days)
- Implement runtime_api.cpp with 6 core functions
- Integrate with existing Runtime class
- Unit test single task submission and explicit allocation

### Phase 3: Scheduler Integration (2-3 days)
- Modify aicpu_executor.cpp to use ready queues
- Replace static graph iteration with dynamic dispatch
- Test multi-task pipelines

### Phase 4: End-to-End Testing (2-3 days)
- Write example orchestration function (tiled matmul)
- Compile orchestration to .so (using existing compiler infrastructure)
- Verify on hardware with real kernels
- Performance profiling

**Total: 8-11 days**

---

## Testing Strategy

**Unit Tests:**
- Ring buffer allocation/reclamation
- TensorMap insert/lookup/collisions
- Scope stack push/pop
- BufferTracker: register, find, writer tracking
- Explicit allocation (pto_alloc) and deferred reclamation (pto_free)
- Device memory allocation via MemoryAllocator

**Integration Tests:**
- Task dependency chains (A → B → C)
- Fanout (A → {B, C, D})
- Nested scopes
- Multi-writer scenarios (parallel writes to same buffer)
- Mixed implicit and explicit allocation
- AICore accessing ring-allocated buffers

**Performance Tests:**
- Ring allocation: <10 cycles target
- TensorMap lookup: <20 cycles target
- Task submission throughput: >100K tasks/sec
- Buffer tracker overhead: <5 cycles per operation

---

## Design Principles

1. **Plugin Architecture** - All changes in src/runtime/, use platform APIs
2. **Device Memory** - Heap ring MUST be device memory (rtMalloc)
3. **AICore Constraint** - No malloc capability, buffers pre-allocated
4. **Fast Allocation** - Ring bump pointer (~5 cycles) vs malloc (~100-500)
5. **Automatic Dependencies** - TensorMap resolves via buffer addresses
6. **Scope-Based Lifecycle** - fanout_count = scope_depth + consumers
7. **Lazy Cleanup** - No eager removal, ring wrap reclaims
8. **Lock-Free** - Atomics for coordination, no mutexes

---

## Memory Layout

```
┌─────────────────────────────────────────────────────┐
│ Platform Layer (src/platform/)                      │
│ • MemoryAllocator (rtMalloc/rtFree wrapper)         │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│ Runtime Plugin (src/runtime/)                       │
│                                                      │
│ Host/AICPU Memory:         Device Memory:           │
│ • TensorMap                • Heap Ring Buffer       │
│ • ScopeStack               • Task Descriptors       │
│ • BufferTracker            • Scheduler State        │
│ • Heap Ring metadata                                │
│                                                      │
│ APIs:                      Workers:                  │
│ • pto_alloc()              • AICore (CUBE/VECTOR)   │
│ • pto_free()               • AICPU (scheduler)      │
│ • pto_scope_begin()                                 │
│ • pto_submit_task()                                 │
│ • pto_scope_end()                                   │
│ • pto_task_complete()                               │
└─────────────────────────────────────────────────────┘
```

---

## Example Usage

**Example 1: Simple Sequential Pipeline (Implicit Allocation)**
```c
void simple_orch(Runtime* rt, uint64_t* args) {
    float *A = (float*)args[0], *B = (float*)args[1], *C = (float*)args[2];

    // Task outputs implicitly allocated, addresses returned in task descriptor
    int32_t t0 = pto_submit_task(rt, KERNEL_ADD, WORKER_VECTOR,
        (TaskParam[]){
            {PARAM_INPUT,  A, 1024},
            {PARAM_INPUT,  B, 1024},
            {PARAM_OUTPUT, NULL, 1024}  // Implicit allocation
        }, 3);

    // Get output address from task descriptor
    void* t0_out = rt->task_descriptors[t0].packed_buffer_base;

    pto_submit_task(rt, KERNEL_MUL, WORKER_VECTOR,
        (TaskParam[]){
            {PARAM_INPUT,  t0_out, 1024},
            {PARAM_INPUT,  C, 1024},
            {PARAM_OUTPUT, NULL, 1024}
        }, 3);
}
```

**Example 2: Parallel Tiled Matrix Multiply (Explicit Allocation)**
```c
void matmul_orch(Runtime* rt, uint64_t* args) {
    float *A = (float*)args[0], *B = (float*)args[1], *C = (float*)args[2];
    int M = args[3], K = args[4], N = args[5];

    const int TILE_M = 128, TILE_K = 128, TILE_N = 128;

    for (int i = 0; i < M; i += TILE_M) {
        for (int j = 0; j < N; j += TILE_N) {
            pto_scope_begin(rt);

            // Explicit allocation for shared output tile
            size_t tile_size = TILE_M * TILE_N * sizeof(float);
            void* C_tile = pto_alloc(rt, tile_size);

            // Multiple tasks write to DIFFERENT regions of C_tile in parallel
            for (int k = 0; k < K; k += TILE_K) {
                size_t partial_size = TILE_M * TILE_N * sizeof(float);
                TaskParam params[] = {
                    {PARAM_INPUT,  &A[i*K + k], TILE_M*TILE_K*4},
                    {PARAM_INPUT,  &B[k*N + j], TILE_K*TILE_N*4},
                    {PARAM_OUTPUT, .region = {
                        .buffer_ptr = C_tile,
                        .offset = 0,  // All write to same tile (accumulate)
                        .size = partial_size
                    }}
                };
                pto_submit_task(rt, KERNEL_GEMM, WORKER_CUBE, params, 3);
            }

            // Copy result to final output
            pto_submit_task(rt, KERNEL_COPY, WORKER_VECTOR,
                (TaskParam[]){
                    {PARAM_INPUT,  C_tile, tile_size},
                    {PARAM_OUTPUT, &C[i*N + j], tile_size}
                }, 2);

            // Signal no more references will be added to C_tile
            pto_free(rt, C_tile);

            pto_scope_end(rt);  // C_tile freed after copy completes
        }
    }
}
```

---

## Files to Create/Modify

### New Files (src/runtime/runtime/)
- `runtime_api.h` - API declarations (6 functions)
- `runtime_api.cpp` - API implementations
- `tensormap.h` - TensorMap interface
- `tensormap.cpp` - Hash table implementation
- `ring_buffer.h` - Ring allocation helpers
- `ring_buffer.cpp` - Implementation
- `buffer_tracker.h` - Explicit buffer tracking interface
- `buffer_tracker.cpp` - Implementation

### Modified Files
- `src/runtime/runtime/runtime.h` - Add orchestration structures
- `src/runtime/runtime/runtime.cpp` - Add init_orchestration()
- `src/runtime/aicpu/aicpu_executor.cpp` - Dynamic ready queues

### No Changes Needed
- `src/platform/` - Use existing MemoryAllocator
- `src/runtime/aicore/` - Kernels unchanged
- `src/runtime/host/` - Minimal changes if any

---

## Critical Notes

**Why Device Memory**: AICore can only access device memory. Ring buffer base pointer must come from rtMalloc.

**Why Ring Buffer**: Single large allocation managed as ring avoids per-task malloc/free overhead (40-200x faster).

**Why TensorMap**: Automatic dependency resolution - orchestration function just passes buffer addresses, TensorMap finds producers.

**Why Scopes**: Buffer lifetime management - ensures outputs live until all consumers (and enclosing scopes) are done.

**Why Explicit Allocation**: Required for parallel writes to the same buffer (tiled algorithms, multi-writer scenarios).

**pto_free() Semantics**: Does NOT immediately reclaim memory. It signals "no more references will be added" - the device recycles memory at its discretion after all current usage finishes. This allows:
- Orchestration function to release control of buffer lifetime
- Device to wait for all writer tasks to complete
- Automatic reclamation when safe (all writers CONSUMED)

---

## References

- Design document: `runtime_buffer_manager_concise.md`
- Current runtime: `src/runtime/runtime/runtime.h`
- AICPU scheduler: `src/runtime/aicpu/aicpu_executor.cpp`
- Memory allocator: `src/platform/a2a3/host/memoryallocator.h`
