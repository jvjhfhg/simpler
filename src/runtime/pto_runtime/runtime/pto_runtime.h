/**
 * PTO Runtime - Parallel Task Orchestration Runtime
 *
 * This runtime supports dynamic task submission with automatic dependency detection
 * via TensorMap, scope-based buffer lifecycle management, and ring buffer architecture
 * for O(1) allocation with implicit deallocation.
 *
 * Key Features:
 * - Dynamic task submission during execution
 * - Ring buffer-based memory management (Task, Heap, DepList rings)
 * - Automatic producer-consumer tracking via TensorMap
 * - Scope-based buffer lifetime (fanout starts at scope_depth)
 * - Decoupled Orchestrator/Scheduler communication via shared memory
 *
 * Architecture:
 * - Orchestrator: Executes user function, submits tasks, builds dependency graph
 * - Scheduler: Dispatches tasks to workers, tracks completions, reclaims memory
 * - Workers: Execute kernels on AICore/Vector/AICPU units
 */

#ifndef PTO_RUNTIME_H
#define PTO_RUNTIME_H

#include <stdint.h>
#include <stdbool.h>
#include <atomic>

#include "pto_types.h"  // PTOWorkerType, PTOTensorDescriptor, etc.

// =============================================================================
// Configuration Constants
// =============================================================================

// Ring buffer sizes (must be power of 2 for efficient modulo)
#ifndef PTO_TASK_WINDOW_SIZE
#define PTO_TASK_WINDOW_SIZE 1024
#endif

#ifndef PTO_HEAP_SIZE
#define PTO_HEAP_SIZE (64 * 1024 * 1024)  // 64MB
#endif

#ifndef PTO_DEP_LIST_POOL_SIZE
#define PTO_DEP_LIST_POOL_SIZE 8192
#endif

#ifndef PTO_TENSORMAP_POOL_SIZE
#define PTO_TENSORMAP_POOL_SIZE 4096
#endif

#ifndef PTO_TENSORMAP_NUM_BUCKETS
#define PTO_TENSORMAP_NUM_BUCKETS 1024
#endif

// Task and scope limits
#ifndef PTO_MAX_ARGS
#define PTO_MAX_ARGS 16
#endif

#ifndef PTO_MAX_WORKER
#define PTO_MAX_WORKER 72
#endif

#ifndef PTO_MAX_SCOPE_DEPTH
#define PTO_MAX_SCOPE_DEPTH 32
#endif

// Alignment for DMA and cache efficiency
#ifndef PTO_ALIGNMENT
#define PTO_ALIGNMENT 64
#endif

// Helper macro for alignment
#define ALIGN_UP(x, align) (((x) + (align) - 1) & ~((align) - 1))

// =============================================================================
// Task State Machine
// =============================================================================

/**
 * Task State Transitions:
 *
 * PENDING → READY:      When fanin_refcount == fanin_count (all producers done)
 * READY → RUNNING:      When dispatched to worker
 * RUNNING → COMPLETED:  When worker signals completion
 * COMPLETED → CONSUMED: When fanout_refcount == fanout_count (all consumers + scopes released)
 *
 * Memory reclamation happens when task transitions to CONSUMED.
 */
enum class TaskState : int32_t {
    PENDING   = 0,  // Waiting for dependencies
    READY     = 1,  // All dependencies met, in ready queue
    RUNNING   = 2,  // Executing on worker
    COMPLETED = 3,  // Execution done, may have live consumers
    CONSUMED  = 4   // All consumers done, buffer can be freed
};

// =============================================================================
// Shared Memory Structures
// =============================================================================

/**
 * Shared Memory Header - Flow control between Orchestrator and Scheduler
 *
 * Cache-line aligned to prevent false sharing. The Orchestrator and Scheduler
 * communicate only via these volatile pointers - no other shared state.
 *
 * Orchestrator writes: current_task_index, heap_top, orchestrator_done
 * Scheduler writes:    last_task_alive, heap_tail, scheduler_done
 *
 * Back-pressure mechanism:
 * - Orchestrator stalls when rings are full (head catches tail)
 * - Scheduler advances tail pointers as tasks are consumed
 */
struct alignas(64) PTOSharedHeader {
    // === Orchestrator → Scheduler ===
    volatile int32_t current_task_index;  // Next task slot to allocate
    volatile int32_t heap_top;            // Current heap allocation pointer
    volatile int32_t orchestrator_done;   // 1 when orchestration complete
    char pad1[64 - 12];

    // === Scheduler → Orchestrator ===
    volatile int32_t last_task_alive;     // Oldest non-CONSUMED task
    volatile int32_t heap_tail;           // Oldest live buffer start
    volatile int32_t scheduler_done;      // 1 when all tasks complete
    char pad2[64 - 12];
};

/**
 * Task Descriptor - Shared between Orchestrator and Scheduler
 *
 * Stored in a ring buffer (TaskRing). Each task has a unique slot index
 * that may be reused after the task is CONSUMED and last_task_alive advances.
 *
 * Dependency tracking uses linked lists (offsets into DepListPool):
 * - fanin_head: list of producer task IDs (immutable after submission)
 * - fanout_head: list of consumer task IDs (grows dynamically, protected by spinlock)
 *
 * fanout_count initialization:
 * - Starts at scope_depth (number of enclosing scopes)
 * - Incremented for each consumer task
 * - Decremented by scope_end() and consumer completion
 * - Task becomes CONSUMED when fanout_refcount == fanout_count
 */
struct PTOTaskDescriptor {
    // === Task Identity ===
    int32_t task_id;                      // Unique task identifier (may wrap in ring)
    int32_t func_id;                      // Kernel function ID
    PTOWorkerType worker_type;            // PTOWorkerType::CUBE or PTOWorkerType::VECTOR
    int32_t num_args;                     // Number of valid arguments

    // === Kernel Arguments ===
    uint64_t args[PTO_MAX_ARGS];          // Arguments passed to kernel

    // === Kernel Binary ===
    uint64_t function_bin_addr;           // Device GM address of kernel binary

    // === Dependency Counts (immutable after submission) ===
    int32_t fanin_count;                  // Number of producer tasks
    int32_t fanout_count;                 // Number of consumers + scope_depth

    // === Dependency Lists (offsets into DepListPool, 0 = empty) ===
    int32_t fanin_head;                   // Linked list of producer task IDs
    volatile int32_t fanout_head;         // Linked list of consumer task IDs (grows dynamically)

    // === Output Buffer Info (for heap reclamation) ===
    int32_t packed_buffer_offset;         // Offset in HeapRing
    int32_t packed_buffer_size;           // Total size of packed outputs

    // === Concurrency Control ===
    // Spinlock for fanout_head and fanout_count modification
    // Only needed because Orchestrator can add consumers while Scheduler reads count
    volatile int32_t fanout_lock;

    char pad[64 - ((17 * 4 + PTO_MAX_ARGS * 8 + 8) % 64)];  // Align to cache line
};

/**
 * Handshake Structure - AICPU ↔ AICore communication
 *
 * Reused from host_build_graph runtime - no changes needed.
 * Each AICore worker has its own handshake buffer for polling-based dispatch.
 *
 * Protocol:
 * 1. AICPU sets aicpu_ready=1
 * 2. AICore sets aicore_done=core_id+1
 * 3. AICPU assigns task pointer and sets task_status=1
 * 4. AICore executes kernel, sets task_status=0
 * 5. AICPU reads completion, clears task=0
 * 6. AICPU sets control=1 to signal shutdown
 */
struct Handshake {
    volatile uint32_t aicpu_ready;  // AICPU ready signal: 0=not ready, 1=ready
    volatile uint32_t aicore_done;  // AICore ready signal: 0=not ready, core_id+1=ready
    volatile uint64_t task;         // Task pointer: 0=no task, non-zero=PTOTaskDescriptor* address
    volatile int32_t task_status;   // Task execution status: 0=idle/complete, 1=busy
    volatile int32_t control;       // Control signal: 0=execute, 1=quit
    volatile int32_t core_type;     // Core type: 0=AIC, 1=AIV
    char pad[64 - 28];              // Pad to cache line (64 bytes)
} __attribute__((aligned(64)));

// =============================================================================
// Scheduler-Private State (not in shared memory)
// =============================================================================

/**
 * Scheduler State - Private to Scheduler thread
 *
 * Tracks per-task execution state and ready queues. Indexed by task_id % PTO_TASK_WINDOW_SIZE
 * to handle ring buffer wraparound.
 *
 * fanin_refcount:  Number of completed producers (0 to fanin_count)
 * fanout_refcount: Number of completed consumers + scope_end calls (0 to fanout_count)
 *
 * Per-worker-type ready queues avoid global queue lock contention and enable
 * natural load balancing across heterogeneous compute units.
 */
struct PTOSchedulerState {
    // Per-task state (ring buffer indexed by task_id % PTO_TASK_WINDOW_SIZE)
    TaskState task_state[PTO_TASK_WINDOW_SIZE];
    int32_t fanin_refcount[PTO_TASK_WINDOW_SIZE];   // Completed producers
    int32_t fanout_refcount[PTO_TASK_WINDOW_SIZE];  // Completed consumers + scopes

    // Ready queues per worker type (ring buffer, FIFO)
    int32_t ready_queue[PTO_NUM_WORKER_TYPES][PTO_TASK_WINDOW_SIZE];
    int32_t ready_head[PTO_NUM_WORKER_TYPES];  // Dequeue position
    int32_t ready_tail[PTO_NUM_WORKER_TYPES];  // Enqueue position
};

// =============================================================================
// Dependency List Pool (shared, ring buffer)
// =============================================================================

/**
 * Dependency List Entry - Node in linked list of task IDs
 *
 * Used for both fanin and fanout lists. Lists are built using prepend (O(1))
 * and stored as offsets into DepListPool.
 *
 * Offset encoding: 0 means empty list, non-zero is (array_index + 1)
 * This allows distinguishing empty (0) from first entry (1).
 *
 * Memory reclamation: Implicit when task ring wraps around - old lists become
 * unreachable garbage and slots are reused.
 */
struct DepListEntry {
    int32_t task_id;              // The task ID in this list node
    int32_t next_offset;          // Offset to next entry (0 = end of list)
};

/**
 * Dependency List Pool - Ring buffer for linked list nodes
 *
 * Allocation: Simple bump pointer (O(1))
 * Deallocation: Implicit when task ring wraps - old entries become garbage
 *
 * Size calculation: ~8K entries = 1024 tasks × 2 outputs × 4 consumers average
 */
struct DepListPool {
    DepListEntry* base;
    int32_t size;         // PTO_DEP_LIST_POOL_SIZE
    int32_t top;          // Next slot to allocate (wraps around)
};

// =============================================================================
// Host API (for device memory operations)
// =============================================================================

/**
 * Host API - Pluggable device memory backend
 *
 * Reused from host_build_graph. Allows runtime to use different device memory
 * implementations (real hardware, simulator, mock).
 */
struct HostApi {
    void* (*device_malloc)(size_t size);
    void (*device_free)(void* dev_ptr);
    int (*copy_to_device)(void* dev_ptr, const void* host_ptr, size_t size);
    int (*copy_from_device)(void* host_ptr, const void* dev_ptr, size_t size);
};

/**
 * Tensor Pair - For tracking host-device memory mappings
 *
 * Used for copy-back during finalize. Orchestration records output tensors
 * that need to be copied from device back to host.
 */
struct TensorPair {
    void* host_ptr;
    void* dev_ptr;
    size_t size;
};

// =============================================================================
// Runtime Parameter Types (for pto_submit_task API)
// =============================================================================

/**
 * Parameter Type - Distinguishes inputs from outputs
 */
enum class PTOParamType : int32_t {
    INPUT  = 0,  // Read-only input buffer
    OUTPUT = 1   // Write-only output buffer
};

/**
 * Parameter Descriptor - Describes input/output for task submission
 *
 * For INPUT:  base_ptr + offset points to existing buffer
 * For OUTPUT: ptr_to_ptr will receive allocated buffer address
 *
 * Example usage:
 *   void* C = NULL;
 *   PTOParam params[] = {
 *       {PTOParamType::INPUT,  (uint64_t)A, 0, size_A, NULL},
 *       {PTOParamType::INPUT,  (uint64_t)B, 0, size_B, NULL},
 *       {PTOParamType::OUTPUT, 0, 0, size_C, &C}  // Runtime writes address to C
 *   };
 *   pto_submit_task(orch, func_id, worker_type, params, 3);
 *   // After return, C contains allocated device buffer address
 */
struct PTOParam {
    PTOParamType type;      // PTOParamType::INPUT or PTOParamType::OUTPUT
    uint64_t base_ptr;      // For INPUT: buffer address; for OUTPUT: unused
    int32_t offset;         // Byte offset within buffer
    int32_t size;           // Size in bytes
    void** ptr_to_ptr;      // For OUTPUT: receives allocated address; for INPUT: unused
};

// =============================================================================
// Forward Declarations (implementations in separate files)
// =============================================================================

struct PTOOrchestrator;
struct PTOScheduler;
struct TensorMap;
struct TaskRing;
struct HeapRing;

// =============================================================================
// Utility Functions (inline for performance)
// =============================================================================

/**
 * Simple spinlock implementation
 */
static inline void spinlock_acquire(volatile int32_t* lock) {
    while (__sync_lock_test_and_set(lock, 1)) {
        // Spin
    }
}

static inline void spinlock_release(volatile int32_t* lock) {
    __sync_lock_release(lock);
}

/**
 * Ring buffer index calculation (assumes size is power of 2)
 */
static inline int32_t ring_index(int32_t pos, int32_t size) {
    return pos & (size - 1);
}

#endif  // PTO_RUNTIME_H
