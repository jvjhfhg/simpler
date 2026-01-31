/**
 * PTO Runtime - Runtime Interface (Phase 8: PTO-Only Mode)
 *
 * This provides a Runtime class compatible with the existing platform layer,
 * allowing the PTO runtime to work with the same infrastructure as host_build_graph.
 *
 * PTO API: pto_alloc(), pto_submit_task(), pto_free(), pto_version_inc()
 * Dependencies are detected automatically via TensorMap.
 *
 * Note: Legacy API (add_task/add_successor) is retained for internal use by
 * pto_submit_task() but PTO mode is always enabled.
 */

#ifndef RUNTIME_H
#define RUNTIME_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <atomic>

#include "tensor_map.h"  // TensorMap, TensorMapEntry, PTOTensorDescriptor, PTOBufferHandle, PTOParam

// =============================================================================
// Configuration Macros
// =============================================================================

#ifndef RUNTIME_MAX_TASKS
#define RUNTIME_MAX_TASKS 1024
#endif

#ifndef RUNTIME_MAX_ARGS
#define RUNTIME_MAX_ARGS 16
#endif

#ifndef RUNTIME_MAX_FANOUT
#define RUNTIME_MAX_FANOUT 512
#endif

#ifndef RUNTIME_MAX_WORKER
#define RUNTIME_MAX_WORKER 72
#endif

#ifndef RUNTIME_MAX_TENSOR_PAIRS
#define RUNTIME_MAX_TENSOR_PAIRS 64
#endif

// =============================================================================
// Data Structures
// =============================================================================

/**
 * Handshake buffer for AICPU-AICore communication
 *
 * Protocol:
 * 1. AICPU sets aicpu_ready=1
 * 2. AICore sets aicore_done=core_id+1
 * 3. AICPU assigns task pointer and sets task_status=1
 * 4. AICore executes, sets task_status=0
 * 5. AICPU clears task=0
 * 6. AICPU sets control=1 to shutdown
 */
struct Handshake {
    volatile uint32_t aicpu_ready;
    volatile uint32_t aicore_done;
    volatile uint64_t task;
    volatile int32_t task_status;
    volatile int32_t control;
    volatile int32_t core_type;
} __attribute__((aligned(64)));

/**
 * Core type enumeration
 */
enum class CoreType : int {
    AIC = 0,
    AIV = 1
};

/**
 * Tensor pair for tracking host-device memory mappings
 */
struct TensorPair {
    void* host_ptr;
    void* dev_ptr;
    size_t size;
};

/**
 * Host API function pointers for device memory operations
 */
struct HostApi {
    void* (*device_malloc)(size_t size);
    void (*device_free)(void* dev_ptr);
    int (*copy_to_device)(void* dev_ptr, const void* host_ptr, size_t size);
    int (*copy_from_device)(void* host_ptr, const void* dev_ptr, size_t size);
};

/**
 * Task entry in the runtime
 */
typedef struct {
    int task_id;
    int func_id;
    uint64_t args[RUNTIME_MAX_ARGS];
    int num_args;
    uint64_t function_bin_addr;
    int core_type;
    std::atomic<int> fanin;
    int fanout[RUNTIME_MAX_FANOUT];
    int fanout_count;
    uint64_t start_time;
    uint64_t end_time;
} Task;

// =============================================================================
// Runtime Class
// =============================================================================

class Runtime {
public:
    // Handshake buffers for AICPU-AICore communication
    Handshake workers[RUNTIME_MAX_WORKER];
    int worker_count;

    // Execution parameters
    int block_dim;
    int sche_cpu_num;

private:
    Task tasks[RUNTIME_MAX_TASKS];
    int next_task_id;
    int initial_ready_tasks[RUNTIME_MAX_TASKS];
    int initial_ready_count;
    TensorPair tensor_pairs[RUNTIME_MAX_TENSOR_PAIRS];
    int tensor_pair_count;

public:
    Runtime();

    // Internal task management (used by pto_submit_task)
    int add_task(uint64_t *args, int num_args, int func_id, PTOWorkerType core_type = PTOWorkerType::VECTOR);
    void add_successor(int from_task, int to_task);

    Task *get_task(int task_id);
    int get_task_count() const;
    int get_initial_ready_tasks(int *ready_tasks);

    void print_runtime() const;

    void record_tensor_pair(void* host_ptr, void* dev_ptr, size_t size);
    TensorPair* get_tensor_pairs();
    int get_tensor_pair_count() const;
    void clear_tensor_pairs();

    // === PTO API ===

    // Initialize PTO mode (called automatically in constructor)
    void pto_init();

    // --- Buffer Management ---
    // Allocate buffer handle, returns handle with address
    // Address is available BEFORE task submission (required for dependent tasks)
    PTOBufferHandle* pto_alloc(int32_t size);

    // Signal "no more references will be added" (does NOT immediately free)
    // Device recycles memory after all consumers finish
    void pto_free(PTOBufferHandle* handle);

    // --- Version Control for In-Place Updates ---
    // Returns new versioned handle (SSA-style)
    // Write to version v waits for all reads from version v-1
    PTOBufferHandle* pto_version_inc(PTOBufferHandle* handle);

    // --- Task Submission ---
    // Submit task with automatic dependency detection via TensorMap
    int pto_submit_task(int32_t func_id, PTOWorkerType worker_type,
                        PTOParam* params, int32_t param_count);

    HostApi host_api;

private:
    // === PTO Internal State ===

    // TensorMap for automatic dependency tracking
    TensorMap tensor_map_;
    TensorMapEntry tensormap_pool_[PTO_TENSORMAP_POOL_SIZE];
    int32_t tensormap_buckets_[PTO_TENSORMAP_NUM_BUCKETS];

    // Buffer handle tracking
    PTOBufferHandle buffer_handles_[PTO_TENSORMAP_POOL_SIZE];
    int32_t buffer_handle_count_ = 0;
};

#endif  // RUNTIME_H
