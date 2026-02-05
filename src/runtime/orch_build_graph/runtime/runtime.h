/**
 * Orchestration Build Graph Runtime - Runtime Interface
 *
 * This provides a Runtime class compatible with the existing platform layer,
 * allowing the orchestration runtime to work with the same infrastructure as host_build_graph.
 *
 * PTO API: pto_scope_begin(), pto_scope_end(), pto_submit_task()
 * Dependencies are detected automatically via TensorMap.
 * Memory allocation is implicit during pto_submit_task() for OUTPUT params.
 *
 * Note: Legacy API (add_task/add_successor) is retained for internal use by
 * pto_submit_task() but PTO mode is always enabled.
 */

#ifndef ORCH_BUILD_GRAPH_RUNTIME_H
#define ORCH_BUILD_GRAPH_RUNTIME_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <atomic>

#include "common/core_type.h"  // CoreType enum (from platform)
#include "dep_list_pool.h"     // DepListPool, DepListEntry for dynamic dependency lists
#include "ring_buffer.h"       // TaskRing, HeapRing, PTOSharedHeader, TaskState, Handshake, etc.
#include "tensor_map.h"        // TensorMap, TensorMapEntry, TensorDescriptor, PTOBufferHandle, PTOParam

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

// Note: Handshake, TensorPair, HostApi, TaskState are defined in pto_runtime.h
// (included via ring_buffer.h) to avoid duplication.
// Note: CoreType is defined in common/core_type.h (from platform)

/**
 * Task entry in the runtime
 *
 * Dependencies are tracked via DepListPool linked lists (fanin_head/fanout_head).
 * The pool uses offset encoding where 0 = empty list.
 */
typedef struct {
    int task_id;
    int func_id;
    uint64_t args[RUNTIME_MAX_ARGS];
    int num_args;
    uint64_t function_bin_addr;
    int core_type;
    std::atomic<int> fanin;
    int fanout_count;  // Number of consumers (for CONSUMED check)
    uint64_t start_time;
    uint64_t end_time;

    // Task state machine and fanout reference counting
    TaskState state;      // Explicit task state (default: PENDING)
    int fanout_refcount;  // Completed consumers + scope_end count

    // DepListPool-based dependency tracking (offset encoding: 0 = empty)
    int32_t fanin_head;            // Head of producer list (DepListPool offset)
    int32_t fanout_head;           // Head of consumer list (DepListPool offset)
    int32_t fanin_count;           // Number of producers (immutable after submission)
    volatile int32_t fanout_lock;  // Spinlock for concurrent fanout modification

    // Packed output buffer tracking
    int32_t packed_buffer_offset;  // Offset in HeapRing (for heap reclamation)
    int32_t packed_buffer_size;    // Total size of packed outputs
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
    int add_task(uint64_t* args, int num_args, int func_id, PTOWorkerType core_type = PTOWorkerType::VECTOR);
    void add_successor(int from_task, int to_task);

    // Task lifecycle helpers
    void check_consumed(int task_id);

    Task* get_task(int task_id);
    int get_task_count() const;
    int get_initial_ready_tasks(int* ready_tasks);

    void print_runtime() const;

    void record_tensor_pair(void* host_ptr, void* dev_ptr, size_t size);
    TensorPair* get_tensor_pairs();
    int get_tensor_pair_count() const;
    void clear_tensor_pairs();

    // === PTO API ===

    // Initialize PTO mode (called automatically in constructor)
    void pto_init();

    // --- Scope-Based Lifecycle ---
    // Scope controls buffer lifetime (fanout initialized to scope_depth)
    void pto_scope_begin();
    void pto_scope_end();

    // --- Task Submission ---
    // Submit task with automatic dependency detection via TensorMap
    // OUTPUT params are allocated implicitly by runtime
    int pto_submit_task(int32_t func_id, PTOWorkerType worker_type, PTOParam* params, int32_t param_count);

    HostApi host_api;

private:
    // === PTO Internal State ===

    // Oldest non-CONSUMED task
    int32_t last_task_alive_;

    // TensorMap for automatic dependency tracking
    TensorMap tensor_map_;
    TensorMapEntry tensormap_pool_[PTO_TENSORMAP_POOL_SIZE];
    int32_t tensormap_buckets_[PTO_TENSORMAP_NUM_BUCKETS];

    // Scope stack for buffer lifetime management
    // scope_stack_[i] = index into scope_tasks_ where scope i's task list begins
    int32_t scope_stack_[PTO_MAX_SCOPE_DEPTH];
    int32_t scope_stack_top_ = 0;

    // Flat list of task IDs per scope (stack-allocated, RAII pop on scope_end)
    // Each scope owns a contiguous slice [scope_stack_[depth], scope_tasks_top_)
    // When a child scope ends, its slice is removed, so parent never sees child's tasks
    int32_t scope_tasks_[RUNTIME_MAX_TASKS];
    int32_t scope_tasks_top_ = 0;

    // Buffer handle tracking (for version_inc)
    PTOBufferHandle buffer_handles_[PTO_TENSORMAP_POOL_SIZE];
    int32_t buffer_handle_count_ = 0;

    // Ring buffers for task and heap allocation
    TaskRing task_ring_;
    PTOTaskDescriptor task_descriptors_[PTO_TASK_WINDOW_SIZE];
    HeapRing heap_ring_;
    char* heap_base_;                   // Device memory base for HeapRing
    bool use_ring_allocation_ = false;  // Enable ring allocation after pto_init_rings()

    // DepListPool for dynamic dependency lists (fanin/fanout)
    DepListPool dep_list_pool_;
    DepListEntry dep_list_entries_[PTO_DEP_LIST_POOL_SIZE];

public:
    // Initialize ring buffers
    void pto_init_rings();
    PTOSharedHeader* get_shared_header() { return &shared_header_; }
    DepListPool* get_dep_list_pool() { return &dep_list_pool_; }

private:
    // Shared header for Orchestrator ↔ Scheduler communication
    PTOSharedHeader shared_header_;
};

#endif  // ORCH_BUILD_GRAPH_RUNTIME_H
