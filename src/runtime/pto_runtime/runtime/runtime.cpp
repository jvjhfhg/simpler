/**
 * PTO Runtime - Runtime Implementation (Phase 8: PTO-Only Mode)
 *
 * PTO mode is always enabled. The constructor auto-initializes TensorMap
 * for automatic dependency tracking.
 */

#include "runtime.h"

// =============================================================================
// Constructor (auto-initializes PTO mode)
// =============================================================================

Runtime::Runtime() {
    // Initialize task array
    for (int i = 0; i < RUNTIME_MAX_TASKS; i++) {
        tasks[i].task_id = 0;
        tasks[i].func_id = 0;
        tasks[i].num_args = 0;
        tasks[i].function_bin_addr = 0;
        tasks[i].core_type = 0;
        tasks[i].fanin = 0;
        tasks[i].fanout_count = 0;
        tasks[i].start_time = 0;
        tasks[i].end_time = 0;
        memset(tasks[i].args, 0, sizeof(tasks[i].args));
        memset(tasks[i].fanout, 0, sizeof(tasks[i].fanout));
    }
    next_task_id = 0;
    initial_ready_count = 0;
    worker_count = 0;
    block_dim = 0;
    sche_cpu_num = 1;
    tensor_pair_count = 0;
    buffer_handle_count_ = 0;
}

// =============================================================================
// Task Management
// =============================================================================

int Runtime::add_task(uint64_t* args, int num_args, int func_id, PTOWorkerType core_type) {
    if (next_task_id >= RUNTIME_MAX_TASKS) {
        fprintf(stderr, "[PTO Runtime] ERROR: Task table full (max=%d)\n", RUNTIME_MAX_TASKS);
        return -1;
    }

    if (num_args > RUNTIME_MAX_ARGS) {
        fprintf(stderr, "[PTO Runtime] ERROR: Too many args (%d > %d)\n", num_args, RUNTIME_MAX_ARGS);
        return -1;
    }

    int task_id = next_task_id++;
    Task* task = &tasks[task_id];

    task->task_id = task_id;
    task->func_id = func_id;
    task->num_args = num_args;
    if (args && num_args > 0) {
        memcpy(task->args, args, num_args * sizeof(uint64_t));
    }
    task->function_bin_addr = 0;
    task->core_type = core_type;
    task->fanin = 0;
    task->fanout_count = 0;
    memset(task->fanout, 0, sizeof(task->fanout));

    return task_id;
}

void Runtime::add_successor(int from_task, int to_task) {
    if (from_task < 0 || from_task >= next_task_id) {
        fprintf(stderr, "[PTO Runtime] ERROR: Invalid from_task ID %d\n", from_task);
        return;
    }

    if (to_task < 0 || to_task >= next_task_id) {
        fprintf(stderr, "[PTO Runtime] ERROR: Invalid to_task ID %d\n", to_task);
        return;
    }

    Task* from = &tasks[from_task];
    Task* to = &tasks[to_task];

    if (from->fanout_count >= RUNTIME_MAX_FANOUT) {
        fprintf(stderr, "[PTO Runtime] ERROR: Fanout overflow for task %d\n", from_task);
        return;
    }

    from->fanout[from->fanout_count++] = to_task;
    to->fanin++;
}

// =============================================================================
// Query Methods
// =============================================================================

Task* Runtime::get_task(int task_id) {
    if (task_id < 0 || task_id >= next_task_id) {
        return nullptr;
    }
    return &tasks[task_id];
}

int Runtime::get_task_count() const {
    return next_task_id;
}

int Runtime::get_initial_ready_tasks(int* ready_tasks) {
    initial_ready_count = 0;
    for (int i = 0; i < next_task_id; i++) {
        if (tasks[i].fanin == 0) {
            initial_ready_tasks[initial_ready_count] = i;
            if (ready_tasks != nullptr) {
                ready_tasks[initial_ready_count] = i;
            }
            initial_ready_count++;
        }
    }
    return initial_ready_count;
}

// =============================================================================
// Utility Methods
// =============================================================================

void Runtime::print_runtime() const {
    printf("\n========================================================================\n");
    printf("[PTO Runtime] Task Runtime Status\n");
    printf("========================================================================\n");
    printf("  Total tasks: %d\n", next_task_id);

    printf("\nInitially Ready Tasks (fanin==0):\n");
    printf("------------------------------------------------------------------------\n");
    printf("  ");
    int ready_count = 0;
    for (int i = 0; i < next_task_id; i++) {
        if (tasks[i].fanin.load() == 0) {
            if (ready_count > 0) printf(", ");
            printf("%d", i);
            ready_count++;
        }
    }
    if (ready_count == 0) {
        printf("(none)");
    }
    printf("\n  Count: %d\n", ready_count);

    printf("\nTask Table:\n");
    printf("------------------------------------------------------------------------\n");

    for (int i = 0; i < next_task_id; i++) {
        const Task* t = &tasks[i];
        printf("  Task %d: func_id=%d, fanin=%d, fanout=%d, core_type=%d [",
            i, t->func_id, t->fanin.load(), t->fanout_count, t->core_type);
        for (int j = 0; j < t->fanout_count; j++) {
            printf("%d%s", t->fanout[j], j < t->fanout_count - 1 ? "," : "");
        }
        printf("]\n");
    }

    printf("========================================================================\n\n");
}

// =============================================================================
// Tensor Pair Management
// =============================================================================

void Runtime::record_tensor_pair(void* host_ptr, void* dev_ptr, size_t size) {
    if (tensor_pair_count >= RUNTIME_MAX_TENSOR_PAIRS) {
        fprintf(stderr, "[PTO Runtime] ERROR: Tensor pairs full (max=%d)\n", RUNTIME_MAX_TENSOR_PAIRS);
        return;
    }
    tensor_pairs[tensor_pair_count].host_ptr = host_ptr;
    tensor_pairs[tensor_pair_count].dev_ptr = dev_ptr;
    tensor_pairs[tensor_pair_count].size = size;
    tensor_pair_count++;
    printf("[PTO Runtime] Recorded tensor pair: host=%p dev=%p size=%zu\n", host_ptr, dev_ptr, size);
}

TensorPair* Runtime::get_tensor_pairs() {
    return tensor_pairs;
}

int Runtime::get_tensor_pair_count() const {
    return tensor_pair_count;
}

void Runtime::clear_tensor_pairs() {
    tensor_pair_count = 0;
}

// =============================================================================
// PTO API Implementation
// =============================================================================

void Runtime::pto_init() {
    buffer_handle_count_ = 0;

    // Initialize TensorMap
    tensormap_init(&tensor_map_, tensormap_pool_, PTO_TENSORMAP_POOL_SIZE,
                   tensormap_buckets_, PTO_TENSORMAP_NUM_BUCKETS);

    printf("[PTO] PTO mode initialized (TensorMap: %d buckets, %d pool entries)\n",
           PTO_TENSORMAP_NUM_BUCKETS, PTO_TENSORMAP_POOL_SIZE);
}

PTOBufferHandle* Runtime::pto_alloc(int32_t size) {
    if (buffer_handle_count_ >= PTO_TENSORMAP_POOL_SIZE) {
        fprintf(stderr, "[PTO] ERROR: Buffer handle pool full (max=%d)\n", PTO_TENSORMAP_POOL_SIZE);
        return nullptr;
    }

    PTOBufferHandle* handle = &buffer_handles_[buffer_handle_count_++];
    handle->size = size;
    handle->version = 0;
    handle->ref_count = 1;

    // Allocate device memory via host API
    void* dev_ptr = host_api.device_malloc(size);
    handle->addr = (uint64_t)dev_ptr;

    return handle;
}

void Runtime::pto_free(PTOBufferHandle* handle) {
    if (handle == nullptr) return;

    // Decrement reference count
    // Actual memory reclamation deferred until all consumers finish
    handle->ref_count--;

    if (handle->ref_count <= 0) {
        // Memory reclamation
        host_api.device_free((void*)handle->addr);
        handle->addr = 0;
    }
}

PTOBufferHandle* Runtime::pto_version_inc(PTOBufferHandle* handle) {
    if (handle == nullptr) return nullptr;

    // Create a new versioned handle (SSA-style)
    // Same address, incremented version number
    if (buffer_handle_count_ >= PTO_TENSORMAP_POOL_SIZE) {
        fprintf(stderr, "[PTO] ERROR: Buffer handle pool full for version_inc\n");
        return nullptr;
    }

    PTOBufferHandle* new_handle = &buffer_handles_[buffer_handle_count_++];
    new_handle->addr = handle->addr;
    new_handle->size = handle->size;
    new_handle->version = handle->version + 1;
    new_handle->ref_count = 1;

    return new_handle;
}

int Runtime::pto_submit_task(int32_t func_id, PTOWorkerType worker_type,
                             PTOParam* params, int32_t param_count) {
    // Build kernel args array from params
    uint64_t args[RUNTIME_MAX_ARGS];
    int num_args = 0;

    for (int32_t i = 0; i < param_count && num_args < RUNTIME_MAX_ARGS; i++) {
        if (params[i].type == PTO_PARAM_SCALAR) {
            args[num_args++] = params[i].scalar_value;
        } else {
            args[num_args++] = params[i].buffer->addr;
        }
    }

    // Add task using legacy API
    int task_id = add_task(args, num_args, func_id, worker_type);
    if (task_id < 0) return -1;

    // Automatic dependency detection via TensorMap
    for (int32_t i = 0; i < param_count; i++) {
        if (params[i].type == PTO_PARAM_INPUT && params[i].buffer != nullptr) {
            // Look up producer for this input tensor
            int32_t producer = tensormap_lookup(&tensor_map_, &params[i].tensor, 0);
            if (producer >= 0 && producer != task_id) {
                add_successor(producer, task_id);
            }
        }
    }

    // Register output tensors in TensorMap
    for (int32_t i = 0; i < param_count; i++) {
        if (params[i].type == PTO_PARAM_OUTPUT && params[i].buffer != nullptr) {
            tensormap_insert(&tensor_map_, &params[i].tensor,
                           task_id, params[i].buffer->version);
        }
    }

    return task_id;
}
