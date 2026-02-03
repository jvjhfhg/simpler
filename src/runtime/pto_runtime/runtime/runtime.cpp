/**
 * PTO Runtime - Runtime Implementation
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
        tasks[i].state = TaskState::PENDING;
        tasks[i].fanout_refcount = 0;
        tasks[i].fanin_producer_count = 0;
        tasks[i].packed_buffer_offset = 0;
        tasks[i].packed_buffer_size = 0;
        memset(tasks[i].args, 0, sizeof(tasks[i].args));
        memset(tasks[i].fanout, 0, sizeof(tasks[i].fanout));
        memset(tasks[i].fanin_producers, 0, sizeof(tasks[i].fanin_producers));
    }
    next_task_id = 0;
    initial_ready_count = 0;
    worker_count = 0;
    block_dim = 0;
    sche_cpu_num = 1;
    tensor_pair_count = 0;
    buffer_handle_count_ = 0;
    scope_stack_top_ = 0;
    scope_tasks_top_ = 0;
    last_task_alive_ = 0;

    // Initialize ring buffer state
    heap_base_ = nullptr;
    use_ring_allocation_ = false;
    memset(&task_ring_, 0, sizeof(task_ring_));
    memset(&heap_ring_, 0, sizeof(heap_ring_));
    memset(&shared_header_, 0, sizeof(shared_header_));
}

// =============================================================================
// Task Management
// =============================================================================

int Runtime::add_task(uint64_t* args, int num_args, int func_id, PTOWorkerType core_type) {
    if (num_args > RUNTIME_MAX_ARGS) {
        fprintf(stderr, "[PTO Runtime] ERROR: Too many args (%d > %d)\n", num_args, RUNTIME_MAX_ARGS);
        return -1;
    }

    int task_id;
    if (use_ring_allocation_) {
        // Back-pressure: stall if task ring is full
        task_id = task_ring_alloc(&task_ring_, &shared_header_.last_task_alive);
        // Map ring slot back to absolute task ID
        task_id = next_task_id++;
    } else {
        if (next_task_id >= RUNTIME_MAX_TASKS) {
            fprintf(stderr, "[PTO Runtime] ERROR: Task table full (max=%d)\n", RUNTIME_MAX_TASKS);
            return -1;
        }
        task_id = next_task_id++;
    }
    Task* task = &tasks[task_id];

    task->task_id = task_id;
    task->func_id = func_id;
    task->num_args = num_args;
    if (args && num_args > 0) {
        memcpy(task->args, args, num_args * sizeof(uint64_t));
    }
    task->function_bin_addr = 0;
    task->core_type = static_cast<int>(core_type);
    task->fanin = 0;
    task->fanout_count = 0;
    task->state = TaskState::PENDING;
    task->fanout_refcount = 0;
    task->fanin_producer_count = 0;
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

    // Record reverse dependency (producer → consumer tracking)
    if (to->fanin_producer_count < RUNTIME_MAX_ARGS) {
        to->fanin_producers[to->fanin_producer_count++] = from_task;
    } else {
        fprintf(stderr, "[PTO Runtime] WARNING: fanin_producers overflow for task %d\n", to_task);
    }
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
// Task Lifecycle Helpers
// =============================================================================

void Runtime::check_consumed(int task_id) {
    if (task_id < 0 || task_id >= next_task_id) return;

    Task* task = &tasks[task_id];
    // Target = fanout_count + 1 (real consumers + scope ref)
    int target = task->fanout_count + 1;
    if (task->fanout_refcount == target &&
        task->state == TaskState::COMPLETED) {
        task->state = TaskState::CONSUMED;
        printf("[PTO] Task %d → CONSUMED (fanout_refcount=%d == fanout_count+1=%d)\n",
               task_id, task->fanout_refcount, target);
    }
}

// =============================================================================
// Utility Methods
// =============================================================================

void Runtime::print_runtime() const {
    static const char* state_names[] = {"PENDING", "READY", "RUNNING", "COMPLETED", "CONSUMED"};

    printf("\n========================================================================\n");
    printf("[PTO Runtime] Task Runtime Status\n");
    printf("========================================================================\n");
    printf("  Total tasks: %d, last_task_alive: %d\n", next_task_id, last_task_alive_);

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

    int consumed_count = 0;
    for (int i = 0; i < next_task_id; i++) {
        const Task* t = &tasks[i];
        int state_idx = static_cast<int>(t->state);
        const char* state_str = (state_idx >= 0 && state_idx <= 4) ? state_names[state_idx] : "UNKNOWN";
        printf("  Task %d: func_id=%d, state=%s, fanin=%d, fanout_count=%d, fanout_refcount=%d, core_type=%d [",
            i, t->func_id, state_str, t->fanin.load(), t->fanout_count, t->fanout_refcount, t->core_type);
        for (int j = 0; j < t->fanout_count; j++) {
            printf("%d%s", t->fanout[j], j < t->fanout_count - 1 ? "," : "");
        }
        printf("]\n");
        if (t->state == TaskState::CONSUMED) consumed_count++;
    }

    printf("\n  CONSUMED tasks: %d / %d\n", consumed_count, next_task_id);
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
    scope_stack_top_ = 0;
    scope_tasks_top_ = 0;

    // Initialize TensorMap
    tensormap_init(&tensor_map_, tensormap_pool_, PTO_TENSORMAP_POOL_SIZE,
                   tensormap_buckets_, PTO_TENSORMAP_NUM_BUCKETS);

    printf("[PTO] PTO mode initialized (TensorMap: %d buckets, %d pool entries)\n",
           PTO_TENSORMAP_NUM_BUCKETS, PTO_TENSORMAP_POOL_SIZE);
}

// =============================================================================
// Ring Buffer Initialization
// =============================================================================

void Runtime::pto_init_rings() {
    // Initialize TaskRing with task descriptors array
    task_ring_init(&task_ring_, task_descriptors_, PTO_TASK_WINDOW_SIZE);

    // Allocate device memory for HeapRing
    heap_base_ = (char*)host_api.device_malloc(PTO_HEAP_SIZE);
    if (heap_base_ == nullptr) {
        fprintf(stderr, "[PTO] ERROR: Failed to allocate HeapRing memory (%d bytes)\n", PTO_HEAP_SIZE);
        return;
    }
    heap_ring_init(&heap_ring_, heap_base_, PTO_HEAP_SIZE);

    // Initialize shared header
    memset(&shared_header_, 0, sizeof(shared_header_));

    // Enable ring-based allocation
    use_ring_allocation_ = true;

    printf("[PTO] Ring buffers initialized:\n");
    printf("      TaskRing: %d slots\n", PTO_TASK_WINDOW_SIZE);
    printf("      HeapRing: %d bytes at %p\n", PTO_HEAP_SIZE, (void*)heap_base_);
}

// =============================================================================
// Scope-Based Lifecycle
// =============================================================================

void Runtime::pto_scope_begin() {
    if (scope_stack_top_ >= PTO_MAX_SCOPE_DEPTH) {
        fprintf(stderr, "[PTO] ERROR: Scope stack overflow (max=%d)\n", PTO_MAX_SCOPE_DEPTH);
        return;
    }
    // Record current position in scope_tasks_ — this scope owns tasks from here onward
    scope_stack_[scope_stack_top_++] = scope_tasks_top_;
    printf("[PTO] Scope begin (depth=%d, scope_tasks_start=%d)\n", scope_stack_top_, scope_tasks_top_);
}

void Runtime::pto_scope_end() {
    if (scope_stack_top_ <= 0) {
        fprintf(stderr, "[PTO] ERROR: Scope stack underflow\n");
        return;
    }

    int32_t tasks_begin = scope_stack_[--scope_stack_top_];
    int32_t tasks_end = scope_tasks_top_;

    printf("[PTO] Scope end (depth=%d, %d tasks)\n",
           scope_stack_top_ + 1, tasks_end - tasks_begin);

    // Scope releases its reference on each task it directly owns
    // Only tasks pushed to scope_tasks_ by this scope are iterated (RAII)
    for (int32_t i = tasks_begin; i < tasks_end; i++) {
        int32_t tid = scope_tasks_[i];
        tasks[tid].fanout_refcount++;
        int target = tasks[tid].fanout_count + 1;
        printf("[PTO] Task %d: fanout_refcount++ → %d (scope_end, target=%d)\n",
               tid, tasks[tid].fanout_refcount, target);

        // Check if task can transition to CONSUMED
        if (tasks[tid].fanout_refcount == target &&
            tasks[tid].state == TaskState::COMPLETED) {
            tasks[tid].state = TaskState::CONSUMED;
            printf("[PTO] Task %d → CONSUMED (via scope_end: fanout_refcount=%d == fanout_count+1=%d)\n",
                   tid, tasks[tid].fanout_refcount, target);
        }
    }

    // Pop this scope's tasks off the flat list (RAII: child scope tasks already gone)
    scope_tasks_top_ = tasks_begin;
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
    // Note: no ref_count - buffer lifetime tied to task lifetime

    return new_handle;
}

int Runtime::pto_submit_task(int32_t func_id, PTOWorkerType worker_type,
                             PTOParam* params, int32_t param_count) {
    // Packed output buffer allocation using HeapRing
    // When ring allocation is enabled, allocate all outputs as a single packed buffer
    void* packed_base = nullptr;
    int32_t packed_buffer_offset = 0;
    int32_t packed_buffer_size = 0;

    if (use_ring_allocation_) {
        // Calculate total output size for new buffers (addr == 0)
        int32_t total_output_size = 0;
        for (int32_t i = 0; i < param_count; i++) {
            if (params[i].type == PTOParamType::OUTPUT && params[i].buffer != nullptr) {
                if (params[i].buffer->addr == 0) {
                    total_output_size += ALIGN_UP(params[i].buffer->size, PTO_ALIGNMENT);
                }
            }
        }

        // Single allocation from HeapRing if there are new outputs
        if (total_output_size > 0) {
            // Back-pressure: stall if heap ring is full
            packed_base = heap_ring_alloc(&heap_ring_, total_output_size,
                                          &shared_header_.heap_tail);
            packed_buffer_offset = heap_ring_offset(&heap_ring_, packed_base);
            packed_buffer_size = total_output_size;

            printf("[PTO] Task: packed allocation %d bytes at offset %d\n",
                   total_output_size, packed_buffer_offset);
        }

        // Assign sub-offsets within packed buffer
        int32_t offset = 0;
        for (int32_t i = 0; i < param_count; i++) {
            if (params[i].type == PTOParamType::OUTPUT && params[i].buffer != nullptr) {
                if (params[i].buffer->addr == 0) {
                    // New buffer: assign address within packed buffer
                    params[i].buffer->addr = (uint64_t)((char*)packed_base + offset);
                    params[i].buffer->version = 0;
                    offset += ALIGN_UP(params[i].buffer->size, PTO_ALIGNMENT);
                }
                // Update tensor descriptor with address
                params[i].tensor.addr = params[i].buffer->addr;
            }
        }
    } else {
        // Legacy: Allocate OUTPUT buffers individually
        // Only allocate for truly new buffers (addr == 0).
        // Versioned handles from pto_version_inc already have addr set.
        for (int32_t i = 0; i < param_count; i++) {
            if (params[i].type == PTOParamType::OUTPUT && params[i].buffer != nullptr) {
                if (params[i].buffer->addr == 0) {
                    // New buffer: allocate and set version to 0
                    int32_t size = params[i].buffer->size;
                    void* dev_ptr = host_api.device_malloc(size);
                    params[i].buffer->addr = (uint64_t)dev_ptr;
                    params[i].buffer->version = 0;
                }
                // Update tensor descriptor with (possibly pre-existing) address
                params[i].tensor.addr = params[i].buffer->addr;
            }
        }
    }

    // Build kernel args array from params
    uint64_t args[RUNTIME_MAX_ARGS];
    int num_args = 0;

    for (int32_t i = 0; i < param_count && num_args < RUNTIME_MAX_ARGS; i++) {
        if (params[i].type == PTOParamType::SCALAR) {
            args[num_args++] = params[i].scalar_value;
        } else {
            args[num_args++] = params[i].buffer->addr;
        }
    }

    // Add task using legacy API
    int task_id = add_task(args, num_args, func_id, worker_type);
    if (task_id < 0) return -1;

    // Record packed buffer info on task
    tasks[task_id].packed_buffer_offset = packed_buffer_offset;
    tasks[task_id].packed_buffer_size = packed_buffer_size;

    // Update shared header with current task index
    if (use_ring_allocation_) {
        shared_header_.current_task_index = next_task_id;
        shared_header_.heap_top = heap_ring_.top;
    }

    // Tasks must be submitted inside a scope
    if (scope_stack_top_ <= 0) {
        fprintf(stderr, "[PTO] ERROR: Task %d submitted outside of scope\n", task_id);
        return -1;
    }

    // Push task onto the current scope's task list
    if (scope_tasks_top_ < RUNTIME_MAX_TASKS) {
        scope_tasks_[scope_tasks_top_++] = task_id;
    } else {
        fprintf(stderr, "[PTO] ERROR: scope_tasks_ overflow\n");
    }

    // Automatic dependency detection via TensorMap
    for (int32_t i = 0; i < param_count; i++) {
        if (params[i].type == PTOParamType::INPUT && params[i].buffer != nullptr) {
            // INPUT: Look up producer for this tensor (skip stale entries via shared_header_)
            int32_t producer = tensormap_lookup(&tensor_map_, &params[i].tensor,
                                                shared_header_.last_task_alive);
            if (producer >= 0 && producer != task_id) {
                add_successor(producer, task_id);
            }
        } else if (params[i].type == PTOParamType::INOUT && params[i].buffer != nullptr) {
            // INOUT: Look up producer (like INPUT) but do NOT register as new producer
            int32_t producer = tensormap_lookup(&tensor_map_, &params[i].tensor,
                                                shared_header_.last_task_alive);
            if (producer >= 0 && producer != task_id) {
                add_successor(producer, task_id);
            }
            // Note: INOUT does NOT call tensormap_insert - it's not a new producer
        }
    }

    // Register OUTPUT tensors in TensorMap
    for (int32_t i = 0; i < param_count; i++) {
        if (params[i].type == PTOParamType::OUTPUT && params[i].buffer != nullptr) {
            tensormap_insert(&tensor_map_, &params[i].tensor,
                           task_id, params[i].buffer->version);
        }
    }

    // Set READY if no dependencies
    if (tasks[task_id].fanin.load(std::memory_order_acquire) == 0) {
        tasks[task_id].state = TaskState::READY;
    }

    return task_id;
}
