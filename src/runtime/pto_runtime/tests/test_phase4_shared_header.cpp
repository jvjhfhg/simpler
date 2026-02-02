/**
 * Phase 4 Test: PTOSharedHeader and TensorMap Staleness
 *
 * Tests Gap #7 (PTOSharedHeader), Gap #8 (TensorMap Staleness):
 * - Shared header fields updated correctly by orchestrator and scheduler
 * - last_task_alive advances as tasks are CONSUMED
 * - heap_tail advances based on packed buffer of consumed tasks
 * - TensorMap staleness filtering skips entries for consumed tasks
 * - Orchestrator/Scheduler communication via shared memory
 *
 * Compile:
 *   g++ -std=c++17 -I../runtime -o test_phase4 test_phase4_shared_header.cpp ../runtime/runtime.cpp
 */

#include "runtime.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ============================================================================
// Mock host API
// ============================================================================

static std::vector<void*> allocated_blocks;

static void* mock_device_malloc(size_t size) {
    void* ptr;
    if (size >= 1024 * 1024) {
        ptr = aligned_alloc(PTO_ALIGNMENT, size);
    } else {
        ptr = malloc(size);
    }
    allocated_blocks.push_back(ptr);
    return ptr;
}

static void mock_device_free(void* ptr) {
    free(ptr);
}

static int mock_copy_to_device(void* dev, const void* host, size_t size) {
    memcpy(dev, host, size);
    return 0;
}

static int mock_copy_from_device(void* host, const void* dev, size_t size) {
    memcpy(host, dev, size);
    return 0;
}

static void cleanup_allocations() {
    for (void* ptr : allocated_blocks) {
        free(ptr);
    }
    allocated_blocks.clear();
}

// ============================================================================
// Helpers
// ============================================================================

static PTOTensorDescriptor make_tensor_bbox(uint64_t addr, int32_t size) {
    PTOTensorDescriptor t = {};
    t.addr = addr;
    t.start_offset = 0;
    t.strides[0] = 1;
    t.repeats[0] = size;
    t.n_dims = 1;
    t.strategy = PTOOverlapStrategy::BOUNDING_BOX;
    return t;
}

static PTOParam make_scalar_param(uint64_t value) {
    PTOParam p = {};
    p.type = PTOParamType::SCALAR;
    p.buffer = nullptr;
    p.scalar_value = value;
    return p;
}

static PTOParam make_input_param(PTOBufferHandle* buf, int32_t size) {
    PTOParam p = {};
    p.type = PTOParamType::INPUT;
    p.tensor = make_tensor_bbox(buf->addr, size);
    p.buffer = buf;
    p.scalar_value = 0;
    return p;
}

static PTOParam make_output_param(PTOBufferHandle* buf, int32_t size) {
    PTOParam p = {};
    p.type = PTOParamType::OUTPUT;
    p.tensor = make_tensor_bbox(0, size);
    p.buffer = buf;
    p.scalar_value = 0;
    return p;
}

static PTOBufferHandle make_external_handle(void* addr, int32_t size) {
    PTOBufferHandle h = {};
    h.addr = (uint64_t)addr;
    h.size = size;
    h.version = 0;
    return h;
}

static PTOBufferHandle make_output_handle(int32_t size) {
    PTOBufferHandle h = {};
    h.addr = 0;
    h.size = size;
    h.version = 0;
    return h;
}

// ============================================================================
// Test counters
// ============================================================================

static int tests_passed = 0;
static int tests_failed = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        printf("  FAIL: %s (line %d)\n", msg, __LINE__); \
        tests_failed++; \
    } else { \
        printf("  PASS: %s\n", msg); \
        tests_passed++; \
    } \
} while (0)

// ============================================================================
// Test 1: Shared header initialization
// ============================================================================

static void test_shared_header_init() {
    printf("\n=== Test 1: Shared Header Initialization ===\n");

    Runtime runtime;
    runtime.host_api = {mock_device_malloc, mock_device_free, mock_copy_to_device, mock_copy_from_device};
    runtime.pto_init();
    runtime.pto_init_rings();

    PTOSharedHeader* header = runtime.get_shared_header();

    CHECK(header != nullptr, "Shared header accessible");
    CHECK(header->current_task_index == 0, "current_task_index initialized to 0");
    CHECK(header->heap_top == 0, "heap_top initialized to 0");
    CHECK(header->last_task_alive == 0, "last_task_alive initialized to 0");
    CHECK(header->heap_tail == 0, "heap_tail initialized to 0");
    CHECK(header->orchestrator_done == 0, "orchestrator_done initialized to 0");
    CHECK(header->scheduler_done == 0, "scheduler_done initialized to 0");
}

// ============================================================================
// Test 2: Orchestrator updates current_task_index and heap_top
// ============================================================================

static void test_orchestrator_updates() {
    printf("\n=== Test 2: Orchestrator Updates Shared Header ===\n");

    Runtime runtime;
    runtime.host_api = {mock_device_malloc, mock_device_free, mock_copy_to_device, mock_copy_from_device};
    runtime.pto_init();
    runtime.pto_init_rings();

    int32_t BYTES = 64;
    int32_t aligned_size = ALIGN_UP(BYTES, PTO_ALIGNMENT);

    void* dev_a_ptr = mock_device_malloc(BYTES);
    PTOBufferHandle dev_a = make_external_handle(dev_a_ptr, BYTES);
    PTOBufferHandle dev_b = make_output_handle(BYTES);
    PTOBufferHandle dev_c = make_output_handle(BYTES);

    PTOSharedHeader* header = runtime.get_shared_header();

    CHECK(header->current_task_index == 0, "Before submission: current_task_index = 0");
    CHECK(header->heap_top == 0, "Before submission: heap_top = 0");

    runtime.pto_scope_begin();

    // T0: allocates dev_b
    PTOParam params0[] = {
        make_input_param(&dev_a, BYTES),
        make_output_param(&dev_b, BYTES),
        make_scalar_param(64),
    };
    int t0 = runtime.pto_submit_task(0, PTOWorkerType::VECTOR, params0, 3);

    CHECK(header->current_task_index == 1, "After T0: current_task_index = 1");
    CHECK(header->heap_top == aligned_size, "After T0: heap_top = aligned_size");

    // T1: allocates dev_c
    PTOParam params1[] = {
        make_input_param(&dev_b, BYTES),
        make_output_param(&dev_c, BYTES),
        make_scalar_param(64),
    };
    int t1 = runtime.pto_submit_task(1, PTOWorkerType::VECTOR, params1, 3);

    CHECK(header->current_task_index == 2, "After T1: current_task_index = 2");
    CHECK(header->heap_top == 2 * aligned_size, "After T1: heap_top = 2 * aligned_size");

    runtime.pto_scope_end();
}

// ============================================================================
// Test 3: Simulate scheduler advancing last_task_alive and heap_tail
// ============================================================================

static void test_scheduler_advances_pointers() {
    printf("\n=== Test 3: Scheduler Advances last_task_alive and heap_tail ===\n");

    Runtime runtime;
    runtime.host_api = {mock_device_malloc, mock_device_free, mock_copy_to_device, mock_copy_from_device};
    runtime.pto_init();
    runtime.pto_init_rings();

    int32_t BYTES = 64;
    int32_t aligned_size = ALIGN_UP(BYTES, PTO_ALIGNMENT);

    void* dev_a_ptr = mock_device_malloc(BYTES);
    PTOBufferHandle dev_a = make_external_handle(dev_a_ptr, BYTES);
    PTOBufferHandle dev_b = make_output_handle(BYTES);
    PTOBufferHandle dev_c = make_output_handle(BYTES);

    runtime.pto_scope_begin();

    // T0 → T1 chain
    PTOParam params0[] = {
        make_input_param(&dev_a, BYTES),
        make_output_param(&dev_b, BYTES),
        make_scalar_param(64),
    };
    int t0 = runtime.pto_submit_task(0, PTOWorkerType::VECTOR, params0, 3);

    PTOParam params1[] = {
        make_input_param(&dev_b, BYTES),
        make_output_param(&dev_c, BYTES),
        make_scalar_param(64),
    };
    int t1 = runtime.pto_submit_task(1, PTOWorkerType::VECTOR, params1, 3);

    Task* task0 = runtime.get_task(t0);
    Task* task1 = runtime.get_task(t1);
    PTOSharedHeader* header = runtime.get_shared_header();

    CHECK(header->last_task_alive == 0, "Initially: last_task_alive = 0");
    CHECK(header->heap_tail == 0, "Initially: heap_tail = 0");

    // Simulate T0 execution
    task0->state = TaskState::RUNNING;
    task0->state = TaskState::COMPLETED;
    if (task1->fanin.fetch_sub(1) == 1) {
        task1->state = TaskState::READY;
    }
    task0->fanout_refcount++;
    runtime.check_consumed(t0);

    // T0 not CONSUMED yet (scope still holds reference)
    CHECK(task0->state == TaskState::COMPLETED, "T0 still COMPLETED (scope holds ref)");
    CHECK(header->last_task_alive == 0, "last_task_alive unchanged (T0 not CONSUMED)");

    // Simulate T1 execution
    task1->state = TaskState::RUNNING;
    task1->state = TaskState::COMPLETED;

    // End scope - releases scope references
    runtime.pto_scope_end();

    CHECK(task0->state == TaskState::CONSUMED, "T0 → CONSUMED after scope_end");
    CHECK(task1->state == TaskState::CONSUMED, "T1 → CONSUMED after scope_end");

    // Simulate scheduler advancing pointers (as in aicpu_executor.cpp)
    // This is what the scheduler would do after detecting CONSUMED tasks
    int32_t task_count = runtime.get_task_count();
    int32_t last_alive = header->last_task_alive;
    while (last_alive < task_count) {
        Task* t = runtime.get_task(last_alive);
        if (t == nullptr || t->state != TaskState::CONSUMED) {
            break;
        }
        last_alive++;
    }
    header->last_task_alive = last_alive;

    // Advance heap_tail
    if (last_alive > 0) {
        Task* last_consumed = runtime.get_task(last_alive - 1);
        int32_t new_heap_tail = last_consumed->packed_buffer_offset
                              + last_consumed->packed_buffer_size;
        header->heap_tail = new_heap_tail;
    }

    CHECK(header->last_task_alive == 2, "last_task_alive advanced to 2 (all tasks CONSUMED)");
    CHECK(header->heap_tail == 2 * aligned_size, "heap_tail advanced to end of all packed buffers");

    runtime.print_runtime();
}

// ============================================================================
// Test 4: TensorMap staleness filtering
// ============================================================================

static void test_tensormap_staleness() {
    printf("\n=== Test 4: TensorMap Staleness Filtering ===\n");

    Runtime runtime;
    runtime.host_api = {mock_device_malloc, mock_device_free, mock_copy_to_device, mock_copy_from_device};
    runtime.pto_init();
    runtime.pto_init_rings();

    int32_t BYTES = 64;

    void* dev_a_ptr = mock_device_malloc(BYTES);
    PTOBufferHandle dev_a = make_external_handle(dev_a_ptr, BYTES);
    PTOBufferHandle dev_b = make_output_handle(BYTES);
    PTOBufferHandle dev_c = make_output_handle(BYTES);
    PTOBufferHandle dev_d = make_output_handle(BYTES);

    runtime.pto_scope_begin();

    // T0: produces dev_b
    PTOParam params0[] = {
        make_input_param(&dev_a, BYTES),
        make_output_param(&dev_b, BYTES),
        make_scalar_param(64),
    };
    int t0 = runtime.pto_submit_task(0, PTOWorkerType::VECTOR, params0, 3);

    // T1: consumes dev_b, produces dev_c (depends on T0)
    PTOParam params1[] = {
        make_input_param(&dev_b, BYTES),
        make_output_param(&dev_c, BYTES),
        make_scalar_param(64),
    };
    int t1 = runtime.pto_submit_task(1, PTOWorkerType::VECTOR, params1, 3);

    Task* task0 = runtime.get_task(t0);
    Task* task1 = runtime.get_task(t1);

    CHECK(task1->fanin.load() == 1, "T1 has 1 dependency (T0)");
    CHECK(task0->fanout_count == 1, "T0 has 1 consumer (T1)");

    // Simulate T0 complete and CONSUMED
    task0->state = TaskState::RUNNING;
    task0->state = TaskState::COMPLETED;
    if (task1->fanin.fetch_sub(1) == 1) {
        task1->state = TaskState::READY;
    }
    task0->fanout_refcount++;
    runtime.check_consumed(t0);

    // Simulate scope_end to make T0 CONSUMED
    // First increment fanout_refcount for scope (T0 is in scope)
    task0->fanout_refcount++;
    int target = task0->fanout_count + 1;
    if (task0->fanout_refcount == target && task0->state == TaskState::COMPLETED) {
        task0->state = TaskState::CONSUMED;
    }

    CHECK(task0->state == TaskState::CONSUMED, "T0 → CONSUMED");

    // Update shared header to reflect T0 consumed
    PTOSharedHeader* header = runtime.get_shared_header();
    header->last_task_alive = 1;  // T0 is consumed, so last_task_alive = 1

    // Now submit T2 that consumes dev_b (same tensor as T1)
    // With staleness filtering, T0's tensormap entry should be skipped
    // because T0 (producer_task_id=0) < last_task_alive (1)
    PTOParam params2[] = {
        make_input_param(&dev_b, BYTES),  // dev_b was produced by T0 (now stale)
        make_output_param(&dev_d, BYTES),
        make_scalar_param(64),
    };
    int t2 = runtime.pto_submit_task(2, PTOWorkerType::VECTOR, params2, 3);

    Task* task2 = runtime.get_task(t2);

    // T2 should NOT have dependency on T0 because T0's entry is stale
    // The tensormap_lookup should skip T0's entry
    CHECK(task2->fanin.load() == 0, "T2 has 0 dependencies (T0 entry filtered as stale)");

    printf("  Note: With staleness filtering, T2 doesn't depend on consumed T0\n");

    runtime.pto_scope_end();
}

// ============================================================================
// Test 5: Fresh task still creates dependencies (not stale)
// ============================================================================

static void test_fresh_task_dependency() {
    printf("\n=== Test 5: Fresh Task Dependencies (Not Stale) ===\n");

    Runtime runtime;
    runtime.host_api = {mock_device_malloc, mock_device_free, mock_copy_to_device, mock_copy_from_device};
    runtime.pto_init();
    runtime.pto_init_rings();

    int32_t BYTES = 64;

    void* dev_a_ptr = mock_device_malloc(BYTES);
    PTOBufferHandle dev_a = make_external_handle(dev_a_ptr, BYTES);
    PTOBufferHandle dev_b = make_output_handle(BYTES);
    PTOBufferHandle dev_c = make_output_handle(BYTES);

    PTOSharedHeader* header = runtime.get_shared_header();
    CHECK(header->last_task_alive == 0, "Initially: last_task_alive = 0");

    runtime.pto_scope_begin();

    // T0: produces dev_b
    PTOParam params0[] = {
        make_input_param(&dev_a, BYTES),
        make_output_param(&dev_b, BYTES),
        make_scalar_param(64),
    };
    int t0 = runtime.pto_submit_task(0, PTOWorkerType::VECTOR, params0, 3);

    // T1: consumes dev_b (T0 NOT consumed, so dependency should be created)
    PTOParam params1[] = {
        make_input_param(&dev_b, BYTES),
        make_output_param(&dev_c, BYTES),
        make_scalar_param(64),
    };
    int t1 = runtime.pto_submit_task(1, PTOWorkerType::VECTOR, params1, 3);

    Task* task0 = runtime.get_task(t0);
    Task* task1 = runtime.get_task(t1);

    // T0 is NOT consumed (last_task_alive = 0, T0 id = 0, 0 >= 0, so not stale)
    CHECK(task1->fanin.load() == 1, "T1 has 1 dependency (T0 is fresh, not stale)");
    CHECK(task0->fanout_count == 1, "T0 has 1 consumer");
    CHECK(task0->fanout[0] == t1, "T0's consumer is T1");

    runtime.pto_scope_end();
}

// ============================================================================
// Test 6: Multiple consumed tasks - progressive advancement
// ============================================================================

static void test_progressive_advancement() {
    printf("\n=== Test 6: Progressive Advancement of last_task_alive ===\n");

    Runtime runtime;
    runtime.host_api = {mock_device_malloc, mock_device_free, mock_copy_to_device, mock_copy_from_device};
    runtime.pto_init();
    runtime.pto_init_rings();

    int32_t BYTES = 64;
    int32_t aligned_size = ALIGN_UP(BYTES, PTO_ALIGNMENT);

    void* dev_a_ptr = mock_device_malloc(BYTES);
    PTOBufferHandle dev_a = make_external_handle(dev_a_ptr, BYTES);
    PTOBufferHandle dev_b = make_output_handle(BYTES);
    PTOBufferHandle dev_c = make_output_handle(BYTES);
    PTOBufferHandle dev_d = make_output_handle(BYTES);

    runtime.pto_scope_begin();

    // Create 3 independent tasks
    PTOParam params0[] = {
        make_input_param(&dev_a, BYTES),
        make_output_param(&dev_b, BYTES),
        make_scalar_param(64),
    };
    int t0 = runtime.pto_submit_task(0, PTOWorkerType::VECTOR, params0, 3);

    PTOParam params1[] = {
        make_input_param(&dev_a, BYTES),
        make_output_param(&dev_c, BYTES),
        make_scalar_param(64),
    };
    int t1 = runtime.pto_submit_task(1, PTOWorkerType::VECTOR, params1, 3);

    PTOParam params2[] = {
        make_input_param(&dev_a, BYTES),
        make_output_param(&dev_d, BYTES),
        make_scalar_param(64),
    };
    int t2 = runtime.pto_submit_task(2, PTOWorkerType::VECTOR, params2, 3);

    Task* task0 = runtime.get_task(t0);
    Task* task1 = runtime.get_task(t1);
    Task* task2 = runtime.get_task(t2);
    PTOSharedHeader* header = runtime.get_shared_header();

    // All independent
    CHECK(task0->fanin.load() == 0, "T0 has no dependencies");
    CHECK(task1->fanin.load() == 0, "T1 has no dependencies");
    CHECK(task2->fanin.load() == 0, "T2 has no dependencies");

    // Complete and consume T0
    task0->state = TaskState::COMPLETED;
    task0->fanout_refcount++;  // scope releases
    if (task0->fanout_refcount == task0->fanout_count + 1 && task0->state == TaskState::COMPLETED) {
        task0->state = TaskState::CONSUMED;
    }
    CHECK(task0->state == TaskState::CONSUMED, "T0 → CONSUMED");

    // Advance last_task_alive for T0
    int32_t task_count = runtime.get_task_count();
    int32_t last_alive = 0;
    while (last_alive < task_count && runtime.get_task(last_alive)->state == TaskState::CONSUMED) {
        last_alive++;
    }
    header->last_task_alive = last_alive;

    CHECK(header->last_task_alive == 1, "After T0 consumed: last_task_alive = 1");
    CHECK(header->heap_tail == 0, "heap_tail still 0 (not updated yet)");

    // Update heap_tail
    Task* last_consumed = runtime.get_task(last_alive - 1);
    header->heap_tail = last_consumed->packed_buffer_offset + last_consumed->packed_buffer_size;
    CHECK(header->heap_tail == aligned_size, "heap_tail = T0's packed buffer end");

    // T1 is NOT consumed yet, so pointer shouldn't advance
    task1->state = TaskState::COMPLETED;
    // Don't increment fanout_refcount yet - simulating partial completion

    last_alive = header->last_task_alive;
    while (last_alive < task_count && runtime.get_task(last_alive)->state == TaskState::CONSUMED) {
        last_alive++;
    }
    CHECK(last_alive == 1, "last_task_alive stays at 1 (T1 not CONSUMED)");

    // Now complete and consume T1
    task1->fanout_refcount++;
    if (task1->fanout_refcount == task1->fanout_count + 1 && task1->state == TaskState::COMPLETED) {
        task1->state = TaskState::CONSUMED;
    }
    CHECK(task1->state == TaskState::CONSUMED, "T1 → CONSUMED");

    // Advance again
    last_alive = header->last_task_alive;
    while (last_alive < task_count && runtime.get_task(last_alive)->state == TaskState::CONSUMED) {
        last_alive++;
    }
    header->last_task_alive = last_alive;

    CHECK(header->last_task_alive == 2, "After T1 consumed: last_task_alive = 2");

    // Complete and consume T2
    task2->state = TaskState::COMPLETED;
    task2->fanout_refcount++;
    if (task2->fanout_refcount == task2->fanout_count + 1 && task2->state == TaskState::COMPLETED) {
        task2->state = TaskState::CONSUMED;
    }

    last_alive = header->last_task_alive;
    while (last_alive < task_count && runtime.get_task(last_alive)->state == TaskState::CONSUMED) {
        last_alive++;
    }
    header->last_task_alive = last_alive;

    CHECK(header->last_task_alive == 3, "After T2 consumed: last_task_alive = 3 (all tasks)");

    // Final heap_tail update
    if (last_alive > 0) {
        Task* final_consumed = runtime.get_task(last_alive - 1);
        header->heap_tail = final_consumed->packed_buffer_offset + final_consumed->packed_buffer_size;
    }
    CHECK(header->heap_tail == 3 * aligned_size, "Final heap_tail = end of all packed buffers");

    runtime.pto_scope_end();
    runtime.print_runtime();
}

// ============================================================================
// Test 7: Phase 1+2+3 compatibility with shared header
// ============================================================================

static void test_all_phases_compatibility() {
    printf("\n=== Test 7: Phase 1+2+3+4 Compatibility ===\n");

    Runtime runtime;
    runtime.host_api = {mock_device_malloc, mock_device_free, mock_copy_to_device, mock_copy_from_device};
    runtime.pto_init();
    runtime.pto_init_rings();

    int32_t BYTES = 64;
    int32_t aligned_size = ALIGN_UP(BYTES, PTO_ALIGNMENT);

    void* dev_a_ptr = mock_device_malloc(BYTES);
    PTOBufferHandle dev_a = make_external_handle(dev_a_ptr, BYTES);
    PTOBufferHandle dev_b = make_output_handle(BYTES);
    PTOBufferHandle dev_c = make_output_handle(BYTES);

    PTOSharedHeader* header = runtime.get_shared_header();

    runtime.pto_scope_begin();

    // T0 → T1 chain
    PTOParam params0[] = {
        make_input_param(&dev_a, BYTES),
        make_output_param(&dev_b, BYTES),
        make_scalar_param(64),
    };
    int t0 = runtime.pto_submit_task(0, PTOWorkerType::VECTOR, params0, 3);

    PTOParam params1[] = {
        make_input_param(&dev_b, BYTES),
        make_output_param(&dev_c, BYTES),
        make_scalar_param(64),
    };
    int t1 = runtime.pto_submit_task(1, PTOWorkerType::VECTOR, params1, 3);

    Task* task0 = runtime.get_task(t0);
    Task* task1 = runtime.get_task(t1);

    // Phase 1: State machine
    CHECK(task0->state == TaskState::READY, "T0 starts READY");
    CHECK(task1->state == TaskState::PENDING, "T1 starts PENDING");

    // Phase 3: Packed buffers allocated
    CHECK(task0->packed_buffer_size == aligned_size, "T0 has packed buffer");
    CHECK(task1->packed_buffer_size == aligned_size, "T1 has packed buffer");

    // Phase 4: Shared header updated
    CHECK(header->current_task_index == 2, "current_task_index = 2");
    CHECK(header->heap_top == 2 * aligned_size, "heap_top = 2 * aligned_size");
    CHECK(header->last_task_alive == 0, "last_task_alive = 0 (nothing consumed yet)");

    // Execute T0
    task0->state = TaskState::RUNNING;
    task0->state = TaskState::COMPLETED;
    if (task1->fanin.fetch_sub(1) == 1) {
        task1->state = TaskState::READY;
    }
    task0->fanout_refcount++;  // Consumer T1 completed its part
    runtime.check_consumed(t0);

    CHECK(task1->state == TaskState::READY, "T1 → READY after T0 completes");

    // Execute T1
    task1->state = TaskState::RUNNING;
    task1->state = TaskState::COMPLETED;

    // Phase 2: Scope end
    runtime.pto_scope_end();

    CHECK(task0->state == TaskState::CONSUMED, "T0 → CONSUMED after scope_end");
    CHECK(task1->state == TaskState::CONSUMED, "T1 → CONSUMED after scope_end");

    // Phase 4: Simulate scheduler advancing pointers
    int32_t task_count = runtime.get_task_count();
    int32_t last_alive = 0;
    while (last_alive < task_count && runtime.get_task(last_alive)->state == TaskState::CONSUMED) {
        last_alive++;
    }
    header->last_task_alive = last_alive;
    if (last_alive > 0) {
        Task* last_consumed = runtime.get_task(last_alive - 1);
        header->heap_tail = last_consumed->packed_buffer_offset + last_consumed->packed_buffer_size;
    }

    CHECK(header->last_task_alive == 2, "last_task_alive = 2 (all consumed)");
    CHECK(header->heap_tail == 2 * aligned_size, "heap_tail = end of all buffers");

    runtime.print_runtime();
}

// ============================================================================
// Test 8: No packed buffers - heap_tail stays at 0
// ============================================================================

static void test_no_packed_buffers() {
    printf("\n=== Test 8: Tasks Without Packed Buffers ===\n");

    Runtime runtime;
    runtime.host_api = {mock_device_malloc, mock_device_free, mock_copy_to_device, mock_copy_from_device};
    runtime.pto_init();
    runtime.pto_init_rings();

    int32_t BYTES = 64;

    void* dev_a_ptr = mock_device_malloc(BYTES);
    void* dev_b_ptr = mock_device_malloc(BYTES);
    PTOBufferHandle dev_a = make_external_handle(dev_a_ptr, BYTES);
    PTOBufferHandle dev_b = make_external_handle(dev_b_ptr, BYTES);

    PTOSharedHeader* header = runtime.get_shared_header();

    runtime.pto_scope_begin();

    // Task with only inputs (no outputs to allocate)
    PTOParam params0[] = {
        make_input_param(&dev_a, BYTES),
        make_input_param(&dev_b, BYTES),
        make_scalar_param(64),
    };
    int t0 = runtime.pto_submit_task(0, PTOWorkerType::VECTOR, params0, 3);

    Task* task0 = runtime.get_task(t0);

    CHECK(task0->packed_buffer_size == 0, "T0 has no packed buffer");
    CHECK(header->heap_top == 0, "heap_top unchanged");

    // Complete and consume T0
    task0->state = TaskState::COMPLETED;
    task0->fanout_refcount++;  // scope releases
    if (task0->fanout_refcount == task0->fanout_count + 1 && task0->state == TaskState::COMPLETED) {
        task0->state = TaskState::CONSUMED;
    }

    // Advance pointers
    int32_t task_count = runtime.get_task_count();
    int32_t last_alive = 0;
    while (last_alive < task_count && runtime.get_task(last_alive)->state == TaskState::CONSUMED) {
        last_alive++;
    }
    header->last_task_alive = last_alive;
    if (last_alive > 0) {
        Task* last_consumed = runtime.get_task(last_alive - 1);
        header->heap_tail = last_consumed->packed_buffer_offset + last_consumed->packed_buffer_size;
    }

    CHECK(header->last_task_alive == 1, "last_task_alive = 1");
    CHECK(header->heap_tail == 0, "heap_tail = 0 (no packed buffers)");

    runtime.pto_scope_end();
}

// ============================================================================
// Main
// ============================================================================

int main() {
    printf("============================================================\n");
    printf("Phase 4 Test: PTOSharedHeader and TensorMap Staleness\n");
    printf("============================================================\n");

    test_shared_header_init();
    test_orchestrator_updates();
    test_scheduler_advances_pointers();
    test_tensormap_staleness();
    test_fresh_task_dependency();
    test_progressive_advancement();
    test_all_phases_compatibility();
    test_no_packed_buffers();

    printf("\n============================================================\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("============================================================\n");

    cleanup_allocations();

    return tests_failed > 0 ? 1 : 0;
}
