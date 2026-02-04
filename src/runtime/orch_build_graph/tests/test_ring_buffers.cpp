/**
 * Phase 3 Test: TaskRing and HeapRing Integration
 *
 * Tests Gap #1 (Task Ring Buffer), Gap #2 (GM Heap / HeapRing), Gap #11 (Packed Output Buffers):
 * - TaskRing and HeapRing initialization
 * - Packed output buffer allocation via HeapRing
 * - Output addresses are contiguous (packed) per task
 * - HeapRing top advances by total aligned output size
 * - packed_buffer_offset and packed_buffer_size tracked on Task struct
 * - Shared header updates (current_task_index, heap_top)
 *
 * Compile:
 *   g++ -std=c++17 -I../runtime -o test_phase3 test_phase3_ring_buffers.cpp ../runtime/runtime.cpp
 */

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "dep_list_pool.h"
#include "runtime.h"

// ============================================================================
// Mock host API
// ============================================================================

static std::vector<void*> allocated_blocks;

static void* mock_device_malloc(size_t size) {
    // Use aligned allocation for large buffers (like HeapRing)
    // to ensure output addresses meet alignment requirements
    void* ptr;
    if (size >= 1024 * 1024) {  // Large allocations (>= 1MB)
        ptr = aligned_alloc(PTO_ALIGNMENT, size);
    } else {
        ptr = malloc(size);
    }
    allocated_blocks.push_back(ptr);
    return ptr;
}

static void mock_device_free(void* ptr) { free(ptr); }

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

static TensorDescriptor make_tensor_bbox(uint64_t addr, int32_t size) {
    TensorDescriptor t = {};
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

#define CHECK(cond, msg)                                     \
    do {                                                     \
        if (!(cond)) {                                       \
            printf("  FAIL: %s (line %d)\n", msg, __LINE__); \
            tests_failed++;                                  \
        } else {                                             \
            printf("  PASS: %s\n", msg);                     \
            tests_passed++;                                  \
        }                                                    \
    } while (0)

// ============================================================================
// Test 1: Ring buffer initialization
// ============================================================================

static void test_ring_init() {
    printf("\n=== Test 1: Ring Buffer Initialization ===\n");

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

    printf("  Ring buffers initialized successfully\n");
}

// ============================================================================
// Test 2: Single task with packed output allocation
// ============================================================================

static void test_single_task_packed_output() {
    printf("\n=== Test 2: Single Task with Packed Output Allocation ===\n");

    Runtime runtime;
    runtime.host_api = {mock_device_malloc, mock_device_free, mock_copy_to_device, mock_copy_from_device};
    runtime.pto_init();
    runtime.pto_init_rings();

    int32_t BYTES = 100;

    void* dev_a_ptr = mock_device_malloc(BYTES);
    PTOBufferHandle dev_a = make_external_handle(dev_a_ptr, BYTES);
    PTOBufferHandle dev_b = make_output_handle(BYTES);
    PTOBufferHandle dev_c = make_output_handle(BYTES);

    runtime.pto_scope_begin();

    // T0: 2 outputs, should be packed contiguously
    PTOParam params0[] = {
        make_input_param(&dev_a, BYTES),
        make_output_param(&dev_b, BYTES),
        make_output_param(&dev_c, BYTES),
        make_scalar_param(64),
    };
    int t0 = runtime.pto_submit_task(0, PTOWorkerType::VECTOR, params0, 4);

    Task* task0 = runtime.get_task(t0);

    // Check packed buffer info
    int32_t aligned_size = ALIGN_UP(BYTES, PTO_ALIGNMENT);
    int32_t expected_total = 2 * aligned_size;

    CHECK(task0->packed_buffer_offset == 0, "T0 packed_buffer_offset = 0 (first allocation)");
    CHECK(task0->packed_buffer_size == expected_total,
        ("T0 packed_buffer_size = " + std::to_string(expected_total) + " (2 aligned outputs)").c_str());

    // Check output addresses are contiguous
    uint64_t addr_b = dev_b.addr;
    uint64_t addr_c = dev_c.addr;
    CHECK(addr_b != 0, "dev_b.addr allocated");
    CHECK(addr_c != 0, "dev_c.addr allocated");
    CHECK(addr_c == addr_b + aligned_size, "Output addresses are contiguous (packed)");

    // Check shared header updates
    PTOSharedHeader* header = runtime.get_shared_header();
    CHECK(header->current_task_index == 1, "current_task_index advanced to 1");
    CHECK(header->heap_top == expected_total, ("heap_top advanced to " + std::to_string(expected_total)).c_str());

    runtime.pto_scope_end();
    runtime.print_runtime();
}

// ============================================================================
// Test 3: Multiple tasks with cumulative heap allocation
// ============================================================================

static void test_multiple_tasks_heap() {
    printf("\n=== Test 3: Multiple Tasks with Cumulative Heap Allocation ===\n");

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
    PTOBufferHandle dev_e = make_output_handle(BYTES);

    runtime.pto_scope_begin();

    // T0: 1 output (dev_b)
    PTOParam params0[] = {
        make_input_param(&dev_a, BYTES),
        make_output_param(&dev_b, BYTES),
        make_scalar_param(64),
    };
    int t0 = runtime.pto_submit_task(0, PTOWorkerType::VECTOR, params0, 3);

    Task* task0 = runtime.get_task(t0);
    CHECK(task0->packed_buffer_offset == 0, "T0 offset = 0");
    CHECK(task0->packed_buffer_size == aligned_size, "T0 size = 1 aligned buffer");

    // T1: 2 outputs (dev_c, dev_d)
    PTOParam params1[] = {
        make_input_param(&dev_b, BYTES),
        make_output_param(&dev_c, BYTES),
        make_output_param(&dev_d, BYTES),
        make_scalar_param(64),
    };
    int t1 = runtime.pto_submit_task(1, PTOWorkerType::VECTOR, params1, 4);

    Task* task1 = runtime.get_task(t1);
    CHECK(task1->packed_buffer_offset == aligned_size, "T1 offset = 1 aligned buffer");
    CHECK(task1->packed_buffer_size == 2 * aligned_size, "T1 size = 2 aligned buffers");

    // T2: 1 output (dev_e)
    PTOParam params2[] = {
        make_input_param(&dev_c, BYTES),
        make_input_param(&dev_d, BYTES),
        make_output_param(&dev_e, BYTES),
        make_scalar_param(64),
    };
    int t2 = runtime.pto_submit_task(2, PTOWorkerType::VECTOR, params2, 4);

    Task* task2 = runtime.get_task(t2);
    CHECK(task2->packed_buffer_offset == 3 * aligned_size, "T2 offset = 3 aligned buffers");
    CHECK(task2->packed_buffer_size == aligned_size, "T2 size = 1 aligned buffer");

    // Check total heap usage
    PTOSharedHeader* header = runtime.get_shared_header();
    CHECK(header->heap_top == 4 * aligned_size, "Total heap = 4 aligned buffers");

    // Verify output addresses follow heap layout
    CHECK(dev_b.addr != 0, "dev_b allocated");
    CHECK(dev_c.addr == dev_b.addr + aligned_size, "dev_c follows dev_b");
    CHECK(dev_d.addr == dev_c.addr + aligned_size, "dev_d follows dev_c (same task)");
    CHECK(dev_e.addr == dev_d.addr + aligned_size, "dev_e follows dev_d");

    runtime.pto_scope_end();
    runtime.print_runtime();
}

// ============================================================================
// Test 4: Task with no outputs
// ============================================================================

static void test_task_no_outputs() {
    printf("\n=== Test 4: Task with No Outputs ===\n");

    Runtime runtime;
    runtime.host_api = {mock_device_malloc, mock_device_free, mock_copy_to_device, mock_copy_from_device};
    runtime.pto_init();
    runtime.pto_init_rings();

    int32_t BYTES = 64;

    void* dev_a_ptr = mock_device_malloc(BYTES);
    void* dev_b_ptr = mock_device_malloc(BYTES);
    PTOBufferHandle dev_a = make_external_handle(dev_a_ptr, BYTES);
    PTOBufferHandle dev_b = make_external_handle(dev_b_ptr, BYTES);

    runtime.pto_scope_begin();

    // T0: no outputs (pure computation or side-effect task)
    PTOParam params0[] = {
        make_input_param(&dev_a, BYTES),
        make_input_param(&dev_b, BYTES),
        make_scalar_param(64),
    };
    int t0 = runtime.pto_submit_task(0, PTOWorkerType::VECTOR, params0, 3);

    Task* task0 = runtime.get_task(t0);
    CHECK(task0->packed_buffer_offset == 0, "T0 offset = 0 (no allocation)");
    CHECK(task0->packed_buffer_size == 0, "T0 size = 0 (no outputs)");

    PTOSharedHeader* header = runtime.get_shared_header();
    CHECK(header->heap_top == 0, "heap_top unchanged (no outputs)");

    runtime.pto_scope_end();
}

// ============================================================================
// Test 5: Pre-allocated output (version_inc) doesn't allocate again
// ============================================================================

static void test_preallocated_output() {
    printf("\n=== Test 5: Pre-allocated Output (version_inc) ===\n");

    Runtime runtime;
    runtime.host_api = {mock_device_malloc, mock_device_free, mock_copy_to_device, mock_copy_from_device};
    runtime.pto_init();
    runtime.pto_init_rings();

    int32_t BYTES = 64;
    int32_t aligned_size = ALIGN_UP(BYTES, PTO_ALIGNMENT);

    void* dev_a_ptr = mock_device_malloc(BYTES);
    PTOBufferHandle dev_a = make_external_handle(dev_a_ptr, BYTES);
    PTOBufferHandle dev_b = make_output_handle(BYTES);

    runtime.pto_scope_begin();

    // T0: allocates dev_b
    PTOParam params0[] = {
        make_input_param(&dev_a, BYTES),
        make_output_param(&dev_b, BYTES),
        make_scalar_param(64),
    };
    int t0 = runtime.pto_submit_task(0, PTOWorkerType::VECTOR, params0, 3);

    Task* task0 = runtime.get_task(t0);
    CHECK(task0->packed_buffer_size == aligned_size, "T0 allocated 1 buffer");

    uint64_t original_addr = dev_b.addr;
    CHECK(original_addr != 0, "dev_b has address");

    // version_inc creates new versioned handle with same address
    PTOBufferHandle* dev_b_v1 = runtime.pto_version_inc(&dev_b);
    CHECK(dev_b_v1 != nullptr, "version_inc returned handle");
    CHECK(dev_b_v1->addr == original_addr, "Versioned handle has same address");
    CHECK(dev_b_v1->version == 1, "Versioned handle has version 1");

    // T1: uses pre-allocated dev_b_v1, should NOT allocate again
    PTOParam params1[] = {
        make_input_param(&dev_a, BYTES),
        make_output_param(dev_b_v1, BYTES),
        make_scalar_param(64),
    };
    int t1 = runtime.pto_submit_task(1, PTOWorkerType::VECTOR, params1, 3);

    Task* task1 = runtime.get_task(t1);
    CHECK(task1->packed_buffer_size == 0, "T1 did not allocate (pre-allocated handle)");
    CHECK(dev_b_v1->addr == original_addr, "Address unchanged after T1");

    PTOSharedHeader* header = runtime.get_shared_header();
    CHECK(header->heap_top == aligned_size, "heap_top unchanged (no new allocation)");

    runtime.pto_scope_end();
}

// ============================================================================
// Test 6: Mixed allocation - some pre-allocated, some new
// ============================================================================

static void test_mixed_allocation() {
    printf("\n=== Test 6: Mixed Allocation ===\n");

    Runtime runtime;
    runtime.host_api = {mock_device_malloc, mock_device_free, mock_copy_to_device, mock_copy_from_device};
    runtime.pto_init();
    runtime.pto_init_rings();

    int32_t BYTES = 64;
    int32_t aligned_size = ALIGN_UP(BYTES, PTO_ALIGNMENT);

    void* dev_a_ptr = mock_device_malloc(BYTES);
    void* dev_b_ptr = mock_device_malloc(BYTES);
    PTOBufferHandle dev_a = make_external_handle(dev_a_ptr, BYTES);
    PTOBufferHandle dev_b = make_external_handle(dev_b_ptr, BYTES);  // Pre-allocated
    PTOBufferHandle dev_c = make_output_handle(BYTES);               // New allocation

    runtime.pto_scope_begin();

    // T0: dev_b is pre-allocated, dev_c needs allocation
    PTOParam params0[] = {
        make_input_param(&dev_a, BYTES),
        make_output_param(&dev_b, BYTES),  // Pre-allocated (addr != 0)
        make_output_param(&dev_c, BYTES),  // Needs allocation (addr == 0)
        make_scalar_param(64),
    };
    int t0 = runtime.pto_submit_task(0, PTOWorkerType::VECTOR, params0, 4);

    Task* task0 = runtime.get_task(t0);
    // Only dev_c should be allocated
    CHECK(task0->packed_buffer_size == aligned_size, "T0 allocated only 1 buffer (dev_c)");
    CHECK(dev_b.addr == (uint64_t)dev_b_ptr, "dev_b address unchanged (pre-allocated)");
    CHECK(dev_c.addr != 0, "dev_c allocated");
    CHECK(dev_c.addr != (uint64_t)dev_b_ptr, "dev_c has different address than dev_b");

    runtime.pto_scope_end();
}

// ============================================================================
// Test 7: Backward compatibility - legacy allocation mode
// ============================================================================

static void test_legacy_mode() {
    printf("\n=== Test 7: Legacy Allocation Mode (without pto_init_rings) ===\n");

    Runtime runtime;
    runtime.host_api = {mock_device_malloc, mock_device_free, mock_copy_to_device, mock_copy_from_device};
    runtime.pto_init();
    // NOTE: NOT calling pto_init_rings() - legacy mode

    int32_t BYTES = 64;

    void* dev_a_ptr = mock_device_malloc(BYTES);
    PTOBufferHandle dev_a = make_external_handle(dev_a_ptr, BYTES);
    PTOBufferHandle dev_b = make_output_handle(BYTES);
    PTOBufferHandle dev_c = make_output_handle(BYTES);

    runtime.pto_scope_begin();

    // T0: should use individual device_malloc per output
    PTOParam params0[] = {
        make_input_param(&dev_a, BYTES),
        make_output_param(&dev_b, BYTES),
        make_output_param(&dev_c, BYTES),
        make_scalar_param(64),
    };
    int t0 = runtime.pto_submit_task(0, PTOWorkerType::VECTOR, params0, 4);

    Task* task0 = runtime.get_task(t0);

    // In legacy mode, packed_buffer_* should be 0
    CHECK(task0->packed_buffer_offset == 0, "T0 packed_buffer_offset = 0 (legacy mode)");
    CHECK(task0->packed_buffer_size == 0, "T0 packed_buffer_size = 0 (legacy mode)");

    // But outputs should still be allocated
    CHECK(dev_b.addr != 0, "dev_b allocated (legacy mode)");
    CHECK(dev_c.addr != 0, "dev_c allocated (legacy mode)");

    // In legacy mode, addresses are NOT guaranteed to be contiguous
    // (each is a separate malloc)

    runtime.pto_scope_end();
}

// ============================================================================
// Test 8: Alignment verification
// ============================================================================

static void test_alignment() {
    printf("\n=== Test 8: Alignment Verification ===\n");

    Runtime runtime;
    runtime.host_api = {mock_device_malloc, mock_device_free, mock_copy_to_device, mock_copy_from_device};
    runtime.pto_init();
    runtime.pto_init_rings();

    // Use odd sizes that don't align to PTO_ALIGNMENT
    int32_t SIZE_B = 33;
    int32_t SIZE_C = 17;
    int32_t SIZE_D = 100;

    void* dev_a_ptr = mock_device_malloc(64);
    PTOBufferHandle dev_a = make_external_handle(dev_a_ptr, 64);
    PTOBufferHandle dev_b = make_output_handle(SIZE_B);
    PTOBufferHandle dev_c = make_output_handle(SIZE_C);
    PTOBufferHandle dev_d = make_output_handle(SIZE_D);

    runtime.pto_scope_begin();

    PTOParam params0[] = {
        make_input_param(&dev_a, 64),
        make_output_param(&dev_b, SIZE_B),
        make_output_param(&dev_c, SIZE_C),
        make_output_param(&dev_d, SIZE_D),
        make_scalar_param(64),
    };
    int t0 = runtime.pto_submit_task(0, PTOWorkerType::VECTOR, params0, 5);

    // Verify addresses are aligned
    CHECK((dev_b.addr % PTO_ALIGNMENT) == 0, "dev_b address aligned");
    CHECK((dev_c.addr % PTO_ALIGNMENT) == 0, "dev_c address aligned");
    CHECK((dev_d.addr % PTO_ALIGNMENT) == 0, "dev_d address aligned");

    // Verify sizes are aligned in packed_buffer_size
    int32_t aligned_b = ALIGN_UP(SIZE_B, PTO_ALIGNMENT);
    int32_t aligned_c = ALIGN_UP(SIZE_C, PTO_ALIGNMENT);
    int32_t aligned_d = ALIGN_UP(SIZE_D, PTO_ALIGNMENT);
    int32_t expected_total = aligned_b + aligned_c + aligned_d;

    Task* task0 = runtime.get_task(t0);
    CHECK(task0->packed_buffer_size == expected_total,
        ("packed_buffer_size = " + std::to_string(expected_total) + " (aligned sum)").c_str());

    // Verify spacing between addresses
    CHECK(dev_c.addr - dev_b.addr == (uint64_t)aligned_b, "dev_c follows dev_b with aligned spacing");
    CHECK(dev_d.addr - dev_c.addr == (uint64_t)aligned_c, "dev_d follows dev_c with aligned spacing");

    runtime.pto_scope_end();
}

// ============================================================================
// Test 9: Phase 1+2 compatibility - state machine still works with ring buffers
// ============================================================================

static void test_phase1_phase2_compatibility() {
    printf("\n=== Test 9: Phase 1+2 Compatibility ===\n");

    Runtime runtime;
    runtime.host_api = {mock_device_malloc, mock_device_free, mock_copy_to_device, mock_copy_from_device};
    runtime.pto_init();
    runtime.pto_init_rings();

    int32_t BYTES = 64;

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

    // Phase 1: State machine
    CHECK(task0->state == TaskState::READY, "T0 starts READY (no deps)");
    CHECK(task1->state == TaskState::PENDING, "T1 starts PENDING (depends on T0)");
    CHECK(task0->fanout_count == 1, "T0 has 1 consumer");
    CHECK(task1->fanin.load() == 1, "T1 has 1 producer");

    // Execute T0
    task0->state = TaskState::RUNNING;
    task0->state = TaskState::COMPLETED;
    if (task1->fanin.fetch_sub(1) == 1) {
        task1->state = TaskState::READY;
    }
    task0->fanout_refcount++;
    runtime.check_consumed(t0);

    CHECK(task1->state == TaskState::READY, "T1 → READY after T0 completes");
    CHECK(task0->state == TaskState::COMPLETED, "T0 still COMPLETED (scope holds ref)");

    // Execute T1
    task1->state = TaskState::RUNNING;
    task1->state = TaskState::COMPLETED;

    // Phase 2: Scope end triggers CONSUMED
    runtime.pto_scope_end();

    CHECK(task0->state == TaskState::CONSUMED, "T0 → CONSUMED after scope_end");
    CHECK(task1->state == TaskState::CONSUMED, "T1 → CONSUMED after scope_end");

    // Phase 3: Packed buffer info preserved
    CHECK(task0->packed_buffer_size > 0, "T0 has packed buffer (Phase 3)");
    CHECK(task1->packed_buffer_size > 0, "T1 has packed buffer (Phase 3)");

    runtime.print_runtime();
}

// ============================================================================
// Main
// ============================================================================

int main() {
    printf("============================================================\n");
    printf("Phase 3 Test: TaskRing and HeapRing Integration\n");
    printf("============================================================\n");

    test_ring_init();
    test_single_task_packed_output();
    test_multiple_tasks_heap();
    test_task_no_outputs();
    test_preallocated_output();
    test_mixed_allocation();
    test_legacy_mode();
    test_alignment();
    test_phase1_phase2_compatibility();

    printf("\n============================================================\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("============================================================\n");

    cleanup_allocations();

    return tests_failed > 0 ? 1 : 0;
}
