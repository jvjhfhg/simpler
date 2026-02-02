/**
 * Phase 1 Test: Task State Machine and Fanout Reference Counting
 *
 * Tests Gap #3 (TaskState) and Gap #5 (fanout_refcount):
 * - Task state initialization (PENDING vs READY)
 * - State transitions: PENDING → READY → RUNNING → COMPLETED → CONSUMED
 * - fanout_refcount increments and CONSUMED transition via check_consumed()
 * - fanin_producers reverse dependency tracking
 *
 * Test topology: Diamond DAG
 *       T0 (a+b=c)
 *      /  \
 *    T1    T2      (c+1=d, c+2=e)
 *      \  /
 *       T3 (d*e=f)
 *
 * Compile:
 *   g++ -std=c++17 -I../runtime -o test_phase1 test_phase1_state_machine.cpp ../runtime/runtime.cpp
 */

#include "runtime.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// ============================================================================
// Mock host API (simple malloc-based)
// ============================================================================

static void* mock_device_malloc(size_t size) {
    return malloc(size);
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

// ============================================================================
// Helpers (same as pto_example_orch.cpp)
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
// Test 1: Orchestration-side state initialization
// ============================================================================

static void test_state_initialization() {
    printf("\n=== Test 1: Orchestration-Side State Initialization ===\n");

    Runtime runtime;
    runtime.host_api = {mock_device_malloc, mock_device_free, mock_copy_to_device, mock_copy_from_device};
    runtime.pto_init();

    int32_t BYTES = 64;

    // External inputs (pre-allocated)
    void* dev_a_ptr = mock_device_malloc(BYTES);
    void* dev_b_ptr = mock_device_malloc(BYTES);
    PTOBufferHandle dev_a = make_external_handle(dev_a_ptr, BYTES);
    PTOBufferHandle dev_b = make_external_handle(dev_b_ptr, BYTES);

    // Outputs (allocated by runtime)
    PTOBufferHandle dev_c = make_output_handle(BYTES);
    PTOBufferHandle dev_d = make_output_handle(BYTES);
    PTOBufferHandle dev_e = make_output_handle(BYTES);
    PTOBufferHandle dev_f = make_output_handle(BYTES);

    runtime.pto_scope_begin();

    // T0: c = a + b (no deps → READY)
    PTOParam params0[] = {
        make_input_param(&dev_a, BYTES),
        make_input_param(&dev_b, BYTES),
        make_output_param(&dev_c, BYTES),
        make_scalar_param(64),
    };
    int t0 = runtime.pto_submit_task(0, PTOWorkerType::VECTOR, params0, 4);

    // T1: d = c + 1 (depends on T0 → PENDING)
    PTOParam params1[] = {
        make_input_param(&dev_c, BYTES),
        make_scalar_param(1),
        make_output_param(&dev_d, BYTES),
        make_scalar_param(64),
    };
    int t1 = runtime.pto_submit_task(1, PTOWorkerType::VECTOR, params1, 4);

    // T2: e = c + 2 (depends on T0 → PENDING)
    PTOParam params2[] = {
        make_input_param(&dev_c, BYTES),
        make_scalar_param(2),
        make_output_param(&dev_e, BYTES),
        make_scalar_param(64),
    };
    int t2 = runtime.pto_submit_task(1, PTOWorkerType::VECTOR, params2, 4);

    // T3: f = d * e (depends on T1 and T2 → PENDING)
    PTOParam params3[] = {
        make_input_param(&dev_d, BYTES),
        make_input_param(&dev_e, BYTES),
        make_output_param(&dev_f, BYTES),
        make_scalar_param(64),
    };
    int t3 = runtime.pto_submit_task(2, PTOWorkerType::VECTOR, params3, 4);

    runtime.pto_scope_end();

    // Verify task creation
    CHECK(t0 == 0, "T0 created with id 0");
    CHECK(t1 == 1, "T1 created with id 1");
    CHECK(t2 == 2, "T2 created with id 2");
    CHECK(t3 == 3, "T3 created with id 3");
    CHECK(runtime.get_task_count() == 4, "4 tasks total");

    // Verify state: T0 has no deps → READY; T1,T2,T3 have deps → PENDING
    Task* task0 = runtime.get_task(t0);
    Task* task1 = runtime.get_task(t1);
    Task* task2 = runtime.get_task(t2);
    Task* task3 = runtime.get_task(t3);

    CHECK(task0->state == TaskState::READY, "T0 state = READY (no dependencies)");
    CHECK(task1->state == TaskState::PENDING, "T1 state = PENDING (depends on T0)");
    CHECK(task2->state == TaskState::PENDING, "T2 state = PENDING (depends on T0)");
    CHECK(task3->state == TaskState::PENDING, "T3 state = PENDING (depends on T1, T2)");

    // Verify fanin counts
    CHECK(task0->fanin.load() == 0, "T0 fanin = 0");
    CHECK(task1->fanin.load() == 1, "T1 fanin = 1 (from T0)");
    CHECK(task2->fanin.load() == 1, "T2 fanin = 1 (from T0)");
    CHECK(task3->fanin.load() == 2, "T3 fanin = 2 (from T1, T2)");

    // Verify fanout_count (array length: real consumers only)
    // CONSUMED target = fanout_count + 1 (scope ref)
    CHECK(task0->fanout_count == 2, "T0 fanout_count = 2 (T1, T2)");
    CHECK(task1->fanout_count == 1, "T1 fanout_count = 1 (T3)");
    CHECK(task2->fanout_count == 1, "T2 fanout_count = 1 (T3)");
    CHECK(task3->fanout_count == 0, "T3 fanout_count = 0 (leaf)");

    // Verify fanout_refcount (scope_end already called → all have 1)
    CHECK(task0->fanout_refcount == 1, "T0 fanout_refcount = 1 (scope)");
    CHECK(task1->fanout_refcount == 1, "T1 fanout_refcount = 1 (scope)");
    CHECK(task2->fanout_refcount == 1, "T2 fanout_refcount = 1 (scope)");
    CHECK(task3->fanout_refcount == 1, "T3 fanout_refcount = 1 (scope)");

    // Verify fanin_producers (reverse dependency tracking)
    CHECK(task0->fanin_producer_count == 0, "T0 has 0 producers");
    CHECK(task1->fanin_producer_count == 1, "T1 has 1 producer");
    CHECK(task1->fanin_producers[0] == t0, "T1 producer[0] = T0");
    CHECK(task2->fanin_producer_count == 1, "T2 has 1 producer");
    CHECK(task2->fanin_producers[0] == t0, "T2 producer[0] = T0");
    CHECK(task3->fanin_producer_count == 2, "T3 has 2 producers");
    // T3 producers are T1 and T2 (order depends on TensorMap lookup order)
    bool has_t1 = (task3->fanin_producers[0] == t1 || task3->fanin_producers[1] == t1);
    bool has_t2 = (task3->fanin_producers[0] == t2 || task3->fanin_producers[1] == t2);
    CHECK(has_t1 && has_t2, "T3 producers include T1 and T2");

    // Print final state
    runtime.print_runtime();

    free(dev_a_ptr);
    free(dev_b_ptr);
}

// ============================================================================
// Test 2: Manual scheduler-side state transitions
// ============================================================================

static void test_state_transitions() {
    printf("\n=== Test 2: Manual State Transitions (Simulated Scheduler) ===\n");

    Runtime runtime;
    runtime.host_api = {mock_device_malloc, mock_device_free, mock_copy_to_device, mock_copy_from_device};
    runtime.pto_init();

    int32_t BYTES = 64;

    void* dev_a_ptr = mock_device_malloc(BYTES);
    void* dev_b_ptr = mock_device_malloc(BYTES);
    PTOBufferHandle dev_a = make_external_handle(dev_a_ptr, BYTES);
    PTOBufferHandle dev_b = make_external_handle(dev_b_ptr, BYTES);

    PTOBufferHandle dev_c = make_output_handle(BYTES);
    PTOBufferHandle dev_d = make_output_handle(BYTES);
    PTOBufferHandle dev_e = make_output_handle(BYTES);
    PTOBufferHandle dev_f = make_output_handle(BYTES);

    runtime.pto_scope_begin();

    PTOParam params0[] = {
        make_input_param(&dev_a, BYTES),
        make_input_param(&dev_b, BYTES),
        make_output_param(&dev_c, BYTES),
        make_scalar_param(64),
    };
    int t0 = runtime.pto_submit_task(0, PTOWorkerType::VECTOR, params0, 4);

    PTOParam params1[] = {
        make_input_param(&dev_c, BYTES),
        make_scalar_param(1),
        make_output_param(&dev_d, BYTES),
        make_scalar_param(64),
    };
    int t1 = runtime.pto_submit_task(1, PTOWorkerType::VECTOR, params1, 4);

    PTOParam params2[] = {
        make_input_param(&dev_c, BYTES),
        make_scalar_param(2),
        make_output_param(&dev_e, BYTES),
        make_scalar_param(64),
    };
    int t2 = runtime.pto_submit_task(1, PTOWorkerType::VECTOR, params2, 4);

    PTOParam params3[] = {
        make_input_param(&dev_d, BYTES),
        make_input_param(&dev_e, BYTES),
        make_output_param(&dev_f, BYTES),
        make_scalar_param(64),
    };
    int t3 = runtime.pto_submit_task(2, PTOWorkerType::VECTOR, params3, 4);

    runtime.pto_scope_end();

    Task* task0 = runtime.get_task(t0);
    Task* task1 = runtime.get_task(t1);
    Task* task2 = runtime.get_task(t2);
    Task* task3 = runtime.get_task(t3);

    // --- Simulate: Dispatch T0 (READY → RUNNING) ---
    printf("\n--- Dispatch T0 ---\n");
    CHECK(task0->state == TaskState::READY, "T0 starts READY");
    task0->state = TaskState::RUNNING;
    CHECK(task0->state == TaskState::RUNNING, "T0 → RUNNING");

    // --- Simulate: T0 completes (RUNNING → COMPLETED) ---
    printf("\n--- Complete T0 ---\n");
    task0->state = TaskState::COMPLETED;
    CHECK(task0->state == TaskState::COMPLETED, "T0 → COMPLETED");

    // Decrement fanin of T1 and T2 (like aicpu_executor does)
    for (int j = 0; j < task0->fanout_count; j++) {
        int dep_id = task0->fanout[j];
        Task* dep = runtime.get_task(dep_id);
        int prev = dep->fanin.fetch_sub(1, std::memory_order_acq_rel);
        if (prev == 1) {
            dep->state = TaskState::READY;
        }
    }
    CHECK(task1->state == TaskState::READY, "T1 → READY (T0 completed)");
    CHECK(task2->state == TaskState::READY, "T2 → READY (T0 completed)");
    CHECK(task3->state == TaskState::PENDING, "T3 still PENDING (needs T1 and T2)");

    // --- Simulate: Dispatch and complete T1 ---
    printf("\n--- Dispatch+Complete T1 ---\n");
    task1->state = TaskState::RUNNING;
    task1->state = TaskState::COMPLETED;
    CHECK(task1->state == TaskState::COMPLETED, "T1 → COMPLETED");

    // T1 completion: decrement T3's fanin
    for (int j = 0; j < task1->fanout_count; j++) {
        int dep_id = task1->fanout[j];
        Task* dep = runtime.get_task(dep_id);
        int prev = dep->fanin.fetch_sub(1, std::memory_order_acq_rel);
        if (prev == 1) {
            dep->state = TaskState::READY;
        }
    }
    CHECK(task3->state == TaskState::PENDING, "T3 still PENDING (needs T2)");

    // T1 completion: increment fanout_refcount for T1's producers
    for (int j = 0; j < task1->fanin_producer_count; j++) {
        int producer_id = task1->fanin_producers[j];
        Task* producer = runtime.get_task(producer_id);
        producer->fanout_refcount++;
        runtime.check_consumed(producer_id);
    }
    CHECK(task0->fanout_refcount == 2, "T0 fanout_refcount = 2 (T1 + scope)");
    CHECK(task0->state == TaskState::COMPLETED, "T0 still COMPLETED (fanout_refcount=2 < fanout_count+1=3)");

    // --- Simulate: Dispatch and complete T2 ---
    printf("\n--- Dispatch+Complete T2 ---\n");
    task2->state = TaskState::RUNNING;
    task2->state = TaskState::COMPLETED;

    // T2 completion: decrement T3's fanin
    for (int j = 0; j < task2->fanout_count; j++) {
        int dep_id = task2->fanout[j];
        Task* dep = runtime.get_task(dep_id);
        int prev = dep->fanin.fetch_sub(1, std::memory_order_acq_rel);
        if (prev == 1) {
            dep->state = TaskState::READY;
        }
    }
    CHECK(task3->state == TaskState::READY, "T3 → READY (both T1, T2 completed)");

    // T2 completion: increment fanout_refcount for T2's producers
    for (int j = 0; j < task2->fanin_producer_count; j++) {
        int producer_id = task2->fanin_producers[j];
        Task* producer = runtime.get_task(producer_id);
        producer->fanout_refcount++;
        runtime.check_consumed(producer_id);
    }
    CHECK(task0->fanout_refcount == 3, "T0 fanout_refcount = 3 (T1+T2+scope)");
    CHECK(task0->state == TaskState::CONSUMED, "T0 → CONSUMED (fanout_refcount=3 == fanout_count+1=3)");

    // --- Simulate: Dispatch and complete T3 ---
    printf("\n--- Dispatch+Complete T3 ---\n");
    task3->state = TaskState::RUNNING;
    task3->state = TaskState::COMPLETED;

    // T3 completion: increment fanout_refcount for T3's producers (T1, T2)
    for (int j = 0; j < task3->fanin_producer_count; j++) {
        int producer_id = task3->fanin_producers[j];
        Task* producer = runtime.get_task(producer_id);
        producer->fanout_refcount++;
        runtime.check_consumed(producer_id);
    }
    CHECK(task1->fanout_refcount == 2, "T1 fanout_refcount = 2 (T3+scope)");
    CHECK(task1->state == TaskState::CONSUMED, "T1 → CONSUMED (fanout_refcount=2 == fanout_count+1=2)");
    CHECK(task2->fanout_refcount == 2, "T2 fanout_refcount = 2 (T3+scope)");
    CHECK(task2->state == TaskState::CONSUMED, "T2 → CONSUMED (fanout_refcount=2 == fanout_count+1=2)");

    // T3 has fanout_count=0, so target=1 (scope). fanout_refcount=1 (scope) → CONSUMED
    runtime.check_consumed(t3);
    CHECK(task3->state == TaskState::CONSUMED, "T3 → CONSUMED (fanout_refcount=1 == fanout_count+1=1)");

    printf("\n--- Final State ---\n");
    runtime.print_runtime();

    free(dev_a_ptr);
    free(dev_b_ptr);
}

// ============================================================================
// Test 3: Chain topology (A → B → C) - sequential dependencies
// ============================================================================

static void test_chain_topology() {
    printf("\n=== Test 3: Chain Topology (A → B → C) ===\n");

    Runtime runtime;
    runtime.host_api = {mock_device_malloc, mock_device_free, mock_copy_to_device, mock_copy_from_device};
    runtime.pto_init();

    int32_t BYTES = 64;

    void* dev_in_ptr = mock_device_malloc(BYTES);
    PTOBufferHandle dev_in = make_external_handle(dev_in_ptr, BYTES);
    PTOBufferHandle dev_mid = make_output_handle(BYTES);
    PTOBufferHandle dev_out = make_output_handle(BYTES);

    runtime.pto_scope_begin();

    // A: mid = in + 1
    PTOParam paramsA[] = {
        make_input_param(&dev_in, BYTES),
        make_scalar_param(1),
        make_output_param(&dev_mid, BYTES),
        make_scalar_param(64),
    };
    int tA = runtime.pto_submit_task(1, PTOWorkerType::VECTOR, paramsA, 4);

    // B: out = mid + 1
    PTOParam paramsB[] = {
        make_input_param(&dev_mid, BYTES),
        make_scalar_param(1),
        make_output_param(&dev_out, BYTES),
        make_scalar_param(64),
    };
    int tB = runtime.pto_submit_task(1, PTOWorkerType::VECTOR, paramsB, 4);

    runtime.pto_scope_end();

    Task* taskA = runtime.get_task(tA);
    Task* taskB = runtime.get_task(tB);

    CHECK(taskA->state == TaskState::READY, "A starts READY");
    CHECK(taskB->state == TaskState::PENDING, "B starts PENDING");
    CHECK(taskA->fanout_count == 1, "A fanout_count = 1 (B)");
    CHECK(taskB->fanout_count == 0, "B fanout_count = 0 (leaf)");

    // Simulate: A runs and completes
    taskA->state = TaskState::RUNNING;
    taskA->state = TaskState::COMPLETED;

    // Decrement B's fanin
    taskB->fanin.fetch_sub(1, std::memory_order_acq_rel);
    taskB->state = TaskState::READY;

    // B completes → increments A's fanout_refcount
    taskB->state = TaskState::RUNNING;
    taskB->state = TaskState::COMPLETED;

    for (int j = 0; j < taskB->fanin_producer_count; j++) {
        Task* producer = runtime.get_task(taskB->fanin_producers[j]);
        producer->fanout_refcount++;
        runtime.check_consumed(taskB->fanin_producers[j]);
    }
    CHECK(taskA->state == TaskState::CONSUMED, "A → CONSUMED after B completes");

    // B is leaf: check_consumed directly
    runtime.check_consumed(tB);
    CHECK(taskB->state == TaskState::CONSUMED, "B → CONSUMED (leaf)");

    printf("\n--- Final State ---\n");
    runtime.print_runtime();

    free(dev_in_ptr);
}

// ============================================================================
// Main
// ============================================================================

int main() {
    printf("============================================================\n");
    printf("Phase 1 Test: Task State Machine & Fanout Reference Counting\n");
    printf("============================================================\n");

    test_state_initialization();
    test_state_transitions();
    test_chain_topology();

    printf("\n============================================================\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("============================================================\n");

    return tests_failed > 0 ? 1 : 0;
}
