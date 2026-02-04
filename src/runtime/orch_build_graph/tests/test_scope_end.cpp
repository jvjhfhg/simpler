/**
 * Phase 2 Test: Scope End Logic — RAII Scope Semantics
 *
 * Tests Gap #4 (Scope Management):
 * - Tasks must be submitted inside a scope (enforced)
 * - CONSUMED target = fanout_count + 1 (real consumers + scope ref)
 * - scope_end() only increments fanout_refcount (releases scope's consumer ref)
 * - fanout_count = array length (real consumers only)
 * - CONSUMED transition: fanout_refcount == fanout_count + 1
 * - Nested scopes: each task belongs to its parent scope only (RAII)
 *
 * Compile:
 *   g++ -std=c++17 -I../runtime -o test_phase2 test_phase2_scope_end.cpp ../runtime/runtime.cpp
 */

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "dep_list_pool.h"
#include "runtime.h"

// ============================================================================
// Mock host API
// ============================================================================

static void* mock_device_malloc(size_t size) { return malloc(size); }
static void mock_device_free(void* ptr) { free(ptr); }
static int mock_copy_to_device(void* dev, const void* host, size_t size) {
    memcpy(dev, host, size);
    return 0;
}
static int mock_copy_from_device(void* host, const void* dev, size_t size) {
    memcpy(host, dev, size);
    return 0;
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
    t.ndims = 1;
    t.overlap_type = OverlapType::Fuzzy;
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
// Test 1: Basic scope_end increments fanout_refcount only
// ============================================================================

static void test_scope_end_increments() {
    printf("\n=== Test 1: scope_end Increments fanout_refcount Only ===\n");

    Runtime runtime;
    runtime.host_api = {mock_device_malloc, mock_device_free, mock_copy_to_device, mock_copy_from_device};
    runtime.pto_init();

    int32_t BYTES = 64;

    void* dev_a_ptr = mock_device_malloc(BYTES);
    PTOBufferHandle dev_a = make_external_handle(dev_a_ptr, BYTES);
    PTOBufferHandle dev_b = make_output_handle(BYTES);
    PTOBufferHandle dev_c = make_output_handle(BYTES);

    runtime.pto_scope_begin();

    // T0: no consumers yet
    PTOParam params0[] = {
        make_input_param(&dev_a, BYTES),
        make_output_param(&dev_b, BYTES),
        make_scalar_param(64),
    };
    int t0 = runtime.pto_submit_task(0, PTOWorkerType::VECTOR, params0, 3);

    // T1: consumes T0's output → T0.fanout_count = 1
    PTOParam params1[] = {
        make_input_param(&dev_b, BYTES),
        make_output_param(&dev_c, BYTES),
        make_scalar_param(64),
    };
    int t1 = runtime.pto_submit_task(1, PTOWorkerType::VECTOR, params1, 3);

    Task* task0 = runtime.get_task(t0);
    Task* task1 = runtime.get_task(t1);

    // Before scope_end: fanout_count = array length, target = fanout_count + 1
    CHECK(task0->fanout_count == 1, "T0 fanout_count = 1 (1 real consumer)");
    CHECK(task1->fanout_count == 0, "T1 fanout_count = 0 (0 real consumers)");
    CHECK(task0->fanout_refcount == 0, "T0 fanout_refcount = 0 (before scope_end)");
    CHECK(task1->fanout_refcount == 0, "T1 fanout_refcount = 0 (before scope_end)");

    runtime.pto_scope_end();

    // After scope_end: fanout_count unchanged, fanout_refcount incremented by 1
    CHECK(task0->fanout_count == 1, "T0 fanout_count = 1 (unchanged)");
    CHECK(task1->fanout_count == 0, "T1 fanout_count = 0 (unchanged)");
    CHECK(task0->fanout_refcount == 1, "T0 fanout_refcount = 1 (scope_end)");
    CHECK(task1->fanout_refcount == 1, "T1 fanout_refcount = 1 (scope_end)");

    // T1: fanout_refcount(1) == fanout_target(1), but not COMPLETED yet
    CHECK(task1->state == TaskState::PENDING, "T1 still PENDING (not executed)");

    runtime.print_runtime();
    free(dev_a_ptr);
}

// ============================================================================
// Test 2: Diamond DAG — full lifecycle to CONSUMED
// ============================================================================

static void test_diamond_scope_consumed() {
    printf("\n=== Test 2: Diamond DAG — Full Lifecycle with Scope ===\n");

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

    // T0: c = a + b
    PTOParam params0[] = {
        make_input_param(&dev_a, BYTES),
        make_input_param(&dev_b, BYTES),
        make_output_param(&dev_c, BYTES),
        make_scalar_param(64),
    };
    int t0 = runtime.pto_submit_task(0, PTOWorkerType::VECTOR, params0, 4);

    // T1: d = c + 1
    PTOParam params1[] = {
        make_input_param(&dev_c, BYTES),
        make_scalar_param(1),
        make_output_param(&dev_d, BYTES),
        make_scalar_param(64),
    };
    int t1 = runtime.pto_submit_task(1, PTOWorkerType::VECTOR, params1, 4);

    // T2: e = c + 2
    PTOParam params2[] = {
        make_input_param(&dev_c, BYTES),
        make_scalar_param(2),
        make_output_param(&dev_e, BYTES),
        make_scalar_param(64),
    };
    int t2 = runtime.pto_submit_task(1, PTOWorkerType::VECTOR, params2, 4);

    // T3: f = d * e
    PTOParam params3[] = {
        make_input_param(&dev_d, BYTES),
        make_input_param(&dev_e, BYTES),
        make_output_param(&dev_f, BYTES),
        make_scalar_param(64),
    };
    int t3 = runtime.pto_submit_task(2, PTOWorkerType::VECTOR, params3, 4);

    Task* task0 = runtime.get_task(t0);
    Task* task1 = runtime.get_task(t1);
    Task* task2 = runtime.get_task(t2);
    Task* task3 = runtime.get_task(t3);
    DepListPool* pool = runtime.get_dep_list_pool();

    // Before scope_end: fanout_count = real consumers only, target = fanout_count + 1
    CHECK(task0->fanout_count == 2, "T0 fanout_count = 2 (T1, T2)");
    CHECK(task1->fanout_count == 1, "T1 fanout_count = 1 (T3)");
    CHECK(task2->fanout_count == 1, "T2 fanout_count = 1 (T3)");
    CHECK(task3->fanout_count == 0, "T3 fanout_count = 0 (leaf)");

    // --- Simulate execution: T0 → T1, T2 → T3 ---
    printf("\n--- Execute T0 ---\n");
    task0->state = TaskState::RUNNING;
    task0->state = TaskState::COMPLETED;
    dep_list_foreach(
        pool,
        task0->fanout_head,
        [&](int32_t dep_id, void*) {
            Task* dep = runtime.get_task(dep_id);
            if (dep->fanin.fetch_sub(1, std::memory_order_acq_rel) == 1) dep->state = TaskState::READY;
        },
        nullptr);
    CHECK(task1->state == TaskState::READY, "T1 → READY");
    CHECK(task2->state == TaskState::READY, "T2 → READY");

    printf("\n--- Execute T1 ---\n");
    task1->state = TaskState::RUNNING;
    task1->state = TaskState::COMPLETED;
    dep_list_foreach(
        pool,
        task1->fanout_head,
        [&](int32_t dep_id, void*) {
            Task* dep = runtime.get_task(dep_id);
            if (dep->fanin.fetch_sub(1, std::memory_order_acq_rel) == 1) dep->state = TaskState::READY;
        },
        nullptr);
    dep_list_foreach(
        pool,
        task1->fanin_head,
        [&](int32_t producer_id, void*) {
            runtime.get_task(producer_id)->fanout_refcount++;
            runtime.check_consumed(producer_id);
        },
        nullptr);
    CHECK(task0->fanout_refcount == 1, "T0 refcount = 1 (T1 done)");
    CHECK(task0->state == TaskState::COMPLETED, "T0 still COMPLETED (1 < fanout_count+1=3, scope holds ref)");

    printf("\n--- Execute T2 ---\n");
    task2->state = TaskState::RUNNING;
    task2->state = TaskState::COMPLETED;
    dep_list_foreach(
        pool,
        task2->fanout_head,
        [&](int32_t dep_id, void*) {
            Task* dep = runtime.get_task(dep_id);
            if (dep->fanin.fetch_sub(1, std::memory_order_acq_rel) == 1) dep->state = TaskState::READY;
        },
        nullptr);
    CHECK(task3->state == TaskState::READY, "T3 → READY");
    dep_list_foreach(
        pool,
        task2->fanin_head,
        [&](int32_t producer_id, void*) {
            runtime.get_task(producer_id)->fanout_refcount++;
            runtime.check_consumed(producer_id);
        },
        nullptr);
    CHECK(task0->fanout_refcount == 2, "T0 refcount = 2 (T1+T2 done)");
    CHECK(task0->state == TaskState::COMPLETED, "T0 still COMPLETED (2 < fanout_count+1=3, scope holds ref)");

    printf("\n--- Execute T3 ---\n");
    task3->state = TaskState::RUNNING;
    task3->state = TaskState::COMPLETED;
    dep_list_foreach(
        pool,
        task3->fanin_head,
        [&](int32_t producer_id, void*) {
            runtime.get_task(producer_id)->fanout_refcount++;
            runtime.check_consumed(producer_id);
        },
        nullptr);
    CHECK(task1->fanout_refcount == 1, "T1 refcount = 1 (T3 done)");
    CHECK(task1->state == TaskState::COMPLETED, "T1 still COMPLETED (1 < fanout_count+1=2, scope holds ref)");
    CHECK(task2->fanout_refcount == 1, "T2 refcount = 1 (T3 done)");
    CHECK(task2->state == TaskState::COMPLETED, "T2 still COMPLETED (1 < fanout_count+1=2, scope holds ref)");

    // T3: refcount=0, target=fanout_count+1=1 → not CONSUMED (scope holds ref)
    runtime.check_consumed(t3);
    CHECK(task3->state == TaskState::COMPLETED, "T3 still COMPLETED (0 < fanout_count+1=1, scope holds ref)");

    // scope_end: releases scope ref → all tasks become CONSUMED
    printf("\n--- scope_end() ---\n");
    runtime.pto_scope_end();

    CHECK(task0->fanout_refcount == 3, "T0 refcount = 3 (T1+T2+scope)");
    CHECK(task0->state == TaskState::CONSUMED, "T0 → CONSUMED (3 == fanout_count+1=3)");
    CHECK(task1->fanout_refcount == 2, "T1 refcount = 2 (T3+scope)");
    CHECK(task1->state == TaskState::CONSUMED, "T1 → CONSUMED (2 == fanout_count+1=2)");
    CHECK(task2->fanout_refcount == 2, "T2 refcount = 2 (T3+scope)");
    CHECK(task2->state == TaskState::CONSUMED, "T2 → CONSUMED (2 == fanout_count+1=2)");
    CHECK(task3->fanout_refcount == 1, "T3 refcount = 1 (scope)");
    CHECK(task3->state == TaskState::CONSUMED, "T3 → CONSUMED (1 == fanout_count+1=1)");

    int consumed = 0;
    for (int i = 0; i < runtime.get_task_count(); i++) {
        if (runtime.get_task(i)->state == TaskState::CONSUMED) consumed++;
    }
    CHECK(consumed == 4, "All 4 tasks CONSUMED");

    runtime.print_runtime();
    free(dev_a_ptr);
    free(dev_b_ptr);
}

// ============================================================================
// Test 3: Nested scopes (RAII: each task belongs to parent scope only)
// ============================================================================

static void test_nested_scopes() {
    printf("\n=== Test 3: Nested Scopes (RAII) ===\n");

    Runtime runtime;
    runtime.host_api = {mock_device_malloc, mock_device_free, mock_copy_to_device, mock_copy_from_device};
    runtime.pto_init();

    int32_t BYTES = 64;

    void* dev_a_ptr = mock_device_malloc(BYTES);
    PTOBufferHandle dev_a = make_external_handle(dev_a_ptr, BYTES);
    PTOBufferHandle dev_b = make_output_handle(BYTES);
    PTOBufferHandle dev_c = make_output_handle(BYTES);
    PTOBufferHandle dev_d = make_output_handle(BYTES);

    // Outer scope
    runtime.pto_scope_begin();

    // T0 in outer scope only
    PTOParam params0[] = {
        make_input_param(&dev_a, BYTES),
        make_output_param(&dev_b, BYTES),
        make_scalar_param(64),
    };
    int t0 = runtime.pto_submit_task(0, PTOWorkerType::VECTOR, params0, 3);

    // Inner scope
    runtime.pto_scope_begin();

    // T1 in inner scope (NOT outer scope — RAII)
    PTOParam params1[] = {
        make_input_param(&dev_b, BYTES),
        make_output_param(&dev_c, BYTES),
        make_scalar_param(64),
    };
    int t1 = runtime.pto_submit_task(1, PTOWorkerType::VECTOR, params1, 3);

    // T2 in inner scope (NOT outer scope — RAII)
    PTOParam params2[] = {
        make_input_param(&dev_c, BYTES),
        make_output_param(&dev_d, BYTES),
        make_scalar_param(64),
    };
    int t2 = runtime.pto_submit_task(1, PTOWorkerType::VECTOR, params2, 3);

    Task* task0 = runtime.get_task(t0);
    Task* task1 = runtime.get_task(t1);
    Task* task2 = runtime.get_task(t2);

    // Before any scope_end: fanout_count = array length, target = fanout_count + 1
    CHECK(task0->fanout_count == 1, "T0 fanout_count = 1 (T1)");
    CHECK(task1->fanout_count == 1, "T1 fanout_count = 1 (T2)");
    CHECK(task2->fanout_count == 0, "T2 fanout_count = 0");

    // Execute all tasks
    task0->state = TaskState::RUNNING;
    task0->state = TaskState::COMPLETED;
    task1->fanin.fetch_sub(1);
    task1->state = TaskState::READY;

    task1->state = TaskState::RUNNING;
    task1->state = TaskState::COMPLETED;
    task0->fanout_refcount++;
    runtime.check_consumed(t0);
    task2->fanin.fetch_sub(1);
    task2->state = TaskState::READY;

    task2->state = TaskState::RUNNING;
    task2->state = TaskState::COMPLETED;
    task1->fanout_refcount++;
    runtime.check_consumed(t1);

    // Before scope_end: all completed but not consumed (scope holds refs)
    CHECK(task0->fanout_refcount == 1, "T0 refcount = 1 (T1 done)");
    CHECK(task0->state == TaskState::COMPLETED, "T0 still COMPLETED (1 < fanout_count+1=2, outer scope holds ref)");
    CHECK(task1->fanout_refcount == 1, "T1 refcount = 1 (T2 done)");
    CHECK(task1->state == TaskState::COMPLETED, "T1 still COMPLETED (1 < fanout_count+1=2, inner scope holds ref)");
    CHECK(task2->fanout_refcount == 0, "T2 refcount = 0");
    CHECK(task2->state == TaskState::COMPLETED, "T2 still COMPLETED (0 < fanout_count+1=1, inner scope holds ref)");

    // Inner scope_end: affects T1, T2 only (RAII)
    printf("\n--- inner scope_end() ---\n");
    runtime.pto_scope_end();

    CHECK(task0->fanout_count == 1, "T0 fanout_count unchanged");
    CHECK(task0->fanout_refcount == 1, "T0 refcount unchanged (not in inner scope)");
    CHECK(task0->state == TaskState::COMPLETED, "T0 still COMPLETED (inner scope doesn't touch it)");
    CHECK(task1->fanout_count == 1, "T1 fanout_count unchanged");
    CHECK(task1->fanout_refcount == 2, "T1 refcount = 2 (T2 + inner scope)");
    CHECK(task1->state == TaskState::CONSUMED, "T1 → CONSUMED (2 == fanout_count+1=2)");
    CHECK(task2->fanout_count == 0, "T2 fanout_count unchanged");
    CHECK(task2->fanout_refcount == 1, "T2 refcount = 1 (inner scope)");
    CHECK(task2->state == TaskState::CONSUMED, "T2 → CONSUMED (1 == fanout_count+1=1)");

    // Outer scope_end: affects T0 only (RAII — T1, T2 belong to inner scope)
    printf("\n--- outer scope_end() ---\n");
    runtime.pto_scope_end();

    CHECK(task0->fanout_count == 1, "T0 fanout_count unchanged");
    CHECK(task0->fanout_refcount == 2, "T0 refcount = 2 (T1 + outer scope)");
    CHECK(task0->state == TaskState::CONSUMED, "T0 → CONSUMED (2 == fanout_count+1=2)");
    // T1, T2 refcounts unchanged — outer scope doesn't touch them
    CHECK(task1->fanout_refcount == 2, "T1 refcount unchanged");
    CHECK(task2->fanout_refcount == 1, "T2 refcount unchanged");

    runtime.print_runtime();
    free(dev_a_ptr);
}

// ============================================================================
// Test 4: Without scope_end — tasks stay COMPLETED (scope holds ref)
// ============================================================================

static void test_no_scope_end() {
    printf("\n=== Test 4: Without scope_end — Tasks Stay COMPLETED ===\n");

    Runtime runtime;
    runtime.host_api = {mock_device_malloc, mock_device_free, mock_copy_to_device, mock_copy_from_device};
    runtime.pto_init();

    int32_t BYTES = 64;

    void* dev_a_ptr = mock_device_malloc(BYTES);
    PTOBufferHandle dev_a = make_external_handle(dev_a_ptr, BYTES);
    PTOBufferHandle dev_b = make_output_handle(BYTES);

    runtime.pto_scope_begin();

    PTOParam params0[] = {
        make_input_param(&dev_a, BYTES),
        make_output_param(&dev_b, BYTES),
        make_scalar_param(64),
    };
    int t0 = runtime.pto_submit_task(0, PTOWorkerType::VECTOR, params0, 3);

    // DO NOT call scope_end

    Task* task0 = runtime.get_task(t0);

    // Execute T0
    task0->state = TaskState::RUNNING;
    task0->state = TaskState::COMPLETED;
    runtime.check_consumed(t0);

    // T0: fanout_count=0, but target=fanout_count+1=1 (scope holds ref)
    // refcount=0 < target=1 → NOT CONSUMED
    CHECK(task0->fanout_count == 0, "T0 fanout_count = 0 (no real consumers)");
    CHECK(task0->fanout_refcount == 0, "T0 refcount = 0");
    CHECK(task0->state == TaskState::COMPLETED, "T0 stays COMPLETED (scope holds ref, no scope_end)");

    runtime.print_runtime();
    free(dev_a_ptr);
}

// ============================================================================
// Test 5: Wide fanout — tasks stay COMPLETED until scope_end
// ============================================================================

static void test_wide_fanout() {
    printf("\n=== Test 5: Wide Fanout ===\n");

    Runtime runtime;
    runtime.host_api = {mock_device_malloc, mock_device_free, mock_copy_to_device, mock_copy_from_device};
    runtime.pto_init();

    int32_t BYTES = 64;

    void* dev_a_ptr = mock_device_malloc(BYTES);
    PTOBufferHandle dev_a = make_external_handle(dev_a_ptr, BYTES);

    const int N = 8;
    PTOBufferHandle out_handles[N];
    for (int i = 0; i < N; i++) {
        out_handles[i] = make_output_handle(BYTES);
    }

    runtime.pto_scope_begin();

    int task_ids[N];
    for (int i = 0; i < N; i++) {
        PTOParam params[] = {
            make_input_param(&dev_a, BYTES),
            make_output_param(&out_handles[i], BYTES),
            make_scalar_param(64),
        };
        task_ids[i] = runtime.pto_submit_task(i, PTOWorkerType::VECTOR, params, 3);
    }

    // Execute all tasks
    for (int i = 0; i < N; i++) {
        Task* t = runtime.get_task(task_ids[i]);
        t->state = TaskState::RUNNING;
        t->state = TaskState::COMPLETED;
        runtime.check_consumed(task_ids[i]);
    }

    // Before scope_end: all COMPLETED, not CONSUMED (scope holds refs)
    int consumed_before = 0;
    for (int i = 0; i < N; i++) {
        Task* t = runtime.get_task(task_ids[i]);
        CHECK(t->fanout_count == 0,
            (std::string("T") + std::to_string(i) + " fanout_count = 0 (no real consumers)").c_str());
        CHECK(t->state == TaskState::COMPLETED,
            (std::string("T") + std::to_string(i) + " still COMPLETED (scope holds ref)").c_str());
        if (t->state == TaskState::CONSUMED) consumed_before++;
    }
    CHECK(consumed_before == 0, "0 tasks CONSUMED before scope_end (scope holds refs)");

    runtime.pto_scope_end();

    // After scope_end: fanout_refcount=1, all CONSUMED
    int consumed_after = 0;
    for (int i = 0; i < N; i++) {
        Task* t = runtime.get_task(task_ids[i]);
        CHECK(t->fanout_count == 0, (std::string("T") + std::to_string(i) + " fanout_count = 0 (unchanged)").c_str());
        CHECK(t->fanout_refcount == 1, (std::string("T") + std::to_string(i) + " fanout_refcount = 1 (scope)").c_str());
        CHECK(t->state == TaskState::CONSUMED, (std::string("T") + std::to_string(i) + " → CONSUMED").c_str());
        if (t->state == TaskState::CONSUMED) consumed_after++;
    }
    CHECK(consumed_after == N, "All 8 tasks CONSUMED after scope_end");

    runtime.print_runtime();
    free(dev_a_ptr);
}

// ============================================================================
// Main
// ============================================================================

int main() {
    printf("============================================================\n");
    printf("Phase 2 Test: Scope End Logic with Fanout Increment\n");
    printf("============================================================\n");

    test_scope_end_increments();
    test_diamond_scope_consumed();
    test_nested_scopes();
    test_no_scope_end();
    test_wide_fanout();

    printf("\n============================================================\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("============================================================\n");

    return tests_failed > 0 ? 1 : 0;
}
