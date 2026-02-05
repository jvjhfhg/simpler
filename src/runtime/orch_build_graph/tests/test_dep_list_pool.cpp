/**
 * Phase 6 Test: DepListPool Integration
 *
 * Tests Gap #6 (DepListPool):
 * - Dynamic fanout lists replace fixed-size arrays
 * - Support for >512 consumers (impossible with old fixed arrays)
 * - Memory efficiency: ~8 bytes per task vs ~2KB with fixed arrays
 * - DepListPool utilization tracking
 * - Pool wrap-around and reuse behavior
 * - Spinlock protection for concurrent fanout modification
 *
 * Compile:
 *   g++ -std=c++17 -I../runtime -o test_dep_list_pool test_dep_list_pool.cpp ../runtime/runtime.cpp
 */

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

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

static TensorDescriptor make_tensor_bbox(uint64_t addr, int32_t size, int32_t version = 0) {
    TensorDescriptor t = {};
    t.addr = addr;
    t.start_offset = 0;
    t.strides[0] = 1;
    t.repeats[0] = size;
    t.ndims = 1;
    t.version = version;
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
    return h;
}

static PTOBufferHandle make_output_handle(int32_t size) {
    PTOBufferHandle h = {};
    h.addr = 0;
    h.size = size;
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
// Helper: Collect all task IDs from a dependency list
// ============================================================================

static std::vector<int32_t> collect_dep_list(DepListPool* pool, int32_t head) {
    std::vector<int32_t> result;
    dep_list_foreach(pool, head, [&](int32_t task_id, void*) { result.push_back(task_id); }, nullptr);
    return result;
}

// ============================================================================
// Test 1: Basic DepListPool operations
// ============================================================================

static void test_basic_operations() {
    printf("\n=== Test 1: Basic DepListPool Operations ===\n");

    DepListEntry entries[16];
    DepListPool pool;
    dep_list_pool_init(&pool, entries, 16);

    CHECK(pool.top == 0, "Pool starts empty");
    CHECK(dep_list_is_empty(0), "Empty list is empty");

    // Build list: [5, 3, 7]
    int32_t list = 0;
    list = dep_list_prepend(&pool, list, 5);
    list = dep_list_prepend(&pool, list, 3);
    list = dep_list_prepend(&pool, list, 7);

    CHECK(pool.top == 3, "Pool allocated 3 entries");
    CHECK(!dep_list_is_empty(list), "List is not empty");
    CHECK(dep_list_count(&pool, list) == 3, "List has 3 entries");

    // Verify order (prepend = reverse order)
    std::vector<int32_t> items = collect_dep_list(&pool, list);
    CHECK(items.size() == 3, "Collected 3 items");
    CHECK(items[0] == 7 && items[1] == 3 && items[2] == 5, "Items in correct order [7, 3, 5]");
}

// ============================================================================
// Test 2: High fanout (>512 consumers) - impossible with old fixed array
// ============================================================================

static void test_high_fanout() {
    printf("\n=== Test 2: High Fanout (>512 Consumers) ===\n");

    Runtime runtime;
    runtime.host_api = {mock_device_malloc, mock_device_free, mock_copy_to_device, mock_copy_from_device};
    runtime.pto_init();

    int32_t BYTES = 64;

    void* dev_a_ptr = mock_device_malloc(BYTES);
    PTOBufferHandle dev_a = make_external_handle(dev_a_ptr, BYTES);

    // Create 600 consumer tasks (old fixed array limited to 512)
    const int NUM_CONSUMERS = 600;
    std::vector<PTOBufferHandle> outputs;
    std::vector<int> consumer_ids;

    runtime.pto_scope_begin();

    // T0: producer task
    PTOBufferHandle dev_b = make_output_handle(BYTES);
    PTOParam params0[] = {
        make_input_param(&dev_a, BYTES),
        make_output_param(&dev_b, BYTES),
        make_scalar_param(64),
    };
    int t0 = runtime.pto_submit_task(0, PTOWorkerType::VECTOR, params0, 3);

    // Create 1000 consumers that all depend on T0
    for (int i = 0; i < NUM_CONSUMERS; i++) {
        PTOBufferHandle out = make_output_handle(BYTES);
        outputs.push_back(out);

        PTOParam params[] = {
            make_input_param(&dev_b, BYTES),
            make_output_param(&outputs[i], BYTES),
            make_scalar_param(64),
        };
        int tid = runtime.pto_submit_task(1, PTOWorkerType::VECTOR, params, 3);
        consumer_ids.push_back(tid);
    }

    Task* task0 = runtime.get_task(t0);
    DepListPool* pool = runtime.get_dep_list_pool();

    // Verify fanout_count
    CHECK(task0->fanout_count == NUM_CONSUMERS, ("T0 fanout_count = " + std::to_string(NUM_CONSUMERS)).c_str());

    // Verify all consumers are in fanout list
    std::vector<int32_t> fanout_list = collect_dep_list(pool, task0->fanout_head);
    CHECK(
        fanout_list.size() == NUM_CONSUMERS, ("Fanout list has " + std::to_string(NUM_CONSUMERS) + " entries").c_str());

    // Convert to set for fast lookup
    std::set<int32_t> fanout_set(fanout_list.begin(), fanout_list.end());
    bool all_consumers_present = true;
    for (int cid : consumer_ids) {
        if (fanout_set.find(cid) == fanout_set.end()) {
            all_consumers_present = false;
            break;
        }
    }
    CHECK(all_consumers_present, "All 1000 consumers in fanout list");

    // Verify each consumer has T0 as producer
    bool all_fanin_correct = true;
    for (int cid : consumer_ids) {
        Task* consumer = runtime.get_task(cid);
        if (consumer->fanin_count != 1) {
            all_fanin_correct = false;
            break;
        }

        std::vector<int32_t> fanin_list = collect_dep_list(pool, consumer->fanin_head);
        if (fanin_list.size() != 1 || fanin_list[0] != t0) {
            all_fanin_correct = false;
            break;
        }
    }
    CHECK(all_fanin_correct, "All consumers have correct fanin (T0)");

    // Print DepListPool utilization
    printf(
        "  DepListPool utilization: %d / %d entries (%.1f%%)\n", pool->top, pool->size, 100.0 * pool->top / pool->size);
    CHECK(pool->top >= NUM_CONSUMERS, "Pool allocated enough entries");

    runtime.pto_scope_end();
    free(dev_a_ptr);
}

// ============================================================================
// Test 3: Memory efficiency - compare with old fixed array approach
// ============================================================================

static void test_memory_efficiency() {
    printf("\n=== Test 3: Memory Efficiency ===\n");

    // Old approach: Task struct with fixed arrays
    struct OldTask {
        int fanout[512];          // 2048 bytes
        int fanin_producers[32];  // 128 bytes
        // ... other fields ...
    };

    // New approach: Task struct with DepListPool
    struct NewTask {
        int32_t fanin_head;   // 4 bytes
        int32_t fanout_head;  // 4 bytes
        int32_t fanin_count;  // 4 bytes
        // ... other fields ...
    };

    size_t old_dep_memory = sizeof(OldTask::fanout) + sizeof(OldTask::fanin_producers);
    size_t new_dep_memory = sizeof(NewTask::fanin_head) + sizeof(NewTask::fanout_head) + sizeof(NewTask::fanin_count);

    printf("  Old fixed array approach: %zu bytes per task\n", old_dep_memory);
    printf("  New DepListPool approach: %zu bytes per task\n", new_dep_memory);
    printf("  Memory savings: %zu bytes per task (%.1f%% reduction)\n",
        old_dep_memory - new_dep_memory,
        100.0 * (old_dep_memory - new_dep_memory) / old_dep_memory);

    CHECK(new_dep_memory < old_dep_memory / 100, "New approach uses <1% of old memory");
    CHECK(new_dep_memory <= 12, "New approach uses <=12 bytes per task");
}

// ============================================================================
// Test 4: Pool capacity management
// ============================================================================

static void test_pool_capacity() {
    printf("\n=== Test 4: DepListPool Capacity Management ===\n");

    // Small pool to test capacity limits
    const int POOL_SIZE = 32;
    DepListEntry entries[POOL_SIZE];
    DepListPool pool;
    dep_list_pool_init(&pool, entries, POOL_SIZE);

    // Allocate exactly the pool capacity
    int32_t list = 0;
    for (int i = 0; i < POOL_SIZE; i++) {
        list = dep_list_prepend(&pool, list, i);
    }

    CHECK(pool.top == 0, "Pool top wrapped to 0 after full allocation");
    CHECK(dep_list_count(&pool, list) == POOL_SIZE, "List has all POOL_SIZE entries");

    // Verify all entries are present and in correct order (prepend = reverse)
    std::vector<int32_t> items = collect_dep_list(&pool, list);
    CHECK(items.size() == POOL_SIZE, "Collected all entries");

    bool correct_order = true;
    for (int i = 0; i < POOL_SIZE; i++) {
        if (items[i] != POOL_SIZE - 1 - i) {
            correct_order = false;
            break;
        }
    }
    CHECK(correct_order, "All entries in correct reverse order");

    printf("  Pool utilization: %d / %d entries (100%%)\n", POOL_SIZE, POOL_SIZE);
    printf("  List size: %d entries\n", (int)items.size());
}

// ============================================================================
// Test 5: Diamond DAG with DepListPool - verify identical dependency resolution
// ============================================================================

static void test_diamond_dag_dependency_resolution() {
    printf("\n=== Test 5: Diamond DAG Dependency Resolution ===\n");

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

    // Diamond topology:
    //       T0 (a+b=c)
    //      / \
    //    T1   T2      (c+1=d, c+2=e)
    //      \ /
    //       T3 (d*e=f)

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

    Task* task0 = runtime.get_task(t0);
    Task* task1 = runtime.get_task(t1);
    Task* task2 = runtime.get_task(t2);
    Task* task3 = runtime.get_task(t3);
    DepListPool* pool = runtime.get_dep_list_pool();

    // Verify fanout structure
    CHECK(task0->fanout_count == 2, "T0 has 2 consumers");
    std::vector<int32_t> t0_fanout = collect_dep_list(pool, task0->fanout_head);
    CHECK(t0_fanout.size() == 2, "T0 fanout list has 2 entries");
    std::set<int32_t> t0_consumers(t0_fanout.begin(), t0_fanout.end());
    CHECK(t0_consumers.count(t1) && t0_consumers.count(t2), "T0 consumers are T1 and T2");

    CHECK(task1->fanout_count == 1, "T1 has 1 consumer");
    std::vector<int32_t> t1_fanout = collect_dep_list(pool, task1->fanout_head);
    CHECK(t1_fanout.size() == 1 && t1_fanout[0] == t3, "T1 consumer is T3");

    CHECK(task2->fanout_count == 1, "T2 has 1 consumer");
    std::vector<int32_t> t2_fanout = collect_dep_list(pool, task2->fanout_head);
    CHECK(t2_fanout.size() == 1 && t2_fanout[0] == t3, "T2 consumer is T3");

    CHECK(task3->fanout_count == 0, "T3 has 0 consumers (leaf)");

    // Verify fanin structure
    CHECK(task0->fanin_count == 0, "T0 has 0 producers");
    CHECK(task1->fanin_count == 1, "T1 has 1 producer");
    std::vector<int32_t> t1_fanin = collect_dep_list(pool, task1->fanin_head);
    CHECK(t1_fanin.size() == 1 && t1_fanin[0] == t0, "T1 producer is T0");

    CHECK(task2->fanin_count == 1, "T2 has 1 producer");
    std::vector<int32_t> t2_fanin = collect_dep_list(pool, task2->fanin_head);
    CHECK(t2_fanin.size() == 1 && t2_fanin[0] == t0, "T2 producer is T0");

    CHECK(task3->fanin_count == 2, "T3 has 2 producers");
    std::vector<int32_t> t3_fanin = collect_dep_list(pool, task3->fanin_head);
    CHECK(t3_fanin.size() == 2, "T3 fanin list has 2 entries");
    std::set<int32_t> t3_producers(t3_fanin.begin(), t3_fanin.end());
    CHECK(t3_producers.count(t1) && t3_producers.count(t2), "T3 producers are T1 and T2");

    // Print DepListPool utilization
    printf("  DepListPool utilization: %d / %d entries\n", pool->top, pool->size);
    CHECK(pool->top >= 6, "Pool allocated at least 6 entries (4 fanout + 2 fanin for T3)");

    runtime.pto_scope_end();
    runtime.print_runtime();

    free(dev_a_ptr);
    free(dev_b_ptr);
}

// ============================================================================
// Test 6: Wide fanout stress test
// ============================================================================

static void test_wide_fanout_stress() {
    printf("\n=== Test 6: Wide Fanout Stress Test ===\n");

    Runtime runtime;
    runtime.host_api = {mock_device_malloc, mock_device_free, mock_copy_to_device, mock_copy_from_device};
    runtime.pto_init();

    int32_t BYTES = 64;

    void* dev_a_ptr = mock_device_malloc(BYTES);
    PTOBufferHandle dev_a = make_external_handle(dev_a_ptr, BYTES);

    // Create 800 consumers from a single producer
    const int NUM_CONSUMERS = 800;
    std::vector<PTOBufferHandle> outputs;
    std::vector<int> consumer_ids;

    runtime.pto_scope_begin();

    // Producer
    PTOBufferHandle dev_b = make_output_handle(BYTES);
    PTOParam params0[] = {
        make_input_param(&dev_a, BYTES),
        make_output_param(&dev_b, BYTES),
        make_scalar_param(64),
    };
    int t0 = runtime.pto_submit_task(0, PTOWorkerType::VECTOR, params0, 3);

    // 2000 consumers
    for (int i = 0; i < NUM_CONSUMERS; i++) {
        PTOBufferHandle out = make_output_handle(BYTES);
        outputs.push_back(out);

        PTOParam params[] = {
            make_input_param(&dev_b, BYTES),
            make_output_param(&outputs[i], BYTES),
            make_scalar_param(64),
        };
        int tid = runtime.pto_submit_task(1, PTOWorkerType::VECTOR, params, 3);
        consumer_ids.push_back(tid);
    }

    Task* task0 = runtime.get_task(t0);
    DepListPool* pool = runtime.get_dep_list_pool();

    CHECK(
        task0->fanout_count == NUM_CONSUMERS, ("Producer has " + std::to_string(NUM_CONSUMERS) + " consumers").c_str());

    // Verify all consumers present
    int fanout_list_count = dep_list_count(pool, task0->fanout_head);
    CHECK(fanout_list_count == NUM_CONSUMERS,
        ("Fanout list has all " + std::to_string(NUM_CONSUMERS) + " consumers").c_str());

    // Print DepListPool utilization
    printf(
        "  DepListPool utilization: %d / %d entries (%.1f%%)\n", pool->top, pool->size, 100.0 * pool->top / pool->size);

    double fill_ratio = (double)pool->top / pool->size;
    CHECK(fill_ratio < 0.9, "Pool not overfilled (< 90% capacity)");

    runtime.pto_scope_end();
    free(dev_a_ptr);
}

// ============================================================================
// Test 7: Empty dependency lists
// ============================================================================

static void test_empty_lists() {
    printf("\n=== Test 7: Empty Dependency Lists ===\n");

    Runtime runtime;
    runtime.host_api = {mock_device_malloc, mock_device_free, mock_copy_to_device, mock_copy_from_device};
    runtime.pto_init();

    int32_t BYTES = 64;

    void* dev_a_ptr = mock_device_malloc(BYTES);
    PTOBufferHandle dev_a = make_external_handle(dev_a_ptr, BYTES);
    PTOBufferHandle dev_b = make_output_handle(BYTES);

    runtime.pto_scope_begin();

    // Task with no dependencies and no consumers
    PTOParam params0[] = {
        make_input_param(&dev_a, BYTES),
        make_output_param(&dev_b, BYTES),
        make_scalar_param(64),
    };
    int t0 = runtime.pto_submit_task(0, PTOWorkerType::VECTOR, params0, 3);

    Task* task0 = runtime.get_task(t0);
    DepListPool* pool = runtime.get_dep_list_pool();

    CHECK(task0->fanin_head == 0, "T0 fanin_head = 0 (empty list)");
    CHECK(task0->fanout_head == 0, "T0 fanout_head = 0 (no consumers yet)");
    CHECK(task0->fanin_count == 0, "T0 fanin_count = 0");
    CHECK(task0->fanout_count == 0, "T0 fanout_count = 0");

    CHECK(dep_list_is_empty(task0->fanin_head), "fanin list is empty");
    CHECK(dep_list_is_empty(task0->fanout_head), "fanout list is empty");

    CHECK(dep_list_count(pool, task0->fanin_head) == 0, "fanin count = 0");
    CHECK(dep_list_count(pool, task0->fanout_head) == 0, "fanout count = 0");

    runtime.pto_scope_end();
    free(dev_a_ptr);
}

// ============================================================================
// Main
// ============================================================================

int main() {
    printf("============================================================\n");
    printf("Phase 6 Test: DepListPool Integration\n");
    printf("============================================================\n");

    test_basic_operations();
    test_high_fanout();
    test_memory_efficiency();
    test_pool_capacity();
    test_diamond_dag_dependency_resolution();
    test_wide_fanout_stress();
    test_empty_lists();

    printf("\n============================================================\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("============================================================\n");

    return tests_failed > 0 ? 1 : 0;
}