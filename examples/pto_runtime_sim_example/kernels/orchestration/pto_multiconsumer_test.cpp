/**
 * PTO Multi-Consumer Buffer Test (Phase 7)
 *
 * Tests buffer-level reference counting with multiple consumers.
 *
 * Formula: f = (a + 1) + (a + 2) + (a + 3)
 * Buffer 'a' has 3 consumers (diamond pattern with fan-out of 3).
 *
 *        a
 *      / | \
 *    b   c   d     (b=a+1, c=a+2, d=a+3)
 *      \ | /
 *        e         (e=b+c)
 *        |
 *        f         (f=e+d)
 *
 * With a=1.0, expected result is:
 *   b = 2, c = 3, d = 4
 *   e = b + c = 5
 *   f = e + d = 9
 *
 * This validates:
 * - Multiple consumers of a single buffer
 * - Correct reference counting (buffer stays alive until all consumers done)
 * - Complex DAG execution order
 */

#include "runtime.h"
#include <iostream>

// Helper: create a BoundingBox tensor descriptor
static PTOTensorDescriptor make_tensor_bbox(uint64_t addr, int32_t size) {
    PTOTensorDescriptor t = {};
    t.addr = addr;
    t.start_offset = 0;
    t.strides[0] = 1;
    t.repeats[0] = size;
    t.n_dims = 1;
    t.strategy = PTO_OVERLAP_BOUNDING_BOX;
    return t;
}

// Helper: create a scalar PTOParam
static PTOParam make_scalar_param(uint64_t value) {
    PTOParam p = {};
    p.type = PTO_PARAM_SCALAR;
    p.buffer = nullptr;
    p.scalar_value = value;
    return p;
}

// Helper: create an input PTOParam
static PTOParam make_input_param(PTOBufferHandle* buf, int32_t size) {
    PTOParam p = {};
    p.type = PTO_PARAM_INPUT;
    p.tensor = make_tensor_bbox(buf->addr, size);
    p.buffer = buf;
    p.scalar_value = 0;
    return p;
}

// Helper: create an output PTOParam
static PTOParam make_output_param(PTOBufferHandle* buf, int32_t size) {
    PTOParam p = {};
    p.type = PTO_PARAM_OUTPUT;
    p.tensor = make_tensor_bbox(buf->addr, size);
    p.buffer = buf;
    p.scalar_value = 0;
    return p;
}

extern "C" {

int build_multiconsumer_test_graph(Runtime* runtime, uint64_t* args, int arg_count) {
    if (arg_count < 7) {
        std::cerr << "build_multiconsumer_test_graph: Expected at least 7 args, got " << arg_count << '\n';
        return -1;
    }

    void* host_a = reinterpret_cast<void*>(args[0]);
    void* host_b = reinterpret_cast<void*>(args[1]);  // unused
    void* host_f = reinterpret_cast<void*>(args[2]);
    size_t size_a = static_cast<size_t>(args[3]);
    size_t size_b = static_cast<size_t>(args[4]);  // unused
    size_t size_f = static_cast<size_t>(args[5]);
    int SIZE = static_cast<int>(args[6]);

    (void)host_b;
    (void)size_b;

    std::cout << "\n=== build_multiconsumer_test_graph: Multi-Consumer Test ===" << '\n';
    std::cout << "Formula: f = (a+1) + (a+2) + (a+3) = 3a + 6\n";
    std::cout << "Tests multiple consumers of buffer 'a'\n";
    std::cout << "SIZE: " << SIZE << " elements\n";

    // Initialize PTO mode
    runtime->pto_init();

    // Allocate device buffers via PTO API
    std::cout << "\n=== Allocating Device Memory (PTO API) ===" << '\n';
    int32_t BYTES = SIZE * sizeof(float);

    // Input buffer (has 3 consumers)
    PTOBufferHandle* dev_a = runtime->pto_alloc(size_a);
    runtime->host_api.copy_to_device((void*)dev_a->addr, host_a, size_a);
    std::cout << "Tensor a: " << size_a << " bytes copied to device (3 consumers)\n";

    // Intermediate buffers
    PTOBufferHandle* dev_b = runtime->pto_alloc(BYTES);
    PTOBufferHandle* dev_c = runtime->pto_alloc(BYTES);
    PTOBufferHandle* dev_d = runtime->pto_alloc(BYTES);
    PTOBufferHandle* dev_e = runtime->pto_alloc(BYTES);
    std::cout << "Allocated intermediate tensors b, c, d, e\n";

    // Output buffer
    PTOBufferHandle* dev_f = runtime->pto_alloc(size_f);
    runtime->record_tensor_pair(host_f, (void*)dev_f->addr, size_f);
    std::cout << "Tensor f (output): " << size_f << " bytes allocated\n";

    // Helper to encode float as uint64_t
    union { float f32; uint64_t u64; } conv;

    // Task 0: b = a + 1 (consumer 1 of 'a')
    conv.f32 = 1.0f;
    PTOParam params0[] = {
        make_input_param(dev_a, BYTES),
        make_scalar_param(conv.u64),
        make_output_param(dev_b, BYTES),
        make_scalar_param((uint64_t)SIZE),
    };
    int t0 = runtime->pto_submit_task(1, 1, params0, 4);  // kernel_add_scalar
    std::cout << "Task " << t0 << ": b = a + 1 (consumer 1 of 'a')\n";

    // Task 1: c = a + 2 (consumer 2 of 'a')
    conv.f32 = 2.0f;
    PTOParam params1[] = {
        make_input_param(dev_a, BYTES),
        make_scalar_param(conv.u64),
        make_output_param(dev_c, BYTES),
        make_scalar_param((uint64_t)SIZE),
    };
    int t1 = runtime->pto_submit_task(1, 1, params1, 4);
    std::cout << "Task " << t1 << ": c = a + 2 (consumer 2 of 'a')\n";

    // Task 2: d = a + 3 (consumer 3 of 'a')
    conv.f32 = 3.0f;
    PTOParam params2[] = {
        make_input_param(dev_a, BYTES),
        make_scalar_param(conv.u64),
        make_output_param(dev_d, BYTES),
        make_scalar_param((uint64_t)SIZE),
    };
    int t2 = runtime->pto_submit_task(1, 1, params2, 4);
    std::cout << "Task " << t2 << ": d = a + 3 (consumer 3 of 'a')\n";

    // Task 3: e = b + c (waits for t0 and t1)
    PTOParam params3[] = {
        make_input_param(dev_b, BYTES),
        make_input_param(dev_c, BYTES),
        make_output_param(dev_e, BYTES),
        make_scalar_param((uint64_t)SIZE),
    };
    int t3 = runtime->pto_submit_task(0, 1, params3, 4);  // kernel_add
    std::cout << "Task " << t3 << ": e = b + c\n";

    // Task 4: f = e + d (waits for t2 and t3)
    PTOParam params4[] = {
        make_input_param(dev_e, BYTES),
        make_input_param(dev_d, BYTES),
        make_output_param(dev_f, BYTES),
        make_scalar_param((uint64_t)SIZE),
    };
    int t4 = runtime->pto_submit_task(0, 1, params4, 4);  // kernel_add
    std::cout << "Task " << t4 << ": f = e + d\n";

    std::cout << "\nCreated runtime with " << runtime->get_task_count() << " tasks\n";
    std::cout << "DAG structure:\n";
    std::cout << "       a\n";
    std::cout << "     / | \\\n";
    std::cout << "    b  c  d   (t0, t1, t2 all depend on 'a')\n";
    std::cout << "     \\ | /\n";
    std::cout << "       e      (t3 depends on b, c)\n";
    std::cout << "       |\n";
    std::cout << "       f      (t4 depends on e, d)\n";
    runtime->print_runtime();

    return 0;
}

}  // extern "C"
