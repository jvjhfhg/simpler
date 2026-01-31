/**
 * PTO In-Place Update Test (Phase 7)
 *
 * Tests pto_version_inc() for SSA-style versioning of in-place updates.
 *
 * Formula: f = (((a + 1) + 1) + 1) + 1 = a + 4
 * Using in-place updates on buffer 'x':
 *   x_v0 = a + 1    (first operation)
 *   x_v1 = x_v0 + 1 (in-place, version 0 -> 1)
 *   x_v2 = x_v1 + 1 (in-place, version 1 -> 2)
 *   f    = x_v2 + 1 (final result)
 *
 * With a=2.0, expected result is 6.0 for all elements.
 *
 * This validates:
 * - pto_version_inc() creates new versioned handles
 * - Dependencies are correctly tracked across versions
 * - In-place updates execute in correct order
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

int build_inplace_test_graph(Runtime* runtime, uint64_t* args, int arg_count) {
    if (arg_count < 7) {
        std::cerr << "build_inplace_test_graph: Expected at least 7 args, got " << arg_count << '\n';
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

    std::cout << "\n=== build_inplace_test_graph: In-Place Update Test ===" << '\n';
    std::cout << "Formula: f = ((a + 1) + 1) + 1 = a + 3\n";
    std::cout << "Using in-place updates with pto_version_inc()\n";
    std::cout << "SIZE: " << SIZE << " elements\n";

    // Initialize PTO mode
    runtime->pto_init();

    // Allocate device buffers via PTO API
    std::cout << "\n=== Allocating Device Memory (PTO API) ===" << '\n';
    int32_t BYTES = SIZE * sizeof(float);

    // Input buffer
    PTOBufferHandle* dev_a = runtime->pto_alloc(size_a);
    runtime->host_api.copy_to_device((void*)dev_a->addr, host_a, size_a);
    std::cout << "Tensor a: " << size_a << " bytes copied to device\n";

    // In-place working buffer (will be versioned)
    PTOBufferHandle* dev_x_v0 = runtime->pto_alloc(BYTES);
    std::cout << "Tensor x (in-place): " << BYTES << " bytes allocated\n";

    // Output buffer
    PTOBufferHandle* dev_f = runtime->pto_alloc(size_f);
    runtime->record_tensor_pair(host_f, (void*)dev_f->addr, size_f);
    std::cout << "Tensor f (output): " << size_f << " bytes allocated\n";

    // Helper to encode float as uint64_t
    union { float f32; uint64_t u64; } conv;
    conv.f32 = 1.0f;

    // Task 0: x_v0 = a + 0 (effectively copy, using kernel_add_scalar with 0)
    // We use kernel_add with a and zeros, but simpler: just use kernel_add_scalar with 0
    // Actually, let's just use x = a + 1 directly for first step
    PTOParam params0[] = {
        make_input_param(dev_a, BYTES),
        make_scalar_param(conv.u64),  // 1.0f
        make_output_param(dev_x_v0, BYTES),
        make_scalar_param((uint64_t)SIZE),
    };
    int t0 = runtime->pto_submit_task(1, 1, params0, 4);  // kernel_add_scalar
    std::cout << "Task " << t0 << ": x_v0 = a + 1\n";

    // Create version 1 of x for in-place update
    PTOBufferHandle* dev_x_v1 = runtime->pto_version_inc(dev_x_v0);
    std::cout << "Created x_v1 via pto_version_inc() (version=" << dev_x_v1->version << ")\n";

    // Task 1: x_v1 = x_v0 + 1 (in-place update)
    PTOParam params1[] = {
        make_input_param(dev_x_v0, BYTES),
        make_scalar_param(conv.u64),  // 1.0f
        make_output_param(dev_x_v1, BYTES),
        make_scalar_param((uint64_t)SIZE),
    };
    int t1 = runtime->pto_submit_task(1, 1, params1, 4);
    std::cout << "Task " << t1 << ": x_v1 = x_v0 + 1 (in-place)\n";

    // Create version 2 of x
    PTOBufferHandle* dev_x_v2 = runtime->pto_version_inc(dev_x_v1);
    std::cout << "Created x_v2 via pto_version_inc() (version=" << dev_x_v2->version << ")\n";

    // Task 2: x_v2 = x_v1 + 1 (in-place update)
    PTOParam params2[] = {
        make_input_param(dev_x_v1, BYTES),
        make_scalar_param(conv.u64),  // 1.0f
        make_output_param(dev_x_v2, BYTES),
        make_scalar_param((uint64_t)SIZE),
    };
    int t2 = runtime->pto_submit_task(1, 1, params2, 4);
    std::cout << "Task " << t2 << ": x_v2 = x_v1 + 1 (in-place)\n";

    // Task 3: f = x_v2 + 1 (final result)
    PTOParam params3[] = {
        make_input_param(dev_x_v2, BYTES),
        make_scalar_param(conv.u64),  // 1.0f
        make_output_param(dev_f, BYTES),
        make_scalar_param((uint64_t)SIZE),
    };
    int t3 = runtime->pto_submit_task(1, 1, params3, 4);
    std::cout << "Task " << t3 << ": f = x_v2 + 1 (final)\n";

    std::cout << "\nCreated runtime with " << runtime->get_task_count() << " tasks\n";
    std::cout << "Expected dependency chain: t0 -> t1 -> t2 -> t3\n";
    runtime->print_runtime();

    return 0;
}

}  // extern "C"
