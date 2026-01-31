/**
 * PTO-Native Orchestration Example (Phase 6)
 *
 * Same formula as legacy: f = (a + b + 1)(a + b + 2)
 * But uses PTO API: pto_alloc(), pto_submit_task(), pto_free()
 * Dependencies are detected automatically via TensorMap.
 */

#include "runtime.h"
#include <iostream>

// Helper: create a BoundingBox tensor descriptor (simplest strategy)
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

int build_pto_example_graph(Runtime* runtime, uint64_t* args, int arg_count) {
    if (arg_count < 7) {
        std::cerr << "build_pto_example_graph: Expected at least 7 args, got " << arg_count << '\n';
        return -1;
    }

    void* host_a = reinterpret_cast<void*>(args[0]);
    void* host_b = reinterpret_cast<void*>(args[1]);
    void* host_f = reinterpret_cast<void*>(args[2]);
    size_t size_a = static_cast<size_t>(args[3]);
    size_t size_b = static_cast<size_t>(args[4]);
    size_t size_f = static_cast<size_t>(args[5]);
    int SIZE = static_cast<int>(args[6]);

    std::cout << "\n=== build_pto_example_graph: PTO-Native Orchestration ===" << '\n';
    std::cout << "Formula: (a + b + 1)(a + b + 2)\n";
    std::cout << "SIZE: " << SIZE << " elements\n";

    // Initialize PTO mode
    runtime->pto_init();

    // Allocate device buffers via PTO API
    std::cout << "\n=== Allocating Device Memory (PTO API) ===" << '\n';
    int32_t BYTES = SIZE * sizeof(float);

    // External buffers: input a, b
    PTOBufferHandle* dev_a = runtime->pto_alloc(size_a);
    runtime->host_api.copy_to_device((void*)dev_a->addr, host_a, size_a);
    std::cout << "Tensor a: " << size_a << " bytes copied to device\n";

    PTOBufferHandle* dev_b = runtime->pto_alloc(size_b);
    runtime->host_api.copy_to_device((void*)dev_b->addr, host_b, size_b);
    std::cout << "Tensor b: " << size_b << " bytes copied to device\n";

    // Output buffer
    PTOBufferHandle* dev_f = runtime->pto_alloc(size_f);
    runtime->record_tensor_pair(host_f, (void*)dev_f->addr, size_f);
    std::cout << "Tensor f (output): " << size_f << " bytes allocated\n";

    // Intermediate buffers
    PTOBufferHandle* dev_c = runtime->pto_alloc(BYTES);
    PTOBufferHandle* dev_d = runtime->pto_alloc(BYTES);
    PTOBufferHandle* dev_e = runtime->pto_alloc(BYTES);
    std::cout << "Allocated intermediate tensors c, d, e\n";

    // Helper to encode float as uint64_t
    union { float f32; uint64_t u64; } conv;

    // Task 0: c = a + b (func_id=0: kernel_add, AIV)
    // kernel_add expects: src0, src1, out, size
    PTOParam params0[] = {
        make_input_param(dev_a, BYTES),
        make_input_param(dev_b, BYTES),
        make_output_param(dev_c, BYTES),
        make_scalar_param((uint64_t)SIZE),
    };
    int t0 = runtime->pto_submit_task(0, 1, params0, 4);

    // Task 1: d = c + 1 (func_id=1: kernel_add_scalar, AIV)
    // kernel_add_scalar expects: src, scalar, out, size
    conv.f32 = 1.0f;
    PTOParam params1[] = {
        make_input_param(dev_c, BYTES),
        make_scalar_param(conv.u64),
        make_output_param(dev_d, BYTES),
        make_scalar_param((uint64_t)SIZE),
    };
    int t1 = runtime->pto_submit_task(1, 1, params1, 4);

    // Task 2: e = c + 2 (func_id=1: kernel_add_scalar, AIV)
    conv.f32 = 2.0f;
    PTOParam params2[] = {
        make_input_param(dev_c, BYTES),
        make_scalar_param(conv.u64),
        make_output_param(dev_e, BYTES),
        make_scalar_param((uint64_t)SIZE),
    };
    int t2 = runtime->pto_submit_task(1, 1, params2, 4);

    // Task 3: f = d * e (func_id=2: kernel_mul, AIV)
    // kernel_mul expects: src0, src1, out, size
    PTOParam params3[] = {
        make_input_param(dev_d, BYTES),
        make_input_param(dev_e, BYTES),
        make_output_param(dev_f, BYTES),
        make_scalar_param((uint64_t)SIZE),
    };
    int t3 = runtime->pto_submit_task(2, 1, params3, 4);

    std::cout << "\nTasks (auto-dependency via TensorMap):\n";
    std::cout << "  task" << t0 << ": c = a + b\n";
    std::cout << "  task" << t1 << ": d = c + 1\n";
    std::cout << "  task" << t2 << ": e = c + 2\n";
    std::cout << "  task" << t3 << ": f = d * e\n";

    std::cout << "Created runtime with " << runtime->get_task_count() << " tasks\n";
    runtime->print_runtime();

    // Note: In Phase 6, we don't call pto_free() here since the buffers
    // need to stay alive during execution. In Phase 7, with proper deferred
    // freeing via scheduler, pto_free() will mark "no more references"
    // but actual memory reclamation happens after all consumers finish.

    return 0;
}

}  // extern "C"
