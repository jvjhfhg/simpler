# Midway Review

1. `PTOParam` has `TensorDescriptor tensor` and `PTOBufferHandle* buffer` but is the second one really in need? Like `TensorDescriptor` holds all the information (`addr` and `size`). `PTOBufferHandle` is sort of redundant. We need to remove the `PTOBufferHandle` and directly create tensor descriptor with solid tensor metadata (strides, repeats, etc.) in orchestration function. Like what's done, address should be specified right at when it's submitted inside `pto_submit_task`.

   > **Analysis (2026-02-05)**: Partially addressed. `PTOBufferHandle::version` was removed (#04). However, `PTOBufferHandle` still exists with `addr` and `size`. It serves as a user-facing handle for buffer identity tracking (e.g., `addr == 0` triggers allocation). `TensorDescriptor` is per-param and describes access patterns. There's redundancy (`addr`/`size` in both), but removing `PTOBufferHandle` entirely requires API redesign. **Status: Open - low priority.**

2. In `pto_submit_task`, it allocates for tensors without a specific address (not external and not a new version of existing tensors). It should be verified that tensors which are gonna be allocated need to be an entire tensor (contiguous contents and maybe other requirements).

   > **Analysis (2026-02-05)**: Still a gap. Current code ([runtime.cpp:363-412](runtime/runtime.cpp#L363-L412)) allocates `buffer->size` bytes without validating tensor contiguity or other properties. No `is_contiguous()` check before allocation. **Status: Open.**

3. To do tensor calculation, aicores may need the layout information of tensor which is involved in tensor metadata `TensorDescriptor`. But in the present implementation, tensor metadata is not passed into INCORE functions (or aicore kernels, can be seen at `examples/orch_build_graph_example/kernels/aiv`). It needs to be passed in.

   > **Analysis (2026-02-05)**: Still a gap. Kernels receive raw pointers and scalars only ([runtime.cpp:414-424](runtime/runtime.cpp#L414-L424)). `TensorDescriptor` is used solely for dependency detection, not passed to kernels. Example: `kernel_add.cpp` takes `args[0..2]` as pointers, `args[3]` as size - no stride/repeat info. **Status: Open.**

4. `TensorDescriptor` now represents in-byte repeats and strides, needs to be modified into dtype-wise. The following dtypes should be supported:

    ```c++
    enum class DataType : uint32_t {
        FLOAT32,    // 4 bytes
        FLOAT16,    // 2 bytes
        INT32,      // 4 bytes
        INT16,      // 2 bytes
        INT8,       // 1 bytes
        UINT8,      // 1 bytes
        BFLOAT16,   // 2 bytes
        INT64,      // 8 bytes
        UINT64,     // 8 bytes
    };

    inline uint64_t get_element_size(DataType dtype) {
        switch (dtype) {
            case DataType::FLOAT32: return 4;
            case DataType::FLOAT16: return 2;
            case DataType::INT32:   return 4;
            case DataType::INT16:   return 2;
            case DataType::INT8:    return 1;
            case DataType::UINT8:   return 1;
            case DataType::BFLOAT16: return 2;
            case DataType::INT64:   return 8;
            case DataType::UINT64:  return 8;
            default: return 0;
        }
    }
    ```

    New `TensorDescriptor` definition:

    ```c++
    struct PTOBufferHandle {
        uint64_t addr;  // Device memory address (bytes)
        int32_t size;   // Total buffer size in bytes
    };

    struct TensorDescriptor {
        PTOBufferHandle buffer;                         // Underlying memory buffer
        uint64_t start_offset;                          // 起始偏移 (元素个数)
        uint64_t strides[RUNTIME_MAX_TENSOR_DIMS];      // 各维度步长 (元素个数), 索引0为高维
        uint64_t repeats[RUNTIME_MAX_TENSOR_DIMS];      // 各维度重复次数, 索引0为高维
        uint64_t ndims;                                 // 使用的维度数
        DataType dtype;                                 // 数据类型
        int32_t version;                                // 张量版本号
        OverlapType overlap_type;                       // 重叠检测类型
    };
    ```

    `is_overlap` needs to be adapted when the two `TensorDescriptor`'s have different dtypes.

   > **Analysis & Implementation (2026-02-05)**: **COMPLETED**.
   >
   > **Key Design Decision**: Refactored `TensorDescriptor` to embed `PTOBufferHandle buffer` instead of separate `addr` and `size` fields. This eliminates redundancy and clarifies that:
   > - `buffer` represents the underlying memory allocation (addr in bytes, size in bytes)
   > - `dtype` specifies how to interpret the buffer contents
   > - `start_offset`, `strides[]`, `repeats[]` are in element units
   >
   > **Implementation Details**:
   > - Added `DataType` enum in [runtime/data_type.h](runtime/data_type.h) with FLOAT32, FLOAT16, INT32, INT16, INT8, UINT8, BFLOAT16, INT64, UINT64
   > - Moved `PTOBufferHandle` definition from `pto_types.h` to `tensor_descriptor.h` (more appropriate location)
   > - Updated `TensorDescriptor` structure:
   >   - Replaced `addr` + `size` with embedded `PTOBufferHandle buffer`
   >   - Added `DataType dtype` field
   >   - All dimensional fields (`start_offset`, `strides[]`, `repeats[]`) now use element units
   > - Updated `is_overlap()` and `complex_overlap()` to convert element offsets to bytes when comparing different dtypes
   > - Updated all call sites: runtime.cpp, test files, orchestration examples, tensor_map.h
   > - All tests pass (307 runtime tests + 152 tensor descriptor tests)
   >
   > **Status: Closed.**

---

## Implementation Plan

### Phase 1: Add DataType support (addresses #4) - ✅ COMPLETED

**Implemented (2026-02-05)**:

1. **Added `DataType` enum** in [runtime/data_type.h](runtime/data_type.h)
   - Defined enum with FLOAT32, FLOAT16, INT32, INT16, INT8, UINT8, BFLOAT16, INT64, UINT64
   - Added `get_element_size(DataType)` helper
   - Added `get_dtype_name(DataType)` helper for debugging

2. **Refactored `TensorDescriptor`** ([tensor_descriptor.h](runtime/tensor_descriptor.h))
   - **Key change**: Replaced separate `addr` and `size` fields with embedded `PTOBufferHandle buffer`
   - Added `DataType dtype` field
   - `buffer.addr` and `buffer.size` remain in bytes (hardware/allocation requirement)
   - Converted `start_offset`, `strides[]`, `repeats[]` to element-wise units
   - Removed `size_bytes()` helper (no longer needed - just use `buffer.size`)

3. **Updated `is_overlap()`** ([tensor_descriptor.cpp](runtime/tensor_descriptor.cpp))
   - Converts element offsets to bytes when comparing descriptors with different dtypes
   - Core logic: compares byte ranges regardless of dtype
   - Falls back to `complex_overlap()` for different dtypes

4. **Updated all call sites**
   - [runtime.cpp](runtime/runtime.cpp): Uses `buffer.size` for allocation (already in bytes)
   - [tensor_map.h](runtime/tensor_map.h): Uses `buffer.addr` for hashing
   - Test files: Updated `make_tensor_bbox()` helpers to use new structure
   - [orch_example_orch.cpp](../../../examples/orch_build_graph_example/kernels/orchestration/orch_example_orch.cpp): Updated helper functions

**Test Results**: All 459 tests pass (307 runtime + 152 tensor descriptor tests)

### Phase 2: Validate allocation constraints (addresses #2)

1. **Add `is_contiguous()` to `TensorDescriptor`**
   - Check if strides match dense layout for given repeats

2. **Add validation in `pto_submit_task`**
   - Before allocation, assert `tensor.is_contiguous()`
   - Log warning or error if non-contiguous tensor needs allocation

### Phase 3: Pass tensor metadata to kernels (addresses #3)

1. **Define kernel parameter convention**
   - Option A: Pass `TensorDescriptor*` as first N args (one per tensor param)
   - Option B: Pack all descriptors into a metadata struct

2. **Update kernel dispatch in `runtime.cpp`**
   - Serialize `TensorDescriptor` into kernel args

3. **Update example kernels**
   - Modify `kernel_add.cpp` etc. to receive and use `TensorDescriptor`
   - Demonstrate strided access pattern

### Phase 4: Remove PTOBufferHandle redundancy (addresses #1) - Low Priority

#### Problem Analysis

Current design has two overlapping concepts:
- **`PTOBufferHandle`**: Mutable identity handle with `addr` and `size`. Used for:
  - Buffer identity tracking (multiple params can share one buffer)
  - Deferred allocation detection (`addr == 0` triggers allocation)
  - Address propagation (runtime fills `addr`, orchestration reads it back)
- **`TensorDescriptor`**: Access pattern descriptor with `addr`, `size`, strides, repeats, etc.

The redundancy:
- Both store `addr` and `size`
- `PTOParam` contains both `TensorDescriptor tensor` and `PTOBufferHandle* buffer`
- After `pto_submit_task`, code does `params[i].tensor.addr = params[i].buffer->addr`

#### Design Goals

1. Eliminate `PTOBufferHandle` as a separate type
2. Preserve buffer identity semantics (multiple tensors sharing one buffer)
3. Preserve deferred allocation semantics
4. Simplify orchestration API

#### Proposed API Design

##### 4.1 New Types

```cpp
// Buffer identity handle - opaque ID for tracking shared buffers
using PTOBufferId = int32_t;
constexpr PTOBufferId PTO_BUFFER_INVALID = -1;
constexpr PTOBufferId PTO_BUFFER_EXTERNAL = -2;  // Pre-allocated by host

// Extended TensorDescriptor (after Phase 1 changes)
struct TensorDescriptor {
    uint64_t addr;                              // GM base address (0 = needs allocation)
    uint64_t size;                              // Total size in elements
    uint64_t start_offset;                      // Start offset in elements
    uint64_t strides[RUNTIME_MAX_TENSOR_DIMS];  // Per-dimension strides in elements
    uint64_t repeats[RUNTIME_MAX_TENSOR_DIMS];  // Per-dimension repeats
    uint64_t ndims;                             // Number of dimensions used
    DataType dtype;                             // Data type (from Phase 1)
    int32_t version;                            // Tensor version for in-place updates
    OverlapType overlap_type;                   // Overlap detection type
    PTOBufferId buffer_id;                      // NEW: Buffer identity (-1 = unique, -2 = external)
};

// Simplified PTOParam - no more PTOBufferHandle*
struct PTOParam {
    PTOParamType type;          // INPUT, OUTPUT, or SCALAR
    TensorDescriptor tensor;    // Tensor descriptor (includes buffer_id)
    uint64_t scalar_value;      // For SCALAR params
    // REMOVED: PTOBufferHandle* buffer
};
```

##### 4.2 Buffer Registry (Internal to Runtime)

```cpp
// Internal runtime structure for tracking allocated buffers
struct BufferEntry {
    PTOBufferId id;
    uint64_t addr;      // 0 until allocated
    uint64_t size;      // Size in bytes
    int32_t ref_count;  // Number of pending consumers
};

class BufferRegistry {
public:
    // Register a new buffer, returns buffer_id
    PTOBufferId register_buffer(uint64_t size_bytes);

    // Get/set address for a buffer
    uint64_t get_addr(PTOBufferId id) const;
    void set_addr(PTOBufferId id, uint64_t addr);

    // Check if buffer needs allocation
    bool needs_allocation(PTOBufferId id) const;

private:
    std::vector<BufferEntry> entries_;
    PTOBufferId next_id_ = 0;
};
```

##### 4.3 New Orchestration API

```cpp
// Create a new buffer identity (replaces make_output_handle)
// Returns a buffer_id that can be used in multiple TensorDescriptors
PTOBufferId pto_create_buffer(Runtime* rt, uint64_t size_bytes);

// Create TensorDescriptor for an external (pre-allocated) buffer
TensorDescriptor pto_external_tensor(
    uint64_t addr,           // Pre-allocated device address
    uint64_t size,           // Size in elements
    DataType dtype,
    std::initializer_list<uint64_t> strides = {},
    std::initializer_list<uint64_t> repeats = {}
);

// Create TensorDescriptor for a runtime-managed buffer
TensorDescriptor pto_managed_tensor(
    PTOBufferId buffer_id,   // From pto_create_buffer()
    uint64_t size,           // Size in elements
    DataType dtype,
    int32_t version = 0,
    std::initializer_list<uint64_t> strides = {},
    std::initializer_list<uint64_t> repeats = {}
);

// Create input param from tensor descriptor
PTOParam pto_input(const TensorDescriptor& tensor);

// Create output param from tensor descriptor
PTOParam pto_output(const TensorDescriptor& tensor);

// Create scalar param
PTOParam pto_scalar(uint64_t value);

// Submit task (unchanged signature, but PTOParam no longer has buffer pointer)
int32_t pto_submit_task(
    int32_t func_id,
    PTOWorkerType worker_type,
    PTOParam* params,
    int32_t param_count
);

// Query allocated address after submission (for debugging/verification)
uint64_t pto_get_buffer_addr(Runtime* rt, PTOBufferId buffer_id);
```

##### 4.4 Usage Example (Before vs After)

**Before (current API):**
```cpp
// Create buffer handles
PTOBufferHandle dev_a = make_external_handle(dev_a_ptr, size_a);
PTOBufferHandle dev_c = make_output_handle(BYTES);

// Create params with both tensor and buffer
PTOParam params[] = {
    make_input_param(&dev_a, BYTES),      // Has tensor + buffer*
    make_output_param(&dev_c, BYTES),     // Has tensor + buffer*
    make_scalar_param((uint64_t)SIZE),
};
runtime->pto_submit_task(0, PTOWorkerType::VECTOR, params, 3);

// Access allocated address via buffer handle
uint64_t addr = dev_c.addr;
```

**After (new API):**
```cpp
// Create buffer identity
PTOBufferId buf_c = pto_create_buffer(runtime, BYTES);

// Create tensor descriptors directly
TensorDescriptor tensor_a = pto_external_tensor(dev_a_ptr, SIZE, DataType::FLOAT32);
TensorDescriptor tensor_c = pto_managed_tensor(buf_c, SIZE, DataType::FLOAT32);

// Create params with tensor only (no buffer pointer)
PTOParam params[] = {
    pto_input(tensor_a),
    pto_output(tensor_c),
    pto_scalar((uint64_t)SIZE),
};
runtime->pto_submit_task(0, PTOWorkerType::VECTOR, params, 3);

// Access allocated address via buffer registry
uint64_t addr = pto_get_buffer_addr(runtime, buf_c);
```

##### 4.5 In-Place Update Example (Before vs After)

**Before:**
```cpp
PTOBufferHandle dev_x_v0 = make_output_handle(BYTES);
PTOBufferHandle dev_x_v1 = dev_x_v0;  // Copy handle for version tracking

PTOParam params5[] = {
    make_input_param(&dev_x_v0, BYTES, 0),   // version 0
    make_output_param(&dev_x_v1, BYTES, 1),  // version 1, same buffer
};
```

**After:**
```cpp
PTOBufferId buf_x = pto_create_buffer(runtime, BYTES);

TensorDescriptor x_v0 = pto_managed_tensor(buf_x, SIZE, DataType::FLOAT32, /*version=*/0);
TensorDescriptor x_v1 = pto_managed_tensor(buf_x, SIZE, DataType::FLOAT32, /*version=*/1);

PTOParam params5[] = {
    pto_input(x_v0),   // version 0
    pto_output(x_v1),  // version 1, same buffer_id
};
```

##### 4.6 Runtime Implementation Changes

In `pto_submit_task`:
```cpp
int32_t Runtime::pto_submit_task(...) {
    // Phase 1: Collect buffers needing allocation
    for (int i = 0; i < param_count; i++) {
        if (params[i].type == PTOParamType::OUTPUT) {
            PTOBufferId bid = params[i].tensor.buffer_id;
            if (bid >= 0 && buffer_registry_.needs_allocation(bid)) {
                // Mark for allocation
                buffers_to_alloc.push_back(bid);
            }
        }
    }

    // Phase 2: Allocate buffers
    for (PTOBufferId bid : buffers_to_alloc) {
        uint64_t size = buffer_registry_.get_size(bid);
        void* ptr = heap_ring_alloc(..., size, ...);
        buffer_registry_.set_addr(bid, (uint64_t)ptr);
    }

    // Phase 3: Resolve addresses in tensor descriptors
    for (int i = 0; i < param_count; i++) {
        PTOBufferId bid = params[i].tensor.buffer_id;
        if (bid >= 0) {
            params[i].tensor.addr = buffer_registry_.get_addr(bid);
        }
        // External buffers (bid == PTO_BUFFER_EXTERNAL) already have addr set
    }

    // Phase 4: Build kernel args and submit
    ...
}
```

##### 4.7 Migration Path

1. **Step 1**: Add `buffer_id` field to `TensorDescriptor` (backward compatible)
2. **Step 2**: Add `BufferRegistry` to Runtime
3. **Step 3**: Add new API functions (`pto_create_buffer`, `pto_external_tensor`, etc.)
4. **Step 4**: Update example orchestration functions to use new API
5. **Step 5**: Deprecate `PTOBufferHandle` and old helper functions
6. **Step 6**: Remove `PTOBufferHandle* buffer` from `PTOParam`
7. **Step 7**: Remove `PTOBufferHandle` type entirely

##### 4.8 Benefits

- **Simpler mental model**: One type (`TensorDescriptor`) describes everything about a tensor
- **No pointer aliasing**: Buffer identity is explicit via `buffer_id`, not implicit via pointer sharing
- **Type safety**: `PTOBufferId` is a value type, not a pointer that can dangle
- **Cleaner API**: `pto_input(tensor)` vs `make_input_param(&buffer, size, version)`

---

**Recommended order**: Phase 1 → Phase 2 → Phase 3 → Phase 4 (optional)
