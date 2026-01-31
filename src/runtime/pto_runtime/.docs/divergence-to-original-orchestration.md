# Divergence to The Original Orchestration Implementation

1. **Add explicit `pto_alloc()` API**
    - Orchestration function needs buffer address BEFORE submitting tasks
    - Cannot use implicit allocation (inside submit) because address must be passed to dependent tasks

2. **Add explicit `pto_free()` API**
    - Does NOT immediately free memory
    - Signals "no more references will be added to this buffer"
    - Device recycles memory at its discretion after all usage finishes

3. **Two sources of buffer addresses for INCORE calls**
    - Runtime-allocated: via `pto_alloc()`
    - External: from orchestration function's parameter list (passed in by caller)

4. **Buffer lifecycle needs its own consumer tracking**
    - Cannot rely on producer's fanout counter: a buffer may have multiple producers
    - Explicitly allocated buffers require buffer-level reference counting, independent of task-level fanout

5. **Version control for in-place updates**
    - Support in-place tensor updates at the same address with different versions
    - API: `pto_version_inc(tensor)` returns new versioned handle (SSA-style)
    - Write to version `v` waits for all reads from version `v-1` to complete
    - Read from version `v` waits for writes to version `v` to complete (region-dependent)

6. **Strided tensor descriptors for TensorMap**
    - Simple `(base_ptr, offset, size)` cannot express non-contiguous tiles
    - Need strided access pattern: `(addr, start_offset, strides[], repeats[], n_dims)`
    - Enables arbitrary regular memory layouts (tiled, transposed, sliced)

    ```c++
    /**
     * Example: addr=base, start_offset=7, strides=[10, 1], repeats=[3, 6]
     * Access pattern: [addr+7..addr+12], [addr+17..addr+22], [addr+27..addr+32]
     */
    struct TensorDescriptor {
        uint64_t addr;                              // Base address in GM
        uint64_t start_offset;                      // Starting offset from addr
        uint64_t strides[RUNTIME_MAX_TENSOR_DIMS];  // Stride per dimension
        uint64_t repeats[RUNTIME_MAX_TENSOR_DIMS];  // Elements per dimension
        int n_dims;                                 // Number of dimensions
    };
    ```

7. TensorMap overlap judgment: several levels, bound on tensor. Faster level indicates low accuracy data. Use the most accurate common level overlaps.


