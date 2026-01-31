# Divergence to The Original Orchestration Implementation

1. **Add explicit `pto_alloc()` API**
    - Orchestration function needs buffer address BEFORE submitting tasks
    - Cannot use implicit allocation (inside submit) because address must be passed to dependent tasks

2. **Add explicit `pto_free()` API**
    - Does NOT immediately free memory
    - Signals "no more references will be added to this buffer"
    - Device recycles memory at its discretion after all usage finishes

3. **Deprecate scope-based lifecycle**
    - Scope (`pto_scope_begin` / `pto_scope_end`) is no longer needed
    - Buffer lifetime is fully managed by explicit `pto_alloc()` / `pto_free()` paired with buffer-level reference counting (see §5)

4. **Two sources of buffer addresses for INCORE calls**
    - Runtime-allocated: via `pto_alloc()`
    - External: from orchestration function's parameter list (passed in by caller)

5. **Buffer lifecycle needs its own consumer tracking**
    - Cannot rely on producer's fanout counter: a buffer may have multiple producers
    - Explicitly allocated buffers require buffer-level reference counting, independent of task-level fanout

6. **Version control for in-place updates**
    - Support in-place tensor updates at the same address with different versions
    - API: `pto_version_inc(tensor)` returns new versioned handle (SSA-style)
    - Write to version `v` waits for all reads from version `v-1` to complete
    - Read from version `v` waits for writes to version `v` to complete (region-dependent)

7. **Strided tensor descriptors for TensorMap**
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

8. **TensorMap overlap judgment**
    - Achieving both high efficiency and high accuracy simultaneously is not feasible. We provide several strategies ranging from fastest/low-accuracy (allows false positives but not false negatives) to slowest/high-accuracy. Each strategy is assigned per tensor, with higher accuracy requiring more information. When comparing two tensors, we perform the most accurate overlap judgment possible based on their shared information.

    ```
    Example: Two tensors A and B with different strategy levels

    Tensor A: strategy = BoundingBox   (knows addr, total_size)
    Tensor B: strategy = StridedExact  (knows addr, strides, repeats, n_dims)

    Overlap check picks the common level → BoundingBox (fast, may false-positive):
      A: [addr=0x1000, size=256]  → range [0x1000, 0x1100)
      B: [addr=0x1080, size=128]  → range [0x1080, 0x1100)
      Result: overlapping ✓ (conservative — may be a false positive if the
              actual strided elements of B don't touch A's region)

    If both had StridedExact, we could compare element-by-element
    and potentially determine they do NOT overlap — but at higher cost.
    ```