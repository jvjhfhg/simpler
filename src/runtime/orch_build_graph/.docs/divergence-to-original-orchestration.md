# Divergence to The Original Orchestration Implementation

This document records the original incorrect divergence points and their corrections.
All corrections have been applied via Plan #02 (pto_runtime_correction_plan.md).

## Original Divergence Points (with corrections applied)

1. **~~Add explicit `pto_alloc()` API~~** — REMOVED
    - ~~Orchestration function needs buffer address BEFORE submitting tasks~~
    - **Correction:** Memory allocation is implicit during `pto_submit_task()` for OUTPUT params. The runtime allocates and fills the buffer address during submission.

2. **~~Add explicit `pto_free()` API~~** — REMOVED
    - ~~Signals "no more references will be added to this buffer"~~
    - **Correction:** No explicit free needed. Buffer lifetime is tied to producer task lifetime. Scope-based lifecycle handles reclamation.

3. **~~Deprecate scope-based lifecycle~~** — RESTORED
    - ~~Scope is no longer needed~~
    - **Correction:** Scope (`pto_scope_begin`/`pto_scope_end`) is the primary lifecycle mechanism. Fanout initialized to `scope_depth` provides automatic memory reclamation.

4. **Two sources of buffer addresses for INCORE calls** — SIMPLIFIED
    - ~~Runtime-allocated: via `pto_alloc()`~~
    - **Correction:** Only external (params from caller) and implicit (OUTPUT allocated during submit). No separate `pto_alloc()` call.

5. **~~Buffer lifecycle needs its own consumer tracking~~** — REMOVED
    - ~~Explicitly allocated buffers require buffer-level reference counting~~
    - **Correction:** Buffer lifetime = task lifetime. When all consumers finish and all scopes exit, the producer task becomes CONSUMED and its packed output buffer is freed. No separate buffer ref count.

6. **Version control for in-place updates** — KEPT
    - Support in-place tensor updates at the same address with different versions
    - API: `pto_version_inc(tensor)` returns new versioned handle (SSA-style)
    - Write to version `v` waits for all reads from version `v-1` to complete
    - Read from version `v` waits for writes to version `v` to complete (region-dependent)
    - **Clarification:** INOUT params create a dependency (like INPUT) but do NOT register as a new producer in TensorMap.

7. **Strided tensor descriptors for TensorMap** — KEPT
    - Need strided access pattern: `(addr, start_offset, strides[], repeats[], n_dims)`
    - Enables arbitrary regular memory layouts (tiled, transposed, sliced)

    ```c++
    struct TensorDescriptor {
        uint64_t addr;
        uint64_t start_offset;
        uint64_t strides[RUNTIME_MAX_TENSOR_DIMS];
        uint64_t repeats[RUNTIME_MAX_TENSOR_DIMS];
        int n_dims;
    };
    ```

8. **TensorMap overlap judgment** — KEPT
    - Multiple strategies from fastest/low-accuracy (BoundingBox) to slowest/high-accuracy (StridedExact)
    - Each strategy is assigned per tensor; overlap check uses the common level

## Correction Summary

| Original Divergence | Status | Correction |
|---------------------|--------|------------|
| §1 Explicit `pto_alloc()` | REMOVED | Allocation is implicit during `pto_submit_task()` |
| §2 Explicit `pto_free()` | REMOVED | No explicit free; scope + task lifetime manages buffers |
| §3 Deprecate scope | RESTORED | Scope is the primary lifecycle mechanism |
| §4 Two buffer sources | SIMPLIFIED | Only external (params) and implicit (OUTPUT) |
| §5 Buffer-level ref counting | REMOVED | Buffer lifetime = task lifetime (task fanout + scope) |
| §6 Version control | KEPT | Added INOUT param type (dependency, not producer) |
| §7 Strided descriptors | KEPT | No changes |
| §8 Overlap judgment | KEPT | No changes |

## Core Design Principle

Buffer lifetime is tied to its producer task's lifetime. When a task becomes CONSUMED
(all consumers done + all scopes exited), its packed output buffer is implicitly freed.
No separate buffer-level reference counting is needed.

