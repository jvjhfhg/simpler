# Plan: Deprecate PTOBufferHandle::version

**Status**: ✅ Completed
**Date**: 2026-02-05
**Context**: Version tracking has been migrated from PTOBufferHandle to TensorDescriptor

## Background

Version tracking has been migrated from `PTOBufferHandle` to `TensorDescriptor`. The `PTOBufferHandle::version` field and `pto_version_inc()` are now redundant because:

1. **TensorDescriptor owns version tracking**: The `TensorDescriptor` struct has a `version` field that is set by user code when constructing tensors
2. **Dependency logic uses TensorDescriptor::version**: The `is_overlap()` method already uses `TensorDescriptor::version` for dependency buildup
3. **TensorMap stores version from TensorDescriptor**: The version is extracted from `params[i].tensor.version` during registration

Since the dependency buildup logic is fully implemented using `TensorDescriptor::version` inside the overlap check, `PTOBufferHandle::version` and `pto_version_inc()` serve no purpose and can be removed directly.

## Changes Completed

### 1. ✅ Removed version field from PTOBufferHandle

**File**: [src/runtime/orch_build_graph/runtime/pto_types.h](src/runtime/orch_build_graph/runtime/pto_types.h)

Removed `int32_t version` field from `PTOBufferHandle` struct.

### 2. ✅ Removed pto_version_inc() function

**Files**:
- [src/runtime/orch_build_graph/runtime/runtime.h](src/runtime/orch_build_graph/runtime/runtime.h) - Removed declaration
- [src/runtime/orch_build_graph/runtime/runtime.cpp](src/runtime/orch_build_graph/runtime/runtime.cpp) - Removed implementation

### 3. ✅ Removed version initialization in runtime

**File**: [src/runtime/orch_build_graph/runtime/runtime.cpp](src/runtime/orch_build_graph/runtime/runtime.cpp)

Removed lines that initialized `params[i].buffer->version = 0` for OUTPUT buffers.

### 4. ✅ Updated tensormap_insert() call

**File**: [src/runtime/orch_build_graph/runtime/runtime.cpp:474](src/runtime/orch_build_graph/runtime/runtime.cpp#L474)

Changed from `params[i].buffer->version` to `params[i].tensor.version`.

### 5. ✅ Updated test code

Removed `test_preallocated_output()` test that specifically tested `pto_version_inc()`.

Updated all test helper functions to:
- Remove `h.version = 0` initialization in `make_external_handle()` and `make_output_handle()`
- Add `version` parameter to `make_tensor_bbox()` with default value of 0

**Files updated**:
- [test_ring_buffers.cpp](src/runtime/orch_build_graph/tests/test_ring_buffers.cpp)
- [test_state_machine.cpp](src/runtime/orch_build_graph/tests/test_state_machine.cpp)
- [test_scope_end.cpp](src/runtime/orch_build_graph/tests/test_scope_end.cpp)
- [test_shared_header.cpp](src/runtime/orch_build_graph/tests/test_shared_header.cpp)
- [test_dep_list_pool.cpp](src/runtime/orch_build_graph/tests/test_dep_list_pool.cpp)

## Verification

✅ All tests pass:
- test_state_machine: 52 passed
- test_scope_end: 102 passed
- test_ring_buffers: 50 passed
- test_shared_header: 57 passed
- test_dep_list_pool: 46 passed

**Total: 307 tests passed, 0 failed**

Dependency tracking via `TensorDescriptor::version` in overlap detection works correctly.

## How TensorDescriptor::version Affects Dependency Buildup

Version tracking enables SSA-style in-place updates with automatic dependency detection:

**Overlap Detection Logic** ([tensor_descriptor.cpp:396-405](src/runtime/orch_build_graph/runtime/tensor_descriptor.cpp#L396-L405)):
```cpp
bool TensorDescriptor::is_overlap(const TensorDescriptor& pre_task_output) const {
    if (!is_same_memref(pre_task_output)) {
        return false;  // Different buffers, no overlap
    }
    debug_assert(version >= pre_task_output.version);
    if (version > pre_task_output.version) {
        return true;   // Same buffer, different version → BARRIER (always dependency)
    }
    // Same version: check spatial overlap...
}
```

**Key Behavior**:
- **Different versions** (`input.version > output.version`): **BARRIER** - always creates dependency regardless of spatial overlap, enabling in-place updates
- **Same version**: Checks spatial overlap (bounding box intersection) to determine dependency
- **Assertion**: Input version must be ≥ output version (enforces SSA ordering)

**Example** (from orch_example_orch.cpp):
```cpp
// Task 4: x_v0 = a + 1 (version=0)
make_output_param(&dev_x_v0, BYTES, 0);

// Task 5: x_v1 = x_v0 + 1 (read v0, write v1)
make_input_param(&dev_x_v0, BYTES, 0);   // Read from version 0
make_output_param(&dev_x_v1, BYTES, 1);  // Write to version 1
// → BARRIER: Task 5 always waits for Task 4 (version 1 > version 0)
```

This replaces the old `pto_version_inc()` API with explicit version management in TensorDescriptor.
