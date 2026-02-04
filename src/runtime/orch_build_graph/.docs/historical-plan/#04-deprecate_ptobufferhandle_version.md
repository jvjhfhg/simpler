# Plan: Deprecate PTOBufferHandle::version

**Status**: Ready to implement
**Date**: 2026-02-04
**Context**: Version tracking has been migrated from PTOBufferHandle to TensorDescriptor

## Background

Version tracking has been migrated from `PTOBufferHandle` to `TensorDescriptor`. The `PTOBufferHandle::version` field and `pto_version_inc()` are now redundant because:

1. **TensorDescriptor owns version tracking**: The `TensorDescriptor` struct has a `version` field that is set by user code when constructing tensors
2. **Dependency logic uses TensorDescriptor::version**: The `is_overlap()` method already uses `TensorDescriptor::version` for dependency buildup
3. **TensorMap stores version from TensorDescriptor**: The version is extracted from `params[i].tensor.version` during registration

Since the dependency buildup logic is fully implemented using `TensorDescriptor::version` inside the overlap check, `PTOBufferHandle::version` and `pto_version_inc()` serve no purpose and can be removed directly.

## Changes Required

### 1. Remove version field from PTOBufferHandle

**File**: `src/runtime/orch_build_graph/runtime/pto_types.h`

```cpp
// Before
struct PTOBufferHandle {
    uint64_t addr;
    int32_t size;
    int32_t version;  // For in-place updates
};

// After
struct PTOBufferHandle {
    uint64_t addr;
    int32_t size;
};
```

### 2. Remove pto_version_inc() function

**Files**:
- `src/runtime/orch_build_graph/runtime/pto_runtime.h` - Remove declaration
- `src/runtime/orch_build_graph/runtime/runtime.cpp` - Remove implementation

### 3. Remove version initialization in runtime

**File**: `src/runtime/orch_build_graph/runtime/runtime.cpp`

Remove lines that initialize `params[i].buffer->version = 0` for OUTPUT buffers.

### 4. Update tensormap_insert() call

**File**: `src/runtime/orch_build_graph/runtime/runtime.cpp`

Already uses `params[i].tensor.version` - verify this is correct.

### 5. Update test code

Remove any tests that specifically test `pto_version_inc()` or `PTOBufferHandle::version`.

## Verification

1. Build and run all tests
2. Verify dependency tracking still works via `TensorDescriptor::version` in overlap detection
