# `use_ring_allocation_` Removal And Ring Buffer By Default

The main purpose is as the title.

## Current State Analysis

### Definition
- **Location**: [runtime.h:186](src/runtime/orch_build_graph/runtime/runtime.h#L186)
- **Type**: `bool use_ring_allocation_ = false`
- **Purpose**: Guards ring-based allocation, enabled only after `pto_init_rings()` is called

### Lifecycle
1. **Reset** ([runtime.cpp:49](src/runtime/orch_build_graph/runtime/runtime.cpp#L49)): Set to `false` in `pto_reset()`
2. **Activation** ([runtime.cpp:301](src/runtime/orch_build_graph/runtime/runtime.cpp#L301)): Set to `true` in `pto_init_rings()` after ring buffers are initialized

### Usage Points (3 locations)

| Location | Line | Ring Mode (`true`) | Legacy Mode (`false`) |
|----------|------|--------------------|-----------------------|
| `add_task()` | [runtime.cpp:69](src/runtime/orch_build_graph/runtime/runtime.cpp#L69) | Uses `task_ring_alloc()` with back-pressure | Simple array bounds check against `RUNTIME_MAX_TASKS` |
| `pto_add_task()` | [runtime.cpp:389](src/runtime/orch_build_graph/runtime/runtime.cpp#L389) | Packed buffer allocation via `heap_ring_alloc()` | Individual `device_malloc()` per OUTPUT buffer |
| `pto_add_task()` | [runtime.cpp:461](src/runtime/orch_build_graph/runtime/runtime.cpp#L461) | Updates `shared_header_.current_task_index` and `heap_top` | No shared header updates |

## Example Analysis: `orch_build_graph_example`

The example at [orch_example_orch.cpp:128](examples/orch_build_graph_example/kernels/orchestration/orch_example_orch.cpp#L128) currently uses:

```cpp
runtime->pto_init();  // Only basic init, NO pto_init_rings()
```

This means the example is running in **legacy mode** (not ring allocation). After this change, the example will automatically use ring allocation without any code changes needed.

## Current Call Patterns

| Pattern | Mode | Used By |
|---------|------|---------|
| `pto_init()` only | Legacy | `orch_build_graph_example`, `test_dep_list_pool`, `test_scope_end`, `test_state_machine` |
| `pto_init()` + `pto_init_rings()` | Ring | `test_shared_header`, `test_ring_buffers` |

## Implementation Plan

### Step 1: Move Ring Initialization into `pto_init()`
- Move the ring buffer initialization logic from `pto_init_rings()` into `pto_init()`
- This ensures rings are always ready after basic init

### Step 2: Remove `use_ring_allocation_` Flag
- Delete the member variable declaration from `runtime.h`
- Remove all conditional checks (`if (use_ring_allocation_)`)
- Keep only the ring-based code paths

### Step 3: Make `pto_init_rings()` a No-Op
- Keep the function signature for backward compatibility
- Make it an empty function body
- This allows existing test code to compile without changes

### Step 4: Remove Legacy Allocation Code
- Delete the `else` branches that use individual `device_malloc()` calls
- Delete the simple array bounds check in `add_task()`

### Step 5: Update Tests
- Remove `pto_init_rings()` calls from tests (optional cleanup)
- **Remove Test 6** in `test_ring_buffers.cpp` ("Legacy Allocation Mode") since legacy mode will no longer exist

## Files to Modify

| File | Changes |
|------|---------|
| [runtime.h:186](src/runtime/orch_build_graph/runtime/runtime.h#L186) | Remove `use_ring_allocation_` member |
| [runtime.cpp:267](src/runtime/orch_build_graph/runtime/runtime.cpp#L267) | Move ring init from `pto_init_rings()` into `pto_init()` |
| [runtime.cpp:69,389,461](src/runtime/orch_build_graph/runtime/runtime.cpp#L69) | Remove conditionals, keep ring-only paths |
| [runtime.cpp:285](src/runtime/orch_build_graph/runtime/runtime.cpp#L285) | Make `pto_init_rings()` empty (no-op) |

### Test Files (Optional Cleanup)

| File | Action |
|------|--------|
| [test_ring_buffers.cpp:381-420](src/runtime/orch_build_graph/tests/test_ring_buffers.cpp#L381-L420) | **Remove Test 6** (Legacy Allocation Mode) |
| [test_shared_header.cpp](src/runtime/orch_build_graph/tests/test_shared_header.cpp) | Remove `pto_init_rings()` calls (8 locations) |
| [test_ring_buffers.cpp](src/runtime/orch_build_graph/tests/test_ring_buffers.cpp) | Remove `pto_init_rings()` calls (7 locations) |

## Risk Assessment

- **Low Risk**: Ring allocation is already the intended production path
- **Example Impact**: `orch_build_graph_example` will automatically upgrade from legacy to ring mode
- **Test Impact**: Test 6 must be removed; other tests will work unchanged (no-op `pto_init_rings()`)
- **Rollback**: Can revert if issues arise since this is a simplification, not new functionality