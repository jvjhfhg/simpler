/**
 * Dependency List Pool for Orchestration Build Graph Runtime
 *
 * Provides efficient linked list storage for task dependencies (fanin/fanout).
 * Uses ring buffer allocation with implicit deallocation - no explicit free operations.
 *
 * Key Features:
 * - O(1) prepend operation for building dependency lists
 * - Offset-based encoding (0 = empty list, non-zero = index + 1)
 * - Implicit memory reclamation when task ring wraps
 * - Lock-free allocation (single Orchestrator thread)
 *
 * Usage Pattern:
 * 1. Orchestrator prepends entries during task submission
 * 2. Scheduler traverses lists during task completion
 * 3. Memory automatically reclaimed when old tasks are consumed
 *
 * Memory Layout:
 *   [GARBAGE | LIVE ENTRIES | FREE SPACE]
 *             ^implicit_tail  ^top
 *
 * Size Calculation:
 *   PTO_DEP_LIST_POOL_SIZE = TASK_WINDOW_SIZE × AVG_OUTPUTS × AVG_FANOUT
 *   Example: 1024 tasks × 2 outputs × 4 consumers = ~8K entries
 */

#ifndef ORCH_BUILD_GRAPH_DEP_LIST_POOL_H
#define ORCH_BUILD_GRAPH_DEP_LIST_POOL_H

#include "pto_runtime.h"

// =============================================================================
// Dependency List Operations
// =============================================================================

/**
 * Initialize dependency list pool
 *
 * @param pool Dependency list pool to initialize
 * @param base Base pointer to DepListEntry array
 * @param size Number of entries (PTO_DEP_LIST_POOL_SIZE)
 */
static inline void dep_list_pool_init(DepListPool* pool, DepListEntry* base, int32_t size) {
    pool->base = base;
    pool->size = size;
    pool->top = 0;
}

/**
 * Prepend entry to dependency list (O(1) operation)
 *
 * Allocates a new entry from the pool and links it to the current list head.
 * Returns new head offset.
 *
 * Offset Encoding:
 * - 0 means empty list
 * - Non-zero means (array_index + 1)
 * - This allows distinguishing empty list from first entry
 *
 * Example:
 *   int32_t fanout_list = 0;  // Empty initially
 *   fanout_list = dep_list_prepend(pool, fanout_list, task_5);  // [5]
 *   fanout_list = dep_list_prepend(pool, fanout_list, task_3);  // [3, 5]
 *   fanout_list = dep_list_prepend(pool, fanout_list, task_7);  // [7, 3, 5]
 *
 * @param pool         Dependency list pool
 * @param current_head Current list head offset (0 for empty list)
 * @param task_id      Task ID to prepend
 * @return New list head offset
 */
static inline int32_t dep_list_prepend(DepListPool* pool, int32_t current_head, int32_t task_id) {
    // Allocate new entry (bump pointer)
    int32_t new_index = pool->top;
    pool->top = (pool->top + 1) % pool->size;  // Wrap around

    // Initialize entry
    DepListEntry* entry = &pool->base[new_index];
    entry->task_id = task_id;
    entry->next_offset = current_head;  // Link to old head

    // Return new head (encode as index + 1, so 0 means empty)
    return new_index + 1;
}

/**
 * Get entry from offset
 *
 * Converts offset encoding back to pointer.
 * Returns NULL if offset is 0 (empty list).
 *
 * @param pool   Dependency list pool
 * @param offset List offset (0 for empty, or index + 1)
 * @return Pointer to entry, or NULL if offset is 0
 */
static inline DepListEntry* dep_list_get(DepListPool* pool, int32_t offset) {
    if (offset == 0) {
        return nullptr;
    }
    return &pool->base[offset - 1];
}

/**
 * Traverse dependency list
 *
 * Visits each entry in the list via callback function.
 * Useful for Scheduler when processing task completions.
 *
 * Example:
 *   auto callback = [](int32_t task_id, void* ctx) {
 *       printf("Task %d\n", task_id);
 *   };
 *   dep_list_foreach(pool, fanout_list, callback, nullptr);
 *
 * @param pool     Dependency list pool
 * @param head     List head offset
 * @param callback Function to call for each entry
 * @param ctx      User context passed to callback
 */
template<typename Func>
static inline void dep_list_foreach(DepListPool* pool, int32_t head, Func callback, void* ctx) {
    int32_t offset = head;
    while (offset != 0) {
        DepListEntry* entry = dep_list_get(pool, offset);
        callback(entry->task_id, ctx);
        offset = entry->next_offset;
    }
}

/**
 * Count entries in dependency list
 *
 * O(n) operation - use sparingly (mainly for debugging).
 *
 * @param pool Dependency list pool
 * @param head List head offset
 * @return Number of entries in list
 */
static inline int32_t dep_list_count(DepListPool* pool, int32_t head) {
    int32_t count = 0;
    int32_t offset = head;
    while (offset != 0) {
        count++;
        DepListEntry* entry = dep_list_get(pool, offset);
        offset = entry->next_offset;
    }
    return count;
}

/**
 * Check if list is empty
 *
 * @param head List head offset
 * @return true if list is empty (head == 0)
 */
static inline bool dep_list_is_empty(int32_t head) {
    return head == 0;
}

// =============================================================================
// Debug Utilities
// =============================================================================

#ifdef PTO_DEBUG

#include <stdio.h>

/**
 * Print dependency list (debug only)
 *
 * @param pool Dependency list pool
 * @param head List head offset
 * @param name List name for display
 */
static inline void dep_list_print(DepListPool* pool, int32_t head, const char* name) {
    printf("%s: [", name);
    int32_t offset = head;
    bool first = true;
    while (offset != 0) {
        if (!first) printf(", ");
        DepListEntry* entry = dep_list_get(pool, offset);
        printf("%d", entry->task_id);
        offset = entry->next_offset;
        first = false;
    }
    printf("]\n");
}

#endif  // PTO_DEBUG

#endif  // ORCH_BUILD_GRAPH_DEP_LIST_POOL_H
