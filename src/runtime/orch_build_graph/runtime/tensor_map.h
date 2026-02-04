/**
 * TensorMap for Orchestration Build Graph Runtime
 *
 * Provides automatic dependency tracking via memory region lookup with
 * strided tensor support and overlap strategies.
 *
 * Key Features:
 * - Hash-based lookup by tensor base address
 * - Strided tensor descriptors for non-contiguous tiles (see tensor_descriptor.h)
 * - Configurable overlap strategies (BoundingBox vs StridedExact)
 * - Ring buffer allocation with implicit deallocation
 *
 * Overlap Strategies:
 * - BoundingBox:  O(d) check, fast but may produce false-positive dependencies
 * - StridedExact: Element-by-element comparison, slow but no false-positives
 *
 * See: divergence-to-original-orchestration.md §6, §7
 */

#ifndef ORCH_BUILD_GRAPH_TENSOR_MAP_H
#define ORCH_BUILD_GRAPH_TENSOR_MAP_H

#include "pto_types.h"
#include <stdint.h>

// =============================================================================
// TensorMap Entry
// =============================================================================

/**
 * TensorMap Entry - stores full tensor descriptor for overlap checking
 *
 * Each entry records a producer task's output tensor, enabling consumers
 * to find dependencies by looking up overlapping memory regions.
 */
struct TensorMapEntry {
    PTOTensorDescriptor tensor;   // Full strided tensor descriptor
    int32_t producer_task_id;     // Task that produces this tensor
    int32_t version;              // For in-place update tracking
    int32_t next_in_bucket;       // Linked list within hash bucket (1-indexed, 0=end)
};

// =============================================================================
// TensorMap Structure
// =============================================================================

/**
 * TensorMap - Hash table for producer-consumer dependency tracking
 *
 * Uses open hashing (chained buckets) with ring buffer allocation.
 * Entries are automatically reclaimed when old tasks are consumed.
 */
struct TensorMap {
    TensorMapEntry* pool;         // Ring buffer of entries
    int32_t pool_size;            // PTO_TENSORMAP_POOL_SIZE
    int32_t pool_head;            // Next slot to allocate (wraps around)
    int32_t* buckets;             // Hash bucket heads (1-indexed, 0=empty)
    int32_t num_buckets;          // PTO_TENSORMAP_NUM_BUCKETS
};

// =============================================================================
// Initialization
// =============================================================================

/**
 * Initialize TensorMap
 *
 * @param tm          TensorMap to initialize
 * @param pool        Array of TensorMapEntry (size = pool_size)
 * @param pool_size   Number of entries in pool
 * @param buckets     Array of bucket heads (size = num_buckets)
 * @param num_buckets Number of hash buckets
 */
static inline void tensormap_init(TensorMap* tm, TensorMapEntry* pool, int32_t pool_size,
                                  int32_t* buckets, int32_t num_buckets) {
    tm->pool = pool;
    tm->pool_size = pool_size;
    tm->pool_head = 0;
    tm->buckets = buckets;
    tm->num_buckets = num_buckets;

    // Clear all buckets
    for (int i = 0; i < num_buckets; i++) {
        buckets[i] = 0;
    }
}

// =============================================================================
// Hash Function
// =============================================================================

/**
 * Hash tensor address to bucket index
 *
 * Uses address >> 6 to account for cache line alignment (64 bytes).
 *
 * @param tm   TensorMap
 * @param addr Base address of tensor
 * @return Bucket index (0 to num_buckets-1)
 */
static inline int32_t tensormap_hash(TensorMap* tm, uint64_t addr) {
    return (addr >> 6) % tm->num_buckets;
}

// =============================================================================
// Insert Operation
// =============================================================================

/**
 * Insert tensor into TensorMap
 *
 * Records a producer task's output tensor for future dependency lookups.
 * Uses ring buffer allocation - old entries are implicitly overwritten.
 *
 * @param tm               TensorMap
 * @param tensor           Tensor descriptor to insert
 * @param producer_task_id Task ID that produces this tensor
 * @param version          Version number for in-place update tracking
 */
static inline void tensormap_insert(TensorMap* tm, const PTOTensorDescriptor* tensor,
                                    int32_t producer_task_id, int32_t version) {
    // Compute hash bucket
    int32_t bucket = tensormap_hash(tm, tensor->addr);

    // Allocate entry from ring buffer (bump pointer)
    int32_t slot = tm->pool_head;
    tm->pool_head = (tm->pool_head + 1) % tm->pool_size;

    // Initialize entry
    TensorMapEntry* entry = &tm->pool[slot];
    entry->tensor = *tensor;
    entry->producer_task_id = producer_task_id;
    entry->version = version;

    // Link into bucket (prepend to list)
    entry->next_in_bucket = tm->buckets[bucket];
    tm->buckets[bucket] = slot + 1;  // 1-indexed (0 means empty)
}

// =============================================================================
// Lookup Operation
// =============================================================================

/**
 * Lookup producer for overlapping tensor
 *
 * Searches the TensorMap for a tensor that overlaps with the query tensor.
 * Skips stale entries (producer_task_id < last_task_alive).
 *
 * @param tm              TensorMap
 * @param tensor          Query tensor descriptor
 * @param last_task_alive Oldest non-consumed task (for staleness check)
 * @return Producer task_id if found, -1 if no overlap
 */
static inline int32_t tensormap_lookup(TensorMap* tm, const PTOTensorDescriptor* tensor,
                                       int32_t last_task_alive) {
    // Compute hash bucket
    int32_t bucket = tensormap_hash(tm, tensor->addr);

    // Traverse bucket chain
    int32_t entry_idx = tm->buckets[bucket];
    while (entry_idx != 0) {
        TensorMapEntry* entry = &tm->pool[entry_idx - 1];

        // Skip stale entries (task already consumed)
        if (entry->producer_task_id < last_task_alive) {
            entry_idx = entry->next_in_bucket;
            continue;
        }

        // Check for overlap using appropriate strategy
        if (tensors_overlap(&entry->tensor, tensor)) {
            return entry->producer_task_id;
        }

        entry_idx = entry->next_in_bucket;
    }

    return -1;  // No overlapping producer found
}

/**
 * Lookup all producers for overlapping tensor
 *
 * Variant that finds ALL overlapping producers (for multi-producer scenarios).
 * Calls callback for each overlapping entry.
 *
 * @param tm              TensorMap
 * @param tensor          Query tensor descriptor
 * @param last_task_alive Oldest non-consumed task
 * @param callback        Function called for each overlapping producer
 * @param ctx             User context passed to callback
 */
template<typename Func>
static inline void tensormap_lookup_all(TensorMap* tm, const PTOTensorDescriptor* tensor,
                                        int32_t last_task_alive, Func callback, void* ctx) {
    // Compute hash bucket
    int32_t bucket = tensormap_hash(tm, tensor->addr);

    // Traverse bucket chain
    int32_t entry_idx = tm->buckets[bucket];
    while (entry_idx != 0) {
        TensorMapEntry* entry = &tm->pool[entry_idx - 1];

        // Skip stale entries
        if (entry->producer_task_id >= last_task_alive) {
            // Check for overlap
            if (tensors_overlap(&entry->tensor, tensor)) {
                callback(entry->producer_task_id, entry->version, ctx);
            }
        }

        entry_idx = entry->next_in_bucket;
    }
}

// =============================================================================
// Debug Utilities
// =============================================================================

#ifdef PTO_DEBUG

#include <stdio.h>

/**
 * Print TensorMap statistics (debug only)
 */
static inline void tensormap_print_stats(TensorMap* tm) {
    int32_t total_entries = 0;
    int32_t max_chain = 0;

    for (int32_t b = 0; b < tm->num_buckets; b++) {
        int32_t chain_len = 0;
        int32_t entry_idx = tm->buckets[b];
        while (entry_idx != 0) {
            chain_len++;
            TensorMapEntry* entry = &tm->pool[entry_idx - 1];
            entry_idx = entry->next_in_bucket;
        }
        total_entries += chain_len;
        if (chain_len > max_chain) {
            max_chain = chain_len;
        }
    }

    printf("TensorMap stats: %d entries, max chain length = %d, pool_head = %d\n",
           total_entries, max_chain, tm->pool_head);
}

#endif  // PTO_DEBUG

#endif  // ORCH_BUILD_GRAPH_TENSOR_MAP_H
