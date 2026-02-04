/**
 * Ring Buffer Infrastructure for Orchestration Build Graph Runtime
 *
 * Provides O(1) allocation with implicit deallocation and zero fragmentation.
 * All dynamic data uses ring buffers - no malloc/free needed.
 *
 * Key Properties:
 * - FIFO allocation order matches task lifecycle (tasks created and retired in order)
 * - Allocation: O(1) bump pointer
 * - Deallocation: Implicit when tail advances (scheduler reclaims memory)
 * - Back-pressure: Orchestrator stalls when ring is full, waiting for Scheduler
 * - Zero fragmentation: Contiguous FIFO allocation
 *
 * Ring Buffers:
 * 1. Task Ring: Allocates PTOTaskDescriptor slots
 * 2. Heap Ring: Allocates device memory buffers (GM heap)
 *
 * Design Decision:
 * - Never split allocation across ring boundary (wrap-around rule)
 * - Skip remaining space and allocate from beginning instead
 * - Prevents complex wraparound handling in kernels
 */

#ifndef ORCH_BUILD_GRAPH_RING_BUFFER_H
#define ORCH_BUILD_GRAPH_RING_BUFFER_H

#include "pto_runtime.h"

// =============================================================================
// Task Ring - Allocates PTOTaskDescriptor slots
// =============================================================================

/**
 * Task Ring - Fixed-size ring buffer for task descriptors
 *
 * Orchestrator writes to head (current_task_index in shared header)
 * Scheduler advances tail (last_task_alive in shared header)
 *
 * Invariant: (head - tail) < size (leave 1 slot empty to distinguish full/empty)
 *
 * Memory layout:
 *   [CONSUMED | LIVE TASKS | FREE SPACE]
 *    ^tail                   ^head
 */
struct TaskRing {
    PTOTaskDescriptor* base;  // Base pointer to task array
    int32_t size;             // PTO_TASK_WINDOW_SIZE (must be power of 2)
    int32_t head;             // Next slot to allocate (Orchestrator local)
};

/**
 * Allocate a task slot from the ring
 *
 * May stall if ring is full, waiting for Scheduler to consume tasks and advance tail.
 *
 * @param ring      Task ring state
 * @param tail_ptr  Pointer to shared tail (last_task_alive)
 * @return Slot index (0 to size-1)
 */
static inline int32_t task_ring_alloc(TaskRing* ring, volatile int32_t* tail_ptr) {
    while (true) {
        int32_t head = ring->head;
        int32_t tail = *tail_ptr;

        // Calculate used slots (handle wraparound)
        int32_t used = (head - tail + ring->size) % ring->size;

        // Check if ring has space (leave 1 slot empty)
        if (used < ring->size - 1) {
            int32_t slot = ring_index(head, ring->size);
            ring->head = head + 1;
            return slot;
        }

        // Ring full - stall and wait for Scheduler to advance tail
        // In real implementation, could yield or sleep here
    }
}

/**
 * Get pointer to task descriptor by absolute task ID
 *
 * @param ring     Task ring state
 * @param task_id  Absolute task ID (may be > size)
 * @return Pointer to task descriptor
 */
static inline PTOTaskDescriptor* task_ring_get(TaskRing* ring, int32_t task_id) {
    int32_t slot = ring_index(task_id, ring->size);
    return &ring->base[slot];
}

// =============================================================================
// Heap Ring - Allocates device memory buffers
// =============================================================================

/**
 * Heap Ring - Ring buffer for device memory (GM heap)
 *
 * Orchestrator allocates from top (heap_top in shared header)
 * Scheduler advances tail (heap_tail in shared header) as buffers are freed
 *
 * Allocation Strategy:
 * - Never split buffer across ring boundary
 * - If insufficient space at end, wrap to beginning
 * - Skip remaining space (internal fragmentation, but simple and safe)
 *
 * Memory layout when top >= tail:
 *   [FREE | CONSUMED | LIVE BUFFERS | FREE]
 *         ^tail      ^top
 *
 * Memory layout when top < tail (wrapped):
 *   [LIVE BUFFERS | FREE | CONSUMED | LIVE BUFFERS]
 *                 ^top   ^tail
 */
struct HeapRing {
    char* base;       // Base pointer to heap buffer
    int32_t size;     // PTO_HEAP_SIZE
    int32_t top;      // Next allocation offset (Orchestrator local)
};

/**
 * Allocate buffer from heap ring
 *
 * May stall if insufficient space, waiting for Scheduler to reclaim memory.
 * Never splits allocation across ring boundary - wraps to beginning instead.
 *
 * @param ring       Heap ring state
 * @param alloc_size Size in bytes (will be aligned to PTO_ALIGNMENT)
 * @param tail_ptr   Pointer to shared tail (heap_tail)
 * @return Pointer to allocated buffer
 */
static inline void* heap_ring_alloc(HeapRing* ring, int32_t alloc_size,
                                   volatile int32_t* tail_ptr) {
    // Align to cache line for DMA efficiency
    alloc_size = ALIGN_UP(alloc_size, PTO_ALIGNMENT);

    while (true) {
        int32_t tail = *tail_ptr;
        int32_t top = ring->top;

        if (top >= tail) {
            // Case 1: [....tail====top......]
            // Free space: [0, tail) and [top, size)

            // Try allocating at end
            int32_t space_at_end = ring->size - top;
            if (space_at_end >= alloc_size) {
                void* ptr = ring->base + top;
                ring->top = top + alloc_size;
                return ptr;
            }

            // Not enough space at end, try wrapping to beginning
            if (tail > alloc_size) {
                void* ptr = ring->base;
                ring->top = alloc_size;
                return ptr;
            }

            // Insufficient space at both ends - stall
        } else {
            // Case 2: [====top....tail=====]
            // Free space: [top, tail)

            int32_t gap = tail - top;
            if (gap >= alloc_size) {
                void* ptr = ring->base + top;
                ring->top = top + alloc_size;
                return ptr;
            }

            // Insufficient space - stall
        }

        // Ring full - stall and wait for Scheduler to advance tail
        // In real implementation, could yield or sleep here
    }
}

/**
 * Get offset of pointer within heap ring
 *
 * @param ring Heap ring state
 * @param ptr  Pointer within ring
 * @return Offset in bytes from base
 */
static inline int32_t heap_ring_offset(HeapRing* ring, void* ptr) {
    return (char*)ptr - ring->base;
}

/**
 * Get pointer from offset within heap ring
 *
 * @param ring   Heap ring state
 * @param offset Offset in bytes from base
 * @return Pointer within ring
 */
static inline void* heap_ring_ptr(HeapRing* ring, int32_t offset) {
    return ring->base + offset;
}

// =============================================================================
// Initialization Helpers
// =============================================================================

/**
 * Initialize task ring
 *
 * @param ring Task ring to initialize
 * @param base Base pointer to task descriptor array
 * @param size Number of slots (PTO_TASK_WINDOW_SIZE)
 */
static inline void task_ring_init(TaskRing* ring, PTOTaskDescriptor* base, int32_t size) {
    ring->base = base;
    ring->size = size;
    ring->head = 0;
}

/**
 * Initialize heap ring
 *
 * @param ring Heap ring to initialize
 * @param base Base pointer to heap buffer
 * @param size Size in bytes (PTO_HEAP_SIZE)
 */
static inline void heap_ring_init(HeapRing* ring, char* base, int32_t size) {
    ring->base = base;
    ring->size = size;
    ring->top = 0;
}

#endif  // ORCH_BUILD_GRAPH_RING_BUFFER_H
