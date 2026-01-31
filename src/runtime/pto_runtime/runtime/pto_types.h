/**
 * PTO Types - Data structures for PTO runtime extensions
 *
 * Adds strided tensor descriptors, buffer handles, and overlap strategies
 * on top of the base types in pto_runtime.h.
 *
 * Key additions:
 * - PTOTensorDescriptor: Strided tensor descriptor for non-contiguous tiles
 * - PTOBufferHandle: Buffer with version tracking for in-place updates
 * - PTOOverlapStrategy: Trade-off between speed and accuracy for dependency detection
 *
 * Phase 1: Header-only, no behavior change.
 */

#ifndef PTO_TYPES_H
#define PTO_TYPES_H

#include "pto_runtime.h"

// =============================================================================
// Configuration
// =============================================================================

#ifndef PTO_MAX_TENSOR_DIMS
#define PTO_MAX_TENSOR_DIMS 8
#endif

// =============================================================================
// Overlap Judgment Strategies
// =============================================================================

/**
 * Overlap strategy for TensorMap dependency detection
 *
 * Trade-off between speed and accuracy:
 * - BoundingBox:  Fast O(d) check, may produce false-positive dependencies
 * - StridedExact: Slow element-by-element check, no false-positives
 *
 * See: divergence-to-original-orchestration.md §7
 */
enum PTOOverlapStrategy : int32_t {
    PTO_OVERLAP_BOUNDING_BOX  = 0,  // Fast: (addr, total_size) only, may false-positive
    PTO_OVERLAP_STRIDED_EXACT = 1,  // Slow: element-by-element comparison, no false-positive
};

// =============================================================================
// Strided Tensor Descriptor
// =============================================================================

/**
 * Strided Tensor Descriptor for TensorMap
 *
 * Supports non-contiguous tiles with arbitrary strides and repeats per dimension.
 * Format: (addr, start_offset, strides[], repeats[], n_dims)
 *
 * Example - contiguous 1D buffer of 4096 bytes:
 *   addr=0x1000, start_offset=0, strides=[1], repeats=[4096], n_dims=1
 *
 * Example - 2D tile (32x64) with stride 128:
 *   addr=0x1000, start_offset=0, strides=[128, 1], repeats=[32, 64], n_dims=2
 *
 * See: divergence-to-original-orchestration.md §6
 */
struct PTOTensorDescriptor {
    uint64_t addr;                            // Base address in GM
    uint64_t start_offset;                    // Starting offset from addr
    uint64_t strides[PTO_MAX_TENSOR_DIMS];    // Stride per dimension
    uint64_t repeats[PTO_MAX_TENSOR_DIMS];    // Elements per dimension
    int32_t n_dims;                           // Number of dimensions
    PTOOverlapStrategy strategy;              // Overlap judgment strategy
};

// =============================================================================
// Buffer Handle
// =============================================================================

/**
 * Buffer Handle returned by pto_alloc()
 *
 * Supports versioning for in-place updates (SSA-style):
 * - Write to version v waits for all reads from version v-1
 * - Read from version v waits for writes to version v to complete
 *
 * Reference counting is at buffer level, independent of task fanout.
 *
 * See: divergence-to-original-orchestration.md §5, §6
 */
struct PTOBufferHandle {
    uint64_t addr;           // Device memory address
    int32_t size;            // Buffer size in bytes
    int32_t version;         // Version number (for in-place updates)
    int32_t ref_count;       // Buffer-level reference count (independent of task fanout)
};

#endif // PTO_TYPES_H