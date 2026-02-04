/**
 * Tensor Descriptor - Strided tensor representation and overlap detection
 *
 * Provides:
 * - PTOTensorDescriptor: Strided tensor descriptor for non-contiguous tiles
 * - Overlap detection strategies (BoundingBox vs StridedExact)
 * - Fast overlap checking for automatic dependency tracking
 *
 * See: divergence-to-original-orchestration.md §6, §7
 */

#ifndef ORCH_BUILD_GRAPH_TENSOR_DESCRIPTOR_H
#define ORCH_BUILD_GRAPH_TENSOR_DESCRIPTOR_H

#include <stdint.h>

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
enum class PTOOverlapStrategy : int32_t {
    BOUNDING_BOX  = 0,  // Fast: (addr, total_size) only, may false-positive
    STRIDED_EXACT = 1,  // Slow: element-by-element comparison, no false-positive
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
// Overlap Detection Functions
// =============================================================================

/**
 * Determine if two tensors overlap using appropriate strategy
 *
 * Picks the most accurate strategy based on both tensors' preferences.
 * If either tensor requests exact checking, uses exact; otherwise uses bounding box.
 *
 * Implementation strategies:
 * - BoundingBox:  Fast O(d) check, may produce false-positives for sparse tensors
 * - StridedExact: Slow element-by-element comparison, no false-positives
 *
 * See: divergence-to-original-orchestration.md §7
 *
 * @param a First tensor descriptor
 * @param b Second tensor descriptor
 * @return true if tensors overlap
 */
bool tensors_overlap(const PTOTensorDescriptor* a, const PTOTensorDescriptor* b);

#endif  // ORCH_BUILD_GRAPH_TENSOR_DESCRIPTOR_H