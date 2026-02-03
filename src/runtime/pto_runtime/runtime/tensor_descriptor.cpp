/**
 * Tensor Descriptor - Overlap detection implementations
 */

#include "tensor_descriptor.h"

// =============================================================================
// Internal Helper Functions
// =============================================================================

/**
 * BoundingBox Overlap Check - Fast O(d) check
 *
 * Computes the bounding box (min/max byte offset) for each tensor and
 * checks if they overlap. May produce false-positives for sparse tensors.
 *
 * Internal helper function - not exposed in public API.
 */
static bool overlap_bounding_box(const PTOTensorDescriptor* a, const PTOTensorDescriptor* b) {
    // Different base addresses never overlap
    if (a->addr != b->addr) return false;

    // Compute bounding box for tensor a
    uint64_t a_start = a->start_offset;
    uint64_t a_end = a->start_offset;
    for (int d = 0; d < a->n_dims; d++) {
        a_end += a->strides[d] * (a->repeats[d] - 1);
    }

    // Compute bounding box for tensor b
    uint64_t b_start = b->start_offset;
    uint64_t b_end = b->start_offset;
    for (int d = 0; d < b->n_dims; d++) {
        b_end += b->strides[d] * (b->repeats[d] - 1);
    }

    // Check for overlap: [a_start, a_end] ∩ [b_start, b_end] ≠ ∅
    return (a_start <= b_end) && (b_start <= a_end);
}

/**
 * StridedExact Overlap Check - Slow element-by-element comparison
 *
 * Compares individual elements accessed by each tensor to detect exact
 * overlaps. No false-positives, but O(n*m) complexity.
 *
 * Internal helper function - not exposed in public API.
 */
static bool overlap_strided_exact(const PTOTensorDescriptor* a, const PTOTensorDescriptor* b) {
    // Different base addresses never overlap
    if (a->addr != b->addr) return false;

    // TODO: Implement element-by-element comparison for exact overlap detection
    // For now, fall back to bounding box (conservative approximation)
    return overlap_bounding_box(a, b);
}

// =============================================================================
// Public API
// =============================================================================

bool tensors_overlap(const PTOTensorDescriptor* a, const PTOTensorDescriptor* b) {
    // Use the more conservative (accurate) strategy of the two
    // Lower enum value = faster but less accurate
    PTOOverlapStrategy common = (a->strategy > b->strategy) ? a->strategy : b->strategy;

    switch (common) {
        case PTOOverlapStrategy::STRIDED_EXACT:
            return overlap_strided_exact(a, b);
        case PTOOverlapStrategy::BOUNDING_BOX:
        default:
            return overlap_bounding_box(a, b);
    }
}
