/**
 * PTO Types - Data structures for PTO runtime extensions
 *
 * Standalone header defining PTO-specific types for:
 * - PTOTensorDescriptor: Strided tensor descriptor for non-contiguous tiles
 * - PTOBufferHandle: Buffer with version tracking for in-place updates
 * - PTOOverlapStrategy: Trade-off between speed and accuracy for dependency detection
 * - PTOParam: Parameter descriptor for pto_submit_task API
 *
 * This header is independent of pto_runtime.h to allow inclusion from runtime.h
 * without type conflicts (Handshake, TensorPair, HostApi).
 */

#ifndef PTO_TYPES_H
#define PTO_TYPES_H

#include <stdint.h>

// =============================================================================
// Configuration
// =============================================================================

#ifndef PTO_MAX_TENSOR_DIMS
#define PTO_MAX_TENSOR_DIMS 8
#endif

#ifndef PTO_TENSORMAP_POOL_SIZE
#define PTO_TENSORMAP_POOL_SIZE 4096
#endif

#ifndef PTO_TENSORMAP_NUM_BUCKETS
#define PTO_TENSORMAP_NUM_BUCKETS 1024
#endif

// =============================================================================
// Worker Types
// =============================================================================

/**
 * Worker types for heterogeneous scheduling
 *
 * Tasks are routed to different ready queues based on worker_type:
 * - PTO_WORKER_CUBE:   AICore-CUBE (matrix ops, convolution)
 * - PTO_WORKER_VECTOR: AICore-VECTOR (element-wise ops, activation)
 *
 * Note: AICPU is not a worker type - AICPU threads act as schedulers that
 * dispatch tasks to AICore workers.
 */
enum PTOWorkerType : int32_t {
    PTO_WORKER_CUBE    = 0,  // AICore-CUBE
    PTO_WORKER_VECTOR  = 1,  // AICore-VECTOR
    PTO_NUM_WORKER_TYPES = 2
};

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

// =============================================================================
// Parameter Types (for pto_submit_task API)
// =============================================================================

/**
 * Parameter Type - Distinguishes inputs from outputs
 */
enum PTOParamType : int32_t {
    PTO_PARAM_INPUT  = 0,  // Read-only input buffer
    PTO_PARAM_OUTPUT = 1,  // Write-only output buffer
    PTO_PARAM_SCALAR = 2   // Raw scalar value (no buffer, no dependency tracking)
};

/**
 * Parameter Descriptor for pto_submit_task
 *
 * Each parameter carries a full tensor descriptor for automatic
 * dependency detection via TensorMap overlap checking.
 *
 * Example:
 *   PTOParam params[] = {
 *       {PTO_PARAM_INPUT,  make_tensor_bbox(dev_a->addr, size), dev_a},
 *       {PTO_PARAM_OUTPUT, make_tensor_bbox(dev_c->addr, size), dev_c},
 *   };
 *   runtime->pto_submit_task(func_id, worker_type, params, 2);
 */
struct PTOParam {
    PTOParamType type;            // PTO_PARAM_INPUT, PTO_PARAM_OUTPUT, or PTO_PARAM_SCALAR
    PTOTensorDescriptor tensor;   // Full strided descriptor for overlap checking (unused for SCALAR)
    PTOBufferHandle* buffer;      // Associated buffer handle (nullptr for SCALAR)
    uint64_t scalar_value;        // Raw value for PTO_PARAM_SCALAR (e.g., encoded float, int size)
};

#endif // PTO_TYPES_H