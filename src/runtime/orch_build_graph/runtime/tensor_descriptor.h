
#include <cstdint>
#include <vector>

#include "common.h"

#pragma once

#define RUNTIME_MAX_TENSOR_DIMS 8

#ifndef NDEBUG
// 用于测试追踪 complex_overlap 是否被调用
struct OverlapPathTracker {
    static int& complex_overlap_call_count() {
        static int count = 0;
        return count;
    }
    static void reset() { complex_overlap_call_count() = 0; }
    static void record_complex_call() { complex_overlap_call_count()++; }
    static bool was_complex_called() { return complex_overlap_call_count() > 0; }
};
#endif

enum class OverlapType {
    Accurate = 0,
    Fuzzy = 1,
};

struct Segment {
    uint64_t begin;
    uint64_t end;

    bool line_segment_intersection(const Segment& other) const { return end > other.begin && other.end > begin; }
};

// 特殊值，表示 reshape 后需要分配新地址
static constexpr uint64_t RESHAPE_NEEDS_ALLOC = UINT64_MAX;

/**
 * Tensor descriptor for Task input/output
 *
 * Describes a strided memory access pattern on Global Memory (GM).
 * This allows expressing non-contiguous but regular memory layouts.
 *
 * Example: addr=base, start_offset=7, strides=[10, 1], repeats=[3, 6]
 * Memory access pattern (from innermost to outermost dimension):
 *   - Start at addr + 7
 *   - Inner dim (strides[1]=1, repeats[1]=6): access 6 consecutive elements
 *   - Outer dim (strides[0]=10, repeats[0]=3): repeat 3 times with stride 10
 * Result: [addr+7..addr+12], [addr+17..addr+22], [addr+27..addr+32]
 */
struct TensorDescriptor {
    class ContiguousMemSegIterator {
    public:
        ContiguousMemSegIterator(const TensorDescriptor& tensor);

        void operator++();
        void operator++(int) { ++*this; }

        const Segment& operator*() const { return cur_seg; }

        bool is_end() const { return indexes_[0] >= tensor_.repeats[0]; }

    private:
        const TensorDescriptor& tensor_;
        Segment cur_seg;
        std::vector<uint64_t> indexes_;
    };

    uint64_t addr;                              // Base address in GM, unit: uint8_t *
    uint64_t size;                              // total memory size, unit: byte
    uint64_t start_offset;                      // Starting offset from addr
    uint64_t strides[RUNTIME_MAX_TENSOR_DIMS];  // Stride for each dimension
    uint64_t repeats[RUNTIME_MAX_TENSOR_DIMS];  // Repeat count for each dimension
    uint64_t ndims;                             // Number of dimensions used
    int32_t version;                            // tensor的版本
    OverlapType overlap_type;                   // 判断覆盖的方式

    TensorDescriptor() : addr(0), size(0), ndims(0) {}

    explicit TensorDescriptor(uint64_t addr,
        uint64_t size,
        uint64_t start_offset,
        uint64_t strides[],
        uint64_t repeats[],
        uint64_t ndims,
        int32_t version,
        OverlapType overlap_type = OverlapType::Accurate);

    explicit TensorDescriptor(uint64_t addr,
        uint64_t size,
        uint64_t start_offset,
        const std::vector<uint64_t>& strides,
        const std::vector<uint64_t>& repeats,
        uint64_t ndims,
        int32_t version,
        OverlapType overlap_type = OverlapType::Accurate);

    TensorDescriptor(TensorDescriptor&& other);
    TensorDescriptor(const TensorDescriptor& other);

    TensorDescriptor& operator=(const TensorDescriptor& other);

    std::string dump() const;

    bool is_valid_tensor() const;

    TensorDescriptor& optimize();

    void resort_strides();

    void remove_redundant_dims();

#ifndef NDEBUG
    bool validate_memory_access_preserved(
        uint64_t original_strides[], uint64_t original_repeats[], int32_t original_ndims) const;

    void collect_all_offsets(
        const uint64_t strides_arr[], const uint64_t repeats_arr[], int32_t dims, std::vector<uint64_t>& offsets) const;
#endif

    std::vector<uint64_t> get_last_shapes() const { return std::vector<uint64_t>(repeats, repeats + ndims); }

    bool valid_view(const std::vector<uint64_t>& shapes, const std::vector<uint64_t>& offsets) const;

    bool valid_reshape(const std::vector<uint64_t>& shapes) const;

    bool valid_transpose(uint64_t x, uint64_t y) const { return x < ndims && y < ndims; }

    TensorDescriptor view(const std::vector<uint64_t>& shapes, const std::vector<uint64_t>& offsets) const;

    bool is_contiguous() const;

    TensorDescriptor reshape(const std::vector<uint64_t>& shapes) const;

    TensorDescriptor transpose(uint64_t x, uint64_t y) const;

    Segment get_fuzzy_seg() const;

    bool is_same_memref(const TensorDescriptor& other) const { return addr == other.addr; }

    bool is_same_strides(const TensorDescriptor& other) const;

    std::vector<uint64_t> offset_to_ndims() const;

    uint64_t offset_ndim_to_1d(const std::vector<uint64_t>& offset_ndims) const;

    bool is_overlap(const TensorDescriptor& pre_task_output) const;

    bool complex_overlap(const TensorDescriptor& pre_task_output) const;
};
