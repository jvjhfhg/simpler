/**
 * TensorDescriptor is_overlap 功能测试
 *
 * 测试覆盖场景：
 * 1. 基本场景：不同基地址、版本依赖、模糊段不相交
 * 2. Fuzzy 模式测试
 * 3. 一维场景测试
 * 4. 多维超矩形精确判断测试
 * 5. 复杂场景（不同 strides、稀疏访问模式）
 * 6. 边界情况（单元素、完全相同、维度合并后判断）
 *
 * 编译: filedir=rc/runtime/orch_build_graph/runtime; mkdir -p build && g++ -std=c++17 -g -I ${filedir} -o \
   build/test_tensor ${filedir}/tensor_descriptor.cpp ${filedir}/../tests/test_tensor_descriptor.cpp 2>&1
 * 运行: build/test_tensor
 * build/test_tensor                      # 运行所有测试
 * build/test_tensor --list               # 列出所有测试用例
 * build/test_tensor test_different_addr  # 运行指定测试
 * build/test_tensor test_1d              # 运行名称包含 'test_1d' 的测试
 * build/test_tensor Fuzzy                # 运行分类包含 'Fuzzy' 的测试
 * build/test_tensor 边界                 # 运行分类包含 '边界' 的测试
 */

#include <cassert>
#include <cstdio>
#include <cstring>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "../runtime/tensor_descriptor.h"

// ==================== 自动注册测试框架 ====================

struct TestCase {
    std::string name;
    std::string category;
    std::function<void()> func;
};

struct TestResult {
    std::string name;
    std::string category;
    bool passed;
    std::string error_message;
    int complex_overlap_calls;  // complex_overlap 被调用次数
};

class TestRegistry {
public:
    static TestRegistry& instance() {
        static TestRegistry registry;
        return registry;
    }

    void add(const char* name, const char* category, std::function<void()> func) {
        tests_.push_back({name, category, std::move(func)});
    }

    // 列出所有可用的测试用例
    void list_tests() {
        printf("可用的测试用例:\n\n");
        std::string current_cat;
        for (const auto& test : tests_) {
            if (test.category != current_cat) {
                printf("[%s]\n", test.category.c_str());
                current_cat = test.category;
            }
            printf("  %s\n", test.name.c_str());
        }
        printf("\n共 %zu 个测试用例\n", tests_.size());
    }

    // 运行所有测试
    int run_all() { return run_filtered(""); }

    // 运行匹配过滤器的测试（支持子串匹配）
    int run_filtered(const std::string& filter) {
        std::vector<TestResult> results;
        std::string current_category;
        int skipped = 0;

        printf("========================================\n");
        printf("TensorDescriptor is_overlap 测试\n");
        if (!filter.empty()) {
            printf("过滤器: %s\n", filter.c_str());
        }
        printf("========================================\n\n");

        for (const auto& test : tests_) {
            // 检查是否匹配过滤器（名称或分类包含过滤字符串）
            if (!filter.empty() && test.name.find(filter) == std::string::npos &&
                test.category.find(filter) == std::string::npos) {
                skipped++;
                continue;
            }

            if (test.category != current_category) {
                if (!current_category.empty()) {
                    printf("\n");
                }
                printf("--- %s ---\n", test.category.c_str());
                current_category = test.category;
            }

            printf("Running %s...\n", test.name.c_str());
            TestResult result{test.name, test.category, false, "", 0};

            try {
                OverlapPathTracker::reset();
                test.func();
                result.complex_overlap_calls = OverlapPathTracker::complex_overlap_call_count();
                printf("  PASSED\n");
                result.passed = true;
            } catch (const std::exception& e) {
                result.complex_overlap_calls = OverlapPathTracker::complex_overlap_call_count();
                result.error_message = e.what();
            } catch (...) {
                printf("  FAILED: Unknown exception\n");
                result.error_message = "Unknown exception";
            }

            results.push_back(result);
        }

        if (results.empty()) {
            printf("没有匹配的测试用例: %s\n", filter.c_str());
            return 1;
        }

        // 打印详细汇总
        print_summary(results, skipped);

        int failed_count = 0;
        for (const auto& r : results) {
            if (!r.passed) failed_count++;
        }
        return failed_count;
    }

    // 打印帮助信息
    static void print_help(const char* program) {
        printf("用法: %s [选项] [过滤器]\n\n", program);
        printf("选项:\n");
        printf("  -h, --help     显示帮助信息\n");
        printf("  -l, --list     列出所有测试用例\n");
        printf("\n");
        printf("过滤器:\n");
        printf("  指定测试名称或分类的子串来过滤测试\n");
        printf("  例如: %s test_1d       运行所有名称包含 'test_1d' 的测试\n", program);
        printf("  例如: %s 边界          运行所有分类包含 '边界' 的测试\n", program);
        printf("\n");
        printf("示例:\n");
        printf("  %s                     运行所有测试\n", program);
        printf("  %s test_different_addr 运行指定测试\n", program);
        printf("  %s Fuzzy               运行 Fuzzy 相关测试\n", program);
    }

private:
    void print_summary(const std::vector<TestResult>& results, int skipped = 0) {
        int passed_count = 0, failed_count = 0;
        for (const auto& r : results) {
            if (r.passed)
                passed_count++;
            else
                failed_count++;
        }

        printf("\n========================================\n");
        printf("测试结果汇总\n");
        printf("========================================\n\n");

        // 按分类分组显示所有测试结果
        std::string current_cat;
        for (const auto& t : results) {
            if (t.category != current_cat) {
                printf("[%s]\n", t.category.c_str());
                current_cat = t.category;
            }
            const char* status = t.passed ? "✅ PASSED" : "❌ FAILED";
            // 变换测试分类不显示判交信息
            bool is_transform_test = t.category.find("暴力变换验证") != std::string::npos ||
                                     t.category.find("复杂操作序列") != std::string::npos ||
                                     t.category.find("大规模数据") != std::string::npos;
            if (is_transform_test) {
                printf("  %s %s\n", status, t.name.c_str());
            } else if (t.complex_overlap_calls > 0) {
                printf("  %s [复杂判交 ×%d] %s\n", status, t.complex_overlap_calls, t.name.c_str());
            } else {
                printf("  %s [快速判交]    %s\n", status, t.name.c_str());
            }
            if (!t.passed && !t.error_message.empty()) {
                printf("     错误: %s\n", t.error_message.c_str());
            }
        }

        // 最终统计
        printf("\n========================================\n");
        printf("总计: %d 通过, %d 失败 (共 %zu 个测试", passed_count, failed_count, results.size());
        if (skipped > 0) {
            printf(", 跳过 %d 个", skipped);
        }
        printf(")\n");
        if (failed_count == 0) {
            printf("🎉 所有测试通过!\n");
        }
        printf("========================================\n");
    }

    std::vector<TestCase> tests_;
};

// 自动注册辅助类
struct TestRegistrar {
    TestRegistrar(const char* name, const char* category, std::function<void()> func) {
        TestRegistry::instance().add(name, category, std::move(func));
    }
};

// 测试定义宏 - 自动注册测试用例
#define TEST(category, name)                                      \
    void name();                                                  \
    static TestRegistrar name##_registrar(#name, category, name); \
    void name()

// 断言宏
#define ASSERT_TRUE(cond)                                        \
    do {                                                         \
        if (!(cond)) {                                           \
            printf("  FAILED: %s (line %d)\n", #cond, __LINE__); \
            throw std::runtime_error("Assertion failed");        \
        }                                                        \
    } while (0)

#define ASSERT_FALSE(cond) ASSERT_TRUE(!(cond))

// ==================== 辅助函数 ====================

/**
 * 暴力验证两个 tensor 是否存在内存交集
 *
 * 算法：
 * 1. 逐点遍历第一个 tensor 的所有 offset，在 vector<bool> 中标记
 * 2. 逐点遍历第二个 tensor 的所有 offset，检查是否被标记
 *
 * 注意：此函数只检查纯内存重叠，不涉及 version/addr/overlap_type 语义
 *
 * @param t1 第一个 tensor
 * @param t2 第二个 tensor
 * @return 是否存在内存交集
 */
bool brute_force_memory_overlap(const TensorDescriptor& t1, const TensorDescriptor& t2) {
    // 计算需要的数组大小（取两个 tensor 的最大可能 offset）
    uint64_t max_size = std::max(t1.size, t2.size);

    // 使用 vector<bool> 作为标记数组
    std::vector<bool> marked(max_size, false);

    // 遍历 t1 的所有点并标记
    // 使用多维索引遍历: idx[0..ndims-1]
    std::vector<uint64_t> idx1(t1.ndims, 0);
    while (true) {
        // 计算当前点的 offset
        uint64_t offset = t1.start_offset;
        for (uint64_t i = 0; i < t1.ndims; i++) {
            offset += idx1[i] * t1.strides[i];
        }
        if (offset < max_size) {
            marked[offset] = true;
        }

        // 递增多维索引（从最内层开始）
        int dim = static_cast<int>(t1.ndims) - 1;
        while (dim >= 0) {
            idx1[dim]++;
            if (idx1[dim] < t1.repeats[dim]) {
                break;
            }
            idx1[dim] = 0;
            dim--;
        }
        if (dim < 0) {
            break;  // 遍历完成
        }
    }

    // 遍历 t2 的所有点，检查是否被标记
    std::vector<uint64_t> idx2(t2.ndims, 0);
    while (true) {
        // 计算当前点的 offset
        uint64_t offset = t2.start_offset;
        for (uint64_t i = 0; i < t2.ndims; i++) {
            offset += idx2[i] * t2.strides[i];
        }
        if (offset < max_size && marked[offset]) {
            return true;  // 找到交集
        }

        // 递增多维索引
        int dim = static_cast<int>(t2.ndims) - 1;
        while (dim >= 0) {
            idx2[dim]++;
            if (idx2[dim] < t2.repeats[dim]) {
                break;
            }
            idx2[dim] = 0;
            dim--;
        }
        if (dim < 0) {
            break;
        }
    }

    return false;
}

/**
 * 验证 is_overlap 结果与暴力方法一致（用于同 addr、同 version、Accurate 模式）
 */
void verify_overlap_consistency(
    TensorDescriptor input, TensorDescriptor output, bool expected_overlap, const char* test_name) {
    // 优化不能在构造时自动执行，需要显式调用
    input.optimize();
    output.optimize();
    bool is_overlap_result = input.is_overlap(output);
    bool brute_force_result = brute_force_memory_overlap(input, output);

    // 对于同 addr、同 version 的情况，两者应该一致
    if (input.addr == output.addr && input.version == output.version && output.overlap_type == OverlapType::Accurate) {
        if (is_overlap_result != brute_force_result) {
            printf(
                "  [MISMATCH] %s: is_overlap=%d, brute_force=%d\n", test_name, is_overlap_result, brute_force_result);
        }
        ASSERT_TRUE(is_overlap_result == brute_force_result);
    }

    // 验证预期值
    if (is_overlap_result != expected_overlap) {
        printf("  [UNEXPECTED] %s: expected=%d, got=%d\n", test_name, expected_overlap, is_overlap_result);
    }
    ASSERT_TRUE(is_overlap_result == expected_overlap);

    // 额外输出暴力验证结果用于调试
    if (brute_force_result != expected_overlap && input.addr == output.addr && input.version == output.version) {
        printf("  [BRUTE_FORCE] actual_memory_overlap=%d\n", brute_force_result);
    }
}

// 便捷宏：自动使用当前函数名作为测试名
#define verify_overlap(input, output, expected) verify_overlap_consistency(input, output, expected, __func__)

/**
 * 创建 TensorDescriptor 的辅助函数
 */
TensorDescriptor make_tensor(uint64_t addr,
    uint64_t size,
    uint64_t start_offset,
    std::vector<uint64_t> strides_vec,
    std::vector<uint64_t> repeats_vec,
    int32_t version,
    OverlapType overlap_type = OverlapType::Accurate) {
    uint64_t strides[RUNTIME_MAX_TENSOR_DIMS] = {0};
    uint64_t repeats[RUNTIME_MAX_TENSOR_DIMS] = {0};
    uint64_t ndims = strides_vec.size();

    for (uint64_t i = 0; i < ndims; i++) {
        strides[i] = strides_vec[i];
        repeats[i] = repeats_vec[i];
    }

    return TensorDescriptor(addr, size, start_offset, strides, repeats, ndims, version, overlap_type);
}

/**
 * 打印 tensor 的内存访问段（用于调试）
 */
void print_tensor_segments(const TensorDescriptor& tensor, const char* name) {
    printf("  %s segments: ", name);
    TensorDescriptor::ContiguousMemSegIterator iter(tensor);
    int count = 0;
    while (!iter.is_end() && count < 20) {
        const Segment& seg = *iter;
        printf("[%lu..%lu] ", seg.begin, seg.end);
        iter++;
        count++;
    }
    if (!iter.is_end()) {
        printf("...");
    }
    printf("\n");
}

// ==================== 基本场景测试 ====================

/**
 * 测试：不同基地址应该无重叠
 */
TEST("基本场景测试", test_different_addr) {
    auto input = make_tensor(1000, 100, 0, {1}, {10}, 1);
    auto output = make_tensor(2000, 100, 0, {1}, {10}, 1);

    // 不同 addr，is_overlap 直接返回 false
    verify_overlap(input, output, false);
}

/**
 * 测试：版本号 input > output 时应该返回 true（存在依赖）
 */
TEST("基本场景测试", test_version_dependency) {
    auto input = make_tensor(1000, 100, 0, {1}, {10}, 2);   // version = 2
    auto output = make_tensor(1000, 100, 0, {1}, {10}, 1);  // version = 1

    // 版本不同，is_overlap 返回 true（语义依赖）
    verify_overlap(input, output, true);
}

/**
 * 测试：版本号 input > output 时即使没有overlap也应该返回 true（存在依赖）
 */
TEST("基本场景测试", test_version_dependency_but_not_memory_overlap) {
    auto input = make_tensor(1000, 100, 0, {1}, {10}, 2);    // version = 2
    auto output = make_tensor(1000, 100, 10, {1}, {10}, 1);  // version = 1

    // 版本不同，is_overlap 返回 true（语义依赖，即使实际内存不重叠）
    verify_overlap(input, output, true);
}

/**
 * 测试：相同版本但模糊段完全不相交
 */
TEST("基本场景测试", test_same_version_no_fuzzy_overlap) {
    auto input = make_tensor(1000, 200, 0, {1}, {10}, 1);
    auto output = make_tensor(1000, 200, 100, {1}, {10}, 1);

    verify_overlap(input, output, false);
}

// ==================== Fuzzy 模式测试 ====================

/**
 * 测试：Fuzzy 模式下，模糊段相交应返回 true
 */
TEST("Fuzzy 模式测试", test_fuzzy_overlap_true) {
    auto input = make_tensor(1000, 100, 0, {1}, {10}, 1);
    auto output = make_tensor(1000, 100, 5, {1}, {10}, 1, OverlapType::Fuzzy);

    verify_overlap(input, output, true);
}

/**
 * 测试：Fuzzy 模式下，模糊段不相交应返回 false
 */
TEST("Fuzzy 模式测试", test_fuzzy_no_intersection) {
    auto input = make_tensor(1000, 200, 0, {1}, {10}, 1);
    auto output = make_tensor(1000, 200, 50, {1}, {10}, 1, OverlapType::Fuzzy);

    verify_overlap(input, output, false);
}

// ==================== 一维场景测试 ====================

/**
 * 测试：一维连续段相交
 */
TEST("一维场景测试", test_1d_overlap) {
    auto input = make_tensor(1000, 100, 0, {1}, {10}, 1);
    auto output = make_tensor(1000, 100, 5, {1}, {10}, 1);

    verify_overlap(input, output, true);
}

/**
 * 测试：一维段相邻但不重叠
 */
TEST("一维场景测试", test_1d_adjacent) {
    auto input = make_tensor(1000, 100, 0, {1}, {10}, 1);
    auto output = make_tensor(1000, 100, 10, {1}, {10}, 1);

    verify_overlap(input, output, false);
}

/**
 * 测试：一维段部分重叠一个元素
 */
TEST("一维场景测试", test_1d_single_element_overlap) {
    auto input = make_tensor(1000, 100, 0, {1}, {10}, 1);
    auto output = make_tensor(1000, 100, 9, {1}, {10}, 1);

    verify_overlap(input, output, true);
}

// ==================== 多维超矩形场景测试 ====================

/**
 * 测试：2D 相同 strides，超矩形相交
 */
TEST("多维超矩形场景测试", test_2d_same_strides_overlap) {
    auto input = make_tensor(1000, 100, 0, {10, 1}, {3, 5}, 1);
    auto output = make_tensor(1000, 100, 2, {10, 1}, {3, 5}, 1);

    verify_overlap(input, output, true);
}

/**
 * 测试：2D 相同 strides，超矩形完全分离
 */
TEST("多维超矩形场景测试", test_2d_same_strides_no_overlap) {
    auto input = make_tensor(1000, 100, 0, {10, 1}, {2, 3}, 1);
    auto output = make_tensor(1000, 100, 5, {10, 1}, {2, 3}, 1);

    verify_overlap(input, output, false);
}

/**
 * 测试：2D 不同 offset 在第一维度上分离
 */
TEST("多维超矩形场景测试", test_2d_different_outer_dim) {
    auto input = make_tensor(1000, 100, 0, {10, 1}, {2, 5}, 1);
    auto output = make_tensor(1000, 100, 20, {10, 1}, {2, 5}, 1);

    verify_overlap(input, output, false);
}

/**
 * 测试：3D 超矩形部分重叠
 */
TEST("多维超矩形场景测试", test_3d_hyperrect_partial_overlap) {
    auto input = make_tensor(1000, 500, 0, {100, 10, 1}, {2, 3, 4}, 1);
    auto output = make_tensor(1000, 500, 2, {100, 10, 1}, {2, 3, 4}, 1);

    verify_overlap(input, output, true);
}

/**
 * 测试：3D 超矩形在中间维度分离
 */
TEST("多维超矩形场景测试", test_3d_hyperrect_middle_dim_separate) {
    auto input = make_tensor(1000, 500, 0, {100, 10, 1}, {2, 2, 4}, 1);
    auto output = make_tensor(1000, 500, 50, {100, 10, 1}, {2, 2, 4}, 1);

    verify_overlap(input, output, false);
}

// ==================== 复杂场景测试（需要 complex_overlap）====================

/**
 * 测试：不同 strides 有实际重叠
 */
TEST("复杂场景测试", test_different_strides_overlap) {
    auto input = make_tensor(1000, 100, 0, {20, 1}, {2, 5}, 1);
    auto output = make_tensor(1000, 100, 2, {10, 1}, {3, 3}, 1);

    print_tensor_segments(input, "input");
    print_tensor_segments(output, "output");

    verify_overlap(input, output, true);
}

/**
 * 测试：不同 strides 无实际重叠
 */
TEST("复杂场景测试", test_different_strides_no_overlap) {
    auto input = make_tensor(1000, 100, 0, {20, 1}, {2, 5}, 1);
    auto output = make_tensor(1000, 100, 7, {10, 1}, {2, 3}, 1);

    print_tensor_segments(input, "input");
    print_tensor_segments(output, "output");

    verify_overlap(input, output, false);
}

/**
 * 测试：稀疏访问模式交错有重叠
 */
TEST("复杂场景测试", test_sparse_access_interleaved) {
    auto input = make_tensor(1000, 100, 0, {10, 1}, {5, 2}, 1);
    auto output = make_tensor(1000, 100, 0, {20, 1}, {3, 2}, 1);

    print_tensor_segments(input, "input");
    print_tensor_segments(output, "output");

    verify_overlap(input, output, true);
}

/**
 * 测试：稀疏访问模式交错无重叠
 */
TEST("复杂场景测试", test_sparse_access_separate) {
    auto input = make_tensor(1000, 100, 0, {10, 1}, {5, 2}, 1);
    auto output = make_tensor(1000, 100, 5, {20, 1}, {3, 2}, 1);

    print_tensor_segments(input, "input");
    print_tensor_segments(output, "output");

    verify_overlap(input, output, false);
}

/**
 * 测试：不同 ndims 的复杂重叠
 */
TEST("复杂场景测试", test_different_ndims_overlap) {
    auto input = make_tensor(1000, 200, 0, {10, 1}, {3, 5}, 1);
    auto output = make_tensor(1000, 200, 0, {100, 10, 1}, {1, 3, 5}, 1);

    print_tensor_segments(input, "input");
    print_tensor_segments(output, "output");

    verify_overlap(input, output, true);
}

// ==================== 边界情况测试 ====================

/**
 * 测试：单元素 tensor 重叠
 */
TEST("边界情况测试", test_single_element_overlap) {
    auto input = make_tensor(1000, 100, 50, {1}, {1}, 1);
    auto output = make_tensor(1000, 100, 50, {1}, {1}, 1);

    verify_overlap(input, output, true);
}

/**
 * 测试：单元素 tensor 不重叠
 */
TEST("边界情况测试", test_single_element_no_overlap) {
    auto input = make_tensor(1000, 100, 50, {1}, {1}, 1);
    auto output = make_tensor(1000, 100, 51, {1}, {1}, 1);

    verify_overlap(input, output, false);
}

/**
 * 测试：完全相同的 tensor 应该重叠
 */
TEST("边界情况测试", test_same_tensor) {
    auto input = make_tensor(1000, 100, 0, {10, 1}, {3, 5}, 1);
    auto output = make_tensor(1000, 100, 0, {10, 1}, {3, 5}, 1);

    verify_overlap(input, output, true);
}

/**
 * 测试：验证 remove_redundant_dims 后的判断正确性
 */
TEST("边界情况测试", test_stride_merge_equivalence) {
    auto tensor1 = make_tensor(1000, 200, 0, {10, 1}, {3, 10}, 1);
    auto tensor2 = make_tensor(1000, 200, 0, {100, 10, 1}, {1, 3, 10}, 1);
    auto output = make_tensor(1000, 200, 5, {10, 1}, {2, 5}, 1);

    print_tensor_segments(tensor1, "tensor1");
    print_tensor_segments(tensor2, "tensor2");
    print_tensor_segments(output, "output");

    bool brute1 = brute_force_memory_overlap(tensor1, output);
    bool brute2 = brute_force_memory_overlap(tensor2, output);

    // 两个等价 tensor 对 output 的覆盖判断结果应一致
    verify_overlap(tensor1, output, brute1);
    verify_overlap(tensor2, output, brute2);
    ASSERT_TRUE(brute1 == brute2);
    ASSERT_TRUE(brute1);
}

/**
 * 测试：大规模稀疏访问模式
 */
TEST("边界情况测试", test_large_sparse_pattern) {
    auto input = make_tensor(1000, 10000, 0, {1000, 100, 10, 1}, {2, 3, 4, 5}, 1);
    auto output = make_tensor(1000, 10000, 0, {500, 50, 1}, {3, 5, 8}, 1);

    print_tensor_segments(input, "input");
    print_tensor_segments(output, "output");

    verify_overlap(input, output, true);
}

/**
 * 测试：边界情况 - 内存段刚好相邻
 */
TEST("边界情况测试", test_exact_adjacent_boundary) {
    auto input = make_tensor(1000, 100, 0, {10, 1}, {2, 5}, 1);
    auto output = make_tensor(1000, 100, 5, {10, 1}, {2, 5}, 1);

    print_tensor_segments(input, "input");
    print_tensor_segments(output, "output");

    verify_overlap(input, output, false);
}

/**
 * 测试：边界情况 - 内存段刚好有一个字节重叠
 */
TEST("边界情况测试", test_one_byte_overlap_boundary) {
    auto input = make_tensor(1000, 100, 0, {10, 1}, {2, 5}, 1);
    auto output = make_tensor(1000, 100, 4, {10, 1}, {2, 5}, 1);

    print_tensor_segments(input, "input");
    print_tensor_segments(output, "output");

    verify_overlap(input, output, true);
}

// ==================== 完全不同 strides 组合测试 ====================

/**
 * 测试：2D vs 3D 完全不同 strides，有重叠
 */
TEST("完全不同 strides 组合测试", test_2d_vs_3d_different_strides_overlap) {
    auto input = make_tensor(1000, 500, 0, {20, 1}, {5, 8}, 1);
    auto output = make_tensor(1000, 500, 0, {100, 10, 1}, {2, 3, 4}, 1);

    print_tensor_segments(input, "input");
    print_tensor_segments(output, "output");

    verify_overlap(input, output, true);
}

/**
 * 测试：2D vs 3D 完全不同 strides，无重叠
 */
TEST("完全不同 strides 组合测试", test_2d_vs_3d_different_strides_no_overlap) {
    auto input = make_tensor(1000, 500, 0, {20, 1}, {3, 5}, 1);
    auto output = make_tensor(1000, 500, 100, {50, 10, 1}, {2, 2, 3}, 1);

    print_tensor_segments(input, "input");
    print_tensor_segments(output, "output");

    verify_overlap(input, output, false);
}

/**
 * 测试：3D vs 4D 完全不同 strides，有重叠
 */
TEST("完全不同 strides 组合测试", test_3d_vs_4d_different_strides_overlap) {
    auto input = make_tensor(1000, 2000, 0, {100, 10, 1}, {3, 4, 5}, 1);
    auto output = make_tensor(1000, 2000, 5, {500, 50, 5, 1}, {2, 2, 3, 3}, 1);

    print_tensor_segments(input, "input");
    print_tensor_segments(output, "output");

    verify_overlap(input, output, true);
}

/**
 * 测试：3D vs 4D 完全不同 strides，无重叠
 */
TEST("完全不同 strides 组合测试", test_3d_vs_4d_different_strides_no_overlap) {
    auto input = make_tensor(1000, 2000, 0, {100, 10, 1}, {2, 3, 4}, 1);
    auto output = make_tensor(1000, 2000, 500, {200, 20, 2, 1}, {2, 2, 3, 2}, 1);

    print_tensor_segments(input, "input");
    print_tensor_segments(output, "output");

    verify_overlap(input, output, false);
}

/**
 * 测试：大 stride 对小 stride 交错模式，有重叠
 */
TEST("完全不同 strides 组合测试", test_large_vs_small_stride_interleaved_overlap) {
    auto input = make_tensor(1000, 500, 0, {100, 1}, {3, 10}, 1);
    auto output = make_tensor(1000, 500, 5, {20, 1}, {8, 5}, 1);

    print_tensor_segments(input, "input");
    print_tensor_segments(output, "output");

    verify_overlap(input, output, true);
}

/**
 * 测试：大 stride 对小 stride 交错模式，无重叠
 */
TEST("完全不同 strides 组合测试", test_large_vs_small_stride_interleaved_no_overlap) {
    auto input = make_tensor(1000, 500, 0, {100, 1}, {3, 5}, 1);
    auto output = make_tensor(1000, 500, 10, {20, 1}, {3, 5}, 1);

    print_tensor_segments(input, "input");
    print_tensor_segments(output, "output");

    verify_overlap(input, output, false);
}

/**
 * 测试：质数 stride 组合，有重叠
 * 注意：size 必须能被所有 tensor 的 strides[0] 整除
 * LCM(17, 13) = 221, 使用 size=442
 */
TEST("完全不同 strides 组合测试", test_prime_strides_overlap) {
    auto input = make_tensor(1000, 442, 0, {17, 1}, {5, 7}, 1);
    auto output = make_tensor(1000, 442, 3, {13, 1}, {6, 5}, 1);

    print_tensor_segments(input, "input");
    print_tensor_segments(output, "output");

    verify_overlap(input, output, true);
}

/**
 * 测试：质数 stride 组合，无重叠
 * 注意：size 必须能被所有 tensor 的 strides[0] 整除
 * LCM(23, 19) = 437, 使用 size=437
 */
TEST("完全不同 strides 组合测试", test_prime_strides_no_overlap) {
    auto input = make_tensor(1000, 437, 0, {23, 1}, {3, 5}, 1);
    auto output = make_tensor(1000, 437, 100, {19, 1}, {4, 5}, 1);

    print_tensor_segments(input, "input");
    print_tensor_segments(output, "output");

    verify_overlap(input, output, false);
}

// ==================== 维度缩减测试 ====================

/**
 * 测试：高维可合并为低维，相同内存模式
 */
TEST("维度缩减测试", test_dim_reduction_equivalent) {
    // 3D tensor: strides=[100, 10, 1], repeats=[2, 10, 10]
    // 可以合并为 1D: strides=[1], repeats=[200] (因为 10*10=100, 10*10=100)
    auto tensor_3d = make_tensor(1000, 500, 0, {100, 10, 1}, {2, 10, 10}, 1);
    auto tensor_1d = make_tensor(1000, 500, 0, {1}, {200}, 1);

    print_tensor_segments(tensor_3d, "tensor_3d");
    print_tensor_segments(tensor_1d, "tensor_1d");

    // 两者应该完全重叠
    verify_overlap(tensor_3d, tensor_1d, true);
}

/**
 * 测试：高维可合并为低维，与另一个 tensor 判断
 */
TEST("维度缩减测试", test_dim_reduction_with_other_tensor) {
    // 2D tensor: strides=[10, 1], repeats=[5, 10] -> 可合并为 1D: [1], [50]
    auto tensor_2d = make_tensor(1000, 500, 0, {10, 1}, {5, 10}, 1);
    auto other = make_tensor(1000, 500, 25, {1}, {20}, 1);

    print_tensor_segments(tensor_2d, "tensor_2d");
    print_tensor_segments(other, "other");

    verify_overlap(tensor_2d, other, true);
}

/**
 * 测试：多级连续合并 4D -> 2D
 */
TEST("维度缩减测试", test_multi_level_dim_reduction) {
    // 4D: strides=[1000, 100, 10, 1], repeats=[1, 10, 10, 10]
    // 合并后: strides=[1000, 1], repeats=[1, 1000]
    // fuzzy_seg.end = 0 + 1000*1 + 100*10 + 10*10 + 1*10 = 2110, 需要 size >= 2110
    auto tensor_4d = make_tensor(1000, 3000, 0, {1000, 100, 10, 1}, {1, 10, 10, 10}, 1);
    auto tensor_2d = make_tensor(1000, 3000, 0, {1000, 1}, {1, 1000}, 1);

    print_tensor_segments(tensor_4d, "tensor_4d");
    print_tensor_segments(tensor_2d, "tensor_2d");

    verify_overlap(tensor_4d, tensor_2d, true);
}

/**
 * 测试：部分维度可合并，部分不可合并
 */
TEST("维度缩减测试", test_partial_dim_reduction) {
    // 3D: strides=[50, 10, 1], repeats=[2, 5, 10]
    // 内两维 10*5=50，刚好等于外层 stride，可合并为 2D
    auto tensor_3d = make_tensor(1000, 500, 0, {50, 10, 1}, {2, 5, 10}, 1);
    auto other = make_tensor(1000, 500, 30, {20, 1}, {3, 10}, 1);

    print_tensor_segments(tensor_3d, "tensor_3d");
    print_tensor_segments(other, "other");

    verify_overlap(tensor_3d, other, true);
}

/**
 * 测试：合并后维度不同但内存等价，无重叠
 */
TEST("维度缩减测试", test_dim_reduction_equivalent_no_overlap) {
    auto tensor_3d = make_tensor(1000, 500, 0, {100, 10, 1}, {2, 10, 10}, 1);
    auto other = make_tensor(1000, 500, 200, {1}, {50}, 1);

    print_tensor_segments(tensor_3d, "tensor_3d");
    print_tensor_segments(other, "other");

    verify_overlap(tensor_3d, other, false);
}

// ==================== 高维合并低维不合并测试 ====================

/**
 * 测试：高维可合并低维不可合并，有重叠
 *
 * 4D: strides=[1000, 100, 10, 1], repeats=[2, 10, 3, 5]
 * dim0↔dim1: 1000 == 100*10 → 合并
 * 合并后 3D: strides=[100, 10, 1], repeats=[20, 3, 5]
 * 低维不合并: 10 != 1*5, 产生间隙
 * 段: [0,5), [10,15), [20,25), [100,105), ...
 * offset=2 的段: [2,7), [12,17), ... 与上面有交集
 */
TEST("高维合并低维不合并测试", test_high_merge_low_no_merge_overlap) {
    // fuzzy_seg.end = 0 + 1000*2 + 100*10 + 10*3 + 1*5 = 3035, 需要 size >= 3035
    auto input = make_tensor(1000, 4000, 0, {1000, 100, 10, 1}, {2, 10, 3, 5}, 1);
    auto output = make_tensor(1000, 4000, 2, {1000, 100, 10, 1}, {2, 10, 3, 5}, 1);

    print_tensor_segments(input, "input");
    print_tensor_segments(output, "output");

    verify_overlap(input, output, true);
}

/**
 * 测试：高维可合并低维不可合并，无重叠
 *
 * 同上结构，offset=5 使低维段恰好错开
 * 段: [0,5), [10,15), [20,25), ...
 * offset=5 的段: [5,10), [15,20), [25,30), ...
 * 两组段恰好互补，无交集
 */
TEST("高维合并低维不合并测试", test_high_merge_low_no_merge_no_overlap) {
    auto input = make_tensor(1000, 4000, 0, {1000, 100, 10, 1}, {2, 10, 3, 5}, 1);
    auto output = make_tensor(1000, 4000, 5, {1000, 100, 10, 1}, {2, 10, 3, 5}, 1);

    print_tensor_segments(input, "input");
    print_tensor_segments(output, "output");

    verify_overlap(input, output, false);
}

/**
 * 测试：高维合并低维不合并，合并前后等价性验证
 *
 * 4D tensor（高维可合并）与手动合并后的 3D tensor 对同一个 output 判断应一致
 * 4D: strides=[1000, 100, 10, 1], repeats=[2, 10, 3, 5] → 合并后 3D: [100, 10, 1], [20, 3, 5]
 */
TEST("高维合并低维不合并测试", test_high_merge_low_no_merge_equivalence) {
    auto tensor_4d = make_tensor(1000, 4000, 0, {1000, 100, 10, 1}, {2, 10, 3, 5}, 1);
    auto tensor_3d = make_tensor(1000, 4000, 0, {100, 10, 1}, {20, 3, 5}, 1);
    auto other = make_tensor(1000, 4000, 3, {100, 10, 1}, {10, 2, 4}, 1);

    print_tensor_segments(tensor_4d, "tensor_4d");
    print_tensor_segments(tensor_3d, "tensor_3d");
    print_tensor_segments(other, "other");

    bool brute_4d = brute_force_memory_overlap(tensor_4d, other);
    bool brute_3d = brute_force_memory_overlap(tensor_3d, other);

    // 合并前后结果应一致
    verify_overlap(tensor_4d, other, brute_4d);
    verify_overlap(tensor_3d, other, brute_3d);
    ASSERT_TRUE(brute_4d == brute_3d);
}

/**
 * 测试：3D 高维合并为 2D，低维不合并，无重叠
 *
 * 3D: strides=[60, 10, 1], repeats=[2, 6, 3]
 * dim0↔dim1: 60 == 10*6 → 合并为 2D: [10, 1], [12, 3]
 * 低维不合并: 10 != 1*3=3，间隙 7 个元素
 * 段: [0,3), [10,13), [20,23), ..., [110,113)
 * 对手 offset=5: [5,8), [15,18), ..., 恰在间隙中
 */
TEST("高维合并低维不合并测试", test_high_merge_low_no_merge_3d_to_2d) {
    // size 必须能被 strides[0]=60 整除，使用 240
    auto input = make_tensor(1000, 240, 0, {60, 10, 1}, {2, 6, 3}, 1);
    auto output = make_tensor(1000, 240, 5, {60, 10, 1}, {2, 6, 3}, 1);

    print_tensor_segments(input, "input");
    print_tensor_segments(output, "output");

    verify_overlap(input, output, false);
}

/**
 * 测试：高维合并低维不合并，低维段恰好边界相邻
 *
 * 4D: strides=[500, 50, 10, 1], repeats=[2, 10, 2, 5]
 * dim0↔dim1: 500 == 50*10 → 合并
 * 合并后 3D: [50, 10, 1], [20, 2, 5]
 * 段: [0,5), [10,15), [50,55), [60,65), ...
 * offset=5 的段: [5,10), [15,20), [55,60), [65,70), ...
 * 恰好相邻（end == begin），无重叠
 */
TEST("高维合并低维不合并测试", test_high_merge_low_no_merge_boundary) {
    auto input = make_tensor(1000, 2000, 0, {500, 50, 10, 1}, {2, 10, 2, 5}, 1);
    auto output = make_tensor(1000, 2000, 5, {500, 50, 10, 1}, {2, 10, 2, 5}, 1);

    print_tensor_segments(input, "input");
    print_tensor_segments(output, "output");

    verify_overlap(input, output, false);
}

/**
 * 测试：两个 tensor 都是高维合并低维不合并，但 strides 不同，走 complex_overlap 路径
 *
 * tensor1: 4D [1000, 100, 10, 1], [2, 10, 3, 5] → 3D [100, 10, 1], [20, 3, 5]
 * tensor2: 4D [360, 60, 10, 1], [2, 6, 3, 4] → 3D [60, 10, 1], [12, 3, 4]
 * strides 不同 → 走 complex_overlap
 */
TEST("高维合并低维不合并测试", test_high_merge_low_no_merge_different_strides) {
    auto input = make_tensor(1000, 9000, 0, {1000, 100, 10, 1}, {2, 10, 3, 5}, 1);
    // output fuzzy_seg.end = 0 + 360*2 + 60*6 + 10*3 + 1*4 = 720 + 360 + 30 + 4 = 1114
    // LCM(1000, 360) = 9000
    auto output = make_tensor(1000, 9000, 0, {360, 60, 10, 1}, {2, 6, 3, 4}, 1);

    print_tensor_segments(input, "input");
    print_tensor_segments(output, "output");

    bool brute = brute_force_memory_overlap(input, output);
    verify_overlap(input, output, brute);
}

// ==================== 超矩形 vs 非超矩形测试 ====================

/**
 * 测试：同 strides，offset 导致非超矩形（input 非超矩形）
 */
TEST("超矩形 vs 非超矩形测试", test_non_hyperrect_input) {
    // strides=[10, 1], offset=8, repeats=[2, 5]
    // 第一行: [8, 13), 第二行: [18, 23)
    // offset 8 + repeats 5 = 13 > stride 10，所以 input 不是超矩形
    auto input = make_tensor(1000, 100, 8, {10, 1}, {2, 5}, 1);
    auto output = make_tensor(1000, 100, 0, {10, 1}, {2, 5}, 1);

    print_tensor_segments(input, "input");
    print_tensor_segments(output, "output");

    verify_overlap(input, output, true);
}

/**
 * 测试：同 strides，offset 导致非超矩形（output 非超矩形）
 */
TEST("超矩形 vs 非超矩形测试", test_non_hyperrect_output) {
    auto input = make_tensor(1000, 100, 0, {10, 1}, {2, 4}, 1);
    // offset=7, repeats=5 -> 7+5=12 > 10，output 不是超矩形
    auto output = make_tensor(1000, 100, 7, {10, 1}, {2, 5}, 1);

    print_tensor_segments(input, "input");
    print_tensor_segments(output, "output");

    verify_overlap(input, output, true);
}

/**
 * 测试：两方都是非超矩形，有重叠
 */
TEST("超矩形 vs 非超矩形测试", test_both_non_hyperrect_overlap) {
    // 两个都越界
    auto input = make_tensor(1000, 100, 7, {10, 1}, {3, 6}, 1);
    auto output = make_tensor(1000, 100, 8, {10, 1}, {3, 5}, 1);

    print_tensor_segments(input, "input");
    print_tensor_segments(output, "output");

    verify_overlap(input, output, true);
}

/**
 * 测试：两方都是非超矩形，无重叠
 */
TEST("超矩形 vs 非超矩形测试", test_both_non_hyperrect_no_overlap) {
    auto input = make_tensor(1000, 200, 6, {20, 1}, {3, 8}, 1);
    auto output = make_tensor(1000, 200, 100, {20, 1}, {3, 8}, 1);

    print_tensor_segments(input, "input");
    print_tensor_segments(output, "output");

    verify_overlap(input, output, false);
}

/**
 * 测试：超矩形 vs 非超矩形，边界有重叠
 */
TEST("超矩形 vs 非超矩形测试", test_hyperrect_vs_non_hyperrect_boundary) {
    // input 是超矩形: offset=0, repeats=3 < stride=10
    auto input = make_tensor(1000, 100, 0, {10, 1}, {2, 3}, 1);
    // output 不是超矩形: offset=9, repeats=4 -> 9+4=13 > 10
    auto output = make_tensor(1000, 100, 9, {10, 1}, {2, 4}, 1);

    print_tensor_segments(input, "input");
    print_tensor_segments(output, "output");

    // input: [0,3), [10,13)
    // output: [9,13), [19,23) -> 因为跨stride所以是 [9,10), [10,13), [19,20), [20,23)
    // 实际上暴力验证会给出准确答案
    bool brute = brute_force_memory_overlap(input, output);
    verify_overlap(input, output, brute);
}

/**
 * 测试：3D 非超矩形复杂场景
 */
TEST("超矩形 vs 非超矩形测试", test_3d_non_hyperrect_complex) {
    // 中间维度超出
    auto input = make_tensor(1000, 1000, 5, {100, 10, 1}, {3, 8, 5}, 1);
    auto output = make_tensor(1000, 1000, 50, {100, 10, 1}, {3, 6, 4}, 1);

    print_tensor_segments(input, "input");
    print_tensor_segments(output, "output");

    bool brute = brute_force_memory_overlap(input, output);
    verify_overlap(input, output, brute);
}

// ==================== 模糊段相交但实际不重叠测试 ====================

/**
 * 测试：fuzzy seg 相交但精确判断无重叠
 */
TEST("模糊段相交但实际不重叠测试", test_fuzzy_intersect_but_no_actual_overlap) {
    // input: offset=0, strides=[20,1], repeats=[3,5] -> 访问 [0,5), [20,25), [40,45)
    // fuzzy_seg: [0, 0+20*3+1*5) = [0, 65)
    auto input = make_tensor(1000, 200, 0, {20, 1}, {3, 5}, 1);
    // output: offset=10, strides=[20,1], repeats=[3,5] -> 访问 [10,15), [30,35), [50,55)
    // fuzzy_seg: [10, 10+20*3+1*5) = [10, 75)
    // fuzzy 相交: [10, 65)，但实际无重叠
    auto output = make_tensor(1000, 200, 10, {20, 1}, {3, 5}, 1);

    print_tensor_segments(input, "input");
    print_tensor_segments(output, "output");

    verify_overlap(input, output, false);
}

/**
 * 测试：fuzzy seg 大范围相交但稀疏无重叠
 */
TEST("模糊段相交但实际不重叠测试", test_fuzzy_intersect_but_sparse_no_overlap) {
    // 偶数位置
    auto input = make_tensor(1000, 200, 0, {20, 1}, {5, 2}, 1);
    // 奇数位置（错开）
    auto output = make_tensor(1000, 200, 5, {20, 1}, {5, 2}, 1);

    print_tensor_segments(input, "input");
    print_tensor_segments(output, "output");

    verify_overlap(input, output, false);
}

/**
 * 测试：3D fuzzy 相交但实际不重叠
 */
TEST("模糊段相交但实际不重叠测试", test_3d_fuzzy_intersect_no_actual_overlap) {
    auto input = make_tensor(1000, 1000, 0, {100, 20, 1}, {3, 2, 5}, 1);
    auto output = make_tensor(1000, 1000, 10, {100, 20, 1}, {3, 2, 5}, 1);

    print_tensor_segments(input, "input");
    print_tensor_segments(output, "output");

    verify_overlap(input, output, false);
}

/**
 * 测试：不同 strides 导致 fuzzy 相交但无实际重叠
 */
TEST("模糊段相交但实际不重叠测试", test_different_strides_fuzzy_intersect_no_overlap) {
    auto input = make_tensor(1000, 500, 0, {50, 1}, {4, 10}, 1);
    auto output = make_tensor(1000, 500, 15, {50, 1}, {4, 10}, 1);

    print_tensor_segments(input, "input");
    print_tensor_segments(output, "output");

    verify_overlap(input, output, false);
}

// ==================== 高维稀疏测试 ====================

/**
 * 测试：5D tensor 交叉，有重叠
 */
TEST("高维稀疏测试", test_5d_overlap) {
    auto input = make_tensor(1000, 50000, 0, {10000, 1000, 100, 10, 1}, {2, 2, 2, 2, 3}, 1);
    auto output = make_tensor(1000, 50000, 1, {10000, 1000, 100, 10, 1}, {2, 2, 2, 2, 3}, 1);

    verify_overlap(input, output, true);
}

/**
 * 测试：5D tensor 交叉，无重叠
 */
TEST("高维稀疏测试", test_5d_no_overlap) {
    auto input = make_tensor(1000, 50000, 0, {10000, 1000, 100, 10, 1}, {2, 2, 2, 2, 3}, 1);
    auto output = make_tensor(1000, 50000, 5, {10000, 1000, 100, 10, 1}, {2, 2, 2, 2, 3}, 1);

    verify_overlap(input, output, false);
}

/**
 * 测试：6D tensor 仅在某一维有交叉
 */
TEST("高维稀疏测试", test_6d_single_dim_overlap) {
    auto input = make_tensor(1000, 200000, 0, {100000, 10000, 1000, 100, 10, 1}, {1, 1, 1, 2, 3, 4}, 1);
    auto output = make_tensor(1000, 200000, 2, {100000, 10000, 1000, 100, 10, 1}, {1, 1, 1, 2, 3, 4}, 1);

    verify_overlap(input, output, true);
}

/**
 * 测试：7D tensor 稀疏访问
 */
TEST("高维稀疏测试", test_7d_sparse) {
    auto input = make_tensor(1000, 1000000, 0, {500000, 50000, 5000, 500, 50, 5, 1}, {1, 1, 1, 2, 2, 2, 3}, 1);
    auto output = make_tensor(1000, 1000000, 1, {500000, 50000, 5000, 500, 50, 5, 1}, {1, 1, 1, 2, 2, 2, 3}, 1);

    verify_overlap(input, output, true);
}

/**
 * 测试：高维不同 strides
 */
TEST("高维稀疏测试", test_5d_different_strides) {
    auto input = make_tensor(1000, 10000, 0, {2000, 200, 20, 2, 1}, {2, 2, 2, 2, 2}, 1);
    auto output = make_tensor(1000, 10000, 0, {5000, 500, 50, 5, 1}, {1, 2, 2, 2, 3}, 1);

    bool brute = brute_force_memory_overlap(input, output);
    verify_overlap(input, output, brute);
}

// ==================== 特殊对称/互补模式测试 ====================

/**
 * 测试：棋盘格访问模式 - 偶数位置 vs 奇数位置
 */
TEST("特殊对称/互补模式测试", test_checkerboard_even_vs_odd) {
    // 偶数位置: 0, 2, 4, 6, ...
    auto even = make_tensor(1000, 100, 0, {2, 1}, {10, 1}, 1);
    // 奇数位置: 1, 3, 5, 7, ...
    auto odd = make_tensor(1000, 100, 1, {2, 1}, {10, 1}, 1);

    print_tensor_segments(even, "even");
    print_tensor_segments(odd, "odd");

    verify_overlap(even, odd, false);
}

/**
 * 测试：一个 tensor 恰好占据另一个的间隙（互补）
 */
TEST("特殊对称/互补模式测试", test_complementary_pattern) {
    // tensor1: [0,3), [10,13), [20,23)
    auto tensor1 = make_tensor(1000, 100, 0, {10, 1}, {3, 3}, 1);
    // tensor2: [5,8), [15,18), [25,28) 恰好在间隙
    auto tensor2 = make_tensor(1000, 100, 5, {10, 1}, {3, 3}, 1);

    print_tensor_segments(tensor1, "tensor1");
    print_tensor_segments(tensor2, "tensor2");

    verify_overlap(tensor1, tensor2, false);
}

/**
 * 测试：2D 棋盘格模式
 */
TEST("特殊对称/互补模式测试", test_2d_checkerboard) {
    // 2D 棋盘格: 每行交替
    // tensor1: 行0列偶数, 行1列奇数, ...
    auto tensor1 = make_tensor(1000, 200, 0, {20, 2, 1}, {3, 5, 1}, 1);
    // tensor2: 偏移1
    auto tensor2 = make_tensor(1000, 200, 1, {20, 2, 1}, {3, 5, 1}, 1);

    print_tensor_segments(tensor1, "tensor1");
    print_tensor_segments(tensor2, "tensor2");

    verify_overlap(tensor1, tensor2, false);
}

/**
 * 测试：完全互补的 2D pattern
 */
TEST("特殊对称/互补模式测试", test_2d_fully_complementary) {
    // 第一个访问每行前半部分
    auto tensor1 = make_tensor(1000, 200, 0, {20, 1}, {5, 10}, 1);
    // 第二个访问每行后半部分
    auto tensor2 = make_tensor(1000, 200, 10, {20, 1}, {5, 10}, 1);

    print_tensor_segments(tensor1, "tensor1");
    print_tensor_segments(tensor2, "tensor2");

    verify_overlap(tensor1, tensor2, false);
}

/**
 * 测试：周期性交错但有一个点重叠
 */
TEST("特殊对称/互补模式测试", test_periodic_single_overlap) {
    auto tensor1 = make_tensor(1000, 100, 0, {10, 1}, {5, 3}, 1);
    auto tensor2 = make_tensor(1000, 100, 2, {10, 1}, {5, 3}, 1);

    print_tensor_segments(tensor1, "tensor1");
    print_tensor_segments(tensor2, "tensor2");

    verify_overlap(tensor1, tensor2, true);
}

/**
 * 测试：多层嵌套周期模式
 */
TEST("特殊对称/互补模式测试", test_nested_periodic) {
    // 3层嵌套周期
    auto tensor1 = make_tensor(1000, 2000, 0, {200, 20, 1}, {3, 4, 5}, 1);
    auto tensor2 = make_tensor(1000, 2000, 10, {200, 20, 1}, {3, 4, 5}, 1);

    print_tensor_segments(tensor1, "tensor1");
    print_tensor_segments(tensor2, "tensor2");

    verify_overlap(tensor1, tensor2, false);
}

// ==================== 额外边界和极端情况测试 ====================

/**
 * 测试：最大维度 8D
 */
TEST("额外边界测试", test_max_8d_dimensions) {
    auto input =
        make_tensor(1000, 10000000, 0, {1000000, 100000, 10000, 1000, 100, 10, 2, 1}, {1, 1, 1, 1, 2, 2, 2, 2}, 1);
    auto output =
        make_tensor(1000, 10000000, 1, {1000000, 100000, 10000, 1000, 100, 10, 2, 1}, {1, 1, 1, 1, 2, 2, 2, 2}, 1);

    verify_overlap(input, output, true);
}

/**
 * 测试：1D 连续内存访问的重叠
 */
TEST("额外边界测试", test_1d_continuous_overlap) {
    // 1D 连续访问: 两个 tensor 部分重叠
    auto tensor1 = make_tensor(1000, 200, 0, {1}, {100}, 1);
    auto tensor2 = make_tensor(1000, 200, 50, {1}, {100}, 1);

    print_tensor_segments(tensor1, "tensor1");
    print_tensor_segments(tensor2, "tensor2");

    verify_overlap(tensor1, tensor2, true);
}

/**
 * 测试：非常大的 stride 跨度
 */
TEST("额外边界测试", test_very_large_stride) {
    auto input = make_tensor(1000, 1000000, 0, {100000, 1}, {3, 10}, 1);
    auto output = make_tensor(1000, 1000000, 50000, {100000, 1}, {3, 10}, 1);

    verify_overlap(input, output, false);
}

/**
 * 测试：repeats 为 1 的多维
 */
TEST("额外边界测试", test_repeats_one) {
    auto tensor1 = make_tensor(1000, 100, 0, {50, 10, 1}, {1, 1, 10}, 1);
    auto tensor2 = make_tensor(1000, 100, 5, {50, 10, 1}, {1, 1, 10}, 1);

    print_tensor_segments(tensor1, "tensor1");
    print_tensor_segments(tensor2, "tensor2");

    verify_overlap(tensor1, tensor2, true);
}

/**
 * 测试：只有第一维有多个 repeat，重叠
 */
TEST("额外边界测试", test_only_first_dim_multiple_repeats_with_overlap) {
    auto tensor1 = make_tensor(1000, 56, 0, {8, 1}, {5, 1}, 1);
    auto tensor2 = make_tensor(1000, 56, 2, {7, 1}, {5, 1}, 1);

    print_tensor_segments(tensor1, "tensor1");
    print_tensor_segments(tensor2, "tensor2");

    verify_overlap(tensor1, tensor2, true);
}

/**
 * 测试：只有第一维有多个 repeat，不重叠
 */
TEST("额外边界测试", test_only_first_dim_multiple_repeats_with_non_overlap) {
    auto tensor1 = make_tensor(1000, 56, 0, {8, 1}, {5, 1}, 1);
    auto tensor2 = make_tensor(1000, 56, 6, {7, 1}, {5, 1}, 1);

    print_tensor_segments(tensor1, "tensor1");
    print_tensor_segments(tensor2, "tensor2");

    verify_overlap(tensor1, tensor2, false);
}

/**
 * 测试：完全相同的非超矩形 tensor
 */
TEST("额外边界测试", test_identical_non_hyperrect) {
    auto tensor1 = make_tensor(1000, 100, 8, {10, 1}, {3, 6}, 1);
    auto tensor2 = make_tensor(1000, 100, 8, {10, 1}, {3, 6}, 1);

    print_tensor_segments(tensor1, "tensor1");
    print_tensor_segments(tensor2, "tensor2");

    verify_overlap(tensor1, tensor2, true);
}

/**
 * 测试：一个 tensor 完全包含另一个
 */
TEST("额外边界测试", test_one_contains_other) {
    auto outer = make_tensor(1000, 200, 0, {10, 1}, {10, 10}, 1);
    auto inner = make_tensor(1000, 200, 25, {10, 1}, {3, 5}, 1);

    print_tensor_segments(outer, "outer");
    print_tensor_segments(inner, "inner");

    verify_overlap(inner, outer, true);
}

/**
 * 测试：交替块模式（块大小不同）
 */
TEST("额外边界测试", test_alternating_blocks_different_sizes) {
    // 块大小为 3
    auto tensor1 = make_tensor(1000, 200, 0, {10, 1}, {10, 3}, 1);
    // 块大小为 4，从偏移 5 开始
    auto tensor2 = make_tensor(1000, 200, 5, {10, 1}, {10, 4}, 1);

    print_tensor_segments(tensor1, "tensor1");
    print_tensor_segments(tensor2, "tensor2");

    verify_overlap(tensor1, tensor2, false);
}

/**
 * 测试：复杂的 4D 交叉模式
 */
TEST("额外边界测试", test_4d_complex_crossing) {
    auto tensor1 = make_tensor(1000, 50000, 0, {10000, 1000, 100, 1}, {2, 3, 4, 50}, 1);
    auto tensor2 = make_tensor(1000, 50000, 25, {10000, 1000, 100, 1}, {2, 3, 4, 50}, 1);

    bool brute = brute_force_memory_overlap(tensor1, tensor2);
    verify_overlap(tensor1, tensor2, brute);
}

// ==================== 暴力拷贝模拟验证 ====================

/**
 * 计算多维索引对应的一维偏移（row-major 布局）
 */
uint64_t indices_to_offset(const std::vector<uint64_t>& indices, const std::vector<uint64_t>& shape) {
    uint64_t offset = 0;
    uint64_t stride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; i--) {
        offset += indices[i] * stride;
        stride *= shape[i];
    }
    return offset;
}

/**
 * 将一维偏移转换为多维索引（row-major 布局）
 */
std::vector<uint64_t> offset_to_indices(uint64_t offset, const std::vector<uint64_t>& shape) {
    std::vector<uint64_t> indices(shape.size());
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; i--) {
        indices[i] = offset % shape[i];
        offset /= shape[i];
    }
    return indices;
}

/**
 * 计算 shape 的总元素数
 */
uint64_t total_elements(const std::vector<uint64_t>& shape) {
    uint64_t total = 1;
    for (auto s : shape) {
        total *= s;
    }
    return total;
}

/**
 * 模拟 view 操作：从原数据中拷贝子区域
 *
 * @param data 原始数据（按 row-major 布局存储）
 * @param original_shape 原始形状
 * @param view_shape 视图形状
 * @param offsets 各维度的偏移
 * @return 新的数据数组（按 row-major 布局）
 */
template <typename T>
std::vector<T> simulate_view(const std::vector<T>& data,
    const std::vector<uint64_t>& original_shape,
    const std::vector<uint64_t>& view_shape,
    const std::vector<uint64_t>& offsets) {
    uint64_t total = total_elements(view_shape);
    std::vector<T> result(total);

    // 遍历 view_shape 的所有索引
    std::vector<uint64_t> view_indices(view_shape.size(), 0);
    for (uint64_t i = 0; i < total; i++) {
        // 计算原数据中的索引 = offsets + view_indices
        std::vector<uint64_t> orig_indices(view_shape.size());
        for (size_t d = 0; d < view_shape.size(); d++) {
            orig_indices[d] = offsets[d] + view_indices[d];
        }

        // 从原数据中读取
        uint64_t orig_offset = indices_to_offset(orig_indices, original_shape);
        result[i] = data[orig_offset];

        // 递增 view_indices
        for (int d = static_cast<int>(view_shape.size()) - 1; d >= 0; d--) {
            view_indices[d]++;
            if (view_indices[d] < view_shape[d]) {
                break;
            }
            view_indices[d] = 0;
        }
    }

    return result;
}

/**
 * 模拟 reshape 操作：数据不变，只改变逻辑形状
 * reshape 不改变数据的存储顺序，直接返回原数据的拷贝
 */
template <typename T>
std::vector<T> simulate_reshape(const std::vector<T>& data, const std::vector<uint64_t>& new_shape) {
    (void)new_shape;
    // reshape 不改变数据顺序
    return data;
}

/**
 * 模拟 transpose 操作：真实拷贝数据到新布局
 *
 * @param data 原始数据（按 row-major 布局存储）
 * @param shape 原始形状
 * @param dim_x 要交换的第一个维度
 * @param dim_y 要交换的第二个维度
 * @return 新的数据数组（按转置后的 row-major 布局）
 */
template <typename T>
std::vector<T> simulate_transpose(
    const std::vector<T>& data, const std::vector<uint64_t>& shape, uint64_t dim_x, uint64_t dim_y) {
    uint64_t total = total_elements(shape);
    std::vector<T> result(total);

    // 计算转置后的形状
    std::vector<uint64_t> new_shape = shape;
    std::swap(new_shape[dim_x], new_shape[dim_y]);

    // 遍历原数据的所有索引
    std::vector<uint64_t> orig_indices(shape.size(), 0);
    for (uint64_t i = 0; i < total; i++) {
        // 计算转置后的索引（交换 dim_x 和 dim_y）
        std::vector<uint64_t> new_indices = orig_indices;
        std::swap(new_indices[dim_x], new_indices[dim_y]);

        // 计算新位置的偏移
        uint64_t new_offset = indices_to_offset(new_indices, new_shape);
        result[new_offset] = data[i];

        // 递增 orig_indices
        for (int d = static_cast<int>(shape.size()) - 1; d >= 0; d--) {
            orig_indices[d]++;
            if (orig_indices[d] < shape[d]) {
                break;
            }
            orig_indices[d] = 0;
        }
    }

    return result;
}

/**
 * 通过 TensorDescriptor 遍历数据，按逻辑顺序（row-major）收集所有数据
 *
 * @param data 原始数据数组
 * @param desc TensorDescriptor 描述符
 * @return 按逻辑顺序收集的数据
 */
template <typename T>
std::vector<T> collect_tensor_data(const T* data, const TensorDescriptor& desc) {
    uint64_t total = 1;
    for (uint64_t i = 0; i < desc.ndims; i++) {
        total *= desc.repeats[i];
    }

    std::vector<T> result(total);
    std::vector<uint64_t> indices(desc.ndims, 0);

    for (uint64_t i = 0; i < total; i++) {
        // 计算物理偏移 = start_offset + sum(indices[k] * strides[k])
        uint64_t offset = desc.start_offset;
        for (uint64_t d = 0; d < desc.ndims; d++) {
            offset += indices[d] * desc.strides[d];
        }
        result[i] = data[offset];

        // 递增 indices（从最内层开始）
        for (int d = static_cast<int>(desc.ndims) - 1; d >= 0; d--) {
            indices[d]++;
            if (indices[d] < desc.repeats[d]) {
                break;
            }
            indices[d] = 0;
        }
    }

    return result;
}

/**
 * 操作类型枚举
 */
enum class TransformOpType { View, Reshape, Transpose };

/**
 * 变换操作结构
 */
struct TransformOp {
    TransformOpType type;
    std::vector<uint64_t> shapes;   // view_shape 或 reshape_shape
    std::vector<uint64_t> offsets;  // view 的 offsets
    uint64_t dim_x;                 // transpose 的第一个维度
    uint64_t dim_y;                 // transpose 的第二个维度

    static TransformOp make_view(const std::vector<uint64_t>& shapes, const std::vector<uint64_t>& offsets) {
        TransformOp op;
        op.type = TransformOpType::View;
        op.shapes = shapes;
        op.offsets = offsets;
        op.dim_x = 0;
        op.dim_y = 0;
        return op;
    }

    static TransformOp make_reshape(const std::vector<uint64_t>& shapes) {
        TransformOp op;
        op.type = TransformOpType::Reshape;
        op.shapes = shapes;
        op.dim_x = 0;
        op.dim_y = 0;
        return op;
    }

    static TransformOp make_transpose(uint64_t x, uint64_t y) {
        TransformOp op;
        op.type = TransformOpType::Transpose;
        op.dim_x = x;
        op.dim_y = y;
        return op;
    }
};

/**
 * 验证变换序列的正确性
 *
 * @param original_data 原始数据
 * @param original_shape 原始形状
 * @param transformed_desc 变换后的 TensorDescriptor
 * @param ops 操作序列
 * @return 是否验证通过
 */
template <typename T>
bool verify_transform_sequence(const std::vector<T>& original_data,
    const std::vector<uint64_t>& original_shape,
    const TensorDescriptor& transformed_desc,
    const std::vector<TransformOp>& ops) {
    // 1. 暴力模拟：依次应用每个操作，真实拷贝数据
    std::vector<T> simulated_data = original_data;
    std::vector<uint64_t> current_shape = original_shape;

    for (const auto& op : ops) {
        switch (op.type) {
            case TransformOpType::View:
                simulated_data = simulate_view(simulated_data, current_shape, op.shapes, op.offsets);
                current_shape = op.shapes;
                break;
            case TransformOpType::Reshape:
                simulated_data = simulate_reshape(simulated_data, op.shapes);
                current_shape = op.shapes;
                break;
            case TransformOpType::Transpose:
                simulated_data = simulate_transpose(simulated_data, current_shape, op.dim_x, op.dim_y);
                std::swap(current_shape[op.dim_x], current_shape[op.dim_y]);
                break;
        }
    }

    // 2. 通过 TensorDescriptor 遍历原始数据
    std::vector<T> descriptor_data = collect_tensor_data(original_data.data(), transformed_desc);

    // 3. 比较全部数据，必须完全一致
    if (simulated_data.size() != descriptor_data.size()) {
        printf("  [SIZE MISMATCH] simulated=%zu, descriptor=%zu\n", simulated_data.size(), descriptor_data.size());
        return false;
    }

    for (size_t i = 0; i < simulated_data.size(); i++) {
        if (simulated_data[i] != descriptor_data[i]) {
            printf("  [DATA MISMATCH] index=%zu, simulated=%d, descriptor=%d\n",
                i,
                static_cast<int>(simulated_data[i]),
                static_cast<int>(descriptor_data[i]));
            return false;
        }
    }

    return true;
}

/**
 * 打印数据数组（用于调试）
 */
template <typename T>
void print_data(const std::vector<T>& data, const char* name, size_t max_elements = 20) {
    printf("  %s: [", name);
    for (size_t i = 0; i < std::min(data.size(), max_elements); i++) {
        printf("%d", static_cast<int>(data[i]));
        if (i < std::min(data.size(), max_elements) - 1) printf(", ");
    }
    if (data.size() > max_elements) printf(", ...");
    printf("] (size=%zu)\n", data.size());
}

// ==================== 暴力验证测试用例 ====================

/**
 * 测试：view 操作的暴力验证
 */
TEST("暴力变换验证", test_view_brute_force) {
    // 创建 3x4 数据: [0,1,2,3, 4,5,6,7, 8,9,10,11]
    std::vector<int> data(12);
    std::iota(data.begin(), data.end(), 0);

    // 原始 tensor: 3x4, strides=[4,1]
    auto tensor = make_tensor(0, 12, 0, {4, 1}, {3, 4}, 1);

    // view: 取 [1:3, 1:4] 子区域 (2x3)
    auto viewed = tensor.view({2, 3}, {1, 1});

    // 记录操作
    std::vector<TransformOp> ops = {TransformOp::make_view({2, 3}, {1, 1})};

    // 验证
    bool result = verify_transform_sequence(data, {3, 4}, viewed, ops);

    // 打印调试信息
    print_data(data, "original");
    auto simulated = simulate_view(data, {3, 4}, {2, 3}, {1, 1});
    print_data(simulated, "simulated");
    auto from_desc = collect_tensor_data(data.data(), viewed);
    print_data(from_desc, "from_desc");

    ASSERT_TRUE(result);
}

/**
 * 测试：reshape 操作的暴力验证
 */
TEST("暴力变换验证", test_reshape_brute_force) {
    // 创建 3x4 数据
    std::vector<int> data(12);
    std::iota(data.begin(), data.end(), 0);

    auto tensor = make_tensor(0, 12, 0, {4, 1}, {3, 4}, 1);
    auto reshaped = tensor.reshape({2, 6});

    std::vector<TransformOp> ops = {TransformOp::make_reshape({2, 6})};

    bool result = verify_transform_sequence(data, {3, 4}, reshaped, ops);

    print_data(data, "original");
    auto from_desc = collect_tensor_data(data.data(), reshaped);
    print_data(from_desc, "from_desc");

    ASSERT_TRUE(result);
}

/**
 * 测试：transpose 操作的暴力验证
 */
TEST("暴力变换验证", test_transpose_brute_force) {
    // 创建 3x4 数据
    std::vector<int> data(12);
    std::iota(data.begin(), data.end(), 0);

    auto tensor = make_tensor(0, 12, 0, {4, 1}, {3, 4}, 1);
    auto transposed = tensor.transpose(0, 1);

    std::vector<TransformOp> ops = {TransformOp::make_transpose(0, 1)};

    bool result = verify_transform_sequence(data, {3, 4}, transposed, ops);

    print_data(data, "original");
    auto simulated = simulate_transpose(data, {3, 4}, 0, 1);
    print_data(simulated, "simulated (4x3)");
    auto from_desc = collect_tensor_data(data.data(), transposed);
    print_data(from_desc, "from_desc");

    ASSERT_TRUE(result);
}

/**
 * 测试：view -> reshape 序列
 */
TEST("暴力变换验证", test_view_then_reshape) {
    std::vector<int> data(12);
    std::iota(data.begin(), data.end(), 0);

    auto tensor = make_tensor(0, 12, 0, {4, 1}, {3, 4}, 1);
    auto viewed = tensor.view({2, 4}, {0, 0});  // 取前 2 行
    auto reshaped = viewed.reshape({8});        // 展平为 8 个元素

    std::vector<TransformOp> ops = {TransformOp::make_view({2, 4}, {0, 0}), TransformOp::make_reshape({8})};

    bool result = verify_transform_sequence(data, {3, 4}, reshaped, ops);

    print_data(data, "original");
    auto from_desc = collect_tensor_data(data.data(), reshaped);
    print_data(from_desc, "from_desc");

    // 预期: 0,1,2,3,4,5,6,7
    ASSERT_TRUE(result);
    ASSERT_TRUE(from_desc.size() == 8);
    for (int i = 0; i < 8; i++) {
        ASSERT_TRUE(from_desc[i] == i);
    }
}

/**
 * 测试：transpose -> view 序列
 */
TEST("暴力变换验证", test_transpose_then_view) {
    std::vector<int> data(12);
    std::iota(data.begin(), data.end(), 0);

    auto tensor = make_tensor(0, 12, 0, {4, 1}, {3, 4}, 1);
    auto transposed = tensor.transpose(0, 1);       // 变为 4x3
    auto viewed = transposed.view({2, 2}, {1, 1});  // 取子区域

    std::vector<TransformOp> ops = {TransformOp::make_transpose(0, 1), TransformOp::make_view({2, 2}, {1, 1})};

    bool result = verify_transform_sequence(data, {3, 4}, viewed, ops);

    print_data(data, "original");
    auto from_desc = collect_tensor_data(data.data(), viewed);
    print_data(from_desc, "from_desc");

    ASSERT_TRUE(result);
}

/**
 * 测试：双重 transpose 恢复原状态
 */
TEST("暴力变换验证", test_double_transpose) {
    std::vector<int> data(12);
    std::iota(data.begin(), data.end(), 0);

    auto tensor = make_tensor(0, 12, 0, {4, 1}, {3, 4}, 1);
    auto t1 = tensor.transpose(0, 1);  // 3x4 -> 4x3
    auto t2 = t1.transpose(0, 1);      // 4x3 -> 3x4 (恢复)

    std::vector<TransformOp> ops = {TransformOp::make_transpose(0, 1), TransformOp::make_transpose(0, 1)};

    bool result = verify_transform_sequence(data, {3, 4}, t2, ops);

    print_data(data, "original");
    auto from_desc = collect_tensor_data(data.data(), t2);
    print_data(from_desc, "from_desc");

    // 双重 transpose 后应该恢复原数据顺序
    ASSERT_TRUE(result);
    for (int i = 0; i < 12; i++) {
        ASSERT_TRUE(from_desc[i] == i);
    }
}

/**
 * 测试：3D tensor 的 view 操作
 */
TEST("暴力变换验证", test_3d_view) {
    // 创建 2x3x4 数据
    std::vector<int> data(24);
    std::iota(data.begin(), data.end(), 0);

    // 2x3x4, strides=[12, 4, 1]
    auto tensor = make_tensor(0, 24, 0, {12, 4, 1}, {2, 3, 4}, 1);
    auto viewed = tensor.view({1, 2, 3}, {1, 1, 0});

    std::vector<TransformOp> ops = {TransformOp::make_view({1, 2, 3}, {1, 1, 0})};

    bool result = verify_transform_sequence(data, {2, 3, 4}, viewed, ops);

    print_data(data, "original");
    auto from_desc = collect_tensor_data(data.data(), viewed);
    print_data(from_desc, "from_desc");

    ASSERT_TRUE(result);
}

/**
 * 测试：3D tensor 的 transpose 操作
 */
TEST("暴力变换验证", test_3d_transpose) {
    std::vector<int> data(24);
    std::iota(data.begin(), data.end(), 0);

    auto tensor = make_tensor(0, 24, 0, {12, 4, 1}, {2, 3, 4}, 1);
    auto transposed = tensor.transpose(0, 2);  // 2x3x4 -> 4x3x2

    std::vector<TransformOp> ops = {TransformOp::make_transpose(0, 2)};

    bool result = verify_transform_sequence(data, {2, 3, 4}, transposed, ops);

    print_data(data, "original");
    auto from_desc = collect_tensor_data(data.data(), transposed);
    print_data(from_desc, "from_desc");

    ASSERT_TRUE(result);
}

/**
 * 测试：复杂变换序列 view -> transpose -> view
 */
TEST("暴力变换验证", test_complex_sequence_view_transpose_view) {
    std::vector<int> data(24);
    std::iota(data.begin(), data.end(), 0);

    auto tensor = make_tensor(0, 24, 0, {12, 4, 1}, {2, 3, 4}, 1);
    auto t1 = tensor.view({2, 2, 3}, {0, 1, 0});
    auto t2 = t1.transpose(0, 2);
    auto t3 = t2.view({2, 2, 2}, {0, 0, 0});

    std::vector<TransformOp> ops = {TransformOp::make_view({2, 2, 3}, {0, 1, 0}),
        TransformOp::make_transpose(0, 2),
        TransformOp::make_view({2, 2, 2}, {0, 0, 0})};

    bool result = verify_transform_sequence(data, {2, 3, 4}, t3, ops);

    print_data(data, "original");
    auto from_desc = collect_tensor_data(data.data(), t3);
    print_data(from_desc, "from_desc");

    ASSERT_TRUE(result);
}

/**
 * 测试：reshape -> view 序列
 */
TEST("暴力变换验证", test_reshape_then_view) {
    std::vector<int> data(12);
    std::iota(data.begin(), data.end(), 0);

    auto tensor = make_tensor(0, 12, 0, {4, 1}, {3, 4}, 1);
    auto reshaped = tensor.reshape({6, 2});       // 3x4 -> 6x2
    auto viewed = reshaped.view({3, 2}, {2, 0});  // 取子区域

    std::vector<TransformOp> ops = {TransformOp::make_reshape({6, 2}), TransformOp::make_view({3, 2}, {2, 0})};

    bool result = verify_transform_sequence(data, {3, 4}, viewed, ops);

    print_data(data, "original");
    auto from_desc = collect_tensor_data(data.data(), viewed);
    print_data(from_desc, "from_desc");

    ASSERT_TRUE(result);
}

/**
 * 测试：多次 view 操作
 */
TEST("暴力变换验证", test_multiple_views) {
    std::vector<int> data(24);
    std::iota(data.begin(), data.end(), 0);

    auto tensor = make_tensor(0, 24, 0, {12, 4, 1}, {2, 3, 4}, 1);
    auto v1 = tensor.view({2, 2, 3}, {0, 1, 1});
    auto v2 = v1.view({1, 2, 2}, {1, 0, 0});

    std::vector<TransformOp> ops = {
        TransformOp::make_view({2, 2, 3}, {0, 1, 1}), TransformOp::make_view({1, 2, 2}, {1, 0, 0})};

    bool result = verify_transform_sequence(data, {2, 3, 4}, v2, ops);

    print_data(data, "original");
    auto from_desc = collect_tensor_data(data.data(), v2);
    print_data(from_desc, "from_desc");

    ASSERT_TRUE(result);
}

/**
 * 测试：4D tensor 复杂变换
 */
TEST("暴力变换验证", test_4d_complex_transform) {
    // 创建 2x2x3x2 数据
    std::vector<int> data(24);
    std::iota(data.begin(), data.end(), 0);

    // 2x2x3x2, strides=[12, 6, 2, 1]
    auto tensor = make_tensor(0, 24, 0, {12, 6, 2, 1}, {2, 2, 3, 2}, 1);
    auto t1 = tensor.transpose(1, 3);  // 交换 dim1 和 dim3
    auto v1 = t1.view({2, 2, 2, 2}, {0, 0, 1, 0});

    std::vector<TransformOp> ops = {
        TransformOp::make_transpose(1, 3), TransformOp::make_view({2, 2, 2, 2}, {0, 0, 1, 0})};

    bool result = verify_transform_sequence(data, {2, 2, 3, 2}, v1, ops);

    print_data(data, "original");
    auto from_desc = collect_tensor_data(data.data(), v1);
    print_data(from_desc, "from_desc");

    ASSERT_TRUE(result);
}

/**
 * 测试：带偏移的 tensor view
 */
TEST("暴力变换验证", test_view_with_start_offset) {
    std::vector<int> data(20);
    std::iota(data.begin(), data.end(), 0);

    // 从 offset=2 开始的 3x4 tensor
    auto tensor = make_tensor(0, 20, 2, {4, 1}, {3, 4}, 1);
    auto viewed = tensor.view({2, 3}, {1, 0});

    // 注意：对于带 start_offset 的 tensor，我们需要调整验证方式
    // 这里直接验证 collect_tensor_data 的结果
    auto from_desc = collect_tensor_data(data.data(), viewed);
    print_data(from_desc, "from_desc");

    // 预期: tensor 从 offset=2 开始，view 从 (1,0) 开始取 2x3
    // 原始数据布局: [2,3,4,5], [6,7,8,9], [10,11,12,13]
    // view 取 row 1-2, col 0-2: [6,7,8], [10,11,12]
    std::vector<int> expected = {6, 7, 8, 10, 11, 12};
    ASSERT_TRUE(from_desc == expected);
}

// ==================== 复杂操作序列测试 ====================

/**
 * 测试：5步复杂变换序列（不含 reshape，因为 view/transpose 后通常不连续）
 * view -> transpose -> view -> transpose -> view
 */
TEST("复杂操作序列", test_5_step_sequence) {
    // 创建 4x6x8 数据
    std::vector<int> data(192);
    std::iota(data.begin(), data.end(), 0);

    // 4x6x8, strides=[48, 8, 1]
    auto tensor = make_tensor(0, 192, 0, {48, 8, 1}, {4, 6, 8}, 1);

    // 步骤1: view 取 3x5x7
    auto t1 = tensor.view({3, 5, 7}, {1, 1, 1});
    // 步骤2: transpose(0, 2) -> 7x5x3
    auto t2 = t1.transpose(0, 2);
    // 步骤3: view 取 5x4x2
    auto t3 = t2.view({5, 4, 2}, {1, 1, 1});
    // 步骤4: transpose(1, 2) -> 5x2x4
    auto t4 = t3.transpose(1, 2);
    // 步骤5: view 取 4x2x3
    auto t5 = t4.view({4, 2, 3}, {1, 0, 1});

    std::vector<TransformOp> ops = {TransformOp::make_view({3, 5, 7}, {1, 1, 1}),
        TransformOp::make_transpose(0, 2),
        TransformOp::make_view({5, 4, 2}, {1, 1, 1}),
        TransformOp::make_transpose(1, 2),
        TransformOp::make_view({4, 2, 3}, {1, 0, 1})};

    bool result = verify_transform_sequence(data, {4, 6, 8}, t5, ops);

    printf("  5-step sequence (view-transpose): %s\n", result ? "PASSED" : "FAILED");
    auto from_desc = collect_tensor_data(data.data(), t5);
    print_data(from_desc, "from_desc");

    ASSERT_TRUE(result);
}

/**
 * 测试：多次 transpose 交换不同维度
 */
TEST("复杂操作序列", test_multiple_transpose_different_dims) {
    // 创建 2x3x4x5 数据
    std::vector<int> data(120);
    std::iota(data.begin(), data.end(), 0);

    // 2x3x4x5, strides=[60, 20, 5, 1]
    auto tensor = make_tensor(0, 120, 0, {60, 20, 5, 1}, {2, 3, 4, 5}, 1);

    // transpose(0,1) -> 3x2x4x5
    auto t1 = tensor.transpose(0, 1);
    // transpose(2,3) -> 3x2x5x4
    auto t2 = t1.transpose(2, 3);
    // transpose(1,2) -> 3x5x2x4
    auto t3 = t2.transpose(1, 2);

    std::vector<TransformOp> ops = {
        TransformOp::make_transpose(0, 1), TransformOp::make_transpose(2, 3), TransformOp::make_transpose(1, 2)};

    bool result = verify_transform_sequence(data, {2, 3, 4, 5}, t3, ops);

    printf("  Multiple transpose: shape should be 3x5x2x4\n");
    printf("  Result shape: %lux%lux%lux%lu\n", t3.repeats[0], t3.repeats[1], t3.repeats[2], t3.repeats[3]);

    ASSERT_TRUE(result);
    ASSERT_TRUE(t3.repeats[0] == 3 && t3.repeats[1] == 5 && t3.repeats[2] == 2 && t3.repeats[3] == 4);
}

/**
 * 测试：交替 view 和 transpose
 */
TEST("复杂操作序列", test_alternating_view_transpose) {
    std::vector<int> data(120);
    std::iota(data.begin(), data.end(), 0);

    auto tensor = make_tensor(0, 120, 0, {60, 20, 5, 1}, {2, 3, 4, 5}, 1);

    // view -> transpose -> view -> transpose -> view
    auto t1 = tensor.view({2, 3, 3, 4}, {0, 0, 1, 1});
    auto t2 = t1.transpose(0, 3);  // 4x3x3x2
    auto t3 = t2.view({3, 2, 3, 2}, {1, 1, 0, 0});
    auto t4 = t3.transpose(1, 2);  // 3x3x2x2
    auto t5 = t4.view({2, 2, 2, 2}, {1, 0, 0, 0});

    std::vector<TransformOp> ops = {TransformOp::make_view({2, 3, 3, 4}, {0, 0, 1, 1}),
        TransformOp::make_transpose(0, 3),
        TransformOp::make_view({3, 2, 3, 2}, {1, 1, 0, 0}),
        TransformOp::make_transpose(1, 2),
        TransformOp::make_view({2, 2, 2, 2}, {1, 0, 0, 0})};

    bool result = verify_transform_sequence(data, {2, 3, 4, 5}, t5, ops);

    auto from_desc = collect_tensor_data(data.data(), t5);
    print_data(from_desc, "from_desc");

    ASSERT_TRUE(result);
}

/**
 * 测试：reshape 链
 */
TEST("复杂操作序列", test_reshape_chain) {
    std::vector<int> data(120);
    std::iota(data.begin(), data.end(), 0);

    auto tensor = make_tensor(0, 120, 0, {60, 20, 5, 1}, {2, 3, 4, 5}, 1);

    // 多次 reshape
    auto t1 = tensor.reshape({6, 20});
    auto t2 = t1.reshape({3, 40});
    auto t3 = t2.reshape({12, 10});
    auto t4 = t3.reshape({2, 6, 10});
    auto t5 = t4.reshape({120});

    std::vector<TransformOp> ops = {TransformOp::make_reshape({6, 20}),
        TransformOp::make_reshape({3, 40}),
        TransformOp::make_reshape({12, 10}),
        TransformOp::make_reshape({2, 6, 10}),
        TransformOp::make_reshape({120})};

    bool result = verify_transform_sequence(data, {2, 3, 4, 5}, t5, ops);

    auto from_desc = collect_tensor_data(data.data(), t5);
    // 展平后应该是 0,1,2,...,119
    ASSERT_TRUE(result);
    ASSERT_TRUE(from_desc.size() == 120);
    for (int i = 0; i < 120; i++) {
        ASSERT_TRUE(from_desc[i] == i);
    }
}

/**
 * 测试：5D tensor 复杂变换
 */
TEST("复杂操作序列", test_5d_complex_transform) {
    // 创建 2x2x3x2x3 数据
    std::vector<int> data(72);
    std::iota(data.begin(), data.end(), 0);

    // strides = [36, 18, 6, 3, 1]
    auto tensor = make_tensor(0, 72, 0, {36, 18, 6, 3, 1}, {2, 2, 3, 2, 3}, 1);

    // transpose(1, 4) -> 2x3x3x2x2
    auto t1 = tensor.transpose(1, 4);
    // view
    auto t2 = t1.view({2, 2, 2, 2, 2}, {0, 1, 1, 0, 0});
    // transpose(0, 2)
    auto t3 = t2.transpose(0, 2);

    std::vector<TransformOp> ops = {TransformOp::make_transpose(1, 4),
        TransformOp::make_view({2, 2, 2, 2, 2}, {0, 1, 1, 0, 0}),
        TransformOp::make_transpose(0, 2)};

    bool result = verify_transform_sequence(data, {2, 2, 3, 2, 3}, t3, ops);

    auto from_desc = collect_tensor_data(data.data(), t3);
    print_data(from_desc, "from_desc");

    ASSERT_TRUE(result);
}

// ==================== 大规模数据测试 ====================

/**
 * 测试：大规模 2D tensor (1000x1000)
 */
TEST("大规模数据", test_large_2d_tensor) {
    const size_t N = 1000;
    std::vector<int> data(N * N);
    std::iota(data.begin(), data.end(), 0);

    // 1000x1000, strides=[1000, 1]
    auto tensor = make_tensor(0, N * N, 0, {N, 1}, {N, N}, 1);

    // view 取中间 500x500
    auto viewed = tensor.view({500, 500}, {250, 250});

    std::vector<TransformOp> ops = {TransformOp::make_view({500, 500}, {250, 250})};

    bool result = verify_transform_sequence(data, {N, N}, viewed, ops);

    printf("  Large 2D view: 1000x1000 -> 500x500 (offset 250,250)\n");
    printf("  Total elements verified: %zu\n", 500UL * 500);

    ASSERT_TRUE(result);
}

/**
 * 测试：大规模 3D tensor transpose
 */
TEST("大规模数据", test_large_3d_transpose) {
    const size_t D1 = 50, D2 = 60, D3 = 70;
    std::vector<int> data(D1 * D2 * D3);
    std::iota(data.begin(), data.end(), 0);

    // 50x60x70, strides=[4200, 70, 1]
    auto tensor = make_tensor(0, D1 * D2 * D3, 0, {D2 * D3, D3, 1}, {D1, D2, D3}, 1);

    // transpose(0, 2) -> 70x60x50
    auto transposed = tensor.transpose(0, 2);

    std::vector<TransformOp> ops = {TransformOp::make_transpose(0, 2)};

    bool result = verify_transform_sequence(data, {D1, D2, D3}, transposed, ops);

    printf("  Large 3D transpose: 50x60x70 -> 70x60x50\n");
    printf("  Total elements verified: %zu\n", D1 * D2 * D3);

    ASSERT_TRUE(result);
    ASSERT_TRUE(transposed.repeats[0] == D3);
    ASSERT_TRUE(transposed.repeats[1] == D2);
    ASSERT_TRUE(transposed.repeats[2] == D1);
}

/**
 * 测试：大规模 4D tensor 复杂变换
 */
TEST("大规模数据", test_large_4d_complex) {
    const size_t D1 = 10, D2 = 20, D3 = 30, D4 = 40;
    std::vector<int> data(D1 * D2 * D3 * D4);
    std::iota(data.begin(), data.end(), 0);

    // 10x20x30x40, strides=[24000, 1200, 40, 1]
    auto tensor = make_tensor(0, D1 * D2 * D3 * D4, 0, {D2 * D3 * D4, D3 * D4, D4, 1}, {D1, D2, D3, D4}, 1);

    // view -> transpose -> view
    auto t1 = tensor.view({8, 15, 25, 35}, {1, 2, 3, 2});
    auto t2 = t1.transpose(1, 3);  // 8x35x25x15
    auto t3 = t2.view({6, 30, 20, 10}, {1, 2, 3, 2});

    std::vector<TransformOp> ops = {TransformOp::make_view({8, 15, 25, 35}, {1, 2, 3, 2}),
        TransformOp::make_transpose(1, 3),
        TransformOp::make_view({6, 30, 20, 10}, {1, 2, 3, 2})};

    bool result = verify_transform_sequence(data, {D1, D2, D3, D4}, t3, ops);

    printf("  Large 4D complex: 10x20x30x40 -> view -> transpose -> view\n");
    printf("  Final shape: %lux%lux%lux%lu\n", t3.repeats[0], t3.repeats[1], t3.repeats[2], t3.repeats[3]);
    printf("  Total elements verified: %zu\n", 6UL * 30 * 20 * 10);

    ASSERT_TRUE(result);
}

/**
 * 测试：大规模 reshape 链
 */
TEST("大规模数据", test_large_reshape_chain) {
    const size_t TOTAL = 100000;
    std::vector<int> data(TOTAL);
    std::iota(data.begin(), data.end(), 0);

    // 100000 元素
    auto tensor = make_tensor(0, TOTAL, 0, {1}, {TOTAL}, 1);

    // 多次 reshape
    auto t1 = tensor.reshape({100, 1000});
    auto t2 = t1.reshape({10, 10, 1000});
    auto t3 = t2.reshape({10, 10, 10, 100});
    auto t4 = t3.reshape({10, 10, 10, 10, 10});
    auto t5 = t4.reshape({100000});

    std::vector<TransformOp> ops = {TransformOp::make_reshape({100, 1000}),
        TransformOp::make_reshape({10, 10, 1000}),
        TransformOp::make_reshape({10, 10, 10, 100}),
        TransformOp::make_reshape({10, 10, 10, 10, 10}),
        TransformOp::make_reshape({100000})};

    bool result = verify_transform_sequence(data, {TOTAL}, t5, ops);

    printf("  Large reshape chain: 100000 elements through 5 reshapes\n");

    // 最终应该恢复原顺序
    auto from_desc = collect_tensor_data(data.data(), t5);
    ASSERT_TRUE(result);
    ASSERT_TRUE(from_desc.size() == TOTAL);
    // 抽样检查
    ASSERT_TRUE(from_desc[0] == 0);
    ASSERT_TRUE(from_desc[50000] == 50000);
    ASSERT_TRUE(from_desc[99999] == 99999);
}

/**
 * 测试：大规模多步变换（不含 reshape，避免非连续问题）
 */
TEST("大规模数据", test_large_multi_step) {
    const size_t D1 = 20, D2 = 25, D3 = 30;
    std::vector<int> data(D1 * D2 * D3);
    std::iota(data.begin(), data.end(), 0);

    // 20x25x30
    auto tensor = make_tensor(0, D1 * D2 * D3, 0, {D2 * D3, D3, 1}, {D1, D2, D3}, 1);

    // 6步变换（view 和 transpose 交替）
    auto t1 = tensor.view({18, 20, 25}, {1, 2, 3});
    auto t2 = t1.transpose(0, 2);  // 25x20x18
    auto t3 = t2.view({20, 15, 15}, {2, 3, 2});
    auto t4 = t3.transpose(1, 2);  // 20x15x15
    auto t5 = t4.view({15, 12, 12}, {3, 2, 2});
    auto t6 = t5.transpose(0, 1);  // 12x15x12

    std::vector<TransformOp> ops = {TransformOp::make_view({18, 20, 25}, {1, 2, 3}),
        TransformOp::make_transpose(0, 2),
        TransformOp::make_view({20, 15, 15}, {2, 3, 2}),
        TransformOp::make_transpose(1, 2),
        TransformOp::make_view({15, 12, 12}, {3, 2, 2}),
        TransformOp::make_transpose(0, 1)};

    bool result = verify_transform_sequence(data, {D1, D2, D3}, t6, ops);

    printf("  Large 6-step transform: 20x25x30 through 6 operations\n");
    printf("  Final shape: %lux%lux%lu\n", t6.repeats[0], t6.repeats[1], t6.repeats[2]);
    printf("  Total elements verified: %lu\n", t6.repeats[0] * t6.repeats[1] * t6.repeats[2]);

    ASSERT_TRUE(result);
}

/**
 * 测试：边界情况 - 单元素 tensor
 */
TEST("大规模数据", test_single_element) {
    std::vector<int> data = {42};

    auto tensor = make_tensor(0, 1, 0, {1}, {1}, 1);

    // reshape 到不同维度
    auto t1 = tensor.reshape({1, 1});
    auto t2 = t1.reshape({1, 1, 1});
    auto t3 = t2.transpose(0, 2);
    auto t4 = t3.view({1, 1, 1}, {0, 0, 0});

    std::vector<TransformOp> ops = {TransformOp::make_reshape({1, 1}),
        TransformOp::make_reshape({1, 1, 1}),
        TransformOp::make_transpose(0, 2),
        TransformOp::make_view({1, 1, 1}, {0, 0, 0})};

    bool result = verify_transform_sequence(data, {1}, t4, ops);

    auto from_desc = collect_tensor_data(data.data(), t4);
    ASSERT_TRUE(result);
    ASSERT_TRUE(from_desc.size() == 1);
    ASSERT_TRUE(from_desc[0] == 42);
}

/**
 * 测试：大规模 - 验证内存访问模式
 */
TEST("大规模数据", test_large_strided_access) {
    const size_t D1 = 100, D2 = 100, D3 = 100;
    std::vector<int> data(D1 * D2 * D3);
    std::iota(data.begin(), data.end(), 0);

    // 100x100x100
    auto tensor = make_tensor(0, D1 * D2 * D3, 0, {D2 * D3, D3, 1}, {D1, D2, D3}, 1);

    // transpose 创建非连续访问模式
    auto t1 = tensor.transpose(0, 2);  // 100x100x100 但 strides 变化
    auto t2 = t1.view({50, 50, 50}, {25, 25, 25});

    std::vector<TransformOp> ops = {
        TransformOp::make_transpose(0, 2), TransformOp::make_view({50, 50, 50}, {25, 25, 25})};

    bool result = verify_transform_sequence(data, {D1, D2, D3}, t2, ops);

    printf("  Large strided access: 100x100x100 -> transpose -> view 50x50x50\n");
    printf("  Total elements verified: %zu\n", 50UL * 50 * 50);

    ASSERT_TRUE(result);
}

// ==================== 变换后 Overlap 测试 ====================

/**
 * 测试：View vs View - 同基地址两个 view 有交集
 */
TEST("变换后Overlap-ViewVsView", test_view_vs_view_same_base_overlap) {
    // 创建 10x10 的 tensor，size=100
    auto base = make_tensor(0, 100, 0, {10, 1}, {10, 10}, 1);

    // view1: 从 (0,0) 开始取 5x5
    auto view1 = base.view({5, 5}, {0, 0});

    // view2: 从 (2,2) 开始取 5x5，与 view1 有重叠
    auto view2 = base.view({5, 5}, {2, 2});

    verify_overlap(view1, view2, true);
}

/**
 * 测试：View vs View - 同基地址两个 view 无交集
 */
TEST("变换后Overlap-ViewVsView", test_view_vs_view_same_base_no_overlap) {
    // 创建 10x10 的 tensor，size=100
    auto base = make_tensor(0, 100, 0, {10, 1}, {10, 10}, 1);

    // view1: 从 (0,0) 开始取 4x4
    auto view1 = base.view({4, 4}, {0, 0});

    // view2: 从 (5,5) 开始取 4x4，与 view1 无重叠
    auto view2 = base.view({4, 4}, {5, 5});

    verify_overlap(view1, view2, false);
}

/**
 * 测试：View vs View - 3D tensor 的 view 重叠
 */
TEST("变换后Overlap-ViewVsView", test_view_vs_view_3d_overlap) {
    // 创建 8x8x8 的 tensor，size=512
    auto base = make_tensor(0, 512, 0, {64, 8, 1}, {8, 8, 8}, 1);

    // view1: 从 (0,0,0) 开始取 4x4x4
    auto view1 = base.view({4, 4, 4}, {0, 0, 0});

    // view2: 从 (2,2,2) 开始取 4x4x4，与 view1 有重叠
    auto view2 = base.view({4, 4, 4}, {2, 2, 2});

    verify_overlap(view1, view2, true);
}

/**
 * 测试：View vs View - 边界刚好相邻（无重叠）
 */
TEST("变换后Overlap-ViewVsView", test_view_vs_view_adjacent_boundary) {
    // 创建 10x10 的 tensor
    auto base = make_tensor(0, 100, 0, {10, 1}, {10, 10}, 1);

    // view1: 行 0-4
    auto view1 = base.view({5, 10}, {0, 0});

    // view2: 行 5-9，刚好相邻
    auto view2 = base.view({5, 10}, {5, 0});

    verify_overlap(view1, view2, false);
}

/**
 * 测试：View vs View - 单元素重叠
 */
TEST("变换后Overlap-ViewVsView", test_view_vs_view_single_element_overlap) {
    // 创建 10x10 的 tensor
    auto base = make_tensor(0, 100, 0, {10, 1}, {10, 10}, 1);

    // view1: 从 (0,0) 开始取 5x5，覆盖 (4,4)
    auto view1 = base.view({5, 5}, {0, 0});

    // view2: 从 (4,4) 开始取 5x5，(4,4) 是唯一重叠点
    auto view2 = base.view({5, 5}, {4, 4});

    verify_overlap(view1, view2, true);
}

/**
 * 测试：Reshape vs Reshape - reshape 后维度相同有重叠
 */
TEST("变换后Overlap-ReshapeVsReshape", test_reshape_vs_reshape_same_dims_overlap) {
    // 创建 1D tensor，24 元素
    auto base = make_tensor(0, 24, 0, {1}, {24}, 1);

    // reshape 成 4x6
    auto t1 = base.reshape({4, 6});

    // 同样 reshape 成 4x6，完全重叠
    auto t2 = base.reshape({4, 6});

    verify_overlap(t1, t2, true);
}

/**
 * 测试：Reshape vs Reshape - reshape 后 view 访问不同区域
 */
TEST("变换后Overlap-ReshapeVsReshape", test_reshape_vs_reshape_diff_view_no_overlap) {
    // 创建 1D tensor，24 元素
    auto base = make_tensor(0, 24, 0, {1}, {24}, 1);

    // reshape 成 4x6，然后取前 2 行
    auto t1 = base.reshape({4, 6}).view({2, 6}, {0, 0});

    // reshape 成 4x6，然后取后 2 行
    auto t2 = base.reshape({4, 6}).view({2, 6}, {2, 0});

    verify_overlap(t1, t2, false);
}

/**
 * 测试：Reshape vs Reshape - reshape 到高维后重叠
 */
TEST("变换后Overlap-ReshapeVsReshape", test_reshape_to_higher_dim_overlap) {
    // 创建 1D tensor，60 元素
    auto base = make_tensor(0, 60, 0, {1}, {60}, 1);

    // reshape 成 3x4x5
    auto t1 = base.reshape({3, 4, 5});

    // reshape 成 5x12，然后取部分
    auto t2 = base.reshape({5, 12}).view({3, 10}, {1, 1});

    // 使用暴力方法验证
    bool brute_result = brute_force_memory_overlap(t1, t2);
    verify_overlap(t1, t2, brute_result);
}

/**
 * 测试：Transpose vs Transpose - 相同 transpose 后完全重叠
 */
TEST("变换后Overlap-TransposeVsTranspose", test_transpose_vs_transpose_same_overlap) {
    // 创建 4x6 的 tensor
    auto base = make_tensor(0, 24, 0, {6, 1}, {4, 6}, 1);

    // 两个都做相同的 transpose
    auto t1 = base.transpose(0, 1);
    auto t2 = base.transpose(0, 1);

    verify_overlap(t1, t2, true);
}

/**
 * 测试：Transpose vs Transpose - 不同 transpose 后有重叠
 */
TEST("变换后Overlap-TransposeVsTranspose", test_transpose_vs_transpose_diff_overlap) {
    // 创建 3x4x5 的 tensor
    auto base = make_tensor(0, 60, 0, {20, 5, 1}, {3, 4, 5}, 1);

    // t1: transpose(0,1) -> 4x3x5
    auto t1 = base.transpose(0, 1);

    // t2: transpose(1,2) -> 3x5x4
    auto t2 = base.transpose(1, 2);

    // 两者访问相同的底层内存，应该有重叠
    verify_overlap(t1, t2, true);
}

/**
 * 测试：Transpose vs Transpose - transpose 后 view 无重叠
 */
TEST("变换后Overlap-TransposeVsTranspose", test_transpose_then_view_no_overlap) {
    // 创建 6x8 的 tensor
    auto base = make_tensor(0, 48, 0, {8, 1}, {6, 8}, 1);

    // t1: transpose 后取前半部分
    auto t1 = base.transpose(0, 1).view({4, 6}, {0, 0});

    // t2: transpose 后取后半部分
    auto t2 = base.transpose(0, 1).view({4, 6}, {4, 0});

    // 使用暴力方法验证
    bool brute_result = brute_force_memory_overlap(t1, t2);
    verify_overlap(t1, t2, brute_result);
}

/**
 * 测试：View vs Reshape - view 与 reshape 后有重叠
 */
TEST("变换后Overlap-ViewVsReshape", test_view_vs_reshape_overlap) {
    // 创建 1D tensor，24 元素
    auto base = make_tensor(0, 24, 0, {1}, {24}, 1);

    // t1: view 取前 12 个元素
    auto t1 = base.view({12}, {0});

    // t2: reshape 成 4x6，访问全部 24 元素
    auto t2 = base.reshape({4, 6});

    // t1 是 t2 的子集，应该有重叠
    verify_overlap(t1, t2, true);
}

/**
 * 测试：View vs Reshape - view 与 reshape 后无重叠
 */
TEST("变换后Overlap-ViewVsReshape", test_view_vs_reshape_no_overlap) {
    // 创建 1D tensor，24 元素
    auto base = make_tensor(0, 24, 0, {1}, {24}, 1);

    // t1: view 取前 10 个元素 (索引 0-9)
    auto t1 = base.view({10}, {0});

    // t2: view 取后 10 个元素 (索引 14-23)
    auto t2 = base.view({10}, {14});

    // 两个区域不重叠
    verify_overlap(t1, t2, false);
}

/**
 * 测试：View vs Transpose - view 子集与 transpose 全集重叠
 */
TEST("变换后Overlap-ViewVsTranspose", test_view_vs_transpose_overlap) {
    // 创建 4x6 的 tensor
    auto base = make_tensor(0, 24, 0, {6, 1}, {4, 6}, 1);

    // t1: view 取 2x3 子区域
    auto t1 = base.view({2, 3}, {1, 2});

    // t2: transpose 后访问全部
    auto t2 = base.transpose(0, 1);

    // t1 是 base 的子集，t2 访问全部，应该有重叠
    verify_overlap(t1, t2, true);
}

/**
 * 测试：View vs Transpose - view 与 transpose+view 无重叠
 */
TEST("变换后Overlap-ViewVsTranspose", test_view_vs_transpose_view_no_overlap) {
    // 创建 6x8 的 tensor
    auto base = make_tensor(0, 48, 0, {8, 1}, {6, 8}, 1);

    // t1: view 取左上角 3x4
    auto t1 = base.view({3, 4}, {0, 0});

    // t2: transpose 后 view 取右下角区域
    auto t2 = base.transpose(0, 1).view({4, 3}, {4, 3});

    // 使用暴力方法验证
    bool brute_result = brute_force_memory_overlap(t1, t2);
    verify_overlap(t1, t2, brute_result);
}

/**
 * 测试：Reshape vs Transpose - reshape 和 transpose 后重叠
 */
TEST("变换后Overlap-ReshapeVsTranspose", test_reshape_vs_transpose_overlap) {
    // 创建 1D tensor，24 元素
    auto base = make_tensor(0, 24, 0, {1}, {24}, 1);

    // t1: reshape 成 4x6
    auto t1 = base.reshape({4, 6});

    // t2: reshape 成 6x4 然后 transpose
    auto t2 = base.reshape({6, 4}).transpose(0, 1);

    // 两者访问相同的底层内存
    verify_overlap(t1, t2, true);
}

/**
 * 测试：Reshape vs Transpose - 复杂变换后无重叠
 */
TEST("变换后Overlap-ReshapeVsTranspose", test_reshape_view_vs_transpose_view_no_overlap) {
    // 创建 1D tensor，48 元素
    auto base = make_tensor(0, 48, 0, {1}, {48}, 1);

    // t1: reshape 成 6x8，取前 3 行
    auto t1 = base.reshape({6, 8}).view({3, 8}, {0, 0});

    // t2: reshape 成 8x6，transpose，取后 3 行
    auto t2 = base.reshape({8, 6}).transpose(0, 1).view({3, 8}, {3, 0});

    // 使用暴力方法验证
    bool brute_result = brute_force_memory_overlap(t1, t2);
    verify_overlap(t1, t2, brute_result);
}

/**
 * 测试：复合变换 - view->transpose vs view->transpose
 */
TEST("变换后Overlap-复合变换", test_compound_view_transpose_overlap) {
    // 创建 8x8 的 tensor
    auto base = make_tensor(0, 64, 0, {8, 1}, {8, 8}, 1);

    // t1: view(6x6, offset 0,0) -> transpose
    auto t1 = base.view({6, 6}, {0, 0}).transpose(0, 1);

    // t2: view(6x6, offset 1,1) -> transpose
    auto t2 = base.view({6, 6}, {1, 1}).transpose(0, 1);

    // 使用暴力方法验证
    bool brute_result = brute_force_memory_overlap(t1, t2);
    verify_overlap(t1, t2, brute_result);
}

/**
 * 测试：复合变换 - reshape->view vs transpose->view
 */
TEST("变换后Overlap-复合变换", test_compound_reshape_view_vs_transpose_view) {
    // 创建 6x8 的 tensor
    auto base = make_tensor(0, 48, 0, {8, 1}, {6, 8}, 1);

    // t1: reshape(12x4) -> view(6x4, offset 0,0)
    auto t1 = base.reshape({12, 4}).view({6, 4}, {0, 0});

    // t2: transpose -> view(4x3, offset 2,2)
    auto t2 = base.transpose(0, 1).view({4, 3}, {2, 2});

    // 使用暴力方法验证
    bool brute_result = brute_force_memory_overlap(t1, t2);
    verify_overlap(t1, t2, brute_result);
}

/**
 * 测试：复合变换 - 三步变换 vs 三步变换
 */
TEST("变换后Overlap-复合变换", test_compound_3_step_vs_3_step) {
    // 创建 6x8x10 的 tensor
    auto base = make_tensor(0, 480, 0, {80, 10, 1}, {6, 8, 10}, 1);

    // t1: view(4x6x8) -> transpose(0,2) -> view(6x4x3)
    auto t1 = base.view({4, 6, 8}, {0, 0, 0}).transpose(0, 2).view({6, 4, 3}, {0, 0, 0});

    // t2: transpose(1,2) -> view(5x8x6) -> transpose(0,1)
    auto t2 = base.transpose(1, 2).view({5, 8, 6}, {0, 0, 0}).transpose(0, 1);

    // 使用暴力方法验证
    bool brute_result = brute_force_memory_overlap(t1, t2);
    verify_overlap(t1, t2, brute_result);
}

/**
 * 测试：复合变换 - 五步变换序列
 */
TEST("变换后Overlap-复合变换", test_compound_5_step_sequence) {
    // 创建 8x10x6 的 tensor
    auto base = make_tensor(0, 480, 0, {60, 6, 1}, {8, 10, 6}, 1);

    // t1: 简化的多步变换
    auto t1 = base.view({6, 8, 5}, {0, 0, 0}).transpose(0, 1).view({6, 5, 4}, {0, 0, 0});

    // t2: 不同路径的多步变换
    auto t2 = base.transpose(0, 2).view({5, 8, 6}, {0, 0, 0}).transpose(0, 1);

    // 使用暴力方法验证
    bool brute_result = brute_force_memory_overlap(t1, t2);
    verify_overlap(t1, t2, brute_result);
}

// ==================== 复合变换补充测试 ====================

/**
 * 测试：复合变换 - transpose->view vs view->transpose (顺序不同)
 */
TEST("变换后Overlap-复合变换", test_compound_transpose_view_vs_view_transpose) {
    // 创建 8x6 的 tensor
    auto base = make_tensor(0, 48, 0, {6, 1}, {8, 6}, 1);

    // t1: 先 transpose 再 view
    auto t1 = base.transpose(0, 1).view({4, 6}, {0, 0});

    // t2: 先 view 再 transpose
    auto t2 = base.view({6, 4}, {0, 0}).transpose(0, 1);

    // 使用暴力方法验证
    bool brute_result = brute_force_memory_overlap(t1, t2);
    verify_overlap(t1, t2, brute_result);
}

/**
 * 测试：复合变换 - reshape->transpose vs transpose (不同路径到相似结果)
 */
TEST("变换后Overlap-复合变换", test_compound_reshape_transpose_vs_transpose) {
    // 创建 1D tensor，24 元素
    auto base = make_tensor(0, 24, 0, {1}, {24}, 1);

    // t1: reshape(4x6) -> transpose -> view(4x4)
    auto t1 = base.reshape({4, 6}).transpose(0, 1).view({4, 4}, {0, 0});

    // t2: reshape(6x4) -> view(4x4)
    auto t2 = base.reshape({6, 4}).view({4, 4}, {0, 0});

    // 使用暴力方法验证
    bool brute_result = brute_force_memory_overlap(t1, t2);
    verify_overlap(t1, t2, brute_result);
}

/**
 * 测试：复合变换 - 双重 transpose
 */
TEST("变换后Overlap-复合变换", test_compound_double_transpose) {
    // 创建 4x6x8 的 tensor
    auto base = make_tensor(0, 192, 0, {48, 8, 1}, {4, 6, 8}, 1);

    // t1: transpose(0,1) -> transpose(1,2)
    auto t1 = base.transpose(0, 1).transpose(1, 2);

    // t2: transpose(0,2) -> view
    auto t2 = base.transpose(0, 2).view({6, 5, 3}, {0, 0, 0});

    // 使用暴力方法验证
    bool brute_result = brute_force_memory_overlap(t1, t2);
    verify_overlap(t1, t2, brute_result);
}

/**
 * 测试：复合变换 - 三重 transpose 恢复原状
 */
TEST("变换后Overlap-复合变换", test_compound_triple_transpose_restore) {
    // 创建 3x4x5 的 tensor
    auto base = make_tensor(0, 60, 0, {20, 5, 1}, {3, 4, 5}, 1);

    // t1: transpose(0,1) -> transpose(1,2) -> transpose(0,1)
    auto t1 = base.transpose(0, 1).transpose(1, 2).transpose(0, 1);

    // t2: 原始 tensor 的 view
    auto t2 = base.view({2, 3, 4}, {0, 0, 0});

    // 使用暴力方法验证
    bool brute_result = brute_force_memory_overlap(t1, t2);
    verify_overlap(t1, t2, brute_result);
}

/**
 * 测试：复合变换 - 嵌套 view
 */
TEST("变换后Overlap-复合变换", test_compound_nested_view) {
    // 创建 10x10 的 tensor
    auto base = make_tensor(0, 100, 0, {10, 1}, {10, 10}, 1);

    // t1: view -> view -> view (逐步缩小)
    auto t1 = base.view({8, 8}, {0, 0}).view({6, 6}, {1, 1}).view({4, 4}, {1, 1});

    // t2: view -> view (不同路径)
    auto t2 = base.view({7, 7}, {2, 2}).view({4, 4}, {1, 1});

    // 使用暴力方法验证
    bool brute_result = brute_force_memory_overlap(t1, t2);
    verify_overlap(t1, t2, brute_result);
}

/**
 * 测试：复合变换 - 嵌套 view 无重叠
 */
TEST("变换后Overlap-复合变换", test_compound_nested_view_no_overlap) {
    // 创建 10x10 的 tensor
    auto base = make_tensor(0, 100, 0, {10, 1}, {10, 10}, 1);

    // t1: 左上角区域
    auto t1 = base.view({5, 5}, {0, 0}).view({3, 3}, {0, 0});

    // t2: 右下角区域
    auto t2 = base.view({5, 5}, {5, 5}).view({3, 3}, {1, 1});

    // 两个区域不重叠
    verify_overlap(t1, t2, false);
}

/**
 * 测试：复合变换 - 同一 base 不同 view 各自变换后重叠
 */
TEST("变换后Overlap-复合变换", test_compound_same_base_diff_view_overlap) {
    // 创建 8x8 的 tensor
    auto base = make_tensor(0, 64, 0, {8, 1}, {8, 8}, 1);

    // 从 base 创建两个有重叠的 view
    auto view1 = base.view({6, 6}, {0, 0});
    auto view2 = base.view({6, 6}, {2, 2});

    // t1: view1 -> transpose
    auto t1 = view1.transpose(0, 1);

    // t2: view2 -> transpose
    auto t2 = view2.transpose(0, 1);

    // 使用暴力方法验证
    bool brute_result = brute_force_memory_overlap(t1, t2);
    verify_overlap(t1, t2, brute_result);
}

/**
 * 测试：复合变换 - 同一 base 不同 view 各自变换后无重叠
 */
TEST("变换后Overlap-复合变换", test_compound_same_base_diff_view_no_overlap) {
    // 创建 10x10 的 tensor
    auto base = make_tensor(0, 100, 0, {10, 1}, {10, 10}, 1);

    // 从 base 创建两个不重叠的 view
    auto view1 = base.view({4, 10}, {0, 0});  // 前 4 行
    auto view2 = base.view({4, 10}, {6, 0});  // 后 4 行

    // t1: view1 -> transpose
    auto t1 = view1.transpose(0, 1);

    // t2: view2 -> transpose
    auto t2 = view2.transpose(0, 1);

    // 两个区域不重叠
    verify_overlap(t1, t2, false);
}

/**
 * 测试：复合变换 - 边界相邻（刚好不重叠）
 */
TEST("变换后Overlap-复合变换", test_compound_adjacent_boundary) {
    // 创建 8x8 的 tensor
    auto base = make_tensor(0, 64, 0, {8, 1}, {8, 8}, 1);

    // t1: 前半部分 -> transpose
    auto t1 = base.view({4, 8}, {0, 0}).transpose(0, 1);

    // t2: 后半部分 -> transpose
    auto t2 = base.view({4, 8}, {4, 0}).transpose(0, 1);

    // 刚好相邻，不重叠
    verify_overlap(t1, t2, false);
}

/**
 * 测试：复合变换 - 单元素重叠
 */
TEST("变换后Overlap-复合变换", test_compound_single_element_overlap) {
    // 创建 8x8 的 tensor
    auto base = make_tensor(0, 64, 0, {8, 1}, {8, 8}, 1);

    // t1: view(4x4, offset 0,0) -> transpose，覆盖 (3,3)
    auto t1 = base.view({4, 4}, {0, 0}).transpose(0, 1);

    // t2: view(4x4, offset 3,3) -> transpose，(0,0) 对应原始 (3,3)
    auto t2 = base.view({4, 4}, {3, 3}).transpose(0, 1);

    // 单元素重叠
    verify_overlap(t1, t2, true);
}

/**
 * 测试：复合变换 - 完全包含关系
 */
TEST("变换后Overlap-复合变换", test_compound_fully_contained) {
    // 创建 8x8 的 tensor
    auto base = make_tensor(0, 64, 0, {8, 1}, {8, 8}, 1);

    // t1: 全部 -> transpose
    auto t1 = base.transpose(0, 1);

    // t2: 子区域 -> transpose
    auto t2 = base.view({4, 4}, {2, 2}).transpose(0, 1);

    // t2 完全包含在 t1 中
    verify_overlap(t1, t2, true);
}

/**
 * 测试：复合变换 - 4D tensor 复合变换
 */
TEST("变换后Overlap-复合变换", test_compound_4d_tensor) {
    // 创建 2x3x4x5 的 tensor
    auto base = make_tensor(0, 120, 0, {60, 20, 5, 1}, {2, 3, 4, 5}, 1);

    // t1: view -> transpose(0,2)
    auto t1 = base.view({2, 3, 3, 4}, {0, 0, 0, 0}).transpose(0, 2);

    // t2: transpose(1,3) -> view
    auto t2 = base.transpose(1, 3).view({2, 4, 3, 2}, {0, 0, 0, 0});

    // 使用暴力方法验证
    bool brute_result = brute_force_memory_overlap(t1, t2);
    verify_overlap(t1, t2, brute_result);
}

/**
 * 测试：复合变换 - 4D tensor 多步变换
 */
TEST("变换后Overlap-复合变换", test_compound_4d_multi_step) {
    // 创建 2x4x3x5 的 tensor
    auto base = make_tensor(0, 120, 0, {60, 15, 5, 1}, {2, 4, 3, 5}, 1);

    // t1: transpose(0,1) -> view -> transpose(2,3)
    auto t1 = base.transpose(0, 1).view({3, 2, 3, 4}, {0, 0, 0, 0}).transpose(2, 3);

    // t2: view -> transpose(0,3) -> view
    auto t2 = base.view({2, 3, 3, 4}, {0, 0, 0, 0}).transpose(0, 3).view({3, 2, 2, 2}, {0, 0, 0, 0});

    // 使用暴力方法验证
    bool brute_result = brute_force_memory_overlap(t1, t2);
    verify_overlap(t1, t2, brute_result);
}

/**
 * 测试：复合变换 - reshape 后多步变换
 */
TEST("变换后Overlap-复合变换", test_compound_reshape_multi_step) {
    // 创建 1D tensor，120 元素
    auto base = make_tensor(0, 120, 0, {1}, {120}, 1);

    // t1: reshape(4x5x6) -> transpose(0,2) -> view
    auto t1 = base.reshape({4, 5, 6}).transpose(0, 2).view({5, 4, 3}, {0, 0, 0});

    // t2: reshape(6x4x5) -> view -> transpose(0,1)
    auto t2 = base.reshape({6, 4, 5}).view({5, 3, 4}, {0, 0, 0}).transpose(0, 1);

    // 使用暴力方法验证
    bool brute_result = brute_force_memory_overlap(t1, t2);
    verify_overlap(t1, t2, brute_result);
}

/**
 * 测试：复合变换 - 交替 view 和 transpose
 */
TEST("变换后Overlap-复合变换", test_compound_alternating_view_transpose) {
    // 创建 8x10 的 tensor
    auto base = make_tensor(0, 80, 0, {10, 1}, {8, 10}, 1);

    // t1: view -> transpose -> view -> transpose
    auto t1 = base.view({6, 8}, {0, 0}).transpose(0, 1).view({6, 5}, {0, 0}).transpose(0, 1);

    // t2: transpose -> view -> transpose -> view
    auto t2 = base.transpose(0, 1).view({8, 6}, {0, 0}).transpose(0, 1).view({5, 6}, {0, 0});

    // 使用暴力方法验证
    bool brute_result = brute_force_memory_overlap(t1, t2);
    verify_overlap(t1, t2, brute_result);
}

/**
 * 测试：复合变换 - 不同维度的 transpose 组合
 */
TEST("变换后Overlap-复合变换", test_compound_different_dim_transpose) {
    // 创建 3x4x5x6 的 tensor
    auto base = make_tensor(0, 360, 0, {120, 30, 6, 1}, {3, 4, 5, 6}, 1);

    // t1: transpose(0,1) -> transpose(2,3)
    auto t1 = base.transpose(0, 1).transpose(2, 3);

    // t2: transpose(0,3) -> transpose(1,2)
    auto t2 = base.transpose(0, 3).transpose(1, 2);

    // 两者访问相同的底层内存
    verify_overlap(t1, t2, true);
}

/**
 * 测试：复合变换 - view 后不同 transpose 组合
 */
TEST("变换后Overlap-复合变换", test_compound_view_then_diff_transpose) {
    // 创建 6x8x10 的 tensor
    auto base = make_tensor(0, 480, 0, {80, 10, 1}, {6, 8, 10}, 1);

    // 共同的 view
    auto common_view = base.view({4, 6, 8}, {0, 0, 0});

    // t1: transpose(0,1)
    auto t1 = common_view.transpose(0, 1);

    // t2: transpose(0,2)
    auto t2 = common_view.transpose(0, 2);

    // 两者访问相同的底层内存
    verify_overlap(t1, t2, true);
}

/**
 * 测试：复合变换 - 复杂路径无重叠
 */
TEST("变换后Overlap-复合变换", test_compound_complex_path_no_overlap) {
    // 创建 12x12 的 tensor
    auto base = make_tensor(0, 144, 0, {12, 1}, {12, 12}, 1);

    // t1: 左上角 6x6 -> transpose -> view 3x4
    auto t1 = base.view({6, 6}, {0, 0}).transpose(0, 1).view({3, 4}, {0, 0});

    // t2: 右下角 6x6 -> transpose -> view 3x4
    auto t2 = base.view({6, 6}, {6, 6}).transpose(0, 1).view({3, 4}, {0, 0});

    // 两个区域不重叠
    verify_overlap(t1, t2, false);
}

/**
 * 测试：复合变换 - 部分重叠的复杂变换
 */
TEST("变换后Overlap-复合变换", test_compound_partial_overlap) {
    // 创建 10x10 的 tensor
    auto base = make_tensor(0, 100, 0, {10, 1}, {10, 10}, 1);

    // t1: 行 0-5，列 0-7 -> transpose
    auto t1 = base.view({6, 8}, {0, 0}).transpose(0, 1);

    // t2: 行 3-8，列 2-9 -> transpose
    auto t2 = base.view({6, 8}, {3, 2}).transpose(0, 1);

    // 使用暴力方法验证
    bool brute_result = brute_force_memory_overlap(t1, t2);
    verify_overlap(t1, t2, brute_result);
}

/**
 * 测试：复合变换 - reshape 到不同形状后 view
 */
TEST("变换后Overlap-复合变换", test_compound_reshape_diff_shape_view) {
    // 创建 1D tensor，60 元素
    auto base = make_tensor(0, 60, 0, {1}, {60}, 1);

    // t1: reshape(3x4x5) -> view(2x3x4)
    auto t1 = base.reshape({3, 4, 5}).view({2, 3, 4}, {0, 0, 0});

    // t2: reshape(5x4x3) -> view(4x3x2)
    auto t2 = base.reshape({5, 4, 3}).view({4, 3, 2}, {0, 0, 0});

    // 使用暴力方法验证
    bool brute_result = brute_force_memory_overlap(t1, t2);
    verify_overlap(t1, t2, brute_result);
}

/**
 * 测试：复合变换 - 对称变换
 */
TEST("变换后Overlap-复合变换", test_compound_symmetric_transform) {
    // 创建 6x6 的方阵
    auto base = make_tensor(0, 36, 0, {6, 1}, {6, 6}, 1);

    // t1: view(4x4) -> transpose
    auto t1 = base.view({4, 4}, {0, 0}).transpose(0, 1);

    // t2: transpose -> view(4x4)
    auto t2 = base.transpose(0, 1).view({4, 4}, {0, 0});

    // 使用暴力方法验证
    bool brute_result = brute_force_memory_overlap(t1, t2);
    verify_overlap(t1, t2, brute_result);
}

/**
 * 测试：复合变换 - 链式 view 缩小到单元素
 */
TEST("变换后Overlap-复合变换", test_compound_chain_view_to_single) {
    // 创建 8x8 的 tensor
    auto base = make_tensor(0, 64, 0, {8, 1}, {8, 8}, 1);

    // t1: 链式 view 缩小到 (2,2) 位置的单元素
    // view({6,6}, {0,0}) -> 从(0,0)开始
    // view({4,4}, {1,1}) -> 从(1,1)开始，相对原始是(1,1)
    // view({2,2}, {1,1}) -> 从(1,1)开始，相对原始是(2,2)
    // view({1,1}, {0,0}) -> 从(0,0)开始，相对原始是(2,2)
    auto t1 = base.view({6, 6}, {0, 0}).view({4, 4}, {1, 1}).view({2, 2}, {1, 1}).view({1, 1}, {0, 0});

    // t2: 直接 view 到 (2,2) 位置
    auto t2 = base.view({1, 1}, {2, 2});

    // 两者访问同一个元素
    verify_overlap(t1, t2, true);
}

/**
 * 测试：复合变换 - 链式 view 缩小到不同单元素
 */
TEST("变换后Overlap-复合变换", test_compound_chain_view_to_diff_single) {
    // 创建 8x8 的 tensor
    auto base = make_tensor(0, 64, 0, {8, 1}, {8, 8}, 1);

    // t1: 链式 view 缩小到 (2,2) 位置
    auto t1 = base.view({4, 4}, {0, 0}).view({1, 1}, {2, 2});

    // t2: 链式 view 缩小到 (5,5) 位置
    auto t2 = base.view({4, 4}, {4, 4}).view({1, 1}, {1, 1});

    // 两者访问不同元素
    verify_overlap(t1, t2, false);
}

/**
 * 测试：特殊场景 - 变换后变成 contiguous 无重叠
 */
TEST("变换后Overlap-特殊场景", test_transform_to_contiguous_no_overlap) {
    // 创建 8x8 的 tensor
    auto base = make_tensor(0, 64, 0, {8, 1}, {8, 8}, 1);

    // t1: view 取前 4 行（contiguous）
    auto t1 = base.view({4, 8}, {0, 0});

    // t2: view 取后 4 行（contiguous）
    auto t2 = base.view({4, 8}, {4, 0});

    // 两个 contiguous 区域不重叠
    verify_overlap(t1, t2, false);
}

/**
 * 测试：特殊场景 - 变换后变成 non-contiguous 有重叠
 */
TEST("变换后Overlap-特殊场景", test_transform_to_non_contiguous_overlap) {
    // 创建 8x8 的 tensor
    auto base = make_tensor(0, 64, 0, {8, 1}, {8, 8}, 1);

    // t1: transpose 后变成 non-contiguous
    auto t1 = base.transpose(0, 1);

    // t2: view 取中间区域
    auto t2 = base.view({4, 4}, {2, 2});

    // t2 是 base 的子集，t1 访问全部，应该有重叠
    verify_overlap(t1, t2, true);
}

/**
 * 测试：特殊场景 - 高维 tensor (5D) 变换
 */
TEST("变换后Overlap-特殊场景", test_5d_tensor_transform) {
    // 创建 2x3x4x5x6 的 tensor
    auto base = make_tensor(0, 720, 0, {360, 120, 30, 6, 1}, {2, 3, 4, 5, 6}, 1);

    // t1: view 取子区域
    auto t1 = base.view({2, 2, 3, 4, 5}, {0, 0, 0, 0, 0});

    // t2: transpose(1,3) 后 view - 修正维度
    // transpose(1,3) 后形状变为 2x5x4x3x6
    auto t2 = base.transpose(1, 3).view({2, 4, 3, 2, 5}, {0, 0, 0, 0, 0});

    // 使用暴力方法验证
    bool brute_result = brute_force_memory_overlap(t1, t2);
    verify_overlap(t1, t2, brute_result);
}

/**
 * 测试：特殊场景 - 单元素 tensor 变换有重叠
 */
TEST("变换后Overlap-特殊场景", test_single_element_transform_overlap) {
    // 创建单元素 tensor
    auto base = make_tensor(0, 1, 0, {1}, {1}, 1);

    // t1: reshape 到 1x1
    auto t1 = base.reshape({1, 1});

    // t2: reshape 到 1x1x1
    auto t2 = base.reshape({1, 1, 1});

    // 两者都访问同一个元素
    verify_overlap(t1, t2, true);
}

/**
 * 测试：特殊场景 - 单元素 tensor 变换无重叠（不同基地址）
 */
TEST("变换后Overlap-特殊场景", test_single_element_transform_no_overlap) {
    // 创建两个不同位置的单元素 tensor
    auto base1 = make_tensor(0, 10, 0, {1}, {1}, 1);
    auto base2 = make_tensor(0, 10, 5, {1}, {1}, 1);

    // t1: reshape
    auto t1 = base1.reshape({1, 1});

    // t2: reshape
    auto t2 = base2.reshape({1, 1});

    // 不同位置，无重叠
    verify_overlap(t1, t2, false);
}

/**
 * 测试：特殊场景 - 跨步访问模式的无重叠
 * 通过 view 操作创建跨步访问的 tensor
 */
TEST("变换后Overlap-特殊场景", test_strided_access_pattern_no_overlap) {
    // 创建 10x10 的 tensor
    auto base = make_tensor(0, 100, 0, {10, 1}, {10, 10}, 1);

    // t1: 取前 5 行
    auto t1 = base.view({5, 10}, {0, 0});

    // t2: 取后 5 行
    auto t2 = base.view({5, 10}, {5, 0});

    // 两个区域不重叠
    verify_overlap(t1, t2, false);
}

/**
 * 测试：特殊场景 - 跨步访问模式有重叠
 */
TEST("变换后Overlap-特殊场景", test_strided_access_pattern_with_overlap) {
    // 创建 10x10 的 tensor
    auto base = make_tensor(0, 100, 0, {10, 1}, {10, 10}, 1);

    // t1: 取行 0-5
    auto t1 = base.view({6, 10}, {0, 0});

    // t2: 取行 4-9，与 t1 在行 4-5 重叠
    auto t2 = base.view({6, 10}, {4, 0});

    // 两个区域有重叠
    verify_overlap(t1, t2, true);
}

// ==================== 主函数 ====================
int main(int argc, char* argv[]) {
    auto& registry = TestRegistry::instance();

    if (argc > 1) {
        std::string arg = argv[1];
        if (arg == "-h" || arg == "--help") {
            TestRegistry::print_help(argv[0]);
            return 0;
        }
        if (arg == "-l" || arg == "--list") {
            registry.list_tests();
            return 0;
        }
        // 其他参数作为过滤器
        return registry.run_filtered(arg);
    }

    return registry.run_all();
}
