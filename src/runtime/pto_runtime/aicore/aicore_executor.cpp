/**
 * PTO AICore Executor - AICore worker kernel
 *
 * Compatible with host_build_graph pattern for Phase 1.
 * Uses standard Runtime and Task structures.
 */

#include "runtime.h"

// Platform-specific includes
#ifdef __CCE_KT_TEST__
#define __aicore__
#define __gm__
static inline void dcci(void* addr, int scope, int type) { (void)addr; (void)scope; (void)type; }
#define ENTIRE_DATA_CACHE 0
#define CACHELINE_OUT 0
#else
#include "aicore/aicore.h"
#endif

/**
 * Unified function pointer type for kernel dispatch
 */
typedef void (*UnifiedKernelFunc)(__gm__ int64_t*);

/**
 * Execute a task via function pointer dispatch
 */
__aicore__ __attribute__((always_inline))
static void execute_task(__gm__ Task* task) {
    if (task == nullptr) {
        return;
    }

    if (task->function_bin_addr == 0) {
        return;
    }

    UnifiedKernelFunc kernel = (UnifiedKernelFunc)task->function_bin_addr;
    kernel(reinterpret_cast<__gm__ int64_t*>(task->args));
}

/**
 * AICore worker kernel - main entry point
 */
__aicore__ __attribute__((weak))
void aicore_execute(__gm__ Runtime* runtime, int block_idx, CoreType core_type) {
    (void)core_type;
    __gm__ Handshake* my_hank = (__gm__ Handshake*)(&runtime->workers[block_idx]);

    // Phase 1: Wait for AICPU ready signal
    while (my_hank->aicpu_ready == 0) {
        dcci(my_hank, ENTIRE_DATA_CACHE, CACHELINE_OUT);
    }

    // Phase 2: Signal AICore ready
    my_hank->aicore_done = block_idx + 1;

    // Phase 3: Main execution loop
    while (true) {
        dcci(my_hank, ENTIRE_DATA_CACHE, CACHELINE_OUT);

        if (my_hank->control == 1) {
            break;
        }

        if (my_hank->task_status == 1 && my_hank->task != 0) {
            __gm__ Task* task_ptr = reinterpret_cast<__gm__ Task*>(my_hank->task);
            execute_task(task_ptr);
            my_hank->task_status = 0;
        }
    }
}
