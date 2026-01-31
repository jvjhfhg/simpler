/**
 * PTO Scheduler - AICPU task scheduling and dispatch
 *
 * Compatible with host_build_graph pattern for Phase 1.
 * Uses standard Runtime and Task structures.
 */

#include "runtime.h"

#include <atomic>
#include <cstdint>
#include <mutex>

// Platform-specific logging
#ifndef DEV_INFO
#define DEV_INFO(fmt, ...) do { } while(0)
#endif
#ifndef DEV_ERROR
#define DEV_ERROR(fmt, ...) do { } while(0)
#endif
#ifndef DEV_WARN
#define DEV_WARN(fmt, ...) do { } while(0)
#endif

constexpr int MAX_AICPU_THREADS = 4;
constexpr int MAX_AIC_PER_THREAD = 24;
constexpr int MAX_AIV_PER_THREAD = 48;
constexpr int MAX_CORES_PER_THREAD = MAX_AIC_PER_THREAD + MAX_AIV_PER_THREAD;

struct AicpuExecutor {
    std::atomic<int> thread_idx_{0};
    std::atomic<bool> initialized_{false};
    std::atomic<bool> init_done_{false};
    std::atomic<bool> init_failed_{false};
    std::atomic<bool> finished_{false};

    int thread_num_{0};
    int cores_total_num_{0};
    int blockdim_cores_num_{3};
    int thread_cores_num_{0};
    int core_assignments_[MAX_AICPU_THREADS][MAX_CORES_PER_THREAD];

    std::mutex ready_queue_aic_mutex_;
    int ready_queue_aic_[RUNTIME_MAX_TASKS];
    std::atomic<int> ready_count_aic_{0};

    std::mutex ready_queue_aiv_mutex_;
    int ready_queue_aiv_[RUNTIME_MAX_TASKS];
    std::atomic<int> ready_count_aiv_{0};

    std::atomic<int> completed_tasks_{0};
    std::atomic<int> total_tasks_{0};
    std::atomic<int> finished_count_{0};

    int init(Runtime* runtime);
    int hank_aicore(Runtime* runtime, int thread_idx, const int* cur_thread_cores);
    int resolve_and_dispatch(Runtime& runtime, int thread_idx, const int* cur_thread_cores, int core_num);
    int shutdown_aicore(Runtime* runtime, int thread_idx, const int* cur_thread_cores);
    int run(Runtime* runtime);
    void deinit();
};

static AicpuExecutor g_aicpu_executor;

int AicpuExecutor::init(Runtime* runtime) {
    bool expected = false;
    if (!initialized_.compare_exchange_strong(expected, true, std::memory_order_acq_rel, std::memory_order_acquire)) {
        return 0;
    }

    DEV_INFO("PTO AicpuExecutor: Initializing");

    if (runtime == nullptr) {
        DEV_ERROR("runtime is nullptr");
        init_failed_.store(true, std::memory_order_release);
        return -1;
    }

    thread_num_ = runtime->sche_cpu_num;
    if (thread_num_ == 0) thread_num_ = 1;

    if (thread_num_ < 1 || thread_num_ > MAX_AICPU_THREADS) {
        DEV_ERROR("Invalid thread_num: %d", thread_num_);
        init_failed_.store(true, std::memory_order_release);
        return -1;
    }

    cores_total_num_ = runtime->block_dim * blockdim_cores_num_;
    thread_cores_num_ = cores_total_num_ / thread_num_;

    if (cores_total_num_ > MAX_CORES_PER_THREAD) {
        DEV_ERROR("Total cores %d exceeds maximum %d", cores_total_num_, MAX_CORES_PER_THREAD);
        init_failed_.store(true, std::memory_order_release);
        return -1;
    }

    DEV_INFO("Config: threads=%d, cores=%d, cores_per_thread=%d", thread_num_, cores_total_num_, thread_cores_num_);

    int num_aic = runtime->block_dim;
    int blocks_per_thread = runtime->block_dim / thread_num_;

    if (runtime->block_dim % thread_num_ != 0) {
        DEV_ERROR("block_dim (%d) must be divisible by thread_num (%d)", runtime->block_dim, thread_num_);
        init_failed_.store(true, std::memory_order_release);
        return -1;
    }

    for (int t = 0; t < thread_num_; t++) {
        int start_block = t * blocks_per_thread;
        int end_block = (t + 1) * blocks_per_thread;
        int core_idx = 0;

        for (int b = start_block; b < end_block; b++) {
            core_assignments_[t][core_idx++] = b;
        }

        for (int b = start_block; b < end_block; b++) {
            int aiv_base = num_aic;
            core_assignments_[t][core_idx++] = aiv_base + b * 2;
            core_assignments_[t][core_idx++] = aiv_base + b * 2 + 1;
        }
    }

    total_tasks_.store(runtime->get_task_count(), std::memory_order_release);
    completed_tasks_.store(0, std::memory_order_release);

    int initial_ready[RUNTIME_MAX_TASKS];
    int initial_count = runtime->get_initial_ready_tasks(initial_ready);

    DEV_INFO("Init: Found %d initially ready tasks", initial_count);

    int aic_count = 0;
    int aiv_count = 0;
    for (int i = 0; i < initial_count; i++) {
        Task* task = runtime->get_task(initial_ready[i]);
        if (task->core_type == 0) {
            ready_queue_aic_[aic_count++] = initial_ready[i];
        } else {
            ready_queue_aiv_[aiv_count++] = initial_ready[i];
        }
    }
    ready_count_aic_.store(aic_count, std::memory_order_release);
    ready_count_aiv_.store(aiv_count, std::memory_order_release);

    finished_count_.store(0, std::memory_order_release);

    init_done_.store(true, std::memory_order_release);
    DEV_INFO("PTO AicpuExecutor: Init complete");
    return 0;
}

int AicpuExecutor::hank_aicore(Runtime* runtime, int thread_idx, const int* cur_thread_cores) {
    Handshake* all_hanks = (Handshake*)runtime->workers;

    DEV_INFO("Thread %d: Handshaking with %d cores", thread_idx, thread_cores_num_);

    for (int i = 0; i < thread_cores_num_; i++) {
        int core_id = cur_thread_cores[i];
        Handshake* hank = &all_hanks[core_id];
        hank->aicpu_ready = 1;
    }

    for (int i = 0; i < thread_cores_num_; i++) {
        int core_id = cur_thread_cores[i];
        Handshake* hank = &all_hanks[core_id];
        while (hank->aicore_done == 0) {
        }
    }
    return 0;
}

int AicpuExecutor::shutdown_aicore(Runtime* runtime, int thread_idx, const int* cur_thread_cores) {
    Handshake* all_hanks = (Handshake*)runtime->workers;

    DEV_INFO("Thread %d: Shutting down %d cores", thread_idx, thread_cores_num_);

    for (int i = 0; i < thread_cores_num_; i++) {
        int core_id = cur_thread_cores[i];
        Handshake* hank = &all_hanks[core_id];
        hank->control = 1;
    }
    return 0;
}

int AicpuExecutor::resolve_and_dispatch(Runtime& runtime, int thread_idx, const int* cur_thread_cores, int core_num) {
    Handshake* hank = (Handshake*)runtime.workers;

    DEV_INFO("Thread %d: Starting execution with %d cores", thread_idx, core_num);

    int cur_thread_completed = 0;
    int cur_thread_tasks_in_flight = 0;
    int task_count = total_tasks_.load(std::memory_order_acquire);

    int idle_iterations = 0;
    const int MAX_IDLE_ITERATIONS = 1000000;
    bool made_progress = false;

    while (true) {
        if (completed_tasks_.load(std::memory_order_acquire) >= task_count) {
            bool all_cores_idle = true;

            for (int i = 0; i < core_num; i++) {
                int core_id = cur_thread_cores[i];
                Handshake* h = &hank[core_id];

                if (h->task_status != 0 || h->task != 0) {
                    all_cores_idle = false;
                    break;
                }
            }

            if (all_cores_idle) {
                break;
            }
        }

        made_progress = false;

        // Process completed tasks
        for (int i = 0; i < core_num; i++) {
            int core_id = cur_thread_cores[i];
            Handshake* h = &hank[core_id];

            if (h->task_status == 0 && h->task != 0) {
                Task* task = reinterpret_cast<Task*>(h->task);
                h->task = 0;

                int task_id = task->task_id;

                DEV_INFO("Thread %d: Core %d completed task %d", thread_idx, core_id, task_id);

                for (int j = 0; j < task->fanout_count; j++) {
                    int dep_id = task->fanout[j];
                    Task* dep = runtime.get_task(dep_id);

                    int prev_fanin = dep->fanin.fetch_sub(1, std::memory_order_acq_rel);

                    if (prev_fanin == 1) {
                        if (dep->core_type == 0) {
                            std::lock_guard<std::mutex> lock(ready_queue_aic_mutex_);
                            int idx = ready_count_aic_.load(std::memory_order_relaxed);
                            ready_queue_aic_[idx] = dep_id;
                            ready_count_aic_.fetch_add(1, std::memory_order_release);
                        } else {
                            std::lock_guard<std::mutex> lock(ready_queue_aiv_mutex_);
                            int idx = ready_count_aiv_.load(std::memory_order_relaxed);
                            ready_queue_aiv_[idx] = dep_id;
                            ready_count_aiv_.fetch_add(1, std::memory_order_release);
                        }
                    }
                }

                cur_thread_tasks_in_flight--;
                cur_thread_completed++;
                made_progress = true;
                completed_tasks_.fetch_add(1, std::memory_order_release);
            }
        }

        // Dispatch ready tasks
        if (cur_thread_tasks_in_flight < core_num) {
            for (int i = 0; i < core_num; i++) {
                int core_id = cur_thread_cores[i];
                Handshake* h = &hank[core_id];

                if (h->task_status == 0 && h->task == 0) {
                    if (h->core_type == 0) {
                        if (ready_count_aic_.load(std::memory_order_acquire) > 0) {
                            std::lock_guard<std::mutex> lock(ready_queue_aic_mutex_);
                            int count = ready_count_aic_.load(std::memory_order_relaxed);
                            if (count > 0) {
                                ready_count_aic_.fetch_sub(1, std::memory_order_release);
                                int task_id = ready_queue_aic_[count - 1];
                                Task* task = runtime.get_task(task_id);

                                h->task = reinterpret_cast<uint64_t>(task);
                                h->task_status = 1;
                                cur_thread_tasks_in_flight++;
                                made_progress = true;
                            }
                        }
                    } else if (h->core_type == 1) {
                        if (ready_count_aiv_.load(std::memory_order_acquire) > 0) {
                            std::lock_guard<std::mutex> lock(ready_queue_aiv_mutex_);
                            int count = ready_count_aiv_.load(std::memory_order_relaxed);
                            if (count > 0) {
                                ready_count_aiv_.fetch_sub(1, std::memory_order_release);
                                int task_id = ready_queue_aiv_[count - 1];
                                Task* task = runtime.get_task(task_id);

                                h->task = reinterpret_cast<uint64_t>(task);
                                h->task_status = 1;
                                cur_thread_tasks_in_flight++;
                                made_progress = true;
                            }
                        }
                    }
                }
            }
        }

        if (!made_progress) {
            idle_iterations++;
            if (idle_iterations > MAX_IDLE_ITERATIONS) {
                DEV_ERROR("Thread %d: Timeout after %d idle iterations!", thread_idx, idle_iterations);
                return -1;
            }
        } else {
            idle_iterations = 0;
        }
    }

    DEV_INFO("Thread %d: Execution complete, completed %d tasks", thread_idx, cur_thread_completed);
    return cur_thread_completed;
}

int AicpuExecutor::run(Runtime* runtime) {
    int thread_idx = thread_idx_++;

    DEV_INFO("Thread %d: Start", thread_idx);

    const int* cur_thread_cores = core_assignments_[thread_idx];

    auto rc = hank_aicore(runtime, thread_idx, cur_thread_cores);
    if (rc != 0) {
        return rc;
    }

    DEV_INFO("Thread %d: Runtime has %d tasks", thread_idx, runtime->get_task_count());
    int completed = resolve_and_dispatch(*runtime, thread_idx, cur_thread_cores, thread_cores_num_);
    DEV_INFO("Thread %d: Executed %d tasks from runtime", thread_idx, completed);

    rc = shutdown_aicore(runtime, thread_idx, cur_thread_cores);
    if (rc != 0) {
        return rc;
    }

    DEV_INFO("Thread %d: Completed", thread_idx);

    int prev_finished = finished_count_.fetch_add(1, std::memory_order_acq_rel);
    if (prev_finished + 1 == thread_num_) {
        finished_.store(true, std::memory_order_release);
        DEV_INFO("Thread %d: Last thread, marking executor finished", thread_idx);
    }

    return 0;
}

void AicpuExecutor::deinit() {
    ready_count_aic_.store(0, std::memory_order_release);
    ready_count_aiv_.store(0, std::memory_order_release);
    completed_tasks_.store(0, std::memory_order_release);
    total_tasks_.store(0, std::memory_order_release);
    finished_count_.store(0, std::memory_order_release);

    initialized_.store(false, std::memory_order_release);
    init_done_.store(false, std::memory_order_release);
    init_failed_.store(false, std::memory_order_release);
    thread_idx_.store(0, std::memory_order_release);
    finished_.store(false, std::memory_order_release);

    DEV_INFO("DeInit: PTO AicpuExecutor reset complete");
}

extern "C" int aicpu_execute(Runtime* runtime) {
    if (runtime == nullptr) {
        DEV_ERROR("%s", "Invalid runtime argument: null pointer");
        return -1;
    }

    DEV_INFO("%s", "aicpu_execute: Starting PTO AICPU kernel execution");

    g_aicpu_executor.init(runtime);

    while (!g_aicpu_executor.init_done_.load(std::memory_order_acquire)) {
        if (g_aicpu_executor.init_failed_.load(std::memory_order_acquire)) {
            DEV_ERROR("%s", "aicpu_execute: Initialization failed, aborting execution");
            return -1;
        }
    }

    int rc = g_aicpu_executor.run(runtime);
    if (rc != 0) {
        DEV_ERROR("aicpu_execute: Thread execution failed with rc=%d", rc);
        return rc;
    }

    if (g_aicpu_executor.finished_.load(std::memory_order_acquire)) {
        DEV_INFO("aicpu_execute: Last thread finished, cleaning up");
        g_aicpu_executor.deinit();
    }

    DEV_INFO("%s", "aicpu_execute: PTO kernel execution completed successfully");
    return 0;
}
