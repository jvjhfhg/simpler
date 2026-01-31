/**
 * PTO AICPU Executor - Dual-mode task scheduling and dispatch
 *
 * Supports two scheduling modes:
 * 1. Legacy mode: Uses AicpuExecutor with AIC/AIV ready queues (stack-based)
 * 2. PTO mode:    Uses PtoScheduler with per-worker-type FIFO ready queues
 *
 * Mode selection: runtime->is_pto_mode() determines which scheduler runs
 *
 * Phase 5: Added PTO-native scheduler with FIFO ready queues per worker type
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

// =============================================================================
// Common Constants
// =============================================================================

constexpr int MAX_AICPU_THREADS = 4;
constexpr int MAX_AIC_PER_THREAD = 24;
constexpr int MAX_AIV_PER_THREAD = 48;
constexpr int MAX_CORES_PER_THREAD = MAX_AIC_PER_THREAD + MAX_AIV_PER_THREAD;

// Worker types (matching pto_runtime.h)
constexpr int WORKER_CUBE = 0;    // AIC (Compute)
constexpr int WORKER_VECTOR = 1;  // AIV (Vector)
constexpr int NUM_WORKER_TYPES = 2;

// =============================================================================
// Legacy Scheduler (AicpuExecutor) - for non-PTO mode
// =============================================================================

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

static AicpuExecutor g_legacy_executor;

int AicpuExecutor::init(Runtime* runtime) {
    bool expected = false;
    if (!initialized_.compare_exchange_strong(expected, true, std::memory_order_acq_rel, std::memory_order_acquire)) {
        return 0;
    }

    DEV_INFO("Legacy AicpuExecutor: Initializing");

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
    DEV_INFO("Legacy AicpuExecutor: Init complete");
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

    DEV_INFO("DeInit: Legacy AicpuExecutor reset complete");
}

// =============================================================================
// PTO Scheduler - for PTO mode (Phase 5)
// =============================================================================

/**
 * PTO Scheduler - Native PTO mode scheduler with FIFO ready queues
 *
 * Key differences from legacy scheduler:
 * - Per-worker-type FIFO ready queues (head/tail indices, not stack)
 * - Structured for future dynamic task submission (polling scan_position)
 * - Uses worker_type from task instead of core_type mapping
 *
 * Ready queue design matches PTOSchedulerState from pto_runtime.h:
 *   ready_queue[worker_type][task_window_size]
 *   ready_head[worker_type]  - dequeue position
 *   ready_tail[worker_type]  - enqueue position
 */
struct PtoScheduler {
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

    // Per-worker-type FIFO ready queues (matching PTOSchedulerState design)
    std::mutex ready_queue_mutex_[NUM_WORKER_TYPES];
    int ready_queue_[NUM_WORKER_TYPES][RUNTIME_MAX_TASKS];
    std::atomic<int> ready_head_[NUM_WORKER_TYPES];  // Dequeue position
    std::atomic<int> ready_tail_[NUM_WORKER_TYPES];  // Enqueue position

    std::atomic<int> completed_tasks_{0};
    std::atomic<int> total_tasks_{0};
    std::atomic<int> finished_count_{0};

    int init(Runtime* runtime);
    int handshake_cores(Runtime* runtime, int thread_idx, const int* cur_thread_cores);
    int schedule_and_dispatch(Runtime& runtime, int thread_idx, const int* cur_thread_cores, int core_num);
    int shutdown_cores(Runtime* runtime, int thread_idx, const int* cur_thread_cores);
    int run(Runtime* runtime);
    void deinit();

    // Helper: enqueue task to appropriate ready queue
    void enqueue_ready_task(int task_id, int worker_type);
    // Helper: dequeue task from ready queue (-1 if empty)
    int dequeue_ready_task(int worker_type);
    // Helper: check if ready queue has tasks
    bool has_ready_tasks(int worker_type);
};

static PtoScheduler g_pto_scheduler;

void PtoScheduler::enqueue_ready_task(int task_id, int worker_type) {
    if (worker_type < 0 || worker_type >= NUM_WORKER_TYPES) {
        worker_type = WORKER_VECTOR;  // Default to vector
    }

    std::lock_guard<std::mutex> lock(ready_queue_mutex_[worker_type]);
    int tail = ready_tail_[worker_type].load(std::memory_order_relaxed);
    ready_queue_[worker_type][tail % RUNTIME_MAX_TASKS] = task_id;
    ready_tail_[worker_type].store(tail + 1, std::memory_order_release);
}

int PtoScheduler::dequeue_ready_task(int worker_type) {
    if (worker_type < 0 || worker_type >= NUM_WORKER_TYPES) {
        return -1;
    }

    std::lock_guard<std::mutex> lock(ready_queue_mutex_[worker_type]);
    int head = ready_head_[worker_type].load(std::memory_order_relaxed);
    int tail = ready_tail_[worker_type].load(std::memory_order_relaxed);

    if (head >= tail) {
        return -1;  // Queue empty
    }

    int task_id = ready_queue_[worker_type][head % RUNTIME_MAX_TASKS];
    ready_head_[worker_type].store(head + 1, std::memory_order_release);
    return task_id;
}

bool PtoScheduler::has_ready_tasks(int worker_type) {
    if (worker_type < 0 || worker_type >= NUM_WORKER_TYPES) {
        return false;
    }

    int head = ready_head_[worker_type].load(std::memory_order_acquire);
    int tail = ready_tail_[worker_type].load(std::memory_order_acquire);
    return head < tail;
}

int PtoScheduler::init(Runtime* runtime) {
    bool expected = false;
    if (!initialized_.compare_exchange_strong(expected, true, std::memory_order_acq_rel, std::memory_order_acquire)) {
        return 0;
    }

    DEV_INFO("PTO Scheduler: Initializing (PTO mode)");

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

    DEV_INFO("PTO Config: threads=%d, cores=%d, cores_per_thread=%d", thread_num_, cores_total_num_, thread_cores_num_);

    int num_aic = runtime->block_dim;
    int blocks_per_thread = runtime->block_dim / thread_num_;

    if (runtime->block_dim % thread_num_ != 0) {
        DEV_ERROR("block_dim (%d) must be divisible by thread_num (%d)", runtime->block_dim, thread_num_);
        init_failed_.store(true, std::memory_order_release);
        return -1;
    }

    // Assign cores to threads (same as legacy)
    for (int t = 0; t < thread_num_; t++) {
        int start_block = t * blocks_per_thread;
        int end_block = (t + 1) * blocks_per_thread;
        int core_idx = 0;

        // AIC cores first
        for (int b = start_block; b < end_block; b++) {
            core_assignments_[t][core_idx++] = b;
        }

        // AIV cores (2 per block)
        for (int b = start_block; b < end_block; b++) {
            int aiv_base = num_aic;
            core_assignments_[t][core_idx++] = aiv_base + b * 2;
            core_assignments_[t][core_idx++] = aiv_base + b * 2 + 1;
        }
    }

    total_tasks_.store(runtime->get_task_count(), std::memory_order_release);
    completed_tasks_.store(0, std::memory_order_release);

    // Initialize FIFO ready queues
    for (int wt = 0; wt < NUM_WORKER_TYPES; wt++) {
        ready_head_[wt].store(0, std::memory_order_release);
        ready_tail_[wt].store(0, std::memory_order_release);
    }

    // Find initially ready tasks and enqueue them (FIFO order)
    int initial_ready[RUNTIME_MAX_TASKS];
    int initial_count = runtime->get_initial_ready_tasks(initial_ready);

    DEV_INFO("PTO Init: Found %d initially ready tasks", initial_count);

    for (int i = 0; i < initial_count; i++) {
        Task* task = runtime->get_task(initial_ready[i]);
        // core_type: 0=AIC (CUBE), 1=AIV (VECTOR)
        int worker_type = task->core_type;
        enqueue_ready_task(initial_ready[i], worker_type);
    }

    finished_count_.store(0, std::memory_order_release);

    init_done_.store(true, std::memory_order_release);
    DEV_INFO("PTO Scheduler: Init complete");
    return 0;
}

int PtoScheduler::handshake_cores(Runtime* runtime, int thread_idx, const int* cur_thread_cores) {
    Handshake* all_hanks = (Handshake*)runtime->workers;

    DEV_INFO("PTO Thread %d: Handshaking with %d cores", thread_idx, thread_cores_num_);

    // Signal ready to all cores
    for (int i = 0; i < thread_cores_num_; i++) {
        int core_id = cur_thread_cores[i];
        Handshake* hank = &all_hanks[core_id];
        hank->aicpu_ready = 1;
    }

    // Wait for all cores to acknowledge
    for (int i = 0; i < thread_cores_num_; i++) {
        int core_id = cur_thread_cores[i];
        Handshake* hank = &all_hanks[core_id];
        while (hank->aicore_done == 0) {
            // Spin wait
        }
    }
    return 0;
}

int PtoScheduler::shutdown_cores(Runtime* runtime, int thread_idx, const int* cur_thread_cores) {
    Handshake* all_hanks = (Handshake*)runtime->workers;

    DEV_INFO("PTO Thread %d: Shutting down %d cores", thread_idx, thread_cores_num_);

    for (int i = 0; i < thread_cores_num_; i++) {
        int core_id = cur_thread_cores[i];
        Handshake* hank = &all_hanks[core_id];
        hank->control = 1;
    }
    return 0;
}

int PtoScheduler::schedule_and_dispatch(Runtime& runtime, int thread_idx, const int* cur_thread_cores, int core_num) {
    Handshake* hank = (Handshake*)runtime.workers;

    DEV_INFO("PTO Thread %d: Starting execution with %d cores", thread_idx, core_num);

    int cur_thread_completed = 0;
    int cur_thread_tasks_in_flight = 0;
    int task_count = total_tasks_.load(std::memory_order_acquire);

    int idle_iterations = 0;
    const int MAX_IDLE_ITERATIONS = 1000000;
    bool made_progress = false;

    while (true) {
        // Check termination condition
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

        // === Phase 1: Process completed tasks ===
        for (int i = 0; i < core_num; i++) {
            int core_id = cur_thread_cores[i];
            Handshake* h = &hank[core_id];

            // Check if task completed (status=0 means idle, task!=0 means has result)
            if (h->task_status == 0 && h->task != 0) {
                Task* task = reinterpret_cast<Task*>(h->task);
                h->task = 0;  // Clear task pointer

                int task_id = task->task_id;

                DEV_INFO("PTO Thread %d: Core %d completed task %d", thread_idx, core_id, task_id);

                // Notify dependent tasks (decrement their fanin)
                for (int j = 0; j < task->fanout_count; j++) {
                    int dep_id = task->fanout[j];
                    Task* dep = runtime.get_task(dep_id);

                    int prev_fanin = dep->fanin.fetch_sub(1, std::memory_order_acq_rel);

                    // If this was the last dependency, task becomes ready
                    if (prev_fanin == 1) {
                        int worker_type = dep->core_type;
                        enqueue_ready_task(dep_id, worker_type);
                        DEV_INFO("PTO Thread %d: Task %d now ready (worker_type=%d)", thread_idx, dep_id, worker_type);
                    }
                }

                cur_thread_tasks_in_flight--;
                cur_thread_completed++;
                made_progress = true;
                completed_tasks_.fetch_add(1, std::memory_order_release);
            }
        }

        // === Phase 2: Dispatch ready tasks to idle cores ===
        if (cur_thread_tasks_in_flight < core_num) {
            for (int i = 0; i < core_num; i++) {
                int core_id = cur_thread_cores[i];
                Handshake* h = &hank[core_id];

                // Check if core is idle
                if (h->task_status == 0 && h->task == 0) {
                    // Map core_type to worker_type for queue selection
                    // core_type: 0=AIC, 1=AIV
                    int worker_type = h->core_type;

                    // Try to dequeue from matching ready queue
                    if (has_ready_tasks(worker_type)) {
                        int task_id = dequeue_ready_task(worker_type);
                        if (task_id >= 0) {
                            Task* task = runtime.get_task(task_id);

                            h->task = reinterpret_cast<uint64_t>(task);
                            h->task_status = 1;  // Mark as busy
                            cur_thread_tasks_in_flight++;
                            made_progress = true;

                            DEV_INFO("PTO Thread %d: Dispatched task %d to core %d (worker_type=%d)",
                                     thread_idx, task_id, core_id, worker_type);
                        }
                    }
                }
            }
        }

        // Idle detection for timeout
        if (!made_progress) {
            idle_iterations++;
            if (idle_iterations > MAX_IDLE_ITERATIONS) {
                DEV_ERROR("PTO Thread %d: Timeout after %d idle iterations!", thread_idx, idle_iterations);
                return -1;
            }
        } else {
            idle_iterations = 0;
        }
    }

    DEV_INFO("PTO Thread %d: Execution complete, completed %d tasks", thread_idx, cur_thread_completed);
    return cur_thread_completed;
}

int PtoScheduler::run(Runtime* runtime) {
    int thread_idx = thread_idx_++;

    DEV_INFO("PTO Thread %d: Start", thread_idx);

    const int* cur_thread_cores = core_assignments_[thread_idx];

    // Handshake with AICore workers
    auto rc = handshake_cores(runtime, thread_idx, cur_thread_cores);
    if (rc != 0) {
        return rc;
    }

    DEV_INFO("PTO Thread %d: Runtime has %d tasks", thread_idx, runtime->get_task_count());

    // Main scheduling loop
    int completed = schedule_and_dispatch(*runtime, thread_idx, cur_thread_cores, thread_cores_num_);
    DEV_INFO("PTO Thread %d: Executed %d tasks from runtime", thread_idx, completed);

    // Shutdown AICore workers
    rc = shutdown_cores(runtime, thread_idx, cur_thread_cores);
    if (rc != 0) {
        return rc;
    }

    DEV_INFO("PTO Thread %d: Completed", thread_idx);

    // Track thread completion
    int prev_finished = finished_count_.fetch_add(1, std::memory_order_acq_rel);
    if (prev_finished + 1 == thread_num_) {
        finished_.store(true, std::memory_order_release);
        DEV_INFO("PTO Thread %d: Last thread, marking scheduler finished", thread_idx);
    }

    return 0;
}

void PtoScheduler::deinit() {
    // Reset ready queues
    for (int wt = 0; wt < NUM_WORKER_TYPES; wt++) {
        ready_head_[wt].store(0, std::memory_order_release);
        ready_tail_[wt].store(0, std::memory_order_release);
    }

    completed_tasks_.store(0, std::memory_order_release);
    total_tasks_.store(0, std::memory_order_release);
    finished_count_.store(0, std::memory_order_release);

    initialized_.store(false, std::memory_order_release);
    init_done_.store(false, std::memory_order_release);
    init_failed_.store(false, std::memory_order_release);
    thread_idx_.store(0, std::memory_order_release);
    finished_.store(false, std::memory_order_release);

    DEV_INFO("DeInit: PTO Scheduler reset complete");
}

// =============================================================================
// Entry Point - Routes to legacy or PTO scheduler based on runtime mode
// =============================================================================

static int legacy_scheduler_run(Runtime* runtime) {
    DEV_INFO("%s", "legacy_scheduler_run: Starting legacy AICPU kernel execution");

    g_legacy_executor.init(runtime);

    while (!g_legacy_executor.init_done_.load(std::memory_order_acquire)) {
        if (g_legacy_executor.init_failed_.load(std::memory_order_acquire)) {
            DEV_ERROR("%s", "legacy_scheduler_run: Initialization failed");
            return -1;
        }
    }

    int rc = g_legacy_executor.run(runtime);
    if (rc != 0) {
        DEV_ERROR("legacy_scheduler_run: Thread execution failed with rc=%d", rc);
        return rc;
    }

    if (g_legacy_executor.finished_.load(std::memory_order_acquire)) {
        DEV_INFO("legacy_scheduler_run: Last thread finished, cleaning up");
        g_legacy_executor.deinit();
    }

    return 0;
}

static int pto_scheduler_run(Runtime* runtime) {
    DEV_INFO("%s", "pto_scheduler_run: Starting PTO AICPU kernel execution");

    g_pto_scheduler.init(runtime);

    while (!g_pto_scheduler.init_done_.load(std::memory_order_acquire)) {
        if (g_pto_scheduler.init_failed_.load(std::memory_order_acquire)) {
            DEV_ERROR("%s", "pto_scheduler_run: Initialization failed");
            return -1;
        }
    }

    int rc = g_pto_scheduler.run(runtime);
    if (rc != 0) {
        DEV_ERROR("pto_scheduler_run: Thread execution failed with rc=%d", rc);
        return rc;
    }

    if (g_pto_scheduler.finished_.load(std::memory_order_acquire)) {
        DEV_INFO("pto_scheduler_run: Last thread finished, cleaning up");
        g_pto_scheduler.deinit();
    }

    return 0;
}

extern "C" int aicpu_execute(Runtime* runtime) {
    if (runtime == nullptr) {
        DEV_ERROR("%s", "Invalid runtime argument: null pointer");
        return -1;
    }

    // Route to appropriate scheduler based on runtime mode
    if (runtime->is_pto_mode()) {
        DEV_INFO("%s", "aicpu_execute: PTO mode detected, using PTO scheduler");
        return pto_scheduler_run(runtime);
    } else {
        DEV_INFO("%s", "aicpu_execute: Legacy mode, using legacy scheduler");
        return legacy_scheduler_run(runtime);
    }
}