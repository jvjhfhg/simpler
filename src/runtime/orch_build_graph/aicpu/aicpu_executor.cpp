/**
 * PTO AICPU Executor
 *
 * Uses PtoScheduler with per-worker-type FIFO ready queues.
 * Legacy scheduler has been removed - PTO is the only mode.
 */

#include "runtime.h"
#include "dep_list_pool.h"

#include <atomic>
#include <cstdint>
#include <mutex>

// Platform-specific logging
#include "aicpu/device_log.h"

// =============================================================================
// Common Constants
// =============================================================================

constexpr int MAX_AICPU_THREADS = 4;
constexpr int MAX_AIC_PER_THREAD = 24;
constexpr int MAX_AIV_PER_THREAD = 48;
constexpr int MAX_CORES_PER_THREAD = MAX_AIC_PER_THREAD + MAX_AIV_PER_THREAD;

// =============================================================================
// PTO Scheduler - FIFO Ready Queues
// =============================================================================

/**
 * PTO Scheduler - Native PTO mode scheduler with FIFO ready queues
 *
 * Key features:
 * - Per-worker-type FIFO ready queues (head/tail indices)
 * - Structured for future dynamic task submission (polling scan_position)
 * - Uses worker_type from task for queue routing
 *
 * Ready queue design:
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

    // Per-worker-type FIFO ready queues
    std::mutex ready_queue_mutex_[PTO_NUM_WORKER_TYPES];
    int ready_queue_[PTO_NUM_WORKER_TYPES][RUNTIME_MAX_TASKS];
    std::atomic<int> ready_head_[PTO_NUM_WORKER_TYPES];  // Dequeue position
    std::atomic<int> ready_tail_[PTO_NUM_WORKER_TYPES];  // Enqueue position

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
    if (worker_type < 0 || worker_type >= PTO_NUM_WORKER_TYPES) {
        worker_type = static_cast<int>(PTOWorkerType::VECTOR);  // Default to vector
    }

    std::lock_guard<std::mutex> lock(ready_queue_mutex_[worker_type]);
    int tail = ready_tail_[worker_type].load(std::memory_order_relaxed);
    ready_queue_[worker_type][tail % RUNTIME_MAX_TASKS] = task_id;
    ready_tail_[worker_type].store(tail + 1, std::memory_order_release);
}

int PtoScheduler::dequeue_ready_task(int worker_type) {
    if (worker_type < 0 || worker_type >= PTO_NUM_WORKER_TYPES) {
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
    if (worker_type < 0 || worker_type >= PTO_NUM_WORKER_TYPES) {
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

    DEV_INFO("PTO Scheduler: Initializing");

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

    // Use worker_count from platform (like host_build_graph does)
    cores_total_num_ = runtime->worker_count;
    thread_cores_num_ = cores_total_num_ / thread_num_;

    if (cores_total_num_ == 0) {
        DEV_ERROR("worker_count is 0, no cores to schedule");
        init_failed_.store(true, std::memory_order_release);
        return -1;
    }

    if (cores_total_num_ > MAX_CORES_PER_THREAD) {
        DEV_ERROR("Total cores %d exceeds maximum %d", cores_total_num_, MAX_CORES_PER_THREAD);
        init_failed_.store(true, std::memory_order_release);
        return -1;
    }

    DEV_INFO("PTO Config: threads=%d, cores=%d, cores_per_thread=%d", thread_num_, cores_total_num_, thread_cores_num_);

    // Platform sets core_type on each Handshake, so we just need to assign cores to threads
    int cores_per_thread = cores_total_num_ / thread_num_;

    if (cores_total_num_ % thread_num_ != 0) {
        DEV_ERROR("worker_count (%d) must be divisible by thread_num (%d)", cores_total_num_, thread_num_);
        init_failed_.store(true, std::memory_order_release);
        return -1;
    }

    // Assign cores to threads (simple sequential assignment)
    for (int t = 0; t < thread_num_; t++) {
        int start_core = t * cores_per_thread;
        for (int i = 0; i < cores_per_thread; i++) {
            core_assignments_[t][i] = start_core + i;
        }
    }

    total_tasks_.store(runtime->get_task_count(), std::memory_order_release);
    completed_tasks_.store(0, std::memory_order_release);

    // Initialize FIFO ready queues
    for (int wt = 0; wt < PTO_NUM_WORKER_TYPES; wt++) {
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
    PTOSharedHeader* shared_header = runtime.get_shared_header();

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

        // === Process completed tasks ===
        for (int i = 0; i < core_num; i++) {
            int core_id = cur_thread_cores[i];
            Handshake* h = &hank[core_id];

            // Check if task completed (status=0 means idle, task!=0 means has result)
            if (h->task_status == 0 && h->task != 0) {
                Task* task = reinterpret_cast<Task*>(h->task);
                h->task = 0;  // Clear task pointer

                // Transition to COMPLETED
                task->state = TaskState::COMPLETED;

                DEV_INFO("PTO Thread %d: Core %d completed task %d", thread_idx, core_id, task->task_id);

                // Get DepListPool for traversing dependency lists
                DepListPool* dep_pool = runtime.get_dep_list_pool();

                // Notify dependent tasks (decrement their fanin) using DepListPool
                dep_list_foreach(dep_pool, task->fanout_head,
                    [&](int32_t dep_id, void*) {
                        Task* dep = runtime.get_task(dep_id);
                        if (dep == nullptr) return;

                        int prev_fanin = dep->fanin.fetch_sub(1, std::memory_order_acq_rel);

                        // If this was the last dependency, task becomes ready
                        if (prev_fanin == 1) {
                            // Transition dependent to READY
                            dep->state = TaskState::READY;
                            int worker_type = dep->core_type;
                            enqueue_ready_task(dep_id, worker_type);
                            DEV_INFO("PTO Thread %d: Task %d now ready (worker_type=%d)", thread_idx, dep_id, worker_type);
                        }
                    }, nullptr);

                // Increment fanout_refcount for each producer using DepListPool
                dep_list_foreach(dep_pool, task->fanin_head,
                    [&](int32_t producer_id, void*) {
                        Task* producer = runtime.get_task(producer_id);
                        if (producer != nullptr) {
                            producer->fanout_refcount++;
                            runtime.check_consumed(producer_id);
                        }
                    }, nullptr);

                // Advance last_task_alive and heap_tail
                // Scan forward from current last_task_alive while tasks are CONSUMED
                if (shared_header != nullptr) {
                    int32_t last_alive = shared_header->last_task_alive;
                    while (last_alive < task_count) {
                        Task* t = runtime.get_task(last_alive);
                        if (t == nullptr || t->state != TaskState::CONSUMED) {
                            break;
                        }
                        last_alive++;
                    }
                    // Update shared header if we advanced
                    if (last_alive > shared_header->last_task_alive) {
                        shared_header->last_task_alive = last_alive;

                        // Advance heap_tail based on last CONSUMED task's packed buffer
                        if (last_alive > 0) {
                            Task* last_consumed = runtime.get_task(last_alive - 1);
                            if (last_consumed != nullptr) {
                                int32_t new_heap_tail = last_consumed->packed_buffer_offset
                                                      + last_consumed->packed_buffer_size;
                                shared_header->heap_tail = new_heap_tail;
                            }
                        }
                    }
                }

                cur_thread_tasks_in_flight--;
                cur_thread_completed++;
                made_progress = true;
                completed_tasks_.fetch_add(1, std::memory_order_release);
            }
        }

        // === Dispatch ready tasks to idle cores ===
        if (cur_thread_tasks_in_flight < core_num) {
            for (int i = 0; i < core_num; i++) {
                int core_id = cur_thread_cores[i];
                Handshake* h = &hank[core_id];

                // Check if core is idle
                if (h->task_status == 0 && h->task == 0) {
                    // Map core_type to worker_type for queue selection
                    // core_type: CoreType::AIC=0, CoreType::AIV=1
                    int worker_type = static_cast<int>(h->core_type);

                    // Try to dequeue from matching ready queue
                    if (has_ready_tasks(worker_type)) {
                        int task_id = dequeue_ready_task(worker_type);
                        if (task_id >= 0) {
                            Task* task = runtime.get_task(task_id);

                            // Transition to RUNNING
                            task->state = TaskState::RUNNING;

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
    for (int wt = 0; wt < PTO_NUM_WORKER_TYPES; wt++) {
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
// Entry Point - PTO Scheduler
// =============================================================================

extern "C" int aicpu_execute(Runtime* runtime) {
    if (runtime == nullptr) {
        DEV_ERROR("%s", "Invalid runtime argument: null pointer");
        return -1;
    }

    DEV_INFO("%s", "aicpu_execute: Starting PTO scheduler");

    g_pto_scheduler.init(runtime);

    while (!g_pto_scheduler.init_done_.load(std::memory_order_acquire)) {
        if (g_pto_scheduler.init_failed_.load(std::memory_order_acquire)) {
            DEV_ERROR("%s", "aicpu_execute: PTO scheduler initialization failed");
            return -1;
        }
    }

    int rc = g_pto_scheduler.run(runtime);
    if (rc != 0) {
        DEV_ERROR("aicpu_execute: PTO scheduler execution failed with rc=%d", rc);
        return rc;
    }

    if (g_pto_scheduler.finished_.load(std::memory_order_acquire)) {
        DEV_INFO("aicpu_execute: Last thread finished, cleaning up");
        g_pto_scheduler.deinit();
    }

    return 0;
}
