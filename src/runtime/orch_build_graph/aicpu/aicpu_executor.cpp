/**
 * PTO AICPU Executor
 *
 * Uses PtoScheduler with per-worker-type FIFO ready queues.
 * Legacy scheduler has been removed - PTO is the only mode.
 *
 * Device orchestration mode:
 * - Thread 3 runs as orchestrator (loads SO, builds task graph)
 * - Threads 0-2 run as schedulers (wait for orchestration, dispatch tasks)
 */

#include "runtime.h"
#include "dep_list_pool.h"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/syscall.h>

// Platform-specific logging
#include "aicpu/device_log.h"

// Orchestration function signature
typedef int (*OrchestrationFunc)(Runtime* runtime, uint64_t* args, int arg_count);

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

    // Orchestration synchronization (for device orchestration mode)
    std::atomic<bool> orchestration_done_{false};
    std::atomic<bool> orchestration_failed_{false};

    int thread_num_{0};
    int scheduler_thread_num_{0};  // Number of scheduler threads (excludes orchestrator)
    int cores_total_num_{0};
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
    int run_orchestrator(Runtime* runtime);
    int run_scheduler(Runtime* runtime, int thread_idx);
    void deinit();

    // Helper: enqueue task to appropriate ready queue
    void enqueue_ready_task(int task_id, int worker_type);
    // Helper: dequeue task from ready queue (-1 if empty)
    int dequeue_ready_task(int worker_type);
    // Helper: check if ready queue has tasks
    bool has_ready_tasks(int worker_type);

    // Helper: load SO from device memory
    void* dlopen_from_memory(const void* so_data, size_t so_size);
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

    // CRITICAL: Reinitialize DepListPool base pointer after host-to-device copy
    // The original pointer was set on host and is invalid on device
    runtime->reinit_dep_list_pool_base();
    DEV_INFO("PTO Scheduler: Reinitialized DepListPool base pointer");

    thread_num_ = runtime->sche_cpu_num;
    if (thread_num_ == 0) thread_num_ = 1;

    if (thread_num_ < 1 || thread_num_ > MAX_AICPU_THREADS) {
        DEV_ERROR("Invalid thread_num: %d", thread_num_);
        init_failed_.store(true, std::memory_order_release);
        return -1;
    }

    // Check orchestration mode
    int orch_mode = runtime->get_orchestration_mode();
    DEV_INFO("PTO Scheduler: orchestration_mode=%d", orch_mode);

    // In device orchestration mode, thread 3 is the orchestrator
    // Scheduler threads are 0 to (thread_num - 2) if orch_mode == 1
    if (orch_mode == 1) {
        scheduler_thread_num_ = thread_num_ - 1;  // Reserve thread 3 for orchestrator
        if (scheduler_thread_num_ < 1) {
            DEV_ERROR("Device orchestration requires at least 2 threads (1 scheduler + 1 orchestrator)");
            init_failed_.store(true, std::memory_order_release);
            return -1;
        }
        DEV_INFO("PTO Scheduler: Device orchestration mode, %d scheduler threads + 1 orchestrator", scheduler_thread_num_);
    } else {
        scheduler_thread_num_ = thread_num_;  // All threads are schedulers
    }

    // Use worker_count from platform (like host_build_graph does)
    cores_total_num_ = runtime->worker_count;
    thread_cores_num_ = cores_total_num_ / scheduler_thread_num_;

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

    DEV_INFO("PTO Config: threads=%d, scheduler_threads=%d, cores=%d, cores_per_thread=%d",
             thread_num_, scheduler_thread_num_, cores_total_num_, thread_cores_num_);

    // Platform sets core_type on each Handshake, so we just need to assign cores to threads
    int cores_per_thread = cores_total_num_ / scheduler_thread_num_;

    if (cores_total_num_ % scheduler_thread_num_ != 0) {
        DEV_ERROR("worker_count (%d) must be divisible by scheduler_thread_num (%d)", cores_total_num_, scheduler_thread_num_);
        init_failed_.store(true, std::memory_order_release);
        return -1;
    }

    // Assign cores to scheduler threads (simple sequential assignment)
    for (int t = 0; t < scheduler_thread_num_; t++) {
        int start_core = t * cores_per_thread;
        for (int i = 0; i < cores_per_thread; i++) {
            core_assignments_[t][i] = start_core + i;
        }
    }

    // CENTRALIZED HANDSHAKE: Handshake with ALL cores during init (like host_build_graph)
    // This ensures all cores are ready before any thread starts scheduling
    DEV_INFO("PTO Scheduler: Handshaking with all %d cores", cores_total_num_);
    Handshake* all_hanks = (Handshake*)runtime->workers;

    // Step 1: Send handshake signal to all cores
    for (int i = 0; i < cores_total_num_; i++) {
        all_hanks[i].aicpu_ready = 1;
    }

    // Step 2: Wait for all cores to respond
    for (int i = 0; i < cores_total_num_; i++) {
        Handshake* hank = &all_hanks[i];
        while (hank->aicore_done == 0) {
            // Busy wait for core response
        }
        DEV_INFO("PTO Scheduler: Core %d ready (type=%d)", i, static_cast<int>(hank->core_type));
    }
    DEV_INFO("PTO Scheduler: All cores handshake complete");

    // In host orchestration mode, tasks are already built
    // In device orchestration mode, tasks will be built by orchestrator thread
    if (orch_mode == 0) {
        total_tasks_.store(runtime->get_task_count(), std::memory_order_release);
    } else {
        total_tasks_.store(0, std::memory_order_release);  // Will be set after orchestration
    }
    completed_tasks_.store(0, std::memory_order_release);

    // Initialize FIFO ready queues
    for (int wt = 0; wt < PTO_NUM_WORKER_TYPES; wt++) {
        ready_head_[wt].store(0, std::memory_order_release);
        ready_tail_[wt].store(0, std::memory_order_release);
    }

    // In host orchestration mode, find initially ready tasks and enqueue them
    if (orch_mode == 0) {
        int initial_ready[RUNTIME_MAX_TASKS];
        int initial_count = runtime->get_initial_ready_tasks(initial_ready);

        DEV_INFO("PTO Init: Found %d initially ready tasks", initial_count);

        for (int i = 0; i < initial_count; i++) {
            Task* task = runtime->get_task(initial_ready[i]);
            // core_type: 0=AIC (CUBE), 1=AIV (VECTOR)
            int worker_type = task->core_type;
            enqueue_ready_task(initial_ready[i], worker_type);
        }
    }

    // Reset orchestration synchronization
    orchestration_done_.store(false, std::memory_order_release);
    orchestration_failed_.store(false, std::memory_order_release);

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

    // Check orchestration mode
    int orch_mode = runtime->get_orchestration_mode();

    // In device orchestration mode, thread (thread_num - 1) is the orchestrator
    if (orch_mode == 1 && thread_idx == thread_num_ - 1) {
        return run_orchestrator(runtime);
    }

    // All other threads are schedulers
    return run_scheduler(runtime, thread_idx);
}

// Helper: Load SO from device memory via memfd_create or temp file
void* PtoScheduler::dlopen_from_memory(const void* so_data, size_t so_size) {
#ifdef __linux__
    // Try memfd_create first (Linux 3.17+)
    int fd = syscall(SYS_memfd_create, "orch_so", 0);
    if (fd >= 0) {
        ssize_t written = write(fd, so_data, so_size);
        if (written < 0 || static_cast<size_t>(written) != so_size) {
            DEV_ERROR("Failed to write SO to memfd: wrote %zd of %zu bytes", written, so_size);
            close(fd);
        } else {
            char fd_path[64];
            snprintf(fd_path, sizeof(fd_path), "/proc/self/fd/%d", fd);
            void* handle = dlopen(fd_path, RTLD_NOW | RTLD_LOCAL);
            close(fd);

            if (handle != nullptr) {
                DEV_INFO("Loaded SO via memfd_create");
                return handle;
            }
            DEV_WARN("dlopen from memfd failed: %s, trying temp file", dlerror());
        }
    }
#endif

    // Fallback: Write SO to temp file and dlopen it
    // Use PID to avoid conflicts between processes
    char fd_path[128];
    snprintf(fd_path, sizeof(fd_path), "/tmp/pto_orch_so_aicpu_%d.so", getpid());

    int fd2 = open(fd_path, O_WRONLY | O_CREAT | O_TRUNC, 0700);
    if (fd2 < 0) {
        DEV_ERROR("Failed to create temp SO file: %s", fd_path);
        return nullptr;
    }

    ssize_t written = write(fd2, so_data, so_size);
    if (written < 0 || static_cast<size_t>(written) != so_size) {
        DEV_ERROR("Failed to write SO to temp file: wrote %zd of %zu bytes", written, so_size);
        close(fd2);
        unlink(fd_path);
        return nullptr;
    }
    close(fd2);

    void* handle = dlopen(fd_path, RTLD_NOW | RTLD_LOCAL);
    unlink(fd_path);  // Remove temp file after loading

    if (handle == nullptr) {
        DEV_ERROR("dlopen failed: %s", dlerror());
        return nullptr;
    }

    DEV_INFO("Loaded SO via temp file");
    return handle;
}

int PtoScheduler::run_orchestrator(Runtime* runtime) {
    DEV_INFO("Orchestrator thread starting");

    // Get orchestration SO from device memory
    void* so_data = runtime->get_device_orch_so();
    size_t so_size = runtime->get_device_orch_so_size();
    const char* func_name = runtime->get_orch_func_name();

    if (so_data == nullptr || so_size == 0 || func_name == nullptr || func_name[0] == '\0') {
        DEV_ERROR("Invalid orchestration parameters: so_data=%p, so_size=%zu, func_name=%s",
                  so_data, so_size, func_name ? func_name : "(null)");
        orchestration_failed_.store(true, std::memory_order_release);
        return -1;
    }

    DEV_INFO("Orchestrator: Loading SO (%zu bytes), function: %s", so_size, func_name);

    // Load SO from device memory
    void* handle = dlopen_from_memory(so_data, so_size);
    if (handle == nullptr) {
        DEV_ERROR("Orchestrator: Failed to load orchestration SO");
        orchestration_failed_.store(true, std::memory_order_release);
        return -1;
    }

    // Get orchestration function
    dlerror();  // Clear any existing error
    OrchestrationFunc orch_func = reinterpret_cast<OrchestrationFunc>(dlsym(handle, func_name));
    const char* dlsym_error = dlerror();
    if (dlsym_error != nullptr) {
        DEV_ERROR("Orchestrator: dlsym failed for '%s': %s", func_name, dlsym_error);
        dlclose(handle);
        orchestration_failed_.store(true, std::memory_order_release);
        return -1;
    }

    DEV_INFO("Orchestrator: Loaded function %s at %p", func_name, (void*)orch_func);

    // Note: pto_init() is already called on host side in init_runtime_impl
    // to pre-allocate HeapRing memory (host_api is not available on AICPU)

    // Call orchestration function (builds task graph)
    uint64_t* args = runtime->get_device_args();
    int args_count = runtime->get_device_args_count();

    DEV_INFO("Orchestrator: Calling orchestration function with %d args", args_count);

    int rc = orch_func(runtime, args, args_count);
    if (rc != 0) {
        DEV_ERROR("Orchestrator: Orchestration function failed with code %d", rc);
        dlclose(handle);
        orchestration_failed_.store(true, std::memory_order_release);
        return rc;
    }

    dlclose(handle);

    // Update total tasks count after orchestration
    int task_count = runtime->get_task_count();
    total_tasks_.store(task_count, std::memory_order_release);

    DEV_INFO("Orchestrator: Task graph built with %d tasks", task_count);

    // Find initially ready tasks and enqueue them
    int initial_ready[RUNTIME_MAX_TASKS];
    int initial_count = runtime->get_initial_ready_tasks(initial_ready);

    DEV_INFO("Orchestrator: Found %d initially ready tasks", initial_count);

    for (int i = 0; i < initial_count; i++) {
        Task* task = runtime->get_task(initial_ready[i]);
        int worker_type = task->core_type;
        enqueue_ready_task(initial_ready[i], worker_type);
    }

    // Signal schedulers that orchestration is complete
    orchestration_done_.store(true, std::memory_order_release);

    DEV_INFO("Orchestrator: Orchestration complete, signaled schedulers");

    // Track thread completion
    int prev_finished = finished_count_.fetch_add(1, std::memory_order_acq_rel);
    if (prev_finished + 1 == thread_num_) {
        finished_.store(true, std::memory_order_release);
        DEV_INFO("Orchestrator: Last thread, marking scheduler finished");
    }

    return 0;
}

int PtoScheduler::run_scheduler(Runtime* runtime, int thread_idx) {
    // Check orchestration mode
    int orch_mode = runtime->get_orchestration_mode();

    // In device orchestration mode, wait for orchestration to complete
    if (orch_mode == 1) {
        DEV_INFO("Scheduler %d: Waiting for orchestration to complete", thread_idx);

        while (!orchestration_done_.load(std::memory_order_acquire)) {
            if (orchestration_failed_.load(std::memory_order_acquire)) {
                DEV_ERROR("Scheduler %d: Orchestration failed, aborting", thread_idx);
                return -1;
            }
            // Spin wait (could add yield or sleep for better power efficiency)
        }

        DEV_INFO("Scheduler %d: Orchestration complete, proceeding", thread_idx);
    }

    const int* cur_thread_cores = core_assignments_[thread_idx];

    // Handshaking is already done in init() - no per-thread handshake needed
    DEV_INFO("PTO Thread %d: Runtime has %d tasks", thread_idx, runtime->get_task_count());

    // Main scheduling loop
    int completed = schedule_and_dispatch(*runtime, thread_idx, cur_thread_cores, thread_cores_num_);
    DEV_INFO("PTO Thread %d: Executed %d tasks from runtime", thread_idx, completed);

    // Shutdown AICore workers
    int rc = shutdown_cores(runtime, thread_idx, cur_thread_cores);
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

    // Reset orchestration synchronization
    orchestration_done_.store(false, std::memory_order_release);
    orchestration_failed_.store(false, std::memory_order_release);

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
