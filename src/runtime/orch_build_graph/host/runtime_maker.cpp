/**
 * PTO Runtime Maker - Host-side initialization and finalization
 *
 * Provides init_runtime_impl and validate_runtime_impl functions that work with
 * pluggable orchestration functions for building task graphs.
 */

#include "runtime.h"
#include <stdint.h>
#include <stddef.h>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <iostream>
#include <unistd.h>
#include <sys/mman.h>

/**
 * Orchestration function signature.
 */
typedef int (*OrchestrationFunc)(Runtime* runtime, uint64_t* args, int arg_count);

/**
 * Argument types for parameter conversion (matches ArgType in pto_runtime_c_api.h)
 */
enum {
    ARG_SCALAR = 0,
    ARG_INPUT_PTR = 1,
    ARG_OUTPUT_PTR = 2,
    ARG_INOUT_PTR = 3,
};

/**
 * Orchestration modes (matches OrchestrationMode in pto_runtime_c_api.h)
 */
enum {
    ORCH_MODE_HOST = 0,
    ORCH_MODE_DEVICE = 1,
};

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize a pre-allocated runtime with dynamic orchestration.
 *
 * @param runtime           Initialized Runtime pointer
 * @param orch_so_binary    Orchestration SO binary data
 * @param orch_so_size      Size of orchestration SO
 * @param orch_func_name    Name of orchestration function
 * @param func_args         Arguments for orchestration
 * @param func_args_count   Number of arguments
 * @param arg_types         Array of ArgType values (can be NULL for host mode)
 * @param arg_sizes         Array of sizes for pointer args (can be NULL for host mode)
 * @param orchestration_mode ORCH_MODE_HOST or ORCH_MODE_DEVICE
 */
int init_runtime_impl(Runtime *runtime,
                    const uint8_t* orch_so_binary,
                    size_t orch_so_size,
                    const char* orch_func_name,
                    uint64_t* func_args,
                    int func_args_count,
                    int* arg_types,
                    uint64_t* arg_sizes,
                    int orchestration_mode) {
    if (runtime == nullptr) {
        std::cerr << "[PTO] Error: Runtime pointer is null\n";
        return -1;
    }
    if (orch_so_binary == nullptr || orch_so_size == 0 || orch_func_name == nullptr) {
        std::cerr << "[PTO] Error: Invalid orchestration parameters\n";
        return -1;
    }

    runtime->clear_tensor_pairs();
    runtime->set_orchestration_mode(orchestration_mode);

    // Device orchestration mode: prepare for AICPU thread 3 execution
    if (orchestration_mode == ORCH_MODE_DEVICE) {
        std::cout << "[PTO] Device orchestration mode: preparing for AICPU execution\n";

        // Copy orchestration SO to device memory
        void* dev_so = runtime->host_api.device_malloc(orch_so_size);
        if (dev_so == nullptr) {
            std::cerr << "[PTO] Error: Failed to allocate device memory for orchestration SO\n";
            return -1;
        }
        int copy_rc = runtime->host_api.copy_to_device(dev_so, orch_so_binary, orch_so_size);
        if (copy_rc != 0) {
            std::cerr << "[PTO] Error: Failed to copy orchestration SO to device\n";
            runtime->host_api.device_free(dev_so);
            return -1;
        }
        runtime->set_device_orch_so(dev_so, orch_so_size);
        runtime->set_orch_func_name(orch_func_name);

        std::cout << "[PTO] Orchestration SO copied to device: " << orch_so_size << " bytes\n";

        // Convert host pointers to device pointers based on arg_types
        uint64_t device_args[RUNTIME_MAX_ARGS];
        for (int i = 0; i < func_args_count && i < RUNTIME_MAX_ARGS; i++) {
            int arg_type = (arg_types != nullptr) ? arg_types[i] : ARG_SCALAR;
            uint64_t arg_size = (arg_sizes != nullptr) ? arg_sizes[i] : 0;

            switch (arg_type) {
                case ARG_INPUT_PTR: {
                    // Allocate device memory and copy input data
                    void* host_ptr = reinterpret_cast<void*>(func_args[i]);
                    void* dev_ptr = runtime->host_api.device_malloc(arg_size);
                    if (dev_ptr == nullptr) {
                        std::cerr << "[PTO] Error: Failed to allocate device memory for arg " << i << "\n";
                        return -1;
                    }
                    int rc = runtime->host_api.copy_to_device(dev_ptr, host_ptr, arg_size);
                    if (rc != 0) {
                        std::cerr << "[PTO] Error: Failed to copy arg " << i << " to device\n";
                        return -1;
                    }
                    device_args[i] = reinterpret_cast<uint64_t>(dev_ptr);
                    // Record for cleanup (no copy-back needed)
                    runtime->record_tensor_pair(nullptr, dev_ptr, arg_size);
                    std::cout << "[PTO] Arg " << i << ": INPUT_PTR, " << arg_size << " bytes copied to device\n";
                    break;
                }
                case ARG_OUTPUT_PTR: {
                    // Allocate device memory, record for copy-back
                    void* host_ptr = reinterpret_cast<void*>(func_args[i]);
                    void* dev_ptr = runtime->host_api.device_malloc(arg_size);
                    if (dev_ptr == nullptr) {
                        std::cerr << "[PTO] Error: Failed to allocate device memory for arg " << i << "\n";
                        return -1;
                    }
                    device_args[i] = reinterpret_cast<uint64_t>(dev_ptr);
                    // Record for copy-back during finalize
                    runtime->record_tensor_pair(host_ptr, dev_ptr, arg_size);
                    std::cout << "[PTO] Arg " << i << ": OUTPUT_PTR, " << arg_size << " bytes allocated on device\n";
                    break;
                }
                case ARG_INOUT_PTR: {
                    // Allocate, copy input, record for copy-back
                    void* host_ptr = reinterpret_cast<void*>(func_args[i]);
                    void* dev_ptr = runtime->host_api.device_malloc(arg_size);
                    if (dev_ptr == nullptr) {
                        std::cerr << "[PTO] Error: Failed to allocate device memory for arg " << i << "\n";
                        return -1;
                    }
                    int rc = runtime->host_api.copy_to_device(dev_ptr, host_ptr, arg_size);
                    if (rc != 0) {
                        std::cerr << "[PTO] Error: Failed to copy arg " << i << " to device\n";
                        return -1;
                    }
                    device_args[i] = reinterpret_cast<uint64_t>(dev_ptr);
                    // Record for copy-back during finalize
                    runtime->record_tensor_pair(host_ptr, dev_ptr, arg_size);
                    std::cout << "[PTO] Arg " << i << ": INOUT_PTR, " << arg_size << " bytes copied to device\n";
                    break;
                }
                case ARG_SCALAR:
                default:
                    // Pass scalar value directly
                    device_args[i] = func_args[i];
                    std::cout << "[PTO] Arg " << i << ": SCALAR, value=" << func_args[i] << "\n";
                    break;
            }
        }

        // Store converted device args in runtime for AICPU thread 3
        runtime->set_device_args(device_args, func_args_count);

        // Pre-initialize PTO mode on host side (allocates HeapRing memory)
        // This must be done before AICPU execution because host_api is not available on AICPU
        runtime->pto_init();

        std::cout << "[PTO] Device orchestration prepared. Orchestration will run on AICPU thread 3.\n";
        return 0;
    }

    // Host orchestration mode: load and execute SO on host
    std::cout << "[PTO] Host orchestration mode: executing on host\n";

    // Load orchestration SO from binary data via temp file
    char fd_path[128];
    snprintf(fd_path, sizeof(fd_path), "/tmp/pto_orch_so_%d.so", getpid());

    int fd = open(fd_path, O_WRONLY | O_CREAT | O_TRUNC, 0700);
    if (fd < 0) {
        std::cerr << "[PTO] Error: Failed to create temp SO file\n";
        return -1;
    }

    ssize_t written = write(fd, orch_so_binary, orch_so_size);
    if (written < 0 || static_cast<size_t>(written) != orch_so_size) {
        std::cerr << "[PTO] Error: Failed to write orchestration SO to temp file\n";
        close(fd);
        unlink(fd_path);
        return -1;
    }
    close(fd);

    void* handle = dlopen(fd_path, RTLD_NOW | RTLD_LOCAL);
    unlink(fd_path);
    if (handle == nullptr) {
        std::cerr << "[PTO] Error: dlopen failed: " << dlerror() << "\n";
        return -1;
    }

    dlerror();
    OrchestrationFunc orch_func =
        reinterpret_cast<OrchestrationFunc>(dlsym(handle, orch_func_name));
    const char* dlsym_error = dlerror();
    if (dlsym_error != nullptr) {
        std::cerr << "[PTO] Error: dlsym failed for '" << orch_func_name << "': " << dlsym_error << "\n";
        dlclose(handle);
        return -1;
    }

    std::cout << "[PTO] Loaded orchestration function: " << orch_func_name << "\n";

    std::cout << "\n=== [PTO] Calling Orchestration Function ===\n";
    std::cout << "Args count: " << func_args_count << '\n';

    int rc = orch_func(runtime, func_args, func_args_count);
    if (rc != 0) {
        std::cerr << "[PTO] Error: Orchestration function failed with code " << rc << '\n';
        runtime->clear_tensor_pairs();
        dlclose(handle);
        return rc;
    }

    std::cout << "\n[PTO] Runtime initialized. Ready for execution.\n";

    return 0;
}

/**
 * Validate runtime results and cleanup.
 */
int validate_runtime_impl(Runtime *runtime) {
    if (runtime == nullptr) {
        std::cerr << "[PTO] Error: Runtime pointer is null\n";
        return -1;
    }

    int rc = 0;

    std::cout << "\n=== [PTO] Copying Results Back to Host ===\n";

    TensorPair* tensor_pairs = runtime->get_tensor_pairs();
    int tensor_pair_count = runtime->get_tensor_pair_count();

    for (int i = 0; i < tensor_pair_count; i++) {
        const TensorPair& pair = tensor_pairs[i];
        // Skip copy-back for input-only tensors (host_ptr == nullptr)
        if (pair.host_ptr == nullptr) {
            std::cout << "[PTO] Tensor " << i << ": skipping copy-back (input-only)\n";
            continue;
        }
        int copy_rc = runtime->host_api.copy_from_device(pair.host_ptr, pair.dev_ptr, pair.size);
        if (copy_rc != 0) {
            std::cerr << "[PTO] Error: Failed to copy tensor " << i << " from device: " << copy_rc << '\n';
            rc = copy_rc;
        } else {
            std::cout << "[PTO] Tensor " << i << ": " << pair.size << " bytes copied to host\n";
        }
    }

    std::cout << "\n=== [PTO] Cleaning Up ===\n";
    for (int i = 0; i < tensor_pair_count; i++) {
        runtime->host_api.device_free(tensor_pairs[i].dev_ptr);
    }
    std::cout << "[PTO] Freed " << tensor_pair_count << " device tensors\n";

    runtime->clear_tensor_pairs();

    std::cout << "=== [PTO] Finalize Complete ===\n";

    return rc;
}

#ifdef __cplusplus
}
#endif
