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

/**
 * Orchestration function signature.
 */
typedef int (*OrchestrationFunc)(Runtime* runtime, uint64_t* args, int arg_count);

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize a pre-allocated runtime with dynamic orchestration.
 */
int init_runtime_impl(Runtime *runtime,
                    const uint8_t* orch_so_binary,
                    size_t orch_so_size,
                    const char* orch_func_name,
                    uint64_t* func_args,
                    int func_args_count) {
    if (runtime == nullptr) {
        std::cerr << "[PTO] Error: Runtime pointer is null\n";
        return -1;
    }
    if (orch_so_binary == nullptr || orch_so_size == 0 || orch_func_name == nullptr) {
        std::cerr << "[PTO] Error: Invalid orchestration parameters\n";
        return -1;
    }

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

    runtime->clear_tensor_pairs();

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
