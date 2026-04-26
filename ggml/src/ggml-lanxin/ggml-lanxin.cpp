#include "ggml-lanxin.h"

#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace fs = std::filesystem;

#define GGML_LANXIN_NAME "LANXIN"
#define GGML_LANXIN_LOG  "ggml-lanxin: "

struct pacc_Device;
struct pacc_Program;
struct pacc_Kernel;

using pacc_Result = int32_t;
static constexpr pacc_Result pacc_Result_Success = 0;

static constexpr uint32_t PACC_KERNEL_ARG_KIND_SCALAR = 0;
static constexpr uint32_t PACC_KERNEL_ARG_KIND_POINTER = 1;
static constexpr uint32_t PACC_KERNEL_ARG_FLAG_SIGNED = 1u << 0;
static constexpr uint32_t PACC_KERNEL_ARG_FLAG_BUFFER_INPUT = 1u << 8;
static constexpr uint32_t PACC_KERNEL_ARG_FLAG_BUFFER_OUTPUT = 1u << 9;

struct PaccKernelArgRecord {
    uint32_t kind;
    uint32_t size;
    uint32_t flags;
    uint32_t reserved;
    uint64_t value;
};

struct PaccKernelBufferBinding {
    uint32_t arg_index;
    uint32_t flags;
    uint64_t addr;
    uint64_t size;
};

using fn_pacc_CreateDevice = pacc_Device * (*)(uint32_t device_id);
using fn_pacc_DestroyDevice = void (*)(pacc_Device * device);
using fn_pacc_CreateProgram = pacc_Program * (*)(void);
using fn_pacc_DestroyProgram = void (*)(pacc_Program * program);
using fn_pacc_LoadProgram = pacc_Result (*)(pacc_Program * program, const void * data, uint64_t size);
using fn_pacc_LoadProgramSource = pacc_Result (*)(
        pacc_Program * program,
        const char * target_arch,
        const char * source_name,
        const uint8_t * source_buffer,
        uint64_t source_len,
        const char * working_directory,
        const char * const * options,
        size_t option_count,
        const uint8_t * linked_bitcode,
        uint64_t linked_bitcode_len);
using fn_pacc_CreateKernelOnDevice = pacc_Kernel * (*)(pacc_Program * program, pacc_Device * device, const char * name);
using fn_pacc_DestroyKernel = void (*)(pacc_Kernel * kernel);
using fn_pacc_KernelClearLaunchState = pacc_Result (*)(pacc_Kernel * kernel);
using fn_pacc_KernelPushArgRecord = pacc_Result (*)(pacc_Kernel * kernel, const PaccKernelArgRecord * record);
using fn_pacc_KernelAddBufferBinding = pacc_Result (*)(pacc_Kernel * kernel, const PaccKernelBufferBinding * binding);
using fn_pacc_LaunchKernel = pacc_Result (*)(pacc_Kernel * kernel, uint32_t grid_x, uint32_t grid_y, uint32_t grid_z, uint32_t block_x, uint32_t block_y, uint32_t block_z);

struct lanxin_runtime_api {
    void * runtime_handle = nullptr;

    fn_pacc_CreateDevice create_device = nullptr;
    fn_pacc_DestroyDevice destroy_device = nullptr;
    fn_pacc_CreateProgram create_program = nullptr;
    fn_pacc_DestroyProgram destroy_program = nullptr;
    fn_pacc_LoadProgram load_program = nullptr;
    fn_pacc_LoadProgramSource load_program_source = nullptr;
    fn_pacc_CreateKernelOnDevice create_kernel_on_device = nullptr;
    fn_pacc_DestroyKernel destroy_kernel = nullptr;
    fn_pacc_KernelClearLaunchState clear_launch_state = nullptr;
    fn_pacc_KernelPushArgRecord push_arg_record = nullptr;
    fn_pacc_KernelAddBufferBinding add_buffer_binding = nullptr;
    fn_pacc_LaunchKernel launch_kernel = nullptr;
};

struct lanxin_loader_state {
    std::mutex mutex;
    bool probed = false;
    bool ready = false;
    bool reported_failure = false;
    std::string error;
    lanxin_runtime_api api;
};

struct lanxin_device_context {
    int index = 0;
    std::string name;
    std::string description;
};

struct lanxin_module_state {
    std::string source_name;
    pacc_Program * program = nullptr;
    std::unordered_map<std::string, pacc_Kernel *> kernels;
};

struct lanxin_backend_context {
    lanxin_device_context * dev = nullptr;
    pacc_Device * device = nullptr;
    std::mutex mutex;
    std::unordered_map<std::string, lanxin_module_state> modules;
};

static lanxin_loader_state & lanxin_loader() {
    static lanxin_loader_state loader;
    return loader;
}

static bool lanxin_log_enabled() {
    return std::getenv("GGML_LANXIN_LOG") != nullptr;
}

static std::string lanxin_dl_error() {
#if defined(_WIN32)
    const DWORD error = GetLastError();
    if (error == 0) {
        return "unknown error";
    }
    LPSTR message = nullptr;
    const DWORD size = FormatMessageA(
            FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
            nullptr,
            error,
            MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
            (LPSTR) &message,
            0,
            nullptr);
    std::string text = size > 0 && message != nullptr ? std::string(message, size) : std::string("unknown error");
    if (message != nullptr) {
        LocalFree(message);
    }
    return text;
#else
    const char * error = dlerror();
    return error != nullptr ? std::string(error) : std::string("unknown error");
#endif
}

static void * lanxin_dlopen(const fs::path & path) {
#if defined(_WIN32)
    return (void *) LoadLibraryA(path.string().c_str());
#else
    return dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
}

static void * lanxin_dlsym(void * handle, const char * symbol) {
#if defined(_WIN32)
    return (void *) GetProcAddress((HMODULE) handle, symbol);
#else
    return dlsym(handle, symbol);
#endif
}

static fs::path lanxin_repo_root() {
#ifdef GGML_LANXIN_SOURCE_DIR
    return fs::path(GGML_LANXIN_SOURCE_DIR).parent_path().parent_path().parent_path();
#else
    return fs::current_path();
#endif
}

static fs::path lanxin_default_hetgpu_root() {
    if (const char * env = std::getenv("GGML_LANXIN_HETGPU_ROOT")) {
        return fs::path(env);
    }
    return lanxin_repo_root().parent_path() / "hetGPU";
}

static std::vector<fs::path> lanxin_candidate_paths(const char * specific_env, std::initializer_list<const char *> relatives) {
    std::vector<fs::path> result;
    if (specific_env != nullptr) {
        if (const char * env = std::getenv(specific_env)) {
            result.emplace_back(env);
        }
    }

    const fs::path hetgpu_root = lanxin_default_hetgpu_root();
    for (const char * relative : relatives) {
        result.push_back(hetgpu_root / relative);
    }
    return result;
}

static bool lanxin_load_symbol(void * handle, const char * symbol, void ** out, std::string & error) {
    *out = lanxin_dlsym(handle, symbol);
    if (*out != nullptr) {
        return true;
    }
    error = "missing symbol ";
    error += symbol;
    error += ": ";
    error += lanxin_dl_error();
    return false;
}

static bool lanxin_probe_locked(lanxin_loader_state & loader) {
    if (loader.probed) {
        return loader.ready;
    }

    loader.probed = true;

    const auto runtime_candidates = lanxin_candidate_paths(
            "GGML_LANXIN_PACC_LIB",
            {
                "ext/pacc_runtime-sys/target/debug/libpacc_runtime_sys.so",
                "ext/pacc_runtime-sys/target/debug/libpacc_runtime_sys.dylib",
                "ext/pacc_runtime-sys/target/release/libpacc_runtime_sys.so",
                "ext/pacc_runtime-sys/target/release/libpacc_runtime_sys.dylib",
            });
    for (const auto & path : runtime_candidates) {
        if (!fs::exists(path)) {
            continue;
        }
        loader.api.runtime_handle = lanxin_dlopen(path);
        if (loader.api.runtime_handle != nullptr) {
            break;
        }
        loader.error = "failed to dlopen ";
        loader.error += path.string();
        loader.error += ": ";
        loader.error += lanxin_dl_error();
    }
    if (loader.api.runtime_handle == nullptr) {
        if (loader.error.empty()) {
            loader.error = "unable to locate libpacc_runtime_sys for Lanxin backend";
        }
        return false;
    }

    if (!lanxin_load_symbol(loader.api.runtime_handle, "pacc_CreateDevice", (void **) &loader.api.create_device, loader.error) ||
        !lanxin_load_symbol(loader.api.runtime_handle, "pacc_DestroyDevice", (void **) &loader.api.destroy_device, loader.error) ||
        !lanxin_load_symbol(loader.api.runtime_handle, "pacc_CreateProgram", (void **) &loader.api.create_program, loader.error) ||
        !lanxin_load_symbol(loader.api.runtime_handle, "pacc_DestroyProgram", (void **) &loader.api.destroy_program, loader.error) ||
        !lanxin_load_symbol(loader.api.runtime_handle, "pacc_LoadProgram", (void **) &loader.api.load_program, loader.error) ||
        !lanxin_load_symbol(loader.api.runtime_handle, "pacc_LoadProgramSource", (void **) &loader.api.load_program_source, loader.error) ||
        !lanxin_load_symbol(loader.api.runtime_handle, "pacc_CreateKernelOnDevice", (void **) &loader.api.create_kernel_on_device, loader.error) ||
        !lanxin_load_symbol(loader.api.runtime_handle, "pacc_DestroyKernel", (void **) &loader.api.destroy_kernel, loader.error) ||
        !lanxin_load_symbol(loader.api.runtime_handle, "pacc_KernelClearLaunchState", (void **) &loader.api.clear_launch_state, loader.error) ||
        !lanxin_load_symbol(loader.api.runtime_handle, "pacc_KernelPushArgRecord", (void **) &loader.api.push_arg_record, loader.error) ||
        !lanxin_load_symbol(loader.api.runtime_handle, "pacc_KernelAddBufferBinding", (void **) &loader.api.add_buffer_binding, loader.error) ||
        !lanxin_load_symbol(loader.api.runtime_handle, "pacc_LaunchKernel", (void **) &loader.api.launch_kernel, loader.error)) {
        return false;
    }

    loader.ready = true;
    return true;
}

static bool lanxin_runtime_available() {
    lanxin_loader_state & loader = lanxin_loader();
    std::lock_guard<std::mutex> lock(loader.mutex);
    const bool ready = lanxin_probe_locked(loader);
    if (!ready && !loader.reported_failure) {
        GGML_LOG_WARN(GGML_LANXIN_LOG "runtime unavailable: %s\n", loader.error.c_str());
        loader.reported_failure = true;
    }
    return ready;
}

static lanxin_runtime_api & lanxin_api() {
    lanxin_loader_state & loader = lanxin_loader();
    std::lock_guard<std::mutex> lock(loader.mutex);
    lanxin_probe_locked(loader);
    return loader.api;
}

static fs::path lanxin_kernel_source_path() {
    return lanxin_repo_root() / "ggml" / "src" / "ggml-cpu" / "lanxin" / "pacc-kernel.cpp";
}

static std::vector<uint8_t> lanxin_read_file(const fs::path & path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return {};
    }
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

static std::vector<uint8_t> lanxin_load_linked_bitcode() {
    const char * env = std::getenv("GGML_LANXIN_LINKED_BITCODE");
    if (env == nullptr || env[0] == '\0') {
        return {};
    }
    const fs::path path(env);
    std::vector<uint8_t> data = lanxin_read_file(path);
    if (data.empty()) {
        GGML_LOG_WARN(GGML_LANXIN_LOG "failed to read linked bitcode from %s\n", path.string().c_str());
    }
    return data;
}

static bool lanxin_push_scalar_arg(const lanxin_runtime_api & api, pacc_Kernel * kernel, uint32_t size, uint32_t flags, uint64_t value) {
    const PaccKernelArgRecord record = {
        /* .kind     = */ PACC_KERNEL_ARG_KIND_SCALAR,
        /* .size     = */ size,
        /* .flags    = */ flags,
        /* .reserved = */ 0,
        /* .value    = */ value,
    };
    return api.push_arg_record(kernel, &record) == pacc_Result_Success;
}

static bool lanxin_push_pointer_arg(const lanxin_runtime_api & api, pacc_Kernel * kernel, uint32_t arg_index, const void * ptr, uint64_t size, uint32_t flags) {
    const PaccKernelArgRecord record = {
        /* .kind     = */ PACC_KERNEL_ARG_KIND_POINTER,
        /* .size     = */ 8,
        /* .flags    = */ flags,
        /* .reserved = */ 0,
        /* .value    = */ (uint64_t) (uintptr_t) ptr,
    };
    const PaccKernelBufferBinding binding = {
        /* .arg_index = */ arg_index,
        /* .flags     = */ flags,
        /* .addr      = */ (uint64_t) (uintptr_t) ptr,
        /* .size      = */ size,
    };
    return api.push_arg_record(kernel, &record) == pacc_Result_Success &&
           api.add_buffer_binding(kernel, &binding) == pacc_Result_Success;
}

static std::string lanxin_kernel_symbol(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return "lanxin_pacc_mul_mat_f32_tile";
        case GGML_TYPE_F16:
            return "lanxin_pacc_mul_mat_fp16_tile";
        case GGML_TYPE_BF16:
            return "lanxin_pacc_mul_mat_bf16_tile";
        default:
            return {};
    }
}

static bool lanxin_supported_mul_mat(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_MUL_MAT) {
        return false;
    }

    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    if (src0 == nullptr || src1 == nullptr) {
        return false;
    }
    if (src0->type != src1->type) {
        return false;
    }
    if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_BF16) {
        return false;
    }
    if (op->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(op)) {
        return false;
    }
    if (src0->ne[2] != 1 || src0->ne[3] != 1) {
        return false;
    }
    if (src0->ne[0] != src1->ne[0]) {
        return false;
    }

    const int64_t m = src0->ne[1];
    const int64_t n = op->ne[1] * op->ne[2] * op->ne[3];
    const int64_t k = src0->ne[0];
    if (m <= 0 || n <= 0 || k <= 0) {
        return false;
    }
    if (m > std::numeric_limits<int32_t>::max() ||
        n > std::numeric_limits<int32_t>::max() ||
        k > std::numeric_limits<int32_t>::max()) {
        return false;
    }

    return lanxin_runtime_available();
}

static bool lanxin_compile_module(lanxin_module_state & module, const std::string & source_name) {
    const fs::path source_path = lanxin_kernel_source_path();
    std::vector<uint8_t> source = lanxin_read_file(source_path);
    if (source.empty()) {
        GGML_LOG_ERROR(GGML_LANXIN_LOG "failed to read kernel source %s\n", source_path.string().c_str());
        return false;
    }

    std::vector<uint8_t> linked_bc = lanxin_load_linked_bitcode();
    std::vector<std::string> option_storage = {
        "-std=c++17",
        "-O3",
        "-DNDEBUG",
    };
    std::vector<const char *> option_ptrs;
    option_ptrs.reserve(option_storage.size());
    for (const std::string & option : option_storage) {
        option_ptrs.push_back(option.c_str());
    }

    const lanxin_runtime_api & api = lanxin_api();
    if (api.create_program == nullptr || api.load_program_source == nullptr) {
        GGML_LOG_ERROR(GGML_LANXIN_LOG "runtime API incomplete while compiling %s\n", source_name.c_str());
        return false;
    }
    const fs::path workdir = lanxin_repo_root();
    module.source_name = source_name;
    module.program = api.create_program();
    if (module.program == nullptr) {
        GGML_LOG_ERROR(GGML_LANXIN_LOG "pacc_CreateProgram failed for %s\n", source_name.c_str());
        return false;
    }
    const pacc_Result rc = api.load_program_source(
            module.program,
            "riscv64-linux-gnu",
            source_name.c_str(),
            source.data(),
            (uint64_t) source.size(),
            workdir.string().c_str(),
            option_ptrs.data(),
            option_ptrs.size(),
            linked_bc.empty() ? nullptr : linked_bc.data(),
            (uint64_t) linked_bc.size());
    if (rc != pacc_Result_Success) {
        if (api.destroy_program != nullptr && module.program != nullptr) {
            api.destroy_program(module.program);
            module.program = nullptr;
        }
        GGML_LOG_ERROR(GGML_LANXIN_LOG "pacc_LoadProgramSource failed for %s\n", source_name.c_str());
        return false;
    }

    if (lanxin_log_enabled()) {
        GGML_LOG_INFO(GGML_LANXIN_LOG "compiled and loaded %s via pacc runtime\n", source_name.c_str());
    }

    return true;
}

static pacc_Kernel * lanxin_ensure_kernel_locked(lanxin_backend_context * ctx, const std::string & symbol) {
    const std::string module_key = "pacc-kernel.cpp";
    lanxin_module_state & module = ctx->modules[module_key];
    if (module.program == nullptr) {
        if (!lanxin_compile_module(module, "ggml/src/ggml-cpu/lanxin/pacc-kernel.cpp")) {
            return nullptr;
        }
    }

    auto it = module.kernels.find(symbol);
    if (it != module.kernels.end()) {
        return it->second;
    }

    const lanxin_runtime_api & api = lanxin_api();
    if (api.create_kernel_on_device == nullptr) {
        GGML_LOG_ERROR(GGML_LANXIN_LOG "runtime API missing pacc_CreateKernelOnDevice\n");
        return nullptr;
    }
    pacc_Kernel * kernel = api.create_kernel_on_device(module.program, ctx->device, symbol.c_str());
    if (kernel == nullptr) {
        GGML_LOG_ERROR(GGML_LANXIN_LOG "pacc_CreateKernelOnDevice failed for %s\n", symbol.c_str());
        return nullptr;
    }
    module.kernels.emplace(symbol, kernel);
    return kernel;
}

static bool lanxin_launch_mul_mat_locked(lanxin_backend_context * ctx, ggml_tensor * node) {
    const ggml_tensor * src0 = node->src[0];
    const ggml_tensor * src1 = node->src[1];
    if (src0 == nullptr || src1 == nullptr || src0->data == nullptr || src1->data == nullptr || node->data == nullptr) {
        return false;
    }

    const int32_t m = (int32_t) src0->ne[1];
    const int32_t n = (int32_t) (node->ne[1] * node->ne[2] * node->ne[3]);
    const int32_t k = (int32_t) src0->ne[0];
    const std::string symbol = lanxin_kernel_symbol(src0->type);
    if (symbol.empty()) {
        return false;
    }

    pacc_Kernel * kernel = lanxin_ensure_kernel_locked(ctx, symbol);
    if (kernel == nullptr) {
        return false;
    }

    const lanxin_runtime_api & api = lanxin_api();
    if (api.clear_launch_state == nullptr || api.push_arg_record == nullptr || api.add_buffer_binding == nullptr || api.launch_kernel == nullptr) {
        GGML_LOG_ERROR(GGML_LANXIN_LOG "runtime API incomplete while launching %s\n", symbol.c_str());
        return false;
    }
    if (api.clear_launch_state(kernel) != pacc_Result_Success) {
        GGML_LOG_ERROR(GGML_LANXIN_LOG "pacc_KernelClearLaunchState failed for %s\n", symbol.c_str());
        return false;
    }

    if (!lanxin_push_scalar_arg(api, kernel, 4, PACC_KERNEL_ARG_FLAG_SIGNED, (uint32_t) m) ||
        !lanxin_push_scalar_arg(api, kernel, 4, PACC_KERNEL_ARG_FLAG_SIGNED, (uint32_t) n) ||
        !lanxin_push_scalar_arg(api, kernel, 4, PACC_KERNEL_ARG_FLAG_SIGNED, (uint32_t) k) ||
        !lanxin_push_pointer_arg(api, kernel, 3, src0->data, (uint64_t) ggml_nbytes(src0), PACC_KERNEL_ARG_FLAG_BUFFER_INPUT) ||
        !lanxin_push_pointer_arg(api, kernel, 4, src1->data, (uint64_t) ggml_nbytes(src1), PACC_KERNEL_ARG_FLAG_BUFFER_INPUT) ||
        !lanxin_push_pointer_arg(api, kernel, 5, node->data, (uint64_t) ggml_nbytes(node), PACC_KERNEL_ARG_FLAG_BUFFER_OUTPUT)) {
        GGML_LOG_ERROR(GGML_LANXIN_LOG "failed to pack launch ABI for %s\n", symbol.c_str());
        return false;
    }

    if (lanxin_log_enabled()) {
        GGML_LOG_INFO(
                GGML_LANXIN_LOG "launching %s m=%d n=%d k=%d on %s\n",
                symbol.c_str(),
                m,
                n,
                k,
                ctx->dev->name.c_str());
    }

    if (api.launch_kernel(kernel, 1, 1, 1, 1, 1, 1) != pacc_Result_Success) {
        GGML_LOG_ERROR(GGML_LANXIN_LOG "pacc_LaunchKernel failed for %s\n", symbol.c_str());
        return false;
    }

    return true;
}

static const char * ggml_backend_lanxin_get_name(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    return GGML_LANXIN_NAME;
}

static void ggml_backend_lanxin_free(ggml_backend_t backend) {
    if (backend == nullptr) {
        return;
    }

    auto * ctx = (lanxin_backend_context *) backend->context;
    if (ctx != nullptr) {
        const lanxin_runtime_api & api = lanxin_api();
        for (auto & module_it : ctx->modules) {
            for (auto & kernel_it : module_it.second.kernels) {
                if (kernel_it.second != nullptr && api.destroy_kernel != nullptr) {
                    api.destroy_kernel(kernel_it.second);
                }
            }
            if (module_it.second.program != nullptr && api.destroy_program != nullptr) {
                api.destroy_program(module_it.second.program);
            }
        }
        if (ctx->device != nullptr && api.destroy_device != nullptr) {
            api.destroy_device(ctx->device);
        }
        delete ctx;
    }

    delete backend;
}

static enum ggml_status ggml_backend_lanxin_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    auto * ctx = (lanxin_backend_context *) backend->context;
    if (ctx == nullptr || cgraph == nullptr) {
        return GGML_STATUS_FAILED;
    }

    std::lock_guard<std::mutex> lock(ctx->mutex);
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        ggml_tensor * node = cgraph->nodes[i];
        switch (node->op) {
            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                continue;
            case GGML_OP_MUL_MAT:
                if (!lanxin_launch_mul_mat_locked(ctx, node)) {
                    return GGML_STATUS_FAILED;
                }
                continue;
            default:
                GGML_LOG_ERROR(GGML_LANXIN_LOG "unsupported op %s in LANXIN subgraph\n", ggml_op_name(node->op));
                return GGML_STATUS_FAILED;
        }
    }

    return GGML_STATUS_SUCCESS;
}

static const ggml_backend_i ggml_backend_lanxin_i = {
    /* .get_name                = */ ggml_backend_lanxin_get_name,
    /* .free                    = */ ggml_backend_lanxin_free,
    /* .set_tensor_async        = */ nullptr,
    /* .get_tensor_async        = */ nullptr,
    /* .cpy_tensor_async        = */ nullptr,
    /* .synchronize             = */ nullptr,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ ggml_backend_lanxin_graph_compute,
    /* .event_record            = */ nullptr,
    /* .event_wait              = */ nullptr,
    /* .graph_optimize          = */ nullptr,
};

static ggml_guid_t ggml_backend_lanxin_guid() {
    static ggml_guid guid = {
        0x0f, 0x28, 0x5e, 0x11, 0x2d, 0x90, 0x44, 0x97,
        0xb0, 0x37, 0x4b, 0x43, 0x3a, 0x53, 0xa1, 0x10
    };
    return &guid;
}

static ggml_backend_t ggml_backend_lanxin_device_init(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);

    if (!lanxin_runtime_available()) {
        return nullptr;
    }

    auto * dev_ctx = (lanxin_device_context *) dev->context;
    auto * ctx = new lanxin_backend_context();
    ctx->dev = dev_ctx;
    const lanxin_runtime_api & api = lanxin_api();
    if (api.create_device == nullptr) {
        delete ctx;
        return nullptr;
    }
    ctx->device = api.create_device((uint32_t) dev_ctx->index);
    if (ctx->device == nullptr) {
        delete ctx;
        return nullptr;
    }

    return new ggml_backend {
        /* .guid      = */ ggml_backend_lanxin_guid(),
        /* .iface     = */ ggml_backend_lanxin_i,
        /* .device    = */ dev,
        /* .context   = */ ctx,
    };
}

static const char * ggml_backend_lanxin_device_get_name(ggml_backend_dev_t dev) {
    return ((lanxin_device_context *) dev->context)->name.c_str();
}

static const char * ggml_backend_lanxin_device_get_description(ggml_backend_dev_t dev) {
    return ((lanxin_device_context *) dev->context)->description.c_str();
}

static void ggml_backend_lanxin_device_get_memory(ggml_backend_dev_t dev, size_t * free_mem, size_t * total_mem) {
    GGML_UNUSED(dev);
    *free_mem = 0;
    *total_mem = 0;
}

static enum ggml_backend_dev_type ggml_backend_lanxin_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_lanxin_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    auto * ctx = (lanxin_device_context *) dev->context;
    props->name = ctx->name.c_str();
    props->description = ctx->description.c_str();
    props->memory_free = 0;
    props->memory_total = 0;
    props->type = GGML_BACKEND_DEVICE_TYPE_GPU;
    props->device_id = nullptr;
    props->caps.async = false;
    props->caps.host_buffer = false;
    props->caps.buffer_from_host_ptr = true;
    props->caps.events = false;
}

static ggml_backend_buffer_type_t ggml_backend_lanxin_device_get_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return ggml_backend_cpu_buffer_type();
}

static ggml_backend_buffer_t ggml_backend_lanxin_device_buffer_from_host_ptr(
        ggml_backend_dev_t dev,
        void * ptr,
        size_t size,
        size_t max_tensor_size) {
    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);
}

static bool ggml_backend_lanxin_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    if (op == nullptr) {
        return false;
    }
    if (op->op == GGML_OP_NONE || op->op == GGML_OP_RESHAPE || op->op == GGML_OP_VIEW ||
        op->op == GGML_OP_PERMUTE || op->op == GGML_OP_TRANSPOSE) {
        return true;
    }
    return lanxin_supported_mul_mat(op);
}

static bool ggml_backend_lanxin_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(dev);
    return ggml_backend_buft_is_host(buft);
}

static bool ggml_backend_lanxin_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    return lanxin_supported_mul_mat(op);
}

static const ggml_backend_device_i ggml_backend_lanxin_device_i = {
    /* .get_name             = */ ggml_backend_lanxin_device_get_name,
    /* .get_description      = */ ggml_backend_lanxin_device_get_description,
    /* .get_memory           = */ ggml_backend_lanxin_device_get_memory,
    /* .get_type             = */ ggml_backend_lanxin_device_get_type,
    /* .get_props            = */ ggml_backend_lanxin_device_get_props,
    /* .init_backend         = */ ggml_backend_lanxin_device_init,
    /* .get_buffer_type      = */ ggml_backend_lanxin_device_get_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ ggml_backend_lanxin_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_lanxin_device_supports_op,
    /* .supports_buft        = */ ggml_backend_lanxin_device_supports_buft,
    /* .offload_op           = */ ggml_backend_lanxin_device_offload_op,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

static const char * ggml_backend_lanxin_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_LANXIN_NAME;
}

static size_t ggml_backend_lanxin_reg_get_device_count(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return 1;
}

static ggml_backend_dev_t ggml_backend_lanxin_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    static lanxin_device_context ctx = {
        /* .index       = */ 0,
        /* .name        = */ "LANXIN0",
        /* .description = */ "Lanxin PACC backend (remote link + ELF launch)",
    };
    static ggml_backend_device dev = {
        /* .iface   = */ ggml_backend_lanxin_device_i,
        /* .reg     = */ reg,
        /* .context = */ &ctx,
    };
    return &dev;
}

static const ggml_backend_reg_i ggml_backend_lanxin_reg_i = {
    /* .get_name         = */ ggml_backend_lanxin_reg_get_name,
    /* .get_device_count = */ ggml_backend_lanxin_reg_get_device_count,
    /* .get_device       = */ ggml_backend_lanxin_reg_get_device,
    /* .get_proc_address = */ nullptr,
};

ggml_backend_reg_t ggml_backend_lanxin_reg(void) {
    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_lanxin_reg_i,
        /* .context     = */ nullptr,
    };
    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_lanxin_reg)
