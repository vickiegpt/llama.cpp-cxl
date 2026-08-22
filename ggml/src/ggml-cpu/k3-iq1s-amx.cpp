#include "k3-iq1s-amx.h"

#include "ggml-quants.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(__linux__) && defined(__x86_64__) && defined(__AMX_TILE__) && defined(__AMX_BF16__)
#include <immintrin.h>
#include <sys/syscall.h>
#include <unistd.h>

#ifndef ARCH_REQ_XCOMP_PERM
#define ARCH_REQ_XCOMP_PERM 0x1023
#endif
#ifndef XFEATURE_XTILEDATA
#define XFEATURE_XTILEDATA 18
#endif

namespace {

struct alignas(64) amx_tile_config {
    uint8_t palette;
    uint8_t start_row;
    uint8_t reserved[14];
    uint16_t colsb[8];
    uint8_t rows[8];
};

std::atomic<uint64_t> g_amx_calls{0};
std::atomic<uint64_t> g_amx_tiles{0};
std::atomic<bool> g_amx_announced{false};

static bool enabled() {
    const char * value = std::getenv("GGML_K3_IQ1S_AMX");
    return value && std::strcmp(value, "0") != 0;
}

static bool acquire_tile_permission() {
    thread_local int state = -1;
    if (state == -1) {
        state = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) == 0 ? 1 : 0;
    }
    return state == 1;
}

static uint16_t fp32_to_bf16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t rounding = 0x7fffU + ((bits >> 16) & 1U);
    return static_cast<uint16_t>((bits + rounding) >> 16);
}

static void amx_bf16_row_matmul(
        const float * activation,
        const float * weights,
        int64_t k,
        int outputs,
        float * destination) {
    alignas(64) uint16_t tile_a[16][32] = {};
    alignas(64) uint16_t tile_b[16][32] = {};
    alignas(64) float tile_c[16][16] = {};
    amx_tile_config config = {};
    config.palette = 1;
    config.colsb[0] = 64;
    config.rows[0] = 16;
    config.colsb[1] = 64;
    config.rows[1] = 16;
    config.colsb[2] = 64;
    config.rows[2] = 16;

    _tile_loadconfig(&config);
    _tile_zero(0);
    for (int64_t offset = 0; offset < k; offset += 32) {
        const int count = static_cast<int>(k - offset < 32 ? k - offset : 32);
        std::memset(tile_a, 0, sizeof(tile_a));
        std::memset(tile_b, 0, sizeof(tile_b));
        for (int index = 0; index < count; ++index) {
            tile_a[0][index] = fp32_to_bf16(activation[offset + index]);
        }
        for (int pair = 0; pair < 16; ++pair) {
            const int first = pair * 2;
            for (int output = 0; output < outputs; ++output) {
                tile_b[pair][2 * output] = first < count
                    ? fp32_to_bf16(weights[output * k + offset + first]) : 0;
                tile_b[pair][2 * output + 1] = first + 1 < count
                    ? fp32_to_bf16(weights[output * k + offset + first + 1]) : 0;
            }
        }
        _tile_loadd(1, tile_a, 64);
        _tile_loadd(2, tile_b, 64);
        _tile_dpbf16ps(0, 1, 2);
    }
    _tile_stored(0, tile_c, 64);
    _tile_release();
    std::memcpy(destination, tile_c[0], static_cast<size_t>(outputs) * sizeof(float));
}

} // namespace
#endif

extern "C" bool ggml_k3_iq1s_amx_mul_mat_id(
        const struct ggml_compute_params * params,
        struct ggml_tensor * dst) {
#if !defined(__linux__) || !defined(__x86_64__) || !defined(__AMX_TILE__) || !defined(__AMX_BF16__)
    (void) params;
    (void) dst;
    return false;
#else
    const ggml_tensor * weights = dst->src[0];
    const ggml_tensor * input = dst->src[1];
    const ggml_tensor * ids = dst->src[2];
    if (params->use_ref || !enabled() || weights->type != GGML_TYPE_IQ1_S || input->type != GGML_TYPE_F32 ||
        ids->type != GGML_TYPE_I32 || !acquire_tile_permission()) {
        return false;
    }
    if (!g_amx_announced.exchange(true)) {
        std::fprintf(stderr,
                "K3_IQ1S_AMX_INIT kernel=tdpbf16ps source=iq1_s conversion=route-selected-bf16\n");
    }

    const int64_t k = weights->ne[0];
    const int64_t n = weights->ne[1];
    const int64_t assignments = ids->ne[0] * ids->ne[1];
    const int64_t output_blocks = (n + 15) / 16;
    const int64_t tasks = assignments * output_blocks;
    std::vector<float> decoded(static_cast<size_t>(16 * k));

    for (int64_t task = params->ith; task < tasks; task += params->nth) {
        const int64_t assignment = task / output_blocks;
        const int64_t output_block = task % output_blocks;
        const int64_t slot = assignment % ids->ne[0];
        const int64_t token = assignment / ids->ne[0];
        const int32_t expert = *reinterpret_cast<const int32_t *>(
                static_cast<const char *>(ids->data) + token * ids->nb[1] + slot * ids->nb[0]);
        if (expert < 0 || expert >= weights->ne[2]) {
            continue;
        }
        const int outputs = static_cast<int>(n - output_block * 16 < 16 ? n - output_block * 16 : 16);
        for (int output = 0; output < outputs; ++output) {
            const char * row = static_cast<const char *>(weights->data) +
                expert * weights->nb[2] + (output_block * 16 + output) * weights->nb[1];
            dequantize_row_iq1_s(reinterpret_cast<const block_iq1_s *>(row),
                                 decoded.data() + static_cast<size_t>(output) * k, k);
        }
        const float * activation = reinterpret_cast<const float *>(
                static_cast<const char *>(input->data) + token * input->nb[2] + slot * input->nb[1]);
        float * output = reinterpret_cast<float *>(
                static_cast<char *>(dst->data) + token * dst->nb[2] + slot * dst->nb[1]) + output_block * 16;
        amx_bf16_row_matmul(activation, decoded.data(), k, outputs, output);
        g_amx_tiles.fetch_add(static_cast<uint64_t>((k + 31) / 32), std::memory_order_relaxed);
    }
    if (params->ith == 0) {
        const uint64_t calls = g_amx_calls.fetch_add(1, std::memory_order_relaxed) + 1;
        const uint64_t every = [] {
            const char * value = std::getenv("GGML_K3_IQ1S_AMX_STATS_EVERY");
            return value ? std::strtoull(value, nullptr, 10) : 100ULL;
        }();
        if (every && calls % every == 0) {
            std::fprintf(stderr, "K3_IQ1S_AMX_STATS calls=%llu bf16_tiles=%llu tensor=%s\n",
                    static_cast<unsigned long long>(calls),
                    static_cast<unsigned long long>(g_amx_tiles.load(std::memory_order_relaxed)),
                    weights->name);
        }
    }
    return true;
#endif
}
