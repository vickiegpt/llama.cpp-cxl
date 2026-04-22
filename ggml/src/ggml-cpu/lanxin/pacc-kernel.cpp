// Minimal Lanxin/PACC mul_mat kernels compiled remotely through hetGPU COMGR.
// The ABI is intentionally small so the host backend can pack launch records
// without depending on the full ggml CPU runtime.

#include <stddef.h>
#include <stdint.h>
#include <string.h>

using ggml_fp16_t = uint16_t;
struct ggml_bf16_t {
    uint16_t bits;
};

static inline float lanxin_bf16_to_fp32(ggml_bf16_t value) {
    uint32_t bits = (uint32_t) value.bits << 16;
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static inline float lanxin_fp16_to_fp32(ggml_fp16_t value) {
    const uint32_t sign = ((uint32_t) value & 0x8000u) << 16;
    const uint32_t exp  = ((uint32_t) value >> 10) & 0x1fu;
    const uint32_t frac = (uint32_t) value & 0x03ffu;

    uint32_t bits = 0;
    if (exp == 0) {
        if (frac == 0) {
            bits = sign;
        } else {
            uint32_t mant = frac;
            int32_t shift = -1;
            do {
                shift++;
                mant <<= 1;
            } while ((mant & 0x0400u) == 0);
            mant &= 0x03ffu;
            const uint32_t exp32 = (uint32_t) (127 - 15 - shift);
            bits = sign | (exp32 << 23) | (mant << 13);
        }
    } else if (exp == 0x1fu) {
        bits = sign | 0x7f800000u | (frac << 13);
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (frac << 13);
    }

    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

extern "C" void lanxin_pacc_mul_mat_f32_tile(
        int m,
        int n,
        int k,
        const float * a,
        const float * b,
        float * c) {
    for (int row_b = 0; row_b < n; ++row_b) {
        const float * b_row = b + (size_t) row_b * (size_t) k;
        float * c_row = c + (size_t) row_b * (size_t) m;
        for (int row_a = 0; row_a < m; ++row_a) {
            const float * a_row = a + (size_t) row_a * (size_t) k;
            float sum = 0.0f;
            for (int col = 0; col < k; ++col) {
                sum += a_row[col] * b_row[col];
            }
            c_row[row_a] = sum;
        }
    }
}

extern "C" void lanxin_pacc_mul_mat_fp16_tile(
        int m,
        int n,
        int k,
        const ggml_fp16_t * a,
        const ggml_fp16_t * b,
        float * c) {
    for (int row_b = 0; row_b < n; ++row_b) {
        const ggml_fp16_t * b_row = b + (size_t) row_b * (size_t) k;
        float * c_row = c + (size_t) row_b * (size_t) m;
        for (int row_a = 0; row_a < m; ++row_a) {
            const ggml_fp16_t * a_row = a + (size_t) row_a * (size_t) k;
            float sum = 0.0f;
            for (int col = 0; col < k; ++col) {
                sum += lanxin_fp16_to_fp32(a_row[col]) * lanxin_fp16_to_fp32(b_row[col]);
            }
            c_row[row_a] = sum;
        }
    }
}

extern "C" void lanxin_pacc_mul_mat_bf16_tile(
        int m,
        int n,
        int k,
        const ggml_bf16_t * a,
        const ggml_bf16_t * b,
        float * c) {
    for (int row_b = 0; row_b < n; ++row_b) {
        const ggml_bf16_t * b_row = b + (size_t) row_b * (size_t) k;
        float * c_row = c + (size_t) row_b * (size_t) m;
        for (int row_a = 0; row_a < m; ++row_a) {
            const ggml_bf16_t * a_row = a + (size_t) row_a * (size_t) k;
            float sum = 0.0f;
            for (int col = 0; col < k; ++col) {
                sum += lanxin_bf16_to_fp32(a_row[col]) * lanxin_bf16_to_fp32(b_row[col]);
            }
            c_row[row_a] = sum;
        }
    }
}
