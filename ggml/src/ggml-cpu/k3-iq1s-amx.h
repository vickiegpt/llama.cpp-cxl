#pragma once

#include "ggml.h"
#include "ggml-cpu-impl.h"

#ifdef __cplusplus
extern "C" {
#endif

// Executes K3 IQ1_S MUL_MAT_ID with a BF16 AMX tile kernel when explicitly
// enabled. Returns false when the operation should use the regular CPU path.
bool ggml_k3_iq1s_amx_mul_mat_id(
        const struct ggml_compute_params * params,
        struct ggml_tensor * dst);

#ifdef __cplusplus
}
#endif
