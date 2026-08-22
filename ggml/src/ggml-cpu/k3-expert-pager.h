#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// Route-aware page-cache control for mmap-backed Kimi K3 expert tensors.
// The hook is intentionally a no-op unless GGML_K3_EXPERT_PAGER=1.
void ggml_k3_expert_pager_prefetch(
        const struct ggml_tensor * experts,
        const struct ggml_tensor * ids);

#ifdef __cplusplus
}
#endif
