#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// Route-aware cache control for Kimi K3 expert tensors. The cxl backend lazily
// streams routed expert slices through CXLMemSim into a bounded resident cache.
// The hook is intentionally a no-op unless GGML_K3_EXPERT_PAGER=1.
void ggml_k3_expert_pager_prefetch(
        const struct ggml_tensor * experts,
        const struct ggml_tensor * ids);

// The GGUF mapping stays MADV_RANDOM so opening K3 never triggers whole-model
// readahead. Immediately before a dense weight is consumed, temporarily give
// only that tensor sequential/WILLNEED advice. Expert tensors are deliberately
// excluded because their sparse ranges are handled by the router above.
void ggml_k3_dense_tensor_prefetch(const struct ggml_tensor * tensor);

#ifdef __cplusplus
}
#endif
