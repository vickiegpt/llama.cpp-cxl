#include "ggml.h"
#include "k3-expert-pager.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

int main() {
    ggml_init_params params = {
        /*.mem_size   =*/ 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    ggml_context * context = ggml_init(params);
    if (!context) {
        return 1;
    }

    ggml_tensor * experts = ggml_new_tensor_3d(context, GGML_TYPE_IQ1_S, 256, 1, 4);
    ggml_set_name(experts, "blk.0.ffn_up_exps.weight");
    ggml_tensor * ids = ggml_new_tensor_2d(context, GGML_TYPE_I32, 2, 1);
    static_cast<int32_t *>(ids->data)[0] = 1;
    static_cast<int32_t *>(ids->data)[1] = 3;

    const size_t bytes = ggml_nbytes(experts);
    const size_t expert_bytes = experts->nb[2];
    for (size_t index = 0; index < bytes; ++index) {
        static_cast<uint8_t *>(experts->data)[index] = static_cast<uint8_t>(index * 131U + 17U);
    }
    std::vector<uint8_t> expected(static_cast<uint8_t *>(experts->data),
                                  static_cast<uint8_t *>(experts->data) + bytes);
    void * original = experts->data;

    ggml_k3_expert_pager_prefetch(experts, ids);
    bool passed = experts->data != original;
    for (int selected : {1, 3}) {
        passed = passed && std::memcmp(
            static_cast<uint8_t *>(experts->data) + selected * expert_bytes,
            expected.data() + selected * expert_bytes,
            expert_bytes) == 0;
    }
    for (int unselected : {0, 2}) {
        std::vector<uint8_t> zero(expert_bytes);
        passed = passed && std::memcmp(
            static_cast<uint8_t *>(experts->data) + unselected * expert_bytes,
            zero.data(), expert_bytes) == 0;
    }
    ggml_k3_expert_pager_prefetch(experts, ids);
    std::printf("K3_EXPERT_PAGER_CXL_ROUNDTRIP=%s bytes=%zu expert_bytes=%zu\n",
                passed ? "PASS" : "FAIL", bytes, expert_bytes);
    ggml_free(context);
    return passed ? 0 : 1;
}
