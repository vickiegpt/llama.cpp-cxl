#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 MODEL_FIRST_SHARD OUTPUT_DIR [TOKENS]" >&2
    exit 2
fi

model=$1
output_dir=$2
tokens=${3:-4}
binary=${LLAMA_CLI:-./build/bin/llama-cli}
cache_mib=${K3_EXPERT_CACHE_MIB:-65536}
mkdir -p "$output_dir"

common=(
    -m "$model"
    -p "Reply with one word: hello"
    -n "$tokens"
    -c 128
    -t "${K3_THREADS:-16}"
    -tb "${K3_BATCH_THREADS:-16}"
    --no-warmup
    --no-display-prompt
    --no-conversation
)

if [[ -n ${K3_RPC_ARGS:-} ]]; then
    # Intentional word splitting: this variable contains complete llama CLI arguments.
    read -r -a rpc_args <<< "$K3_RPC_ARGS"
    common+=("${rpc_args[@]}")
fi

run_case() {
    local name=$1
    shift
    echo "K3_BENCH_START case=$name utc=$(date -u +%FT%TZ)"
    /usr/bin/time -v env "$@" "$binary" "${common[@]}" \
        >"$output_dir/$name.log" 2>"$output_dir/$name.time.log"
    grep -E "(prompt eval time|eval time|K3_EXPERT_PAGER_STATS|K3_IQ1S_AMX_STATS)" \
        "$output_dir/$name.log" "$output_dir/$name.time.log" || true
    echo "K3_BENCH_END case=$name utc=$(date -u +%FT%TZ)"
}

run_case no_legomem_cpu \
    GGML_K3_EXPERT_PAGER=0 \
    GGML_K3_IQ1S_AMX=0

run_case no_legomem_amx \
    GGML_K3_EXPERT_PAGER=0 \
    GGML_K3_IQ1S_AMX=1 \
    GGML_K3_IQ1S_AMX_STATS_EVERY=1

run_case legomem_cxl_amx \
    GGML_K3_EXPERT_PAGER=1 \
    GGML_K3_EXPERT_PAGER_BACKEND=cxl \
    GGML_K3_EXPERT_CACHE_MIB="$cache_mib" \
    GGML_K3_EXPERT_STATS_EVERY=1 \
    GGML_K3_EXPERT_CXL_LIBRARY="${K3_CXL_LIBRARY:-/home/ubuntu/legomem/lib/liblegomem_kv.so}" \
    GGML_K3_EXPERT_CXL_HOST="${K3_CXL_HOST:-127.0.0.1}" \
    GGML_K3_EXPERT_CXL_PORT="${K3_CXL_PORT:-9999}" \
    GGML_K3_EXPERT_CXL_BASE_MIB="${K3_CXL_BASE_MIB:-1024}" \
    GGML_K3_EXPERT_CXL_CAPACITY_MIB="${K3_CXL_CAPACITY_MIB:-393216}" \
    GGML_K3_IQ1S_AMX=1 \
    GGML_K3_IQ1S_AMX_STATS_EVERY=1
