#include "k3-expert-pager.h"

#include <errno.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

struct pager_entry {
    uintptr_t begin;
    size_t length;
    uint64_t stamp;
};

struct pager_state {
    pthread_mutex_t mutex;
    struct pager_entry * entries;
    size_t count;
    size_t capacity;
    size_t tracked_bytes;
    size_t budget_bytes;
    uint64_t stamp;
    uint64_t calls;
    uint64_t selected_experts;
    uint64_t requested_bytes;
    uint64_t resident_bytes;
    uint64_t evicted_bytes;
    uint64_t advise_errors;
    int enabled;
    uint64_t stats_every;
};

static struct pager_state g_pager = {
    .mutex = PTHREAD_MUTEX_INITIALIZER,
    .enabled = -1,
};

static uint64_t parse_u64_env(const char * name, uint64_t fallback) {
    const char * value = getenv(name);
    if (!value || !*value) {
        return fallback;
    }
    char * end = NULL;
    errno = 0;
    const unsigned long long parsed = strtoull(value, &end, 10);
    return errno == 0 && end && *end == '\0' ? (uint64_t) parsed : fallback;
}

static bool pager_tensor_is_k3_expert(const struct ggml_tensor * tensor) {
    if (!tensor || !tensor->data || tensor->ne[2] <= 1) {
        return false;
    }
    return strstr(tensor->name, "ffn_") != NULL &&
           (strstr(tensor->name, "_exps") != NULL || strstr(tensor->name, "experts") != NULL);
}

#if defined(__linux__)
static void page_aligned_range(const void * ptr, size_t size, uintptr_t * begin, size_t * length) {
    static size_t page_size;
    if (page_size == 0) {
        page_size = (size_t) sysconf(_SC_PAGESIZE);
    }
    const uintptr_t mask = page_size - 1;
    const uintptr_t first = (uintptr_t) ptr & ~mask;
    const uintptr_t last = ((uintptr_t) ptr + size + mask) & ~mask;
    *begin = first;
    *length = last - first;
}

static size_t resident_bytes(uintptr_t begin, size_t length) {
    static size_t page_size;
    if (page_size == 0) {
        page_size = (size_t) sysconf(_SC_PAGESIZE);
    }
    const size_t pages = length / page_size;
    unsigned char * vec = (unsigned char *) malloc(pages);
    if (!vec) {
        return 0;
    }
    size_t resident = 0;
    if (mincore((void *) begin, length, vec) == 0) {
        for (size_t index = 0; index < pages; ++index) {
            resident += (vec[index] & 1U) ? page_size : 0;
        }
    }
    free(vec);
    return resident;
}
#endif

static void pager_init_locked(void) {
    if (g_pager.enabled != -1) {
        return;
    }
    const char * enabled = getenv("GGML_K3_EXPERT_PAGER");
    g_pager.enabled = enabled && strcmp(enabled, "0") != 0;
    g_pager.budget_bytes = parse_u64_env("GGML_K3_EXPERT_CACHE_MIB", 64ULL * 1024ULL) * 1024ULL * 1024ULL;
    g_pager.stats_every = parse_u64_env("GGML_K3_EXPERT_STATS_EVERY", 100);
    if (g_pager.enabled) {
        fprintf(stderr,
                "K3_EXPERT_PAGER_INIT backend=mmap-route-aware cache_mib=%zu stats_every=%" PRIu64 "\n",
                g_pager.budget_bytes / 1024 / 1024, g_pager.stats_every);
    }
}

static struct pager_entry * find_entry_locked(uintptr_t begin, size_t length) {
    for (size_t index = 0; index < g_pager.count; ++index) {
        if (g_pager.entries[index].begin == begin && g_pager.entries[index].length == length) {
            return &g_pager.entries[index];
        }
    }
    return NULL;
}

static void track_entry_locked(uintptr_t begin, size_t length) {
    struct pager_entry * entry = find_entry_locked(begin, length);
    if (entry) {
        entry->stamp = ++g_pager.stamp;
        return;
    }
    if (g_pager.count == g_pager.capacity) {
        const size_t new_capacity = g_pager.capacity ? g_pager.capacity * 2 : 256;
        void * resized = realloc(g_pager.entries, new_capacity * sizeof(*g_pager.entries));
        if (!resized) {
            return;
        }
        g_pager.entries = (struct pager_entry *) resized;
        g_pager.capacity = new_capacity;
    }
    g_pager.entries[g_pager.count++] = (struct pager_entry) {
        .begin = begin,
        .length = length,
        .stamp = ++g_pager.stamp,
    };
    g_pager.tracked_bytes += length;
}

static void evict_to_budget_locked(void) {
#if defined(__linux__)
    while (g_pager.tracked_bytes > g_pager.budget_bytes && g_pager.count > 0) {
        size_t victim = 0;
        for (size_t index = 1; index < g_pager.count; ++index) {
            if (g_pager.entries[index].stamp < g_pager.entries[victim].stamp) {
                victim = index;
            }
        }
        const struct pager_entry entry = g_pager.entries[victim];
        if (madvise((void *) entry.begin, entry.length, MADV_DONTNEED) != 0) {
            ++g_pager.advise_errors;
        } else {
            g_pager.evicted_bytes += entry.length;
        }
        g_pager.tracked_bytes -= entry.length;
        g_pager.entries[victim] = g_pager.entries[g_pager.count - 1];
        --g_pager.count;
    }
#endif
}

void ggml_k3_expert_pager_prefetch(
        const struct ggml_tensor * experts,
        const struct ggml_tensor * ids) {
#if !defined(__linux__)
    (void) experts;
    (void) ids;
    return;
#else
    if (!pager_tensor_is_k3_expert(experts) || !ids || !ids->data || ids->type != GGML_TYPE_I32) {
        return;
    }

    pthread_mutex_lock(&g_pager.mutex);
    pager_init_locked();
    if (!g_pager.enabled) {
        pthread_mutex_unlock(&g_pager.mutex);
        return;
    }

    const int64_t expert_count = experts->ne[2];
    bool * selected = (bool *) calloc((size_t) expert_count, sizeof(bool));
    if (!selected) {
        pthread_mutex_unlock(&g_pager.mutex);
        return;
    }

    for (int64_t token = 0; token < ids->ne[1]; ++token) {
        for (int64_t slot = 0; slot < ids->ne[0]; ++slot) {
            const int32_t expert = *(const int32_t *) ((const char *) ids->data + token * ids->nb[1] + slot * ids->nb[0]);
            if (expert >= 0 && expert < expert_count) {
                selected[expert] = true;
            }
        }
    }

    size_t selected_count = 0;
    size_t selected_bytes = 0;
    size_t selected_resident = 0;
    for (int64_t expert = 0; expert < expert_count; ++expert) {
        if (!selected[expert]) {
            continue;
        }
        const void * ptr = (const char *) experts->data + expert * experts->nb[2];
        uintptr_t begin = 0;
        size_t length = 0;
        page_aligned_range(ptr, experts->nb[2], &begin, &length);
        selected_resident += resident_bytes(begin, length);
        selected_bytes += length;
        ++selected_count;
        track_entry_locked(begin, length);
        if (madvise((void *) begin, length, MADV_WILLNEED) != 0) {
            ++g_pager.advise_errors;
        }
    }
    free(selected);

    ++g_pager.calls;
    g_pager.selected_experts += selected_count;
    g_pager.requested_bytes += selected_bytes;
    g_pager.resident_bytes += selected_resident;
    evict_to_budget_locked();

    if (g_pager.stats_every && g_pager.calls % g_pager.stats_every == 0) {
        const double hit_ratio = g_pager.requested_bytes
            ? (double) g_pager.resident_bytes / (double) g_pager.requested_bytes
            : 0.0;
        fprintf(stderr,
                "K3_EXPERT_PAGER_STATS calls=%" PRIu64 " selected=%" PRIu64
                " requested_mib=%.3f resident_hit_ratio=%.6f tracked_mib=%.3f"
                " evicted_mib=%.3f advise_errors=%" PRIu64 " tensor=%s\n",
                g_pager.calls, g_pager.selected_experts,
                (double) g_pager.requested_bytes / 1024.0 / 1024.0,
                hit_ratio,
                (double) g_pager.tracked_bytes / 1024.0 / 1024.0,
                (double) g_pager.evicted_bytes / 1024.0 / 1024.0,
                g_pager.advise_errors, experts->name);
    }
    pthread_mutex_unlock(&g_pager.mutex);
#endif
}
