#include "k3-expert-pager.h"

#include <errno.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__linux__)
#include <dlfcn.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

typedef void * (*legomem_open_fn)(const char *, int);
typedef int (*legomem_io_fn)(void *, uint64_t, void *, size_t);
typedef void (*legomem_close_fn)(void *);

struct pager_region {
    struct ggml_tensor * tensor;
    const void * source;
    void * shadow;
    size_t bytes;
    size_t expert_bytes;
    int64_t expert_count;
    uint64_t cxl_base;
    uint64_t cxl_header;
    char layer[64];
    bool * stored;
    bool * resident;
    uint64_t * stamps;
};

#define K3_PAGER_HEADER_BYTES 4096U
#define K3_PAGER_MAX_PERSISTED_EXPERTS 512U
#define K3_PAGER_MAGIC UINT64_C(0x4b33455850524731)

struct pager_disk_header {
    uint64_t magic;
    uint64_t version;
    uint64_t tensor_hash;
    uint64_t tensor_bytes;
    uint64_t expert_bytes;
    uint64_t expert_count;
    uint8_t stored[K3_PAGER_MAX_PERSISTED_EXPERTS / 8];
};

struct pager_state {
    pthread_mutex_t mutex;
    struct pager_region * regions;
    size_t region_count;
    size_t region_capacity;
    size_t resident_bytes;
    size_t budget_bytes;
    uint64_t next_cxl_address;
    uint64_t cxl_limit;
    uint64_t stamp;
    uint64_t calls;
    uint64_t selected_experts;
    uint64_t requested_bytes;
    uint64_t cache_hit_bytes;
    uint64_t evicted_bytes;
    uint64_t cxl_read_bytes;
    uint64_t cxl_write_bytes;
    uint64_t io_errors;
    uint64_t advise_errors;
    uint64_t stats_every;
    void * library;
    void * client;
    legomem_open_fn client_open;
    legomem_io_fn client_read;
    legomem_io_fn client_write;
    legomem_close_fn client_close;
    int enabled;
    bool cxl_requested;
    bool cxl_ready;
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

static uint64_t tensor_name_hash(const char * name) {
    uint64_t hash = UINT64_C(1469598103934665603);
    for (const unsigned char * current = (const unsigned char *) name; *current; ++current) {
        hash ^= *current;
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static bool pager_tensor_is_k3_expert(const struct ggml_tensor * tensor) {
    if (!tensor || !tensor->data || tensor->ne[2] <= 1) {
        return false;
    }
    return strstr(tensor->name, "ffn_") != NULL &&
           (strstr(tensor->name, "_exps") != NULL || strstr(tensor->name, "experts") != NULL);
}

static bool pager_tensor_is_k3_dense_weight(const struct ggml_tensor * tensor) {
    if (!tensor || !tensor->data || !tensor->name[0] || pager_tensor_is_k3_expert(tensor)) {
        return false;
    }
    if (strstr(tensor->name, ".weight") == NULL) {
        return false;
    }
    return strncmp(tensor->name, "blk.", 4) == 0 ||
           strcmp(tensor->name, "token_embd.weight") == 0 ||
           strcmp(tensor->name, "output.weight") == 0;
}

#if defined(__linux__)
static size_t page_size(void) {
    static size_t value;
    if (value == 0) {
        value = (size_t) sysconf(_SC_PAGESIZE);
    }
    return value;
}

static void page_aligned_range(const void * ptr, size_t size, uintptr_t * begin, size_t * length) {
    const uintptr_t mask = page_size() - 1;
    const uintptr_t first = (uintptr_t) ptr & ~mask;
    const uintptr_t last = ((uintptr_t) ptr + size + mask) & ~mask;
    *begin = first;
    *length = last - first;
}

static void interior_page_range(const void * ptr, size_t size, uintptr_t * begin, size_t * length) {
    const uintptr_t mask = page_size() - 1;
    const uintptr_t first = ((uintptr_t) ptr + mask) & ~mask;
    const uintptr_t last = ((uintptr_t) ptr + size) & ~mask;
    *begin = first;
    *length = last > first ? last - first : 0;
}

static size_t resident_bytes(uintptr_t begin, size_t length) {
    const size_t pages = length / page_size();
    unsigned char * vec = (unsigned char *) malloc(pages ? pages : 1);
    if (!vec) {
        return 0;
    }
    size_t resident = 0;
    if (pages && mincore((void *) begin, length, vec) == 0) {
        for (size_t index = 0; index < pages; ++index) {
            resident += (vec[index] & 1U) ? page_size() : 0;
        }
    }
    free(vec);
    return resident;
}

static bool load_cxl_api_locked(void) {
    const char * path = getenv("GGML_K3_EXPERT_CXL_LIBRARY");
    const char * host = getenv("GGML_K3_EXPERT_CXL_HOST");
    const int port = (int) parse_u64_env("GGML_K3_EXPERT_CXL_PORT", 9999);
    if (!path || !*path) {
        path = "/home/ubuntu/legomem/lib/liblegomem_kv.so";
    }
    if (!host || !*host) {
        host = "127.0.0.1";
    }
    g_pager.library = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!g_pager.library) {
        fprintf(stderr, "K3_EXPERT_PAGER_ERROR dlopen=%s error=%s\n", path, dlerror());
        return false;
    }
    *(void **) (&g_pager.client_open) = dlsym(g_pager.library, "legomem_client_open");
    *(void **) (&g_pager.client_read) = dlsym(g_pager.library, "legomem_client_read");
    *(void **) (&g_pager.client_write) = dlsym(g_pager.library, "legomem_client_write");
    *(void **) (&g_pager.client_close) = dlsym(g_pager.library, "legomem_client_close");
    if (!g_pager.client_open || !g_pager.client_read || !g_pager.client_write || !g_pager.client_close) {
        fprintf(stderr, "K3_EXPERT_PAGER_ERROR incomplete LegoMem client API\n");
        dlclose(g_pager.library);
        g_pager.library = NULL;
        return false;
    }
    g_pager.client = g_pager.client_open(host, port);
    if (!g_pager.client) {
        fprintf(stderr, "K3_EXPERT_PAGER_ERROR connect=%s:%d\n", host, port);
        dlclose(g_pager.library);
        g_pager.library = NULL;
        return false;
    }
    return true;
}
#endif

static void pager_init_locked(void) {
    if (g_pager.enabled != -1) {
        return;
    }
    const char * enabled = getenv("GGML_K3_EXPERT_PAGER");
    const char * backend = getenv("GGML_K3_EXPERT_PAGER_BACKEND");
    g_pager.enabled = enabled && strcmp(enabled, "0") != 0;
    g_pager.cxl_requested = backend && strcmp(backend, "cxl") == 0;
    g_pager.budget_bytes = parse_u64_env("GGML_K3_EXPERT_CACHE_MIB", 64ULL * 1024ULL) * 1024ULL * 1024ULL;
    g_pager.stats_every = parse_u64_env("GGML_K3_EXPERT_STATS_EVERY", 100);
    g_pager.next_cxl_address = parse_u64_env("GGML_K3_EXPERT_CXL_BASE_MIB", 1024) * 1024ULL * 1024ULL;
    g_pager.cxl_limit = parse_u64_env("GGML_K3_EXPERT_CXL_CAPACITY_MIB", 384ULL * 1024ULL) * 1024ULL * 1024ULL;
#if defined(__linux__)
    if (g_pager.enabled && g_pager.cxl_requested) {
        g_pager.cxl_ready = load_cxl_api_locked();
    }
#endif
    if (g_pager.enabled) {
        fprintf(stderr,
                "K3_EXPERT_PAGER_INIT backend=%s cache_mib=%zu stats_every=%" PRIu64
                " cxl_base_mib=%" PRIu64 " cxl_capacity_mib=%" PRIu64 "\n",
                g_pager.cxl_ready ? "cxl-route-aware" : "mmap-route-aware",
                g_pager.budget_bytes / 1024 / 1024, g_pager.stats_every,
                g_pager.next_cxl_address / 1024 / 1024, g_pager.cxl_limit / 1024 / 1024);
    }
}

#if defined(__linux__)
static void extract_layer_name(const char * tensor_name, char * layer, size_t capacity) {
    const char * marker = strstr(tensor_name, "ffn_");
    const size_t length = marker ? (size_t) (marker - tensor_name) : strlen(tensor_name);
    const size_t copy = length < capacity - 1 ? length : capacity - 1;
    memcpy(layer, tensor_name, copy);
    layer[copy] = '\0';
}

static struct pager_region * find_region_locked(const struct ggml_tensor * tensor) {
    for (size_t index = 0; index < g_pager.region_count; ++index) {
        if (g_pager.regions[index].tensor == tensor) {
            return &g_pager.regions[index];
        }
    }
    return NULL;
}

static struct pager_region * register_region_locked(const struct ggml_tensor * tensor) {
    struct pager_region * existing = find_region_locked(tensor);
    if (existing || !g_pager.cxl_ready) {
        return existing;
    }
    if (g_pager.region_count == g_pager.region_capacity) {
        const size_t capacity = g_pager.region_capacity ? g_pager.region_capacity * 2 : 64;
        void * resized = realloc(g_pager.regions, capacity * sizeof(*g_pager.regions));
        if (!resized) {
            return NULL;
        }
        g_pager.regions = (struct pager_region *) resized;
        g_pager.region_capacity = capacity;
    }

    const size_t bytes = ggml_nbytes(tensor);
    const uint64_t aligned_bytes = (bytes + page_size() - 1) & ~(uint64_t) (page_size() - 1);
    const uint64_t allocation_bytes = K3_PAGER_HEADER_BYTES + aligned_bytes;
    if (g_pager.next_cxl_address > g_pager.cxl_limit ||
        allocation_bytes > g_pager.cxl_limit - g_pager.next_cxl_address) {
        fprintf(stderr, "K3_EXPERT_PAGER_ERROR CXL address space exhausted tensor=%s bytes=%zu\n",
                tensor->name, bytes);
        ++g_pager.io_errors;
        return NULL;
    }
    void * shadow = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (shadow == MAP_FAILED) {
        ++g_pager.io_errors;
        return NULL;
    }
    const size_t count = (size_t) tensor->ne[2];
    bool * stored = (bool *) calloc(count, sizeof(bool));
    bool * resident = (bool *) calloc(count, sizeof(bool));
    uint64_t * stamps = (uint64_t *) calloc(count, sizeof(uint64_t));
    if (!stored || !resident || !stamps) {
        free(stored);
        free(resident);
        free(stamps);
        munmap(shadow, bytes);
        return NULL;
    }

    const uint64_t header_address = g_pager.next_cxl_address;
    const uint64_t data_address = header_address + K3_PAGER_HEADER_BYTES;
    uint64_t disk_header_words[K3_PAGER_HEADER_BYTES / sizeof(uint64_t)] = { 0 };
    uint8_t * disk_header_bytes = (uint8_t *) disk_header_words;
    struct pager_disk_header * disk_header = (struct pager_disk_header *) disk_header_words;
    const uint64_t name_hash = tensor_name_hash(tensor->name);
    bool header_valid = g_pager.client_read(
            g_pager.client, header_address, disk_header_bytes, sizeof(disk_header_words)) == 0 &&
        disk_header->magic == K3_PAGER_MAGIC && disk_header->version == 1 &&
        disk_header->tensor_hash == name_hash && disk_header->tensor_bytes == bytes &&
        disk_header->expert_bytes == tensor->nb[2] && disk_header->expert_count == (uint64_t) tensor->ne[2];
    size_t persisted_count = 0;
    if (header_valid && tensor->ne[2] <= K3_PAGER_MAX_PERSISTED_EXPERTS) {
        for (size_t index = 0; index < count; ++index) {
            stored[index] = (disk_header->stored[index / 8] & (uint8_t) (1U << (index % 8))) != 0;
            persisted_count += stored[index] ? 1 : 0;
        }
    } else {
        memset(disk_header_bytes, 0, sizeof(disk_header_words));
        disk_header->magic = K3_PAGER_MAGIC;
        disk_header->version = 1;
        disk_header->tensor_hash = name_hash;
        disk_header->tensor_bytes = bytes;
        disk_header->expert_bytes = tensor->nb[2];
        disk_header->expert_count = (uint64_t) tensor->ne[2];
        if (g_pager.client_write(
                g_pager.client, header_address, disk_header_bytes, sizeof(disk_header_words)) != 0) {
            ++g_pager.io_errors;
        }
    }

    struct pager_region * region = &g_pager.regions[g_pager.region_count++];
    *region = (struct pager_region) {
        .tensor = (struct ggml_tensor *) tensor,
        .source = tensor->data,
        .shadow = shadow,
        .bytes = bytes,
        .expert_bytes = tensor->nb[2],
        .expert_count = tensor->ne[2],
        .cxl_base = data_address,
        .cxl_header = header_address,
        .stored = stored,
        .resident = resident,
        .stamps = stamps,
    };
    extract_layer_name(tensor->name, region->layer, sizeof(region->layer));
    g_pager.next_cxl_address += allocation_bytes;
    region->tensor->data = shadow;
    fprintf(stderr,
            "K3_EXPERT_PAGER_MAP tensor=%s experts=%" PRId64
            " expert_mib=%.3f cxl_base=%" PRIu64 " persisted=%zu\n",
            tensor->name, region->expert_count,
            (double) region->expert_bytes / 1024.0 / 1024.0, region->cxl_base, persisted_count);
    return region;
}

static bool ensure_expert_locked(struct pager_region * region, int64_t expert, uint64_t current_stamp) {
    if (expert < 0 || expert >= region->expert_count) {
        return false;
    }
    const size_t index = (size_t) expert;
    region->stamps[index] = current_stamp;
    if (region->resident[index]) {
        g_pager.cache_hit_bytes += region->expert_bytes;
        return true;
    }

    void * destination = (char *) region->shadow + index * region->expert_bytes;
    void * source = (char *) region->source + index * region->expert_bytes;
    const uint64_t address = region->cxl_base + index * region->expert_bytes;
    if (!region->stored[index]) {
        if (g_pager.client_write(g_pager.client, address, source, region->expert_bytes) != 0) {
            ++g_pager.io_errors;
            return false;
        }
        region->stored[index] = true;
        g_pager.cxl_write_bytes += region->expert_bytes;
        uint8_t stored_byte = 0;
        const size_t byte_index = index / 8;
        const size_t first = byte_index * 8;
        for (size_t bit = 0; bit < 8 && first + bit < (size_t) region->expert_count; ++bit) {
            stored_byte |= region->stored[first + bit] ? (uint8_t) (1U << bit) : 0;
        }
        const uint64_t bitmap_address = region->cxl_header +
            offsetof(struct pager_disk_header, stored) + byte_index;
        if (g_pager.client_write(g_pager.client, bitmap_address, &stored_byte, 1) != 0) {
            ++g_pager.io_errors;
        }
        uintptr_t source_begin = 0;
        size_t source_length = 0;
        interior_page_range(source, region->expert_bytes, &source_begin, &source_length);
        if (source_length && madvise((void *) source_begin, source_length, MADV_DONTNEED) != 0) {
            ++g_pager.advise_errors;
        }
    }
    if (g_pager.client_read(g_pager.client, address, destination, region->expert_bytes) != 0) {
        ++g_pager.io_errors;
        return false;
    }
    region->resident[index] = true;
    g_pager.resident_bytes += region->expert_bytes;
    g_pager.cxl_read_bytes += region->expert_bytes;
    return true;
}

static bool evict_one_locked(uint64_t current_stamp) {
    struct pager_region * victim_region = NULL;
    size_t victim_expert = 0;
    uint64_t victim_stamp = UINT64_MAX;
    for (size_t region_index = 0; region_index < g_pager.region_count; ++region_index) {
        struct pager_region * region = &g_pager.regions[region_index];
        for (size_t expert = 0; expert < (size_t) region->expert_count; ++expert) {
            if (region->resident[expert] && region->stamps[expert] < current_stamp &&
                region->stamps[expert] < victim_stamp) {
                victim_region = region;
                victim_expert = expert;
                victim_stamp = region->stamps[expert];
            }
        }
    }
    if (!victim_region) {
        return false;
    }
    void * ptr = (char *) victim_region->shadow + victim_expert * victim_region->expert_bytes;
    uintptr_t begin = 0;
    size_t length = 0;
    interior_page_range(ptr, victim_region->expert_bytes, &begin, &length);
    if (length && madvise((void *) begin, length, MADV_DONTNEED) != 0) {
        ++g_pager.advise_errors;
    }
    victim_region->resident[victim_expert] = false;
    g_pager.resident_bytes -= victim_region->expert_bytes;
    g_pager.evicted_bytes += victim_region->expert_bytes;
    return true;
}

static void evict_to_budget_locked(uint64_t current_stamp) {
    while (g_pager.resident_bytes > g_pager.budget_bytes && evict_one_locked(current_stamp)) {
    }
}

static void prefetch_mmap_locked(const struct ggml_tensor * experts, const bool * selected) {
    for (int64_t expert = 0; expert < experts->ne[2]; ++expert) {
        if (!selected[expert]) {
            continue;
        }
        const void * ptr = (const char *) experts->data + expert * experts->nb[2];
        uintptr_t begin = 0;
        size_t length = 0;
        page_aligned_range(ptr, experts->nb[2], &begin, &length);
        g_pager.cache_hit_bytes += resident_bytes(begin, length);
        if (madvise((void *) begin, length, MADV_WILLNEED) != 0) {
            ++g_pager.advise_errors;
        }
    }
}
#endif

void ggml_k3_expert_pager_prefetch(
        const struct ggml_tensor * experts,
        const struct ggml_tensor * ids) {
#if !defined(__linux__)
    (void) experts;
    (void) ids;
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
    size_t selected_count = 0;
    for (int64_t token = 0; token < ids->ne[1]; ++token) {
        for (int64_t slot = 0; slot < ids->ne[0]; ++slot) {
            const int32_t expert = *(const int32_t *) ((const char *) ids->data + token * ids->nb[1] + slot * ids->nb[0]);
            if (expert >= 0 && expert < expert_count && !selected[expert]) {
                selected[expert] = true;
                ++selected_count;
            }
        }
    }

    ++g_pager.calls;
    const uint64_t current_stamp = ++g_pager.stamp;
    g_pager.selected_experts += selected_count;
    if (g_pager.cxl_ready) {
        struct pager_region * current = register_region_locked(experts);
        if (current) {
            for (size_t region_index = 0; region_index < g_pager.region_count; ++region_index) {
                struct pager_region * region = &g_pager.regions[region_index];
                if (strcmp(region->layer, current->layer) != 0 || region->expert_count != expert_count) {
                    continue;
                }
                g_pager.requested_bytes += selected_count * region->expert_bytes;
                for (int64_t expert = 0; expert < expert_count; ++expert) {
                    if (selected[expert]) {
                        ensure_expert_locked(region, expert, current_stamp);
                    }
                }
            }
            evict_to_budget_locked(current_stamp);
        }
    } else {
        g_pager.requested_bytes += selected_count * experts->nb[2];
        prefetch_mmap_locked(experts, selected);
    }
    free(selected);

    if (g_pager.stats_every && g_pager.calls % g_pager.stats_every == 0) {
        const double hit_ratio = g_pager.requested_bytes
            ? (double) g_pager.cache_hit_bytes / (double) g_pager.requested_bytes : 0.0;
        fprintf(stderr,
                "K3_EXPERT_PAGER_STATS calls=%" PRIu64 " selected=%" PRIu64
                " requested_mib=%.3f hit_ratio=%.6f resident_mib=%.3f evicted_mib=%.3f"
                " cxl_read_mib=%.3f cxl_write_mib=%.3f io_errors=%" PRIu64
                " advise_errors=%" PRIu64 " tensor=%s\n",
                g_pager.calls, g_pager.selected_experts,
                (double) g_pager.requested_bytes / 1024.0 / 1024.0, hit_ratio,
                (double) g_pager.resident_bytes / 1024.0 / 1024.0,
                (double) g_pager.evicted_bytes / 1024.0 / 1024.0,
                (double) g_pager.cxl_read_bytes / 1024.0 / 1024.0,
                (double) g_pager.cxl_write_bytes / 1024.0 / 1024.0,
                g_pager.io_errors, g_pager.advise_errors, experts->name);
    }
    pthread_mutex_unlock(&g_pager.mutex);
#endif
}

void ggml_k3_dense_tensor_prefetch(const struct ggml_tensor * tensor) {
#if !defined(__linux__)
    (void) tensor;
#else
    if (!pager_tensor_is_k3_dense_weight(tensor)) {
        return;
    }

    pthread_mutex_lock(&g_pager.mutex);
    pager_init_locked();
    const bool enabled = g_pager.enabled;
    pthread_mutex_unlock(&g_pager.mutex);
    if (!enabled) {
        return;
    }

    uintptr_t begin = 0;
    size_t length = 0;
    page_aligned_range(tensor->data, ggml_nbytes(tensor), &begin, &length);
    if (!length) {
        return;
    }

    // MADV_SEQUENTIAL restores clustered fault-around/readahead only for the
    // active dense tensor. WILLNEED starts I/O while worker setup is finishing.
    int errors = 0;
    errors += madvise((void *) begin, length, MADV_SEQUENTIAL) != 0;
    errors += madvise((void *) begin, length, MADV_WILLNEED) != 0;
    if (errors) {
        pthread_mutex_lock(&g_pager.mutex);
        g_pager.advise_errors += (uint64_t) errors;
        pthread_mutex_unlock(&g_pager.mutex);
    }
#endif
}
