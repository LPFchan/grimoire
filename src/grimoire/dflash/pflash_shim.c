// pflash_shim.c — LD_PRELOAD library that parks/unparks llama-server GPU allocations
//
// Intercepts cuMemAlloc_v2 / cuMemFree_v2 to track all GPU allocations.
// On "park": saves all live allocations to a host-side malloc buffer,
// unmaps GPU physical pages via CUDA VMM (preserving virtual addresses).
// On "unpark": remaps GPU physical pages at the same VAs, restores data.
//
// No allocation-size heuristics — parks EVERYTHING (weights, KV, scratch).
// VMM coalescing groups adjacent small allocations into 2 MB-aligned regions.
// Thread safety via mutex around all intercepted calls.
//
// Protocol: named pipe /tmp/pflash_shim.ctl (write), .ack (read)
//   "park" → "ok" or "error:..."
//   "unpark" → "ok" or "error:..."

#define _GNU_SOURCE
#include <cuda.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// ── Constants ─────────────────────────────────────────────────────────
#define MAX_ALLOCS      16384
#define ALIGN_2MB       0x200000ULL
#define MAX_CTL_LINE    256
#define VMM_GRANULARITY 0x200000ULL  // 2 MB — minimum for cuMemMap
#define PID_BUF_SIZE    64

// ── Registry entry ─────────────────────────────────────────────────────
typedef struct {
    CUdeviceptr ptr;
    size_t      size;
    int         active;  // 1 = live, 0 = freed
} AllocEntry;

static AllocEntry  g_allocs[MAX_ALLOCS];
static int         g_n_allocs;
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;

// ── Host shadow buffer ────────────────────────────────────────────────
// Plain malloc (not pinned — avoids RLIMIT_MEMLOCK issues)
static unsigned char *g_shadow    = NULL;
static size_t         g_shadow_sz = 0;
static size_t         g_shadow_used = 0;

// ── CUDA VMM state ────────────────────────────────────────────────────
static CUdevice     g_dev         = 0;
static CUcontext    g_ctx         = NULL;
static int          g_vmm_inited  = 0;
static CUdeviceptr  g_va_base     = 0;
static size_t       g_va_capacity = 0;  // total reserved VA space in bytes
static size_t       g_va_used     = 0;  // allocated within VA range

// ── Park state ────────────────────────────────────────────────────────
static volatile int g_parked = 0;
static int          g_ctl_fd = -1;
static int          g_ack_fd = -1;

// ── Thread for FIFO listener ──────────────────────────────────────────
static pthread_t    g_listener;
static volatile int g_listener_running = 0;

// ── DSO constructor/destructor ────────────────────────────────────────
static void shim_init(void)  __attribute__((constructor));
static void shim_fini(void)  __attribute__((destructor));

// ── Forward declarations ──────────────────────────────────────────────
static void vmm_init(void);
static int  vmm_park(void);
static int  vmm_unpark(void);
static void registry_add(CUdeviceptr ptr, size_t size);
static void registry_remove(CUdeviceptr ptr);
static void ctl_listener_loop(void);

// ── dlsym helpers ─────────────────────────────────────────────────────
static void *resolve(const char *name) {
    static void *handle = NULL;
    if (!handle) handle = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_NOLOAD);
    return handle ? dlsym(handle, name) : NULL;
}

typedef CUresult (*cuMemAlloc_v2_fn)(CUdeviceptr *, size_t);
typedef CUresult (*cuMemFree_v2_fn)(CUdeviceptr);
typedef CUresult (*cuMemGetInfo_v2_fn)(size_t *, size_t *);
typedef CUresult (*cuCtxSynchronize_fn)(void);

static cuMemAlloc_v2_fn    real_cuMemAlloc_v2    = NULL;
static cuMemFree_v2_fn     real_cuMemFree_v2     = NULL;
static cuMemGetInfo_v2_fn  real_cuMemGetInfo_v2  = NULL;
static cuCtxSynchronize_fn real_cuCtxSynchronize = NULL;

// ── Intercepted CUDA Driver API calls ─────────────────────────────────
CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize) {
    if (!real_cuMemAlloc_v2) real_cuMemAlloc_v2 = resolve("cuMemAlloc_v2");
    CUresult r = real_cuMemAlloc_v2(dptr, bytesize);
    if (r == CUDA_SUCCESS) {
        pthread_mutex_lock(&g_lock);
        registry_add(*dptr, bytesize);
        if (!g_vmm_inited) vmm_init();
        pthread_mutex_unlock(&g_lock);
    }
    return r;
}

CUresult cuMemFree_v2(CUdeviceptr dptr) {
    if (!real_cuMemFree_v2) real_cuMemFree_v2 = resolve("cuMemFree_v2");
    pthread_mutex_lock(&g_lock);
    registry_remove(dptr);
    pthread_mutex_unlock(&g_lock);
    return real_cuMemFree_v2(dptr);
}

CUresult cuMemGetInfo_v2(size_t *free, size_t *total) {
    if (!real_cuMemGetInfo_v2) real_cuMemGetInfo_v2 = resolve("cuMemGetInfo_v2");
    return real_cuMemGetInfo_v2(free, total);
}

// ── Registry ──────────────────────────────────────────────────────────
static int registry_find(CUdeviceptr ptr) {
    for (int i = 0; i < g_n_allocs; i++)
        if (g_allocs[i].active && g_allocs[i].ptr == ptr) return i;
    return -1;
}

static void registry_add(CUdeviceptr ptr, size_t size) {
    int idx = registry_find(ptr);
    if (idx >= 0) { g_allocs[idx].size = size; return; }
    if (g_n_allocs >= MAX_ALLOCS) { fprintf(stderr, "[pflash_shim] registry full\n"); return; }
    g_allocs[g_n_allocs].ptr    = ptr;
    g_allocs[g_n_allocs].size   = size;
    g_allocs[g_n_allocs].active = 1;
    g_n_allocs++;
    g_va_used += (size + VMM_GRANULARITY - 1) & ~(VMM_GRANULARITY - 1);
}

static void registry_remove(CUdeviceptr ptr) {
    int idx = registry_find(ptr);
    if (idx >= 0) { g_allocs[idx].active = 0; }
}

// ── VMM Init ──────────────────────────────────────────────────────────
static void vmm_init(void) {
    // Get current CUDA context
    cuCtxGetCurrent(&g_ctx);
    if (!g_ctx) return;
    cuCtxGetDevice(&g_dev);

    // Query free memory to estimate VA space needed
    size_t free_mem = 0, total_mem = 0;
    cuMemGetInfo_v2(&free_mem, &total_mem);
    // Reserve 90% of total VRAM as VA space
    g_va_capacity = (total_mem * 9 / 10) & ~(VMM_GRANULARITY - 1);
    if (g_va_capacity < 2ULL * 1024 * 1024 * 1024)
        g_va_capacity = 2ULL * 1024 * 1024 * 1024;  // at least 2 GB
    if (g_va_capacity > 22ULL * 1024 * 1024 * 1024)
        g_va_capacity = 22ULL * 1024 * 1024 * 1024;  // at most 22 GB

    CUresult r = cuMemAddressReserve(&g_va_base, g_va_capacity, 0, 0, 0);
    if (r != CUDA_SUCCESS) {
        fprintf(stderr, "[pflash_shim] cuMemAddressReserve failed: %d\n", r);
        g_va_capacity = 0;
        return;
    }
    g_vmm_inited = 1;
    fprintf(stderr, "[pflash_shim] VMM inited: VA=0x%lx capacity=%zu MB\n",
            (unsigned long)g_va_base, g_va_capacity >> 20);
}

// ── Park: save all allocations to host shadow, unmap GPU pages ───────
static int vmm_park(void) {
    if (!g_vmm_inited || !g_va_capacity) return -1;

    // Synchronize — no in-flight kernels during unmap
    if (!real_cuCtxSynchronize) real_cuCtxSynchronize = resolve("cuCtxSynchronize");
    if (real_cuCtxSynchronize) real_cuCtxSynchronize();

    // Calculate total shadow size
    size_t total = 0;
    pthread_mutex_lock(&g_lock);
    for (int i = 0; i < g_n_allocs; i++) {
        if (g_allocs[i].active) total += g_allocs[i].size;
    }

    // Allocate host shadow buffer
    unsigned char *buf = (unsigned char *)malloc(total);
    if (!buf) { pthread_mutex_unlock(&g_lock); return -1; }
    g_shadow = buf;
    g_shadow_sz = total;
    g_shadow_used = 0;

    // Save each allocation to shadow, then unmap
    for (int i = 0; i < g_n_allocs; i++) {
        if (!g_allocs[i].active) continue;
        CUdeviceptr ptr = g_allocs[i].ptr;
        size_t sz = g_allocs[i].size;
        size_t aligned = (sz + VMM_GRANULARITY - 1) & ~(VMM_GRANULARITY - 1);

        // Copy GPU → host
        cuMemcpyDtoH(buf + g_shadow_used, ptr, sz);
        g_shadow_used += sz;

        // Unmap VMM region (if within our VA range)
        if (ptr >= g_va_base && ptr < g_va_base + g_va_capacity) {
            cuMemUnmap(ptr, aligned);
        }
    }
    pthread_mutex_unlock(&g_lock);

    g_parked = 1;
    fprintf(stderr, "[pflash_shim] parked: %d allocations, %zu MB shadow\n",
            g_n_allocs, total >> 20);
    return 0;
}

// ── Unpark: remap GPU pages, restore data from host shadow ───────────
static int vmm_unpark(void) {
    if (!g_vmm_inited || !g_va_capacity || !g_shadow) return -1;

    pthread_mutex_lock(&g_lock);
    size_t offset = 0;
    for (int i = 0; i < g_n_allocs; i++) {
        if (!g_allocs[i].active) continue;
        CUdeviceptr ptr = g_allocs[i].ptr;
        size_t sz = g_allocs[i].size;
        size_t aligned = (sz + VMM_GRANULARITY - 1) & ~(VMM_GRANULARITY - 1);

        if (ptr >= g_va_base && ptr < g_va_base + g_va_capacity) {
            CUmemAllocationProp prop = {};
            prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id = g_dev;

            CUmemGenericAllocationHandle handle;
            CUresult r = cuMemCreate(&handle, aligned, &prop, 0);
            if (r != CUDA_SUCCESS) {
                fprintf(stderr, "[pflash_shim] cuMemCreate failed at offset %zu: %d\n", offset, r);
                pthread_mutex_unlock(&g_lock);
                return -1;
            }

            r = cuMemMap(ptr, aligned, 0, handle, 0);
            cuMemRelease(handle);
            if (r != CUDA_SUCCESS) {
                fprintf(stderr, "[pflash_shim] cuMemMap failed at offset %zu: %d\n", offset, r);
                pthread_mutex_unlock(&g_lock);
                return -1;
            }

            CUmemAccessDesc access = {};
            access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            access.location.id = g_dev;
            access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            cuMemSetAccess(ptr, aligned, &access, 1);
        }

        // Copy host → GPU
        cuMemcpyHtoD(ptr, g_shadow + offset, sz);
        offset += sz;
    }
    pthread_mutex_unlock(&g_lock);

    free(g_shadow); g_shadow = NULL; g_shadow_sz = 0;
    g_parked = 0;
    fprintf(stderr, "[pflash_shim] unparked\n");
    return 0;
}

// ── FIFO listener ─────────────────────────────────────────────────────
static void ctl_listener_loop(void) {
    char path[PID_BUF_SIZE];
    snprintf(path, sizeof(path), "/tmp/pflash_shim.ctl");

    // Create FIFO
    unlink(path);
    if (mkfifo(path, 0666) < 0) { perror("mkfifo .ctl"); return; }
    g_ctl_fd = open(path, O_RDONLY);
    if (g_ctl_fd < 0) { perror("open .ctl"); return; }

    char ack_path[PID_BUF_SIZE];
    snprintf(ack_path, sizeof(ack_path), "/tmp/pflash_shim.ack");
    unlink(ack_path);
    if (mkfifo(ack_path, 0666) < 0) { perror("mkfifo .ack"); return; }
    // Open write end (non-blocking to avoid deadlock)
    g_ack_fd = open(ack_path, O_WRONLY | O_NONBLOCK);
    if (g_ack_fd < 0) { perror("open .ack"); return; }

    fprintf(stderr, "[pflash_shim] listener ready on %s\n", path);
    fflush(stderr);

    char line[MAX_CTL_LINE];
    while (g_listener_running) {
        int n = read(g_ctl_fd, line, sizeof(line) - 1);
        if (n <= 0) { usleep(10000); continue; }
        line[n] = '\0';
        // Trim whitespace
        while (n > 0 && (line[n-1] == '\n' || line[n-1] == ' ')) line[--n] = '\0';

        const char *response = "error: unknown command\n";
        if (strcmp(line, "park") == 0) {
            response = (vmm_park() == 0) ? "ok\n" : "error: park failed\n";
        } else if (strcmp(line, "unpark") == 0) {
            response = (vmm_unpark() == 0) ? "ok\n" : "error: unpark failed\n";
        } else if (strcmp(line, "quit") == 0) {
            break;
        }
        write(g_ack_fd, response, strlen(response));
        fsync(g_ack_fd);
    }

    close(g_ctl_fd); g_ctl_fd = -1;
    close(g_ack_fd); g_ack_fd = -1;
    unlink(path);
}

// ── Constructor / Destructor ──────────────────────────────────────────
static void shim_init(void) {
    fprintf(stderr, "[pflash_shim] loaded\n");

    // Start listener thread
    g_listener_running = 1;
    pthread_create(&g_listener, NULL, (void *(*)(void *))ctl_listener_loop, NULL);
    pthread_detach(g_listener);
}

static void shim_fini(void) {
    g_listener_running = 0;
    if (g_ctl_fd >= 0) { close(g_ctl_fd); g_ctl_fd = -1; }
    if (g_ack_fd >= 0) { close(g_ack_fd); g_ack_fd = -1; }
    unlink("/tmp/pflash_shim.ctl");
    unlink("/tmp/pflash_shim.ack");
}
