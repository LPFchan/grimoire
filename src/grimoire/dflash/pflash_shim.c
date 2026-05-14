#define _GNU_SOURCE
#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define MAX_A 16384
#define VMM_G 0x200000ULL
#define LOG(fmt, ...) do { fprintf(stderr,"[pflash_shim] "fmt"\n",##__VA_ARGS__); fflush(stderr); } while(0)

typedef struct { void *p; size_t sz; } E;
static E es[MAX_A];
static int ne;
static pthread_mutex_t lk = PTHREAD_MUTEX_INITIALIZER;

static CUdevice dv;
static CUcontext cx;
static int vmm_ok;
static CUdeviceptr va_base, va_next;
static size_t va_cap;
static unsigned char *sh;
static size_t sh_sz;
static int cf=-1, af=-1;
static pthread_t th;
static volatile int thr;

__attribute__((constructor)) static void shim_init(void);
__attribute__((destructor))  static void shim_fini(void);

static void *sym(const char *n) {
    void *h;
    if ((h = dlopen("libcuda.so.1",RTLD_LAZY|RTLD_NOLOAD))) { void *p = dlsym(h,n); if(p) return p; }
    if ((h = dlopen("libcudart.so",RTLD_LAZY|RTLD_NOLOAD))) { void *p = dlsym(h,n); if(p) return p; }
    if ((h = dlopen("libcudart.so.12",RTLD_LAZY|RTLD_NOLOAD))) { void *p = dlsym(h,n); if(p) return p; }
    return NULL;
}

static void vmm_init(void) {
    cuCtxGetCurrent(&cx);
    if (!cx) { LOG("VMM: no context yet"); return; }
    cuCtxGetDevice(&dv);
    size_t fr, tt; cuMemGetInfo_v2(&fr, &tt);
    va_cap = (tt * 9 / 10) & ~(VMM_G - 1);
    if (va_cap < 2ULL<<30) va_cap = 2ULL<<30;
    if (va_cap > 22ULL<<30) va_cap = 22ULL<<30;
    CUresult r = cuMemAddressReserve(&va_base, va_cap, 0, 0, 0);
    LOG("VMM: reserve %zu MB -> %d va=0x%lx", va_cap>>20, r, (unsigned long)va_base);
    if (r == CUDA_SUCCESS) { va_next = va_base; vmm_ok = 1; }
    else { va_cap = 0; }
}

static CUdeviceptr vmm_alloc(size_t sz) {
    // Chunk large allocations into 128 MB pieces to respect per-alloc limits
    const size_t CHUNK = 128ULL * 1024 * 1024;  // 128 MB per VMM create
    size_t al = (sz + VMM_G - 1) & ~(VMM_G - 1);
    if (va_next + al > va_base + va_cap) { LOG("VMM OOM: need %zu, cap %zu", al, va_cap); return 0; }

    size_t remaining = al;
    CUdeviceptr current = va_next;

    while (remaining > 0) {
        size_t chunk = (remaining < CHUNK) ? remaining : CHUNK;
        chunk = (chunk + VMM_G - 1) & ~(VMM_G - 1);  // align to VMM granularity

        CUmemAllocationProp pr = {};
        pr.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        pr.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        pr.location.id = dv;

        CUmemGenericAllocationHandle h;
        CUresult r = cuMemCreate(&h, chunk, &pr, 0);
        if (r != CUDA_SUCCESS) { LOG("cuMemCreate(%zu) fail %d at offset %zu", chunk, r, al - remaining); return 0; }

        r = cuMemMap(current, chunk, 0, h, 0);
        cuMemRelease(h);
        if (r != CUDA_SUCCESS) { LOG("cuMemMap(%zu) fail %d", chunk, r); return 0; }

        CUmemAccessDesc ac = {};
        ac.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        ac.location.id = dv;
        ac.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        cuMemSetAccess(current, chunk, &ac, 1);

        current += chunk;
        remaining -= chunk;
    }

    CUdeviceptr ret = va_next;
    va_next = current;
    return ret;
}

static void vmm_free(CUdeviceptr p, size_t sz) {
    // Chunked unmap (though cuMemUnmap handles large ranges, chunked for consistency)
    const size_t CHUNK = 128ULL * 1024 * 1024;
    size_t remaining = (sz + VMM_G - 1) & ~(VMM_G - 1);
    CUdeviceptr current = p;
    while (remaining > 0) {
        size_t chunk = (remaining < CHUNK) ? remaining : CHUNK;
        chunk = (chunk + VMM_G - 1) & ~(VMM_G - 1);
        cuMemUnmap(current, chunk);
        current += chunk;
        remaining -= chunk;
    }
}

cudaError_t cudaMalloc(void **p, size_t sz) {
    static cudaError_t (*real)(void**,size_t);
    if (!real) real = sym("cudaMalloc");
    if (!vmm_ok) vmm_init();
    if (!vmm_ok) return real(p, sz);
    CUdeviceptr va = vmm_alloc(sz);
    if (!va) return real(p, sz);
    *p = (void*)va;
    pthread_mutex_lock(&lk);
    if (ne < MAX_A) { es[ne].p = *p; es[ne].sz = sz; ne++; }
    LOG("cudaMalloc(%p,%zu) VMM tot=%d", *p, sz, ne);
    pthread_mutex_unlock(&lk);
    return cudaSuccess;
}

cudaError_t cudaFree(void *p) {
    static cudaError_t (*real)(void*);
    if (!real) real = sym("cudaFree");
    if (p) {
        pthread_mutex_lock(&lk);
        size_t sz = 0;
        for (int i = 0; i < ne; i++) {
            if (es[i].p == p) { sz = es[i].sz; es[i] = es[--ne]; break; }
        }
        pthread_mutex_unlock(&lk);
        if (vmm_ok && sz) { vmm_free((CUdeviceptr)(uintptr_t)p, sz); return cudaSuccess; }
    }
    return real(p);
}

static int park(void) {
    LOG("park vmm=%d ne=%d", vmm_ok, ne);
    if (!vmm_ok) return -1;
    cuCtxSetCurrent(cx); cuCtxSynchronize();
    pthread_mutex_lock(&lk);
    size_t t = 0;
    for (int i = 0; i < ne; i++) t += es[i].sz;
    unsigned char *b = malloc(t);
    if (!b) { pthread_mutex_unlock(&lk); return -1; }
    sh = b; sh_sz = t; size_t off = 0;
    for (int i = 0; i < ne; i++) {
        CUdeviceptr dp = (CUdeviceptr)(uintptr_t)es[i].p;
        size_t s = es[i].sz, al = (s + VMM_G - 1) & ~(VMM_G - 1);
        cuMemcpyDtoH(b + off, dp, s); off += s;
        cuMemUnmap(dp, al);
        LOG("  unmap[%d]: %p -> host (%zu MB)", i, es[i].p, s>>20);
    }
    pthread_mutex_unlock(&lk);
    LOG("park done: %.0f MB", (double)t/(1<<20));
    return 0;
}

static int unpark(void) {
    LOG("unpark vmm=%d sh=%p", vmm_ok, (void*)sh);
    if (!vmm_ok || !sh) return -1;
    cuCtxSetCurrent(cx);
    pthread_mutex_lock(&lk);
    size_t off = 0; int n = 0;

    const size_t CHUNK = 128ULL * 1024 * 1024;

    for (int i = 0; i < ne; i++) {
        CUdeviceptr dp = (CUdeviceptr)(uintptr_t)es[i].p;
        size_t s = es[i].sz, al = (s + VMM_G - 1) & ~(VMM_G - 1);

        // Chunked VMM re-creation at the same VA (supports large 15 GB allocations)
        size_t remaining = al;
        CUdeviceptr current = dp;
        int ok = 1;
        while (remaining > 0) {
            size_t chunk = (remaining < CHUNK) ? remaining : CHUNK;
            chunk = (chunk + VMM_G - 1) & ~(VMM_G - 1);

            CUmemAllocationProp pr = {};
            pr.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            pr.location.type = CU_MEM_LOCATION_TYPE_DEVICE; pr.location.id = dv;
            CUmemGenericAllocationHandle h;
            CUresult r = cuMemCreate(&h, chunk, &pr, 0);
            if (r != CUDA_SUCCESS) { LOG("  cuMemCreate[%d] chunk fail %d", i, r); ok = 0; break; }
            cuMemMap(current, chunk, 0, h, 0); cuMemRelease(h);
            CUmemAccessDesc ac = {};
            ac.location.type = CU_MEM_LOCATION_TYPE_DEVICE; ac.location.id = dv;
            ac.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            cuMemSetAccess(current, chunk, &ac, 1);
            current += chunk; remaining -= chunk;
        }
        if (!ok) continue;

        cuMemcpyHtoD(dp, sh + off, s); off += s; n++;
        LOG("  remap[%d]: %p <- host (%zu MB)", i, es[i].p, s>>20);
    }
    pthread_mutex_unlock(&lk);
    free(sh); sh = NULL; sh_sz = 0;
    LOG("unpark done: %d entries", n);
    return 0;
}

static void listener(void) {
    cf = open("/tmp/pflash_shim.ctl", O_RDONLY);
    if (cf < 0) return;
    af = open("/tmp/pflash_shim.ack", O_WRONLY | O_NONBLOCK);
    if (af < 0) return;
    LOG("listener ready");
    char buf[256];
    while (thr) {
        int n = read(cf, buf, sizeof(buf)-1);
        if (n <= 0) { usleep(10000); continue; }
        buf[n] = 0; while (n>0 && (buf[n-1]=='\n'||buf[n-1]==' ')) buf[--n]=0;
        const char *r = "error\n";
        if (!strcmp(buf,"park")) r = park()==0?"ok\n":"err:park\n";
        else if (!strcmp(buf,"unpark")) r = unpark()==0?"ok\n":"err:unpark\n";
        else if (!strcmp(buf,"quit")) break;
        write(af, r, strlen(r)); fsync(af);
    }
    close(cf); close(af);
    unlink("/tmp/pflash_shim.ctl"); unlink("/tmp/pflash_shim.ack");
}

static void shim_init(void) {
    LOG("loading");
    mkfifo("/tmp/pflash_shim.ctl",0666);
    mkfifo("/tmp/pflash_shim.ack",0666);
    thr = 1;
    pthread_create(&th, NULL, (void*(*)(void*))listener, NULL);
    pthread_detach(th);
    LOG("loaded (VMM lazy-init on first cudaMalloc)");
}

static void shim_fini(void) {
    thr = 0;
    if (cf>=0) close(cf); if (af>=0) close(af);
    unlink("/tmp/pflash_shim.ctl"); unlink("/tmp/pflash_shim.ack");
}
