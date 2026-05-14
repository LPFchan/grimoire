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

typedef struct { CUdeviceptr p; size_t s; int a; } E;
static E es[MAX_A];
static int ne;
static pthread_mutex_t lk = PTHREAD_MUTEX_INITIALIZER;

static unsigned char *sh;
static size_t sh_sz, sh_used;
static CUcontext cx;
static CUdevice dv;
static int vmm;
static CUdeviceptr va;
static size_t vc;

static int cf=-1, af=-1;
static pthread_t th;
static volatile int thr;

__attribute__((constructor)) static void init(void);
__attribute__((destructor))  static void fini(void);

static void *sym(const char *n) {
    void *h;
    if ((h = dlopen("libcuda.so.1", RTLD_LAZY|RTLD_NOLOAD))) { void *p = dlsym(h,n); if(p) return p; }
    if ((h = dlopen("libcudart.so", RTLD_LAZY|RTLD_NOLOAD))) { void *p = dlsym(h,n); if(p) return p; }
    if ((h = dlopen("libcudart.so.12", RTLD_LAZY|RTLD_NOLOAD))) { void *p = dlsym(h,n); if(p) return p; }
    if ((h = dlopen("libggml-cuda.so.0", RTLD_LAZY|RTLD_NOLOAD))) { void *p = dlsym(h,n); if(p) return p; }
    return NULL;
}

static void reg(CUdeviceptr p, size_t s) {
    pthread_mutex_lock(&lk);
    if (ne < MAX_A) { es[ne].p = p; es[ne].s = s; es[ne].a = 1; ne++; }
    LOG("reg[%d]: 0x%lx sz=%zu total=%d", ne-1, (unsigned long)p, s, ne);
    if (!vmm && p) {
        cuCtxGetCurrent(&cx);
        if (cx) {
            cuCtxGetDevice(&dv);
            size_t fr, tt; cuMemGetInfo_v2(&fr, &tt);
            vc = (tt * 9 / 10) & ~(VMM_G - 1);
            if (vc < 2ULL<<30) vc = 2ULL<<30;
            if (vc > 22ULL<<30) vc = 22ULL<<30;
            CUresult r = cuMemAddressReserve(&va, vc, 0, 0, 0);
            LOG("VMM: reserve(%zu MB) → %d va=0x%lx", vc>>20, r, (unsigned long)va);
            vmm = (r == CUDA_SUCCESS);
        }
    }
    pthread_mutex_unlock(&lk);
}

static void unreg(CUdeviceptr p) {
    pthread_mutex_lock(&lk);
    for (int i = 0; i < ne; i++) if (es[i].a && es[i].p == p) { es[i].a = 0; break; }
    pthread_mutex_unlock(&lk);
}

// ── Intercept ALL possible CUDA allocators ──────────────────────────

// CUDA Runtime API
cudaError_t cudaMalloc(void **p, size_t s) {
    static cudaError_t (*real)(void**,size_t);
    if (!real) real = sym("cudaMalloc");
    cudaError_t r = real(p,s);
    if (r==cudaSuccess && p && *p) reg((CUdeviceptr)(uintptr_t)*p, s);
    else LOG("cudaMalloc(%zu) → %d", s, r);
    return r;
}

cudaError_t cudaFree(void *p) {
    static cudaError_t (*real)(void*);
    if (!real) real = sym("cudaFree");
    if (p) unreg((CUdeviceptr)(uintptr_t)p);
    return real(p);
}

// CUDA Driver API — undefine macros that alias cuMemAlloc -> cuMemAlloc_v2
// We define our own cuMemAlloc and cuMemAlloc_v2 that both route through reg()
#pragma push_macro("cuMemAlloc")
#pragma push_macro("cuMemFree")
#undef cuMemAlloc
#undef cuMemFree

CUresult cuMemAlloc(CUdeviceptr *p, size_t s) {
    static CUresult (*real)(CUdeviceptr*,size_t);
    if (!real) real = sym("cuMemAlloc");
    CUresult r = real(p,s);
    if (r==CUDA_SUCCESS && p && *p) reg(*p, s);
    else LOG("cuMemAlloc(%zu) → %d", s, r);
    return r;
}

CUresult cuMemAlloc_v2(CUdeviceptr *p, size_t s) {
    static CUresult (*real)(CUdeviceptr*,size_t);
    if (!real) real = sym("cuMemAlloc_v2");
    CUresult r = real(p,s);
    if (r==CUDA_SUCCESS && p && *p) reg(*p, s);
    else LOG("cuMemAlloc_v2(%zu) → %d", s, r);
    return r;
}

CUresult cuMemFree(CUdeviceptr p) {
    static CUresult (*real)(CUdeviceptr);
    if (!real) real = sym("cuMemFree");
    unreg(p);
    return real(p);
}

CUresult cuMemFree_v2(CUdeviceptr p) {
    static CUresult (*real)(CUdeviceptr);
    if (!real) real = sym("cuMemFree_v2");
    unreg(p);
    return real(p);
}

#pragma pop_macro("cuMemAlloc")
#pragma pop_macro("cuMemFree")

// ── Park ────────────────────────────────────────────────────────────
static int park(void) {
    LOG("park (vmm=%d ne=%d)", vmm, ne);
    if (!vmm) return -1;
    cuCtxSetCurrent(cx);
    pthread_mutex_lock(&lk);
    size_t t = 0; int n = 0;
    for (int i = 0; i < ne; i++) { if (es[i].a) { n++; t += es[i].s; } }
    LOG("park: %d active, %.0f MB", n, (double)t/(1<<20));
    unsigned char *b = malloc(t);
    if (!b) { pthread_mutex_unlock(&lk); return -1; }
    sh = b; sh_sz = t; sh_used = 0;
    for (int i = 0; i < ne; i++) {
        if (!es[i].a) continue;
        size_t al = (es[i].s + VMM_G - 1) & ~(VMM_G - 1);
        LOG("  save[%d]: %zu bytes to off %zu", i, es[i].s, sh_used);
        cuMemcpyDtoH(b + sh_used, es[i].p, es[i].s);
        sh_used += es[i].s;
        if (es[i].p >= va && es[i].p < va + vc) cuMemUnmap(es[i].p, al);
    }
    pthread_mutex_unlock(&lk);
    LOG("park done: %.1f MB", (double)t/(1<<20));
    return 0;
}

// ── Unpark ──────────────────────────────────────────────────────────
static int unpark(void) {
    LOG("unpark (vmm=%d sh=%p)", vmm, (void*)sh);
    if (!vmm || !sh) return -1;
    cuCtxSetCurrent(cx);
    pthread_mutex_lock(&lk);
    size_t off = 0; int n = 0;
    for (int i = 0; i < ne; i++) {
        if (!es[i].a) continue;
        size_t al = (es[i].s + VMM_G - 1) & ~(VMM_G - 1);
        n++;
        if (es[i].p >= va && es[i].p < va + vc) {
            CUmemAllocationProp pr = {}; pr.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            pr.location.type = CU_MEM_LOCATION_TYPE_DEVICE; pr.location.id = dv;
            CUmemGenericAllocationHandle h;
            if (cuMemCreate(&h, al, &pr, 0) != CUDA_SUCCESS) continue;
            cuMemMap(es[i].p, al, 0, h, 0); cuMemRelease(h);
            CUmemAccessDesc ac = {};
            ac.location.type = CU_MEM_LOCATION_TYPE_DEVICE; ac.location.id = dv;
            ac.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            cuMemSetAccess(es[i].p, al, &ac, 1);
        }
        cuMemcpyHtoD(es[i].p, sh + off, es[i].s);
        off += es[i].s;
    }
    pthread_mutex_unlock(&lk);
    free(sh); sh = NULL; sh_sz = 0;
    LOG("unpark done: %d entries", n);
    return 0;
}

// ── FIFO ────────────────────────────────────────────────────────────
static void loop(void) {
    cf = open("/tmp/pflash_shim.ctl", O_RDONLY);
    if (cf < 0) return;
    af = open("/tmp/pflash_shim.ack", O_WRONLY | O_NONBLOCK);
    if (af < 0) return;
    LOG("listener ready");
    char buf[256];
    while (thr) {
        int n = read(cf, buf, sizeof(buf)-1);
        if (n <= 0) { usleep(10000); continue; }
        buf[n] = 0;
        while (n>0 && (buf[n-1]=='\n'||buf[n-1]==' ')) buf[--n]=0;
        const char *r = "error\n";
        if (!strcmp(buf,"park"))    { int ok = park();    r = ok==0?"ok\n":"err:park\n"; }
        else if (!strcmp(buf,"unpark")) { int ok = unpark(); r = ok==0?"ok\n":"err:unpark\n"; }
        else if (!strcmp(buf,"quit")) break;
        write(af, r, strlen(r)); fsync(af);
    }
    close(cf); close(af);
    unlink("/tmp/pflash_shim.ctl"); unlink("/tmp/pflash_shim.ack");
}

static void init(void) {
    LOG("loading");
    unlink("/tmp/pflash_shim.ctl"); mkfifo("/tmp/pflash_shim.ctl",0666);
    unlink("/tmp/pflash_shim.ack"); mkfifo("/tmp/pflash_shim.ack",0666);
    thr = 1;
    pthread_create(&th, NULL, (void*(*)(void*))loop, NULL);
    pthread_detach(th);
    LOG("loaded");
}

static void fini(void) {
    thr = 0;
    if (cf>=0) close(cf); if (af>=0) close(af);
    unlink("/tmp/pflash_shim.ctl"); unlink("/tmp/pflash_shim.ack");
}
