/* gpuprof: count what a GPU process asks of the CUDA driver, cuSPARSE and MPI.
 *
 * An LD_PRELOAD library.  It answers one question: what does a Krylov iteration with a
 * BoomerAMG V-cycle do on a GPU besides arithmetic?  It counts, and times on the calling
 * thread, the calls that cost a fixed amount whatever the size of the problem:
 *
 *   driver   cuMemAlloc / cuMemFree          device allocations
 *            cuMemcpy{DtoH,HtoD,DtoD}*       copies (count, bytes, time on the caller)
 *            cuLaunchKernel*                 kernel launches
 *            cuStreamSynchronize, cuCtxSynchronize
 *   cuSPARSE cusparseSpMV, cusparseCreateCsr, cusparseSpMV_bufferSize
 *   MPI      MPI_Isend, MPI_Irecv, MPI_Waitall, MPI_Allreduce (profiling interface)
 *
 * hypre links the CUDA runtime statically, so the runtime entry points (cudaMalloc, ...)
 * cannot be interposed.  Every CUDA 12 runtime, static or shared, opens libcuda.so.1,
 * finds the driver's lookup function cuGetProcAddress with dlsym, and asks it for each
 * entry point by name.  This library interposes dlsym for that one name and wraps the
 * entry points the lookup hands out.  Any other name goes to the real dlsym by a tail
 * call, which keeps the caller's return address and so the meaning of RTLD_NEXT for the
 * libraries that use it.
 *
 * In the synchronized mode (gpuprof_sync_mode) every kernel launch and every cuSPARSE
 * product is bracketed by a synchronization of the context and its device time is
 * tallied by kernel and grid size, or by the rows and nonzeros of the matrix
 * (gpuprof_dump).  The library also applies BoomerAMG options that MFEM does not expose,
 * from environment variables, just before the setup (see apply_amg_options).
 *
 * Read from Python with ctypes:
 *   lib = ctypes.CDLL("libgpuprof.so"); lib.gpuprof_reset(); ...; lib.gpuprof_get(c, t, b)
 *
 * Build:  mpicc -O2 -fPIC -shared -o libgpuprof.so tools/gpuprof.c -ldl
 * Use:    mpirun -n 4 -x LD_PRELOAD=$PWD/libgpuprof.so tools/mpirun_pinned.sh \
 *             python benchmarks/krylov_anatomy.py ...
 *
 * Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.  This file is part of
 * hIPPyMFEM, free software under the GNU General Public License version 2.0 dated June
 * 1991 (GPL-2.0-only); see the files LICENSE and COPYRIGHT.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <mpi.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

enum { K_ALLOC, K_FREE, K_D2H, K_H2D, K_D2D, K_UCOPY, K_LAUNCH, K_SYNC,
       K_SPMV, K_SPCREATE, K_SPBUF,
       K_ISEND, K_IRECV, K_WAITALL, K_ALLREDUCE, K_AMGSETUP, K_AMGSOLVE, K_N };

static const char *const k_names[K_N] = {
   "dev_alloc", "dev_free", "copy_d2h", "copy_h2d", "copy_d2d", "copy", "launch", "sync",
   "cusparse_spmv", "cusparse_create", "cusparse_bufsize",
   "mpi_isend", "mpi_irecv", "mpi_waitall", "mpi_allreduce", "amg_setup", "amg_solve" };

static long   k_count[K_N];
static long   k_nsec[K_N];
static double k_bytes[K_N];

static inline long now_ns(void)
{
   struct timespec ts;
   clock_gettime(CLOCK_MONOTONIC, &ts);
   return ts.tv_sec * 1000000000L + ts.tv_nsec;
}

static inline void tally(int k, long t0, double bytes)
{
   __atomic_fetch_add(&k_count[k], 1, __ATOMIC_RELAXED);
   __atomic_fetch_add(&k_nsec[k], now_ns() - t0, __ATOMIC_RELAXED);
   k_bytes[k] += bytes;
}

/* Synchronized mode: every kernel launch and every cuSPARSE product is bracketed by a
 * synchronization of the context, so its time on the device is known, and tallied by
 * kernel (function and grid size) or by matrix (rows and nonzeros).  It slows the run and
 * is used for one solve at a time, to see which kernel the time of an iteration goes to. */

static int sync_mode = 0;
static void *ctx_sync_fn = NULL;                 /* the driver's cuCtxSynchronize */

struct krow { const void *f; long a, b; long count; long nsec; };
#define KMAX 1024
static struct krow ktab[KMAX];
static int nk = 0;

static void ktally(const void *f, long a, long b, long nsec)
{
   for (int i = 0; i < nk; i++)
   {
      if (ktab[i].f == f && ktab[i].a == a && ktab[i].b == b)
      {
         ktab[i].count++;
         ktab[i].nsec += nsec;
         return;
      }
   }
   if (nk < KMAX)
   {
      ktab[nk].f = f; ktab[nk].a = a; ktab[nk].b = b; ktab[nk].count = 1; ktab[nk].nsec = nsec;
      nk++;
   }
}

static inline void ctx_sync(void)
{
   if (ctx_sync_fn) { ((int (*)(void)) ctx_sync_fn)(); }
}

void gpuprof_sync_mode(int on)
{
   sync_mode = on;
   nk = 0;
}

/* One line per kernel: "L <function> <grid x> <block x> <count> <seconds>", and per
 * matrix handed to cuSPARSE: "S 0 <rows> <nonzeros> <count> <seconds>".              */
int gpuprof_dump(const char *path)
{
   FILE *f = fopen(path, "w");
   if (!f) { return -1; }
   for (int i = 0; i < nk; i++)
   {
      fprintf(f, "%s %p %ld %ld %ld %.9f\n", ktab[i].f ? "L" : "S", ktab[i].f, ktab[i].a,
              ktab[i].b, ktab[i].count, 1e-9 * (double) ktab[i].nsec);
   }
   fclose(f);
   return nk;
}

int gpuprof_nkinds(void) { return K_N; }
const char *gpuprof_name(int k) { return (k >= 0 && k < K_N) ? k_names[k] : ""; }

void gpuprof_reset(void)
{
   memset(k_count, 0, sizeof k_count);
   memset(k_nsec, 0, sizeof k_nsec);
   memset(k_bytes, 0, sizeof k_bytes);
}

void gpuprof_get(long *count, double *seconds, double *bytes)
{
   for (int k = 0; k < K_N; k++)
   {
      count[k] = k_count[k];
      seconds[k] = 1e-9 * (double) k_nsec[k];
      bytes[k] = k_bytes[k];
   }
}

/* ---------------------------------------------------------------- the real dlsym */

static void *(*real_dlsym)(void *, const char *) = NULL;

static void find_real_dlsym(void)
{
   real_dlsym = (void *(*)(void *, const char *)) dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.34");
   if (!real_dlsym)
   {
      real_dlsym = (void *(*)(void *, const char *)) dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
   }
}

/* ---------------------------------------------------------------- driver wrappers
 * The CUDA 12 runtime asks the driver for its entry points by name, through
 * cuGetProcAddress_v2, which it finds with dlsym.  That one function is wrapped, and it
 * hands back a counting wrapper for the names below.  A name can be requested with two
 * meanings (flags select the per-thread default stream), and several runtimes live in
 * one process (hypre's static one, the shared one of MFEM and JAX, the one inside
 * cuSPARSE), so every name has one slot per value of the flags.                      */

typedef int CUresult;
typedef unsigned long long CUdeviceptr;

#define DEF(name, PARAMS, ARGS, KIND, BYTES)                                        \
   static void *real_##name[2];                                                     \
   static CUresult w0_##name PARAMS                                                 \
   {                                                                                \
      long t0 = now_ns();                                                           \
      CUresult r = ((CUresult (*) PARAMS) real_##name[0]) ARGS;                     \
      tally(KIND, t0, (double) (BYTES));                                            \
      return r;                                                                     \
   }                                                                                \
   static CUresult w1_##name PARAMS                                                 \
   {                                                                                \
      long t0 = now_ns();                                                           \
      CUresult r = ((CUresult (*) PARAMS) real_##name[1]) ARGS;                     \
      tally(KIND, t0, (double) (BYTES));                                            \
      return r;                                                                     \
   }

DEF(cuMemAlloc, (CUdeviceptr *p, size_t n), (p, n), K_ALLOC, n)
DEF(cuMemAllocAsync, (CUdeviceptr *p, size_t n, void *st), (p, n, st), K_ALLOC, n)
DEF(cuMemAllocFromPoolAsync, (CUdeviceptr *p, size_t n, void *pool, void *st),
    (p, n, pool, st), K_ALLOC, n)
DEF(cuMemAllocManaged, (CUdeviceptr *p, size_t n, unsigned flags), (p, n, flags), K_ALLOC, n)
DEF(cuMemFree, (CUdeviceptr p), (p), K_FREE, 0)
DEF(cuMemFreeAsync, (CUdeviceptr p, void *st), (p, st), K_FREE, 0)
DEF(cuMemcpyDtoH, (void *d, CUdeviceptr s_, size_t n), (d, s_, n), K_D2H, n)
DEF(cuMemcpyHtoD, (CUdeviceptr d, const void *s_, size_t n), (d, s_, n), K_H2D, n)
DEF(cuMemcpyDtoD, (CUdeviceptr d, CUdeviceptr s_, size_t n), (d, s_, n), K_D2D, n)
DEF(cuMemcpyDtoHAsync, (void *d, CUdeviceptr s_, size_t n, void *st), (d, s_, n, st), K_D2H, n)
DEF(cuMemcpyHtoDAsync, (CUdeviceptr d, const void *s_, size_t n, void *st), (d, s_, n, st),
    K_H2D, n)
DEF(cuMemcpyDtoDAsync, (CUdeviceptr d, CUdeviceptr s_, size_t n, void *st), (d, s_, n, st),
    K_D2D, n)
/* cuMemcpy and cuMemcpyAsync take unified addresses: the direction is not known here */
DEF(cuMemcpy, (CUdeviceptr d, CUdeviceptr s_, size_t n), (d, s_, n), K_UCOPY, n)
DEF(cuMemcpyAsync, (CUdeviceptr d, CUdeviceptr s_, size_t n, void *st), (d, s_, n, st),
    K_UCOPY, n)
typedef CUresult (*launch_t)(void *, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                             unsigned, void *, void **, void **);
typedef CUresult (*launchex_t)(const void *, void *, void **, void **);
static void *real_cuLaunchKernel[2];
static void *real_cuLaunchKernelEx[2];

static inline CUresult launch(int v, void *f, unsigned gx, unsigned gy, unsigned gz, unsigned bx,
                              unsigned by, unsigned bz, unsigned shared, void *st, void **params,
                              void **extra)
{
   if (__builtin_expect(sync_mode, 0))
   {
      ctx_sync();
      long t0 = now_ns();
      CUresult r = ((launch_t) real_cuLaunchKernel[v])(f, gx, gy, gz, bx, by, bz, shared, st,
                                                       params, extra);
      ctx_sync();
      ktally(f, (long) gx * gy * gz, (long) bx * by * bz, now_ns() - t0);
      tally(K_LAUNCH, t0, 0.0);
      return r;
   }
   long t0 = now_ns();
   CUresult r = ((launch_t) real_cuLaunchKernel[v])(f, gx, gy, gz, bx, by, bz, shared, st, params,
                                                    extra);
   tally(K_LAUNCH, t0, 0.0);
   return r;
}

static CUresult w0_cuLaunchKernel(void *f, unsigned gx, unsigned gy, unsigned gz, unsigned bx,
                                  unsigned by, unsigned bz, unsigned shared, void *st,
                                  void **params, void **extra)
{
   return launch(0, f, gx, gy, gz, bx, by, bz, shared, st, params, extra);
}

static CUresult w1_cuLaunchKernel(void *f, unsigned gx, unsigned gy, unsigned gz, unsigned bx,
                                  unsigned by, unsigned bz, unsigned shared, void *st,
                                  void **params, void **extra)
{
   return launch(1, f, gx, gy, gz, bx, by, bz, shared, st, params, extra);
}

static inline CUresult launchex(int v, const void *config, void *f, void **params, void **extra)
{
   if (__builtin_expect(sync_mode, 0))
   {
      const unsigned *c = (const unsigned *) config;     /* grid x, y, z, block x, y, z */
      ctx_sync();
      long t0 = now_ns();
      CUresult r = ((launchex_t) real_cuLaunchKernelEx[v])(config, f, params, extra);
      ctx_sync();
      ktally(f, (long) c[0] * c[1] * c[2], (long) c[3] * c[4] * c[5], now_ns() - t0);
      tally(K_LAUNCH, t0, 0.0);
      return r;
   }
   long t0 = now_ns();
   CUresult r = ((launchex_t) real_cuLaunchKernelEx[v])(config, f, params, extra);
   tally(K_LAUNCH, t0, 0.0);
   return r;
}

static CUresult w0_cuLaunchKernelEx(const void *config, void *f, void **params, void **extra)
{
   return launchex(0, config, f, params, extra);
}

static CUresult w1_cuLaunchKernelEx(const void *config, void *f, void **params, void **extra)
{
   return launchex(1, config, f, params, extra);
}

DEF(cuStreamSynchronize, (void *st), (st), K_SYNC, 0)
DEF(cuCtxSynchronize, (void), (), K_SYNC, 0)
DEF(cuEventSynchronize, (void *ev), (ev), K_SYNC, 0)

struct hook { const char *name; void **real; void *wrap[2]; };
#define HOOK(name) { #name, real_##name, { (void *) w0_##name, (void *) w1_##name } }
static const struct hook hooks[] = {
   HOOK(cuMemAlloc), HOOK(cuMemAllocAsync), HOOK(cuMemAllocFromPoolAsync),
   HOOK(cuMemAllocManaged), HOOK(cuMemFree), HOOK(cuMemFreeAsync),
   HOOK(cuMemcpyDtoH), HOOK(cuMemcpyHtoD), HOOK(cuMemcpyDtoD),
   HOOK(cuMemcpyDtoHAsync), HOOK(cuMemcpyHtoDAsync), HOOK(cuMemcpyDtoDAsync),
   HOOK(cuMemcpy), HOOK(cuMemcpyAsync),
   HOOK(cuLaunchKernel), HOOK(cuLaunchKernelEx),
   HOOK(cuStreamSynchronize), HOOK(cuCtxSynchronize), HOOK(cuEventSynchronize),
};

static int n_hooked = 0;
int gpuprof_hooked(void) { return n_hooked; }

static int debug_on(void)
{
   static int on = -1;
   if (on < 0) { on = getenv("GPUPROF_DEBUG") != NULL; }
   return on;
}

static void substitute(const char *proc, void **fptr, unsigned long long flags)
{
   int v = flags ? 1 : 0;
   for (size_t i = 0; i < sizeof hooks / sizeof hooks[0]; i++)
   {
      if (strcmp(proc, hooks[i].name) == 0)
      {
         if (*fptr == hooks[i].wrap[0] || *fptr == hooks[i].wrap[1]) { return; }
         if (hooks[i].real[v] && hooks[i].real[v] != *fptr)
         {
            /* a third meaning of the name: leave this request alone */
            if (debug_on()) { fprintf(stderr, "gpuprof: %s has another address, not wrapped\n", proc); }
            return;
         }
         hooks[i].real[v] = *fptr;
         if (strcmp(proc, "cuCtxSynchronize") == 0 && !ctx_sync_fn) { ctx_sync_fn = *fptr; }
         *fptr = hooks[i].wrap[v];
         n_hooked++;
         return;
      }
   }
}

/* The runtime first asks cuGetProcAddress_v2 (found with dlsym) for "cuGetProcAddress"
 * itself, in the version it was built for, and then makes every other request through the
 * function it got back: five arguments from CUDA 12.0, four before.                    */

typedef int (*gpa5_t)(const char *, void **, int, unsigned long long, void *);
typedef int (*gpa4_t)(const char *, void **, int, unsigned long long);
static gpa5_t real_gpa5[2] = { NULL, NULL };    /* [0] from dlsym, [1] from the driver */
static gpa4_t real_gpa4 = NULL;
static int w_gpa5_0(const char *, void **, int, unsigned long long, void *);
static int w_gpa5_1(const char *, void **, int, unsigned long long, void *);
static int w_gpa4(const char *, void **, int, unsigned long long);

static void after_lookup(const char *proc, void **fptr, int version, unsigned long long flags,
                         int r)
{
   if (debug_on())
   {
      fprintf(stderr, "gpuprof: cuGetProcAddress(%s, version %d, flags %llu) = %d\n",
              proc ? proc : "(null)", version, flags, r);
   }
   if (r != 0 || !proc || !fptr || !*fptr) { return; }
   if (strcmp(proc, "cuGetProcAddress") == 0)
   {
      if (*fptr == (void *) w_gpa5_0 || *fptr == (void *) w_gpa5_1 || *fptr == (void *) w_gpa4)
      {
         return;
      }
      if (version >= 12000)
      {
         real_gpa5[1] = (gpa5_t) *fptr;
         *fptr = (void *) w_gpa5_1;
      }
      else
      {
         real_gpa4 = (gpa4_t) *fptr;
         *fptr = (void *) w_gpa4;
      }
      return;
   }
   substitute(proc, fptr, flags);
}

static int w_gpa5_0(const char *proc, void **fptr, int version, unsigned long long flags,
                    void *status)
{
   int r = real_gpa5[0](proc, fptr, version, flags, status);
   after_lookup(proc, fptr, version, flags, r);
   return r;
}

static int w_gpa5_1(const char *proc, void **fptr, int version, unsigned long long flags,
                    void *status)
{
   int r = real_gpa5[1](proc, fptr, version, flags, status);
   after_lookup(proc, fptr, version, flags, r);
   return r;
}

static int w_gpa4(const char *proc, void **fptr, int version, unsigned long long flags)
{
   int r = real_gpa4(proc, fptr, version, flags);
   after_lookup(proc, fptr, version, flags, r);
   return r;
}

static __attribute__((noinline)) void *driver_wrapper(void *handle, const char *name)
{
   if (strcmp(name, "cuGetProcAddress_v2") == 0)
   {
      void *f = real_dlsym(handle, name);
      if (!f) { return NULL; }
      real_gpa5[0] = (gpa5_t) f;
      return (void *) w_gpa5_0;
   }
   if (strcmp(name, "cuGetProcAddress") == 0)
   {
      void *f = real_dlsym(handle, name);
      if (!f) { return NULL; }
      real_gpa4 = (gpa4_t) f;
      return (void *) w_gpa4;
   }
   return NULL;
}

void *dlsym(void *handle, const char *name)
{
   if (__builtin_expect(!real_dlsym, 0)) { find_real_dlsym(); }
   if (name && name[0] == 'c' && name[1] == 'u' && name[2] == 'G' && handle != RTLD_NEXT
       && handle != RTLD_DEFAULT && !getenv("GPUPROF_OFF"))
   {
      void *w = driver_wrapper(handle, name);
      if (w) { return w; }
   }
   return real_dlsym(handle, name);            /* a tail call: the caller's frame is kept */
}

/* ---------------------------------------------------------------- finding the real functions
 * Python loads its extension modules, and with them MFEM, hypre and cuSPARSE, outside
 * the global scope, where RTLD_NEXT does not look.  The real function is therefore taken
 * from the library itself, found by name among the mappings of the process.           */

static void *in_library(const char *libname, const char *symbol)
{
   char line[4096];
   void *f = NULL;
   if (!real_dlsym) { find_real_dlsym(); }
   FILE *maps = fopen("/proc/self/maps", "r");
   if (maps)
   {
      while (!f && fgets(line, sizeof line, maps))
      {
         char *path = strchr(line, '/');
         if (!path || !strstr(path, libname)) { continue; }
         path[strcspn(path, "\n")] = 0;
         void *h = dlopen(path, RTLD_LAZY | RTLD_NOLOAD);
         if (h) { f = real_dlsym(h, symbol); }
      }
      fclose(maps);
   }
   if (!f) { f = real_dlsym(RTLD_NEXT, symbol); }
   if (!f)
   {
      fprintf(stderr, "gpuprof: cannot find %s in %s\n", symbol, libname);
      abort();
   }
   return f;
}

/* ---------------------------------------------------------------- cuSPARSE (PLT) */

static long last_rows = 0, last_nnz = 0;       /* of the descriptor created last */

int cusparseSpMV(void *handle, int op, const void *alpha, void *matA, void *vecX,
                 const void *beta, void *vecY, int type, int alg, void *buffer)
{
   static int (*real)(void *, int, const void *, void *, void *, const void *, void *, int,
                      int, void *) = NULL;
   if (!real) { real = in_library("libcusparse", "cusparseSpMV"); }
   if (__builtin_expect(sync_mode, 0))
   {
      int saved = sync_mode;
      ctx_sync();
      sync_mode = 0;                           /* its kernels are counted as one product */
      long t0 = now_ns();
      int r = real(handle, op, alpha, matA, vecX, beta, vecY, type, alg, buffer);
      ctx_sync();
      sync_mode = saved;
      ktally(NULL, last_rows, last_nnz, now_ns() - t0);
      tally(K_SPMV, t0, 0.0);
      return r;
   }
   long t0 = now_ns();
   int r = real(handle, op, alpha, matA, vecX, beta, vecY, type, alg, buffer);
   tally(K_SPMV, t0, 0.0);
   return r;
}

int cusparseCreateCsr(void *descr, long rows, long cols, long nnz, void *rowptr, void *colind,
                      void *values, int rowtype, int coltype, int base, int valtype)
{
   static int (*real)(void *, long, long, long, void *, void *, void *, int, int, int, int) = NULL;
   if (!real) { real = in_library("libcusparse", "cusparseCreateCsr"); }
   long t0 = now_ns();
   int r = real(descr, rows, cols, nnz, rowptr, colind, values, rowtype, coltype, base, valtype);
   tally(K_SPCREATE, t0, (double) nnz);
   last_rows = rows;
   last_nnz = nnz;
   return r;
}

int cusparseSpMV_bufferSize(void *handle, int op, const void *alpha, void *matA, void *vecX,
                            const void *beta, void *vecY, int type, int alg, size_t *size)
{
   static int (*real)(void *, int, const void *, void *, void *, const void *, void *, int,
                      int, size_t *) = NULL;
   if (!real) { real = in_library("libcusparse", "cusparseSpMV_bufferSize"); }
   long t0 = now_ns();
   int r = real(handle, op, alpha, matA, vecX, beta, vecY, type, alg, size);
   tally(K_SPBUF, t0, 0.0);
   return r;
}

/* ---------------------------------------------------------------- MPI (PMPI) */

int MPI_Isend(const void *buf, int count, MPI_Datatype type, int dest, int tag, MPI_Comm comm,
              MPI_Request *req)
{
   long t0 = now_ns();
   int size = 0;
   PMPI_Type_size(type, &size);
   int r = PMPI_Isend(buf, count, type, dest, tag, comm, req);
   tally(K_ISEND, t0, (double) count * size);
   return r;
}

int MPI_Irecv(void *buf, int count, MPI_Datatype type, int src, int tag, MPI_Comm comm,
              MPI_Request *req)
{
   long t0 = now_ns();
   int r = PMPI_Irecv(buf, count, type, src, tag, comm, req);
   tally(K_IRECV, t0, 0.0);
   return r;
}

int MPI_Waitall(int count, MPI_Request reqs[], MPI_Status stats[])
{
   long t0 = now_ns();
   int r = PMPI_Waitall(count, reqs, stats);
   tally(K_WAITALL, t0, 0.0);
   return r;
}

int MPI_Allreduce(const void *in, void *out, int count, MPI_Datatype type, MPI_Op op,
                  MPI_Comm comm)
{
   long t0 = now_ns();
   int r = PMPI_Allreduce(in, out, count, type, op, comm);
   tally(K_ALLREDUCE, t0, 0.0);
   return r;
}

/* ---------------------------------------------------------------- BoomerAMG (PLT)
 * PyMFEM gives no handle to the HYPRE_Solver, so the options MFEM does not expose are
 * applied here, just before the setup, from environment variables that the driving
 * script sets between variants:
 *   GPUPROF_AMG_<OPT>=<int>      HYPRE_BoomerAMGSet<Opt>(solver, int)
 *   GPUPROF_AMGR_<OPT>=<float>   HYPRE_BoomerAMGSet<Opt>(solver, double)
 *   GPUPROF_AMG2_<OPT>=<a>:<b>   HYPRE_BoomerAMGSet<Opt>(solver, int, int)
 * with <OPT> the hypre name, e.g. GPUPROF_AMG_PMaxElmts=2, GPUPROF_AMG_AggInterpType=5,
 * GPUPROF_AMGR_TruncFactor=0.1.                                                       */

extern char **environ;

static void apply_amg_options(void *solver)
{
   char fn[128];
   for (char **e = environ; e && *e; e++)
   {
      int real;
      const char *p;
      if (strncmp(*e, "GPUPROF_AMG_", 12) == 0) { real = 0; p = *e + 12; }
      else if (strncmp(*e, "GPUPROF_AMGR_", 13) == 0) { real = 1; p = *e + 13; }
      else if (strncmp(*e, "GPUPROF_AMG2_", 13) == 0) { real = 2; p = *e + 13; }
      else { continue; }
      const char *eq = strchr(p, '=');
      if (!eq || eq == p || (size_t) (eq - p) > 60 || !eq[1]) { continue; }
      snprintf(fn, sizeof fn, "HYPRE_BoomerAMGSet%.*s", (int) (eq - p), p);
      void *f = in_library("libHYPRE", fn);
      if (real == 1) { ((int (*)(void *, double)) f)(solver, atof(eq + 1)); }
      else if (real == 2)
      {
         const char *colon = strchr(eq + 1, ':');
         ((int (*)(void *, int, int)) f)(solver, atoi(eq + 1), colon ? atoi(colon + 1) : 0);
      }
      else { ((int (*)(void *, int)) f)(solver, atoi(eq + 1)); }
   }
}

int HYPRE_BoomerAMGSetup(void *solver, void *A, void *b, void *x)
{
   static int (*real)(void *, void *, void *, void *) = NULL;
   if (!real) { real = in_library("libHYPRE", "HYPRE_BoomerAMGSetup"); }
   apply_amg_options(solver);
   long t0 = now_ns();
   int r = real(solver, A, b, x);
   tally(K_AMGSETUP, t0, 0.0);
   return r;
}

int HYPRE_BoomerAMGSolve(void *solver, void *A, void *b, void *x)
{
   static int (*real)(void *, void *, void *, void *) = NULL;
   if (!real) { real = in_library("libHYPRE", "HYPRE_BoomerAMGSolve"); }
   long t0 = now_ns();
   int r = real(solver, A, b, x);
   tally(K_AMGSOLVE, t0, 0.0);
   return r;
}
