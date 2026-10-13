# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""NVIDIA's cuDSS on the monolithic Stokes Jacobian, through nvmath-python's low-level
bindings: an LDL^T factorization of the symmetric indefinite saddle-point matrix on one
GPU, the analysis done once and the values replaced in place for every new Jacobian.

The matrix is scaled symmetrically by its rows' largest entries before it is factorized:
without that the friction and the penalty rows beside the viscous and the pressure rows
leave solutions wrong by several percent.  Solutions are refined against the
double-precision matrix.  One rank holds the whole matrix.

Needs ``cupy``, ``nvmath-python`` and ``nvidia-cudss-cu12``.
"""
import time

import numpy as np

from hippymfem.common.linalg import _diag_block
from hippymfem.common.parvector import host_sync

_CUDA_TYPE = {"float64": 1, "float32": 0, "int32": 10, "int64": 24}


class CuDSSDirect:
    """``factorize(J)`` and ``solve(x, b)`` for the problem's Newton, adjoint and
    incremental systems (``J`` symmetric)."""

    def __init__(self, comm, precision="double", ir_tol=1e-12, ir_max=6, spd=False):
        """``precision``: of the factors ("single" halves their memory; with refinement the
        answers are still double precision).  ``spd``: Cholesky, for a symmetric positive
        definite matrix such as the velocity block."""
        import cupy as cp
        from nvmath.bindings import cudss

        if comm.size != 1:
            raise ValueError("cuDSS runs on one rank here")
        self.cp, self.cudss = cp, cudss
        self.precision = precision
        self.low = np.float32 if precision == "single" else np.float64
        self.ir_tol, self.ir_max = float(ir_tol), int(ir_max)
        self.spd = bool(spd)
        self.handle = None
        self.sig = None
        self.times = {"analysis": 0.0, "factorize": 0.0, "solve": 0.0, "transfer": 0.0}
        self.counts = {"analysis": 0, "factorize": 0, "solve": 0, "refine": 0}
        self.max_residual = 0.0
        self.memory_estimates = None

    def _csr(self, J):
        blk = _diag_block(J)
        host_sync(blk)
        n = blk.Height()
        indptr = np.asarray(blk.GetIArray())[:n + 1]
        nnz = int(indptr[-1])
        return n, indptr, np.asarray(blk.GetJArray())[:nnz], np.asarray(blk.GetDataArray())[:nnz]

    def _setup(self, n, indptr, indices):
        cp, c = self.cp, self.cudss
        self.free()
        self.n = n
        otype = np.int64 if indptr[-1] >= 2 ** 31 else np.int32
        self.indptr = cp.asarray(indptr.astype(otype))
        self.indices = cp.asarray(indices.astype(np.int32))
        self.rows = cp.asarray(np.repeat(np.arange(n, dtype=np.int32), np.diff(indptr)))
        self.vals64 = cp.zeros(int(indptr[-1]), dtype=np.float64)
        self.vals = self.vals64 if self.low is np.float64 else cp.zeros(int(indptr[-1]), dtype=np.float32)
        self.Acsr = None
        self.device = cp.cuda.runtime.getDevice()
        self.handle = c.create()
        self.config = c.config_create()
        # no cudaHostRegister of cuDSS's host buffers: with several ranks on a node the
        # page-locked memory runs out in the analysis
        self._set(c.ConfigParam.USE_CUDA_REGISTER_MEMORY, np.array([0], np.int32))
        self.data = c.data_create(self.handle)
        vt = _CUDA_TYPE[np.dtype(self.low).name]
        self.matrix = c.matrix_create_csr(n, n, int(indptr[-1]), self.indptr.data.ptr, 0, self.indices.data.ptr,
                                          self.vals.data.ptr, _CUDA_TYPE[np.dtype(otype).name], _CUDA_TYPE["int32"],
                                          vt, c.MatrixType.SPD if self.spd else c.MatrixType.SYMMETRIC,
                                          c.MatrixViewType.FULL, c.IndexBase.ZERO)
        self.x = cp.zeros(n, dtype=self.low)
        self.b = cp.zeros(n, dtype=self.low)
        self.xmat = c.matrix_create_dn(n, 1, n, self.x.data.ptr, vt, c.Layout.COL_MAJOR)
        self.bmat = c.matrix_create_dn(n, 1, n, self.b.data.ptr, vt, c.Layout.COL_MAJOR)
        self._run(c.Phase.ANALYSIS)
        est = np.zeros(16, dtype=np.int64)
        written = np.zeros(1, dtype=np.uint64)
        try:
            c.data_get(self.handle, self.data, c.DataParam.MEMORY_ESTIMATES, est.ctypes.data, est.nbytes,
                       written.ctypes.data)
            self.memory_estimates = [int(v) for v in est[:5]]
        except Exception:
            pass

    def _set(self, param, array):
        self.cudss.config_set(self.config, param, array.ctypes.data, array.nbytes)

    def _run(self, phase):
        rt = self.cp.cuda.runtime
        try:
            self.cudss.execute(self.handle, phase, self.config, self.data, self.matrix, self.xmat, self.bmat)
        finally:
            rt.setDevice(self.device)
        rt.deviceSynchronize()

    def factorize(self, J):
        return self.factorize_csr(*self._csr(J))

    def factorize_csr(self, n, indptr, indices, data):
        """The same for a matrix given as host CSR arrays (both triangles of a symmetric one)."""
        cp = self.cp
        sig = (n, int(indptr[-1]), int(indices[::9973].sum()))
        if sig != self.sig:
            t0 = time.perf_counter()
            self._setup(n, indptr, indices)
            self.sig = sig
            self.times["analysis"] += time.perf_counter() - t0
            self.counts["analysis"] += 1
        t0 = time.perf_counter()
        self.vals64.set(np.ascontiguousarray(data, dtype=np.float64))
        import cupyx.scipy.sparse as csp

        # S J S with S = diag(1 / sqrt(max_j |J_ij|)): rows of very different size (the
        # friction and the penalty against the viscous terms, the pressure rows) are what
        # makes the pivots small
        rowmax = cp.zeros(n, dtype=np.float64)
        cp.maximum.at(rowmax, self.rows, cp.abs(self.vals64))
        self.s = 1.0 / cp.sqrt(cp.maximum(rowmax, 1e-300))
        self.vals64 *= self.s[self.rows] * self.s[self.indices]
        if self.vals is not self.vals64:
            self.vals[...] = self.vals64
        self.Ascaled = csp.csr_matrix((self.vals64, self.indices, self.indptr), shape=(n, n))
        cp.get_default_memory_pool().free_all_blocks()
        self.times["transfer"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        self._run(self.cudss.Phase.FACTORIZATION)
        self.times["factorize"] += time.perf_counter() - t0
        self.counts["factorize"] += 1
        return self

    def _lowsolve(self, r):
        s = float(self.cp.linalg.norm(r)) or 1.0
        self.b[...] = (r / s).astype(self.low)
        self._run(self.cudss.Phase.SOLVE)
        return self.x.astype(np.float64) * s

    #: the refinement's tolerance for the solves of a Hessian action (``solve_incremental``):
    #: the default 1e-12 is at the floor of what refinement reaches, and getting there
    #: costs 3.5 more triangular solves a solve on average; one is enough for 1e-9
    ir_tol_incremental = 1e-9

    def solve_incremental(self, x, b):
        return self.solve(x, b, ir_tol=self.ir_tol_incremental)

    def solve(self, x, b, ir_tol=None):
        cp = self.cp
        ir_tol = self.ir_tol if ir_tol is None else ir_tol
        t0 = time.perf_counter()
        rs = cp.asarray(np.asarray(b.array, dtype=np.float64)) * self.s
        nrs = float(cp.linalg.norm(rs)) or 1.0
        ys = self._lowsolve(rs)
        rr = rs - self.Ascaled @ ys
        for _ in range(self.ir_max):
            if float(cp.linalg.norm(rr)) <= ir_tol * nrs:
                break
            ys2 = ys + self._lowsolve(rr)
            rr2 = rs - self.Ascaled @ ys2
            self.counts["refine"] += 1
            if float(cp.linalg.norm(rr2)) >= float(cp.linalg.norm(rr)):
                break
            ys, rr = ys2, rr2
        # the residual of the system factorized (the scaled one): no unscaled copy is kept
        self.last_residual = float(cp.linalg.norm(rr)) / nrs
        self.max_residual = max(self.max_residual, self.last_residual)
        x.array[:] = cp.asnumpy(ys * self.s)
        self.times["solve"] += time.perf_counter() - t0
        self.counts["solve"] += 1
        return x

    def solve_many(self, B, refine=1):
        """``A^{-1} B`` for a block of right-hand sides (a CuPy array ``(n, k)``), in one
        solve phase per sweep of refinement: much faster than ``k`` single solves."""
        cp, c = self.cp, self.cudss
        k = B.shape[1]
        if getattr(self, "_k", None) != k:
            for name in ("_xm", "_bm"):
                if getattr(self, name, None) is not None:
                    c.matrix_destroy(getattr(self, name))
            vt = _CUDA_TYPE[np.dtype(self.low).name]
            self._X = cp.zeros((self.n, k), dtype=self.low, order="F")
            self._B = cp.zeros((self.n, k), dtype=self.low, order="F")
            self._xm = c.matrix_create_dn(self.n, k, self.n, self._X.data.ptr, vt, c.Layout.COL_MAJOR)
            self._bm = c.matrix_create_dn(self.n, k, self.n, self._B.data.ptr, vt, c.Layout.COL_MAJOR)
            self._k = k

        def low(R):
            sc = cp.maximum(cp.linalg.norm(R, axis=0), 1e-300)
            self._B[...] = (R / sc).astype(self.low)
            try:
                c.execute(self.handle, c.Phase.SOLVE, self.config, self.data, self.matrix, self._xm, self._bm)
            finally:
                cp.cuda.runtime.setDevice(self.device)
            cp.cuda.runtime.deviceSynchronize()
            return self._X.astype(np.float64) * sc
        t0 = time.perf_counter()
        Bs = B * self.s[:, None]
        Ys = low(Bs)
        for _ in range(refine):
            Ys = Ys + low(Bs - self.Ascaled @ Ys)
        Y = Ys * self.s[:, None]
        self.times["solve"] += time.perf_counter() - t0
        self.counts["solve"] += k
        return Y

    def inertia(self):
        """The numbers of positive and of negative pivots of the factorization of a
        symmetric matrix (the signs of its eigenvalues), or ``None`` where cuDSS does not
        say."""
        if self.spd or self.handle is None:
            return None
        buf = np.zeros(2, dtype=np.int32)                        # the size cuDSS asks for
        written = np.zeros(1, dtype=np.uint64)
        try:
            self.cudss.data_get(self.handle, self.data, self.cudss.DataParam.INERTIA, buf.ctypes.data, buf.nbytes,
                                written.ctypes.data)
        except Exception:                                        # noqa: BLE001
            return None
        return int(buf[0]), int(buf[1])

    def free(self):
        if self.handle is None:
            return
        c = self.cudss
        for m in (self.matrix, self.xmat, self.bmat):
            c.matrix_destroy(m)
        c.data_destroy(self.handle, self.data)
        c.config_destroy(self.config)
        c.destroy(self.handle)
        self.handle = None
        self.sig = None

    def __del__(self):
        try:
            self.free()
        except Exception:                                        # noqa: BLE001
            pass

    def describe(self):
        return {"precision": self.precision, "times": dict(self.times),
                "counts": dict(self.counts), "max_refined_residual": self.max_residual,
                "memory_estimates": self.memory_estimates}
