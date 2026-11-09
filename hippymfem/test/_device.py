# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""What a check that compares two computations may expect on a GPU.

On the host the routes that a suite compares add the same numbers in the same order,
and their results agree bit for bit; the checks say so with ``== 0``.  On a GPU the
element kernels' scatter adds with atomic operations and hypre forms its products on
the device, so the order of a sum changes from one run to the next and two results
that are equal in exact arithmetic differ by round-off.  :func:`same` keeps such a
check exact on the host and allows :data:`DEVICE_ROUNDOFF` of the result's scale on a
GPU.  A suite runs in that mode when ``HIPPYMFEM_DEVICE=gpu`` (the element kernels) or
``HIPPYMFEM_HYPRE_DEVICE=1`` (MFEM and hypre) is set, as ``run_tests.sh`` does with
``DEVICE_SUITES=1``.
"""

#: the relative difference two results that are equal in exact arithmetic may show on
#: a GPU: about 450 units in the last place of a double; the differences measured on
#: L40S, Blackwell and MI210 cards were 1e-19 to 3e-14 of the blocks' largest entries
DEVICE_ROUNDOFF = 1e-13


def on_device():
    """Whether this process assembles or applies its operators on a GPU: MFEM is
    configured on one, or the element kernels run on one."""
    from hippymfem.common.parvector import device_active

    if device_active():
        return True
    try:
        from hippymfem.fem import kernel

        return kernel.device().platform != "cpu"
    except Exception:                                                # noqa: BLE001
        return False


def roundoff():
    """The relative difference allowed between results that are equal in exact
    arithmetic: 0 on the host, :data:`DEVICE_ROUNDOFF` on a GPU."""
    return DEVICE_ROUNDOFF if on_device() else 0.0


def same(diff, scale=1.0):
    """Whether two results that differ by ``diff`` (an absolute difference, measured
    against ``scale``; pass a relative difference with the default 1) agree: exactly on
    the host, to round-off on a GPU.  NaN never agrees."""
    return bool(diff <= roundoff() * abs(scale))
