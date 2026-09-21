# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""The geothermal model: steady heat conduction with a tabulated, temperature-dependent
conductivity per lithology, borehole temperature logs, an anisotropic BiLaplacian prior.

Scaled coordinates: the block ``L x L x H`` (8 km x 8 km x 4 km) is the unit cube, so the
vertical derivative carries a factor ``L/H`` and the conductivity tensor in unit
coordinates is ``diag(1, 1, (L/H)^2)``.  Temperature is in units of ``T_SCALE`` kelvin
above the surface; the unknown is ``m = log(k / k_ref)``, the deviation of the rock's
reference conductivity from ``k_ref``, and the temperature dependence of each lithology
is a *table* (Vosteen & Schellschmidt, 2003: ``k(T) = k0 / (0.99 + T (a - b/k0))``,
tabulated at 16 temperatures and read with a Gaussian kernel so it is smooth in ``T``).
The residual density is therefore code, not a form:

    k(m, u, x) = exp(m) * f_{lith(x)}(T_SCALE u),    r = k grad u . D grad p - q(x) p

with a basal heat flux as a boundary density on the bottom face and the surface
temperature as a Dirichlet condition on the top face.
"""

import numpy as np
import jax.numpy as jnp
import mfem.par as mfem

import hippymfem as hm
from hippymfem.modeling.variables import STATE, PARAMETER, ADJOINT

L_HORIZ = 8000.0            # m
H_DEPTH = 4000.0            # m
T_SCALE = 100.0             # K per unit of u
K_REF = 2.5                 # W/(m K); m = log(k0 / K_REF)
ASPECT2 = (L_HORIZ / H_DEPTH) ** 2

#: (name, k0 [W/(m K)], a, b) per lithology; a, b from Vosteen & Schellschmidt (2003)
LITHOLOGIES = [("sediments", 2.2, 0.0034, 0.0039),
               ("carbonates", 2.9, 0.0034, 0.0039),
               ("basement", 3.3, 0.0030, 0.0042)]
# Scaling of the sources.  Dividing the physical weak form by H k_ref T_SCALE, a
# volumetric heat production A [W/m^3] becomes A L^2 / (k_ref T_SCALE) and a basal
# flux q [W/m^2] becomes q L^2 / (H k_ref T_SCALE); the vertical gradient in the unit
# cube is then Q_BASAL / (k ASPECT2), i.e. 1.05 (105 K over 4 km) at k = k_ref.
#: radiogenic heat production per lithology: 2.0, 1.2, 3.1 microW/m^3
Q_RAD = [0.5, 0.3, 0.8]
#: basal heat flux of 66 mW/m^2 (a continental value), in the scaled units above
Q_BASAL = 4.2
#: the BiLaplacian prior: gamma, delta, and the vertical anisotropy of Theta
# Marginal std of log k about 0.5, a horizontal correlation length sqrt(gamma/delta) of
# 0.25 (2 km) and a vertical one of sqrt(theta_z) times that (0.5 km).  Measured on the
# 16^3 mesh by Monte Carlo (48 samples): std 0.495, max 0.78.  The first calibration,
# gamma 0.004 / delta 1, gave a std of 16 (exp(60) contrasts and a forward solve that
# could not converge); the variance scales as 1/(gamma^1.5 delta^0.5) at fixed Theta.
PRIOR = {"gamma": 0.3, "delta": 4.8, "theta_z": 0.25, "robin_bc": True}
#: the buried anomaly of the synthetic truth: centre, radius, amplitude in log k
# Centred at 2.3 km depth, inside the logged interval.  With a Dirichlet top and a flux
# bottom the temperature signature of a conductive body lives at and below the body
# (above it the field is set by the resistance to the surface), so boreholes have to
# reach the body's depth to see it: at 2.6 km depth under 2 km logs the first 32^3 run
# recovered nothing (MAP -0.06 against a truth of 0.93 in the box).
ANOMALY = {"centre": (0.62, 0.48, 0.42), "radius": 0.12, "amplitude": 1.0}
#: bottom of the boreholes in unit coordinates (2.8 km): deep exploration wells
BOREHOLE_BOTTOM = 0.3


def conductivity_tables(n_nodes=16, T_max=300.0):
    """Temperature nodes (deg C above surface) and the ratio ``k(T)/K_REF`` per lithology."""
    T = np.linspace(0.0, T_max, n_nodes)
    tabs = []
    for _, k0, a, b in LITHOLOGIES:
        tabs.append(k0 / (0.99 + T * (a - b / k0)) / K_REF)
    return T, np.array(tabs)


def lithology_index(x):
    """0 sediments (top), 1 carbonates, 2 basement, with two dipping interfaces (JAX)."""
    z1 = 0.72 + 0.06 * jnp.sin(2.0 * jnp.pi * x[0]) + 0.03 * jnp.cos(2.0 * jnp.pi * x[1])
    z2 = 0.42 + 0.05 * jnp.cos(2.0 * jnp.pi * x[0] + 1.0)
    return jnp.where(x[2] > z1, 0, jnp.where(x[2] > z2, 1, 2))


def lithology_index_np(x):
    """The same map for numpy points ``(n, 3)``."""
    z1 = 0.72 + 0.06 * np.sin(2.0 * np.pi * x[:, 0]) + 0.03 * np.cos(2.0 * np.pi * x[:, 1])
    z2 = 0.42 + 0.05 * np.cos(2.0 * np.pi * x[:, 0] + 1.0)
    return np.where(x[:, 2] > z1, 0, np.where(x[:, 2] > z2, 1, 2))


def make_densities(nodes, tabs, kernel_width=None):
    """The volume and boundary residual densities."""
    nodes_j = jnp.asarray(nodes)
    tabs_j = jnp.asarray(tabs)
    q_j = jnp.asarray(Q_RAD)
    w = kernel_width if kernel_width is not None else 1.2 * (nodes[1] - nodes[0])
    D = jnp.array([1.0, 1.0, ASPECT2])

    def varf(u, m, p, x):
        li = lithology_index(x)
        T = T_SCALE * u.val
        wts = jnp.exp(-0.5 * ((T - nodes_j) / w) ** 2)
        f = jnp.sum(wts * jnp.take(tabs_j, li, axis=0)) / jnp.sum(wts)
        k = jnp.exp(m.val) * f
        return k * jnp.dot(u.grad * D, p.grad) - jnp.take(q_j, li) * p.val

    def bdr_varf(u, m, p, x, n):
        # -k grad u . n_out = q_basal on the bottom face: heat enters from below
        return -Q_BASAL * p.val

    return varf, bdr_varf


class Geothermal:
    """Everything the workflow needs, built once."""

    def __init__(self, n, comm, order=2, nboreholes=60, noise_kelvin=0.5, seed=1,
                 quadrature_degree=None):
        self.comm = comm
        self.n = n
        self.pmesh = mfem.ParMesh(comm, mfem.Mesh.MakeCartesian3D(n, n, n, mfem.Element.HEXAHEDRON))
        self.Vu = hm.FunctionSpace.H1(self.pmesh, order)
        self.Vm = hm.FunctionSpace.H1(self.pmesh, 1)
        self.Vh = [self.Vu, self.Vm, self.Vu]
        self.nodes, self.tabs = conductivity_tables()
        varf, bdr = make_densities(self.nodes, self.tabs)
        # attribute 6 is the top (z = 1) of MakeCartesian3D, attribute 1 the bottom
        self.bc = hm.DirichletBC(self.Vu, None, bdr_attributes=[6])
        self.pde = hm.PDEVariationalProblem(self.Vh, varf, self.bc, self.bc.homogeneous(),
                                            is_fwd_linear=False, bdr_varf=bdr, bdr_attributes=[1],
                                            quadrature_degree=quadrature_degree,
                                            symmetric_jacobian=False, transpose_free_adjoint=True,
                                            release_linearization_on_move=True)
        # dR/du carries k'(u) du grad u . grad p, so the Jacobian is *not* symmetric:
        # GMRES rather than CG, and the adjoint uses the true transpose.  It is applied
        # through A's MultTranspose with A's AMG hierarchy as the preconditioner
        # (transpose_free_adjoint): no transposed copy and no second hierarchy, which
        # is what decides whether the largest cases fit the cards.
        self.pde.newton_parameters["max_iter"] = 30
        self.pde.newton_parameters["rel_tolerance"] = 1e-9
        # the three solves the records were taken with; the adjoint keeps its default
        self.pde.set_solvers(hm.auto_solver, self.Vu, comm, max_direct=0, method="gmres",
                             rel_tolerance=1e-10, max_iter=3000,
                             attributes=("solver", "solver_fwd_inc", "solver_adj_inc"))
        Theta = np.diag([1.0, 1.0, PRIOR["theta_z"]])
        self.prior = hm.BiLaplacianPrior(self.Vm, PRIOR["gamma"], PRIOR["delta"], Theta=Theta,
                                         robin_bc=PRIOR["robin_bc"], solver_type="krylov")
        for name in ("Asolver", "Msolver"):
            sol = getattr(self.prior, name, None)
            if sol is not None and hasattr(sol, "parameters"):
                sol.parameters["rel_tolerance"] = 1e-8
        # the synthetic truth: a prior sample plus a buried high-conductivity body
        hm.parRandom.set_seed(seed)
        noise = self.prior.noise_vector()
        self.prior.sample_noise(1.0, noise)
        self.mtrue = self.Vm.vector()
        self.prior.sample(noise, self.mtrue)
        c, r, a = ANOMALY["centre"], ANOMALY["radius"], ANOMALY["amplitude"]
        bump = self.Vm.project(lambda z: a * np.exp(-0.5 * (((z[0] - c[0]) ** 2 + (z[1] - c[1]) ** 2) / r ** 2
                                                            + ((z[2] - c[2]) / (0.6 * r)) ** 2)))
        self.mtrue.axpy(1.0, bump)
        # boreholes: vertical logs from the surface to BOREHOLE_BOTTOM, one sample per element layer
        rng = np.random.default_rng(seed + 10)
        xy = rng.uniform(0.12, 0.88, (nboreholes, 2))
        nz = int(round((1.0 - BOREHOLE_BOTTOM) * n))
        zs = 1.0 - (np.arange(nz) + 0.5) / n
        self.targets = np.array([[x, y, z] for x, y in xy for z in zs])
        self.B = hm.assemblePointwiseObservation(self.Vu, self.targets)
        utrue = self.pde.generate_state()
        self.pde.solveFwd(utrue, [utrue, self.mtrue, None])
        self.utrue = utrue
        data = self.B.createVecLeft()
        self.B.mult(utrue, data)
        self.noise_std = noise_kelvin / T_SCALE
        hm.parRandom.set_seed(seed + 20)
        self.B.perturb(data, self.noise_std)
        self.data = data
        self.misfit = hm.DiscreteStateObservation(self.B, data, self.noise_std ** 2)
        self.model = hm.Model(self.pde, self.prior, self.misfit)

    def summary(self):
        return {"n": self.n, "state_dofs": int(self.Vu.GlobalTrueVSize()), "param_dofs": int(self.Vm.GlobalTrueVSize()),
                "observations": int(self.targets.shape[0]), "boreholes": int(self.targets.shape[0] // max(1, int(round((1.0 - BOREHOLE_BOTTOM) * self.n)))),
                "noise_kelvin": self.noise_std * T_SCALE, "u_true_max_kelvin": float(self.utrue.norm("linf")) * T_SCALE,
                "forward_newton_iterations": int(getattr(self.pde, "fwd_iterations", -1)),
                "prior": dict(PRIOR), "anomaly": dict(ANOMALY), "lithologies": [l[0] for l in LITHOLOGIES]}
