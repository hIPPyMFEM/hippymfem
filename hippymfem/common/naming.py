# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Both spellings of the programming model's method names.

hIPPYlib spells its methods in camelCase (``solveFwd``, ``setLinearizationPoint``,
``applyWuu``); this library's own names are snake_case (``generate_state``,
``set_operator``, ``norm``).  Every class of the programming model answers to both:
a class that defines either spelling of a name in :data:`PAIRS` gets the other as an
alias, and a subclass that overrides either is reached through both.  The library
itself calls the camelCase names, so an object that speaks only hIPPYlib keeps
working without inheriting from anything here; the snake_case spelling is the one
the documentation prefers.
"""

#: ``(snake_case, camelCase)`` for every method name that has both spellings
PAIRS = (
    ("solve_fwd", "solveFwd"),
    ("solve_adj", "solveAdj"),
    ("eval_gradient_parameter", "evalGradientParameter"),
    ("set_linearization_point", "setLinearizationPoint"),
    ("set_point_for_hessian_evaluations", "setPointForHessianEvaluations"),
    ("solve_incremental", "solveIncremental"),
    ("solve_fwd_incremental", "solveFwdIncremental"),
    ("solve_adj_incremental", "solveAdjIncremental"),
    ("apply_c", "applyC"),
    ("apply_ct", "applyCt"),
    ("apply_wuu", "applyWuu"),
    ("apply_wum", "applyWum"),
    ("apply_wmu", "applyWmu"),
    ("apply_wmm", "applyWmm"),
    ("apply_r", "applyR"),
    ("mult_transpose", "multTranspose"),
    ("create_vec_left", "createVecLeft"),
    ("create_vec_right", "createVecRight"),
    ("get_size", "getSize"),
    ("get_comm", "getComm"),
    ("get_local_size", "getLocalSize"),
    ("set_size_from_vector", "setSizeFromVector"),
    ("get_hessian_preconditioner", "getHessianPreconditioner"),
    ("derivative_info", "derivativeInfo"),
    ("show_me", "showMe"),
    ("export_state", "exportState"),
    ("kl_distance_from_prior", "klDistanceFromPrior"),
    ("compute_low_rank_factorization", "computeLowRankFactorization"),
    ("expected_value", "expectedValue"),
)
CAMEL_OF = dict(PAIRS)
SNAKE_OF = {camel: snake for snake, camel in PAIRS}


def sync_spellings(cls):
    """Give ``cls`` both spellings of every name in :data:`PAIRS` it defines itself.

    Usable as a class decorator.  Only the class's own namespace counts, so a
    subclass that overrides one spelling has the other point at its override.
    """
    own = cls.__dict__
    for snake, camel in PAIRS:
        if snake in own and camel not in own:
            setattr(cls, camel, own[snake])
        elif camel in own and snake not in own:
            setattr(cls, snake, own[camel])
    return cls


class SnakeCamel(object):
    """Mixin: every subclass gets both spellings of the names it defines."""

    __slots__ = ()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        sync_spellings(cls)
