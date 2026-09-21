"""
Multi-resolution momentum-transfer force operator for the Neon backend.
"""

from typing import Any

import warp as wp

from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator
from xlb.operator.force import MomentumTransfer
from xlb.mres_perf_optimization_type import MresPerfOptimizationType


class MultiresMomentumTransfer(MomentumTransfer):
    """Momentum-transfer force computation on a multi-resolution grid.

    Extends :class:`MomentumTransfer` with Neon-specific container code that
    iterates over all grid levels.  The LBM operation sequence (collide-then-
    stream vs. stream-then-collide) is inferred from the performance
    optimization type.

    Parameters
    ----------
    no_slip_bc_instance : BoundaryCondition
        The no-slip BC whose tagged voxels define the force integration
        surface.
    mres_perf_opt : MresPerfOptimizationType
        Multi-resolution performance strategy.
    velocity_set : VelocitySet, optional
    precision_policy : PrecisionPolicy, optional
    compute_backend : ComputeBackend, optional
    force_levels : sequence of int, optional
        Grid levels to integrate the force over.  ``None`` (the default) uses
        every level, which is the historical behaviour.  Because this operator
        already requires all boundary voxels of the BC to sit on one level,
        passing e.g. ``[0]`` for a body voxelized only at the finest level
        gives the same force and skips the other levels entirely.
    """

    def __init__(
        self,
        no_slip_bc_instance,
        mres_perf_opt=MresPerfOptimizationType.NAIVE_COLLIDE_STREAM,
        velocity_set: VelocitySet = None,
        precision_policy: PrecisionPolicy = None,
        compute_backend: ComputeBackend = None,
        force_levels=None,
    ):
        from xlb.operator.force.momentum_transfer import LBMOperationSequence

        if compute_backend in [ComputeBackend.JAX, ComputeBackend.WARP]:
            raise NotImplementedError(f"Operator {self.__class__.__name__} not supported in {compute_backend} backend.")

        # Set the sequence of operations based on the performance optimization type
        if mres_perf_opt == MresPerfOptimizationType.NAIVE_COLLIDE_STREAM:
            operation_sequence = LBMOperationSequence.COLLIDE_THEN_STREAM
        elif mres_perf_opt in (
            MresPerfOptimizationType.FUSION_AT_FINEST,
            MresPerfOptimizationType.FUSION_AT_FINEST_SFV,
            MresPerfOptimizationType.FUSION_AT_FINEST_SFV_ALL,
        ):
            operation_sequence = LBMOperationSequence.STREAM_THEN_COLLIDE
        else:
            raise ValueError(f"Unknown performance optimization type: {mres_perf_opt}")

        # Check if the performance optimization type is compatible with the use of mesh distance
        if operation_sequence != LBMOperationSequence.STREAM_THEN_COLLIDE:
            assert not no_slip_bc_instance.needs_mesh_distance, (
                "Mesh distance is only supported in the MultiresMomentumTransfer operator when the LBM operation sequence is STREAM_THEN_COLLIDE."
            )

        # Print a warning to the user about the boundary voxels
        print(
            "WARNING! make sure boundary voxels are all at the same level and not among the transition regions from one level to another. "
            "Otherwise, the results of force calculation are not correct!\n"
        )

        # Levels to integrate the force over. None means every level.
        self.force_levels = force_levels
        # Built Neon containers, keyed by (level, field-handle identities).
        self._container_cache = {}

        # Call super
        super().__init__(no_slip_bc_instance, operation_sequence, velocity_set, precision_policy, compute_backend)

    def _construct_neon(self):
        import neon

        # Use the warp functional for the NEON backend
        functional, _ = self._construct_warp()

        @neon.Container.factory(name="MomentumTransfer")
        def container(
            f_0: Any,
            f_1: Any,
            bc_mask: Any,
            missing_mask: Any,
            force: Any,
            _rho: Any,
            _u: Any,
            _relax: Any,           
            _norm_vec: Any,
            _norm_dist: Any,
            level: Any,
        ):
            def container_launcher(loader: neon.Loader):
                loader.set_mres_grid(bc_mask.get_grid(), level)
                bc_mask_pn = loader.get_mres_write_handle(bc_mask)
                missing_mask_pn = loader.get_mres_write_handle(missing_mask)
                f_0_pn = loader.get_mres_write_handle(f_0)
                f_1_pn = loader.get_mres_write_handle(f_1)
                _rho0_pn = loader.get_mres_read_handle(_rho)
                _u0_pn = loader.get_mres_read_handle(_u)
                _relax_pn = loader.get_mres_read_handle(_relax)
                _norm_vec_pn = loader.get_mres_read_handle(_norm_vec)
                _norm_dist_pn = loader.get_mres_read_handle(_norm_dist)


                @wp.func
                def container_kernel(index: Any):
                    # apply the functional
                    functional(
                        index,
                        f_0_pn,
                        f_1_pn,
                        bc_mask_pn,
                        missing_mask_pn,
                        force,
                        _rho0_pn,
                        _u0_pn,
                        _relax_pn,                        
                        _norm_vec_pn,
                        _norm_dist_pn
                    )

                loader.declare_kernel(container_kernel)

            return container_launcher

        return functional, container

    @Operator.register_backend(ComputeBackend.NEON)
    def neon_implementation(
        self,
        f_0,
        f_1,
        bc_mask,
        missing_mask,
        _rho=None,
        _u=None,
        _relax=None,
        _norm_vec_pn=None,
        _norm_dist_pn=None,
        stream=0,
        levels=None,
    ):
        import neon
        if _rho is None or _u is None:
            raise TypeError("rho and u must be provided: momentum_transfer(f_0, f_1, bc_mask, missing_mask, rho, u)")

        # Ensure the force is initialized to zero
        self.force *= self.compute_dtype(0.0)

        # Define the neon functionals needed for this operation
        self.fetcher_functional = self.fetcher.neon_functional

        grid = bc_mask.get_grid()

        # Which levels to integrate over. The operator already requires that all
        # boundary voxels of this BC sit on a single level (see the warning in
        # __init__), so integrating over levels that hold none of them is pure
        # overhead. `levels=None` keeps the original all-levels behaviour.
        if levels is None:
            levels = self.force_levels
        if levels is None:
            levels = range(grid.num_levels)

        # Cache the Neon containers. Building one re-traces and re-hashes the
        # Warp kernel, so the original code paid that cost on every call, for
        # every level, for every body. A container is only valid for the exact
        # field handles it was built with, so the cache key carries their
        # identities and a container is rebuilt if any handle is replaced.
        key_fields = (
            id(f_0),
            id(f_1),
            id(bc_mask),
            id(missing_mask),
            id(self.force),
            id(_rho),
            id(_u),
            id(_relax),
            id(_norm_vec_pn),
            id(_norm_dist_pn),
        )
        for level in levels:
            key = (level, key_fields)
            c = self._container_cache.get(key)
            if c is None:
                c = self.neon_container(
                    f_0, f_1, bc_mask, missing_mask, self.force, _rho, _u, _relax, _norm_vec_pn, _norm_dist_pn, level
                )
                self._container_cache[key] = c
            # Launch the neon container
            c.run(stream, container_runtime=neon.Container.ContainerRuntime.neon)
        return self.force.numpy()[0]
