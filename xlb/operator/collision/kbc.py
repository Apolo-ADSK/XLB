import jax.numpy as jnp
from jax import jit
import warp as wp
from typing import Any
from functools import partial

from xlb.velocity_set import VelocitySet, D2Q9, D3Q27
from xlb.compute_backend import ComputeBackend
from xlb.operator.collision.collision import Collision
from xlb.operator import Operator
from xlb.operator.macroscopic import SecondMoment as MomentumFlux


class KBC(Collision):
    """
    KBC collision operator for LBM.

    This class implements the Karlin-Bösch-Chikatamarla (KBC) model for the collision step in the Lattice Boltzmann Method,
    optimized with fused scalar products to reduce redundant division operations.
    """

    def __init__(
        self,
        velocity_set: VelocitySet = None,
        precision_policy=None,
        compute_backend=None,
    ):
        """Initialize the KBC collision operator."""
        self.momentum_flux = MomentumFlux()
        self.epsilon = 1e-32  # Small constant to prevent division by zero

        super().__init__(
            velocity_set=velocity_set,
            precision_policy=precision_policy,
            compute_backend=compute_backend,
        )

    ### JAX Backend Implementation ###

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0,), donate_argnums=(1, 2, 3))
    def jax_implementation(
        self,
        f: jnp.ndarray,
        feq: jnp.ndarray,
        rho: jnp.ndarray,
        u: jnp.ndarray,
        omega,
    ):
        """
        JAX implementation of the KBC collision step with fused scalar products.

        Parameters
        ----------
        f : jax.numpy.ndarray
            Distribution function.
        feq : jax.numpy.ndarray
            Equilibrium distribution function.
        rho : jax.numpy.ndarray
            Density.
        u : jax.numpy.ndarray
            Velocity.
        omega : float
            Relaxation parameter (inverse relaxation time).

        Returns
        -------
        jax.numpy.ndarray
            Post-collision distribution function.
        """
        fneq = f - feq
        if isinstance(self.velocity_set, D2Q9):
            shear = self.decompose_shear_d2q9_jax(fneq)
            delta_s = shear * rho / 4.0
        elif isinstance(self.velocity_set, D3Q27):
            shear = self.decompose_shear_d3q27_jax(fneq)
            delta_s = shear * rho
        else:
            raise NotImplementedError(f"Velocity set not supported: {type(self.velocity_set)}")

        # Compute constants
        beta = self.compute_dtype(0.5) * self.compute_dtype(omega)
        inv_beta = 1.0 / beta

        # Compute fused scalar products and perform collision
        delta_h = fneq - delta_s
        sp1, sp2 = self.compute_scalar_products_jax(delta_s, delta_h, feq)
        gamma = inv_beta - (2.0 - inv_beta) * sp1 / (self.epsilon + sp2)
        fout = f - beta * (2.0 * delta_s + gamma[None, ...] * delta_h)

        return fout

    @partial(jit, static_argnums=(0,), inline=True)
    def compute_scalar_products_jax(self, delta_s, delta_h, feq):
        """
        Compute fused entropic scalar products for JAX backend.

        Reuses the term `delta_h / feq` to compute both scalar products efficiently.

        Parameters
        ----------
        delta_s : jax.numpy.ndarray
            Shear component of the non-equilibrium distribution.
        delta_h : jax.numpy.ndarray
            Higher-order component of the non-equilibrium distribution.
        feq : jax.numpy.ndarray
            Equilibrium distribution function.

        Returns
        -------
        tuple
            (sp1, sp2) where:
            - sp1 = sum(delta_s * delta_h / feq)
            - sp2 = sum(delta_h * delta_h / feq)
        """
        temp = delta_h / feq
        sp1 = jnp.sum(delta_s * temp, axis=0)
        sp2 = jnp.sum(delta_h * temp, axis=0)
        return sp1, sp2

    @partial(jit, static_argnums=(0,), inline=True)
    def decompose_shear_d3q27_jax(self, fneq):
        """
        Decompose the non-equilibrium distribution into shear components for D3Q27.

        Parameters
        ----------
        fneq : jax.numpy.ndarray
            Non-equilibrium distribution function.

        Returns
        -------
        jax.numpy.ndarray
            Shear components.
        """
        Pi = self.momentum_flux(fneq)
        Nxz = Pi[0, ...] - Pi[5, ...]
        Nyz = Pi[3, ...] - Pi[5, ...]
        s = jnp.zeros_like(fneq)
        s = s.at[9, ...].set((2.0 * Nxz - Nyz) / 6.0)
        s = s.at[18, ...].set((2.0 * Nxz - Nyz) / 6.0)
        s = s.at[3, ...].set((-Nxz + 2.0 * Nyz) / 6.0)
        s = s.at[6, ...].set((-Nxz + 2.0 * Nyz) / 6.0)
        s = s.at[1, ...].set((-Nxz - Nyz) / 6.0)
        s = s.at[2, ...].set((-Nxz - Nyz) / 6.0)
        s = s.at[12, ...].set(Pi[1, ...] / 4.0)
        s = s.at[24, ...].set(Pi[1, ...] / 4.0)
        s = s.at[21, ...].set(-Pi[1, ...] / 4.0)
        s = s.at[15, ...].set(-Pi[1, ...] / 4.0)
        s = s.at[10, ...].set(Pi[2, ...] / 4.0)
        s = s.at[20, ...].set(Pi[2, ...] / 4.0)
        s = s.at[19, ...].set(-Pi[2, ...] / 4.0)
        s = s.at[11, ...].set(-Pi[2, ...] / 4.0)
        s = s.at[8, ...].set(Pi[4, ...] / 4.0)
        s = s.at[4, ...].set(Pi[4, ...] / 4.0)
        s = s.at[7, ...].set(-Pi[4, ...] / 4.0)
        s = s.at[5, ...].set(-Pi[4, ...] / 4.0)
        return s

    @partial(jit, static_argnums=(0,), inline=True)
    def decompose_shear_d2q9_jax(self, fneq):
        """
        Decompose the non-equilibrium distribution into shear components for D2Q9.

        Parameters
        ----------
        fneq : jax.numpy.ndarray
            Non-equilibrium distribution function.

        Returns
        -------
        jax.numpy.ndarray
            Shear components.
        """
        Pi = self.momentum_flux(fneq)
        N = Pi[0, ...] - Pi[2, ...]
        s = jnp.zeros_like(fneq)
        s = s.at[3, ...].set(N)
        s = s.at[6, ...].set(N)
        s = s.at[2, ...].set(-N)
        s = s.at[1, ...].set(-N)
        s = s.at[8, ...].set(Pi[1, ...])
        s = s.at[4, ...].set(-Pi[1, ...])
        s = s.at[5, ...].set(-Pi[1, ...])
        s = s.at[7, ...].set(Pi[1, ...])
        return s

    ### Warp Backend Implementation ###

    def _construct_warp(self):
        """Construct Warp functionals and kernel for the KBC collision step."""
        if not (isinstance(self.velocity_set, D3Q27) or isinstance(self.velocity_set, D2Q9)):
            raise NotImplementedError(f"Velocity set not supported for Warp backend: {type(self.velocity_set)}")

        # Define Warp types and constants
        _u_vec = wp.vec(self.velocity_set.d, dtype=self.compute_dtype)
        _f_vec = wp.vec(self.velocity_set.q, dtype=self.compute_dtype)
        _epsilon = wp.constant(self.compute_dtype(self.epsilon))
        _two = wp.constant(self.compute_dtype(2.0))

        @wp.func
        def decompose_shear_d2q9(fneq: Any):
            """Decompose shear components for D2Q9 in Warp."""
            pi = self.momentum_flux.warp_functional(fneq)
            N = pi[0] - pi[2]
            s = _f_vec()
            s[3] = N
            s[6] = N
            s[2] = -N
            s[1] = -N
            s[8] = pi[1]
            s[4] = -pi[1]
            s[5] = -pi[1]
            s[7] = pi[1]
            return s

        @wp.func
        def decompose_shear_d3q27(fneq: Any):
            """Decompose shear components for D3Q27 in Warp."""
            pi = self.momentum_flux.warp_functional(fneq)
            nxz = pi[0] - pi[5]
            nyz = pi[3] - pi[5]
            s = _f_vec()
            s[9] = (_two * nxz - nyz) / 6.0
            s[18] = (_two * nxz - nyz) / 6.0
            s[3] = (-nxz + _two * nyz) / 6.0
            s[6] = (-nxz + _two * nyz) / 6.0
            s[1] = (-nxz - nyz) / 6.0
            s[2] = (-nxz - nyz) / 6.0
            s[12] = pi[1] / 4.0
            s[24] = pi[1] / 4.0
            s[21] = -pi[1] / 4.0
            s[15] = -pi[1] / 4.0
            s[10] = pi[2] / 4.0
            s[20] = pi[2] / 4.0
            s[19] = -pi[2] / 4.0
            s[11] = -pi[2] / 4.0
            s[8] = pi[4] / 4.0
            s[4] = pi[4] / 4.0
            s[7] = -pi[4] / 4.0
            s[5] = -pi[4] / 4.0
            return s

        @wp.func
        def compute_scalar_products(delta_s: Any, delta_h: Any, feq: Any):
            """
            Compute fused entropic scalar products for Warp backend.

            Reuses `delta_h[i] / feq[i]` to compute both scalar products in a single loop.

            Parameters
            ----------
            delta_s : Warp vector
                Shear component.
            delta_h : Warp vector
                Higher-order component.
            feq : Warp vector
                Equilibrium distribution function.

            Returns
            -------
            tuple
                (sp1, sp2) where:
                - sp1 = sum(delta_s * delta_h / feq)
                - sp2 = sum(delta_h * delta_h / feq)
            """
            s1 = self.compute_dtype(0.0)  # Sum for sp1
            c1 = self.compute_dtype(0.0)  # Correction for sp1
            s2 = self.compute_dtype(0.0) # Sum for sp2
            c2 = self.compute_dtype(0.0)  # Correction for sp2
            for i in range(self.velocity_set.q):
                temp = delta_h[i] / feq[i]
                x1 = delta_s[i] * temp
                t1 = s1 + x1
                if abs(s1) >= abs(x1):
                    c1 += (s1 - t1) + x1
                else:
                    c1 += (x1 - t1) + s1
                

                x2 = delta_h[i] * temp
                t2 = s2 + x2
                if abs(s2) >= abs(x2):
                    c2 += (s2 - t2) + x2
                else:
                    c2 += (x2 - t2) + s2
                
            sp1 = t1 + c1
            sp2 = t2 + c2
            return sp1, sp2

        @wp.func
        def functional(
            f: Any,
            feq: Any,
            rho: Any,
            u: Any,
            omega: Any,
        ):
            """Warp functional for KBC collision with fused scalar products."""
            fneq = f - feq
            if wp.static(self.velocity_set.d == 3):
                shear = decompose_shear_d3q27(fneq)
                delta_s = shear * rho
            else:
                shear = decompose_shear_d2q9(fneq)
                delta_s = shear * rho / 4.0

            _beta = self.compute_dtype(0.5) * self.compute_dtype(omega)
            _inv_beta = self.compute_dtype(1.0) / _beta
            delta_h = fneq - delta_s
            sp1, sp2 = compute_scalar_products(delta_s, delta_h, feq)
            gamma = _inv_beta - (_two - _inv_beta) * sp1 / (_epsilon + sp2)
            fout = f - _beta * (_two * delta_s + gamma * delta_h)
            return fout

        @wp.kernel
        def kernel(
            f: wp.array4d(dtype=Any),
            feq: wp.array4d(dtype=Any),
            fout: wp.array4d(dtype=Any),
            rho: wp.array4d(dtype=Any),
            u: wp.array4d(dtype=Any),
            omega: Any,
        ):
            """Warp kernel to launch the KBC collision step."""
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)
            _f = _f_vec()
            _feq = _f_vec()
            for l in range(self.velocity_set.q):
                _f[l] = f[l, index[0], index[1], index[2]]
                _feq[l] = feq[l, index[0], index[1], index[2]]
            _u = _u_vec()
            for l in range(self.velocity_set.d):
                _u[l] = u[l, index[0], index[1], index[2]]
            _rho = rho[0, index[0], index[1], index[2]]
            _fout = functional(_f, _feq, _rho, _u, omega)
            for l in range(self.velocity_set.q):
                fout[l, index[0], index[1], index[2]] = self.store_dtype(_fout[l])

        return functional, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f, feq, fout, rho, u, omega):
        """
        Warp implementation of the KBC collision step.

        Parameters
        ----------
        f, feq, fout, rho, u, omega : Warp arrays and scalar
            Inputs and output for the collision step.

        Returns
        -------
        Warp array
            Post-collision distribution function.
        """
        wp.launch(
            self.warp_kernel,
            inputs=[f, feq, fout, rho, u, omega],
            dim=f.shape[1:],
        )
        return fout