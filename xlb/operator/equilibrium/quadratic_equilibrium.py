from functools import partial
import jax.numpy as jnp
from jax import jit
import warp as wp
from typing import Any

from xlb.compute_backend import ComputeBackend
from xlb.operator.equilibrium.equilibrium import Equilibrium
from xlb.operator import Operator


class QuadraticEquilibrium(Equilibrium):
    """
    Quadratic equilibrium of Boltzmann equation using hermite polynomials.
    Standard equilibrium model for LBM.
    """

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0))
    def jax_implementation(self, rho, u):
        cu = 3.0 * jnp.tensordot(self.velocity_set.c, u, axes=(0, 0))
        usqr = 1.5 * jnp.sum(jnp.square(u), axis=0, keepdims=True)
        w = self.velocity_set.w.reshape((-1,) + (1,) * (len(rho.shape) - 1))
        feq = rho * w * (1.0 + cu * (1.0 + 0.5 * cu) - usqr)
        return feq

    def _construct_warp(self):
        # Set local constants TODO: This is a hack and should be fixed with warp update
        _c = self.velocity_set.c
        _w = self.velocity_set.w
        _f_vec = wp.vec(self.velocity_set.q, dtype=self.compute_dtype)
        _u_vec = wp.vec(self.velocity_set.d, dtype=self.compute_dtype)

        # Construct the equilibrium functional
        @wp.func
        def functional(
            rho: Any,
            u: Any,
        ):
            feq = _f_vec()
            zero = self.compute_dtype(0.0)
            half = self.compute_dtype(0.5)
            one = self.compute_dtype(1.0)
            one_half = self.compute_dtype(1.5)
            three = self.compute_dtype(3.0)

            for l in range(self.velocity_set.q):
                # Use wp.vec3 for D3 (adjust to wp.vec2 or wp.vec4 based on your velocity_set.d)
                cu_vec = wp.vec3(zero, zero, zero)
                comp_vec = wp.vec3(zero, zero, zero)

                # Precompute the terms into a vector to avoid conditionals in the loop
                x_vec = wp.vec3(zero, zero, zero)
                for d in range(self.velocity_set.d):
                    if _c[d, l] == 1:
                        x_vec[d] = u[d]
                    elif _c[d, l] == -1:
                        x_vec[d] = -u[d]
                    else:
                        x_vec[d] = zero

                # SIMD Kahan summation (single step for all components)
                temp_vec = cu_vec + x_vec
                # Compute compensation component-wise without direct vector comparison
                for d in range(self.velocity_set.d):
                    cu_d = wp.extract(cu_vec, d)
                    x_d = wp.extract(x_vec, d)
                    temp_d = wp.extract(temp_vec, d)
                    if wp.abs(cu_d) >= wp.abs(x_d):
                        comp_vec[d] = wp.extract(comp_vec, d) + ((cu_d - temp_d) + x_d)
                    else:
                        comp_vec[d] = wp.extract(comp_vec, d) + ((x_d - temp_d) + cu_d)
                cu_vec = temp_vec

                # Apply correction
                cu_vec = cu_vec + comp_vec

                # Extract scalar cu
                cu = zero
                for d in range(self.velocity_set.d):
                    cu += wp.extract(cu_vec, d)
                cu *= three

                usqr = one_half * wp.dot(u, u)
                feq[l] = rho * _w[l] * (
                    one
                    + cu * (one + half * cu)
                    - usqr
                )

            return feq
        
        # Construct the warp kernel
        @wp.kernel
        def kernel(
            rho: wp.array4d(dtype=Any),
            u: wp.array4d(dtype=Any),
            f: wp.array4d(dtype=Any),
        ):
            # Get the global index
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)

            # Get the equilibrium
            _u = _u_vec()
            for d in range(self.velocity_set.d):
                _u[d] = u[d, index[0], index[1], index[2]]
            _rho = rho[0, index[0], index[1], index[2]]
            feq = functional(_rho, _u)

            # Set the output
            for l in range(self.velocity_set.q):
                f[l, index[0], index[1], index[2]] = self.store_dtype(feq[l])

        return functional, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, rho, u, f):
        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[
                rho,
                u,
                f,
            ],
            dim=rho.shape[1:],
        )
        return f
