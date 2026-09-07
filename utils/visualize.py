from pathlib import Path

import numpy as np

from .backend import _asnumpy


def _to_cpu_mf(mf):
    return mf.to_cpu() if hasattr(mf, "to_cpu") else mf


def _system_from_method(method):
    mf = _to_cpu_mf(method.mf)
    if hasattr(mf, "mol"):
        return mf.mol
    if hasattr(mf, "cell"):
        return mf.cell
    raise AttributeError("method.mf must provide either mol or cell for cube export")


def _default_cubegen_orbital(system):
    if hasattr(system, "lattice_vectors"):
        try:
            from pyscf.pbc.tools import cubegen
        except ImportError:
            from pyscf.tools import cubegen
    else:
        from pyscf.tools import cubegen
    return cubegen.orbital


def _restricted_mo_coeff(method):
    ctx = method.ctx
    mo_coeff = _asnumpy(ctx.mo_coeff)
    occidx_a = _asnumpy(ctx.occidx_a).astype(int)
    viridx_a = _asnumpy(ctx.viridx_a).astype(int)
    mo_order = np.concatenate([occidx_a, viridx_a])
    return mo_coeff[0][:, mo_order]


def _unrestricted_mo_coeffs(method):
    mo_coeff = _asnumpy(method.ctx.mo_coeff)
    return mo_coeff[0], mo_coeff[1]


def _write_orbital(cubegen_orbital, system, outfile, coeff, cube_kwargs):
    coeff = np.asarray(coeff)
    cubegen_orbital(system, str(outfile), coeff, **cube_kwargs)


def _write_restricted_pair(
    method,
    system,
    outdir,
    prefix,
    pair,
    hole,
    particle,
    cubegen_orbital,
    cube_kwargs,
):
    mo = _restricted_mo_coeff(method)
    hole_file = outdir / f"{prefix}_nto{pair}_hole.cube"
    particle_file = outdir / f"{prefix}_nto{pair}_particle.cube"
    _write_orbital(cubegen_orbital, system, hole_file, mo @ hole, cube_kwargs)
    _write_orbital(cubegen_orbital, system, particle_file, mo @ particle, cube_kwargs)
    return [
        {"pair": pair, "kind": "hole", "path": str(hole_file)},
        {"pair": pair, "kind": "particle", "path": str(particle_file)},
    ]


def _write_unrestricted_component(
    cubegen_orbital,
    system,
    outdir,
    prefix,
    pair,
    kind,
    spin,
    mo,
    coeff,
    cube_kwargs,
):
    outfile = outdir / f"{prefix}_nto{pair}_{kind}_{spin}.cube"
    _write_orbital(cubegen_orbital, system, outfile, mo @ coeff, cube_kwargs)
    return {"pair": pair, "kind": kind, "spin": spin, "path": str(outfile)}


def _write_unrestricted_pair(
    method,
    system,
    outdir,
    prefix,
    pair,
    hole,
    particle,
    cubegen_orbital,
    cube_kwargs,
    component_tol,
):
    mo_a, mo_b = _unrestricted_mo_coeffs(method)
    nmo_a = mo_a.shape[1]
    components = []
    for kind, vector in (("hole", hole), ("particle", particle)):
        alpha = vector[:nmo_a]
        beta = vector[nmo_a:]
        if np.linalg.norm(alpha) > component_tol:
            components.append(
                _write_unrestricted_component(
                    cubegen_orbital, system, outdir, prefix, pair,
                    kind, "alpha", mo_a, alpha, cube_kwargs,
                )
            )
        if np.linalg.norm(beta) > component_tol:
            components.append(
                _write_unrestricted_component(
                    cubegen_orbital, system, outdir, prefix, pair,
                    kind, "beta", mo_b, beta, cube_kwargs,
                )
            )
    return components


def write_nto_cubes(
    method,
    state_f=0,
    state_i=0,
    nroots=1,
    outdir=".",
    prefix=None,
    cubegen_orbital=None,
    component_tol=1.0e-12,
    **cube_kwargs,
):
    """Write NTO hole/particle orbitals as PySCF cube files.

    PySCF provides cube output through ``pyscf.tools.cubegen.orbital`` for
    molecular objects, and the same helper is used here by default.  For
    unrestricted references, spin-MO NTOs are split into alpha and beta spatial
    components because the spin parts are orthogonal and should not be merged
    into a single scalar orbital.  For XTDA ground-to-excited NTOs, pass
    ``state_i=None`` to denote the reference determinant.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if prefix is None:
        label_f = "ground" if state_f is None else state_f
        label_i = "ground" if state_i is None else state_i
        prefix = f"stateF{label_f}_stateI{label_i}"

    system = _system_from_method(method)
    if cubegen_orbital is None:
        cubegen_orbital = _default_cubegen_orbital(system)

    singular_values, holes, particles = method.nto(state_f, state_i, nroots=nroots)
    singular_values = np.asarray(singular_values)
    holes = np.asarray(holes)
    particles = np.asarray(particles)

    files = []
    for idx, (hole, particle) in enumerate(zip(holes.T, particles.T), start=1):
        if method.type_u:
            files.extend(
                _write_unrestricted_pair(
                    method, system, outdir, prefix, idx, hole, particle,
                    cubegen_orbital, cube_kwargs, component_tol,
                )
            )
        else:
            files.extend(
                _write_restricted_pair(
                    method, system, outdir, prefix, idx, hole, particle,
                    cubegen_orbital, cube_kwargs,
                )
            )

    return {
        "singular_values": singular_values,
        "files": files,
    }
