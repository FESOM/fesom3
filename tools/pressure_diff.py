#!/usr/bin/env python3
"""Compare two pressure/EOS dumps (FESOM2 oracle vs FESOM3) field-by-field for the
M2.1 operator byte-gate.

Both files use the self-describing FADVHDMP format (shared with the M1 advection
gate) written by FESOM2 src/fesom_pressure_dump.F90 and FESOM3
src/infra/mod_advhor_dump.F90:

  char8  "FADVHDMP" | int32 nod2D,elem2D,edge2D,nl
  records: char24 name | int32 dtype(0=real64,1=int32) | int32 d1,d2 | d1*d2 values

The gated outputs are:
  density_m_rho0  in-situ density - density_ref   (PGF density anomaly)     M2.1
  hpressure       hydrostatic pressure            (top-down integration)    M2.1
  bvfreq_raw      N^2 squared                     (before horizontal smoothing) M2.1
  bvfreq          N^2 squared                     (after smooth_nod, N2smth_hidx=1) M2.1
  pgf_x, pgf_y    pressure gradient force         (gradient_sca . hpressure / rho0) M2.2
  coriolis        f = 2*omega*sin(lat_geo)        (geometry field, via r2g)  M2.3
  uvnode_rhs      momentum advection on nodes     (w*du/dz + u*du/dx, post-normalize) M2.4
  uv_rhsAB_cor    Coriolis + momentum advection   (this-step AB array)       M2.3/M2.4
  uv_rhs_eul      full vel_rhs, first Euler step (ff=1.0)                    M2.3/M2.4
  uv_rhs_ab2      full vel_rhs, AB2 step (ff=ab2=1.6)                        M2.3/M2.4
  visc_u_c        biharmonic viscosity 1st Laplacian (visc_filt_bidiff pass 1) M2.4-visc
  visc_v_c        biharmonic viscosity 1st Laplacian (V component)            M2.4-visc
  uv_rhs_visc     post-viscosity UV_rhs (opt_visc=7 increment added)          M2.4-visc
  uv_rhs_ivv      post-TDMA UV_rhs (impl_vert_visc_ale Thomas solve)          M2.5
The inputs are dumped too (temp, salt, density_ref, zbar_3d_n, Z_3d_n, hnode for the
EOS path; eta_n, uv_in, uv_rhsAB_prev, w_e for vel_rhs/momadv; Av, stress_surf, w_i for
the implicit vertical viscosity) so any divergence is localised: a wrong EOS shows in
density/bvfreq_raw, a wrong integration in hpressure only, a wrong smoother in
bvfreq-but-not-bvfreq_raw, a wrong PGF contraction in pgf_x/pgf_y only, a wrong Coriolis
in coriolis, a wrong momentum advection in uvnode_rhs, a wrong AB blend in
uv_rhs_eul/uv_rhs_ab2, a wrong biharmonic-viscosity 1st Laplacian in visc_u_c/visc_v_c, a
wrong 2nd pass in uv_rhs_visc-but-not-visc_u_c, a wrong Thomas solve in uv_rhs_ivv.

Usage:  pressure_diff.py <fesom2.bin> <fesom3.bin> [--signal 0.0]
Exit 0 iff every common field matches within the signal threshold (default 0 ==
byte-identical) and shapes/dims agree -- except the fields in SKIP_FIELDS, which both
dumps carry but which are by design not comparable.
"""
import sys
import struct
import numpy as np

MAGIC = b"FADVHDMP"

# Dumped by FESOM2 but NOT compared and no longer dumped by FESOM3: FESOM3 splits w into
# w_e/w_i with the smooth Shchepetkin limiting function (src/oce/oce_wsplit.F90,
# docs/plans/completed/2026-10-02-wsplit-smooth.md), FESOM2 with its hard switch at
# wsplit_maxcfl, so the split fields differ wherever CFL_z > wsplit_mincfl by
# construction. Listing them here keeps the FESOM2-side records out of the "MISSING in
# F3" test (checked before it); w, cfl_z and hnode_new -- the split's inputs -- are
# still gated.
SKIP_FIELDS = {"w_split_e", "w_split_i"}


def read_dump(path):
    with open(path, "rb") as f:
        buf = f.read()
    if buf[:8] != MAGIC:
        raise ValueError(f"{path}: bad magic {buf[:8]!r}")
    off = 8
    nod2D, elem2D, edge2D, nl = struct.unpack_from("<4i", buf, off)
    off += 16
    dims = dict(nod2D=nod2D, elem2D=elem2D, edge2D=edge2D, nl=nl)
    fields = {}
    n = len(buf)
    while off < n:
        name = buf[off:off + 24].decode("ascii", "replace").strip()
        off += 24
        dtype, d1, d2 = struct.unpack_from("<3i", buf, off)
        off += 12
        count = d1 * d2
        if dtype == 0:
            data = np.frombuffer(buf, dtype="<f8", count=count, offset=off)
            off += 8 * count
        elif dtype == 1:
            data = np.frombuffer(buf, dtype="<i4", count=count, offset=off)
            off += 4 * count
        else:
            raise ValueError(f"{path}: unknown dtype {dtype} for {name}")
        fields[name] = (dtype, d1, d2, data)
    return dims, fields


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    signal = 0.0
    for a in sys.argv[1:]:
        if a.startswith("--signal"):
            signal = float(a.split("=", 1)[1]) if "=" in a else 0.0
    if len(args) < 2:
        print(__doc__)
        sys.exit(2)
    p2, p3 = args[0], args[1]
    d2, f2 = read_dump(p2)
    d3, f3 = read_dump(p3)

    print(f"FESOM2: {p2}")
    print(f"FESOM3: {p3}")
    print(f"dims  FESOM2={d2}")
    print(f"dims  FESOM3={d3}")
    ok = True
    if d2 != d3:
        print("FAIL: mesh dims differ")
        ok = False

    names = list(f2.keys())
    extra3 = [k for k in f3 if k not in f2]
    print(f"\n{'field':24s} {'shape':>14s} {'dtype':>6s} {'max|delta|':>16s}  status")
    print("-" * 74)
    for name in names:
        if name in SKIP_FIELDS:
            print(f"{name:24s} {'-':>14s} {'-':>6s} {'not comparable':>16s}  SKIP")
            continue
        if name not in f3:
            print(f"{name:24s} {'-':>14s} {'-':>6s} {'MISSING in F3':>16s}  FAIL")
            ok = False
            continue
        t2, a1, a2, v2 = f2[name]
        t3, b1, b2, v3 = f3[name]
        shape = f"{a1}x{a2}"
        if (t2, a1, a2) != (t3, b1, b2):
            print(f"{name:24s} {shape:>14s} {t2:>6d} shape/dtype F3={b1}x{b2}  FAIL")
            ok = False
            continue
        if v2.shape != v3.shape:
            print(f"{name:24s} {shape:>14s} {t2:>6d} {'len mismatch':>16s}  FAIL")
            ok = False
            continue
        d = np.abs(v2 - v3)
        mx = float(d.max()) if d.size else 0.0
        status = "PASS" if mx <= signal else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"{name:24s} {shape:>14s} {t2:>6d} {mx:>16.6e}  {status}")

    if extra3:
        print(f"\n(only in FESOM3, ignored: {extra3})")

    tag = ("M2.1 density+hpressure+bvfreq; M2.2 pgf; M2.3 coriolis+vel_rhs; M2.4 momentum advection -> full "
           "UV_rhs + biharmonic viscosity opt_visc=7; M2.5 implicit vertical viscosity TDMA; M2.6 SSH CG; M2.7 "
           "ALE update; M2.8 PP mixing; M2.8b mo_convect; M2.9a tracer solve diff_tracers_ale [TDMA consumes Kv]")
    print("\n" + (f"PRESSURE/EOS/N2/PGF/VELRHS/VISC/IVERTVISC/SSH/ALE/PP/TRACER-SOLVE BYTE-GATE ({tag}): PASS (max|delta|=0)" if ok else
                  f"PRESSURE/EOS/N2/PGF/VELRHS/VISC/IVERTVISC/SSH/ALE/PP/TRACER-SOLVE BYTE-GATE ({tag}): FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
