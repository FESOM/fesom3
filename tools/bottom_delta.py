#!/usr/bin/env python3
"""Audit the FESOM3 bottom-at-vertices change against the mesh it was predicted from.

Reads a FESOM3 mesh-geometry dump (written by src/infra/mod_geom_dump.F90; format
documented at the top of tools/geom_diff.py) plus the mesh directory's ORIGINAL
elvls.out, and reports how the element bottom moved when it stopped being read from
that file and started being derived as min over the element's vertex columns.

    nlevels_new(e) = min over v in e of nlevels_nod2D(v)      (from the dump)
    nlevels_old(e) = elvls.out                                (no longer read)

Reported:
  - the per-element level-shift histogram and the changed-element count
  - the ocean-volume change, ELEM_AREA-WEIGHTED. The weighting matters: an unweighted
    per-element depth sum gives +1.37% on pi where the weighted figure is +1.55%, and
    comparing the wrong one against the plan's prediction would look like a failure.
  - stagnant bottom cells: vertices whose deepest scalar cell has NO wet adjacent
    element. Must be zero -- three unguarded divides in the model depend on it
    (oce_ale.F90:88, oce_ale.F90:377, oce_pressure_bv.F90:310).
  - a units sanity check on edge_dxdy / edge_len (R7: both in metres).

Usage:  bottom_delta.py <geom_dump.bin> <mesh_dir> [--expect-changed N] [--expect-volume PCT]

Exit status 0 iff the mesh is self-consistent (0 stagnant cells, derivation reproduced)
and any --expect values match.
"""
import sys
import struct
import argparse
from collections import Counter


def read_dump(path, want=None):
    """FGEOMDMP: magic | int32 nod2D,elem2D,edge2D,nl | records.

    `want` limits which records are unpacked. The area arrays are (nl, nod2D) doubles --
    24M values on core2 -- and this tool never looks at them, so skipping the unpack
    turns a multi-minute read into a couple of seconds.
    """
    with open(path, "rb") as f:
        buf = f.read()
    if buf[:8] != b"FGEOMDMP":
        raise ValueError(f"{path}: bad magic {buf[:8]!r}")
    off = 8
    nod2D, elem2D, edge2D, nl = struct.unpack_from("<4i", buf, off)
    off += 16
    dims = dict(nod2D=nod2D, elem2D=elem2D, edge2D=edge2D, nl=nl)
    fields = {}
    while off < len(buf):
        name = buf[off:off + 24].decode("ascii", "replace").strip()
        off += 24
        dtype, d1, d2 = struct.unpack_from("<3i", buf, off)
        off += 12
        n = d1 * d2
        size = (8 if dtype == 0 else 4) * n
        if want is not None and name not in want:
            off += size
            continue
        if dtype == 0:
            vals = struct.unpack_from(f"<{n}d", buf, off)
        else:
            vals = struct.unpack_from(f"<{n}i", buf, off)
        off += size
        fields[name] = (vals, d1, d2)
    return dims, fields


def cols(fields, name, ncol):
    """the first `ncol` rows of a (d1, d2) column-major record, as `ncol` lists.

    Slice ONCE per row, not once per element: `vals[i::d1]` builds a whole new list, so
    calling it inside a per-element loop is quadratic and will not finish on core2.
    """
    vals, d1, _ = fields[name]
    return [vals[i::d1] for i in range(ncol)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("mesh_dir")
    ap.add_argument("--expect-changed", type=int, default=None)
    ap.add_argument("--expect-volume", type=float, default=None)
    ap.add_argument("--tol-volume", type=float, default=0.01,
                    help="absolute tolerance on the volume percentage (default 0.01)")
    args = ap.parse_args()

    dims, f = read_dump(args.dump, want={
        "nlevels", "nlevels_nod2D", "elem_area", "elem2D_nodes", "edge_len", "edge_dxdy"})
    nE, nN, nl = dims["elem2D"], dims["nod2D"], dims["nl"]

    for need in ("nlevels", "nlevels_nod2D", "elem_area", "elem2D_nodes"):
        if need not in f:
            sys.exit(f"bottom_delta: dump is missing '{need}'")

    new = list(f["nlevels"][0])
    nlv = list(f["nlevels_nod2D"][0])
    area = list(f["elem_area"][0])                    # already scaled to m^2
    n1, n2, n3 = cols(f, "elem2D_nodes", 3)

    # zbar from aux3d.out (not carried in the dump)
    tok = open(f"{args.mesh_dir}/aux3d.out").read().split()
    zbar = [float(x) for x in tok[1:1 + nl]]
    if zbar[1] > 0:
        zbar = [-z for z in zbar]

    old = [int(x) for x in open(f"{args.mesh_dir}/elvls.out")]
    if len(old) != nE:
        sys.exit(f"bottom_delta: elvls.out has {len(old)} entries, dump has {nE} elements")

    print(f"=== {args.mesh_dir}   nod2D={nN} elem2D={nE} nl={nl} ===")

    # 1. the derivation actually applied
    bad = [e for e in range(nE)
           if new[e] != min(nlv[n1[e] - 1], nlv[n2[e] - 1], nlv[n3[e] - 1])]
    if bad:
        sys.exit(f"bottom_delta: nlevels is NOT min over the element vertices "
                 f"({len(bad)} elements, first={bad[0] + 1})")
    print("derivation: nlevels(e) == min over element vertices           OK")

    # 2. level shift
    d = [new[e] - old[e] for e in range(nE)]
    hist = Counter(x for x in d if x)
    changed = sum(hist.values())
    print(f"elem bottom vs elvls : mean {sum(d)/nE:+.3f}  min {min(d):+d}  max {max(d):+d}  "
          f"changed {changed}/{nE} ({100*changed/nE:.1f}%)")
    for k in sorted(hist):
        print(f"    {k:+3d} : {hist[k]:6d}")

    # 3. volume, ELEM_AREA-WEIGHTED
    v0 = sum(area[e] * -zbar[old[e] - 1] for e in range(nE))
    v1 = sum(area[e] * -zbar[new[e] - 1] for e in range(nE))
    dv = 100.0 * (v1 - v0) / v0
    print(f"ocean volume (area-weighted) : {dv:+.2f} %")
    print(f"max depth : {max(-zbar[old[e]-1] for e in range(nE)):.0f} -> "
          f"{max(-zbar[new[e]-1] for e in range(nE)):.0f} m")

    # 4. stagnant bottom cells
    deepest = [0] * nN
    for e in range(nE):
        ne = new[e]
        for n in (n1[e], n2[e], n3[e]):
            if ne > deepest[n - 1]:
                deepest[n - 1] = ne
    stag = sum(1 for n in range(nN) if deepest[n] and deepest[n] < nlv[n])
    print(f"stagnant bottom cells : {stag} nodes ({100*stag/nN:.2f} %)")

    # 5. R7 units
    ok_units = True
    if "edge_len" in f:
        elen = [x for x in f["edge_len"][0] if x != 0.0]
        edx = [abs(x) for x in f["edge_dxdy"][0]]
        print(f"edge_len   : min {min(elen):.1f} m  max {max(elen):.1f} m")
        print(f"edge_dxdy  : max |.| {max(edx):.1f} m")
        if min(elen) <= 1.0 or max(edx) < 1.0:
            print("  FAIL: edge geometry is not metre-scale (R7 factor not folded in?)")
            ok_units = False
    else:
        print("edge_len   : absent from the dump (pre-R7 build)")

    fail = False
    if stag != 0:
        print(f"  FAIL: {stag} stagnant bottom cells -- three unguarded divides need zero")
        fail = True
    if not ok_units:
        fail = True
    if args.expect_changed is not None and changed != args.expect_changed:
        print(f"  FAIL: changed-element count {changed} != expected {args.expect_changed}")
        fail = True
    if args.expect_volume is not None and abs(dv - args.expect_volume) > args.tol_volume:
        print(f"  FAIL: volume {dv:+.2f}% != expected {args.expect_volume:+.2f}% "
              f"(tol {args.tol_volume})")
        fail = True

    print("RESULT: " + ("FAIL" if fail else "OK"))
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
