#!/usr/bin/env python3
"""se_twin_diff.py: statistics of a split-explicit (SE) vs semi-implicit (SI) twin run (split-explicit Task 5.2).

Reads the monthly-mean Zarr stores of two fesom_lifecycle_native_mr runs (FESOM3_OUTPUT dirs, one
<var>.fesom.<year>.zarr store per variable, filesplit 'y') and the mesh diagnostics store
fesom.mesh.diag.zarr (FESOM3_MESHDIAG=1), and prints:

  - annual-mean SSH difference SE - SI: area-weighted rms (also with the area-mean offset removed),
    max |diff| with its node (1-based global index), lon, lat and bottom depth; the same per month
  - monthly area-weighted global-mean SSH of each run and its drift (last month - first month)
  - monthly volume-weighted global-mean salt and temp of each run and the drift
  - monthly sea-ice area (sum a_ice * area), extent (area where a_ice > 0.15) and volume
    (sum m_ice * area), NH (lat > 0) and SH (lat < 0), for both runs and SE - SI
  - annual-mean SST / SSS / a_ice differences (rms, max)
  - 3-D temp and salt, SE - SI, per depth band (0-100 m, 100-1000 m, 1000 m-bottom, all) and per
    level: volume-weighted rms and max |diff| (lon, lat, depth) of the annual mean, the first and the
    last month; the band-mean drift (last - first month) of each run; and, as the scale of the
    signal, the rms of the SI run's own change over the year (last - first month)

The two runs need not be SI and SE: --labels names them (e.g. a noise baseline, SI np 256 vs SI np
128, passed as --se / --si); the tables keep the SI / SE column names.

Weights: area = nod_area(1, n) of fesom.mesh.diag.zarr (FESOM_F's depth-independent nodal area);
volume of layer k at node n = nod_area(k, n) * (nz(k+1) - nz(k)), the cold linfs layer thickness
(the model's linfs tracer volume; eta not included); nod_area is 0 below the node's bottom and the
3-D fields are NaN there. Annual means weight the monthly means by the number of days in the month.

Python env: /work/ab0995/a270088/mambaforge/bin/python3 (numpy, xarray, zarr), as tools/zarr_diff.py.

  se_twin_diff.py --si <SI out dir> --se <SE out dir> [--mesh <fesom.mesh.diag.zarr>] [--year 1958]
                  [--json <file>]
Exit 0 = statistics printed, 1 = a store is missing or the grids do not match.
"""
import argparse
import calendar
import json
import os
import sys

import numpy as np
import xarray as xr


def _open(path):
    if not os.path.isdir(path):
        print("FAIL: missing store", path)
        sys.exit(1)
    return xr.open_zarr(path, consolidated=False, decode_times=False)


def _store(d, var, year):
    return os.path.join(d, f"{var}.fesom.{year}.zarr")


def _wrms(x, w):
    m = np.isfinite(x) & (w > 0)
    return float(np.sqrt(np.sum(w[m] * x[m] ** 2) / np.sum(w[m])))


def _wmean(x, w):
    m = np.isfinite(x) & (w > 0)
    return float(np.sum(w[m] * x[m]) / np.sum(w[m]))


def _argmax_abs(x):
    a = np.where(np.isfinite(x), np.abs(x), -1.0)
    i = int(np.argmax(a))
    return i, float(a[i])


def _argmax_abs2(x):
    a = np.where(np.isfinite(x), np.abs(x), -1.0)
    j = int(np.argmax(a))
    k, n = np.unravel_index(j, x.shape)
    return int(k), int(n), float(a.flat[j])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--si", required=True, help="FESOM3_OUTPUT dir of the semi-implicit run")
    ap.add_argument("--se", required=True, help="FESOM3_OUTPUT dir of the split-explicit run")
    ap.add_argument("--mesh", help="fesom.mesh.diag.zarr (default: <si>/fesom.mesh.diag.zarr)")
    ap.add_argument("--year", type=int, default=1958)
    ap.add_argument("--json", help="also write the numbers to this JSON file")
    ap.add_argument("--labels", default="SI,SE", help="names of the --si and --se runs for the header")
    a = ap.parse_args()
    runs = {"SI": a.si, "SE": a.se}
    out = {"si": a.si, "se": a.se, "year": a.year}
    out["labels"] = a.labels
    print(f"runs: SI = {a.labels.split(',')[0]} ({a.si}); SE = {a.labels.split(',')[-1]} ({a.se})")

    # ------------------------------------------------------------------ mesh and weights
    mesh = _open(a.mesh or os.path.join(a.si, "fesom.mesh.diag.zarr"))
    lon = mesh["lon"].values.astype(float)
    lat = mesh["lat"].values.astype(float)
    nod_area = mesh["nod_area"].values.astype(float)          # (nz, nod2), 0 below the bottom
    zlev = mesh["nz"].values.astype(float)                    # level depths, positive down
    zbot = mesh["zbar_n_bottom"].values.astype(float)
    nlev = mesh["nlevels_nod2D"].values.astype(int)
    area = nod_area[0, :]
    dz = np.diff(zlev)                                        # layer thicknesses (nz1)
    vol = nod_area[:-1, :] * dz[:, None]                      # (nz1, nod2)
    nod2 = lon.size
    print(f"mesh: {nod2} nodes, {zlev.size} levels, ocean area {area.sum():.6e} m2, "
          f"volume {vol.sum():.6e} m3 (nod_area(k)*dz(k), linfs)")
    for k in range(1, nod_area.shape[0]):                     # the area is depth-independent in FESOM_F
        wet = nod_area[k] > 0
        if wet.any() and np.max(np.abs(nod_area[k][wet] - area[wet])) > 0:
            print(f"  note: nod_area level {k + 1} differs from level 1")
            break
    out["mesh"] = {"nod2": nod2, "area_m2": float(area.sum()), "volume_m3": float(vol.sum())}

    # ------------------------------------------------------------------ months
    ssh = {}
    tvals = {}
    for r, d in runs.items():
        ds = _open(_store(d, "ssh", a.year))
        ssh[r] = ds["ssh"].values.astype(float)               # (time, nod2)
        tvals[r] = (ds["time"].values, ds["time"].attrs.get("units", "?"))
        dl = float(np.max(np.abs(ds["lon"].values - lon))) if "lon" in ds else float("nan")
        dla = float(np.max(np.abs(ds["lat"].values - lat))) if "lat" in ds else float("nan")
        print(f"{r}: ssh records {ssh[r].shape[0]}, time {list(tvals[r][0])} ({tvals[r][1]}); "
              f"store lon/lat vs mesh max|d| {dl:.1e} / {dla:.1e} deg")
        if ssh[r].shape[1] != nod2:
            print("FAIL: node count differs from the mesh store")
            sys.exit(1)
    nm = min(ssh["SI"].shape[0], ssh["SE"].shape[0])
    if nm < 12:
        print(f"WARNING: only {nm} common monthly records (SI {ssh['SI'].shape[0]}, SE {ssh['SE'].shape[0]})")
    mdays = np.array([calendar.monthrange(a.year, m + 1)[1] for m in range(nm)], dtype=float)
    mnames = [calendar.month_abbr[m + 1] for m in range(nm)]
    out["months"] = mnames
    out["month_days"] = mdays.tolist()

    def annual(x):                                            # (nm, ...) -> day-weighted mean
        return np.tensordot(mdays / mdays.sum(), x[:nm], axes=(0, 0))

    # ------------------------------------------------------------------ SSH difference
    ssh_si = annual(ssh["SI"])
    ssh_se = annual(ssh["SE"])
    d = ssh_se - ssh_si
    off = _wmean(d, area)
    i, mx = _argmax_abs(d)
    res = {
        "rms_m": _wrms(d, area), "mean_offset_m": off, "rms_no_offset_m": _wrms(d - off, area),
        "max_abs_m": mx, "max_node_1based": i + 1, "max_lon": float(lon[i]), "max_lat": float(lat[i]),
        "max_depth_m": float(zbot[i]), "max_nlevels": int(nlev[i]), "max_si_m": float(ssh_si[i]),
        "max_se_m": float(ssh_se[i]),
        "si_annual_ssh_rms_m": _wrms(ssh_si - _wmean(ssh_si, area), area),
    }
    print("\n== annual-mean SSH, SE - SI (area-weighted over", nm, "months; port_kokkos: 0.2 mm rms / 2.9 mm max)")
    print(f"  rms {res['rms_m'] * 1e3:.3f} mm   area-mean offset {off * 1e3:+.4f} mm   "
          f"rms without the offset {res['rms_no_offset_m'] * 1e3:.3f} mm")
    print(f"  max |diff| {mx * 1e3:.3f} mm at node {i + 1} ({lon[i]:.2f}E, {lat[i]:.2f}N, bottom {zbot[i]:.0f} m, "
          f"nlevels {nlev[i]}): SI {ssh_si[i]:.4f} m, SE {ssh_se[i]:.4f} m")
    print(f"  (signal: SI annual-mean SSH rms about its area mean {res['si_annual_ssh_rms_m']:.4f} m)")
    # top 5 locations
    order = np.argsort(-np.where(np.isfinite(d), np.abs(d), -1.0))[:5]
    res["top5"] = [{"node_1based": int(j + 1), "lon": float(lon[j]), "lat": float(lat[j]),
                    "depth_m": float(zbot[j]), "diff_mm": float(d[j] * 1e3)} for j in order]
    print("  largest 5: " + "; ".join(f"{t['diff_mm']:+.2f} mm ({t['lon']:.1f}E {t['lat']:.1f}N, "
                                     f"{t['depth_m']:.0f} m)" for t in res["top5"]))
    print("  per month: rms / max|diff| [mm]")
    res["monthly"] = []
    for m in range(nm):
        dm = ssh["SE"][m] - ssh["SI"][m]
        im, mxm = _argmax_abs(dm)
        res["monthly"].append({"month": mnames[m], "rms_mm": _wrms(dm, area) * 1e3, "max_mm": mxm * 1e3,
                               "max_lon": float(lon[im]), "max_lat": float(lat[im])})
        print(f"    {mnames[m]}  {_wrms(dm, area) * 1e3:8.3f}  {mxm * 1e3:9.3f}  ({lon[im]:7.2f}E {lat[im]:6.2f}N)")
    out["ssh_diff"] = res

    # ------------------------------------------------------------------ global-mean SSH, salt, temp
    gm = {r: [_wmean(ssh[r][m], area) for m in range(nm)] for r in runs}
    tracer = {}
    for var in ("salt", "temp"):
        tracer[var] = {}
        for r, dd in runs.items():
            ds = _open(_store(dd, var, a.year))
            x = ds[var]
            vals = []
            for m in range(nm):
                f = x[m].values.astype(float)                 # (nz1, nod2), NaN below the bottom
                ok = np.isfinite(f) & (vol > 0)
                vals.append(float(np.sum(f[ok] * vol[ok]) / np.sum(vol[ok])))
            tracer[var][r] = vals
    print("\n== monthly global means: SSH (area-weighted) [mm], salt and temp (volume-weighted) [psu, degC]")
    print("  month   SSH_SI      SSH_SE      SE-SI    |  salt_SI        salt_SE        SE-SI      |"
          "  temp_SI       temp_SE       SE-SI")
    for m in range(nm):
        print(f"  {mnames[m]}  {gm['SI'][m] * 1e3:+9.4f}  {gm['SE'][m] * 1e3:+9.4f}  "
              f"{(gm['SE'][m] - gm['SI'][m]) * 1e3:+9.4f}  |  "
              f"{tracer['salt']['SI'][m]:.9f}  {tracer['salt']['SE'][m]:.9f}  "
              f"{tracer['salt']['SE'][m] - tracer['salt']['SI'][m]:+.3e}  |  "
              f"{tracer['temp']['SI'][m]:.8f}  {tracer['temp']['SE'][m]:.8f}  "
              f"{tracer['temp']['SE'][m] - tracer['temp']['SI'][m]:+.3e}")
    drift = {}
    for r in runs:
        drift[r] = {"ssh_mm": (gm[r][-1] - gm[r][0]) * 1e3,
                    "salt_psu": tracer["salt"][r][-1] - tracer["salt"][r][0],
                    "temp_degC": tracer["temp"][r][-1] - tracer["temp"][r][0]}
        print(f"  drift {r} ({mnames[-1]} - {mnames[0]}): SSH {drift[r]['ssh_mm']:+.4f} mm, "
              f"salt {drift[r]['salt_psu']:+.3e} psu, temp {drift[r]['temp_degC']:+.3e} degC")
    out["global_mean"] = {"ssh_m": gm, "salt_psu": tracer["salt"], "temp_degC": tracer["temp"], "drift": drift}

    # ------------------------------------------------------------------ sea ice
    ice = {}
    for r, dd in runs.items():
        ai = _open(_store(dd, "a_ice", a.year))["a_ice"].values.astype(float)[:nm]
        mi = _open(_store(dd, "m_ice", a.year))["m_ice"].values.astype(float)[:nm]
        ice[r] = {"a_ice": ai, "m_ice": mi}
    nh = lat > 0
    sh = lat < 0
    tab = {}
    for r in runs:
        ai, mi = ice[r]["a_ice"], ice[r]["m_ice"]
        tab[r] = {
            "area_NH_1e6km2": [float(np.nansum(ai[m][nh] * area[nh]) / 1e12) for m in range(nm)],
            "area_SH_1e6km2": [float(np.nansum(ai[m][sh] * area[sh]) / 1e12) for m in range(nm)],
            "extent_NH_1e6km2": [float(np.sum(area[nh & (ai[m] > 0.15)]) / 1e12) for m in range(nm)],
            "extent_SH_1e6km2": [float(np.sum(area[sh & (ai[m] > 0.15)]) / 1e12) for m in range(nm)],
            "volume_NH_1e3km3": [float(np.nansum(mi[m][nh] * area[nh]) / 1e12) for m in range(nm)],
            "volume_SH_1e3km3": [float(np.nansum(mi[m][sh] * area[sh]) / 1e12) for m in range(nm)],
        }
    print("\n== sea ice, monthly means: area = sum(a_ice*area) [1e6 km2], volume = sum(m_ice*area) [1e3 km3]")
    print("  month  area_NH SI     SE    SE-SI  |  area_SH SI     SE    SE-SI  |  vol_NH SI    SE   SE-SI"
          "  |  vol_SH SI    SE   SE-SI")
    for m in range(nm):
        row = []
        for key in ("area_NH_1e6km2", "area_SH_1e6km2", "volume_NH_1e3km3", "volume_SH_1e3km3"):
            s, e = tab["SI"][key][m], tab["SE"][key][m]
            row.append(f"{s:7.4f} {e:7.4f} {e - s:+.4f}")
        print(f"  {mnames[m]}   " + "  |  ".join(row))
    print("  extent (a_ice > 0.15) NH SI/SE, SH SI/SE: " + "; ".join(
        f"{mnames[m]} {tab['SI']['extent_NH_1e6km2'][m]:.2f}/{tab['SE']['extent_NH_1e6km2'][m]:.2f} "
        f"{tab['SI']['extent_SH_1e6km2'][m]:.2f}/{tab['SE']['extent_SH_1e6km2'][m]:.2f}" for m in range(nm)))
    out["sea_ice"] = tab

    # ------------------------------------------------------------------ other 2-D annual means
    print("\n== other annual-mean 2-D fields, SE - SI: area-weighted rms, max |diff| (location)")
    out["other_2d"] = {}
    for var, fac, unit in (("sst", 1.0, "degC"), ("sss", 1.0, "psu"), ("a_ice", 1.0, ""), ("m_ice", 1.0, "m")):
        xs = {}
        for r, dd in runs.items():
            xs[r] = annual(_open(_store(dd, var, a.year))[var].values.astype(float))
        dv = (xs["SE"] - xs["SI"]) * fac
        iv, mv = _argmax_abs(dv)
        out["other_2d"][var] = {"rms": _wrms(dv, area), "max_abs": mv, "lon": float(lon[iv]), "lat": float(lat[iv])}
        print(f"  {var:6s} rms {_wrms(dv, area):.3e} {unit}, max {mv:.3e} {unit} ({lon[iv]:.2f}E {lat[iv]:.2f}N)")

    # ------------------------------------------------------------------ 3-D temp and salt
    zmid = 0.5 * (zlev[:-1] + zlev[1:])                       # layer mid-depths (nz1), positive down
    bands = (("0-100 m", 0.0, 100.0), ("100-1000 m", 100.0, 1000.0), ("1000 m-bottom", 1000.0, 1.0e9),
             ("all", 0.0, 1.0e9))
    print("\n== 3-D temp and salt, SE - SI: volume-weighted rms, max |diff| (lon, lat, layer mid-depth)")
    out["tracer_3d"] = {}
    for var, unit in (("temp", "degC"), ("salt", "psu")):
        f = {}
        for r, dd in runs.items():
            x = _open(_store(dd, var, a.year))[var]
            f[r] = np.stack([x[m].values.astype(float) for m in range(nm)])   # (nm, nz1, nod2)
        ann = {r: annual(f[r]) for r in runs}
        cases = (("annual mean", ann["SE"] - ann["SI"]), (f"{mnames[0]}", f["SE"][0] - f["SI"][0]),
                 (f"{mnames[nm - 1]}", f["SE"][nm - 1] - f["SI"][nm - 1]),
                 (f"SI own change {mnames[nm - 1]} - {mnames[0]}", f["SI"][nm - 1] - f["SI"][0]))
        r3 = {}
        print(f"  {var} [{unit}]")
        for cname, dd3 in cases:
            r3[cname] = {}
            for bname, z0, z1 in bands:
                sel = ((zmid >= z0) & (zmid < z1))[:, None] & (vol > 0)
                w = np.where(sel, vol, 0.0)
                k, n, mv = _argmax_abs2(np.where(sel, dd3, np.nan))
                rms = _wrms(dd3, w)
                r3[cname][bname] = {"rms": rms, "max_abs": mv, "lon": float(lon[n]), "lat": float(lat[n]),
                                    "depth_m": float(zmid[k])}
                print(f"    {cname:28s} {bname:14s} rms {rms:.3e}  max {mv:.3e} "
                      f"({lon[n]:7.2f}E {lat[n]:6.2f}N {zmid[k]:6.0f} m)")
        # band means and their drift (last - first month)
        r3["band_mean"] = {}
        print(f"    band-mean {var}: {mnames[0]} / {mnames[nm - 1]} / drift for SI and SE, SE - SI of the drift")
        for bname, z0, z1 in bands:
            sel = ((zmid >= z0) & (zmid < z1))[:, None] & (vol > 0)
            w = np.where(sel, vol, 0.0)
            bm = {r: [_wmean(f[r][0], w), _wmean(f[r][nm - 1], w)] for r in runs}
            dr = {r: bm[r][1] - bm[r][0] for r in runs}
            r3["band_mean"][bname] = {"SI": bm["SI"], "SE": bm["SE"], "drift_SI": dr["SI"], "drift_SE": dr["SE"]}
            print(f"      {bname:14s} SI {bm['SI'][0]:.7f} / {bm['SI'][1]:.7f} / {dr['SI']:+.3e}   "
                  f"SE {bm['SE'][0]:.7f} / {bm['SE'][1]:.7f} / {dr['SE']:+.3e}   SE-SI {dr['SE'] - dr['SI']:+.3e}")
        # per level, annual mean
        d_ann = ann["SE"] - ann["SI"]
        r3["levels"] = []
        print(f"    per level, annual mean SE - SI: level  mid-depth  rms  max |diff| (lon, lat)")
        for k in range(zmid.size):
            w = np.where(vol[k] > 0, vol[k], 0.0)
            if not np.any(w > 0):
                continue
            dk = np.where(w > 0, d_ann[k], np.nan)
            n, mv = _argmax_abs(dk)
            rms = _wrms(d_ann[k], w)
            r3["levels"].append({"level": k + 1, "depth_m": float(zmid[k]), "rms": rms, "max_abs": mv,
                                 "lon": float(lon[n]), "lat": float(lat[n])})
            print(f"      {k + 1:3d} {zmid[k]:7.1f} m  {rms:.3e}  {mv:.3e} ({lon[n]:7.2f}E {lat[n]:6.2f}N)")
        out["tracer_3d"][var] = r3
        del f, ann

    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
        print("\nwrote", a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
