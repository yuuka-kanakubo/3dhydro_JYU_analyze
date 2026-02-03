#!/usr/bin/env python3
"""
Make a movie of hydro evolution (temperature on x-y plane) from a text file.

Usage:
  ./make_hydro_movie.py time_evolutionXYplane.dat --out hydro_temp.mp4


Input format per row:
  tau  x  y  ed  vx  vy  temp

Example:
  python make_hydro_movie.py hydro.txt --out hydro_temp.mp4 --fps 20

Dependencies:
  - numpy
  - matplotlib
To save MP4:
  - ffmpeg installed (recommended)
Fallback:
  - GIF via pillow (if you choose --out something.gif)
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter


def load_data(path: Path) -> np.ndarray:
    # Load numeric columns, skip comment lines beginning with '#'
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data[None, :]  # single row -> shape (1, ncol)
    if data.shape[1] < 7:
        raise ValueError(f"Expected >= 7 columns, got {data.shape[1]}")
    return data


def build_frames(data: np.ndarray, xlim=(-10.0, 10.0), ylim=(-10.0, 10.0), tol=1e-10):
    """
    Group rows by tau and return:
      - taus: sorted unique tau values
      - frames: list of dicts, each contains x, y, temp arrays for a single tau
    Only keeps points within xlim/ylim.
    """
    tau = data[:, 0]
    x = data[:, 1]
    y = data[:, 2]
    temp = data[:, 6]

    mask = (x >= xlim[0]) & (x <= xlim[1]) & (y >= ylim[0]) & (y <= ylim[1])
    tau, x, y, temp = tau[mask], x[mask], y[mask], temp[mask]

    # Robust unique taus (floating comparison)
    taus = np.unique(np.round(tau / tol) * tol)
    taus.sort()

    frames = []
    for t in taus:
        m = np.isclose(tau, t, atol=tol, rtol=0.0)
        frames.append({"tau": float(t), "x": x[m], "y": y[m], "temp": temp[m]})

    if len(frames) == 0:
        raise ValueError("No data points found within the specified x/y range.")
    return taus, frames

def diagnose_xy_mismatch(
    infile: Path,
    frames,
    grid_xs,
    grid_ys,
    xlim=(-10.0, 10.0),
    ylim=(-10.0, 10.0),
    xy_tol=1e-8,
    max_print=50,
):
    """
    Print lines in the input file whose x/y values fail exact grid lookup
    but would match after rounding (floating-point mismatch).
    """

    raw = np.loadtxt(infile, comments="#")
    tau_all = raw[:, 0]
    x_all = raw[:, 1]
    y_all = raw[:, 2]

    # domain cut (must match make_movie)
    mask = (
        (x_all >= xlim[0]) & (x_all <= xlim[1]) &
        (y_all >= ylim[0]) & (y_all <= ylim[1])
    )

    raw = raw[mask]

    grid_xs = np.asarray(grid_xs)
    grid_ys = np.asarray(grid_ys)

    def q(v):
        return np.round(v / xy_tol) * xy_tol

    bad = []

    for idx, row in enumerate(raw):
        tau, x, y = row[0], row[1], row[2]

        # exact lookup fails?
        exact_x = x in grid_xs
        exact_y = y in grid_ys

        # rounded lookup would succeed?
        rx = q(x)
        ry = q(y)
        rounded_x = rx in grid_xs
        rounded_y = ry in grid_ys

        if (not exact_x or not exact_y) and (rounded_x and rounded_y):
            bad.append((idx, tau, x, y, rx, ry))

    print(f"[xy-mismatch] Found {len(bad)} float-mismatched rows")
    print(f"[xy-mismatch] Printing up to {max_print}\n")

    for k, (idx, tau, x, y, rx, ry) in enumerate(bad[:max_print]):
        print(
            f"line ~{idx:6d}: "
            f"tau={tau:.10g}  "
            f"x={x:.12g}  y={y:.12g}   "
            f"→ rounded ({rx:.10g}, {ry:.10g})   "
            f"Δx={x-rx:+.3e}  Δy={y-ry:+.3e}"
        )

    if len(bad) > max_print:
        print(f"... {len(bad) - max_print} more rows omitted")



def diagnose_incomplete_grid(
    infile: Path,
    xlim=(-10.0, 10.0),
    ylim=(-10.0, 10.0),
    tau_tol=1e-10,
    xy_tol=1e-8,
    max_print_missing=50,
    max_print_near=5,
):
    """
    For each tau slice, check whether points form a complete grid (X × Y).
    If not, print missing (x,y) cells and optionally show "nearby" points in the file.

    tau_tol: tolerance for grouping tau
    xy_tol: tolerance for comparing x,y (handles floating jitter)
    """

    data = load_data(infile)

    # filter domain
    tau = data[:, 0]
    x = data[:, 1]
    y = data[:, 2]
    temp = data[:, 6]

    mask = (x >= xlim[0]) & (x <= xlim[1]) & (y >= ylim[0]) & (y <= ylim[1])
    data = data[mask]
    if data.size == 0:
        print("[diagnose] No points in the requested x/y range.")
        return

    # Helper: quantize to reduce floating jitter
    def q(v, tol):
        return np.round(v / tol) * tol

    tau_q = q(data[:, 0], tau_tol)

    taus = np.unique(tau_q)
    taus.sort()

    print(f"[diagnose] Checking {len(taus)} tau slices within x={xlim}, y={ylim}")
    print(f"[diagnose] Using tau_tol={tau_tol:g}, xy_tol={xy_tol:g}\n")

    for t in taus:
        m = np.isclose(tau_q, t, atol=tau_tol, rtol=0.0)
        slice_data = data[m]
        xs = q(slice_data[:, 1], xy_tol)
        ys = q(slice_data[:, 2], xy_tol)

        ux = np.unique(xs); ux.sort()
        uy = np.unique(ys); uy.sort()

        npts = xs.size
        expected = ux.size * uy.size

        # Build set of present (x,y) pairs
        present = set(zip(xs.tolist(), ys.tolist()))

        missing = []
        if expected != npts or len(present) != expected:
            # Identify missing cells from the full Cartesian product
            for xv in ux:
                for yv in uy:
                    if (float(xv), float(yv)) not in present:
                        missing.append((float(xv), float(yv)))

        if missing:
            print(f"=== tau ~ {float(t):.12g} fm ===")
            print(f"  points present: {npts}")
            print(f"  unique x: {ux.size}, unique y: {uy.size}, expected Nx*Ny={expected}")
            print(f"  missing cells: {len(missing)} (printing up to {max_print_missing})")

            for (xmiss, ymiss) in missing[:max_print_missing]:
                print(f"    missing (x,y)=({xmiss:.10g}, {ymiss:.10g})")

                # Near-miss help: find points close to this target that *do* exist
                # This often reveals rounding differences or boundary filtering.
                dx = np.abs(slice_data[:, 1] - xmiss)
                dy = np.abs(slice_data[:, 2] - ymiss)
                close = np.where((dx <= 10*xy_tol) & (dy <= 10*xy_tol))[0]

                if close.size > 0:
                    print(f"      nearby points in file (showing up to {max_print_near}):")
                    for idx in close[:max_print_near]:
                        # print the *original* (un-quantized) row
                        row = slice_data[idx]
                        print(
                            "        "
                            f"tau={row[0]:.10g}  x={row[1]:.10g}  y={row[2]:.10g}  "
                            f"ed={row[3]:.10g}  vx={row[4]:.10g}  vy={row[5]:.10g}  temp={row[6]:.10g}"
                        )
                else:
                    print("      no nearby points found (might be truly missing or outside cut).")

            if len(missing) > max_print_missing:
                print(f"  ... {len(missing) - max_print_missing} more missing cells not shown.")
            print()
        # If no missing, stay quiet for that tau



def infer_regular_grid(frames, tol=1e-8):
    """
    Try to infer a regular (x,y) grid from the first frame.
    If it looks like a rectilinear grid, return sorted unique x and y coordinates.
    Otherwise return (None, None).
    """
    f0 = frames[0]
    xs = np.unique(np.round(f0["x"] / tol) * tol)
    ys = np.unique(np.round(f0["y"] / tol) * tol)
    # If points count matches Nx*Ny, it's likely a full grid.
    if xs.size * ys.size == f0["x"].size:
        xs.sort()
        ys.sort()
        return xs, ys
    return None, None

def centers_to_edges(u):
    u = np.asarray(u)
    du = np.diff(u)
    d = np.median(du)
    edges = np.concatenate([[u[0] - d/2], (u[:-1] + u[1:]) / 2, [u[-1] + d/2]])
    return edges


def make_movie(
    infile: Path,
    outfile: Path,
    fps: int = 20,
    dpi: int = 150,
    xlim=(-10.0, 10.0),
    ylim=(-10.0, 10.0),
):
    data = load_data(infile)
    taus, frames = build_frames(data, xlim=xlim, ylim=ylim)

    # Determine global color scale (so color meaning is consistent over time)
    all_temps = np.concatenate([f["temp"] for f in frames])
    vmin = float(np.nanmin(all_temps))
    vmax = float(np.nanmax(all_temps))

    # Try to use imshow on a regular grid (fast + clean).
    # If not regular, fall back to scatter (works for any sampling).
    grid_xs, grid_ys = infer_regular_grid(frames)

    def snap_to_grid(v, grid, tol=1e-10):
        """
        Return nearest coordinate in `grid` to v if within tol, else None.
        grid must be sorted 1D numpy array.
        """
        i = np.searchsorted(grid, v)
        candidates = []
        if 0 <= i < len(grid):
            candidates.append(grid[i])
        if 0 <= i - 1 < len(grid):
            candidates.append(grid[i - 1])
        if not candidates:
            return None
        best = min(candidates, key=lambda g: abs(v - g))
        return best if abs(v - best) <= tol else None


    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.set_xlabel("x [fm]")
    ax.set_ylabel("y [fm]")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")

    tau_text = ax.text(
        0.98, 0.98, "", transform=ax.transAxes,
        ha="right", va="top", fontsize=12,
        bbox=dict(boxstyle="round,pad=0.25", alpha=0.75)
    )


    ## diagnose_xy_mismatch(
    ##     infile=infile,
    ##     frames=frames,
    ##     grid_xs=grid_xs,
    ##     grid_ys=grid_ys,
    ##     xlim=xlim,
    ##     ylim=ylim,
    ##     xy_tol=1e-8,
    ##     max_print=30,
    ## )


    mappable = None

    if grid_xs is not None and grid_ys is not None:
        nx, ny = grid_xs.size, grid_ys.size
    
        # robust index maps (avoid float mismatch)
        x_to_i = {float(x): i for i, x in enumerate(grid_xs)}
        y_to_j = {float(y): j for j, y in enumerate(grid_ys)}
        nx, ny = len(grid_xs), len(grid_ys)

        GRID_X = 0.15
        GRID_Y = 0.15
        
        def frame_to_image(f, tol=1e-10):
            img = np.full((ny, nx), np.nan, dtype=float)
        
            miss = 0
            for xi, yi, ti in zip(f["x"], f["y"], f["temp"]):
                xs = snap_to_grid(xi, grid_xs, tol=tol)
                ys = snap_to_grid(yi, grid_ys, tol=tol)
                if xs is None or ys is None:
                    miss += 1
                    continue
                img[y_to_j[float(ys)], x_to_i[float(xs)]] = ti
        
            # Optional: print once for debugging
            # print("missed points:", miss, "out of", len(f["x"]))
            return img

        # --- INITIAL IMAGE (THIS DEFINES img0) ---
        img0 = frame_to_image(frames[0], tol=1e-10)
        print("NaNs in img0:", np.isnan(img0).sum(), "/", img0.size)

    
        # build correct physical edges
        x_edges = centers_to_edges(grid_xs)
        y_edges = centers_to_edges(grid_ys)
    
        mappable = ax.imshow(
            img0,
            origin="lower",
            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
            vmin=vmin, vmax=vmax,
            interpolation="none",   # NO smoothing
            resample=False,
        )
    
        # make missing cells transparent instead of white
        cmap = plt.get_cmap().copy()
        cmap.set_bad(color="red")   # show missing cells in red
        mappable.set_cmap(cmap)
    
        cbar = fig.colorbar(mappable, ax=ax, pad=0.02)
        cbar.set_label("Temperature")
    
        print("points per frame:", frames[0]["x"].size)
        print("nx * ny:", nx * ny)
        print(" ==> if not equal, grid is incomplete")
    
        def update(k):
            img = frame_to_image(frames[k])
            mappable.set_data(img)
            tau_text.set_text(f"tau = {frames[k]['tau']:.3f} fm")

            return mappable, tau_text


    else:
        # Fallback: scattered sampling
        mappable = ax.scatter(
            frames[0]["x"], frames[0]["y"],
            c=frames[0]["temp"],
            s=10,
            vmin=vmin, vmax=vmax
        )
        cbar = fig.colorbar(mappable, ax=ax, pad=0.02)
        cbar.set_label("Temperature")

        def update(k):
            f = frames[k]
            offsets = np.column_stack([f["x"], f["y"]])
            mappable.set_offsets(offsets)
            mappable.set_array(f["temp"])
            tau_text.set_text(f"tau = {f['tau']:.3f} fm")
            return mappable, tau_text

    # Initialize tau text
    tau_text.set_text(f"tau = {frames[0]['tau']:.3f} fm")
    ax.set_title("Hydro evolution: Temperature on x–y plane")

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=False)

    out_suffix = outfile.suffix.lower()
    if out_suffix == ".mp4":
        writer = FFMpegWriter(fps=fps, bitrate=1800)
        anim.save(outfile, writer=writer, dpi=dpi)
    elif out_suffix in [".gif"]:
        writer = PillowWriter(fps=fps)
        anim.save(outfile, writer=writer, dpi=dpi)
    else:
        raise ValueError("Output must end with .mp4 or .gif")

    plt.close(fig)
    print(f"Saved: {outfile}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("infile", type=Path, help="Input hydro txt file")
    p.add_argument("--out", type=Path, default=Path("hydro_temp.mp4"), help="Output movie (.mp4 or .gif)")
    p.add_argument("--fps", type=int, default=20, help="Frames per second")
    p.add_argument("--dpi", type=int, default=150, help="Output DPI")
    p.add_argument("--xmin", type=float, default=-10.0)
    p.add_argument("--xmax", type=float, default=10.0)
    p.add_argument("--ymin", type=float, default=-10.0)
    p.add_argument("--ymax", type=float, default=10.0)
    args = p.parse_args()

##    diagnose_incomplete_grid(
##    infile=args.infile,
##    xlim=(args.xmin, args.xmax),
##    ylim=(args.ymin, args.ymax),
##    tau_tol=1e-10,
##    xy_tol=1e-8,
##    max_print_missing=30,
##    max_print_near=3,
##    )
##

    make_movie(
        infile=args.infile,
        outfile=args.out,
        fps=args.fps,
        dpi=args.dpi,
        xlim=(args.xmin, args.xmax),
        ylim=(args.ymin, args.ymax),
    )


if __name__ == "__main__":
    main()

