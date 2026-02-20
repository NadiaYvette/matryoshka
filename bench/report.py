#!/usr/bin/env python3
"""
Matryoshka B+ tree benchmark report generator.

Runs bench_compare across all library x workload x size combinations,
optionally collects perf stat hardware counters and perf record profiles,
generates matplotlib charts, and compiles a LaTeX PDF report.

Usage:
    python3 bench/report.py [--build-dir build] [--output matryoshka_report.pdf]
                             [--no-perf] [--sizes 65536 1048576 4194304]

Prerequisites:
    - The matryoshka library must be built (bench_compare in build_dir)
    - Python 3 with matplotlib, numpy, jinja2
    - pdflatex (texlive)
    - Optional: perf (linux-tools) for hardware counters and profiling
"""

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import datetime
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import jinja2


# ── Configuration ─────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent

LIBRARIES = ["matryoshka", "std_set", "tlx_btree", "libart"]
WORKLOADS = [
    "seq_insert", "rand_insert", "rand_delete",
    "mixed", "ycsb_a", "ycsb_b", "search_after_churn",
]
DEFAULT_SIZES = [65536, 262144, 1048576, 4194304, 16777216]

LIBRARY_COLORS = {
    "matryoshka":    "#2980b9",  # blue
    "std_set":       "#e74c3c",  # red
    "tlx_btree":     "#8e44ad",  # purple
    "libart":        "#f39c12",  # orange
    "abseil_btree":  "#27ae60",  # green
}

LIBRARY_LABELS = {
    "matryoshka":    "Matryoshka B+ tree",
    "std_set":       "std::set (RB tree)",
    "tlx_btree":     "TLX btree\\_set",
    "libart":        "libart (ART)",
    "abseil_btree":  "Abseil btree\\_set",
}

LIBRARY_LABELS_PLAIN = {
    "matryoshka":    "Matryoshka B+ tree",
    "std_set":       "std::set (RB tree)",
    "tlx_btree":     "TLX btree_set",
    "libart":        "libart (ART)",
    "abseil_btree":  "Abseil btree_set",
}

WORKLOAD_LABELS = {
    "seq_insert":         "Sequential Insert",
    "rand_insert":        "Random Insert",
    "rand_delete":        "Random Delete",
    "mixed":              "Mixed Insert/Delete",
    "ycsb_a":             "YCSB-A (95\\% write)",
    "ycsb_b":             "YCSB-B (50\\% delete)",
    "search_after_churn": "Search After Churn",
}

WORKLOAD_LABELS_PLAIN = {
    "seq_insert":         "Sequential Insert",
    "rand_insert":        "Random Insert",
    "rand_delete":        "Random Delete",
    "mixed":              "Mixed Insert/Delete",
    "ycsb_a":             "YCSB-A (95% write)",
    "ycsb_b":             "YCSB-B (50% delete)",
    "search_after_churn": "Search After Churn",
}

# Hardware counters to collect with perf stat
PERF_EVENTS = [
    "cache-misses",
    "cache-references",
    "L1-dcache-load-misses",
    "L1-dcache-loads",
    "LLC-load-misses",
    "LLC-loads",
    "dTLB-load-misses",
    "dTLB-loads",
    "branch-misses",
    "branches",
    "instructions",
    "cycles",
]


# ── Helpers ───────────────────────────────────────────────────────

TEX_SPECIAL = str.maketrans({
    '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#',
    '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}',
    '^': r'\textasciicircum{}',
})


def tex_escape(s):
    """Escape LaTeX special characters in a string."""
    if not isinstance(s, str):
        s = str(s)
    return s.translate(TEX_SPECIAL)


def fmt_size(n):
    """Format an integer as a human-readable size label (64K, 4M, etc.)."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.0f}M" if n % 1_000_000 == 0 else f"{n / 1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.0f}K" if n % 1_000 == 0 else f"{n / 1_000:.1f}K"
    return str(n)


def fmt_bytes(n):
    """Format byte count for display."""
    if n is None:
        return "?"
    if n >= 1_048_576:
        return f"{n // 1_048_576} MB"
    elif n >= 1024:
        return f"{n // 1024} KB"
    return f"{n} B"


# ── System Info ───────────────────────────────────────────────────

def get_system_info():
    """Collect system information (CPU, caches, kernel)."""
    info = {}
    info["hostname"] = platform.node()
    info["kernel"] = platform.release()
    info["arch"] = platform.machine()
    info["date"] = datetime.datetime.now().isoformat(timespec="seconds")
    info["python"] = platform.python_version()

    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        info["cpu"] = "unknown"

    for name, var in [("L1d", "LEVEL1_DCACHE_SIZE"),
                      ("L2", "LEVEL2_CACHE_SIZE"),
                      ("L3", "LEVEL3_CACHE_SIZE")]:
        try:
            val = subprocess.check_output(
                ["getconf", var], stderr=subprocess.DEVNULL, text=True
            ).strip()
            info[name] = int(val)
        except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
            info[name] = None

    return info


# ── Library Detection ─────────────────────────────────────────────

def detect_libraries(bench_binary):
    """Run bench_compare --help and parse which libraries are compiled in."""
    try:
        proc = subprocess.run(
            [str(bench_binary), "--help"],
            capture_output=True, text=True, timeout=10,
        )
        text = proc.stdout + proc.stderr
        available = []
        for lib in LIBRARIES + ["abseil_btree"]:
            if lib in text:
                available.append(lib)
        return available
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return []


# ── Benchmark Execution ──────────────────────────────────────────

def run_bench_compare(bench_binary, library, workload, size):
    """Run bench_compare for a single (library, workload, size) combination.

    Returns a dict parsed from the JSON output line, or None on failure.
    """
    cmd = [
        str(bench_binary),
        "--library", library,
        "--workload", workload,
        "--size", str(size),
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
        )
        for line in proc.stdout.strip().splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    pass
    except (subprocess.TimeoutExpired, OSError):
        pass
    return None


def run_all_benchmarks(bench_binary, libraries, workloads, sizes):
    """Run bench_compare for all combinations.  Returns list of result dicts."""
    results = []
    total = len(libraries) * len(workloads) * len(sizes)
    done = 0

    for lib in libraries:
        for wl in workloads:
            for sz in sizes:
                done += 1
                label = f"{lib}/{wl} N={fmt_size(sz)}"
                print(f"  [{done:3d}/{total}] {label}...", end="", flush=True)
                r = run_bench_compare(bench_binary, lib, wl, sz)
                if r:
                    results.append(r)
                    mops = r.get("mops", 0)
                    print(f" {mops:.2f} Mop/s")
                else:
                    print(" FAILED")

    return results


# ── Perf Stat (Hardware Counters) ─────────────────────────────────

def have_perf():
    """Check if perf is available and usable."""
    try:
        proc = subprocess.run(
            ["perf", "stat", "--", "true"],
            capture_output=True, text=True, timeout=10,
        )
        return proc.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def run_perf_stat(bench_binary, library, workload, size):
    """Run perf stat on bench_compare and return parsed counter dict."""
    events = ",".join(PERF_EVENTS)
    cmd = [
        "perf", "stat", "-e", events, "--",
        str(bench_binary),
        "--library", library,
        "--workload", workload,
        "--size", str(size),
    ]
    env = dict(os.environ, LC_ALL="C")  # force C locale for consistent parsing
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600, env=env,
        )
        counters = {}
        for line in proc.stderr.splitlines():
            line = line.strip()
            # perf stat output formats:
            #   "198137  cache-misses  ..."           (non-hybrid)
            #   "198137  cpu_atom/cache-misses/  ..."  (hybrid Intel P/E)
            # Numbers may have , separators (LC_ALL=C uses none).
            m = re.match(r"^([\d,]+)\s+(?:cpu_\w+/)?([\w-]+)/?", line)
            if m:
                val_str = m.group(1).replace(",", "")
                try:
                    value = int(val_str)
                except ValueError:
                    continue
                name = m.group(2)
                # Accumulate across cpu_core/cpu_atom on hybrid CPUs
                prev = counters.get(name)
                counters[name] = (prev or 0) + value
            # Also handle "<not counted>" or "<not supported>"
            m2 = re.match(r"^<not (?:counted|supported)>\s+(?:cpu_\w+/)?([\w-]+)", line)
            if m2:
                name = m2.group(1)
                if name not in counters:
                    counters[name] = None
        return counters
    except (subprocess.TimeoutExpired, OSError):
        return {}


def collect_perf_stats(bench_binary, libraries, workload, size):
    """Collect perf stat counters for all libraries at a given workload/size.

    Returns dict: library -> counter_dict.
    """
    results = {}
    for lib in libraries:
        label = f"{lib}/{workload} N={fmt_size(size)}"
        print(f"  perf stat: {label}...", end="", flush=True)
        counters = run_perf_stat(bench_binary, lib, workload, size)
        if counters:
            results[lib] = counters
            # Show cache miss rate if available
            refs = counters.get("cache-references")
            misses = counters.get("cache-misses")
            if refs and misses and refs > 0:
                rate = 100.0 * misses / refs
                print(f" cache miss {rate:.1f}%")
            else:
                print(" OK")
        else:
            print(" FAILED")
    return results


# ── Perf Record (Profile Top Functions) ───────────────────────────

def run_perf_record(bench_binary, library, workload, size):
    """Run perf record + perf report on bench_compare.

    Returns list of (percent, symbol) tuples for top functions.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        perf_data = os.path.join(tmpdir, "perf.data")
        # Record
        cmd_record = [
            "perf", "record", "-g", "-o", perf_data, "--",
            str(bench_binary),
            "--library", library,
            "--workload", workload,
            "--size", str(size),
        ]
        try:
            proc = subprocess.run(
                cmd_record, capture_output=True, text=True, timeout=600,
            )
            if proc.returncode != 0:
                print(f"    perf record exit code {proc.returncode}",
                      flush=True)
                if proc.stderr:
                    print(f"    stderr: {proc.stderr.strip()[:200]}",
                          flush=True)
                return []
        except (subprocess.TimeoutExpired, OSError) as e:
            print(f"    perf record exception: {e}", flush=True)
            return []

        # Report
        cmd_report = [
            "perf", "report", "-i", perf_data,
            "--stdio", "--no-children",
            "-g", "none", "--percent-limit", "1.0",
        ]
        try:
            proc = subprocess.run(
                cmd_report, capture_output=True, text=True, timeout=60,
            )
            if proc.returncode != 0:
                print(f"    perf report exit code {proc.returncode}",
                      flush=True)
                if proc.stderr:
                    print(f"    stderr: {proc.stderr.strip()[:200]}",
                          flush=True)
        except (subprocess.TimeoutExpired, OSError) as e:
            print(f"    perf report exception: {e}", flush=True)
            return []

        # Parse all matching lines across event groups (hybrid CPUs
        # produce separate cpu_atom and cpu_core sections).
        by_sym = {}
        for line in proc.stdout.splitlines():
            line = line.strip()
            # Format: "  XX.XX%  command  shared_object  [.] symbol_name"
            m = re.match(r"^(\d+\.\d+)%\s+.+\[\.\]\s+(.+)$", line)
            if m:
                pct = float(m.group(1))
                sym = m.group(2).strip()
                by_sym[sym] = by_sym.get(sym, 0.0) + pct

        # Sort by descending overhead, return top 20
        top = sorted(by_sym.items(), key=lambda x: -x[1])[:20]
        return [(pct, sym) for sym, pct in top]


# ── Perf Cache-Miss Attribution ────────────────────────────────────

def run_perf_cache_misses(bench_binary, library, workload, size):
    """Run perf record -e cache-misses + perf report to attribute cache
    misses per function.

    Returns list of (percent, symbol) tuples for top functions.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        perf_data = os.path.join(tmpdir, "perf.data")
        cmd_record = [
            "perf", "record", "-e", "cache-misses", "-o", perf_data, "--",
            str(bench_binary),
            "--library", library,
            "--workload", workload,
            "--size", str(size),
        ]
        try:
            proc = subprocess.run(
                cmd_record, capture_output=True, text=True, timeout=600,
            )
            if proc.returncode != 0:
                print(f"    perf record (cache-misses) exit code {proc.returncode}",
                      flush=True)
                return []
        except (subprocess.TimeoutExpired, OSError) as e:
            print(f"    perf record (cache-misses) exception: {e}", flush=True)
            return []

        cmd_report = [
            "perf", "report", "-i", perf_data,
            "--stdio", "--no-children",
            "-g", "none", "--percent-limit", "1.0",
        ]
        try:
            proc = subprocess.run(
                cmd_report, capture_output=True, text=True, timeout=60,
            )
            if proc.returncode != 0:
                print(f"    perf report (cache-misses) exit code {proc.returncode}",
                      flush=True)
        except (subprocess.TimeoutExpired, OSError) as e:
            print(f"    perf report (cache-misses) exception: {e}", flush=True)
            return []

        by_sym = {}
        for line in proc.stdout.splitlines():
            line = line.strip()
            m = re.match(r"^(\d+\.\d+)%\s+.+\[\.\]\s+(.+)$", line)
            if m:
                pct = float(m.group(1))
                sym = m.group(2).strip()
                by_sym[sym] = by_sym.get(sym, 0.0) + pct

        top = sorted(by_sym.items(), key=lambda x: -x[1])[:20]
        return [(pct, sym) for sym, pct in top]


# ── Chart Generation ──────────────────────────────────────────────

def _setup_chart_style():
    """Set consistent matplotlib style for all charts."""
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 150,
    })


def make_throughput_bar_charts(results, sizes, libraries, chart_dir):
    """Generate throughput bar charts per workload (grouped by library).

    One chart per workload, bars grouped by library at each size.
    Returns dict: workload -> chart_path.
    """
    _setup_chart_style()
    charts = {}

    # Index results: (library, workload, n) -> mops
    by_key = {}
    for r in results:
        key = (r["library"], r["workload"], r["n"])
        by_key[key] = r.get("mops", 0)

    # Only use libraries that appear in results
    active_libs = [lib for lib in libraries
                   if any(r["library"] == lib for r in results)]

    for wl in WORKLOADS:
        wl_results = [r for r in results if r["workload"] == wl]
        if not wl_results:
            continue

        # Use the largest size present for this workload
        target_size = max(r["n"] for r in wl_results)

        libs_with_data = [lib for lib in active_libs
                          if (lib, wl, target_size) in by_key]
        if not libs_with_data:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(libs_with_data))
        vals = [by_key.get((lib, wl, target_size), 0) for lib in libs_with_data]
        colors = [LIBRARY_COLORS.get(lib, "#7f8c8d") for lib in libs_with_data]

        bars = ax.bar(x, vals, color=colors, zorder=3, width=0.6)
        # Value labels on bars
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.1f}", ha="center", va="bottom", fontsize=8,
                    fontweight="bold")

        labels = [LIBRARY_LABELS_PLAIN.get(lib, lib) for lib in libs_with_data]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
        ax.set_ylabel("Throughput (Mop/s)")
        ax.set_title(f"{WORKLOAD_LABELS_PLAIN.get(wl, wl)} -- N = {fmt_size(target_size)}",
                     fontweight="bold")
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylim(bottom=0)

        fig.tight_layout()
        out = chart_dir / f"bar_{wl}.pdf"
        fig.savefig(out, format="pdf", bbox_inches="tight")
        plt.close(fig)
        charts[wl] = str(out)

    return charts


def make_scaling_line_charts(results, sizes, libraries, chart_dir):
    """Generate scaling line plots (Mop/s vs N, one line per library).

    One chart per workload.  Returns dict: workload -> chart_path.
    """
    _setup_chart_style()
    charts = {}

    # Index results
    by_key = {}
    for r in results:
        key = (r["library"], r["workload"], r["n"])
        by_key[key] = r.get("mops", 0)

    active_libs = [lib for lib in libraries
                   if any(r["library"] == lib for r in results)]

    for wl in WORKLOADS:
        wl_results = [r for r in results if r["workload"] == wl]
        if not wl_results:
            continue

        wl_sizes = sorted(set(r["n"] for r in wl_results))
        if len(wl_sizes) < 2:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))

        for lib in active_libs:
            xs = []
            ys = []
            for sz in wl_sizes:
                mops = by_key.get((lib, wl, sz))
                if mops is not None and mops > 0:
                    xs.append(sz)
                    ys.append(mops)
            if xs:
                color = LIBRARY_COLORS.get(lib, "#7f8c8d")
                ax.plot(xs, ys, "o-", color=color, linewidth=2, markersize=5,
                        label=LIBRARY_LABELS_PLAIN.get(lib, lib))

        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda v, _: fmt_size(int(v)) if v > 0 else ""))
        ax.set_xlabel("N (keys)")
        ax.set_ylabel("Throughput (Mop/s)")
        ax.set_title(f"Scaling: {WORKLOAD_LABELS_PLAIN.get(wl, wl)}",
                     fontweight="bold")
        ax.legend(loc="best")
        ax.grid(alpha=0.3)
        ax.set_axisbelow(True)

        fig.tight_layout()
        out = chart_dir / f"scaling_{wl}.pdf"
        fig.savefig(out, format="pdf", bbox_inches="tight")
        plt.close(fig)
        charts[wl] = str(out)

    return charts


def make_hw_counter_chart(perf_data, libraries, chart_dir):
    """Generate hardware counter comparison bar chart.

    perf_data: dict of library -> counter_dict.
    Returns chart path or None.
    """
    _setup_chart_style()

    active_libs = [lib for lib in libraries if lib in perf_data]
    if not active_libs:
        return None

    # Compute miss rates
    metrics = {}
    metric_names = []

    # Cache miss rate
    for lib in active_libs:
        c = perf_data[lib]
        rates = {}

        refs = c.get("cache-references")
        misses = c.get("cache-misses")
        if refs and misses and refs > 0:
            rates["Cache Miss %"] = 100.0 * misses / refs
        else:
            rates["Cache Miss %"] = 0

        l1_loads = c.get("L1-dcache-loads")
        l1_misses = c.get("L1-dcache-load-misses")
        if l1_loads and l1_misses and l1_loads > 0:
            rates["L1d Miss %"] = 100.0 * l1_misses / l1_loads
        else:
            rates["L1d Miss %"] = 0

        llc_loads = c.get("LLC-loads")
        llc_misses = c.get("LLC-load-misses")
        if llc_loads and llc_misses and llc_loads > 0:
            rates["LLC Miss %"] = 100.0 * llc_misses / llc_loads
        else:
            rates["LLC Miss %"] = 0

        branches = c.get("branches")
        br_misses = c.get("branch-misses")
        if branches and br_misses and branches > 0:
            rates["Branch Miss %"] = 100.0 * br_misses / branches
        else:
            rates["Branch Miss %"] = 0

        cycles = c.get("cycles")
        instructions = c.get("instructions")
        if cycles and instructions and cycles > 0:
            rates["IPC"] = instructions / cycles
        else:
            rates["IPC"] = 0

        metrics[lib] = rates

    # Only plot rates (not IPC) together, IPC on a second axis
    rate_names = ["Cache Miss %", "L1d Miss %", "LLC Miss %", "Branch Miss %"]
    n_metrics = len(rate_names)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5),
                                    gridspec_kw={"width_ratios": [3, 1]})

    # Left panel: miss rates
    x = np.arange(n_metrics)
    n_libs = len(active_libs)
    width = 0.8 / max(n_libs, 1)

    for i, lib in enumerate(active_libs):
        vals = [metrics[lib].get(m, 0) for m in rate_names]
        color = LIBRARY_COLORS.get(lib, "#7f8c8d")
        ax1.bar(x + i * width, vals, width, label=LIBRARY_LABELS_PLAIN.get(lib, lib),
                color=color, zorder=3)

    ax1.set_xticks(x + width * (n_libs - 1) / 2)
    ax1.set_xticklabels(rate_names, rotation=20, ha="right", fontsize=9)
    ax1.set_ylabel("Miss Rate (%)")
    ax1.set_title("Hardware Counter Miss Rates", fontweight="bold")
    ax1.legend(fontsize=8, loc="best")
    ax1.grid(axis="y", alpha=0.3, zorder=0)
    ax1.set_axisbelow(True)

    # Right panel: IPC
    x2 = np.arange(len(active_libs))
    ipc_vals = [metrics[lib].get("IPC", 0) for lib in active_libs]
    colors = [LIBRARY_COLORS.get(lib, "#7f8c8d") for lib in active_libs]
    ax2.bar(x2, ipc_vals, color=colors, zorder=3, width=0.6)
    for i, v in enumerate(ipc_vals):
        ax2.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8,
                 fontweight="bold")
    labels = [LIBRARY_LABELS_PLAIN.get(lib, lib) for lib in active_libs]
    ax2.set_xticks(x2)
    ax2.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax2.set_ylabel("Instructions Per Cycle")
    ax2.set_title("IPC", fontweight="bold")
    ax2.grid(axis="y", alpha=0.3, zorder=0)
    ax2.set_axisbelow(True)

    fig.suptitle("Hardware Counters (largest size, rand_insert)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = chart_dir / "hw_counters.pdf"
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return str(out)


# ── LaTeX Template Loading & Compilation ─────────────────────────

LIBRARY_DESCRIPTIONS = {
    "matryoshka":   "B+ tree with nested CL sub-tree leaves (up to 855 keys/page), SIMD search, hugepage arena",
    "std_set":      "Red-black tree (libstdc++), pointer-chasing, 40--48\\,B/node",
    "tlx_btree":    "Cache-conscious B+ tree, sorted-array leaves ($B{\\approx}128$)",
    "libart":       "Adaptive Radix Tree, 4-byte keys, no predecessor search",
    "abseil_btree": "Google B-tree, sorted-array leaves ($B{\\approx}256$)",
}


def _get_template_path():
    """Return the path to report_template.tex.j2."""
    return Path(__file__).resolve().parent / "report_template.tex.j2"



def render_latex(context):
    """Render the external LaTeX template with the provided context dict."""
    template_path = _get_template_path()
    template_dir = str(template_path.parent)
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(template_dir),
        block_start_string=r'\BLOCK{',
        block_end_string='}',
        variable_start_string=r'\VAR{',
        variable_end_string='}',
        comment_start_string=r'\#{',
        comment_end_string='}',
        line_statement_prefix='%%>',
        line_comment_prefix='%%#',
        trim_blocks=True,
        lstrip_blocks=True,
        autoescape=False,
        undefined=jinja2.Undefined,  # missing vars render as empty
    )
    template = env.get_template(template_path.name)
    return template.render(**context)


def compile_latex(tex_content, output_path, chart_dir):
    """Compile .tex string to PDF using pdflatex (2 passes)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tex_path = Path(tmpdir) / "report.tex"
        tex_path.write_text(tex_content, encoding="utf-8")

        # Symlink chart PDFs into temp directory so includegraphics finds them
        chart_dir = Path(chart_dir)
        if chart_dir.exists():
            for pdf_file in chart_dir.glob("*.pdf"):
                dst = Path(tmpdir) / pdf_file.name
                if not dst.exists():
                    os.symlink(pdf_file.resolve(), dst)

        # Run pdflatex twice (first pass for ToC/refs, second to resolve)
        for pass_num in range(2):
            print(f"  pdflatex pass {pass_num + 1}/2...", end="", flush=True)
            try:
                proc = subprocess.run(
                    ["pdflatex", "-interaction=nonstopmode",
                     "-halt-on-error", "report.tex"],
                    cwd=tmpdir, capture_output=True, text=True, timeout=120,
                )
                if proc.returncode != 0 and pass_num == 1:
                    print(" WARNINGS")
                    for line in proc.stdout.splitlines()[-30:]:
                        print(f"    {line}")
                else:
                    print(" OK")
            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                print(f" ERROR: {e}")
                if isinstance(e, FileNotFoundError):
                    print("  pdflatex not found. Install texlive:")
                    print("    dnf install texlive-scheme-basic texlive-booktabs "
                          "texlive-caption texlive-fancyhdr texlive-float "
                          "texlive-longtable texlive-hyperref")
                return False

        result_pdf = Path(tmpdir) / "report.pdf"
        if result_pdf.exists():
            shutil.copy2(result_pdf, output_path)
            return True
        else:
            print("  Error: pdflatex did not produce output PDF.")
            log_path = Path(tmpdir) / "report.log"
            if log_path.exists():
                print(log_path.read_text(encoding="utf-8", errors="replace")[-3000:])
            return False


# ── Report Assembly ───────────────────────────────────────────────

def _compute_analysis_vars(results, perf_data, profile_top, libraries, sizes):
    """Compute data-driven variables for the analysis sections of the template."""
    ctx = {}
    largest = max(sizes) if sizes else 0
    smallest = min(sizes) if sizes else 0

    # Index: (library, workload) -> mops at given size
    by_key = {}          # at largest size
    by_key_1m = {}       # at 1M
    by_key_small = {}    # at smallest size
    all_by_key = {}      # (library, workload, n) -> mops
    for r in results:
        lib, wl, n = r["library"], r["workload"], r["n"]
        mops = r.get("mops", 0)
        all_by_key[(lib, wl, n)] = mops
        if n == largest:
            by_key[(lib, wl)] = mops
        if n == 1048576:
            by_key_1m[(lib, wl)] = mops
        if n == smallest:
            by_key_small[(lib, wl)] = mops

    # Use 1M data if available, else largest
    ref = by_key_1m if by_key_1m else by_key
    competitors = [lib for lib in libraries if lib != "matryoshka"]

    # ── Per-library throughput at largest and smallest sizes ──
    all_libs = ["matryoshka", "std_set", "tlx_btree", "libart", "abseil_btree"]
    lib_short = {
        "matryoshka": "mat", "std_set": "stdset", "tlx_btree": "tlx",
        "libart": "art", "abseil_btree": "abseil",
    }
    workloads = ["seq_insert", "rand_insert", "rand_delete", "mixed",
                 "ycsb_a", "ycsb_b", "search_after_churn"]

    for lib in all_libs:
        short = lib_short[lib]
        for wl in workloads:
            val_l = by_key.get((lib, wl), 0)
            val_s = by_key_small.get((lib, wl), 0)
            val_m = ref.get((lib, wl), 0)
            ctx[f"{short}_{wl}_mops"] = f"{val_m:.2f}"
            ctx[f"{short}_{wl}_mops_large"] = f"{val_l:.2f}"
            ctx[f"{short}_{wl}_mops_small"] = f"{val_s:.2f}"

    # Specific legacy values (keep backward compat)
    ctx["matryoshka_rand_insert_mops"] = ctx["mat_rand_insert_mops"]
    ctx["stdset_rand_insert_mops"] = ctx["stdset_rand_insert_mops"]
    ctx["matryoshka_search_mops"] = ctx["mat_search_after_churn_mops"]

    # Best non-matryoshka competitor on rand_insert
    best_ri = max((ref.get((c, "rand_insert"), 0) for c in competitors), default=0)
    ctx["best_competitor_rand_insert_mops"] = f"{best_ri:.2f}"

    # Slowdown/speedup factors (at largest size, matryoshka vs each lib)
    mat_ri = by_key.get(("matryoshka", "rand_insert"), 0)
    best_ri_l = max((by_key.get((c, "rand_insert"), 0) for c in competitors), default=0)
    ctx["insert_slowdown_factor"] = f"{best_ri_l / mat_ri:.1f}" if mat_ri > 0 else "N/A"

    mat_rd = by_key.get(("matryoshka", "rand_delete"), 0)
    best_rd = max((by_key.get((c, "rand_delete"), 0) for c in competitors), default=0)
    ctx["delete_slowdown_factor"] = f"{best_rd / mat_rd:.1f}" if mat_rd > 0 else "N/A"

    # Per-workload ratio: matryoshka / each competitor at largest size
    for wl in workloads:
        mat_v = by_key.get(("matryoshka", wl), 0)
        for lib in competitors:
            short = lib_short.get(lib, lib)
            other_v = by_key.get((lib, wl), 0)
            if mat_v > 0 and other_v > 0:
                ctx[f"ratio_mat_{short}_{wl}"] = f"{mat_v / other_v:.2f}"
            else:
                ctx[f"ratio_mat_{short}_{wl}"] = "N/A"

    # Scaling degradation: ratio of small-N to large-N throughput
    for lib in all_libs:
        short = lib_short[lib]
        for wl in workloads:
            vs = by_key_small.get((lib, wl), 0)
            vl = by_key.get((lib, wl), 0)
            if vs > 0 and vl > 0:
                ctx[f"scale_{short}_{wl}"] = f"{vs / vl:.1f}"
            else:
                ctx[f"scale_{short}_{wl}"] = "N/A"

    # B-tree competitors gap
    tlx_ri = by_key.get(("tlx_btree", "rand_insert"), 0)
    abs_ri = by_key.get(("abseil_btree", "rand_insert"), 0)
    if tlx_ri > 0 and abs_ri > 0:
        gap = abs(tlx_ri - abs_ri) / max(tlx_ri, abs_ri) * 100
        ctx["btree_insert_gap_pct"] = f"{gap:.0f}"
    elif tlx_ri > 0:
        ctx["btree_insert_gap_pct"] = "N/A (only TLX available)"
    else:
        ctx["btree_insert_gap_pct"] = "N/A"

    ctx["largest_n"] = str(largest)
    ctx["smallest_n"] = str(smallest)

    # ── Perf-derived metrics (per-library) ──
    def _perf_rate(lib, miss_key, ref_key):
        if not perf_data or lib not in perf_data:
            return "N/A"
        c = perf_data[lib]
        m, r2 = c.get(miss_key), c.get(ref_key)
        if m is not None and r2 and r2 > 0:
            return f"{1000.0 * m / r2:.1f}"
        return "N/A"

    def _perf_ipc(lib):
        if not perf_data or lib not in perf_data:
            return "N/A"
        c = perf_data[lib]
        cyc, ins = c.get("cycles"), c.get("instructions")
        return f"{ins / cyc:.2f}" if cyc and ins and cyc > 0 else "N/A"

    def _perf_pct(lib, miss_key, ref_key):
        """Return miss rate as percentage (e.g. '5.8')."""
        if not perf_data or lib not in perf_data:
            return "N/A"
        c = perf_data[lib]
        m, r2 = c.get(miss_key), c.get(ref_key)
        if m is not None and r2 and r2 > 0:
            return f"{100.0 * m / r2:.1f}"
        return "N/A"

    # Per-library hardware counter rows for table
    hw_rows = []
    for lib in all_libs:
        if perf_data and lib in perf_data:
            short = lib_short[lib]
            hw_rows.append({
                "name": tex_escape(lib),
                "cache_miss_pct": _perf_pct(lib, "cache-misses", "cache-references"),
                "l1d_miss_pct": _perf_pct(lib, "L1-dcache-load-misses", "L1-dcache-loads"),
                "llc_miss_pct": _perf_pct(lib, "LLC-load-misses", "LLC-loads"),
                "dtlb_miss_per_1k": _perf_rate(lib, "dTLB-load-misses", "dTLB-loads"),
                "branch_miss_pct": _perf_pct(lib, "branch-misses", "branches"),
                "ipc": _perf_ipc(lib),
            })
    ctx["hw_rows"] = hw_rows

    # Legacy scalar values (backward compat)
    ctx["matryoshka_dtlb_miss_rate"] = _perf_rate("matryoshka", "dTLB-load-misses", "dTLB-loads")
    ctx["stdset_dtlb_miss_rate"] = _perf_rate("std_set", "dTLB-load-misses", "dTLB-loads")
    ctx["matryoshka_llc_miss_rate"] = _perf_rate("matryoshka", "LLC-load-misses", "LLC-loads")
    ctx["stdset_llc_miss_rate"] = _perf_rate("std_set", "LLC-load-misses", "LLC-loads")
    ctx["matryoshka_ipc"] = _perf_ipc("matryoshka")

    # Profile: percentage in mt_page_insert
    ctx["pct_leaf_build"] = "N/A"
    if profile_top:
        for pct, sym in profile_top:
            if "mt_page_insert" in sym or "mt_page_delete" in sym:
                ctx["pct_leaf_build"] = f"{pct:.0f}"
                break

    return ctx


def generate_report(results, sys_info, libraries, sizes, output_path,
                    build_dir, perf_data=None, profile_top=None,
                    cache_miss_top=None):
    """Orchestrate chart generation, LaTeX rendering, and PDF compilation."""
    if not results:
        print("No benchmark results to report.")
        return False

    chart_dir = Path(build_dir) / "report_charts"
    chart_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate charts
    print("\n  Generating throughput bar charts...", flush=True)
    bar_charts = make_throughput_bar_charts(results, sizes, libraries, chart_dir)

    print("  Generating scaling line charts...", flush=True)
    scaling_charts = make_scaling_line_charts(results, sizes, libraries, chart_dir)

    hw_chart = None
    if perf_data:
        print("  Generating hardware counter chart...", flush=True)
        hw_chart_path = make_hw_counter_chart(perf_data, libraries, chart_dir)
        if hw_chart_path:
            hw_chart = Path(hw_chart_path).name

    # 2. Build context for the external template
    # Chart filenames (individual per-workload vars: chart_bar_<wl>, chart_scale_<wl>)
    context = {}
    for wl in WORKLOADS:
        if wl in bar_charts:
            context[f"chart_bar_{wl}"] = Path(bar_charts[wl]).name
        else:
            context[f"chart_bar_{wl}"] = ""
        if wl in scaling_charts:
            context[f"chart_scale_{wl}"] = Path(scaling_charts[wl]).name
        else:
            context[f"chart_scale_{wl}"] = ""

    # System info as a dict (matching template's \VAR{sysinfo.cpu} etc.)
    page_size = "4096"
    try:
        page_size = subprocess.check_output(
            ["getconf", "PAGESIZE"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    context["date"] = tex_escape(sys_info.get("date", ""))
    context["sysinfo"] = {
        "cpu": tex_escape(sys_info.get("cpu", "unknown")),
        "kernel": tex_escape(sys_info.get("kernel", "unknown")),
        "l1d": fmt_bytes(sys_info.get("L1d")),
        "l2": fmt_bytes(sys_info.get("L2")),
        "l3": fmt_bytes(sys_info.get("L3")),
        "page_size": f"{page_size} B",
    }

    # Libraries list (dicts with name, label, description)
    context["libraries"] = [
        {
            "name": tex_escape(lib),
            "label": LIBRARY_LABELS.get(lib, tex_escape(lib)),
            "description": LIBRARY_DESCRIPTIONS.get(lib, ""),
        }
        for lib in libraries
    ]

    # Hardware counter chart
    context["chart_perf"] = hw_chart

    # Profile functions (list of dicts with overhead, name, source)
    profile_functions = []
    if profile_top:
        for pct, sym in profile_top:
            profile_functions.append({
                "overhead": f"{pct:.1f}",
                "name": tex_escape(sym),
                "source": "bench\\_compare",
            })
    context["profile_functions"] = profile_functions

    # Cache-miss attribution functions (list of dicts with pct, name)
    cache_miss_functions = []
    if cache_miss_top:
        for pct, sym in cache_miss_top:
            cache_miss_functions.append({
                "pct": f"{pct:.1f}",
                "name": tex_escape(sym),
            })
    context["cache_miss_functions"] = cache_miss_functions

    # Results table (list of dicts matching template's \VAR{r.library} etc.)
    context["results"] = []
    for r in sorted(results,
                    key=lambda x: (x.get("library", ""),
                                   x.get("workload", ""),
                                   x.get("n", 0))):
        context["results"].append({
            "library": tex_escape(r.get("library", "")),
            "workload": tex_escape(r.get("workload", "")),
            "n": str(r.get("n", 0)),
            "mops": f"{r.get('mops', 0):.2f}",
            "ns_per_op": f"{r.get('ns_per_op', 0):.1f}",
        })

    # Data-driven analysis variables
    analysis = _compute_analysis_vars(results, perf_data, profile_top,
                                      libraries, sizes)
    context.update(analysis)

    # 3. Render LaTeX
    print("  Rendering LaTeX...", flush=True)
    tex = render_latex(context)

    # 4. Compile to PDF
    print("  Compiling PDF...", flush=True)
    ok = compile_latex(tex, output_path, chart_dir)
    return ok


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Matryoshka B+ tree benchmark report generator")
    parser.add_argument(
        "--build-dir", default="build",
        help="CMake build directory (default: build)")
    parser.add_argument(
        "--output", default="matryoshka_report.pdf",
        help="Output PDF path (default: matryoshka_report.pdf)")
    parser.add_argument(
        "--no-perf", action="store_true",
        help="Skip perf stat and perf record collection")
    parser.add_argument(
        "--sizes", nargs="+", type=int, default=DEFAULT_SIZES,
        help="Tree sizes to test (default: %(default)s)")
    args = parser.parse_args()

    build_dir = Path(args.build_dir)
    bench_binary = ROOT / build_dir / "bench_compare"

    if not bench_binary.exists():
        print(f"Error: {bench_binary} not found.")
        print(f"Build the project first:")
        print(f"  cmake -B {args.build_dir} && cmake --build {args.build_dir}")
        sys.exit(1)

    print("=" * 60)
    print("  Matryoshka Benchmark Report Generator")
    print("=" * 60)

    # ── System info ───────────────────────────────────────────────
    print("\nCollecting system information...")
    sys_info = get_system_info()
    print(f"  CPU:    {sys_info.get('cpu', '?')}")
    print(f"  Kernel: {sys_info.get('kernel', '?')}")
    print(f"  L1d={fmt_bytes(sys_info.get('L1d'))}  "
          f"L2={fmt_bytes(sys_info.get('L2'))}  "
          f"L3={fmt_bytes(sys_info.get('L3'))}")

    # ── Detect libraries ──────────────────────────────────────────
    print("\nDetecting compiled libraries...")
    available_libs = detect_libraries(bench_binary)
    if not available_libs:
        # Fallback: try running each library to see what works
        print("  (help parsing failed, probing each library...)")
        available_libs = []
        for lib in LIBRARIES + ["abseil_btree"]:
            r = run_bench_compare(bench_binary, lib, "seq_insert", 1024)
            if r:
                available_libs.append(lib)
                print(f"    {lib}: available")
            else:
                print(f"    {lib}: not available")

    if not available_libs:
        print("Error: no libraries detected in bench_compare binary.")
        sys.exit(1)

    print(f"  Available: {', '.join(available_libs)}")

    # ── Check perf availability ───────────────────────────────────
    use_perf = not args.no_perf and have_perf()
    if args.no_perf:
        print("\nPerf collection disabled (--no-perf).")
    elif use_perf:
        print("\nPerf is available and will be used for hardware counters.")
    else:
        print("\nPerf not available; skipping hardware counter collection.")

    # ── Run benchmarks ────────────────────────────────────────────
    sizes = sorted(args.sizes)
    total_combos = len(available_libs) * len(WORKLOADS) * len(sizes)
    print(f"\nRunning {total_combos} benchmark combinations "
          f"({len(available_libs)} libraries x {len(WORKLOADS)} workloads "
          f"x {len(sizes)} sizes)...\n")

    results = run_all_benchmarks(bench_binary, available_libs, WORKLOADS, sizes)
    print(f"\nCollected {len(results)} result entries.")

    if not results:
        print("No benchmark results collected. Exiting.")
        sys.exit(1)

    # ── Perf stat (hardware counters) ─────────────────────────────
    perf_data = None
    if use_perf:
        largest_size = max(sizes)
        print(f"\nCollecting perf stat counters at N={fmt_size(largest_size)}, "
              f"workload=rand_insert...")
        perf_data = collect_perf_stats(
            bench_binary, available_libs, "rand_insert", largest_size)

    # ── Perf record (profile top functions) ───────────────────────
    profile_top = None
    if use_perf:
        profile_size = 4194304
        if profile_size not in sizes:
            # Use the largest size available
            profile_size = max(sizes)
        print(f"\nRunning perf record on matryoshka/rand_insert "
              f"N={fmt_size(profile_size)}...", flush=True)
        profile_top = run_perf_record(
            bench_binary, "matryoshka", "rand_insert", profile_size)
        if profile_top:
            print(f"  Captured {len(profile_top)} hot functions.")
            for pct, sym in profile_top[:5]:
                print(f"    {pct:5.1f}%  {sym}")
        else:
            print("  perf record/report failed (may need elevated privileges).")

    # ── Perf cache-miss attribution ──────────────────────────────
    cache_miss_top = None
    if use_perf:
        cm_size = 4194304
        if cm_size not in sizes:
            cm_size = max(sizes)
        print(f"\nRunning perf record -e cache-misses on matryoshka/rand_insert "
              f"N={fmt_size(cm_size)}...", flush=True)
        cache_miss_top = run_perf_cache_misses(
            bench_binary, "matryoshka", "rand_insert", cm_size)
        if cache_miss_top:
            print(f"  Captured {len(cache_miss_top)} cache-miss functions.")
            for pct, sym in cache_miss_top[:5]:
                print(f"    {pct:5.1f}%  {sym}")
        else:
            print("  perf record (cache-misses) failed.")

    # ── Generate report ───────────────────────────────────────────
    print(f"\nGenerating report: {args.output}")
    ok = generate_report(
        results, sys_info, available_libs, sizes, args.output,
        build_dir=str(build_dir),
        perf_data=perf_data,
        profile_top=profile_top,
        cache_miss_top=cache_miss_top,
    )

    if ok:
        print(f"\nDone. Report saved to {args.output}")
    else:
        print(f"\nReport generation failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
