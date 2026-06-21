"""
compare_torch.py

Benchmark: custom CUDA transpose kernel vs PyTorch transpose + contiguous.

Key insight:
  PyTorch's tensor.T is a ZERO-COPY view (swaps strides, no data movement).
  The actual memory reordering only happens when .contiguous() is called.
  This script times the .contiguous() path, which internally calls a CUDA
  kernel similar to our custom implementations.

Prerequisites:
  Run the CUDA binaries first:
    nvcc -o transpose_scalar.exe temp/transpose.cu
    nvcc -o transpose_f4.exe    temp/demo1.cu

  Then:
    python compare_torch.py
"""

import torch
import time
import subprocess
import re
import sys
import os


# =========================================================================
# Run a CUDA binary and parse its output for timing/BW
# =========================================================================
def run_cuda_binary(exe_path, label):
    """Run a CUDA binary and extract timing + bandwidth from its output."""
    if not os.path.exists(exe_path):
        print(f"  [SKIP] {label}: binary not found at {exe_path}")
        print(f"         Compile first: nvcc -o {os.path.basename(exe_path)} ...")
        return None

    result = subprocess.run([exe_path], capture_output=True, text=True)
    output = result.stdout + result.stderr

    versions = []

    # Parse lines like: "  v3               0.946      141.91       73.9 %         3.18 x"
    # or:              "  v1               0.741      181.09       94.3 %        236.7 x"
    for line in output.splitlines():
        # Try both scalar and float4 patterns
        m = re.match(
            r'\s+(v\S+)\s+(\S+)\s+ms\s+\|\s+BW\s+(\S+)\s+GB/s\s*',
            line
        )
        if not m:
            m = re.match(
                r'\s+(v\S+)\s+PASS\s*\n?\s*(\S+)\s+ms\s+\|\s+BW\s+(\S+)\s+GB/s\s*',
                line, re.DOTALL
            )
        if not m:
            # Try the iostream style output: "v1               0.741      181.10 ..."
            m = re.match(
                r'\s*(v\S+)\s+(\S+)\s+(\S+)\s+(\S+)',
                line
            )
            if m:
                name = m.group(1)
                try:
                    t = float(m.group(2))
                    bw = float(m.group(3))
                    versions.append({
                        'label': f'{label}/{name}',
                        'time_ms': t,
                        'bw_gbs': bw,
                    })
                except ValueError:
                    pass
            continue

        versions.append({
            'label': f'{label}/{m.group(1)}',
            'time_ms': float(m.group(2)),
            'bw_gbs': float(m.group(3)),
        })

    if not versions:
        print(f"  [WARN] {label}: could not parse output")
        print(f"  --- raw output ---")
        for line in output.splitlines()[-20:]:
            print(f"    {line}")
        print(f"  --- end ---")

    return versions


# =========================================================================
# PyTorch transpose + contiguous benchmark
# =========================================================================
def bench_torch_transpose(device, H, W, nreps=20, dtype=torch.float32):
    """Time torch.Tensor.T.contiguous() for a matrix transpose."""

    elem_size = 4 if dtype == torch.float32 else 2
    data_per_run = 2 * H * W * elem_size  # read + write

    # Create tensor on GPU
    A = torch.rand(H, W, dtype=dtype, device=device)
    B = torch.empty(W, H, dtype=dtype, device=device)
    torch.cuda.synchronize()

    # Warm-up
    for _ in range(3):
        B.copy_(A.T)
        torch.cuda.synchronize()

    # Timed runs
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)

    times = []
    for _ in range(nreps):
        start_ev.record()
        B.copy_(A.T)
        end_ev.record()
        torch.cuda.synchronize()
        times.append(start_ev.elapsed_time(end_ev))

    avg_ms = sum(times) / len(times)
    bw_gbs = data_per_run / (avg_ms * 1e-3) / 1e9

    # Verify correctness
    ref = A.cpu().T.contiguous()
    gpu_result = B.cpu()
    diff = (ref - gpu_result).abs().max().item()

    return avg_ms, bw_gbs, diff


# =========================================================================
# main
# =========================================================================
def main():
    if not torch.cuda.is_available():
        print("CUDA not available. Install PyTorch with CUDA support.")
        sys.exit(1)

    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)

    print("=" * 56)
    print("  Transpose: Custom CUDA vs PyTorch")
    print("=" * 56)
    print(f"  GPU       : {props.name}")
    print(f"  CUDA      : {torch.version.cuda}")
    print(f"  PyTorch   : {torch.__version__}")
    print()

    H, W = 4096, 4096
    data_MB = 2 * H * W * 4 / (1024 * 1024)
    print(f"  Matrix    : {H} x {W}  ({H * W * 4 // (1024 * 1024)} MB)")
    print(f"  Data/run  : {data_MB:.0f} MB (read + write)")
    print()

    # ===================================================================
    # 1. Run our CUDA binaries
    # ===================================================================
    this_dir = os.path.dirname(os.path.abspath(__file__))
    temp_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(this_dir))), 'temp'
    )

    all_versions = []

    # Scalar transpose
    scalar_exe = os.path.join(temp_dir, 'transpose.exe')
    results = run_cuda_binary(scalar_exe, 'custom/scalar')
    if results:
        all_versions.extend(results)

    # float4 transpose
    f4_exe = os.path.join(temp_dir, 'demo1.exe')
    results = run_cuda_binary(f4_exe, 'custom/float4')
    if results:
        all_versions.extend(results)

    print()

    # ===================================================================
    # 2. PyTorch benchmark
    # ===================================================================
    print("--- PyTorch ---")
    print("  torch.Tensor.T                  : zero-copy view (no time)")
    print("  torch.Tensor.T.contiguous()     : actual memory reorder")
    print("  B.copy_(A.T)                    : optimized contiguous")
    print()

    mem_clock_khz = 6001  # RTX 3050 Laptop typical
    bus_width = props.memory_bus_width  # bits
    if bus_width == 0:
        bus_width = 128  # fallback
    bw_theor = mem_clock_khz * 1e3 * bus_width / 8.0 * 2.0 / 1e9

    # Test float32
    for hw, label in [((H, W), "float32")]:
        h, w = hw

        # .T then .contiguous() -- two separate ops
        t0 = time.perf_counter()
        A = torch.rand(h, w, device=device)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        # Warm-up already done above conceptually
        for _ in range(5):
            _ = A.T.contiguous()
            torch.cuda.synchronize()

        t2 = time.perf_counter()
        for _ in range(20):
            _ = A.T.contiguous()
            torch.cuda.synchronize()
        t3 = time.perf_counter()
        avg_contig = (t3 - t2) / 20 * 1000

        alloc_ms = (t1 - t0) * 1000
        print(f"  {label}: .T.contiguous()  = {avg_contig:.3f} ms")

        # B.copy_(A.T) -- single kernel, most efficient
        ms, bw, err = bench_torch_transpose(device, h, w, nreps=20)
        all_versions.append({
            'label': 'torch/copy_',
            'time_ms': ms,
            'bw_gbs': bw,
        })
        print(f"  {label}: B.copy_(A.T)     = {ms:.3f} ms  |  "
              f"BW {bw:.2f} GB/s  |  {100.0*bw/bw_theor:.1f}% eff"
              f"  |  max err = {err:.2e}")
        print()

    # ===================================================================
    # 3. Final comparison table
    # ===================================================================
    print("=" * 56)
    print("  Comparison Summary")
    print("=" * 56)
    print(f"  {'Version':<24s} {'Time(ms)':>10s} {'BW(GB/s)':>10s} "
          f"{'Eff.BW%':>8s}")
    print("  " + "-" * 52)

    for v in sorted(all_versions, key=lambda x: x['time_ms']):
        eff = 100.0 * v['bw_gbs'] / bw_theor
        print(f"  {v['label']:<24s} {v['time_ms']:10.3f} "
              f"{v['bw_gbs']:10.2f} {eff:7.1f} %")

    print("  " + "-" * 52)
    print(f"  Theoretical peak BW: {bw_theor:.2f} GB/s")
    print()


if __name__ == '__main__':
    main()
