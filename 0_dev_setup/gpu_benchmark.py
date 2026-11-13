import time
import torch

torch.manual_seed(0)
size = 5000

a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)

# CPU warm-up
_ = a_cpu @ b_cpu

start = time.perf_counter()
c_cpu = a_cpu @ b_cpu
cpu_time = time.perf_counter() - start
print(f"CPU: {cpu_time:.2f}s")

if torch.mps.is_available():
    a_gpu = a_cpu.to("mps")
    b_gpu = b_cpu.to("mps")

    # GPU warm-up
    _ = a_gpu @ b_gpu
    torch.mps.synchronize()

    start = time.perf_counter()
    c_gpu = a_gpu @ b_gpu
    torch.mps.synchronize()
    gpu_time = time.perf_counter() - start
    print(f"GPU: {gpu_time:.2f}s")
    print(f"Speedup: {cpu_time / gpu_time:.2f}x")

    print(f"Results match: {torch.allclose(c_cpu, c_gpu.cpu())}")
    diff = (c_cpu - c_gpu.cpu()).abs()
    print(f"Max diff: {diff.max().item()}")
    print(f"Mean diff: {diff.mean().item()}")
    print(f"Mean abs magnitude: {c_cpu.abs().mean().item()}")
