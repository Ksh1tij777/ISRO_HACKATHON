"""GPU sanity check (Phase 0).

Run: python scripts/check_gpu.py
Confirms torch sees CUDA and can allocate/compute on the GPU.
"""
import sys


def main() -> int:
    import torch

    print(f"torch            : {torch.__version__}")
    print(f"cuda available   : {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("\nNo CUDA device visible to torch. Training will fall back to CPU "
              "(too slow for the 40-epoch run). Fix the CUDA install before training.")
        return 1

    dev = torch.device("cuda:0")
    name = torch.cuda.get_device_name(dev)
    total_gb = torch.cuda.get_device_properties(dev).total_memory / 1024**3
    print(f"device           : {name}")
    print(f"total memory     : {total_gb:.1f} GiB")

    # Tiny matmul to confirm the device actually computes.
    a = torch.randn(2048, 2048, device=dev)
    b = torch.randn(2048, 2048, device=dev)
    c = (a @ b).sum().item()
    torch.cuda.synchronize()
    print(f"matmul check     : ok (sum={c:.1f})")
    print(f"peak alloc       : {torch.cuda.max_memory_allocated(dev) / 1024**2:.0f} MiB")

    if total_gb < 10:
        print("\nNote: <10 GiB VRAM. SegFormer-B2 @ batch 8 / 512px will likely OOM.")
        print("      Drop batch_size to 4 (or B0/B1) in configs/segformer_b2.yaml.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
