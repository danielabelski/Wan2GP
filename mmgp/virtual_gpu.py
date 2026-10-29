# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""A smaller GPU (and machine) emulated on a larger one, for simulations: every VRAM figure PyTorch, NVML and MMGP report is the emulated
card's, and MMGP's VRAM allocator reserves no more than such a card leaves the process, so that MMGP and the application decide as they
would on it (memory profiles, budgets, preloads, CPU compute, the application's own VRAM rules).

    MMGP_VIRTUAL_VRAM_GB=8 python app.py                    # read when mmgp is first imported (also MMGP_VIRTUAL_VRAM_LIMIT_GB, MMGP_VIRTUAL_RAM_GB)
    from mmgp import virtual_gpu; virtual_gpu.install(8)    # or in code, before the application reads any VRAM figure

Emulated:
- torch.cuda.get_device_properties(...).total_memory;
- the limit of what the process may reserve, what a real card leaves it beside its CUDA context, the driver and the desktop: the card minus
  LIMIT_OVERHEAD_GB unless given, enforced by the VRAM allocator (allocator.set_vram_limit), or by PyTorch's allocator
  (set_per_process_memory_fraction, limit_torch_allocator);
- the free VRAM (torch.cuda.mem_get_info, pynvml.nvmlDeviceGetMemoryInfo): that limit less what the process reserves, but never more than the
  real GPU has free. The VRAM of the other programs on the GPU counts only when it leaves less than the emulated card would (the VRAM
  allocator's room follows the same rule);
- optionally the RAM: psutil.virtual_memory().total, as MMGP reads it at import (the available RAM stays the machine's: hold the rest with
  another process to emulate it too).
The GPU's speed, the PCIe bandwidth and the CPU stay the machine's."""
import os
import types

import torch

LIMIT_OVERHEAD_GB = 1.2
state = {}  # total, limit (bytes) once installed
_real_properties = None  # the real torch.cuda.get_device_properties


class _Properties:
    """The device's properties, the total memory being the emulated card's."""

    def __init__(self, real, total):
        self._real, self.total_memory = real, total

    def __getattr__(self, name):
        return getattr(self._real, name)

    def __repr__(self):
        return f"{self._real!r} (emulated total_memory={self.total_memory >> 20}MB)"


def install(vram_gb, limit_gb=None, ram_gb=None):
    """vram_gb: the emulated card's VRAM (GiB); limit_gb: what the VRAM allocator may reserve (vram_gb - LIMIT_OVERHEAD_GB by default);
    ram_gb: the RAM (GiB), before mmgp.offload is imported."""
    if state:
        raise RuntimeError("The virtual GPU is already installed.")
    total = int(float(vram_gb) * 2**30)
    limit = int((float(vram_gb) - LIMIT_OVERHEAD_GB if limit_gb is None else float(limit_gb)) * 2**30)
    state.update(total=total, limit=limit)

    global _real_properties
    real_properties = _real_properties = torch.cuda.get_device_properties
    torch.cuda.get_device_properties = lambda device=None: _Properties(real_properties(device), total)

    real_mem_get_info, real_reserved = torch.cuda.memory.mem_get_info, torch.cuda.memory.memory_reserved
    try:
        import pynvml
    except ImportError:
        pynvml = None
    real_nvml_memory = None if pynvml is None else pynvml.nvmlDeviceGetMemoryInfo
    nvml_handles = {}

    def free(real_free, device=None):  # what the limit leaves beside the process' reservations, within what the GPU really has free
        from . import allocator
        reserved = allocator.stats(device)["reserved"] if allocator.active == "vmm" else real_reserved(device)
        return max(0, min(limit - reserved, real_free))

    def gpu_free(device):  # the whole GPU's free VRAM (NVML), the other programs' included: on Windows the driver's own figure leaves them out
        index = torch.cuda.current_device() if device is None or torch.device(device).index is None else torch.device(device).index
        handle = nvml_handles.get(index)
        if handle is None:
            properties = real_properties(index)
            pynvml.nvmlInit()
            handle = nvml_handles[index] = pynvml.nvmlDeviceGetHandleByPciBusId(f"{properties.pci_domain_id:08X}:{properties.pci_bus_id:02X}:{properties.pci_device_id:02X}.0")
        return int(real_nvml_memory(handle).free)

    def mem_get_info(device=None):  # (the VRAM allocator answers free with its room once installed, which takes NVML's figure too)
        real_free = real_mem_get_info(device)[0]
        return free(real_free if pynvml is None else min(real_free, gpu_free(device)), device), total
    torch.cuda.mem_get_info = torch.cuda.memory.mem_get_info = mem_get_info

    if pynvml is not None:
        def nvml_memory(handle, *args, **kwargs):
            info = real_nvml_memory(handle, *args, **kwargs)
            available = free(int(info.free))
            return types.SimpleNamespace(total=total, used=total - available, free=available, reserved=getattr(info, "reserved", 0))
        pynvml.nvmlDeviceGetMemoryInfo = nvml_memory

    if ram_gb is not None:
        import psutil
        real_virtual_memory, ram_total = psutil.virtual_memory, int(float(ram_gb) * 2**30)
        psutil.virtual_memory = lambda: real_virtual_memory()._replace(total=ram_total)

    from . import allocator
    if allocator.active == "vmm":
        allocator.set_vram_limit(limit)
    print(f"[MMGP] Virtual GPU: {float(vram_gb):g} GB card emulated, the VRAM allocator reserving at most {limit / 2**30:.2f} GiB"
          + (f", {float(ram_gb):g} GB of RAM" if ram_gb is not None else ""), flush=True)


def limit_torch_allocator():
    """With PyTorch's allocator (not the VRAM allocator), its reservations stop at the limit, as on the emulated card. CUDA initializes."""
    torch.cuda.set_per_process_memory_fraction(min(1.0, state["limit"] / _real_properties(torch.cuda.current_device()).total_memory))


def install_from_environment():
    vram_gb = os.environ.get("MMGP_VIRTUAL_VRAM_GB")
    if vram_gb and not state:
        install(vram_gb, os.environ.get("MMGP_VIRTUAL_VRAM_LIMIT_GB") or None, os.environ.get("MMGP_VIRTUAL_RAM_GB") or None)
