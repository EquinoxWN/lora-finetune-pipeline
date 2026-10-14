"""Wall time, peak VRAM and GPU energy per phase, plus a free-memory preflight."""

from __future__ import annotations

import gc
import time
from dataclasses import asdict, dataclass
from types import TracebackType
from typing import Any

import torch

MIB = 1024 * 1024


@dataclass
class Measurement:
    """One measured phase."""

    name: str
    seconds: float = 0.0
    peak_allocated_mib: float | None = None
    peak_reserved_mib: float | None = None
    energy_wh: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Plain JSON-ready form, rounded."""
        out = asdict(self)
        for key, value in out.items():
            if isinstance(value, float):
                out[key] = round(value, 4 if key == "energy_wh" else 2)
        return out


class _Nvml:
    """Total-energy counter through NVML, if the driver supports it."""

    def __init__(self) -> None:
        """Open the first GPU, or stay disabled."""
        self.handle: Any = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self.nvml: Any = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
        except Exception:  # missing library, no GPU, or unsupported counter
            self.handle = None

    def millijoules(self) -> int | None:
        """Energy used by the GPU since the driver loaded, in mJ."""
        if self.handle is None:
            return None
        return int(self.nvml.nvmlDeviceGetTotalEnergyConsumption(self.handle))


_NVML: _Nvml | None = None


def _nvml() -> _Nvml:
    """Shared NVML handle, opened on first use."""
    global _NVML
    if _NVML is None:
        _NVML = _Nvml()
    return _NVML


class Meter:
    """Context manager that measures one phase."""

    def __init__(self, name: str, device: torch.device) -> None:
        """Prepare to measure a phase on a device."""
        self.result = Measurement(name)
        self.device = device
        self._t0 = 0.0
        self._mj0: int | None = None

    def __enter__(self) -> Meter:
        """Reset the peak counters and start the clock."""
        if self.device.type == "cuda":
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()  # so reserved memory left by an earlier phase is not counted
            torch.cuda.reset_peak_memory_stats()
            self._mj0 = _nvml().millijoules()
        self._t0 = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Stop the clock and read the peaks."""
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            self.result.peak_allocated_mib = torch.cuda.max_memory_allocated() / MIB
            self.result.peak_reserved_mib = torch.cuda.max_memory_reserved() / MIB
            end = _nvml().millijoules()
            if self._mj0 is not None and end is not None:
                self.result.energy_wh = (end - self._mj0) / 3_600_000
        self.result.seconds = time.perf_counter() - self._t0


def free_ram_mib() -> int:
    """Physical memory available without swapping."""
    import psutil

    return int(psutil.virtual_memory().available // MIB)


def gpu_info() -> dict[str, Any]:
    """GPU name, total VRAM and free VRAM, or an empty dict without CUDA."""
    if not torch.cuda.is_available():
        return {}
    free, total = torch.cuda.mem_get_info()
    return {
        "name": torch.cuda.get_device_name(0),
        "vram_total_mib": total // MIB,
        "vram_free_mib": free // MIB,
        "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
    }


class PreflightError(RuntimeError):
    """The machine does not have the memory a run needs."""


def preflight(min_free_ram_mib: int, min_free_vram_mib: int) -> dict[str, Any]:
    """Refuse to load a model when RAM or VRAM is short (avoids swapping the machine)."""
    if not torch.cuda.is_available():
        raise PreflightError("CUDA is not available; the QLoRA run needs an NVIDIA GPU")
    info = gpu_info()
    info["ram_free_mib"] = free_ram_mib()
    if info["ram_free_mib"] < min_free_ram_mib:
        raise PreflightError(f"only {info['ram_free_mib']} MiB RAM free, need {min_free_ram_mib}")
    if info["vram_free_mib"] < min_free_vram_mib:
        raise PreflightError(
            f"only {info['vram_free_mib']} MiB VRAM free, need {min_free_vram_mib}"
        )
    return info
