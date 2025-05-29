"""
ServerInfo that will be announce to DHT and used for client's routing.
    HardwareInfo: Detects and summarizes hardware information, RAM and FLOPs
    ShardedModelInfo:
"""

import platform
import subprocess
from dataclasses import asdict, dataclass
from typing import Any, ClassVar, Dict

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_reduce
from mlx_lm.tuner.utils import nparams

try:
    import psutil
except ImportError:
    psutil = None


@dataclass
class HardwareInfo:
    """Generic hardware summary for a peer."""

    total_ram_gb: float
    chip: str
    tflops_fp16: float

    def dumps(self) -> Dict[str, Any]:
        """Serializes the HardwareInfo object to a dictionary."""
        return asdict(self)

    @classmethod
    def loads(cls, obj: Dict[str, Any]) -> "HardwareInfo":
        """Deserializes a dictionary into a HardwareInfo object."""
        return cls(**obj)

    @staticmethod
    def detect() -> "HardwareInfo":
        """Dispatch to the correct subclass for the current machine."""
        if platform.system() == "Darwin" and platform.machine().startswith("arm"):
            return AppleSiliconHardwareInfo.detect()
        # @TODO: add NvidiaHardwareInfo.detect() etc.
        raise NotImplementedError("Unsupported hardware; add a subclass.")


@dataclass
class AppleSiliconHardwareInfo(HardwareInfo):
    """HardwareInfo specialised for Apple silicon (M-series)."""

    # From cpu-monkey.com
    _APPLE_PEAK_FP16: ClassVar[Dict[str, float]] = {
        "M1": 4.58,
        "M1 Pro": 10.6,
        "M1 Max": 21.2,
        "M2": 7.1,
        "M2 Pro": 11.36,
        "M2 Max": 26.98,
        "M3": 7.1,
        "M3 Pro": 9.94,
        "M3 Max": 28.4,
        "M4": 8.52,
        "M4 Pro": 17.04,
        "M4 Max": 34.08,
    }

    @classmethod
    def detect(cls) -> "AppleSiliconHardwareInfo":
        if psutil:
            total_gb = psutil.virtual_memory().total / 2**30
        else:
            total_gb = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"])) / 2**30

        chip = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()

        short_name = chip.rsplit("Apple ", maxsplit=1)[-1]
        try:
            flops = cls._APPLE_PEAK_FP16[short_name]
        except KeyError as e:
            raise RuntimeError(
                f"Unknown Apple silicon chip '{short_name}' detected. "
                "Please add it to the _APPLE_PEAK_FP16 dictionary."
            ) from e

        return cls(total_ram_gb=round(total_gb, 1), chip=chip, tflops_fp16=flops)


@dataclass
class ShardedModelInfo:
    """
    Detailed information about the specific model shard hosted by a server.
    """

    model_name: str
    start_layer: int
    end_layer: int
    parameter_count: int
    memory_consumption_mb: float

    def dumps(self) -> Dict[str, Any]:
        """Serializes the HardwareInfo object to a dictionary."""
        data = asdict(self)
        return data

    @classmethod
    def loads(cls, data: Dict[str, Any]) -> "ShardedModelInfo":
        """Deserializes a dictionary into a HardwareInfo object."""
        return cls(**data)

    @classmethod
    def from_sharded_model(
        cls, sharded_model_instance: nn.Module  # Instance of your ShardedModel
    ) -> "ShardedModelInfo":
        """
        Constructs ShardedModelInfo from a loaded ShardedModel instance.
        Assumes sharded_model_instance has start_layer, end_layer, and model_id_original attributes.
        """
        # Calculate parameter count
        count = nparams(sharded_model_instance)

        model_bytes = tree_reduce(
            lambda acc, x: acc + x.nbytes if isinstance(x, mx.array) else acc,
            sharded_model_instance.parameters(),
            0,
        )
        memory_mb = round(model_bytes / (1024 * 1024), 2)

        return cls(
            model_name=sharded_model_instance.model_id,  # Use the cleaned name
            start_layer=sharded_model_instance.start_layer,
            end_layer=sharded_model_instance.end_layer,
            parameter_count=count,
            memory_consumption_mb=memory_mb,
        )
